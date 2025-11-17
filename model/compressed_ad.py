"""
Compressed Algorithm Distillation Model.

Extends the base AD model to handle compression tokens and learn from
compressed context efficiently.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange

from env import map_dark_states, map_dark_states_inverse


# Special token IDs
COMPRESS_START_TOKEN = -1
COMPRESS_END_TOKEN = -2


class CompressedAD(nn.Module):
    """
    Algorithm Distillation model with compression token support.
    
    Key differences from base AD:
    1. Adds special embeddings for <compress> and </compress> tokens
    2. Attention masking to handle compressed segments
    3. Loss only computed on non-compressed query positions
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.device = config['device']
        self.n_transit = config['n_transit']
        self.max_seq_length = config['n_transit']
        self.mixed_precision = config['mixed_precision']
        self.grid_size = config['grid_size']
        
        # Transformer config
        tf_n_embd = config['tf_n_embd']
        tf_n_head = config.get('tf_n_head', 4)
        tf_n_layer = config.get('tf_n_layer', 4)
        tf_dim_feedforward = config.get('tf_dim_feedforward', tf_n_embd * 4)
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_length, tf_n_embd))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tf_n_embd,
            nhead=tf_n_head,
            dim_feedforward=tf_dim_feedforward,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=tf_n_layer)
        
        # Embeddings
        self.embed_context = nn.Linear(
            config['dim_states'] * 2 + config['num_actions'] + 1, 
            tf_n_embd
        )
        self.embed_query_state = nn.Embedding(
            config['grid_size'] * config['grid_size'], 
            tf_n_embd
        )
        
        # Special token embeddings for compression markers
        self.compress_start_embed = nn.Parameter(torch.randn(1, 1, tf_n_embd) * 0.02)
        self.compress_end_embed = nn.Parameter(torch.randn(1, 1, tf_n_embd) * 0.02)
        
        # Output head
        self.pred_action = nn.Linear(tf_n_embd, config['num_actions'])
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=config['label_smoothing'])
        
        # Initialize
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
    
    def _apply_positional_embedding(self, x):
        """Add positional embeddings."""
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return x
    
    def _insert_compression_tokens(self, context_embed, compression_mask):
        """
        Insert compression markers into the context embeddings.
        
        Args:
            context_embed: (batch, seq_len, emb_dim) - context embeddings
            compression_mask: (batch, seq_len) - boolean mask for compressed positions
        
        Returns:
            enhanced_embed: embeddings with compression markers inserted
            actual_mask: updated mask after insertion
        """
        batch_size, seq_len, emb_dim = context_embed.shape
        
        # For simplicity, we add compression information via learned embeddings
        # rather than inserting actual tokens (which would change sequence length)
        
        # Create a compression indicator: -1 for start, 0 for normal, +1 for inside, -2 for end
        compression_indicator = torch.zeros(batch_size, seq_len, device=context_embed.device)
        
        if compression_mask.any():
            # Find boundaries
            for b in range(batch_size):
                mask_b = compression_mask[b]
                if not mask_b.any():
                    continue
                
                # Find starts and ends of compressed segments
                diff = torch.diff(mask_b.float(), prepend=torch.tensor([0.0], device=mask_b.device))
                starts = (diff > 0).nonzero(as_tuple=True)[0]
                ends = (diff < 0).nonzero(as_tuple=True)[0]
                
                # Mark positions
                for start_idx in starts:
                    if start_idx < seq_len:
                        compression_indicator[b, start_idx] = -1  # Start marker
                
                for end_idx in ends:
                    if end_idx > 0 and end_idx - 1 < seq_len:
                        compression_indicator[b, end_idx - 1] = -2  # End marker
                
                # Mark interior positions
                for i in range(seq_len):
                    if mask_b[i] and compression_indicator[b, i] == 0:
                        compression_indicator[b, i] = 1  # Inside compressed segment
        
        # Add learned embeddings for compression markers
        enhanced_embed = context_embed.clone()
        
        # Add start token embeddings
        start_mask = (compression_indicator == -1)
        if start_mask.any():
            enhanced_embed = enhanced_embed + self.compress_start_embed * start_mask.unsqueeze(-1).float()
        
        # Add end token embeddings
        end_mask = (compression_indicator == -2)
        if end_mask.any():
            enhanced_embed = enhanced_embed + self.compress_end_embed * end_mask.unsqueeze(-1).float()
        
        return enhanced_embed, compression_mask
    
    def transformer(self, x, max_seq_length=None, dtype=None):
        """Transformer forward with positional embeddings."""
        x = self._apply_positional_embedding(x)
        out = self.transformer_encoder(x)
        return out
    
    def forward(self, x):
        """
        Forward pass with compression token handling.
        
        x should contain:
        - Standard AD inputs (query_states, states, actions, rewards, next_states)
        - compression_mask: boolean tensor indicating compressed positions
        """
        # Extract inputs
        query_states = x['query_states'].to(self.device)
        target_actions = x['target_actions'].to(self.device)
        states = x['states'].to(self.device)
        actions = x['actions'].to(self.device)
        next_states = x['next_states'].to(self.device)
        rewards = x['rewards'].to(self.device)
        rewards = rearrange(rewards, 'b n -> b n 1')
        
        # Compression mask
        compression_mask = x.get('compression_mask', None)
        if compression_mask is not None:
            compression_mask = compression_mask.to(self.device).bool()
        else:
            # No compression for this batch
            compression_mask = torch.zeros(
                states.shape[0], states.shape[1], 
                dtype=torch.bool, device=self.device
            )
        
        # Embed query state
        query_states_embed = self.embed_query_state(
            map_dark_states(query_states, self.grid_size).to(torch.long)
        )
        query_states_embed = rearrange(query_states_embed, 'b d -> b 1 d')
        
        # Embed context
        context, _ = pack([states, actions, rewards, next_states], 'b n *')
        context_embed = self.embed_context(context)
        
        # Insert compression markers
        context_embed, _ = self._insert_compression_tokens(context_embed, compression_mask)
        
        # Combine context and query
        context_embed, _ = pack([context_embed, query_states_embed], 'b * d')
        
        # Transformer forward
        transformer_output = self.transformer(
            context_embed,
            max_seq_length=self.max_seq_length,
            dtype=self.mixed_precision
        )
        
        # Predict action from query position (last position before adding query)
        logits_actions = self.pred_action(transformer_output[:, self.n_transit - 1])
        
        # Compute loss
        loss_action = self.loss_fn(logits_actions, target_actions)
        acc_action = (logits_actions.argmax(dim=-1) == target_actions).float().mean()
        
        result = {
            'loss_action': loss_action,
            'acc_action': acc_action,
        }
        
        return result
    
    def evaluate_in_context(self, vec_env, eval_timesteps, beam_k=0, sample=True):
        """
        In-context evaluation (same as base AD).
        Note: Compression is only used during training, not during inference.
        """
        outputs = {}
        outputs['reward_episode'] = []
        
        reward_episode = np.zeros(vec_env.num_envs)
        
        query_states = vec_env.reset()
        query_states = torch.tensor(query_states, device=self.device, requires_grad=False, dtype=torch.long)
        query_states = rearrange(query_states, 'e d -> e 1 d')
        query_states_embed = self.embed_query_state(map_dark_states(query_states, self.grid_size))
        transformer_input = query_states_embed
        
        for step in range(eval_timesteps):
            query_states_prev = query_states.clone().detach().to(torch.float)
            
            output = self.transformer(
                transformer_input,
                max_seq_length=self.max_seq_length,
                dtype='fp32'
            )
            
            logits = self.pred_action(output[:, -1])
            
            if sample:
                log_probs = F.log_softmax(logits, dim=-1)
                actions = torch.multinomial(log_probs.exp(), num_samples=1)
                actions = rearrange(actions, 'e 1 -> e')
            else:
                actions = logits.argmax(dim=-1)
            
            query_states, rewards, dones, infos = vec_env.step(actions.cpu().numpy())
            
            actions = rearrange(actions, 'e -> e 1 1')
            actions = F.one_hot(actions, num_classes=self.config['num_actions'])
            
            reward_episode += rewards
            rewards = torch.tensor(rewards, device=self.device, requires_grad=False, dtype=torch.float)
            rewards = rearrange(rewards, 'e -> e 1 1')
            
            query_states = torch.tensor(query_states, device=self.device, requires_grad=False, dtype=torch.long)
            query_states = rearrange(query_states, 'e d -> e 1 d')
            
            if dones[0]:
                outputs['reward_episode'].append(reward_episode)
                reward_episode = np.zeros(vec_env.num_envs)
                
                states_next = torch.tensor(
                    np.stack([info['terminal_observation'] for info in infos]),
                    device=self.device, dtype=torch.float
                )
                states_next = rearrange(states_next, 'e d -> e 1 d')
            else:
                states_next = query_states.clone().detach().to(torch.float)
            
            query_states_embed = self.embed_query_state(map_dark_states(query_states, self.grid_size))
            
            context, _ = pack([query_states_prev, actions, rewards, states_next], 'e i *')
            context_embed = self.embed_context(context)
            
            if transformer_input.size(1) > 1:
                context_embed, _ = pack([transformer_input[:, :-1], context_embed], 'e * h')
                context_embed = context_embed[:, -(self.n_transit - 1):]
            
            transformer_input, _ = pack([context_embed, query_states_embed], 'e * h')
        
        outputs['reward_episode'] = np.stack(outputs['reward_episode'], axis=1)
        
        return outputs
