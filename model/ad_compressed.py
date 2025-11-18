"""
Algorithm Distillation model with compression token support.
Learns to predict actions using context inside <compress> ... \compress tokens.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, repeat

from env import map_dark_states, map_dark_states_inverse


# Special token IDs
COMPRESS_START = -1  # <compress>
COMPRESS_END = -2     # \compress
PAD_TOKEN = -3        # padding


class CompressedAD(torch.nn.Module):
    """
    AD model that handles compression tokens around on-policy segments.
    The model learns to focus on context inside compression tokens.
    """
    def __init__(self, config):
        super(CompressedAD, self).__init__()

        self.config = config
        self.device = config['device']
        self.n_transit = config['n_transit']
        self.max_seq_length = config['n_transit']
        self.mixed_precision = config['mixed_precision']
        self.grid_size = config['grid_size']

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
        self.embed_context = nn.Linear(config['dim_states'] * 2 + config['num_actions'] + 1, tf_n_embd)
        self.embed_query_state = nn.Embedding(config['grid_size'] * config['grid_size'], tf_n_embd)
        
        # Special token embeddings for compression markers
        self.embed_compress_start = nn.Parameter(torch.randn(1, 1, tf_n_embd) * 0.02)
        self.embed_compress_end = nn.Parameter(torch.randn(1, 1, tf_n_embd) * 0.02)
        
        # Action prediction head
        self.pred_action = nn.Linear(tf_n_embd, config['num_actions'])

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=config['label_smoothing'])

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def _apply_positional_embedding(self, x):
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return x

    def transformer(self, x, max_seq_length=None, dtype=None):
        """Wrapper for transformer encoder."""
        x = self._apply_positional_embedding(x)
        out = self.transformer_encoder(x)
        return out

    def _insert_compression_token_embeddings(self, context_embed, compression_mask):
        """
        Add compression token embeddings to context.
        
        Args:
            context_embed: (batch, seq_len, emb_dim) - embedded context
            compression_mask: (batch, seq_len) - 1 for positions inside compression segments
            
        Returns:
            Modified embeddings with compression token information
        """
        batch_size, seq_len, emb_dim = context_embed.shape
        
        # Find boundaries (transitions from 0->1 and 1->0)
        # Pad mask to detect boundaries
        padded_mask = torch.cat([
            torch.zeros(batch_size, 1, device=compression_mask.device),
            compression_mask,
            torch.zeros(batch_size, 1, device=compression_mask.device)
        ], dim=1)
        
        # Start positions: 0 -> 1 transition
        start_positions = (padded_mask[:, 1:] - padded_mask[:, :-1]) == 1
        # End positions: 1 -> 0 transition  
        end_positions = (padded_mask[:, :-1] - padded_mask[:, 1:]) == 1
        
        # Add token embeddings at boundary positions
        start_mask = start_positions[:, :seq_len].float().unsqueeze(-1)  # (batch, seq_len, 1)
        end_mask = end_positions[:, :seq_len].float().unsqueeze(-1)
        
        # Add embeddings (broadcasts across batch)
        context_embed = context_embed + self.embed_compress_start * start_mask
        context_embed = context_embed + self.embed_compress_end * end_mask
        
        return context_embed

    def forward(self, x):
        query_states = x['query_states'].to(self.device)  # (batch_size, dim_state)
        target_actions = x['target_actions'].to(self.device)  # (batch_size,)
        states = x['states'].to(self.device)  # (batch_size, num_transit, dim_state)
        actions = x['actions'].to(self.device)  # (batch_size, num_transit, num_actions)
        next_states = x['next_states'].to(self.device)  # (batch_size, num_transit, dim_state)
        rewards = x['rewards'].to(self.device)  # (batch_size, num_transit)
        rewards = rearrange(rewards, 'b n -> b n 1')
        
        # Compression mask (1 for positions inside compression segments)
        compression_mask = x['compression_mask'].to(self.device)  # (batch_size, num_transit)

        # Embed query state
        query_states_embed = self.embed_query_state(map_dark_states(query_states, self.grid_size).to(torch.long))
        query_states_embed = rearrange(query_states_embed, 'b d -> b 1 d')

        # Embed context
        context, _ = pack([states, actions, rewards, next_states], 'b n *')
        context_embed = self.embed_context(context)
        
        # Insert compression token embeddings
        context_embed = self._insert_compression_token_embeddings(context_embed, compression_mask)
        
        # Concatenate context and query
        context_embed, _ = pack([context_embed, query_states_embed], 'b * d')

        # Pass through transformer
        transformer_output = self.transformer(context_embed,
                                              max_seq_length=self.max_seq_length,
                                              dtype=self.mixed_precision)

        result = {}

        # Predict action from query position
        logits_actions = self.pred_action(transformer_output[:, self.n_transit-1])

        loss_full_action = self.loss_fn(logits_actions, target_actions)
        acc_full_action = (logits_actions.argmax(dim=-1) == target_actions).float().mean()

        result['loss_action'] = loss_full_action
        result['acc_action'] = acc_full_action

        return result

    def evaluate_in_context(self, vec_env, eval_timesteps, beam_k=0, sample=True):
        """Evaluation without compression (inference doesn't use compression tokens)."""
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

            output = self.transformer(transformer_input,
                                        max_seq_length=self.max_seq_length,
                                        dtype='fp32')

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

                states_next = torch.tensor(np.stack([info['terminal_observation'] for info in infos]),
                                           device=self.device, dtype=torch.float)

                states_next = rearrange(states_next, 'e d -> e 1 d')
            else:
                states_next = query_states.clone().detach().to(torch.float)

            query_states_embed = self.embed_query_state(map_dark_states(query_states, self.grid_size))

            context, _ = pack([query_states_prev, actions, rewards, states_next], 'e i *')
            context_embed = self.embed_context(context)

            if transformer_input.size(1) > 1:
                context_embed, _ = pack([transformer_input[:, :-1], context_embed], 'e * h')
                context_embed = context_embed[:, -(self.n_transit-1):]

            transformer_input, _ = pack([context_embed, query_states_embed], 'e * h')

        return outputs
