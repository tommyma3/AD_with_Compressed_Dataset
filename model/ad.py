import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, repeat

from env import map_dark_states, map_dark_states_inverse

class AD(torch.nn.Module):
    def __init__(self, config):
        super(AD, self).__init__()

        self.config = config
        self.device = config['device']
        self.n_transit = config['n_transit']
        self.max_seq_length = config['n_transit']
        self.mixed_precision = config['mixed_precision']
        self.grid_size = config['grid_size']
        
        # Enable compressed context mode
        self.use_compressed_context = config.get('use_compressed_context', False)
        self.compress_interval = config.get('compress_interval', 80)  # n_steps from PPO

        tf_n_embd = config['tf_n_embd']
        tf_n_head = config.get('tf_n_head', 4)
        tf_n_layer = config.get('tf_n_layer', 4)
        tf_dim_feedforward = config.get('tf_dim_feedforward', tf_n_embd * 4)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_length, tf_n_embd))
        
        # Special token types: 0=transition, 1=compress_start, 2=compress_end
        self.token_type_embedding = nn.Embedding(3, tf_n_embd)
        
        # Register causal mask buffer (will be created when needed)
        self.register_buffer('causal_mask', None, persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tf_n_embd,
            nhead=tf_n_head,
            dim_feedforward=tf_dim_feedforward,
            activation='gelu',
            batch_first=True,  # so inputs are (batch, seq, emb)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=tf_n_layer)

        self.embed_context = nn.Linear(config['dim_states'] * 2 + config['num_actions'] + 1, tf_n_embd)
        self.embed_query_state = nn.Embedding(config['grid_size'] * config['grid_size'], tf_n_embd)
        
        # Special token embeddings for compression markers
        self.compress_start_token = nn.Parameter(torch.randn(1, 1, tf_n_embd))
        self.compress_end_token = nn.Parameter(torch.randn(1, 1, tf_n_embd))
        
        self.pred_action = nn.Linear(tf_n_embd, config['num_actions'])
        # Predict token type: 0=action, 1=compress_start, 2=compress_end
        self.pred_token_type = nn.Linear(tf_n_embd, 3)

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=config['label_smoothing'])
        self.token_type_loss_fn = nn.CrossEntropyLoss(reduction='mean')

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.compress_start_token, std=0.02)
        nn.init.trunc_normal_(self.compress_end_token, std=0.02)

    def _apply_positional_embedding(self, x):
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return x

    def _get_causal_mask(self, seq_len):
        """
        Generate causal attention mask for autoregressive prediction.
        Returns upper triangular matrix with -inf above diagonal.
        """
        if self.causal_mask is None or self.causal_mask.size(0) != seq_len:
            # Create causal mask: upper triangular matrix with -inf
            mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
            self.causal_mask = mask.to(self.device)
        return self.causal_mask
    
    def transformer(self, x, max_seq_length=None, dtype=None, use_causal_mask=True):
        """
        Thin wrapper so existing code calling self.transformer(...) still works.
        We accept (batch, seq, emb) and return (batch, seq, emb).
        dtype argument is accepted for API compatibility - we won't dynamically change types here.
        use_causal_mask: if True, apply causal masking to prevent future token attention
        """
        x = self._apply_positional_embedding(x)
        
        if use_causal_mask:
            seq_len = x.size(1)
            attn_mask = self._get_causal_mask(seq_len)
            out = self.transformer_encoder(x, mask=attn_mask)  # (batch, seq, emb)
        else:
            out = self.transformer_encoder(x)  # (batch, seq, emb)
        return out

    def forward(self, x):
        query_states = x['query_states'].to(self.device)  # (batch_size, dim_state)
        target_actions = x['target_actions'].to(self.device)  # (batch_size,)
        states = x['states'].to(self.device)  # (batch_size, num_transit, dim_state)
        actions = x['actions'].to(self.device)  # (batch_size, num_transit, num_actions)
        next_states = x['next_states'].to(self.device)  # (batch_size, num_transit, dim_state)
        rewards = x['rewards'].to(self.device)  # (batch_size, num_transit)
        rewards = rearrange(rewards, 'b n -> b n 1')

        query_states_embed = self.embed_query_state(map_dark_states(query_states, self.grid_size).to(torch.long))
        query_states_embed = rearrange(query_states_embed, 'b d -> b 1 d')

        context, _ = pack([states, actions, rewards, next_states], 'b n *')
        context_embed = self.embed_context(context)
        
        # Add token type embeddings for context (all transitions)
        batch_size, seq_len, _ = context_embed.shape
        token_types = torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device)
        context_embed = context_embed + self.token_type_embedding(token_types)
        
        # Handle compressed context mode
        if self.use_compressed_context and 'compress_markers' in x:
            compress_markers = x['compress_markers'].to(self.device)  # (batch_size, num_transit)
            # compress_markers: 0=normal, 1=compress_start, 2=compress_end
            
            # Insert special tokens at marked positions
            embeds_list = []
            for b in range(batch_size):
                seq_embeds = []
                for t in range(seq_len):
                    if compress_markers[b, t] == 1:  # compress_start
                        token_embed = self.compress_start_token.squeeze(0)
                        token_embed = token_embed + self.token_type_embedding(
                            torch.tensor([1], device=self.device))
                        seq_embeds.append(token_embed)
                    elif compress_markers[b, t] == 2:  # compress_end
                        token_embed = self.compress_end_token.squeeze(0)
                        token_embed = token_embed + self.token_type_embedding(
                            torch.tensor([2], device=self.device))
                        seq_embeds.append(token_embed)
                    else:  # normal transition
                        seq_embeds.append(context_embed[b:b+1, t])
                embeds_list.append(torch.cat(seq_embeds, dim=0))
            context_embed = torch.stack(embeds_list, dim=0)
        
        context_embed, _ = pack([context_embed, query_states_embed], 'b * d')

        # Apply causal masking during training to prevent future information leakage
        transformer_output = self.transformer(context_embed,
                                              max_seq_length=self.max_seq_length,
                                              dtype=self.mixed_precision,
                                              use_causal_mask=True)

        result = {}

        # The last token in the sequence is the query state embedding
        # Predict action from this position
        logits_actions = self.pred_action(transformer_output[:, -1])  # (batch_size, num_actions)
        logits_token_type = self.pred_token_type(transformer_output[:, -1])  # (batch_size, 3)

        loss_full_action = self.loss_fn(logits_actions, target_actions)
        acc_full_action = (logits_actions.argmax(dim=-1) == target_actions).float().mean()

        result['loss_action'] = loss_full_action
        result['acc_action'] = acc_full_action
        
        # In compressed context mode, predict token types for special tokens
        if self.use_compressed_context and 'target_token_types' in x:
            target_token_types = x['target_token_types'].to(self.device)
            loss_token_type = self.token_type_loss_fn(logits_token_type, target_token_types)
            acc_token_type = (logits_token_type.argmax(dim=-1) == target_token_types).float().mean()
            result['loss_token_type'] = loss_token_type
            result['acc_token_type'] = acc_token_type
            result['loss_action'] = loss_full_action + 0.1 * loss_token_type  # Combined loss

        return result

    def evaluate_in_context(self, vec_env, eval_timesteps, beam_k=0, sample=True):
        outputs = {}
        outputs['reward_episode'] = []

        reward_episode = np.zeros(vec_env.num_envs)

        query_states = vec_env.reset()
        query_states = torch.tensor(query_states, device=self.device, requires_grad=False, dtype=torch.long)
        query_states = rearrange(query_states, 'e d -> e 1 d')
        query_states_embed = self.embed_query_state(map_dark_states(query_states, self.grid_size))
        transformer_input = query_states_embed
        
        # Track compression regions if using compressed context
        steps_in_compression = 0
        compression_context = None  # Store previous compression region
        current_compression = []  # Accumulate current compression region

        for step in range(eval_timesteps):
            query_states_prev = query_states.clone().detach().to(torch.float)

            # During evaluation, we still use causal masking for consistency
            output = self.transformer(transformer_input,
                                        max_seq_length=self.max_seq_length,
                                        dtype='fp32',
                                        use_causal_mask=True)

            logits = self.pred_action(output[:, -1])
            
            if self.use_compressed_context:
                # Also predict token type to detect compression boundaries
                logits_token_type = self.pred_token_type(output[:, -1])
                predicted_token_type = logits_token_type.argmax(dim=-1)

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
            
            # Add token type embedding for regular transitions
            token_type_embed = self.token_type_embedding(
                torch.zeros(context_embed.shape[0], context_embed.shape[1], dtype=torch.long, device=self.device))
            context_embed = context_embed + token_type_embed
            
            # Handle compressed context mode during evaluation
            if self.use_compressed_context:
                steps_in_compression += 1
                current_compression.append(context_embed)
                
                # Check if we should insert compression markers
                if steps_in_compression == self.compress_interval:
                    # Add compress_end token
                    compress_end_embed = self.compress_end_token.expand(context_embed.shape[0], -1, -1)
                    compress_end_embed = compress_end_embed + self.token_type_embedding(
                        torch.full((context_embed.shape[0], 1), 2, dtype=torch.long, device=self.device))
                    current_compression.append(compress_end_embed)
                    
                    # Save current compression as previous
                    compression_context = torch.cat(current_compression, dim=1)
                    
                    # Start new compression region
                    current_compression = []
                    steps_in_compression = 0
                    
                    # Add compress_start token
                    compress_start_embed = self.compress_start_token.expand(context_embed.shape[0], -1, -1)
                    compress_start_embed = compress_start_embed + self.token_type_embedding(
                        torch.full((context_embed.shape[0], 1), 1, dtype=torch.long, device=self.device))
                    current_compression.append(compress_start_embed)
                
                # Build context from last compression region + current compression region
                if compression_context is not None and len(current_compression) > 0:
                    combined_context = torch.cat([compression_context] + current_compression, dim=1)
                elif len(current_compression) > 0:
                    combined_context = torch.cat(current_compression, dim=1)
                else:
                    combined_context = context_embed
                    
                # Limit context length
                if combined_context.size(1) > self.n_transit - 1:
                    combined_context = combined_context[:, -(self.n_transit-1):]
                    
                transformer_input, _ = pack([combined_context, query_states_embed], 'e * h')
            else:
                # Original logic without compression
                if transformer_input.size(1) > 1:
                    context_embed, _ = pack([transformer_input[:, :-1], context_embed], 'e * h')
                    context_embed = context_embed[:, -(self.n_transit-1):]

                transformer_input, _ = pack([context_embed, query_states_embed], 'e * h')

        outputs['reward_episode'] = np.stack(outputs['reward_episode'], axis=1)

        return outputs
