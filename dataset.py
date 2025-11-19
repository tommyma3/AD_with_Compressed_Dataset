from torch.utils.data import Dataset
import numpy as np
from utils import get_traj_file_name
import h5py
import random
from einops import rearrange, repeat


class ADDataset(Dataset):
    def __init__(self, config, traj_dir, mode='train', n_stream=None, source_timesteps=None):
        self.config = config
        self.env = config['env']
        self.n_transit = config['n_transit']
        self.dynamics = config['dynamics']
        self.use_compressed_context = config.get('use_compressed_context', False)
        self.compress_interval = config.get('compress_interval', 80)
        
        if self.env == 'darkroom':
            n_total_envs = config['grid_size'] ** 2

        else:
            raise ValueError('Invalid env')

        total_env_idx = list(range(n_total_envs))
        random.seed(config['env_split_seed'])
        random.shuffle(total_env_idx)
        
        n_train_envs = round(n_total_envs * config['train_env_ratio'])
        
        if mode == 'train':
            env_idx = total_env_idx[:n_train_envs]
        elif mode == 'test':
            env_idx = total_env_idx[n_train_envs:]
        elif mode == 'all':
            env_idx = total_env_idx
        else:
            raise ValueError('Invalid mode')

        states = []
        actions = []
        rewards = []
        next_states = []
        compress_markers = []  # Load compression markers

        with h5py.File(f'{traj_dir}/{get_traj_file_name(config)}.hdf5', 'r') as f:
            for i in env_idx:
                states.append(f[f'{i}']['states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                actions.append(f[f'{i}']['actions'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                rewards.append(f[f'{i}']['rewards'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                next_states.append(f[f'{i}']['next_states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                
                # Load compress markers if available
                if 'compress_markers' in f[f'{i}']:
                    compress_markers.append(f[f'{i}']['compress_markers'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                    
        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.rewards = np.concatenate(rewards, axis=0)
        self.next_states = np.concatenate(next_states, axis=0)
        
        if len(compress_markers) > 0:
            self.compress_markers = np.concatenate(compress_markers, axis=0)
        else:
            # If no markers available, create dummy markers (all zeros = normal transitions)
            self.compress_markers = np.zeros_like(self.actions, dtype=np.int32)
        
        # Build compression region index for efficient sampling in compressed mode
        if self.use_compressed_context:
            self._build_compression_index()
    
    def _build_compression_index(self):
        """Build index of compression regions for efficient sampling."""
        self.compression_regions = []
        
        for hist_idx in range(len(self.states)):
            markers = self.compress_markers[hist_idx]
            regions = []
            start_idx = None
            
            for t in range(len(markers)):
                if markers[t] == 1:  # compress_start
                    start_idx = t
                elif markers[t] == 2 and start_idx is not None:  # compress_end
                    regions.append((start_idx, t + 1))  # [start, end)
                    start_idx = None
            
            # Handle incomplete region at the end
            if start_idx is not None:
                regions.append((start_idx, len(markers)))
            
            self.compression_regions.append(regions)
    
    def __len__(self):
        if self.use_compressed_context:
            # In compressed mode, sample is per compression region
            total_samples = 0
            for regions in self.compression_regions:
                for start, end in regions:
                    region_len = end - start
                    if region_len >= self.n_transit:
                        total_samples += region_len - self.n_transit + 1
            return total_samples
        else:
            return (len(self.states[0]) - self.n_transit + 1) * len(self.states)
    
    def _get_compressed_context_sample(self, idx):
        """Sample a sequence that spans current and optionally previous compression region."""
        # Map linear index to (history_idx, region_idx, position_in_region)
        cumulative = 0
        for hist_idx, regions in enumerate(self.compression_regions):
            for region_idx, (start, end) in enumerate(regions):
                region_len = end - start
                if region_len >= self.n_transit:
                    num_samples = region_len - self.n_transit + 1
                    if idx < cumulative + num_samples:
                        position = idx - cumulative
                        transition_idx = start + position
                        
                        # Sample context from current region and optionally previous region
                        context_start = max(0, transition_idx - self.n_transit + 1)
                        
                        # If we need more context and there's a previous region, include it
                        if transition_idx - context_start < self.n_transit - 1 and region_idx > 0:
                            prev_start, prev_end = regions[region_idx - 1]
                            # Take from end of previous region
                            needed = (self.n_transit - 1) - (transition_idx - context_start)
                            prev_context_start = max(prev_start, prev_end - needed)
                            
                            # Concatenate previous region context + current region context
                            prev_states = self.states[hist_idx, prev_context_start:prev_end]
                            prev_actions = self.actions[hist_idx, prev_context_start:prev_end]
                            prev_rewards = self.rewards[hist_idx, prev_context_start:prev_end]
                            prev_next_states = self.next_states[hist_idx, prev_context_start:prev_end]
                            prev_markers = self.compress_markers[hist_idx, prev_context_start:prev_end]
                            
                            curr_states = self.states[hist_idx, start:transition_idx]
                            curr_actions = self.actions[hist_idx, start:transition_idx]
                            curr_rewards = self.rewards[hist_idx, start:transition_idx]
                            curr_next_states = self.next_states[hist_idx, start:transition_idx]
                            curr_markers = self.compress_markers[hist_idx, start:transition_idx]
                            
                            context_states = np.concatenate([prev_states, curr_states])
                            context_actions = np.concatenate([prev_actions, curr_actions])
                            context_rewards = np.concatenate([prev_rewards, curr_rewards])
                            context_next_states = np.concatenate([prev_next_states, curr_next_states])
                            context_markers = np.concatenate([prev_markers, curr_markers])
                        else:
                            context_states = self.states[hist_idx, context_start:transition_idx]
                            context_actions = self.actions[hist_idx, context_start:transition_idx]
                            context_rewards = self.rewards[hist_idx, context_start:transition_idx]
                            context_next_states = self.next_states[hist_idx, context_start:transition_idx]
                            context_markers = self.compress_markers[hist_idx, context_start:transition_idx]
                        
                        # Pad if necessary
                        if len(context_states) < self.n_transit - 1:
                            pad_len = self.n_transit - 1 - len(context_states)
                            context_states = np.concatenate([
                                np.zeros((pad_len,) + context_states.shape[1:], dtype=context_states.dtype),
                                context_states
                            ])
                            context_actions = np.concatenate([
                                np.zeros((pad_len,), dtype=context_actions.dtype),
                                context_actions
                            ])
                            context_rewards = np.concatenate([
                                np.zeros((pad_len,), dtype=context_rewards.dtype),
                                context_rewards
                            ])
                            context_next_states = np.concatenate([
                                np.zeros((pad_len,) + context_next_states.shape[1:], dtype=context_next_states.dtype),
                                context_next_states
                            ])
                            context_markers = np.concatenate([
                                np.zeros((pad_len,), dtype=context_markers.dtype),
                                context_markers
                            ])
                        
                        return {
                            'query_states': self.states[hist_idx, transition_idx],
                            'target_actions': self.actions[hist_idx, transition_idx],
                            'states': context_states[-(self.n_transit-1):],
                            'actions': context_actions[-(self.n_transit-1):],
                            'rewards': context_rewards[-(self.n_transit-1):],
                            'next_states': context_next_states[-(self.n_transit-1):],
                            'compress_markers': context_markers[-(self.n_transit-1):],
                            'target_token_types': 0,  # 0=action (normal case)
                        }
                    
                    cumulative += num_samples
        
        raise IndexError(f"Index {idx} out of range")
    
    def __getitem__(self, i):
        if self.use_compressed_context:
            traj = self._get_compressed_context_sample(i)
        else:
            history_idx = i // (len(self.states[0]) - self.n_transit + 1)
            transition_idx = i % (len(self.states[0]) - self.n_transit + 1)
                
            traj = {
                'query_states': self.states[history_idx, transition_idx + self.n_transit - 1],
                'target_actions': self.actions[history_idx, transition_idx + self.n_transit - 1],
                'states': self.states[history_idx, transition_idx:transition_idx + self.n_transit - 1],
                'actions': self.actions[history_idx, transition_idx:transition_idx + self.n_transit - 1],
                'rewards': self.rewards[history_idx, transition_idx:transition_idx + self.n_transit - 1],
                'next_states': self.next_states[history_idx, transition_idx:transition_idx + self.n_transit - 1],
            }
        
        if self.dynamics:
            if self.use_compressed_context:
                hist_idx = 0  # Will be computed properly in _get_compressed_context_sample
                trans_idx = 0
            else:
                hist_idx = i // (len(self.states[0]) - self.n_transit + 1)
                trans_idx = i % (len(self.states[0]) - self.n_transit + 1) + self.n_transit - 1
            
            if not self.use_compressed_context:
                traj.update({
                    'target_next_states': self.next_states[hist_idx, trans_idx],
                    'target_rewards': self.rewards[hist_idx, trans_idx],
                })
        
        return traj