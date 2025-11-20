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
            # In compressed mode, sample at each position in regions that have a previous region
            total_samples = 0
            for regions in self.compression_regions:
                for region_idx, (start, end) in enumerate(regions):
                    if region_idx == 0:
                        continue  # Skip first region (no previous)
                    total_samples += end - start
            return total_samples
        else:
            return (len(self.states[0]) - self.n_transit + 1) * len(self.states)
    
    def _get_compressed_context_sample(self, idx):
        """Sample a sequence from one compression region + previous region, with markers as separate tokens.
        
        The sequence structure is:
        <compress> t1, t2, ..., tn </compress> <compress> tn+1, tn+2, ..., query_state
        
        Where compression tokens are INSERTED as separate sequence elements.
        """
        # Map linear index to (history_idx, region_idx, position_in_region)
        cumulative = 0
        for hist_idx, regions in enumerate(self.compression_regions):
            for region_idx, (start, end) in enumerate(regions):
                if region_idx == 0:
                    continue  # Skip first region (no previous region to pair with)
                
                region_len = end - start
                num_samples = region_len  # Sample at each position in the current region
                
                if idx < cumulative + num_samples:
                    position = idx - cumulative
                    transition_idx = start + position
                    
                    # Get previous complete region
                    prev_start, prev_end = regions[region_idx - 1]
                    
                    # Build sequence with compression tokens as separate elements
                    # Previous region: <compress> transitions </compress>
                    prev_transitions = self.states[hist_idx, prev_start:prev_end]
                    prev_actions = self.actions[hist_idx, prev_start:prev_end]
                    prev_rewards = self.rewards[hist_idx, prev_start:prev_end]
                    prev_next_states = self.next_states[hist_idx, prev_start:prev_end]
                    
                    # Current region: <compress> transitions up to query
                    curr_transitions = self.states[hist_idx, start:transition_idx]
                    curr_actions = self.actions[hist_idx, start:transition_idx]
                    curr_rewards = self.rewards[hist_idx, start:transition_idx]
                    curr_next_states = self.next_states[hist_idx, start:transition_idx]
                    
                    # Concatenate: prev_region + curr_region (markers will be inserted in model)
                    context_states = np.concatenate([prev_transitions, curr_transitions]) if len(curr_transitions) > 0 else prev_transitions
                    context_actions = np.concatenate([prev_actions, curr_actions]) if len(curr_transitions) > 0 else prev_actions
                    context_rewards = np.concatenate([prev_rewards, curr_rewards]) if len(curr_transitions) > 0 else prev_rewards
                    context_next_states = np.concatenate([prev_next_states, curr_next_states]) if len(curr_transitions) > 0 else prev_next_states
                    
                    # Create marker array: [0, 0, ..., 0 (prev region), 0, 0, ..., 0 (curr region)]
                    # Markers will indicate where to insert compress tokens
                    prev_len = len(prev_transitions)
                    curr_len = len(curr_transitions)
                    context_markers = np.zeros(prev_len + curr_len, dtype=np.int32)
                    
                    # Mark boundaries: insert compress_start before each region, compress_end after prev region
                    # We'll use a special encoding: position of prev region start, prev region end, curr region start
                    region_boundaries = {
                        'prev_region_len': prev_len,
                        'curr_region_len': curr_len,
                    }
                    
                    # Determine target: are we predicting action or compress_end token?
                    # If we're at the end of a compression interval, predict compress_end
                    steps_in_current = transition_idx - start
                    if steps_in_current > 0 and steps_in_current % self.compress_interval == 0:
                        target_token_type = 2  # compress_end
                        target_action = -1  # No action predicted
                    else:
                        target_token_type = 0  # action
                        target_action = self.actions[hist_idx, transition_idx]
                    
                    return {
                        'query_states': self.states[hist_idx, transition_idx],
                        'target_actions': target_action,
                        'states': context_states,
                        'actions': context_actions,
                        'rewards': context_rewards,
                        'next_states': context_next_states,
                        'prev_region_len': prev_len,
                        'curr_region_len': curr_len,
                        'target_token_type': target_token_type,
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