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
        """Build index of compression regions for efficient sampling.
        
        Since PPO uses fixed n_steps=compress_interval, we can directly compute
        region boundaries without searching through markers.
        """
        self.compression_regions = []
        self.valid_samples = []  # (hist_idx, region_idx, position_in_region, has_prev)
        
        for hist_idx in range(len(self.states)):
            traj_len = len(self.states[hist_idx])
            n_regions = traj_len // self.compress_interval
            
            regions = []
            for r in range(n_regions):
                start = r * self.compress_interval
                end = start + self.compress_interval
                regions.append((start, end))
            
            self.compression_regions.append(regions)
            
            # Pre-compute all valid sample indices
            for region_idx in range(len(regions)):
                region_start = regions[region_idx][0]
                region_end = regions[region_idx][1]
                # Sample at each position in the region
                for pos in range(region_start, region_end):
                    # First region: no previous region (matches eval cold start)
                    # Other regions: include previous region
                    has_prev = (region_idx > 0)
                    self.valid_samples.append((hist_idx, region_idx, pos, has_prev))
    
    def __len__(self):
        if self.use_compressed_context:
            return len(self.valid_samples)
        else:
            return (len(self.states[0]) - self.n_transit + 1) * len(self.states)
    
    def _get_compressed_context_sample(self, idx):
        """Fast sample using pre-computed index.
        
        The sequence structure depends on has_prev:
        - With prev: <compress> prev_region(80 steps) </compress> <compress> curr_region(up to query) query_state
        - Without prev: <compress> curr_region(up to query) query_state
        """
        hist_idx, region_idx, transition_idx, has_prev = self.valid_samples[idx]
        regions = self.compression_regions[hist_idx]
        
        # Get current region up to query position
        curr_start = regions[region_idx][0]
        curr_len = transition_idx - curr_start  # Variable: 0 to compress_interval-1
        
        if has_prev:
            # Get previous complete region (always 80 steps)
            prev_start, prev_end = regions[region_idx - 1]
            prev_len = prev_end - prev_start  # Always compress_interval
            
            # Extract sequences
            prev_states = self.states[hist_idx, prev_start:prev_end]
            prev_actions = self.actions[hist_idx, prev_start:prev_end]
            prev_rewards = self.rewards[hist_idx, prev_start:prev_end]
            prev_next_states = self.next_states[hist_idx, prev_start:prev_end]
            
            if curr_len > 0:
                curr_states = self.states[hist_idx, curr_start:transition_idx]
                curr_actions = self.actions[hist_idx, curr_start:transition_idx]
                curr_rewards = self.rewards[hist_idx, curr_start:transition_idx]
                curr_next_states = self.next_states[hist_idx, curr_start:transition_idx]
                
                # Concatenate regions
                context_states = np.concatenate([prev_states, curr_states])
                context_actions = np.concatenate([prev_actions, curr_actions])
                context_rewards = np.concatenate([prev_rewards, curr_rewards])
                context_next_states = np.concatenate([prev_next_states, curr_next_states])
            else:
                context_states = prev_states
                context_actions = prev_actions
                context_rewards = prev_rewards
                context_next_states = prev_next_states
        else:
            # No previous region (first region, matches eval cold start)
            prev_len = 0
            
            if curr_len > 0:
                context_states = self.states[hist_idx, curr_start:transition_idx]
                context_actions = self.actions[hist_idx, curr_start:transition_idx]
                context_rewards = self.rewards[hist_idx, curr_start:transition_idx]
                context_next_states = self.next_states[hist_idx, curr_start:transition_idx]
            else:
                # Very first step: no context at all
                context_states = np.zeros((0,) + self.states.shape[2:], dtype=self.states.dtype)
                context_actions = np.zeros((0,), dtype=self.actions.dtype)
                context_rewards = np.zeros((0,), dtype=self.rewards.dtype)
                context_next_states = np.zeros((0,) + self.next_states.shape[2:], dtype=self.next_states.dtype)
        
        # Determine target: predict compress_end at region boundaries
        steps_in_current = transition_idx - curr_start
        if steps_in_current == self.compress_interval - 1:
            # Last step of region: predict compress_end
            target_token_type = 2
            target_action = -1
        else:
            # Normal step: predict action
            target_token_type = 0
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