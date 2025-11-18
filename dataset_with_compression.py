"""
Dataset loader with compression token insertion.
Inserts <compress> and \compress tokens around on-policy segments.
"""
from torch.utils.data import Dataset
import numpy as np
from utils import get_traj_file_name
import h5py
import random
from einops import rearrange, repeat


# Special token IDs
COMPRESS_START = -1  # <compress>
COMPRESS_END = -2     # \compress
PAD_TOKEN = -3        # padding


class CompressedADDataset(Dataset):
    """
    Dataset that inserts compression tokens around on-policy segments.
    The transformer learns to predict actions using context inside compression tokens.
    """
    def __init__(self, config, traj_dir, mode='train', n_stream=None, source_timesteps=None):
        self.config = config
        self.env = config['env']
        self.n_transit = config['n_transit']
        self.dynamics = config['dynamics']
        
        # Compression tokens
        self.compress_start = COMPRESS_START
        self.compress_end = COMPRESS_END
        self.pad_token = PAD_TOKEN
        
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
        compression_masks = []

        file_path = f'{traj_dir}/{get_traj_file_name(config)}_compressed.hdf5'
        with h5py.File(file_path, 'r') as f:
            for i in env_idx:
                states.append(f[f'{i}']['states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                actions.append(f[f'{i}']['actions'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                rewards.append(f[f'{i}']['rewards'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                next_states.append(f[f'{i}']['next_states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                compression_masks.append(f[f'{i}']['compression_mask'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                    
        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.rewards = np.concatenate(rewards, axis=0)
        self.next_states = np.concatenate(next_states, axis=0)
        self.compression_masks = np.concatenate(compression_masks, axis=0)
    
    def __len__(self):
        return (len(self.states[0]) - self.n_transit + 1) * len(self.states)
    
    def _insert_compression_tokens(self, trajectory_slice, mask_slice):
        """
        Insert <compress> and \compress tokens around on-policy segments.
        
        Args:
            trajectory_slice: Dictionary with states, actions, rewards, next_states
            mask_slice: Boolean array indicating which timesteps are in compression segments
            
        Returns:
            Modified trajectory with compression tokens inserted
        """
        # Find segment boundaries
        segments = []
        in_segment = False
        segment_start = None
        
        for i, is_compressed in enumerate(mask_slice):
            if is_compressed and not in_segment:
                # Start of compression segment
                segment_start = i
                in_segment = True
            elif not is_compressed and in_segment:
                # End of compression segment
                segments.append((segment_start, i))
                in_segment = False
        
        # Handle case where segment extends to end
        if in_segment:
            segments.append((segment_start, len(mask_slice)))
        
        return segments
    
    def __getitem__(self, i):
        history_idx = i // (len(self.states[0]) - self.n_transit + 1)
        transition_idx = i % (len(self.states[0]) - self.n_transit + 1)
        
        # Extract trajectory window
        states_slice = self.states[history_idx, transition_idx:transition_idx + self.n_transit - 1]
        actions_slice = self.actions[history_idx, transition_idx:transition_idx + self.n_transit - 1]
        rewards_slice = self.rewards[history_idx, transition_idx:transition_idx + self.n_transit - 1]
        next_states_slice = self.next_states[history_idx, transition_idx:transition_idx + self.n_transit - 1]
        mask_slice = self.compression_masks[history_idx, transition_idx:transition_idx + self.n_transit - 1]
        
        # Identify compression segments
        segments = self._insert_compression_tokens({}, mask_slice)
        
        traj = {
            'query_states': self.states[history_idx, transition_idx + self.n_transit - 1],
            'target_actions': self.actions[history_idx, transition_idx + self.n_transit - 1],
            'states': states_slice,
            'actions': actions_slice,
            'rewards': rewards_slice,
            'next_states': next_states_slice,
            'compression_mask': mask_slice.astype(np.float32),  # For model to use
            'compression_segments': segments,  # List of (start, end) tuples
        }
        
        if self.dynamics:
            traj.update({
                'target_next_states': self.next_states[history_idx, transition_idx + self.n_transit - 1],
                'target_rewards': self.rewards[history_idx, transition_idx + self.n_transit - 1],
            })
        
        return traj
