"""
Compressed Dataset Loader for Algorithm Distillation.

Loads trajectories with compression markers and creates training samples
where compressed segments are included as context but not prediction targets.
"""
from torch.utils.data import Dataset
import numpy as np
import h5py
import random
from einops import rearrange


# Special token IDs for compression markers
COMPRESS_START_TOKEN = -1  # Special ID for <compress>
COMPRESS_END_TOKEN = -2    # Special ID for </compress>
PAD_TOKEN = -3             # For padding if needed


class CompressedADDataset(Dataset):
    """
    Dataset for training Algorithm Distillation with compressed context.
    
    Each sample contains:
    - Context: sequence of (state, action, reward, next_state) with compression markers
    - Query state: the current state to predict action for
    - Target action: the action to predict
    - Compression mask: indicates which positions are inside <compress>...</compress>
    
    The model learns to:
    1. Use compressed segments as context
    2. Only predict actions for query states outside compression
    """
    
    def __init__(self, config, traj_dir, mode='train', n_stream=None, source_timesteps=None):
        self.config = config
        self.env = config['env']
        self.n_transit = config['n_transit']
        self.dynamics = config.get('dynamics', False)
        
        if self.env == 'darkroom':
            n_total_envs = config['grid_size'] ** 2
        else:
            raise ValueError('Invalid env')
        
        # Split environments
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
        
        # Load compressed data
        file_name = f"{traj_dir}/history_{self.env}_PPO_alg-seed{config['alg_seed']}_compressed.hdf5"
        
        states = []
        actions = []
        rewards = []
        next_states = []
        compression_masks = []
        
        with h5py.File(file_name, 'r') as f:
            # Load metadata
            self.on_policy_threshold = f.attrs.get('on_policy_threshold', 0.7)
            self.compression_window = f.attrs.get('compression_window', 10)
            
            for i in env_idx:
                if str(i) not in f:
                    continue
                    
                env_group = f[str(i)]
                
                # Load data
                env_states = env_group['states'][()]
                env_actions = env_group['actions'][()]
                env_rewards = env_group['rewards'][()]
                env_next_states = env_group['next_states'][()]
                env_compression_mask = env_group['compression_mask'][()].astype(bool)
                
                # Transpose to (stream, timestep, ...) format if needed
                if env_states.ndim == 3:
                    env_states = env_states.transpose(1, 0, 2)
                    env_next_states = env_next_states.transpose(1, 0, 2)
                if env_actions.ndim == 2:
                    env_actions = env_actions.transpose(1, 0)
                if env_rewards.ndim == 2:
                    env_rewards = env_rewards.transpose(1, 0)
                if env_compression_mask.ndim == 2:
                    env_compression_mask = env_compression_mask.transpose(1, 0)
                
                # Slice to requested length
                if n_stream is not None and source_timesteps is not None:
                    env_states = env_states[:n_stream, :source_timesteps]
                    env_actions = env_actions[:n_stream, :source_timesteps]
                    env_rewards = env_rewards[:n_stream, :source_timesteps]
                    env_next_states = env_next_states[:n_stream, :source_timesteps]
                    env_compression_mask = env_compression_mask[:n_stream, :source_timesteps]
                
                states.append(env_states)
                actions.append(env_actions)
                rewards.append(env_rewards)
                next_states.append(env_next_states)
                compression_masks.append(env_compression_mask)
        
        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.rewards = np.concatenate(rewards, axis=0)
        self.next_states = np.concatenate(next_states, axis=0)
        self.compression_masks = np.concatenate(compression_masks, axis=0)
        
        print(f'Loaded {mode} dataset: {len(self.states)} trajectories')
        print(f'  Compression ratio: {self.compression_masks.mean()*100:.1f}%')
    
    def __len__(self):
        # Number of possible query positions per trajectory
        return (self.states.shape[1] - self.n_transit + 1) * len(self.states)
    
    def __getitem__(self, i):
        """
        Get a training sample with compressed context.
        
        Returns a dict with:
        - query_states: state to predict action for
        - target_actions: ground truth action
        - states, actions, rewards, next_states: context history
        - compression_mask: which context positions are compressed
        - has_compressed_context: whether this sample includes compressed segments
        """
        history_idx = i // (self.states.shape[1] - self.n_transit + 1)
        transition_idx = i % (self.states.shape[1] - self.n_transit + 1)
        
        # Get context window
        context_start = transition_idx
        context_end = transition_idx + self.n_transit - 1
        query_idx = context_end
        
        traj = {
            # Query and target
            'query_states': self.states[history_idx, query_idx],
            'target_actions': self.actions[history_idx, query_idx],
            
            # Context history (n_transit - 1 steps)
            'states': self.states[history_idx, context_start:context_end],
            'actions': self.actions[history_idx, context_start:context_end],
            'rewards': self.rewards[history_idx, context_start:context_end],
            'next_states': self.next_states[history_idx, context_start:context_end],
            
            # Compression information
            'compression_mask': self.compression_masks[history_idx, context_start:context_end],
            'has_compressed_context': self.compression_masks[history_idx, context_start:context_end].any(),
        }
        
        if self.dynamics:
            traj.update({
                'target_next_states': self.next_states[history_idx, query_idx],
                'target_rewards': self.rewards[history_idx, query_idx],
            })
        
        return traj
