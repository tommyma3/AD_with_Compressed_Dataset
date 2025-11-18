"""
Compressed History Logger Callback for Algorithm Distillation.

This callback identifies on-policy transitions (high probability under current policy)
and marks them with compression tokens for efficient context learning.
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import torch


class CompressedHistoryLoggerCallback(BaseCallback):
    """
    Callback that logs history with compression markers around on-policy transitions.
    
    Compression Strategy:
    - Identify transitions where current policy agrees (on-policy, high confidence)
    - Mark these segments with <compress> and </compress> tokens
    - The transformer will learn to use these as compressed context without
      explicitly predicting actions within compressed segments
    """
    
    def __init__(self, env_name, env_idx, history=None, 
                 on_policy_threshold=0.7, compression_window=10):
        """
        Args:
            env_name: Environment name
            env_idx: Environment index for multi-env training
            history: Shared dict for storing trajectories (multiprocessing.Manager().dict())
            on_policy_threshold: Probability threshold to consider a transition "on-policy"
            compression_window: Minimum consecutive on-policy steps to compress
        """
        super().__init__()
        self.env_name = env_name
        self.env_idx = env_idx
        self.history = history
        
        # Compression parameters
        self.on_policy_threshold = on_policy_threshold
        self.compression_window = compression_window
        
        # Data storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.action_probs = []  # Store action probabilities to determine on-policy
        
        # Episode tracking
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        """Called at each environment step."""
        # Capture state, action, reward
        self.states.append(self.locals["obs_tensor"].cpu().numpy())
        self.next_states.append(self.locals["new_obs"])
        self.actions.append(self.locals["actions"])
        self.rewards.append(self.locals["rewards"].copy())
        self.dones.append(self.locals["dones"])
        
        # Get action probabilities from the policy
        # This tells us how "on-policy" this transition is
        obs_tensor = self.locals["obs_tensor"]
        with torch.no_grad():
            # Get action distribution from current policy
            action_dist = self.model.policy.get_distribution(obs_tensor)
            action_probs = action_dist.distribution.probs.cpu().numpy()
            
            # Store probability of the action actually taken
            actions = self.locals["actions"]
            taken_action_probs = action_probs[np.arange(len(actions)), actions]
            self.action_probs.append(taken_action_probs)
        
        # Episode reward tracking
        self.episode_rewards.append(self.locals['rewards'])
        
        if self.locals['dones'][0]:
            mean_reward = np.mean(np.mean(self.episode_rewards, axis=0))
            self.logger.record('rollout/mean_reward', mean_reward)
            self.episode_rewards = []
        
        return True
    
    def _identify_compression_segments(self, action_probs_array):
        """
        Identify segments of on-policy transitions to compress.
        
        Returns:
            compression_mask: Boolean array indicating which transitions should be compressed
            segment_boundaries: List of (start, end) tuples for each compressed segment
        """
        n_steps = len(action_probs_array)
        compression_mask = np.zeros(n_steps, dtype=bool)
        segment_boundaries = []
        
        # Find high-confidence (on-policy) transitions
        is_on_policy = action_probs_array >= self.on_policy_threshold
        
        # Find consecutive on-policy segments
        segment_start = None
        for i in range(n_steps):
            if is_on_policy[i]:
                if segment_start is None:
                    segment_start = i
            else:
                if segment_start is not None:
                    # End of segment
                    segment_length = i - segment_start
                    if segment_length >= self.compression_window:
                        # Mark this segment for compression
                        compression_mask[segment_start:i] = True
                        segment_boundaries.append((segment_start, i))
                    segment_start = None
        
        # Handle segment extending to end
        if segment_start is not None:
            segment_length = n_steps - segment_start
            if segment_length >= self.compression_window:
                compression_mask[segment_start:] = True
                segment_boundaries.append((segment_start, n_steps))
        
        return compression_mask, segment_boundaries
    
    def _on_training_end(self):
        """Called when training ends. Save compressed history."""
        # Convert lists to arrays
        states = np.array(self.states, dtype=np.int32)
        actions = np.array(self.actions, dtype=np.int32)
        rewards = np.array(self.rewards, dtype=np.int32)
        next_states = np.array(self.next_states, dtype=np.int32)
        dones = np.array(self.dones, dtype=bool)
        action_probs = np.array(self.action_probs, dtype=np.float32)
        
        # Identify which transitions should be compressed
        compression_mask, segment_boundaries = self._identify_compression_segments(
            action_probs.squeeze() if action_probs.ndim > 1 else action_probs
        )
        
        # Store everything in history
        self.history[self.env_idx] = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'action_probs': action_probs,
            'compression_mask': compression_mask.astype(np.uint8),  # Save as uint8 for HDF5
            'compression_segments': np.array(segment_boundaries, dtype=np.int32) if segment_boundaries else np.array([]).reshape(0, 2).astype(np.int32)
        }
        
        # Log compression statistics
        if len(segment_boundaries) > 0:
            total_compressed = compression_mask.sum()
            compression_ratio = total_compressed / len(compression_mask)
            self.logger.record('compression/num_segments', len(segment_boundaries))
            self.logger.record('compression/ratio', compression_ratio)
            self.logger.record('compression/total_steps_compressed', int(total_compressed))
            
            avg_segment_length = total_compressed / len(segment_boundaries)
            self.logger.record('compression/avg_segment_length', avg_segment_length)
