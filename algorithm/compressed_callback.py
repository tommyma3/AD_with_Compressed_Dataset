"""
Callback to track action probabilities and mark on-policy segments.
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CompressedHistoryCallback(BaseCallback):
    """
    Callback that tracks action probabilities to identify on-policy segments.
    These segments will be marked with compression tokens in the dataset.
    """
    def __init__(self, env_name, env_idx, history=None, on_policy_threshold=0.7, min_segment_length=10):
        super(CompressedHistoryCallback, self).__init__()
        self.env_name = env_name
        self.env_idx = env_idx
        self.history = history
        
        # Compression parameters
        self.on_policy_threshold = on_policy_threshold
        self.min_segment_length = min_segment_length

        # Standard trajectory data
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        # Track action probabilities for on-policy detection
        self.action_probs = []

        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Capture state, action, and reward at each step
        self.states.append(self.locals["obs_tensor"].cpu().numpy())
        self.next_states.append(self.locals["new_obs"])
        self.actions.append(self.locals["actions"])
        self.rewards.append(self.locals["rewards"].copy())
        self.dones.append(self.locals["dones"])
        
        # Capture action probabilities for the taken action
        # This is used to detect on-policy segments
        if hasattr(self.model, 'policy'):
            with np.errstate(invalid='ignore'):  # Ignore any division warnings
                import torch
                
                # Get action distribution from current policy
                obs_tensor = self.locals["obs_tensor"]
                actions = self.locals["actions"]
                
                # Convert actions to tensor if needed
                if not isinstance(actions, torch.Tensor):
                    actions_tensor = torch.as_tensor(actions, device=obs_tensor.device)
                else:
                    actions_tensor = actions
                
                # Get log probabilities
                _, log_prob, _ = self.model.policy.evaluate_actions(obs_tensor, actions_tensor)
                action_prob = np.exp(log_prob.detach().cpu().numpy())
                
                self.action_probs.append(action_prob)
        else:
            # Fallback if policy not available
            self.action_probs.append(np.ones_like(self.locals["rewards"]))

        self.episode_rewards.append(self.locals['rewards'])
        
        if self.locals['dones'][0]:
            mean_reward = np.mean(np.mean(self.episode_rewards, axis=0))
            self.logger.record('rollout/mean_reward', mean_reward)
            self.episode_rewards = []
                        
        return True

    def _identify_compression_segments(self, action_probs):
        """
        Identify segments where action probabilities are consistently high (on-policy).
        Returns a binary mask where 1 indicates a step inside a compression segment.
        """
        T = len(action_probs)
        compression_mask = np.zeros(T, dtype=bool)
        
        # Find consecutive on-policy segments
        is_on_policy = action_probs >= self.on_policy_threshold
        
        segment_start = None
        for i in range(T):
            if is_on_policy[i]:
                if segment_start is None:
                    segment_start = i
            else:
                if segment_start is not None:
                    # End of segment
                    segment_length = i - segment_start
                    if segment_length >= self.min_segment_length:
                        compression_mask[segment_start:i] = True
                    segment_start = None
        
        # Handle final segment
        if segment_start is not None:
            segment_length = T - segment_start
            if segment_length >= self.min_segment_length:
                compression_mask[segment_start:T] = True
        
        return compression_mask

    def _on_training_end(self):
        """Save trajectory data with compression mask."""
        action_probs_array = np.array(self.action_probs, dtype=np.float32).squeeze()
        compression_mask = self._identify_compression_segments(action_probs_array)
        
        self.history[self.env_idx] = {
            'states': np.array(self.states, dtype=np.int32),
            'actions': np.array(self.actions, dtype=np.int32),
            'rewards': np.array(self.rewards, dtype=np.int32),
            'next_states': np.array(self.next_states, dtype=np.int32),
            'dones': np.array(self.dones, dtype=np.bool_),
            'compression_mask': compression_mask.astype(np.uint8),  # 1 = inside compression segment
            'action_probs': action_probs_array,  # For analysis
        }
