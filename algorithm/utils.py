import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class HistoryLoggerCallback(BaseCallback):
    def __init__(self, env_name, env_idx, history=None, compress_interval=80):
        super(HistoryLoggerCallback, self).__init__()
        self.env_name = env_name
        self.env_idx = env_idx
        self.compress_interval = compress_interval  # n_steps from PPO config

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.compress_markers = []  # Track compression markers

        self.history = history

        self.episode_rewards = []
        self.episode_success = []
        
        self.steps_since_compress = 0
        self.insert_compress_start_next = True  # Start first compression region

    def _on_step(self) -> bool:
        # Capture state, action, and reward at each step
        self.states.append(self.locals["obs_tensor"].cpu().numpy())
        self.next_states.append(self.locals["new_obs"])
        self.actions.append(self.locals["actions"])

        self.rewards.append(self.locals["rewards"].copy())
        self.dones.append(self.locals["dones"])
        
        # Handle compression markers
        # compress_markers: 0=normal, 1=compress_start, 2=compress_end
        marker = np.zeros_like(self.locals["dones"], dtype=np.int32)
        
        if self.insert_compress_start_next:
            marker[:] = 1  # compress_start
            self.insert_compress_start_next = False
            self.steps_since_compress = 0
        
        self.steps_since_compress += 1
        
        # Check if we should insert compress_end marker
        if self.steps_since_compress >= self.compress_interval:
            marker[:] = 2  # compress_end
            self.insert_compress_start_next = True  # Next step will be compress_start
            self.steps_since_compress = 0
        
        self.compress_markers.append(marker)

        self.episode_rewards.append(self.locals['rewards'])
        
        if self.locals['dones'][0]:
            mean_reward = np.mean(np.mean(self.episode_rewards, axis=0))
            self.logger.record('rollout/mean_reward', mean_reward)
            self.episode_rewards = []
                        
        return True

    def _on_training_end(self):
        self.history[self.env_idx] = {
            'states': np.array(self.states, dtype=np.int32),
            'actions': np.array(self.actions, dtype=np.int32),
            'rewards': np.array(self.rewards, dtype=np.int32),
            'next_states': np.array(self.next_states, dtype=np.int32),
            'dones': np.array(self.dones, dtype=np.bool_),
            'compress_markers': np.array(self.compress_markers, dtype=np.int32)  # Save markers
        }