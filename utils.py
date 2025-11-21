import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

from env import map_dark_states
from functools import partial
import matplotlib.pyplot as plt


def get_config(config_path):
    with open(config_path, 'r') as f:
        new_config = yaml.full_load(f)
    config = {}
    if 'include' in new_config:
        include_config = get_config(new_config['include'])
        config.update(include_config)
        del new_config['include']
    config.update(new_config)
    return config


def get_traj_file_name(config):
    if config["env"] == 'metaworld':
        task = config['task']
    else:
        task = config['env']

    path = f'history_{task}_{config["alg"]}_alg-seed{config["alg_seed"]}'

    return path

def ad_collate_fn(batch, grid_size):
    res = {}
    res['query_states'] = torch.tensor(np.array([item['query_states'] for item in batch]), requires_grad=False, dtype=torch.float)
    res['target_actions'] = torch.tensor(np.array([item['target_actions'] for item in batch]), requires_grad=False, dtype=torch.long)
    
    # Check if we're in compressed context mode
    if 'prev_region_len' in batch[0]:
        # Compressed context mode: prev_region can be 0 (first region) or fixed (compress_interval)
        # curr_region is variable (0 to compress_interval-1)
        # Handle both cases: with and without previous region
        
        # Find max length across all samples in batch (prev_region_len can vary!)
        batch_size = len(batch)
        max_len = max(len(item['states']) for item in batch)
        
        # Pre-allocate arrays for better performance
        state_shape = batch[0]['states'].shape[1:]
        states_array = np.zeros((batch_size, max_len) + state_shape, dtype=batch[0]['states'].dtype)
        actions_array = np.zeros((batch_size, max_len), dtype=batch[0]['actions'].dtype)
        rewards_array = np.zeros((batch_size, max_len), dtype=batch[0]['rewards'].dtype)
        next_states_array = np.zeros((batch_size, max_len) + state_shape, dtype=batch[0]['next_states'].dtype)
        
        # Fill arrays efficiently
        for i, item in enumerate(batch):
            seq_len = len(item['states'])
            states_array[i, :seq_len] = item['states']
            actions_array[i, :seq_len] = item['actions']
            rewards_array[i, :seq_len] = item['rewards']
            next_states_array[i, :seq_len] = item['next_states']
        
        res['states'] = torch.from_numpy(states_array).float()
        res['actions'] = F.one_hot(torch.from_numpy(actions_array).long(), num_classes=5)
        res['rewards'] = torch.from_numpy(rewards_array).float()
        res['next_states'] = torch.from_numpy(next_states_array).float()
        
        # Store region lengths as tensors for faster GPU transfer
        res['prev_region_len'] = [item['prev_region_len'] for item in batch]
        res['curr_region_len'] = [item['curr_region_len'] for item in batch]
    else:
        # Fixed length sequences - original logic
        res['states'] = torch.tensor(np.array([item['states'] for item in batch]), requires_grad=False, dtype=torch.float)
        res['actions'] = F.one_hot(torch.tensor(np.array([item['actions'] for item in batch]), requires_grad=False, dtype=torch.long), num_classes=5)
        res['rewards'] = torch.tensor(np.array([item['rewards'] for item in batch]), dtype=torch.float, requires_grad=False)
        res['next_states'] = torch.tensor(np.array([item['next_states'] for item in batch]), requires_grad=False, dtype=torch.float)
    
    # Handle target token types (for predicting compress tokens)
    if 'target_token_type' in batch[0]:
        res['target_token_type'] = torch.tensor(np.array([item['target_token_type'] for item in batch]), requires_grad=False, dtype=torch.long)
    
    # Legacy support for old compress_markers field
    if 'compress_markers' in batch[0]:
        res['compress_markers'] = torch.tensor(np.array([item['compress_markers'] for item in batch]), requires_grad=False, dtype=torch.long)
    
    if 'target_next_states' in batch[0].keys():
        res['target_next_states'] = map_dark_states(torch.tensor(np.array([item['target_next_states'] for item in batch]), dtype=torch.long, requires_grad=False), grid_size=grid_size)
        res['target_rewards'] = torch.tensor(np.array([item['target_rewards'] for item in batch]), dtype=torch.long, requires_grad=False)
        
    return res

def get_data_loader(dataset, batch_size, config, shuffle=True):
    collate_fn = partial(ad_collate_fn, grid_size=config['grid_size'])
    num_workers = config.get('num_workers', 4)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, 
                     num_workers=num_workers, 
                     persistent_workers=(num_workers > 0),
                     pin_memory=True,  # Faster CPU-to-GPU transfer
                     prefetch_factor=2 if num_workers > 0 else None)  # Prefetch batches

def log_in_context(values: np.ndarray, max_reward: int, episode_length: int, tag: str, title: str, xlabel: str, ylabel: str, step: int, success=None, writer=None) -> None:
    steps = np.arange(1, len(values[0])+1) * episode_length
    mean_value = values.mean(axis=0)
    
    plt.plot(steps, mean_value)
    
    if success is not None:
        success_rate = success.astype(np.float32).mean(axis=0)

        for i, (xi, yi) in enumerate(zip(steps, mean_value)):
            if (i+1) % 10 == 0:
                plt.annotate(f'{success_rate[i]:.2f}', (xi, yi))
        
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.ylim(-max_reward * 0.05, max_reward * 1.05)
    writer.add_figure(f'{tag}/mean', plt.gcf(), global_step=step)
    plt.close()

def next_dataloader(dataloader: DataLoader):
    """
    Makes the dataloader never end when the dataset is exhausted.
    This is done to remove the notion of an 'epoch' and to count only the amount
    of training steps.
    """
    while True:
        for batch in dataloader:
            yield batch