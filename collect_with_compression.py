"""
Data collection script with compression token marking.
Collects trajectories and marks on-policy segments for compression.
"""
import os
from datetime import datetime
import yaml
import multiprocessing

from env import SAMPLE_ENVIRONMENT, make_env, Darkroom, DarkroomPermuted
from algorithm import ALGORITHM
from algorithm.compressed_callback import CompressedHistoryCallback
import h5py
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv

from utils import get_config, get_traj_file_name


def worker(arg, config, traj_dir, env_idx, history, file_name):
    """Worker function for parallel data collection."""
    
    if config['env'] == 'darkroom':
        env = DummyVecEnv([make_env(config, goal=arg)] * config['n_stream'])
    else:
        raise ValueError('Invalid environment')
    
    alg_name = config['alg']
    seed = config['alg_seed'] + env_idx

    config['device'] = 'cpu'

    alg = ALGORITHM[alg_name](config, env, seed, traj_dir)
    
    # Use compressed callback with on-policy detection
    callback = CompressedHistoryCallback(
        config['env'], 
        env_idx, 
        history,
        on_policy_threshold=config.get('on_policy_threshold', 0.7),
        min_segment_length=config.get('compression_window', 10)
    )
    
    log_name = f'{file_name}_{env_idx}'
    
    alg.learn(
        total_timesteps=config['total_source_timesteps'],
        callback=callback,
        log_interval=1,
        tb_log_name=log_name,
        reset_num_timesteps=True,
        progress_bar=True
    )
    env.close()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    config = get_config("config/env/darkroom.yaml")
    config.update(get_config("config/algorithm/ppo_darkroom.yaml"))
    
    # Compression parameters
    config['on_policy_threshold'] = 0.7  # Probability threshold for on-policy detection
    config['compression_window'] = 10     # Minimum consecutive steps to mark as compressed

    if not os.path.exists("datasets"):
        os.makedirs("datasets", exist_ok=True)
        
    traj_dir = 'datasets'

    train_args, test_args = SAMPLE_ENVIRONMENT[config['env']](config, shuffle=False)
    total_args = train_args + test_args
    n_envs = len(total_args)

    file_name = get_traj_file_name(config)
    path = f'datasets/{file_name}_compressed.hdf5'
    
    if os.path.exists(path):
        print(f"Warning: {path} already exists. It will be overwritten.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            exit(0)
    
    start_time = datetime.now()
    print(f'Training started at {start_time}')
    print(f'Compression parameters:')
    print(f'  - On-policy threshold: {config["on_policy_threshold"]}')
    print(f'  - Minimum segment length: {config["compression_window"]}')

    with multiprocessing.Manager() as manager:
        history = manager.dict()

        # Create a pool with a maximum of n_workers
        with multiprocessing.Pool(processes=config['n_process']) as pool:
            # Map the worker function to the environments
            pool.starmap(worker, [(total_args[i], config, traj_dir, i, history, file_name) for i in range(n_envs)])

        # Save the history dictionary with compression data
        print("\nSaving compressed trajectories...")
        with h5py.File(path, 'w') as f:
            # Save metadata
            f.attrs['on_policy_threshold'] = config['on_policy_threshold']
            f.attrs['compression_window'] = config['compression_window']
            
            total_steps = 0
            compressed_steps = 0
            
            for i in range(n_envs):
                env_group = f.create_group(f'{i}')
                for key, value in history[i].items():
                    env_group.create_dataset(key, data=value, compression='gzip')
                
                # Calculate compression statistics
                mask = history[i]['compression_mask']
                total_steps += len(mask)
                compressed_steps += np.sum(mask)
            
            # Save overall statistics
            compression_ratio = compressed_steps / total_steps if total_steps > 0 else 0
            f.attrs['total_steps'] = total_steps
            f.attrs['compressed_steps'] = compressed_steps
            f.attrs['compression_ratio'] = compression_ratio
            
            print(f"\nCompression Statistics:")
            print(f"  Total steps: {total_steps}")
            print(f"  Compressed steps: {compressed_steps}")
            print(f"  Compression ratio: {compression_ratio*100:.1f}%")

    end_time = datetime.now()
    print()
    print(f'Training ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')
    print(f'Saved to: {path}')
