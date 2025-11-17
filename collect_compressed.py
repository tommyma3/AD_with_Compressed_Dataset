"""
Collect compressed training data for Algorithm Distillation.

This script collects trajectories with compression markers for on-policy segments.
"""
import os
from datetime import datetime
import multiprocessing

from env import SAMPLE_ENVIRONMENT, make_env
from algorithm import ALGORITHM
from algorithm.compressed_history_callback import CompressedHistoryLoggerCallback
import h5py
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv
from utils import get_config, get_traj_file_name


def worker(arg, config, traj_dir, env_idx, history, file_name):
    """Worker function for parallel trajectory collection."""
    if config['env'] == 'darkroom':
        env = DummyVecEnv([make_env(config, goal=arg)] * config['n_stream'])
    else:
        raise ValueError('Invalid environment')
    
    alg_name = config['alg']
    seed = config['alg_seed'] + env_idx
    config['device'] = 'cpu'
    
    # Use compressed history callback
    alg = ALGORITHM[alg_name](config, env, seed, traj_dir)
    callback = CompressedHistoryLoggerCallback(
        env_name=config['env'],
        env_idx=env_idx,
        history=history,
        on_policy_threshold=config.get('on_policy_threshold', 0.7),
        compression_window=config.get('compression_window', 10)
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
    multiprocessing.set_start_method('spawn', force=True)
    
    # Load configuration
    config = get_config("config/env/darkroom.yaml")
    config.update(get_config("config/algorithm/ppo_darkroom.yaml"))
    
    # Add compression config
    config['on_policy_threshold'] = 0.7  # Probability threshold for on-policy
    config['compression_window'] = 10     # Min consecutive steps to compress
    
    # Setup directories
    if not os.path.exists("datasets"):
        os.makedirs("datasets", exist_ok=True)
    
    traj_dir = 'datasets'
    
    # Sample environments
    train_args, test_args = SAMPLE_ENVIRONMENT[config['env']](config, shuffle=False)
    total_args = train_args + test_args
    n_envs = len(total_args)
    
    # Output file path
    file_name = get_traj_file_name(config) + '_compressed'
    path = f'datasets/{file_name}.hdf5'
    
    # Check if file exists (prevent overwrite with w- mode)
    if os.path.exists(path):
        response = input(f"File {path} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            exit(0)
        os.remove(path)
    
    start_time = datetime.now()
    print(f'Compressed data collection started at {start_time}')
    print(f'Configuration:')
    print(f'  On-policy threshold: {config["on_policy_threshold"]}')
    print(f'  Compression window: {config["compression_window"]}')
    print(f'  Number of environments: {n_envs}')
    print(f'  Output file: {path}')
    print()
    
    with multiprocessing.Manager() as manager:
        history = manager.dict()
        
        # Collect data in parallel
        with multiprocessing.Pool(processes=config['n_process']) as pool:
            pool.starmap(worker, [
                (total_args[i], config, traj_dir, i, history, file_name) 
                for i in range(n_envs)
            ])
        
        # Save compressed history to HDF5
        print(f'\nSaving compressed data to {path}...')
        
        with h5py.File(path, 'w') as f:
            # Save metadata
            f.attrs['on_policy_threshold'] = config['on_policy_threshold']
            f.attrs['compression_window'] = config['compression_window']
            f.attrs['n_envs'] = n_envs
            f.attrs['collection_date'] = start_time.isoformat()
            
            total_steps = 0
            total_compressed = 0
            
            for i in range(n_envs):
                if i not in history:
                    print(f'Warning: Environment {i} has no data (worker may have failed)')
                    continue
                
                env_group = f.create_group(f'{i}')
                
                # Save all datasets with compression for efficiency
                for key, value in history[i].items():
                    if isinstance(value, np.ndarray) and value.size > 100:
                        # Use gzip compression for large arrays
                        env_group.create_dataset(
                            key, 
                            data=value,
                            compression='gzip',
                            compression_opts=4
                        )
                    else:
                        env_group.create_dataset(key, data=value)
                
                # Track statistics
                if 'compression_mask' in history[i]:
                    n_steps = len(history[i]['compression_mask'])
                    n_compressed = history[i]['compression_mask'].sum()
                    total_steps += n_steps
                    total_compressed += n_compressed
                    
                    env_group.attrs['n_steps'] = n_steps
                    env_group.attrs['n_compressed'] = int(n_compressed)
                    env_group.attrs['compression_ratio'] = float(n_compressed / n_steps) if n_steps > 0 else 0.0
            
            # Save global statistics
            if total_steps > 0:
                f.attrs['total_steps'] = total_steps
                f.attrs['total_compressed_steps'] = total_compressed
                f.attrs['overall_compression_ratio'] = float(total_compressed / total_steps)
        
        print(f'Saved successfully!')
        print(f'\nCompression Statistics:')
        print(f'  Total steps collected: {total_steps:,}')
        print(f'  Steps marked for compression: {total_compressed:,}')
        print(f'  Overall compression ratio: {total_compressed/total_steps*100:.1f}%')
    
    end_time = datetime.now()
    print()
    print(f'Collection ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')
