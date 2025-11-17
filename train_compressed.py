"""
Train Compressed Algorithm Distillation Model.

This script trains an AD model with compression tokens for efficient learning.
"""
from datetime import datetime
import os
import os.path as path
from glob import glob

from accelerate import Accelerator
import yaml
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from dataset_compressed import CompressedADDataset
from env import SAMPLE_ENVIRONMENT, make_env
from model.compressed_ad import CompressedAD
from utils import get_config, get_data_loader, log_in_context, next_dataloader
from transformers import get_cosine_schedule_with_warmup

import multiprocessing
from tqdm import tqdm
from stable_baselines3.common.vec_env import SubprocVecEnv


def ad_compressed_collate_fn(batch, grid_size):
    """Custom collate function for compressed AD dataset."""
    import torch.nn.functional as F
    import numpy as np
    from functools import partial
    
    res = {}
    res['query_states'] = torch.tensor(
        np.array([item['query_states'] for item in batch]), 
        requires_grad=False, dtype=torch.float
    )
    res['target_actions'] = torch.tensor(
        np.array([item['target_actions'] for item in batch]), 
        requires_grad=False, dtype=torch.long
    )
    res['states'] = torch.tensor(
        np.array([item['states'] for item in batch]), 
        requires_grad=False, dtype=torch.float
    )
    res['actions'] = F.one_hot(
        torch.tensor(np.array([item['actions'] for item in batch]), 
                    requires_grad=False, dtype=torch.long),
        num_classes=5
    )
    res['rewards'] = torch.tensor(
        np.array([item['rewards'] for item in batch]), 
        dtype=torch.float, requires_grad=False
    )
    res['next_states'] = torch.tensor(
        np.array([item['next_states'] for item in batch]), 
        requires_grad=False, dtype=torch.float
    )
    
    # Compression mask
    res['compression_mask'] = torch.tensor(
        np.array([item['compression_mask'] for item in batch]),
        dtype=torch.bool, requires_grad=False
    )
    res['has_compressed_context'] = torch.tensor(
        np.array([item['has_compressed_context'] for item in batch]),
        dtype=torch.bool, requires_grad=False
    )
    
    if 'target_next_states' in batch[0].keys():
        from env import map_dark_states
        res['target_next_states'] = map_dark_states(
            torch.tensor(np.array([item['target_next_states'] for item in batch]), 
                        dtype=torch.long, requires_grad=False), 
            grid_size=grid_size
        )
        res['target_rewards'] = torch.tensor(
            np.array([item['target_rewards'] for item in batch]), 
            dtype=torch.long, requires_grad=False
        )
    
    return res


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    # Load config
    config = get_config('./config/env/darkroom.yaml')
    config.update(get_config('./config/algorithm/ppo_darkroom.yaml'))
    config.update(get_config('./config/model/ad_dr.yaml'))
    
    # Update model name
    config['model'] = 'CompressedAD'
    
    log_dir = path.join('./runs', f"{config['model']}-{config['env']}-seed{config['env_split_seed']}")
    writer = SummaryWriter(log_dir, flush_secs=15)
    
    config['log_dir'] = log_dir
    config_save_path = path.join(config['log_dir'], 'config.yaml')
    
    try:
        with open(config_save_path, 'r') as f:
            f.read(1)
            config_exists = True
    except FileNotFoundError:
        config_exists = False
    
    if config_exists:
        print(f'WARNING: {log_dir} already exists. Skipping...')
        exit(0)
    
    config['traj_dir'] = './datasets'
    config['mixed_precision'] = 'fp32'
    
    # Setup accelerator
    accelerator = Accelerator(mixed_precision='no')
    config['device'] = accelerator.device
    print('Using Device: ', config['device'])
    
    # Create compressed AD model
    model = CompressedAD(config)
    
    load_start_time = datetime.now()
    print(f'Data loading started at {load_start_time}')
    
    # Load compressed datasets
    train_dataset = CompressedADDataset(
        config, config['traj_dir'], 'train',
        config['train_n_stream'], config['train_source_timesteps']
    )
    test_dataset = CompressedADDataset(
        config, config['traj_dir'], 'test',
        1, config['train_source_timesteps']
    )
    
    # Create data loaders with custom collate
    from functools import partial
    from torch.utils.data import DataLoader
    
    train_collate = partial(ad_compressed_collate_fn, grid_size=config['grid_size'])
    test_collate = partial(ad_compressed_collate_fn, grid_size=config['grid_size'])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=True,
        collate_fn=train_collate,
        num_workers=config['num_workers'],
        persistent_workers=True
    )
    train_dataloader = next_dataloader(train_dataloader)
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['test_batch_size'],
        shuffle=False,
        collate_fn=test_collate,
        num_workers=config['num_workers'],
        persistent_workers=True
    )
    
    load_end_time = datetime.now()
    print()
    print(f'Data loading ended at {load_end_time}')
    print(f'Elapsed time: {load_end_time - load_start_time}')
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['lr'],
        betas=(config['beta1'], config['beta2']),
        weight_decay=config['weight_decay']
    )
    lr_sched = get_cosine_schedule_with_warmup(
        optimizer, config['num_warmup_steps'], config['train_timesteps']
    )
    step = 0
    
    # Load checkpoint if exists
    ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
    if len(ckpt_paths) > 0:
        ckpt_path = ckpt_paths[-1]
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_sched.load_state_dict(ckpt['lr_sched'])
        step = ckpt['step']
        print(f'Checkpoint loaded from {ckpt_path}')
    
    # Setup evaluation environments
    env_name = config['env']
    train_env_args, test_env_args = SAMPLE_ENVIRONMENT[env_name](config)
    train_env_args = train_env_args[:10]
    test_env_args = test_env_args[:10]
    env_args = train_env_args + test_env_args
    
    if env_name == "darkroom":
        envs = SubprocVecEnv([make_env(config, goal=arg) for arg in env_args])
    else:
        raise NotImplementedError('Environment not supported')
    
    # Prepare for training
    model, optimizer, train_dataloader, lr_sched = accelerator.prepare(
        model, optimizer, train_dataloader, lr_sched
    )
    
    start_time = datetime.now()
    print(f'Training started at {start_time}')
    
    with tqdm(total=config['train_timesteps'], position=0, leave=True, disable=False) as pbar:
        pbar.update(step)
        
        while True:
            batch = next(train_dataloader)
            step += 1
            
            with accelerator.autocast():
                output = model(batch)
            
            loss = output['loss_action']
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            if not accelerator.optimizer_step_was_skipped:
                lr_sched.step()
            
            pbar.set_postfix(loss=loss.item())
            
            # Logging
            if step % config['summary_interval'] == 0:
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/loss_action', output['loss_action'], step)
                writer.add_scalar('train/lr', lr_sched.get_last_lr()[0], step)
                writer.add_scalar('train/acc_action', output['acc_action'].item(), step)
            
            # Evaluation
            if step % config['eval_interval'] == 0:
                torch.cuda.empty_cache()
                model.eval()
                eval_start_time = datetime.now()
                print(f'Evaluating started at {eval_start_time}')
                
                with torch.no_grad():
                    test_loss_action = 0.0
                    test_acc_action = 0.0
                    test_cnt = 0
                    
                    for j, batch in enumerate(test_dataloader):
                        output = model(batch)
                        cnt = len(batch['states'])
                        test_loss_action += output['loss_action'].item() * cnt
                        test_acc_action += output['acc_action'].item() * cnt
                        test_cnt += cnt
                
                writer.add_scalar('test/loss_action', test_loss_action / test_cnt, step)
                writer.add_scalar('test/acc_action', test_acc_action / test_cnt, step)
                
                eval_end_time = datetime.now()
                print()
                print(f'Evaluating ended at {eval_end_time}')
                print(f'Elapsed time: {eval_end_time - eval_start_time}')
                model.train()
                torch.cuda.empty_cache()
            
            pbar.update(1)
            
            # Save checkpoint
            if step % config['ckpt_interval'] == 0:
                ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
                for ckpt_path in ckpt_paths:
                    os.remove(ckpt_path)
                
                new_ckpt_path = path.join(config['log_dir'], f'ckpt-{step}.pt')
                
                torch.save({
                    'step': step,
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': lr_sched.state_dict(),
                }, new_ckpt_path)
                print(f'\nCheckpoint saved to {new_ckpt_path}')
            
            if step >= config['train_timesteps']:
                break
    
    writer.flush()
    envs.close()
    
    end_time = datetime.now()
    print()
    print(f'Training ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')
