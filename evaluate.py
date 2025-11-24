from datetime import datetime

from glob import glob

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*cudaGetDeviceCount.*')

import torch
import os.path as path

from env import SAMPLE_ENVIRONMENT, make_env
from model import MODEL
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
seed = 0
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate AD model on specific goal')
    parser.add_argument('--goal_idx', type=int, default=None, 
                       help='Index of goal to evaluate (0-7). If not specified, evaluates all goals.')
    parser.add_argument('--ckpt_dir', type=str, default='./runs/AD-darkroom-seed0',
                       help='Directory containing checkpoint files')
    args = parser.parse_args()
    
    ckpt_dir = args.ckpt_dir
    ckpt_paths = sorted(glob(path.join(ckpt_dir, 'ckpt-*.pt')))
    print(f'Using device: {device}')

    if len(ckpt_paths) > 0:
        ckpt_path = ckpt_paths[-1]
        ckpt = torch.load(ckpt_path, map_location=device)
        print(f'Checkpoint loaded from {ckpt_path}')
        config = ckpt['config']
    else:
        raise ValueError('No checkpoint found.')
    
    model_name = config['model']
    model = MODEL[model_name](config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    env_name = config['env']
    _, test_env_args = SAMPLE_ENVIRONMENT[env_name](config)

    print(f"Available test goals ({len(test_env_args)} total):")
    for i, goal in enumerate(test_env_args):
        print(f"  {i}: {goal}")
    
    # Select goal(s) to evaluate
    if args.goal_idx is not None:
        if args.goal_idx < 0 or args.goal_idx >= len(test_env_args):
            raise ValueError(f'Goal index {args.goal_idx} out of range [0, {len(test_env_args)-1}]')
        eval_goals = [test_env_args[args.goal_idx]]
        print(f"\nEvaluating single goal {args.goal_idx}: {eval_goals[0]}")
    else:
        eval_goals = test_env_args
        print(f"\nEvaluating all {len(eval_goals)} goals")

    if env_name == 'darkroom':
        envs = SubprocVecEnv([make_env(config, goal=arg) for arg in eval_goals])
    else:
        raise NotImplemented(f'Environment not supported')
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    start_time = datetime.now()
    print(f'Starting at {start_time}')

    with torch.no_grad():
        test_rewards = model.evaluate_in_context(vec_env=envs, eval_timesteps=config['horizon'] * 500)['reward_episode']
        save_path = path.join(ckpt_dir, f'eval_result_goal{args.goal_idx}.npy' if args.goal_idx is not None else 'eval_result.npy')
    
    end_time = datetime.now()
    print()
    print(f'Ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')

    envs.close()

    with open(save_path, 'wb') as f:
        np.save(f, test_rewards)
    print(f'Results saved to {save_path}')

    # Print results
    print("\nEvaluation Results:")
    print("=" * 60)
    for i in range(len(eval_goals)):
        goal = eval_goals[i]
        reward = test_rewards[i]
        print(f'Goal {args.goal_idx if args.goal_idx is not None else i} {goal}: {reward}')
    
    print()
    print("Mean reward per environment:", test_rewards.mean(axis=1))
    print("Overall mean reward: ", test_rewards.mean())
    print("Std deviation: ", test_rewards.std())