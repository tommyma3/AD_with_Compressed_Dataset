"""
Example script demonstrating compressed context training workflow.

This script shows how to:
1. Collect PPO data with compression markers
2. Train a transformer with compressed context
3. Evaluate the trained model

Usage:
    python example_compressed_training.py [--mode collect|train|eval|all]
"""

import argparse
import os
import subprocess
import sys


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\nError: {description} failed with return code {result.returncode}")
        sys.exit(1)
    
    print(f"\n{description} completed successfully!")


def collect_data():
    """Collect training data with compression markers."""
    print("\n" + "="*60)
    print("STEP 1: Collecting PPO Training Data with Compression Markers")
    print("="*60)
    print("""
This will:
- Run PPO on multiple darkroom environments
- Insert <compress> and </compress> tokens every n_steps
- Save trajectories with compression markers to HDF5 files
- Location: datasets/history_darkroom_PPO_alg-seed42.hdf5
    """)
    
    run_command("python collect.py", "Data Collection")


def train_model():
    """Train the transformer with compressed context."""
    print("\n" + "="*60)
    print("STEP 2: Training Transformer with Compressed Context")
    print("="*60)
    print("""
This will:
- Load trajectories with compression markers
- Sample sequences from current + previous compression regions only
- Train transformer to predict actions and token types
- Save checkpoints to runs/AD-darkroom-seed0/
    """)
    
    # Check if we need to modify train.py to use compressed config
    print("\nNote: Make sure train.py uses the compressed config:")
    print("  config.update(get_config('./config/model/ad_dr_compressed.yaml'))")
    
    response = input("\nHave you updated train.py? (y/n): ")
    if response.lower() != 'y':
        print("\nPlease update train.py first:")
        print("  Change: config.update(get_config('./config/model/ad_dr.yaml'))")
        print("  To:     config.update(get_config('./config/model/ad_dr_compressed.yaml'))")
        sys.exit(0)
    
    run_command("python train.py", "Model Training")


def evaluate_model():
    """Evaluate the trained model."""
    print("\n" + "="*60)
    print("STEP 3: Evaluating Trained Model")
    print("="*60)
    print("""
This will:
- Load the trained model checkpoint
- Evaluate on test environments
- Use compressed context during rollout
- Report mean rewards
    """)
    
    # Check if checkpoint exists
    ckpt_dir = "runs/AD-darkroom-seed0"
    if not os.path.exists(ckpt_dir):
        print(f"\nError: Checkpoint directory not found: {ckpt_dir}")
        print("Please train the model first (step 2)")
        sys.exit(1)
    
    run_command("python evaluate.py", "Model Evaluation")


def main():
    parser = argparse.ArgumentParser(
        description="Run compressed context training workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full workflow
  python example_compressed_training.py --mode all
  
  # Run individual steps
  python example_compressed_training.py --mode collect
  python example_compressed_training.py --mode train
  python example_compressed_training.py --mode eval
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['collect', 'train', 'eval', 'all'],
        default='all',
        help='Which step to run (default: all)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Compressed Context Algorithm Distillation - Example Workflow")
    print("="*60)
    
    if args.mode in ['collect', 'all']:
        collect_data()
    
    if args.mode in ['train', 'all']:
        train_model()
    
    if args.mode in ['eval', 'all']:
        evaluate_model()
    
    print("\n" + "="*60)
    print("WORKFLOW COMPLETE!")
    print("="*60)
    print("""
Results:
- Trained model: runs/AD-darkroom-seed0/ckpt-*.pt
- Training logs: runs/AD-darkroom-seed0/ (view with tensorboard)
- Evaluation results: runs/AD-darkroom-seed0/eval_result.npy

Next Steps:
1. View training curves: tensorboard --logdir runs/
2. Analyze compression metrics: Check loss_token_type and acc_token_type
3. Compare with baseline: Train without compressed context and compare rewards
    """)


if __name__ == '__main__':
    main()
