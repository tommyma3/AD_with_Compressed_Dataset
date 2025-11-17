"""
Quick Start Guide for Compressed Algorithm Distillation

This script demonstrates the complete workflow with a small example.
"""
import os
import sys


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def main():
    print_section("COMPRESSED ALGORITHM DISTILLATION - QUICK START")
    
    print("""
This implementation adds compression tokens to Algorithm Distillation for
efficient learning from on-policy segments.

The workflow has 3 main steps:
    1. Collect compressed training data (with on-policy detection)
    2. Train the Compressed AD model
    3. Evaluate the trained model
    
Let's walk through each step...
    """)
    
    # Check if we're ready to proceed
    if not os.path.exists('config/env/darkroom.yaml'):
        print("❌ Error: Config files not found. Please run from the project root directory.")
        return
    
    print_section("STEP 1: Collect Compressed Training Data")
    
    print("""
This step trains PPO policies and identifies on-policy transitions:
    
    python collect_compressed.py
    
Configuration (you can adjust these):
    - on_policy_threshold: 0.7   (higher = more aggressive compression)
    - compression_window: 10      (minimum consecutive steps to compress)
    
Expected output:
    datasets/history_darkroom_PPO_alg-seed0_compressed.hdf5
    
This file contains:
    - States, actions, rewards (standard)
    - Compression masks (which transitions are on-policy)
    - Compression segments (start/end indices)
    - Metadata (thresholds, statistics)
    """)
    
    response = input("Run data collection now? (y/n, or 'skip' if already done): ")
    
    if response.lower() == 'y':
        print("\n🚀 Starting data collection...")
        print("This will take some time (10-30 minutes depending on configuration)\n")
        os.system('python collect_compressed.py')
    elif response.lower() == 'skip':
        compressed_file = 'datasets/history_darkroom_PPO_alg-seed0_compressed.hdf5'
        if not os.path.exists(compressed_file):
            print(f"❌ Error: {compressed_file} not found. Please run collection first.")
            return
        print("✓ Using existing compressed dataset")
    else:
        print("Skipped data collection. Run manually: python collect_compressed.py")
        return
    
    print_section("STEP 2: Train Compressed AD Model")
    
    print("""
Now we train the transformer with compression token support:
    
    python train_compressed.py
    
The model learns to:
    - Recognize <compress> and </compress> tokens
    - Use compressed segments as efficient context
    - Focus learning on off-policy, diverse transitions
    
Training will:
    - Save checkpoints to ./runs/CompressedAD-darkroom-seed0/
    - Log metrics to TensorBoard
    - Evaluate periodically on test environments
    
Expected training time: 5-15 minutes
    """)
    
    response = input("Start training now? (y/n): ")
    
    if response.lower() == 'y':
        print("\n🚀 Starting training...")
        print("Monitor progress with: tensorboard --logdir=runs\n")
        os.system('python train_compressed.py')
    else:
        print("Skipped training. Run manually: python train_compressed.py")
        return
    
    print_section("STEP 3: Analyze Results")
    
    print("""
After training, you can:

1. Check compression statistics:
    
    python -c "
    import h5py
    with h5py.File('datasets/history_darkroom_PPO_alg-seed0_compressed.hdf5', 'r') as f:
        print(f'Compression ratio: {f.attrs[\"overall_compression_ratio\"]*100:.1f}%')
    "

2. View training curves:
    
    tensorboard --logdir=runs

3. Evaluate the model:
    
    python evaluate.py  # (update to use CompressedAD)

4. Compare with baseline:
    - Train standard AD: python train.py
    - Compare performance and training time
    """)
    
    print_section("NEXT STEPS")
    
    print("""
✨ Congratulations! You've set up Compressed Algorithm Distillation.

To customize for your use case:

1. Adjust compression threshold:
   - Edit collect_compressed.py: config['on_policy_threshold']
   - Higher (0.9): More selective, compress only very confident steps
   - Lower (0.5): More aggressive, compress more steps

2. Modify compression window:
   - Edit collect_compressed.py: config['compression_window']
   - Larger: Fewer, longer compressed segments
   - Smaller: More frequent compression

3. Experiment with model architecture:
   - Edit config/model/ad_dr.yaml
   - Try different n_head, n_layer, tf_n_embd values

4. Extend to your environment:
   - Create new environment in env/
   - Update SAMPLE_ENVIRONMENT mapping
   - Collect compressed data for your task

📚 See COMPRESSED_AD_README.md for detailed documentation.

🐛 Troubleshooting:
   - Check logs in ./runs/
   - Verify compressed dataset with dataset_compressed.py
   - Inspect attention patterns with analyze_attention.py

Happy compressing! 🚀
    """)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
