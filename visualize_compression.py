"""
Visualize compression segments and action probabilities.
Helps understand how on-policy segments are detected.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_compression(hdf5_path, env_idx=0, stream_idx=0, max_timesteps=200):
    """
    Visualize compression segments overlaid with action probabilities.
    
    Args:
        hdf5_path: Path to compressed HDF5 file
        env_idx: Environment index to visualize
        stream_idx: Stream index within environment
        max_timesteps: Maximum timesteps to show
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Load metadata
        threshold = f.attrs['on_policy_threshold']
        min_length = f.attrs['compression_window']
        
        # Load data for specific environment
        group = f[f'{env_idx}']
        action_probs = group['action_probs'][()]  # (n_stream, n_timesteps)
        compression_mask = group['compression_mask'][()]  # (n_stream, n_timesteps)
        rewards = group['rewards'][()]  # (n_stream, n_timesteps)
        
        # Select stream and truncate
        action_probs = action_probs[stream_idx, :max_timesteps]
        compression_mask = compression_mask[stream_idx, :max_timesteps]
        rewards = rewards[stream_idx, :max_timesteps]
        
        # Count segments
        segments = []
        in_segment = False
        segment_start = None
        
        for i, is_compressed in enumerate(compression_mask):
            if is_compressed and not in_segment:
                segment_start = i
                in_segment = True
            elif not is_compressed and in_segment:
                segments.append((segment_start, i))
                in_segment = False
        
        if in_segment:
            segments.append((segment_start, len(compression_mask)))
        
        print(f"\nCompression Analysis:")
        print(f"  Environment: {env_idx}, Stream: {stream_idx}")
        print(f"  Threshold: {threshold}")
        print(f"  Minimum segment length: {min_length}")
        print(f"  Timesteps shown: {max_timesteps}")
        print(f"  Number of segments: {len(segments)}")
        print(f"  Compressed steps: {np.sum(compression_mask)}/{len(compression_mask)} ({np.mean(compression_mask)*100:.1f}%)")
        
        if segments:
            print(f"\n  Segment details:")
            for i, (start, end) in enumerate(segments):
                length = end - start
                avg_prob = np.mean(action_probs[start:end])
                print(f"    Segment {i+1}: timesteps {start}-{end} (length={length}, avg_prob={avg_prob:.3f})")
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        
        timesteps = np.arange(len(action_probs))
        
        # Plot 1: Action probabilities with compression overlay
        ax1.plot(timesteps, action_probs, 'b-', linewidth=1.5, label='Action Probability')
        ax1.axhline(y=threshold, color='r', linestyle='--', linewidth=1, label=f'Threshold ({threshold})')
        ax1.fill_between(timesteps, 0, 1, where=compression_mask.astype(bool), 
                        alpha=0.3, color='green', label='Compressed Segment')
        
        # Mark segment boundaries with vertical lines
        for start, end in segments:
            ax1.axvline(x=start, color='darkgreen', linestyle=':', alpha=0.7)
            ax1.axvline(x=end, color='darkgreen', linestyle=':', alpha=0.7)
        
        ax1.set_ylabel('Probability', fontsize=11)
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'On-Policy Segment Detection (Env={env_idx}, Stream={stream_idx})', fontsize=13, fontweight='bold')
        
        # Plot 2: Compression mask (binary)
        ax2.fill_between(timesteps, 0, 1, where=compression_mask.astype(bool), 
                        color='green', alpha=0.6, step='post')
        ax2.set_ylabel('Compressed', fontsize=11)
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['No', 'Yes'])
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_title('Compression Mask', fontsize=12)
        
        # Plot 3: Rewards
        colors = ['red' if r < 0 else 'blue' for r in rewards]
        ax3.bar(timesteps, rewards, color=colors, alpha=0.6, width=1.0)
        ax3.set_xlabel('Timestep', fontsize=11)
        ax3.set_ylabel('Reward', fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Rewards', fontsize=12)
        
        plt.tight_layout()
        
        # Save figure
        save_path = f'figs/compression_visualization_env{env_idx}_stream{stream_idx}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to: {save_path}")
        
        plt.show()


def compare_environments(hdf5_path, n_envs=4, max_timesteps=100):
    """
    Compare compression across multiple environments.
    
    Args:
        hdf5_path: Path to compressed HDF5 file
        n_envs: Number of environments to compare
        max_timesteps: Timesteps to show per environment
    """
    with h5py.File(hdf5_path, 'r') as f:
        threshold = f.attrs['on_policy_threshold']
        
        fig, axes = plt.subplots(n_envs, 1, figsize=(14, n_envs*2), sharex=True)
        
        if n_envs == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            if str(i) not in f:
                break
            
            group = f[f'{i}']
            action_probs = group['action_probs'][()][0, :max_timesteps]
            compression_mask = group['compression_mask'][()][0, :max_timesteps]
            
            timesteps = np.arange(len(action_probs))
            
            ax.plot(timesteps, action_probs, 'b-', linewidth=1, alpha=0.7)
            ax.axhline(y=threshold, color='r', linestyle='--', linewidth=0.8)
            ax.fill_between(timesteps, 0, 1, where=compression_mask.astype(bool),
                           alpha=0.3, color='green')
            
            compression_ratio = np.mean(compression_mask)
            ax.set_ylabel(f'Env {i}', fontsize=10)
            ax.text(0.98, 0.95, f'{compression_ratio*100:.1f}% compressed', 
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Timestep', fontsize=11)
        fig.suptitle('Compression Comparison Across Environments', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = 'figs/compression_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved comparison to: {save_path}")
        
        plt.show()


def compression_statistics(hdf5_path):
    """Print detailed compression statistics."""
    with h5py.File(hdf5_path, 'r') as f:
        print("\n" + "="*60)
        print("COMPRESSION STATISTICS")
        print("="*60)
        
        # Global metadata
        print(f"\nGlobal Settings:")
        print(f"  On-policy threshold: {f.attrs['on_policy_threshold']}")
        print(f"  Minimum segment length: {f.attrs['compression_window']}")
        print(f"  Total steps: {f.attrs['total_steps']}")
        print(f"  Compressed steps: {f.attrs['compressed_steps']}")
        print(f"  Overall compression ratio: {f.attrs['compression_ratio']*100:.2f}%")
        
        # Per-environment statistics
        print(f"\nPer-Environment Statistics:")
        
        env_ratios = []
        for env_idx in sorted([int(k) for k in f.keys() if k.isdigit()]):
            group = f[f'{env_idx}']
            mask = group['compression_mask'][()]
            ratio = np.mean(mask)
            env_ratios.append(ratio)
            
            n_streams, n_timesteps = mask.shape
            compressed_steps = np.sum(mask)
            
            print(f"  Env {env_idx}: {ratio*100:.1f}% ({compressed_steps}/{n_streams*n_timesteps} steps)")
        
        print(f"\nDistribution:")
        print(f"  Mean: {np.mean(env_ratios)*100:.1f}%")
        print(f"  Std:  {np.std(env_ratios)*100:.1f}%")
        print(f"  Min:  {np.min(env_ratios)*100:.1f}%")
        print(f"  Max:  {np.max(env_ratios)*100:.1f}%")
        
        print("="*60 + "\n")


if __name__ == '__main__':
    import sys
    import os
    
    # Default path
    hdf5_path = 'datasets/history_darkroom_PPO_alg-seed0_compressed.hdf5'
    
    if len(sys.argv) > 1:
        hdf5_path = sys.argv[1]
    
    if not os.path.exists(hdf5_path):
        print(f"Error: File not found: {hdf5_path}")
        print("Usage: python visualize_compression.py [path_to_compressed.hdf5]")
        sys.exit(1)
    
    # Create figs directory if needed
    os.makedirs('figs', exist_ok=True)
    
    # Show statistics
    compression_statistics(hdf5_path)
    
    # Visualize single environment
    print("\nGenerating detailed visualization for Env 0, Stream 0...")
    visualize_compression(hdf5_path, env_idx=0, stream_idx=0, max_timesteps=200)
    
    # Compare environments
    print("\nGenerating comparison across environments...")
    compare_environments(hdf5_path, n_envs=min(4, 81), max_timesteps=150)
    
    print("\nVisualization complete!")
