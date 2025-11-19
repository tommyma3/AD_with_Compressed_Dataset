"""
Visualization script for compression markers in collected trajectories.

This script helps verify that compression markers are correctly inserted
and visualizes the compression regions in the training data.

Usage:
    python visualize_compression.py [--env_idx 0] [--max_steps 500]
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from utils import get_config, get_traj_file_name


def visualize_compression_markers(traj_file, env_idx=0, max_steps=500):
    """Visualize compression markers for a specific environment trajectory."""
    
    with h5py.File(traj_file, 'r') as f:
        if str(env_idx) not in f:
            print(f"Error: Environment {env_idx} not found in {traj_file}")
            print(f"Available environments: {list(f.keys())}")
            return
        
        env_group = f[str(env_idx)]
        
        # Check if compress_markers exist
        if 'compress_markers' not in env_group:
            print(f"Error: No compression markers found in {traj_file}")
            print("This file was collected without compressed context mode.")
            print("Please run collect.py with the updated HistoryLoggerCallback.")
            return
        
        # Load data
        states = env_group['states'][()]
        actions = env_group['actions'][()]
        rewards = env_group['rewards'][()]
        compress_markers = env_group['compress_markers'][()]
        
        # Transpose if needed (depends on storage format)
        if states.shape[0] < states.shape[1]:
            states = states.T
            actions = actions.T
            rewards = rewards.T
            compress_markers = compress_markers.T
        
        # Take first trajectory stream
        states = states[0, :max_steps]
        actions = actions[0, :max_steps]
        rewards = rewards[0, :max_steps]
        markers = compress_markers[0, :max_steps]
        
        print(f"\nTrajectory Statistics:")
        print(f"  Total steps: {len(markers)}")
        print(f"  Compress start tokens: {np.sum(markers == 1)}")
        print(f"  Compress end tokens: {np.sum(markers == 2)}")
        print(f"  Normal transitions: {np.sum(markers == 0)}")
        
        # Find compression regions
        compress_starts = np.where(markers == 1)[0]
        compress_ends = np.where(markers == 2)[0]
        
        print(f"\n  Compression regions found: {len(compress_starts)}")
        
        if len(compress_starts) > 0:
            print(f"  First region starts at: {compress_starts[0]}")
            if len(compress_ends) > 0:
                print(f"  First region ends at: {compress_ends[0]}")
                if len(compress_starts) > 0 and len(compress_ends) > 0:
                    region_length = compress_ends[0] - compress_starts[0]
                    print(f"  First region length: {region_length}")
        
        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        
        timesteps = np.arange(len(markers))
        
        # Plot 1: Rewards with compression regions highlighted
        ax = axes[0]
        ax.plot(timesteps, rewards, 'b-', linewidth=1, label='Rewards')
        
        # Highlight compression regions
        for start, end in zip(compress_starts, compress_ends):
            if end < len(rewards):
                ax.axvspan(start, end, alpha=0.2, color='green', label='Compression Region' if start == compress_starts[0] else '')
        
        ax.set_ylabel('Reward')
        ax.set_title('Rewards with Compression Regions Highlighted')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Actions
        ax = axes[1]
        ax.plot(timesteps, actions, 'r.', markersize=2, label='Actions')
        
        for start, end in zip(compress_starts, compress_ends):
            if end < len(actions):
                ax.axvspan(start, end, alpha=0.2, color='green')
        
        ax.set_ylabel('Action')
        ax.set_title('Actions')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Compression markers
        ax = axes[2]
        
        # Create bar chart for markers
        colors = ['blue' if m == 0 else 'green' if m == 1 else 'red' for m in markers]
        ax.bar(timesteps, markers, color=colors, width=1.0, alpha=0.7)
        
        ax.set_ylabel('Marker Type')
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Normal', '<compress>', '</compress>'])
        ax.set_title('Compression Markers')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Plot 4: Cumulative reward in each compression region
        ax = axes[3]
        
        region_rewards = []
        region_positions = []
        
        for i, (start, end) in enumerate(zip(compress_starts, compress_ends)):
            if end < len(rewards):
                region_reward = np.sum(rewards[start:end])
                region_rewards.append(region_reward)
                region_positions.append((start + end) / 2)
        
        if len(region_rewards) > 0:
            ax.bar(range(len(region_rewards)), region_rewards, alpha=0.7, color='purple')
            ax.set_xlabel('Compression Region Index')
            ax.set_ylabel('Cumulative Reward')
            ax.set_title('Cumulative Reward per Compression Region')
            ax.grid(True, alpha=0.3)
        
        axes[3].set_xlabel('Timestep')
        
        plt.tight_layout()
        
        # Save figure
        output_file = f'figs/compression_visualization_env{env_idx}.png'
        import os
        os.makedirs('figs', exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_file}")
        
        plt.show()


def analyze_compression_patterns(traj_file):
    """Analyze compression patterns across all environments."""
    
    print("\n" + "="*60)
    print("Compression Pattern Analysis")
    print("="*60)
    
    with h5py.File(traj_file, 'r') as f:
        env_ids = list(f.keys())
        
        if 'compress_markers' not in f[env_ids[0]]:
            print("Error: No compression markers found in trajectories")
            return
        
        all_region_lengths = []
        
        for env_id in env_ids:
            markers = f[env_id]['compress_markers'][()]
            
            # Handle transposed data
            if markers.shape[0] < markers.shape[1]:
                markers = markers.T
            
            # Take first stream
            markers = markers[0]
            
            starts = np.where(markers == 1)[0]
            ends = np.where(markers == 2)[0]
            
            for start, end in zip(starts, ends):
                region_length = end - start
                all_region_lengths.append(region_length)
        
        if len(all_region_lengths) > 0:
            all_region_lengths = np.array(all_region_lengths)
            
            print(f"\nCompression Region Statistics:")
            print(f"  Total regions: {len(all_region_lengths)}")
            print(f"  Mean length: {all_region_lengths.mean():.2f} ± {all_region_lengths.std():.2f}")
            print(f"  Min length: {all_region_lengths.min()}")
            print(f"  Max length: {all_region_lengths.max()}")
            print(f"  Median length: {np.median(all_region_lengths):.2f}")
            
            # Plot histogram
            plt.figure(figsize=(10, 6))
            plt.hist(all_region_lengths, bins=30, alpha=0.7, color='blue', edgecolor='black')
            plt.xlabel('Compression Region Length')
            plt.ylabel('Frequency')
            plt.title('Distribution of Compression Region Lengths')
            plt.grid(True, alpha=0.3)
            plt.axvline(all_region_lengths.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {all_region_lengths.mean():.1f}')
            plt.legend()
            
            output_file = 'figs/compression_region_distribution.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\nDistribution plot saved to: {output_file}")
            
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize compression markers in collected trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--env_idx',
        type=int,
        default=0,
        help='Environment index to visualize (default: 0)'
    )
    
    parser.add_argument(
        '--max_steps',
        type=int,
        default=500,
        help='Maximum number of steps to visualize (default: 500)'
    )
    
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Run analysis of compression patterns across all environments'
    )
    
    args = parser.parse_args()
    
    # Load config to get trajectory file name
    config = get_config("config/env/darkroom.yaml")
    config.update(get_config("config/algorithm/ppo_darkroom.yaml"))
    
    traj_file = f'datasets/{get_traj_file_name(config)}.hdf5'
    
    print(f"Loading trajectories from: {traj_file}")
    
    import os
    if not os.path.exists(traj_file):
        print(f"\nError: Trajectory file not found: {traj_file}")
        print("Please run collect.py first to generate training data.")
        return
    
    if args.analyze:
        analyze_compression_patterns(traj_file)
    else:
        visualize_compression_markers(traj_file, args.env_idx, args.max_steps)


if __name__ == '__main__':
    main()
