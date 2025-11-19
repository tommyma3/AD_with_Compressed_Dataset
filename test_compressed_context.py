"""
Test script to validate compressed context implementation.

This script performs unit tests on key components of the compressed context feature
to ensure everything works correctly before running full training.

Usage:
    python test_compressed_context.py
"""

import torch
import numpy as np
import h5py
import os
import tempfile
from dataset import ADDataset
from model import MODEL
from utils import get_config


def test_compression_markers_format():
    """Test that compression markers are in the correct format."""
    print("\n" + "="*60)
    print("TEST 1: Compression Markers Format")
    print("="*60)
    
    # Create dummy markers
    n_steps = 240  # 3 compression regions of 80 steps each
    markers = np.zeros(n_steps, dtype=np.int32)
    
    # Insert compression markers every 80 steps
    compress_interval = 80
    for i in range(0, n_steps, compress_interval):
        if i < n_steps:
            markers[i] = 1  # compress_start
        if i + compress_interval - 1 < n_steps:
            markers[i + compress_interval - 1] = 2  # compress_end
    
    # Verify markers
    starts = np.where(markers == 1)[0]
    ends = np.where(markers == 2)[0]
    
    print(f"Total steps: {n_steps}")
    print(f"Compression starts: {starts}")
    print(f"Compression ends: {ends}")
    print(f"Number of regions: {len(starts)}")
    
    assert len(starts) == len(ends), "Mismatch between start and end markers"
    assert len(starts) == 3, f"Expected 3 regions, got {len(starts)}"
    
    for start, end in zip(starts, ends):
        region_len = end - start + 1
        print(f"Region [{start}, {end}]: length = {region_len}")
        assert region_len == compress_interval, f"Region length {region_len} != {compress_interval}"
    
    print("✅ PASSED: Compression markers are correctly formatted")


def test_model_initialization():
    """Test that the model initializes correctly with compressed context."""
    print("\n" + "="*60)
    print("TEST 2: Model Initialization")
    print("="*60)
    
    config = {
        'device': 'cpu',
        'n_transit': 160,
        'mixed_precision': 'fp32',
        'grid_size': 9,
        'use_compressed_context': True,
        'compress_interval': 80,
        'tf_n_embd': 64,
        'tf_n_head': 4,
        'tf_n_layer': 2,
        'dim_states': 2,
        'num_actions': 5,
        'label_smoothing': 0.0
    }
    
    model = MODEL['AD'](config)
    
    # Check special tokens exist
    assert hasattr(model, 'compress_start_token'), "Missing compress_start_token"
    assert hasattr(model, 'compress_end_token'), "Missing compress_end_token"
    assert hasattr(model, 'token_type_embedding'), "Missing token_type_embedding"
    assert hasattr(model, 'pred_token_type'), "Missing pred_token_type"
    
    print(f"Compress start token shape: {model.compress_start_token.shape}")
    print(f"Compress end token shape: {model.compress_end_token.shape}")
    print(f"Token type embedding: {model.token_type_embedding}")
    print(f"Token type predictor: {model.pred_token_type}")
    
    print("✅ PASSED: Model initialized correctly with compressed context")


def test_forward_pass():
    """Test forward pass with compressed context data."""
    print("\n" + "="*60)
    print("TEST 3: Forward Pass with Compressed Context")
    print("="*60)
    
    config = {
        'device': 'cpu',
        'n_transit': 160,
        'mixed_precision': 'fp32',
        'grid_size': 9,
        'use_compressed_context': True,
        'compress_interval': 80,
        'tf_n_embd': 64,
        'tf_n_head': 4,
        'tf_n_layer': 2,
        'dim_states': 2,
        'num_actions': 5,
        'label_smoothing': 0.0
    }
    
    model = MODEL['AD'](config)
    model.eval()
    
    batch_size = 4
    n_context = 159  # n_transit - 1
    
    # Create dummy batch with valid discrete state values for darkroom
    # States should be integer coordinates [0, grid_size) for darkroom environment
    batch = {
        'query_states': torch.randint(0, config['grid_size'], (batch_size, 2)).float(),
        'target_actions': torch.randint(0, 5, (batch_size,)),
        'states': torch.randint(0, config['grid_size'], (batch_size, n_context, 2)).float(),
        'actions': torch.eye(5)[torch.randint(0, 5, (batch_size, n_context))],
        'rewards': torch.randint(0, 2, (batch_size, n_context)).float(),
        'next_states': torch.randint(0, config['grid_size'], (batch_size, n_context, 2)).float(),
        'compress_markers': torch.zeros(batch_size, n_context, dtype=torch.long),
        'target_token_types': torch.zeros(batch_size, dtype=torch.long),
    }
    
    # Insert some compression markers
    batch['compress_markers'][:, 0] = 1  # compress_start
    batch['compress_markers'][:, 79] = 2  # compress_end
    batch['compress_markers'][:, 80] = 1  # compress_start
    
    with torch.no_grad():
        output = model(batch)
    
    # Check outputs
    assert 'loss_action' in output, "Missing loss_action"
    assert 'acc_action' in output, "Missing acc_action"
    assert 'loss_token_type' in output, "Missing loss_token_type"
    assert 'acc_token_type' in output, "Missing acc_token_type"
    
    print(f"Loss action: {output['loss_action'].item():.4f}")
    print(f"Accuracy action: {output['acc_action'].item():.4f}")
    print(f"Loss token type: {output['loss_token_type'].item():.4f}")
    print(f"Accuracy token type: {output['acc_token_type'].item():.4f}")
    
    print("✅ PASSED: Forward pass works with compressed context")


def test_dataset_loading():
    """Test dataset loading and sampling with compression markers."""
    print("\n" + "="*60)
    print("TEST 4: Dataset Loading with Compression Markers")
    print("="*60)
    
    # Create temporary HDF5 file with dummy data
    with tempfile.TemporaryDirectory() as tmpdir:
        traj_file = os.path.join(tmpdir, 'test_traj.hdf5')
        
        # Create dummy trajectory data
        n_envs = 4  # 2x2 grid for testing
        n_streams = 2
        n_steps = 240
        
        with h5py.File(traj_file, 'w') as f:
            for env_idx in range(n_envs):
                env_group = f.create_group(str(env_idx))
                
                # Create dummy data (transposed format: time x stream x dim)
                states = np.random.randint(0, 4, (n_steps, n_streams, 2), dtype=np.int32)  # 2x2 grid
                actions = np.random.randint(0, 5, (n_steps, n_streams), dtype=np.int32)
                rewards = np.random.randint(0, 2, (n_steps, n_streams), dtype=np.int32)
                next_states = np.random.randint(0, 4, (n_steps, n_streams, 2), dtype=np.int32)  # 2x2 grid
                
                # Create compression markers
                markers = np.zeros((n_steps, n_streams), dtype=np.int32)
                compress_interval = 80
                for i in range(0, n_steps, compress_interval):
                    if i < n_steps:
                        markers[i, :] = 1  # compress_start
                    if i + compress_interval - 1 < n_steps:
                        markers[i + compress_interval - 1, :] = 2  # compress_end
                
                env_group.create_dataset('states', data=states)
                env_group.create_dataset('actions', data=actions)
                env_group.create_dataset('rewards', data=rewards)
                env_group.create_dataset('next_states', data=next_states)
                env_group.create_dataset('compress_markers', data=markers)
        
        # Create config
        config = {
            'env': 'darkroom',
            'grid_size': 2,  # 2x2 grid = 4 total environments (matches n_envs)
            'n_transit': 160,
            'dynamics': False,
            'use_compressed_context': True,
            'compress_interval': 80,
            'env_split_seed': 0,
            'train_env_ratio': 0.5,  # 2 train, 2 test
            'alg': 'PPO',
            'alg_seed': 42
        }
        
        # Override file name for testing
        from utils import get_traj_file_name
        original_fn = get_traj_file_name(config)
        
        # Copy file to expected location
        import shutil
        expected_file = os.path.join(tmpdir, f'{original_fn}.hdf5')
        shutil.copy(traj_file, expected_file)
        
        # Load dataset
        try:
            dataset = ADDataset(config, tmpdir, mode='train', n_stream=n_streams, source_timesteps=n_steps)
            
            print(f"Dataset length: {len(dataset)}")
            print(f"Compression regions built: {len(dataset.compression_regions)}")
            
            # Sample a few items
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                print(f"\nSample {i}:")
                print(f"  Query state shape: {sample['query_states'].shape}")
                print(f"  Context states shape: {sample['states'].shape}")
                print(f"  Compress markers shape: {sample['compress_markers'].shape}")
                print(f"  Compress markers: {sample['compress_markers']}")
                
                assert 'compress_markers' in sample, "Missing compress_markers in sample"
                assert sample['states'].shape[0] == config['n_transit'] - 1, "Incorrect context length"
            
            print("\n✅ PASSED: Dataset loading works with compressed context")
            
        except Exception as e:
            print(f"\n❌ FAILED: Dataset loading error: {e}")
            raise


def test_backward_compatibility():
    """Test that code works without compressed context (backward compatibility)."""
    print("\n" + "="*60)
    print("TEST 5: Backward Compatibility (No Compressed Context)")
    print("="*60)
    
    config = {
        'device': 'cpu',
        'n_transit': 80,
        'mixed_precision': 'fp32',
        'grid_size': 9,
        'use_compressed_context': False,  # Disabled
        'tf_n_embd': 64,
        'tf_n_head': 4,
        'tf_n_layer': 2,
        'dim_states': 2,
        'num_actions': 5,
        'label_smoothing': 0.0
    }
    
    model = MODEL['AD'](config)
    
    batch_size = 4
    n_context = 79
    
    # Create batch without compression markers
    # States should be valid discrete coordinates for darkroom
    batch = {
        'query_states': torch.randint(0, config['grid_size'], (batch_size, 2)).float(),
        'target_actions': torch.randint(0, 5, (batch_size,)),
        'states': torch.randint(0, config['grid_size'], (batch_size, n_context, 2)).float(),
        'actions': torch.eye(5)[torch.randint(0, 5, (batch_size, n_context))],
        'rewards': torch.randint(0, 2, (batch_size, n_context)).float(),
        'next_states': torch.randint(0, config['grid_size'], (batch_size, n_context, 2)).float(),
    }
    
    with torch.no_grad():
        output = model(batch)
    
    assert 'loss_action' in output, "Missing loss_action"
    assert 'acc_action' in output, "Missing acc_action"
    assert 'loss_token_type' not in output, "Should not have loss_token_type when disabled"
    
    print(f"Loss action: {output['loss_action'].item():.4f}")
    print(f"Accuracy action: {output['acc_action'].item():.4f}")
    
    print("✅ PASSED: Backward compatibility maintained")


def main():
    print("="*60)
    print("Compressed Context Implementation Tests")
    print("="*60)
    
    tests = [
        ("Compression Markers Format", test_compression_markers_format),
        ("Model Initialization", test_model_initialization),
        ("Forward Pass", test_forward_pass),
        ("Dataset Loading", test_dataset_loading),
        ("Backward Compatibility", test_backward_compatibility),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ FAILED: {test_name}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✅ All tests passed! Implementation is ready to use.")
    else:
        print(f"\n❌ {failed} test(s) failed. Please fix the issues before proceeding.")
        exit(1)


if __name__ == '__main__':
    main()
