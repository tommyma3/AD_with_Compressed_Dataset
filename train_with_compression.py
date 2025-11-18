"""
Training script for Compressed Algorithm Distillation.
Uses dataset with compression tokens marking on-policy segments.
"""
import os
import yaml
from accelerate import Accelerator
from datetime import datetime
import torch
from torch.utils.data import DataLoader

from dataset_with_compression import CompressedADDataset
from model import MODEL
from utils import get_config


def train_compressed_ad():
    """Train the CompressedAD model."""
    
    # Load configuration
    config = get_config("config/env/darkroom.yaml")
    config.update(get_config("config/algorithm/ppo_darkroom.yaml"))
    config.update(get_config("config/model/ad_dr.yaml"))
    
    # Set model type
    config['model'] = 'CompressedADv2'
    
    # Setup accelerator
    accelerator = Accelerator(mixed_precision=config.get('mixed_precision', 'no'))
    config['device'] = accelerator.device
    
    # Create datasets
    print("Loading compressed dataset...")
    train_dataset = CompressedADDataset(
        config, 
        'datasets', 
        mode='train',
        n_stream=config.get('n_stream'),
        source_timesteps=config.get('source_timesteps')
    )
    
    test_dataset = CompressedADDataset(
        config,
        'datasets',
        mode='test',
        n_stream=config.get('n_stream'),
        source_timesteps=config.get('source_timesteps')
    )
    
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    print(f"Creating {config['model']} model...")
    model = MODEL[config['model']](config)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Prepare with accelerator
    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )
    
    # Training loop
    best_acc = 0.0
    run_name = f"CompressedAD-darkroom-seed{config.get('alg_seed', 0)}"
    save_dir = f"./runs/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nStarting training...")
    print(f"Save directory: {save_dir}")
    
    for epoch in range(config.get('n_epochs', 100)):
        # Training
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            outputs = model(batch)
            loss = outputs['loss_action']
            acc = outputs['acc_action']
            
            accelerator.backward(loss)
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += acc.item()
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(batch)
                test_loss += outputs['loss_action'].item()
                test_acc += outputs['acc_action'].item()
        
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        
        # Logging
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{config.get('n_epochs', 100)}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_acc': test_acc,
                    'config': config,
                }, f"{save_dir}/best_model.pt")
                print(f"  → Saved best model (acc: {best_acc:.4f})")
    
    print(f"\nTraining complete! Best test accuracy: {best_acc:.4f}")
    print(f"Model saved to: {save_dir}/best_model.pt")


if __name__ == '__main__':
    train_compressed_ad()
