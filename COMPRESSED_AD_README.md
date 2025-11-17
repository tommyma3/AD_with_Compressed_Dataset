# Compressed Algorithm Distillation

This implementation extends Algorithm Distillation with compression tokens to enable efficient learning from on-policy segments.

## 🎯 Overview

**Problem**: Standard AD uses all historical transitions equally, even though many transitions are redundant when they're highly on-policy (the current policy would make the same decisions).

**Solution**: Mark on-policy segments with special `<compress>` and `</compress>` tokens, allowing the transformer to:
1. Use compressed segments as compact context
2. Focus learning on diverse, off-policy transitions
3. Achieve better sample efficiency

## 🏗️ Architecture

### Key Components

1. **CompressedHistoryLoggerCallback** (`algorithm/compressed_history_callback.py`)
   - Tracks action probabilities during PPO training
   - Identifies on-policy transitions (high probability under current policy)
   - Marks consecutive on-policy segments for compression

2. **CompressedADDataset** (`dataset_compressed.py`)
   - Loads trajectories with compression masks
   - Provides training samples with compression information
   - Handles variable-length compressed segments

3. **CompressedAD Model** (`model/compressed_ad.py`)
   - Adds learned embeddings for compression tokens
   - Processes compressed context efficiently
   - Only predicts actions for non-compressed query positions

### Compression Strategy

**On-Policy Detection**:
- Transition is "on-policy" if: `P(action|state) >= threshold` (default: 0.7)
- Consecutive on-policy steps form a compression segment
- Minimum segment length to compress: 10 steps (configurable)

**Token Insertion**:
```
Normal context:      s₀ a₀ r₀ s₁ a₁ r₁ s₂ a₂ r₂ ...
With compression:    s₀ a₀ r₀ <compress> s₁ a₁ r₁ ... sₖ </compress> sₖ₊₁ ...
```

The transformer learns:
- Content inside `<compress>...</compress>` is high-confidence, on-policy behavior
- Use it as context but don't waste capacity memorizing obvious actions
- Focus learning on challenging, off-policy transitions

## 📊 Workflow

### Step 1: Collect Compressed Data

```bash
python collect_compressed.py
```

This script:
- Trains PPO policies for each environment
- Identifies on-policy transitions during training
- Saves trajectories with compression masks to HDF5

**Configuration** (in script or config file):
```python
config['on_policy_threshold'] = 0.7  # Probability threshold
config['compression_window'] = 10     # Min segment length
```

**Output**: `datasets/history_darkroom_PPO_alg-seed0_compressed.hdf5`

### Step 2: Train Compressed AD Model

```bash
python train_compressed.py
```

This script:
- Loads compressed dataset
- Trains CompressedAD transformer
- Evaluates on test environments

**Model learns to**:
- Recognize compression tokens
- Use compressed context efficiently
- Predict actions for non-compressed queries

### Step 3: Evaluate

```bash
python evaluate.py  # Update to use CompressedAD model
```

The trained model can generalize to new tasks using the learned compression strategy.

## 🔧 Configuration

### Compression Parameters

```yaml
# In config/algorithm/ppo_darkroom.yaml or hardcoded
on_policy_threshold: 0.7    # Higher = more aggressive compression
                             # 0.9 = only compress very confident steps
                             # 0.5 = compress more liberally

compression_window: 10       # Minimum consecutive on-policy steps
                             # Larger = fewer, longer compressed segments
                             # Smaller = more, shorter segments
```

### Model Parameters

```yaml
# In config/model/ad_dr.yaml
tf_n_embd: 32               # Embedding dimension
tf_n_layer: 4               # Number of transformer layers
tf_n_head: 4                # Number of attention heads
n_transit: 80               # Context window size
```

## 📁 File Structure

```
AD_Compress_Window/
├── algorithm/
│   ├── compressed_history_callback.py  # NEW: Compression callback
│   └── __init__.py                     # Updated
├── model/
│   ├── compressed_ad.py                # NEW: Compressed AD model
│   └── __init__.py                     # Updated
├── collect_compressed.py               # NEW: Data collection script
├── dataset_compressed.py               # NEW: Dataset loader
├── train_compressed.py                 # NEW: Training script
└── COMPRESSED_AD_README.md            # This file
```

## 🧪 Example Usage

### Full Pipeline

```bash
# 1. Collect compressed training data
python collect_compressed.py

# 2. Train compressed AD model
python train_compressed.py

# 3. (Optional) Analyze compression statistics
python -c "
import h5py
import numpy as np

with h5py.File('datasets/history_darkroom_PPO_alg-seed0_compressed.hdf5', 'r') as f:
    print(f'Compression Statistics:')
    print(f'  Overall ratio: {f.attrs[\"overall_compression_ratio\"]*100:.1f}%')
    print(f'  Total steps: {f.attrs[\"total_steps\"]:,}')
    print(f'  Compressed steps: {f.attrs[\"total_compressed_steps\"]:,}')
    
    for i in range(min(5, len(f.keys()))):
        env_group = f[str(i)]
        print(f'  Env {i}: {env_group.attrs[\"compression_ratio\"]*100:.1f}% compressed')
"
```

### Custom Compression Strategy

Modify `CompressedHistoryLoggerCallback._identify_compression_segments()`:

```python
def _identify_compression_segments(self, action_probs_array):
    # Example: Only compress perfect on-policy (prob = 1.0)
    is_on_policy = action_probs_array >= 0.95
    
    # Example: Compress all segments regardless of length
    compression_mask = is_on_policy
    
    # Your custom logic here...
    
    return compression_mask, segment_boundaries
```

## 📈 Expected Results

**Compression Efficiency**:
- Typical compression ratio: 30-60% of steps marked for compression
- Depends on:
  - Policy quality (better policy = more on-policy = more compression)
  - Task difficulty (easy tasks = higher compression)
  - Threshold setting

**Training Benefits**:
- Faster convergence (fewer transitions to memorize)
- Better generalization (focus on diverse transitions)
- Memory efficiency (compressed context uses fewer parameters)

**Performance**:
- Should match or exceed standard AD
- Especially effective when:
  - Training data has many on-policy segments
  - Compression threshold is well-tuned
  - Context window is long (>50 steps)

## 🔍 Debugging Tips

### Check Compression Masks

```python
from dataset_compressed import CompressedADDataset

dataset = CompressedADDataset(config, './datasets', 'train', 1, 1000)

sample = dataset[0]
print(f"Compression mask: {sample['compression_mask']}")
print(f"Has compressed context: {sample['has_compressed_context']}")
print(f"Ratio: {sample['compression_mask'].mean():.2%}")
```

### Verify Token Embeddings

```python
model = CompressedAD(config)
print(f"Compress start embedding shape: {model.compress_start_embed.shape}")
print(f"Compress end embedding shape: {model.compress_end_embed.shape}")

# Check that embeddings are learned
print(f"Start embed mean: {model.compress_start_embed.mean().item():.4f}")
```

### Monitor Training Logs

Look for:
- `compression/num_segments`: Number of compressed segments per trajectory
- `compression/ratio`: Fraction of steps compressed
- `compression/avg_segment_length`: Average compressed segment length

## ⚠️ Limitations

1. **Data Collection Overhead**: Requires tracking action probabilities during PPO training
2. **Compression Quality**: Depends on threshold tuning for your task
3. **Model Complexity**: Additional compression token embeddings add parameters
4. **Evaluation**: Compression is only used during training, not inference

## 🚀 Future Improvements

- **Dynamic Compression**: Learn threshold instead of using fixed value
- **Hierarchical Compression**: Multi-level compression for very long contexts
- **Compression Attention**: Separate attention heads for compressed vs normal context
- **Online Compression**: Apply compression during inference for speedup

## 📚 References

This implementation is inspired by:
- Algorithm Distillation (Laskin et al., 2022)
- Memory-Augmented Transformers
- Efficient Transformers Survey

## 🤝 Contributing

To extend this implementation:
1. Modify compression strategy in `CompressedHistoryLoggerCallback`
2. Update token embeddings in `CompressedAD`
3. Adjust dataset loader in `CompressedADDataset`
4. Test on your environment and report results!

---

**Questions?** Check the code comments or open an issue.
