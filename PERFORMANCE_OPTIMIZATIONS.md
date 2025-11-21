# Performance Optimizations for Multi-GPU Training

## Problem
Training speed was only **7 seconds/iteration** with 4 GPUs, which is extremely slow.

## Root Causes Identified

1. **Insufficient data loading workers** - 12 workers for 4 GPUs
2. **Inefficient collate function** - Using Python lists and repeated numpy operations
3. **CPU-GPU transfer bottleneck** - No pin_memory or prefetching
4. **Model forward pass inefficiency** - Creating tensors in tight loops (80+ tensor creations per sample)
5. **Small batch size** - 512 total (128 per GPU) underutilized GPU capacity

## Optimizations Applied

### 1. Data Loading Improvements

**Before:**
```yaml
num_workers: 12
```

**After:**
```yaml
num_workers: 16  # 4 workers per GPU
```

**Impact:** Better parallelism for data preprocessing

---

### 2. Optimized Collate Function

**Before:**
```python
# Multiple list appends and concatenations
states_list = []
for item in batch:
    if pad_len > 0:
        states_padded = np.concatenate([item['states'], np.zeros(...)])
    states_list.append(states_padded)
res['states'] = torch.tensor(np.stack(states_list), ...)
```

**After:**
```python
# Pre-allocate arrays, single operation
batch_size = len(batch)
states_array = np.zeros((batch_size, max_len) + state_shape, dtype=...)
for i, item in enumerate(batch):
    seq_len = len(item['states'])
    states_array[i, :seq_len] = item['states']
res['states'] = torch.from_numpy(states_array).float()
```

**Impact:** 
- Reduced memory allocations
- Faster numpy operations
- Less CPU overhead

---

### 3. DataLoader Configuration

**Before:**
```python
DataLoader(..., num_workers=num_workers, persistent_workers=True)
```

**After:**
```python
DataLoader(..., 
          num_workers=num_workers,
          persistent_workers=(num_workers > 0),
          pin_memory=True,        # Faster CPU-to-GPU transfer
          prefetch_factor=2)      # Prefetch 2 batches ahead
```

**Impact:**
- `pin_memory=True`: Uses pinned (page-locked) memory for faster transfer
- `prefetch_factor=2`: Pipeline efficiency, GPUs never wait for data
- Expected **2-3x speedup** in data loading

---

### 4. Model Forward Pass Vectorization

**Before (SLOW - 160+ tensor creations per batch):**
```python
for b in range(batch_size):
    for t in range(prev_region_len[b]):  # 80 iterations
        trans_embed = context_embed[b, t] + self.token_type_embedding(
            torch.tensor([0], device=self.device))  # NEW TENSOR EACH TIME!
        seq_embeds.append(trans_embed)
```

**After (FAST - 3 tensor creations per batch):**
```python
# Pre-compute once
token_type_0 = self.token_type_embedding(torch.tensor([0], device=self.device))
token_type_1 = self.token_type_embedding(torch.tensor([1], device=self.device))
token_type_2 = self.token_type_embedding(torch.tensor([2], device=self.device))

for b in range(batch_size):
    # Vectorized operation - no loop!
    prev_embeds = context_embed[b, :prev_region_len[b]] + token_type_0
    seq_embeds.append(prev_embeds)
```

**Impact:**
- **80x fewer tensor creations** (3 vs 240 per batch)
- **No CPU-GPU transfer overhead** in tight loops
- Expected **5-10x speedup** in forward pass
- This was likely the **primary bottleneck**

---

### 5. Increased Batch Size

**Before:**
```yaml
train_batch_size: 512  # ~128 per GPU
```

**After:**
```yaml
train_batch_size: 1024  # 256 per GPU
```

**Impact:**
- Better GPU utilization (GPUs are compute-bound, not memory-bound at 512)
- Amortizes fixed overhead costs
- More stable gradients with larger batches

---

## Expected Performance Improvement

| Component | Before | After | Speedup |
|-----------|--------|-------|---------|
| Data Loading | Bottleneck | Optimized | 2-3x |
| Collate Function | O(n) operations | Pre-allocated | 1.5-2x |
| Model Forward | 240 tensor/batch | 3 tensor/batch | 5-10x |
| Batch Size | 128/GPU | 256/GPU | 1.5x |
| **Total Expected** | **7s/it** | **~0.1-0.3s/it** | **25-70x** |

## Realistic Estimate

With all optimizations, expect training speed to improve from:
- **7 seconds/iteration → 0.2-0.5 seconds/iteration**
- **~15-35x overall speedup**

This translates to:
- **Before:** 194 hours for 100k steps
- **After:** 5.5-13 hours for 100k steps

## Monitoring

After restarting training, monitor:

```bash
# Watch training speed
tensorboard --logdir runs/

# GPU utilization
watch -n 1 nvidia-smi

# Expected metrics:
# - GPU Utilization: 90-100%
# - Iteration Speed: 0.2-0.5s/it (vs 7s/it before)
# - GPU Memory: ~50-70% per GPU
```

## If Still Slow

If training is still slow after these optimizations:

1. **Check GPU utilization**: `nvidia-smi` should show 90-100%
2. **Profile the code**:
   ```python
   import torch.profiler
   with torch.profiler.profile(...) as prof:
       # training loop
   print(prof.key_averages().table())
   ```
3. **Increase batch size further** if GPU memory allows (try 1536 or 2048)
4. **Enable gradient checkpointing** if memory-bound
5. **Check network bandwidth** if using distributed training across nodes

## Key Takeaway

The main bottleneck was **creating torch.tensors in tight loops** during forward pass. This caused:
- Thousands of small CPU→GPU memory transfers per iteration
- Python overhead for each tensor creation
- Synchronization points blocking GPU execution

**Vectorization is critical for performance!**
