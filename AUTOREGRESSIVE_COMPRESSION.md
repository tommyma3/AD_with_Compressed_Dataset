# Autoregressive Compression Training

## Overview

This implementation allows the transformer to **actively manage its context window** by learning to autoregressively predict when to emit compression tokens (`<compress>` and `</compress>`). This mimics PPO's on-policy nature where the agent focuses on recent experiences.

## Key Design Principles

### 1. **Compression Tokens as Separate Sequence Elements**

Compression tokens are **INSERTED** into the sequence, not used as replacements:

```
Sequence structure:
<compress> t1, t2, ..., t80 </compress> <compress> t81, t82, ..., query_state

Where:
- <compress>  : compress_start token (type=1)
- </compress> : compress_end token (type=2)
- t1, t2, ... : normal transitions (type=0)
```

### 2. **Training: Sample Consecutive Region Pairs**

During training, the dataset samples **one compression region + its previous region**:

- **Dataset structure**: 4 compression regions stored (328 tokens capacity)
- **Training samples**: Region pairs (e.g., region 2 + region 1, region 3 + region 2)
- **Context window**: Always contains exactly 2 consecutive compression regions
- **Targets**: 
  - Predict actions at each timestep
  - Predict `</compress>` token at compression interval boundaries (every 80 steps)

### 3. **Evaluation: Autoregressive Context Pruning**

During evaluation, the model:

1. **Predicts token types** at each step using `pred_token_type` head
2. **Detects `</compress>` predictions** (when `predicted_token_type == 2`)
3. **Prunes old compression region** automatically when `</compress>` is emitted
4. **Maintains sliding window** of at most 2 compression regions

**Example evaluation sequence:**

```python
# Step 0-79: First region
<compress> t1, t2, ..., t79, query_state
# Model predicts actions

# Step 80: Model predicts </compress>
<compress> t1, t2, ..., t80 </compress> <compress> query_state
# Old region completed, new region starts

# Step 81-159: Continue in new region
<compress> t1, t2, ..., t80 </compress> <compress> t81, t82, ..., query_state

# Step 160: Model predicts </compress> again
<compress> t1, t2, ..., t80 </compress> <compress> t81, ..., t160 </compress> <compress> query_state
# PRUNE: throw away first region (t1-t80)

# After pruning:
<compress> t81, ..., t160 </compress> <compress> query_state
```

## Configuration

### Key Parameters (`config/model/ad_dr_compressed.yaml`)

```yaml
use_compressed_context: true      # Enable compression mode
compress_interval: 80             # Must match n_steps from PPO config
n_transit: 328                    # 4 regions × 82 tokens/region for flexibility
```

**Why n_transit = 328?**
- Each compression region: 80 transitions + 1 `<compress>` + 1 `</compress>` = 82 tokens
- Store 4 regions for training flexibility: 4 × 82 = 328 tokens
- During evaluation, only 2 regions are maintained (pruned automatically)

### PPO Configuration (`config/algorithm/ppo_darkroom.yaml`)

```yaml
n_steps: 80  # Must match compress_interval
```

## Implementation Details

### Dataset (`dataset.py`)

**Changes:**
- `_get_compressed_context_sample()`: Samples consecutive region pairs
- Returns `prev_region_len` and `curr_region_len` instead of `compress_markers`
- `target_token_type`: 0=action, 2=compress_end (at interval boundaries)

### Model (`model/ad.py`)

**Forward Pass:**
- Builds sequence: `<compress> prev_region </compress> <compress> curr_region query`
- Inserts compression tokens as separate elements
- Predicts both `logits_action` and `logits_token_type`
- Loss computed based on `target_token_type`:
  - If type=0: compute action loss
  - If type=2: compute token type loss
  - Combined loss: `loss_action + 0.1 * loss_token_type`

**Evaluation (`evaluate_in_context`):**
- Maintains `current_region` and `previous_region` buffers
- At each step:
  1. Predict `logits_token_type`
  2. If `predicted_token_type == 2`:
     - Emit `</compress>` token
     - Save current region as previous
     - Prune old previous region
     - Start new region with `<compress>`
  3. Otherwise: predict action and add transition to current region

### Training Loop (`train.py`)

**Metrics Logged:**
- `train/loss_action`: Action prediction loss
- `train/acc_action`: Action prediction accuracy
- `train/loss_token_type`: Compression token prediction loss
- `train/acc_token_type`: Compression token prediction accuracy (whether model correctly predicts when to emit `</compress>`)
- Same metrics for `test/` split

## Workflow

### 1. Data Collection

```bash
python collect.py
```

**What happens:**
- PPO collects trajectories with `n_steps=80`
- `HistoryLoggerCallback` inserts markers at boundaries
- Data saved to `datasets/history_darkroom_PPO_alg-seed42.hdf5`

### 2. Training

```bash
# Single GPU
python train.py

# Multi-GPU
accelerate launch train.py
```

**What the model learns:**
- **Action prediction**: Choose correct actions given context
- **Compression token prediction**: Emit `</compress>` at appropriate intervals
- **Context management**: Focus on recent 2 compression regions (on-policy data)

### 3. Evaluation

```bash
python evaluate.py
```

**What happens:**
- Model autoregressively predicts actions AND compression tokens
- When `</compress>` is predicted:
  - Old compression region is pruned
  - Model focuses on most recent data
- Context window adaptively managed by the model itself

## Advantages Over Fixed-Interval Compression

### Previous Approach (Fixed):
- Compression boundaries hardcoded every 80 steps
- Model passively observes markers
- No active learning of when to compress

### New Approach (Autoregressive):
- Model **learns** when to emit `</compress>`
- Can potentially adapt compression intervals based on task dynamics
- More flexible and learnable context management
- Better alignment with on-policy RL principles

## Testing

Run unit tests to verify implementation:

```bash
python test_compressed_context.py
```

**Tests verify:**
1. Compression token insertion as separate elements
2. Model forward pass with region pairs
3. Token type prediction
4. Dataset sampling with consecutive regions
5. Backward compatibility with non-compressed mode

## Monitoring Training

Use TensorBoard to monitor compression learning:

```bash
tensorboard --logdir runs/
```

**Key metrics to watch:**
- `train/acc_token_type`: Should approach 1.0 (model learns when to compress)
- `train/acc_action`: Should improve (action prediction quality)
- `test/acc_token_type`: Should generalize well (compression is consistent)

## Troubleshooting

**Issue**: `acc_token_type` stays low
- **Cause**: Model not learning to predict compression boundaries
- **Fix**: Increase token type loss weight in model forward pass

**Issue**: `acc_action` drops when compression enabled
- **Cause**: Context pruning removes useful information
- **Fix**: Increase `n_transit` or adjust `compress_interval`

**Issue**: Out of memory during training
- **Cause**: Long sequences with multiple regions
- **Fix**: Reduce batch size or enable gradient accumulation

## Future Enhancements

1. **Adaptive Compression Intervals**: Let model learn variable-length regions
2. **Hierarchical Compression**: Multi-level compression for very long episodes
3. **Compression Quality Metrics**: Track how well pruned context maintains performance
4. **Compression Visualization**: Plot when model decides to compress during evaluation
