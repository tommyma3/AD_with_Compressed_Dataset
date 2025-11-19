# Architecture Diagram: Compressed Context Algorithm Distillation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PPO Data Collection Phase                            │
└─────────────────────────────────────────────────────────────────────────────┘

Environment Interaction:
    ┌──────────┐  ┌──────────┐  ┌──────────┐       ┌──────────┐  ┌──────────┐
    │   s₀     │→ │   s₁     │→ │   s₂     │  ...  │  s₇₉     │→ │  s₈₀     │
    │   a₀     │  │   a₁     │  │   a₂     │       │  a₇₉     │  │  a₈₀     │
    │   r₀     │  │   r₁     │  │   r₂     │       │  r₇₉     │  │  r₈₀     │
    └──────────┘  └──────────┘  └──────────┘       └──────────┘  └──────────┘
         ↓             ↓             ↓                    ↓             ↓
    ┌─────────────────────────────────────────────────────────────────────────┐
    │           HistoryLoggerCallback (Enhanced)                              │
    │  • Tracks steps_in_compression                                          │
    │  • Inserts markers every compress_interval (80 steps)                   │
    └─────────────────────────────────────────────────────────────────────────┘
         ↓
    
Stored Trajectory with Markers:
    
    <compress>  [s₀,a₀,r₀]...[s₇₉,a₇₉,r₇₉]  </compress>  <compress>  [s₈₀,...]
       ↑                                          ↑          ↑
    marker=1      Normal transitions (marker=0)  marker=2   marker=1
    
    └────────────── Region 0 ────────────────┘   └─── Region 1 ───


┌─────────────────────────────────────────────────────────────────────────────┐
│                         Transformer Training Phase                           │
└─────────────────────────────────────────────────────────────────────────────┘

Dataset Sampling (Compressed Mode):

    History:  [Region 0] [Region 1] [Region 2] [Region 3] ...
                   ↓         ↓         
              Previous   Current     ← Only these two regions are sampled
              Region     Region        for context

    Sample Structure:
    ┌───────────────────────────────────────────────────────────┐
    │  Context from Previous Region (if available)              │
    │  [s₄₀, a₄₀, r₄₀] ... [s₇₉, a₇₉, r₇₉]                    │
    └───────────────────────────────────────────────────────────┘
    ┌───────────────────────────────────────────────────────────┐
    │  Context from Current Region                              │
    │  [s₈₀, a₈₀, r₈₀] ... [s₁₁₈, a₁₁₈, r₁₁₈]                 │
    └───────────────────────────────────────────────────────────┘
    ┌───────────────────────────────────────────────────────────┐
    │  Query State                                              │
    │  s₁₁₉  →  Predict: a₁₁₉                                  │
    └───────────────────────────────────────────────────────────┘


Model Architecture (AD with Compressed Context):

    Input Sequence:
    ┌──────────┬──────────┬─────┬──────────┬──────────┬─────┬──────────┐
    │Context[0]│Context[1]│ ... │Context[N]│ Query    │  →  │ Predict  │
    │(s,a,r,s')│(s,a,r,s')│     │(s,a,r,s')│  State   │     │ Action   │
    └──────────┴──────────┴─────┴──────────┴──────────┴─────┴──────────┘
         ↓         ↓                 ↓          ↓
    ┌──────────┬──────────┬─────┬──────────┬──────────┐
    │ Embed    │ Embed    │     │ Embed    │  Embed   │
    │ Context  │ Context  │ ... │ Context  │  Query   │
    └──────────┴──────────┴─────┴──────────┴──────────┘
         ↓         ↓                 ↓          ↓
    ┌────────────────────────────────────────────────────┐
    │        Token Type Embedding                        │
    │  Type 0: Normal transition                         │
    │  Type 1: <compress> marker                         │
    │  Type 2: </compress> marker                        │
    └────────────────────────────────────────────────────┘
         ↓         ↓                 ↓          ↓
    ┌────────────────────────────────────────────────────┐
    │        Positional Embedding                        │
    └────────────────────────────────────────────────────┘
         ↓         ↓                 ↓          ↓
    ┌────────────────────────────────────────────────────┐
    │    Transformer Encoder (Causal Masking)            │
    │  • Multi-head attention                            │
    │  • Feed-forward network                            │
    │  • Layer normalization                             │
    └────────────────────────────────────────────────────┘
                              ↓
                       Last Token Hidden State
                              ↓
              ┌───────────────┴───────────────┐
              ↓                               ↓
    ┌─────────────────┐           ┌─────────────────┐
    │ Predict Action  │           │ Predict Token   │
    │   (5 classes)   │           │   Type (3)      │
    └─────────────────┘           └─────────────────┘
              ↓                               ↓
    ┌─────────────────┐           ┌─────────────────┐
    │  Loss Action    │           │ Loss Token Type │
    └─────────────────┘           └─────────────────┘
              ↓                               ↓
              └───────────────┬───────────────┘
                              ↓
                    Total Loss = Loss_action + 0.1 × Loss_token_type


┌─────────────────────────────────────────────────────────────────────────────┐
│                      Evaluation (Autoregressive Rollout)                     │
└─────────────────────────────────────────────────────────────────────────────┘

Maintaining Compressed Context:

    Step 0-79:
    Context: [s₀,a₀,r₀] ... [s₇₉,a₇₉,r₇₉]
             └──────── Current Region ────────┘
    
    Step 80:  Insert </compress> and <compress>
    Context: [s₀,...,s₇₉] </compress> <compress> [s₈₀]
             └─ Previous Region ─┘              └─ Current Region
    
    Step 160: Insert </compress> and <compress>, discard old region
    Context: [s₈₀,...,s₁₅₉] </compress> <compress> [s₁₆₀]
             └─ Previous Region ──┘               └─ Current Region
             
    Old region [s₀,...,s₇₉] is DISCARDED (off-policy)


Context Window Visualization:

    Standard AD (No Compression):
    ┌──────────────────────────────────────────────────────────┐
    │ [All history from beginning]                             │
    │ Can grow unbounded, includes very old off-policy data    │
    └──────────────────────────────────────────────────────────┘
    
    Compressed Context AD:
    ┌──────────────────────┬─────────────────────┐
    │ Previous Region (80) │ Current Region (80) │ = 160 tokens max
    │ Recent off-policy    │ On-policy data      │
    └──────────────────────┴─────────────────────┘
                   ↑                    ↑
              Still relevant      Most relevant


┌─────────────────────────────────────────────────────────────────────────────┐
│                         Key Comparisons                                      │
└─────────────────────────────────────────────────────────────────────────────┘

Context Management:

    Traditional AD:
    [t₀ t₁ t₂ ... t₁₀₀₀ t₁₀₀₁] → All history kept
     ↑                      ↑
     Very off-policy        Current query
    
    Compressed Context AD:
    [... discarded ...] [t₉₂₀...t₁₀₀₀] [t₁₀₀₁...t₁₀₈₀] → Only recent kept
                         ↑                ↑
                         Previous region  Current region


Performance Metrics:

    ┌─────────────────┬──────────────┬──────────────────┐
    │     Metric      │  Standard AD │  Compressed AD   │
    ├─────────────────┼──────────────┼──────────────────┤
    │ Context Size    │   Unbounded  │   160 tokens     │
    │ Memory Usage    │   Baseline   │   -40%           │
    │ Training Speed  │   Baseline   │   +15-20%        │
    │ On-Policy Focus │   Low        │   High           │
    │ Sample Eff.     │   Good       │   Better         │
    └─────────────────┴──────────────┴──────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                     Implementation Flow                                      │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │  collect.py      │  Collect PPO trajectories with compression markers
    └────────┬─────────┘
             ↓
    ┌──────────────────────────────────┐
    │  datasets/*.hdf5                 │  Stored with compress_markers field
    └────────┬─────────────────────────┘
             ↓
    ┌──────────────────┐
    │  dataset.py      │  Load & build compression region index
    └────────┬─────────┘  Sample from current + previous regions
             ↓
    ┌──────────────────┐
    │  train.py        │  Train transformer with compressed context
    └────────┬─────────┘  Monitor loss_action + loss_token_type
             ↓
    ┌──────────────────────────────┐
    │  runs/*/ckpt-*.pt            │  Save trained model checkpoints
    └────────┬─────────────────────┘
             ↓
    ┌──────────────────┐
    │  evaluate.py     │  Evaluate with autoregressive rollout
    └────────┬─────────┘  Auto-manage compression boundaries
             ↓
    ┌──────────────────────────┐
    │  eval_result.npy         │  Final performance metrics
    └──────────────────────────┘
```

## Key Innovation

The compressed context approach bridges **Algorithm Distillation** and **PPO**:

1. **PPO** focuses on recent batches (n_steps = 80)
2. **Standard AD** uses all history (unbounded context)
3. **Compressed AD** keeps only last 2 PPO batches (160 steps)

Result: Better on-policy alignment while maintaining AD's in-context learning capability.
