# Multi-GPU Training Quick Reference

## 🚀 Quick Start (3 Commands)

```bash
# 1. One-time setup
accelerate config

# 2. Launch training with all GPUs
accelerate launch train.py

# 3. Monitor training
tensorboard --logdir runs/
```

## 📋 Common Commands

### Launch Training

```bash
# Use all available GPUs
accelerate launch train.py

# Use specific number of GPUs
accelerate launch --num_processes 2 train.py

# Use specific GPUs by ID
accelerate launch --gpu_ids 0,1 train.py

# With mixed precision (faster)
accelerate launch --mixed_precision fp16 train.py

# Use launcher script (recommended)
./launch_multi_gpu.sh 2          # Linux/Mac
.\launch_multi_gpu.ps1 2          # Windows
```

### Monitor Training

```bash
# Watch GPU usage
watch -n 1 nvidia-smi             # Linux
while ($true) { nvidia-smi; sleep 1 }  # PowerShell

# View logs
tensorboard --logdir runs/

# Check training log
tail -f logs/multi_gpu_*/training.log
```

## ⚙️ Configuration Options

### In `config/model/ad_dr_compressed.yaml`:

```yaml
# Enable mixed precision (2x faster)
mixed_precision: 'fp16'  # 'no', 'fp16', or 'bf16'

# Gradient accumulation (if OOM)
gradient_accumulation_steps: 1

# Batch size per GPU
train_batch_size: 512

# Number of data loading workers
num_workers: 4
```

## 🎯 Performance Tuning

### GPU Memory Management

| Issue | Solution |
|-------|----------|
| Out of Memory | Reduce `train_batch_size` or increase `gradient_accumulation_steps` |
| Underutilized GPU | Increase `train_batch_size` |
| Slow data loading | Increase `num_workers` |
| Want faster training | Set `mixed_precision: 'fp16'` |

### Recommended Settings

**2 GPUs (e.g., 2x RTX 3090 24GB):**
```yaml
mixed_precision: 'fp16'
train_batch_size: 512
gradient_accumulation_steps: 1
num_workers: 4
```

**4 GPUs (e.g., 4x A100 40GB):**
```yaml
mixed_precision: 'fp16'
train_batch_size: 256
gradient_accumulation_steps: 1
num_workers: 4
```

**8 GPUs (Large cluster):**
```yaml
mixed_precision: 'fp16'
train_batch_size: 128
gradient_accumulation_steps: 2
num_workers: 4
```

## 🔧 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
train_batch_size: 256  # was 512

# Or increase gradient accumulation
gradient_accumulation_steps: 2
```

### Slow Training
```bash
# Enable mixed precision
mixed_precision: 'fp16'

# Increase batch size (if memory allows)
train_batch_size: 1024

# Increase data loading workers
num_workers: 8
```

### Uneven GPU Utilization
```bash
# Check if all GPUs are visible
nvidia-smi

# Verify correct number of processes
accelerate launch --num_processes N train.py
```

### Hanging During Training
```bash
# Set debug mode
export NCCL_DEBUG=INFO
accelerate launch train.py

# Check network (multi-node)
ping other_node_ip
```

## 📊 Expected Performance

Training time for 50k steps (darkroom):

| GPUs | Time | Speedup |
|------|------|---------|
| 1x RTX 3090 | 8h | 1.0x |
| 2x RTX 3090 | 4.5h | 1.8x |
| 4x A100 | 2.5h | 3.2x |
| 8x A100 | 1.5h | 5.3x |

*With `mixed_precision='fp16'`*

## 🔍 Verification Checklist

- [ ] Run `accelerate config` (one-time setup)
- [ ] Set `mixed_precision: 'fp16'` in config
- [ ] Check GPU availability: `nvidia-smi`
- [ ] Launch training: `accelerate launch train.py`
- [ ] Monitor GPU usage: all GPUs should show high utilization
- [ ] Check startup message: "Number of processes: N"
- [ ] Verify checkpoints save correctly
- [ ] Training time should be faster with more GPUs

## 📚 Additional Resources

- **Full Guide**: See `MULTI_GPU_TRAINING.md`
- **Accelerate Docs**: https://huggingface.co/docs/accelerate
- **PyTorch DDP**: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

## 💡 Pro Tips

1. **Always use mixed precision** for 2x speedup
2. **Start with 2 GPUs** to verify setup
3. **Monitor `nvidia-smi`** during training
4. **Use launcher scripts** for reproducibility
5. **Save checkpoints frequently** (every 1k steps)
6. **Profile first** with single GPU to find bottlenecks
7. **Batch size × num_gpus** = effective batch size
8. **Linear speedup** up to 4-8 GPUs typically

## 🐛 Debug Commands

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check number of GPUs
python -c "import torch; print(torch.cuda.device_count())"

# Check accelerate config
accelerate env

# Test accelerate
accelerate test

# Verbose logging
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

---

**Quick Help**: For more details, see `MULTI_GPU_TRAINING.md`
