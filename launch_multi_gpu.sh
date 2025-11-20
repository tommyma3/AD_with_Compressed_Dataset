#!/bin/bash
# Multi-GPU Training Launcher Script
# Usage: ./launch_multi_gpu.sh [num_gpus] [config_path]
#
# Examples:
#   ./launch_multi_gpu.sh 2                                    # Use 2 GPUs with default config
#   ./launch_multi_gpu.sh 4 config/model/ad_dr_compressed.yaml # Use 4 GPUs with custom config
#   ./launch_multi_gpu.sh all                                  # Use all available GPUs

set -e  # Exit on error

# Default values
NUM_GPUS="${1:-2}"
CONFIG_PATH="${2:-config/model/ad_dr_compressed.yaml}"

echo "=================================================="
echo "Multi-GPU Training Launcher"
echo "=================================================="
echo "Configuration:"
echo "  Number of GPUs: $NUM_GPUS"
echo "  Config file: $CONFIG_PATH"
echo "=================================================="
echo ""

# Check if accelerate is installed
if ! command -v accelerate &> /dev/null; then
    echo "ERROR: accelerate is not installed"
    echo "Please install with: pip install accelerate"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Check if accelerate is configured
if [ ! -f "$HOME/.cache/huggingface/accelerate/default_config.yaml" ]; then
    echo "WARNING: accelerate is not configured"
    echo "Running accelerate config..."
    accelerate config
fi

# Check GPU availability
NUM_AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Available GPUs: $NUM_AVAILABLE_GPUS"

if [ "$NUM_GPUS" = "all" ]; then
    NUM_GPUS=$NUM_AVAILABLE_GPUS
    echo "Using all $NUM_GPUS GPUs"
fi

if [ "$NUM_GPUS" -gt "$NUM_AVAILABLE_GPUS" ]; then
    echo "ERROR: Requested $NUM_GPUS GPUs but only $NUM_AVAILABLE_GPUS available"
    exit 1
fi

# List available GPUs
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Create log directory
LOG_DIR="logs/multi_gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Logs will be saved to: $LOG_DIR"
echo ""
echo "Starting training..."
echo "=================================================="
echo ""

# Launch training
accelerate launch \
    --num_processes "$NUM_GPUS" \
    --mixed_precision fp16 \
    train.py 2>&1 | tee "$LOG_DIR/training.log"

echo ""
echo "=================================================="
echo "Training completed!"
echo "Logs saved to: $LOG_DIR/training.log"
echo "=================================================="
