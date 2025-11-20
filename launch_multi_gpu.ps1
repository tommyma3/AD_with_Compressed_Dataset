# Multi-GPU Training Launcher Script (PowerShell)
# Usage: .\launch_multi_gpu.ps1 [num_gpus] [config_path]
#
# Examples:
#   .\launch_multi_gpu.ps1 2                                    # Use 2 GPUs with default config
#   .\launch_multi_gpu.ps1 4 config/model/ad_dr_compressed.yaml # Use 4 GPUs with custom config
#   .\launch_multi_gpu.ps1 all                                  # Use all available GPUs

param(
    [string]$NumGPUs = "2",
    [string]$ConfigPath = "config/model/ad_dr_compressed.yaml"
)

$ErrorActionPreference = "Stop"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Multi-GPU Training Launcher" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Configuration:"
Write-Host "  Number of GPUs: $NumGPUs"
Write-Host "  Config file: $ConfigPath"
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Check if accelerate is installed
try {
    $null = Get-Command accelerate -ErrorAction Stop
} catch {
    Write-Host "ERROR: accelerate is not installed" -ForegroundColor Red
    Write-Host "Please install with: pip install accelerate"
    exit 1
}

# Check if config file exists
if (-not (Test-Path $ConfigPath)) {
    Write-Host "ERROR: Config file not found: $ConfigPath" -ForegroundColor Red
    exit 1
}

# Check if accelerate is configured
$accelConfigPath = Join-Path $env:USERPROFILE ".cache\huggingface\accelerate\default_config.yaml"
if (-not (Test-Path $accelConfigPath)) {
    Write-Host "WARNING: accelerate is not configured" -ForegroundColor Yellow
    Write-Host "Running accelerate config..."
    accelerate config
}

# Check GPU availability
try {
    $gpuInfo = nvidia-smi --query-gpu=name --format=csv,noheader
    $numAvailableGPUs = ($gpuInfo | Measure-Object).Count
    Write-Host "Available GPUs: $numAvailableGPUs" -ForegroundColor Green
} catch {
    Write-Host "ERROR: nvidia-smi not found. Are NVIDIA drivers installed?" -ForegroundColor Red
    exit 1
}

if ($NumGPUs -eq "all") {
    $NumGPUs = $numAvailableGPUs
    Write-Host "Using all $NumGPUs GPUs" -ForegroundColor Green
}

$numGPUsInt = [int]$NumGPUs
if ($numGPUsInt -gt $numAvailableGPUs) {
    Write-Host "ERROR: Requested $NumGPUs GPUs but only $numAvailableGPUs available" -ForegroundColor Red
    exit 1
}

# List available GPUs
Write-Host ""
Write-Host "GPU Information:" -ForegroundColor Cyan
nvidia-smi --query-gpu=index,name,memory.total --format=csv
Write-Host ""

# Create log directory
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "logs\multi_gpu_$timestamp"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

Write-Host "Logs will be saved to: $logDir" -ForegroundColor Yellow
Write-Host ""
Write-Host "Starting training..." -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Launch training
$logFile = Join-Path $logDir "training.log"
$command = "accelerate launch --num_processes $NumGPUs --mixed_precision fp16 train.py"

Write-Host "Running: $command" -ForegroundColor Gray
Write-Host ""

# Run and capture output
& accelerate launch --num_processes $NumGPUs --mixed_precision fp16 train.py 2>&1 | Tee-Object -FilePath $logFile

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Training completed!" -ForegroundColor Green
Write-Host "Logs saved to: $logFile" -ForegroundColor Yellow
Write-Host "==================================================" -ForegroundColor Cyan
