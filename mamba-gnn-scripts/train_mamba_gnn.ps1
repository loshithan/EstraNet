# Mamba-GNN Training Script for Windows (EstraNet-aligned configuration)
# Run this to train Mamba-GNN with same hyperparameters as EstraNet for fair comparison

Write-Host "==================================================" -ForegroundColor Green
Write-Host "  Mamba-GNN Training (EstraNet-aligned config)" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green

# Configuration matching EstraNet defaults
$DATA_PATH = "data/ASCAD.h5"
$CHECKPOINT_DIR = "checkpoints/mamba_gnn_estranet"
$RESULT_PATH = "results/mamba_gnn_estranet"

# Create directories
New-Item -ItemType Directory -Force -Path $CHECKPOINT_DIR | Out-Null
New-Item -ItemType Directory -Force -Path "results" | Out-Null

# Training with EstraNet-matched hyperparameters
python scripts/train_mamba_gnn.py `
    --data_path=$DATA_PATH `
    --checkpoint_dir=$CHECKPOINT_DIR `
    --result_path=$RESULT_PATH `
    --do_train `
    --target_byte=2 `
    --input_length=700 `
    --train_batch_size=256 `
    --eval_batch_size=32 `
    --train_steps=100000 `
    --iterations=500 `
    --eval_steps=500 `
    --save_steps=10000 `
    --max_eval_batch=312 `
    --learning_rate=0.00025 `
    --clip=0.25 `
    --min_lr_ratio=0.004 `
    --warmup_steps=1000 `
    --d_model=128 `
    --mamba_layers=4 `
    --gnn_layers=3 `
    --k_neighbors=8 `
    --dropout=0.1

Write-Host ""
Write-Host "Training completed! Checkpoints saved to: $CHECKPOINT_DIR" -ForegroundColor Cyan
Write-Host ""
