#!/bin/bash

# Evaluate trained Mamba-GNN model using Guessing Entropy (100 trials)

echo "=================================================="
echo "  Mamba-GNN Evaluation (Guessing Entropy)"
echo "=================================================="

CHECKPOINT_DIR="checkpoints/mamba_gnn_estranet"
DATA_PATH="data/ASCAD.h5"
RESULT_PATH="results/mamba_gnn_estranet_eval"

# Evaluate specific checkpoint (change checkpoint_idx as needed)
# checkpoint_idx=0 means latest checkpoint
# checkpoint_idx=10000 means checkpoint at step 10000

python scripts/train_mamba_gnn.py \
    --data_path=$DATA_PATH \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --result_path=$RESULT_PATH \
    --checkpoint_idx=0 \
    --target_byte=2 \
    --input_length=700 \
    --eval_batch_size=32 \
    --d_model=128 \
    --mamba_layers=4 \
    --gnn_layers=3 \
    --k_neighbors=8 \
    --dropout=0.1

echo ""
echo "Evaluation completed! Results saved to: ${RESULT_PATH}.txt"
echo ""
