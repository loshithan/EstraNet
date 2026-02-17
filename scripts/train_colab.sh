#!/bin/bash

# Simple Colab Training Script for EstraNet
# This is a minimal version optimized for Google Colab

# EDIT THIS: Set your data path
DATA_PATH="data/ASCAD.h5"

# EDIT THIS: Set where to save checkpoints (use Google Drive path in Colab)
CKP_DIR="./checkpoints"

# Colab-optimized settings (faster training, ~2-4 hours)
TRAIN_STEPS=50000
WARMUP_STEPS=5000
SAVE_STEPS=5000
ITERATIONS=1000

# Model configuration (same as paper)
N_LAYER=2
D_MODEL=128
BATCH_SIZE=32

# Run training
python train_trans.py \
    --data_path=${DATA_PATH} \
    --checkpoint_dir=${CKP_DIR} \
    --dataset=ASCAD \
    --input_length=10000 \
    --data_desync=200 \
    --train_batch_size=${BATCH_SIZE} \
    --eval_batch_size=${BATCH_SIZE} \
    --train_steps=${TRAIN_STEPS} \
    --warmup_steps=${WARMUP_STEPS} \
    --iterations=${ITERATIONS} \
    --save_steps=${SAVE_STEPS} \
    --n_layer=${N_LAYER} \
    --d_model=${D_MODEL} \
    --d_head=32 \
    --n_head=8 \
    --d_inner=256 \
    --n_head_softmax=8 \
    --d_head_softmax=16 \
    --dropout=0.05 \
    --conv_kernel_size=3 \
    --n_conv_layer=2 \
    --pool_size=20 \
    --d_kernel_map=512 \
    --beta_hat_2=150 \
    --model_normalization=preLC \
    --head_initialization=forward \
    --softmax_attn=True \
    --learning_rate=2.5e-4 \
    --clip=0.25 \
    --min_lr_ratio=0.004 \
    --max_eval_batch=100 \
    --do_train=True

echo ""
echo "Training completed!"
echo "Checkpoints saved to: ${CKP_DIR}"
