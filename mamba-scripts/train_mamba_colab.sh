#!/bin/bash

# Training script for Mamba model in Google Colab
# Optimized for Colab's free GPU (T4)

python train_trans.py \
    --data_path=data/ASCAD.h5 \
    --checkpoint_dir=./checkpoints_mamba \
    --model_type=mamba \
    --dataset=ASCAD \
    --input_length=10000 \
    --eval_batch_size=32 \
    --n_layer=2 \
    --d_model=128 \
    --d_inner=256 \
    --n_head_softmax=8 \
    --d_head_softmax=16 \
    --dropout=0.05 \
    --conv_kernel_size=3 \
    --n_conv_layer=2 \
    --pool_size=20 \
    --beta_hat_2=150 \
    --model_normalization=preLC \
    --softmax_attn=True \
    --do_train=True \
    --learning_rate=0.00025 \
    --clip=0.25 \
    --min_lr_ratio=0.004 \
    --warmup_steps=0 \
    --train_batch_size=256 \
    --train_steps=50000 \
    --iterations=500 \
    --save_steps=5000 \
    --result_path=results/mamba
