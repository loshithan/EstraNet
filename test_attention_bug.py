"""Test script to reproduce the dimension mismatch bug in SelfAttention"""
import tensorflow as tf
import numpy as np
from fast_attention import SelfAttention

# Model config from the notebook
d_model = 128
d_head = 32
n_head = 8
dropout = 0.05
d_kernel_map = 512
head_init_range = (0., 1.0)

print("="*60)
print("Testing SelfAttention initialization and forward pass")
print("=" *60)
print(f"\nConfig:")
print(f"  d_model: {d_model}")
print(f"  d_head: {d_head}")
print(f"  n_head: {n_head}")
print(f"  d_kernel_map: {d_kernel_map}")
print(f"  head_init_range: {head_init_range}")

# Create SelfAttention layer
print("\n Creating SelfAttention layer...")
try:
    attn_layer = SelfAttention(
        d_model=d_model,
        d_head=d_head,
        n_head=n_head,
        attention_dropout=dropout,
        feature_map_type='fourier',
        normalize_attn=False,
        d_kernel_map=d_kernel_map,
        head_init_range=head_init_range
    )
    print("✅ Layer created successfully")
    
    # Check the shape of pos_ft_offsets
    print(f"\n Checking weight shapes:")
    print(f"  pos_ft_offsets shape: {attn_layer.pos_ft_offsets.shape}")
    print(f"  pos_ft_offsets value shape: {attn_layer.pos_ft_offsets.numpy().shape}")
    print(f"  pos_ft_offsets values:\n{attn_layer.pos_ft_offsets.numpy()}")
    
except Exception as e:
    print(f"❌ Failed to create layer: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Create test inputs
batch_size = 16
seq_len = 1
print(f"\nCreating test inputs (batch={batch_size}, seq={seq_len})...")

source_input = tf.random.normal((batch_size, seq_len, d_model))
pos_ft = tf.random.normal((batch_size, seq_len, d_model))
pos_ft_slopes = tf.random.normal((batch_size, seq_len, d_model))

print(f"  source_input shape: {source_input.shape}")
print(f"  pos_ft shape: {pos_ft.shape}")
print(f"  pos_ft_slopes shape: {pos_ft_slopes.shape}")

# Try forward pass
print("\n🔍 Attempting forward pass...")
try:
    output = attn_layer(source_input, pos_ft, pos_ft_slopes, training=True)
    print(f"✅ Forward pass successful!")
    print(f"  Output shape: {output[0].shape}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    
print("\n" + "="*60)
