"""Test full Transformer model to reproduce the actual bug"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import tensorflow as tf
from transformer import Transformer

# Model config from the notebook (same as error trace)
n_layer = 2
d_model = 128
d_head = 32
n_head = 8
d_inner = 256
n_head_softmax = 8
d_head_softmax = 16
dropout = 0.05
n_classes = 256
conv_kernel_size = 3
n_conv_layer = 2
pool_size = 20
d_kernel_map = 512
beta_hat_2 = 150
model_normalization = 'preLC'
head_initialization = 'forward'
softmax_attn = True

print("="*70)
print("Testing Full Transformer Model (matching notebook config)")
print("="*70)

print(f"\nConfig:")
print(f"  n_layer: {n_layer}")
print(f"  d_model: {d_model}, d_head: {d_head}, n_head: {n_head}")
print(f"  d_kernel_map: {d_kernel_map}, beta_hat_2: {beta_hat_2}")
print(f"  head_initialization: {head_initialization}")

# Create model
print("\n📦 Creating Transformer model...")
try:
    model = Transformer(
        n_layer=n_layer,
        d_model=d_model,
        d_head=d_head,
        n_head=n_head,
        d_inner=d_inner,
        d_head_softmax=n_head_softmax,
        n_head_softmax=n_head_softmax,
        dropout=dropout,
        n_classes=n_classes,
        conv_kernel_size=conv_kernel_size,
        n_conv_layer=n_conv_layer,
        pool_size=pool_size,
        d_kernel_map=d_kernel_map,
        beta_hat_2=beta_hat_2,
        model_normalization=model_normalization,
        head_initialization=head_initialization,
        softmax_attn=softmax_attn,
        output_attn=False
    )
    print("✅ Model created successfully")
except Exception as e:
    print(f"❌ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Create test batch (matching error trace dimensions)
batch_size = 16
input_length = 700  # ASCAD trace length

print(f"\n📊 Creating test batch (batch={batch_size}, length={input_length})...")
test_input = tf.random.normal((batch_size, input_length))

print(f"  Input shape: {test_input.shape}")

# Try forward pass
print("\n🔍 Attempting forward pass...")
try:
    output = model(inputs=test_input, softmax_attn_smoothing=1.0, training=True)
    print(f"✅ Forward pass SUCCESSFUL!")
    print(f"  Output shape: {output[0].shape}")
except Exception as e:
    print(f"\n❌ Forward pass FAILED with error:\n")
    print(f"  {type(e).__name__}: {str(e)[:200]}")
    print(f"\n📋 Full traceback:")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
