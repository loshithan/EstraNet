"""
Test script for GNN-based EstraNet model
Verifies model builds correctly and checks parameter count
"""

import tensorflow as tf
import numpy as np
from gnn_estranet import GNNEstraNet

print("="*70)
print("Testing GNN-based EstraNet")
print("="*70)

# Model configuration (same as training)
config = {
    'n_gcn_layers': 2,
    'd_model': 128,
    'k_neighbors': 5,
    'graph_pooling': 'mean',
    'd_head_softmax': 16,
    'n_head_softmax': 8,
    'dropout': 0.05,
    'n_classes': 256,
    'conv_kernel_size': 3,
    'n_conv_layer': 2,
    'pool_size': 2,
    'beta_hat_2': 150,
    'model_normalization': 'preLC',
    'softmax_attn': True,
    'output_attn': False
}

print("\n🏗️  Building GNN model...")
print(f"Configuration:")
for key, val in config.items():
    print(f"  {key}: {val}")

model = GNNEstraNet(**config)

# Build the model with dummy input
print("\n🔨 Building model graph with dummy input...")
dummy_input = tf.zeros((1, 700))
output = model(dummy_input, training=False)[0]

print(f"✅ Model built successfully!")
print(f"   Input shape:  {dummy_input.shape}")
print(f"   Output shape: {output.shape}")

# Count parameters
total_params = model.count_params()
print(f"\n📊 Model Parameters: {total_params:,}")

# Compare to baselines
print(f"\n📈 Comparison:")
print(f"   Transformer: 431,233 parameters")
print(f"   Mamba:       425,569 parameters")
print(f"   GNN:         {total_params:,} parameters")

if total_params < 431233:
    reduction = ((431233 - total_params) / 431233) * 100
    print(f"   🎯 GNN has {reduction:.1f}% fewer parameters than Transformer!")
else:
    print(f"   ⚠️  GNN has more parameters than Transformer")

# Test forward pass with random data
print(f"\n🧪 Testing forward pass with random data...")
test_input = tf.random.normal((4, 700))
test_output = model(test_input, training=True)[0]
print(f"   Batch input shape:  {test_input.shape}")
print(f"   Batch output shape: {test_output.shape}")
print(f"   ✅ Forward pass successful!")

# Summary
print(f"\n{'='*70}")
print("✅ GNN Model Test Complete!")
print(f"{'='*70}")

print(f"\nYou can now use --model_type=gnn in train_trans.py")
print(f"\nExample:")
print(f"  python train_trans.py \\")
print(f"    --model_type=gnn \\")
print(f"    --data_path=data/ASCAD.h5 \\")
print(f"    --train_steps=50000 \\")
print(f"    --checkpoint_dir=./checkpoints_gnn")
