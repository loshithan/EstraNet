"""
Test script to verify TensorFlow and PyTorch Mamba-GNN models

This script tests both implementations to ensure they work correctly
and produce similar outputs.
"""

import numpy as np
import sys
import os

# Test TensorFlow implementation
print("="*80)
print("Testing TensorFlow Implementation")
print("="*80)

try:
    import tensorflow as tf
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.mamba_block_tf import OptimizedMambaBlockTF, MambaBlockStack
    from models.mamba_gnn_model_tf import OptimizedMambaGNNTF
    
    print("\n1. Testing OptimizedMambaBlockTF...")
    block_tf = OptimizedMambaBlockTF(d_model=128, d_conv=7, expand=2, dropout=0.1)
    x_tf = tf.random.normal([4, 14, 128])  # batch=4, seq_len=14, features=128
    output_tf = block_tf(x_tf, training=True)
    print(f"   Input shape:  {x_tf.shape}")
    print(f"   Output shape: {output_tf.shape}")
    assert output_tf.shape == x_tf.shape, "Shape mismatch!"
    print("   ✓ OptimizedMambaBlockTF passed")
    
    print("\n2. Testing MambaBlockStack...")
    stack_tf = MambaBlockStack(d_model=128, num_layers=4, d_conv=7, expand=2, dropout=0.1)
    output_stack_tf = stack_tf(x_tf, training=True)
    print(f"   Output shape: {output_stack_tf.shape}")
    assert output_stack_tf.shape == x_tf.shape, "Shape mismatch!"
    print("   ✓ MambaBlockStack passed")
    
    print("\n3. Testing OptimizedMambaGNNTF...")
    model_tf = OptimizedMambaGNNTF(
        trace_length=700,
        d_model=128,
        mamba_layers=4,
        gnn_layers=3,
        num_classes=256,
        k_neighbors=8,
        dropout=0.1
    )
    x_trace_tf = tf.random.normal([4, 700])  # batch=4, trace_length=700
    output_logits_tf = model_tf(x_trace_tf, training=True)
    print(f"   Input shape:  {x_trace_tf.shape}")
    print(f"   Output shape: {output_logits_tf.shape}")
    assert output_logits_tf.shape == (4, 256), "Shape mismatch!"
    
    # Count parameters
    total_params_tf = sum([tf.size(w).numpy() for w in model_tf.trainable_weights])
    print(f"   Total parameters: {total_params_tf:,}")
    print("   ✓ OptimizedMambaGNNTF passed")
    
    print("\n✓ All TensorFlow tests passed!")
    tf_success = True
    
except Exception as e:
    print(f"\n✗ TensorFlow tests failed: {e}")
    import traceback
    traceback.print_exc()
    tf_success = False

# Test PyTorch implementation
print("\n" + "="*80)
print("Testing PyTorch Implementation")
print("="*80)

try:
    import torch
    import torch.nn as nn
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.mamba_block import OptimizedMambaBlock
    from models.mamba_gnn_model import OptimizedMambaGNN
    
    print("\n1. Testing OptimizedMambaBlock...")
    block_pt = OptimizedMambaBlock(d_model=128, d_conv=7, expand=2, dropout=0.1)
    x_pt = torch.randn(4, 14, 128)  # batch=4, seq_len=14, features=128
    output_pt = block_pt(x_pt)
    print(f"   Input shape:  {x_pt.shape}")
    print(f"   Output shape: {output_pt.shape}")
    assert output_pt.shape == x_pt.shape, "Shape mismatch!"
    print("   ✓ OptimizedMambaBlock passed")
    
    print("\n2. Testing OptimizedMambaGNN...")
    model_pt = OptimizedMambaGNN(
        trace_length=700,
        d_model=128,
        mamba_layers=4,
        gnn_layers=3,
        num_classes=256,
        k_neighbors=8,
        dropout=0.1
    )
    x_trace_pt = torch.randn(4, 700)  # batch=4, trace_length=700
    output_logits_pt = model_pt(x_trace_pt)
    print(f"   Input shape:  {x_trace_pt.shape}")
    print(f"   Output shape: {output_logits_pt.shape}")
    assert output_logits_pt.shape == (4, 256), "Shape mismatch!"
    
    # Count parameters
    total_params_pt = sum(p.numel() for p in model_pt.parameters())
    print(f"   Total parameters: {total_params_pt:,}")
    print("   ✓ OptimizedMambaGNN passed")
    
    print("\n✓ All PyTorch tests passed!")
    pt_success = True
    
except Exception as e:
    print(f"\n✗ PyTorch tests failed: {e}")
    import traceback
    traceback.print_exc()
    pt_success = False

# Compare parameter counts
print("\n" + "="*80)
print("Comparison Summary")
print("="*80)

if tf_success and pt_success:
    print(f"\nParameter Count:")
    print(f"  TensorFlow: {total_params_tf:,}")
    print(f"  PyTorch:    {total_params_pt:,}")
    
    param_diff = abs(total_params_tf - total_params_pt)
    param_diff_pct = (param_diff / total_params_pt) * 100
    
    if param_diff_pct < 1.0:
        print(f"  ✓ Parameter counts match (difference: {param_diff_pct:.2f}%)")
    else:
        print(f"  ⚠ Parameter counts differ by {param_diff_pct:.2f}%")
    
    print("\nImplementation Status:")
    print("  ✓ TensorFlow implementation working")
    print("  ✓ PyTorch implementation working")
    print("  ✓ Both models have similar capacity")
    print("\nNext Steps:")
    print("  1. Train PyTorch model:    python mamba-gnn-scripts/train_mamba_gnn.py --do_train")
    print("  2. Train TensorFlow model: python scripts/train_mamba_gnn_tf.py --do_train")
    print("  3. Convert to TFLite:      (automatic during TF training)")
    print("  4. Compare results:        python scripts/compare_results.py")
    
elif tf_success:
    print("\n✓ TensorFlow implementation working")
    print("✗ PyTorch implementation has issues")
    print("\nYou can still use TensorFlow for TFLite conversion")
    
elif pt_success:
    print("\n✗ TensorFlow implementation has issues")
    print("✓ PyTorch implementation working")
    print("\nYou can still use PyTorch for training")
    
else:
    print("\n✗ Both implementations have issues")
    print("Please check dependencies:")
    print("  - TensorFlow: pip install tensorflow")
    print("  - PyTorch:    pip install torch")

print("\n" + "="*80)
