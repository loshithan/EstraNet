"""
Test script to verify MambaNet architecture builds correctly.
Quick validation that the Mamba integration works.
"""

import tensorflow as tf
import numpy as np
from mamba_transformer import MambaNet

# Configuration matching test_ascad.py settings
CONFIG = {
    'n_layer': 2,
    'd_model': 128,
    'd_state': 16,
    'd_conv': 4,
    'expand_factor': 2,
    'd_inner': 256,
    'd_head_softmax': 16,
    'n_head_softmax': 8,
    'dropout': 0.05,
    'n_classes': 256,
    'conv_kernel_size': 3,
    'n_conv_layer': 2,
    'pool_size': 20,
    'beta_hat_2': 150,
    'model_normalization': 'preLC',
    'use_ff': False,
    'softmax_attn': True,
    'output_attn': False
}

def test_model_build():
    """Test that the model builds without errors."""
    print("=" * 60)
    print("Testing MambaNet Model Build")
    print("=" * 60)
    
    # Create model
    print("\n1. Creating MambaNet model...")
    model = MambaNet(**CONFIG)
    print("✓ Model created successfully")
    
    # Create dummy input
    batch_size = 4
    input_length = 10000  # ASCAD trace length
    dummy_input = tf.random.normal((batch_size, input_length))
    print(f"\n2. Created dummy input: shape {dummy_input.shape}")
    
    # Forward pass
    print("\n3. Running forward pass...")
    try:
        output = model(dummy_input, training=False)
        logits = output[0]
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {logits.shape}")
        print(f"  Expected: ({batch_size}, {CONFIG['n_classes']})")
        
        # Verify output shape
        assert logits.shape == (batch_size, CONFIG['n_classes']), \
            f"Output shape mismatch! Got {logits.shape}, expected ({batch_size}, {CONFIG['n_classes']})"
        print("✓ Output shape verified")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise
    
    # Test training mode
    print("\n4. Testing training mode...")
    try:
        output = model(dummy_input, training=True)
        print("✓ Training mode works")
    except Exception as e:
        print(f"✗ Training mode failed: {e}")
        raise
    
    # Count parameters
    print("\n5. Model Summary:")
    print(f"  Total parameters: {model.count_params():,}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nYou can now use MambaNet in place of Transformer!")
    print("Simply replace:")
    print("  from transformer import Transformer")
    print("with:")
    print("  from mamba_transformer import MambaNet")
    print("=" * 60)

if __name__ == "__main__":
    test_model_build()
