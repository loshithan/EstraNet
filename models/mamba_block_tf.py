"""
TensorFlow implementation of Optimized Mamba Block for Temporal Modeling

This module provides TensorFlow/Keras version of OptimizedMambaBlock for side-channel
analysis, enabling TFLite conversion for deployment.

Key Features:
- Depthwise separable convolutions for efficiency
- Batch normalization for stable training
- Residual connections with learnable scaling
- GELU activation for smooth gradients
- Compatible with EstraNet architecture
- Ready for TFLite conversion
"""

import tensorflow as tf
import numpy as np


class OptimizedMambaBlockTF(tf.keras.layers.Layer):
    """
    TensorFlow implementation of Optimized Mamba Block.
    
    This block processes temporal sequences using:
    1. Layer normalization for input stabilization
    2. Linear projection to expand dimensions
    3. Depthwise convolution for temporal feature extraction
    4. Batch normalization for training stability
    5. Pointwise convolution for feature mixing
    6. Gated activation mechanism
    7. Residual connection with learnable scaling
    
    Args:
        d_model (int): Model dimension (input and output feature size)
        d_conv (int): Convolution kernel size. Default: 7
        expand (int): Expansion factor for internal dimension. Default: 2
        dropout (float): Dropout probability. Default: 0.1
        name (str): Layer name
    
    Shape:
        - Input: (batch_size, sequence_length, d_model)
        - Output: (batch_size, sequence_length, d_model)
    
    Example:
        >>> block = OptimizedMambaBlockTF(d_model=128, d_conv=7, expand=2)
        >>> x = tf.random.normal([32, 14, 128])  # batch=32, seq_len=14, features=128
        >>> output = block(x)
        >>> print(output.shape)  # TensorShape([32, 14, 128])
    """
    
    def __init__(self, d_model, d_conv=7, expand=2, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_conv = d_conv
        self.dropout_rate = dropout
        
        # Layer normalization
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        
        # Input projection (expands to d_inner * 2 for gating)
        self.in_proj = tf.keras.layers.Dense(
            self.d_inner * 2, 
            use_bias=False,
            name='in_proj'
        )
        
        # Depthwise convolution for temporal modeling
        self.conv1d = tf.keras.layers.Conv1D(
            filters=self.d_inner,
            kernel_size=d_conv,
            padding='causal',  # Causal padding for temporal sequence
            groups=self.d_inner,  # Depthwise convolution
            use_bias=False,
            name='depthwise_conv'
        )
        
        # Batch normalization
        self.conv_norm = tf.keras.layers.BatchNormalization(name='conv_norm')
        
        # Pointwise convolution for feature mixing
        self.pointwise = tf.keras.layers.Conv1D(
            filters=self.d_inner,
            kernel_size=1,
            name='pointwise_conv'
        )
        
        # Output projection
        self.out_proj = tf.keras.layers.Dense(
            d_model, 
            use_bias=False,
            name='out_proj'
        )
        
        # Dropout
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def build(self, input_shape):
        # Learnable scaling parameter for residual connection
        self.gamma = self.add_weight(
            name='gamma',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, x, training=False):
        """
        Forward pass of the Mamba block.
        
        Args:
            x (tf.Tensor): Input tensor of shape (batch, seq_len, d_model)
            training (bool): Whether in training mode
            
        Returns:
            tf.Tensor: Output tensor of shape (batch, seq_len, d_model)
        """
        residual = x
        
        # Layer normalization
        x = self.norm1(x)
        
        # Input projection and split for gating
        xz = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x_input, z = tf.split(xz, 2, axis=-1)  # Each (batch, seq_len, d_inner)
        
        # Depthwise convolution
        x_conv = self.conv1d(x_input)  # (batch, seq_len, d_inner)
        
        # Batch normalization
        x_conv = self.conv_norm(x_conv, training=training)
        
        # Pointwise convolution
        x_conv = self.pointwise(x_conv)
        
        # GELU activation
        x_conv = tf.nn.gelu(x_conv)
        
        # Gated activation
        y = x_conv * tf.nn.sigmoid(z)
        
        # Output projection
        output = self.out_proj(y)
        
        # Dropout
        output = self.dropout(output, training=training)
        
        # Residual connection with learnable scaling
        return residual + self.gamma * output
    
    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'd_conv': self.d_conv,
            'expand': self.d_inner // self.d_model,
            'dropout': self.dropout_rate,
        })
        return config


class MambaBlockStack(tf.keras.layers.Layer):
    """
    Stack of multiple Mamba blocks for deep temporal modeling.
    
    Args:
        d_model (int): Model dimension
        num_layers (int): Number of Mamba blocks to stack
        d_conv (int): Convolution kernel size. Default: 7
        expand (int): Expansion factor. Default: 2
        dropout (float): Dropout probability. Default: 0.1
    """
    
    def __init__(self, d_model, num_layers, d_conv=7, expand=2, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Create stack of Mamba blocks
        self.blocks = [
            OptimizedMambaBlockTF(
                d_model=d_model,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                name=f'mamba_block_{i}'
            )
            for i in range(num_layers)
        ]
    
    def call(self, x, training=False):
        """
        Forward pass through stacked blocks.
        
        Args:
            x (tf.Tensor): Input tensor of shape (batch, seq_len, d_model)
            training (bool): Whether in training mode
            
        Returns:
            tf.Tensor: Output tensor of shape (batch, seq_len, d_model)
        """
        for block in self.blocks:
            x = block(x, training=training)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_layers': self.num_layers,
        })
        return config


# Example usage and testing
if __name__ == '__main__':
    print("Testing OptimizedMambaBlockTF...")
    
    # Test single block
    block = OptimizedMambaBlockTF(d_model=128, d_conv=7, expand=2, dropout=0.1)
    x = tf.random.normal([4, 14, 128])  # batch=4, seq_len=14, features=128
    
    print(f"Input shape: {x.shape}")
    output = block(x, training=True)
    print(f"Output shape: {output.shape}")
    print(f"✓ Single block test passed")
    
    # Test stacked blocks
    print("\nTesting MambaBlockStack...")
    stack = MambaBlockStack(d_model=128, num_layers=4, d_conv=7, expand=2, dropout=0.1)
    output = stack(x, training=True)
    print(f"Output shape: {output.shape}")
    print(f"✓ Stacked blocks test passed")
    
    # Count parameters
    total_params = sum([tf.size(w).numpy() for w in block.trainable_weights])
    print(f"\nSingle block parameters: {total_params:,}")
    
    total_params_stack = sum([tf.size(w).numpy() for w in stack.trainable_weights])
    print(f"Stacked blocks (4 layers) parameters: {total_params_stack:,}")
