import tensorflow as tf
import numpy as np
from normalization import LayerScaling, LayerCentering


def shape_list(x):
    """Get shape of tensor as a list."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class MambaBlock(tf.keras.layers.Layer):
    """
    Mamba (Selective State Space Model) Block for sequence modeling.
    
    This implementation provides a drop-in replacement for TransformerLayer
    in the EstraNet architecture for side-channel analysis.
    
    Args:
        d_model: Model dimension
        d_state: State dimension (typically 16)
        d_conv: Convolution kernel size for local context (typically 4)
        expand_factor: Expansion factor for inner dimension (typically 2)
        dropout: Dropout rate
        model_normalization: Type of normalization ('preLC', 'postLC', or 'none')
    """
    
    def __init__(self, 
                 d_model, 
                 d_state=16, 
                 d_conv=4, 
                 expand_factor=2,
                 dropout=0.1,
                 model_normalization='preLC',
                 **kwargs):
        super().__init__(**kwargs)
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand_factor
        self.model_normalization = model_normalization
        
        # Input projection
        self.in_proj = tf.keras.layers.Dense(self.d_inner * 2, use_bias=False)
        
        # Convolution for local context
        self.conv1d = tf.keras.layers.Conv1D(
            filters=self.d_inner,
            kernel_size=d_conv,
            padding='same',
            groups=self.d_inner,  # Depthwise convolution
            use_bias=True
        )
        
        # SSM parameters (Selective State Space)
        self.x_proj = tf.keras.layers.Dense(d_state * 2, use_bias=False)  # For B and C
        self.dt_proj = tf.keras.layers.Dense(self.d_inner, use_bias=True)  # Time step
        
        # Output projection
        self.out_proj = tf.keras.layers.Dense(d_model, use_bias=False)
        
        # Dropout
        self.dropout = tf.keras.layers.Dropout(dropout)
        
        # Layer normalization
        if model_normalization in ['preLC', 'postLC']:
            self.norm = LayerCentering()
        else:
            self.norm = None
            
    def build(self, input_shape):
        # Initialize A (state transition matrix) - learnable
        # Initialize as complex conjugate pairs for stability (similar to S4)
        self.A = self.add_weight(
            name='A',
            shape=(self.d_inner, self.d_state),
            initializer=self._init_A,
            trainable=True
        )
        
        # Initialize D (skip connection) - learnable
        self.D = self.add_weight(
            name='D',
            shape=(self.d_inner,),
            initializer='ones',
            trainable=True
        )
        
    def _init_A(self, shape, dtype=None):
        """Initialize A matrix with negative values for stability."""
        d_inner, d_state = shape
        # Initialize with values similar to S4
        A = np.repeat(
            np.arange(1, d_state + 1, dtype=np.float32)[None, :],
            d_inner,
            axis=0
        )
        A = -0.5 * A  # Negative for stability
        return tf.constant(A, dtype=dtype or tf.float32)
    
    def selective_scan(self, u, delta, B, C):
        """
        Perform selective scan (core of Mamba).
        
        Args:
            u: Input (batch, seq_len, d_inner)
            delta: Discretization step (batch, seq_len, d_inner)
            B: Input matrix (batch, seq_len, d_state)
            C: Output matrix (batch, seq_len, d_state)
            
        Returns:
            y: Output (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = shape_list(u)
        
        # Discretize continuous parameters (A, B)
        # deltaA = exp(delta * A)
        deltaA = tf.exp(delta[:, :, :, None] * self.A[None, None, :, :])  # (B, L, d_inner, d_state)
        
        # deltaB_u = delta * B * u
        deltaB_u = delta[:, :, :, None] * B[:, :, None, :] * u[:, :, :, None]  # (B, L, d_inner, d_state)
        
        # Perform selective scan using cumulative product
        # This is a simplified version - full implementation would use parallel scan
        x = tf.zeros((batch, d_inner, self.d_state), dtype=u.dtype)
        outputs = []
        
        for i in range(seq_len):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = tf.einsum('bij,bj->bi', x, C[:, i])  # (B, d_inner)
            outputs.append(y)
        
        y = tf.stack(outputs, axis=1)  # (B, L, d_inner)
        return y
    
    def call(self, inputs, training=False):
        """
        Forward pass.
        
        Args:
            inputs: List [inp, p_ft, p_s] where:
                - inp: Main input (batch, seq_len, d_model)
                - p_ft: Positional features (not used in Mamba but kept for compatibility)
                - p_s: Positional scaling (not used in Mamba but kept for compatibility)
                
        Returns:
            List [output] where output has same shape as inp
        """
        if isinstance(inputs, list):
            inp = inputs[0]  # Mamba doesn't use positional features the same way
        else:
            inp = inputs
            
        residual = inp
        
        # Pre-normalization
        if self.model_normalization == 'preLC' and self.norm is not None:
            inp = self.norm(inp)
        
        # Input projection: split into x and z for gating
        xz = self.in_proj(inp)  # (B, L, 2*d_inner)
        x, z = tf.split(xz, 2, axis=-1)  # Each (B, L, d_inner)
        
        # Convolution for local context
        x = self.conv1d(x)  # (B, L, d_inner)
        x = tf.nn.silu(x)  # SiLU activation
        
        # SSM parameters
        x_db = self.x_proj(x)  # (B, L, 2*d_state)
        B, C = tf.split(x_db, 2, axis=-1)  # Each (B, L, d_state)
        
        # Compute time step (delta)
        delta = self.dt_proj(x)  # (B, L, d_inner)
        delta = tf.nn.softplus(delta)  # Ensure positive
        
        # Selective scan (SSM)
        y = self.selective_scan(x, delta, B, C)
        
        # Multiply by skip connection
        y = y + x * self.D[None, None, :]
        
        # Gating mechanism (similar to gated linear units)
        y = y * tf.nn.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        output = self.dropout(output, training=training)
        
        # Residual connection
        output = output + residual
        
        # Post-normalization
        if self.model_normalization == 'postLC' and self.norm is not None:
            output = self.norm(output)
        
        return [output]  # Return as list for compatibility with TransformerLayer interface


class PositionwiseFF(tf.keras.layers.Layer):
    """
    Positionwise Feed-Forward layer (optional, can be added after Mamba).
    """
    def __init__(self, d_model, d_inner, dropout, **kwargs):
        super().__init__(**kwargs)
        self.layer_1 = tf.keras.layers.Dense(d_inner, activation='relu')
        self.drop_1 = tf.keras.layers.Dropout(dropout)
        self.layer_2 = tf.keras.layers.Dense(d_model)
        self.drop_2 = tf.keras.layers.Dropout(dropout)

    def call(self, inp, training=False):
        x = self.drop_1(self.layer_1(inp), training=training)
        return [self.drop_2(self.layer_2(x), training=training)]


class MambaLayer(tf.keras.layers.Layer):
    """
    Complete Mamba Layer with optional feed-forward.
    This provides a closer match to TransformerLayer structure.
    """
    def __init__(self, 
                 d_model, 
                 d_state=16,
                 d_conv=4,
                 expand_factor=2,
                 d_inner_ff=None,
                 dropout=0.1,
                 model_normalization='preLC',
                 use_ff=False,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.model_normalization = model_normalization
        self.use_ff = use_ff
        
        # Mamba block
        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            dropout=dropout,
            model_normalization=model_normalization
        )
        
        # Optional feed-forward (like in Transformer)
        if use_ff:
            d_inner_ff = d_inner_ff or d_model * 4
            self.ff = PositionwiseFF(d_model, d_inner_ff, dropout)
            if model_normalization in ['preLC', 'postLC']:
                self.norm_ff = LayerCentering()
        
    def call(self, inputs, training=False):
        """
        Forward pass.
        
        Args:
            inputs: List [inp, p_ft, p_s] (for compatibility with TransformerLayer)
            
        Returns:
            List [output]
        """
        # Mamba block
        out = self.mamba(inputs, training=training)
        x = out[0]
        
        # Optional feed-forward
        if self.use_ff:
            residual = x
            if self.model_normalization == 'preLC':
                x = self.norm_ff(x)
            ff_out = self.ff(x, training=training)[0]
            x = ff_out + residual
            if self.model_normalization == 'postLC':
                x = self.norm_ff(x)
            out = [x]
        
        return out
