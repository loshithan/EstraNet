import tensorflow as tf
from mamba import MambaLayer
from transformer import PositionalFeature, SoftmaxAttention, shape_list


class MambaNet(tf.keras.Model):
    """
    EstraNet architecture with Mamba layers instead of Transformer layers.
    
    This model replaces the self-attention mechanism with Mamba's selective
    state-space model while keeping the Conv1D preprocessing and output layers.
    
    Args:
        n_layer: Number of Mamba layers
        d_model: Model dimension
        d_state: State dimension for Mamba (default: 16)
        d_conv: Convolution kernel size in Mamba (default: 4)
        expand_factor: Expansion factor in Mamba (default: 2)
        d_inner: Dimension for optional feed-forward layers
        d_head_softmax: Head dimension for output softmax attention
        n_head_softmax: Number of heads for output softmax attention
        dropout: Dropout rate
        n_classes: Number of output classes (256 for ASCAD)
        conv_kernel_size: Kernel size for Conv1D preprocessing
        n_conv_layer: Number of Conv1D layers
        pool_size: Pooling size for average pooling
        beta_hat_2: Positional feature scaling (kept for compatibility)
        model_normalization: Type of normalization ('preLC', 'postLC', 'none')
        use_ff: Whether to use feed-forward layers after Mamba
        softmax_attn: Whether to use softmax attention at output
        output_attn: Whether to output attention probabilities
    """
    
    def __init__(self, 
                 n_layer, 
                 d_model, 
                 d_state=16,
                 d_conv=4,
                 expand_factor=2,
                 d_inner=256,
                 d_head_softmax=16, 
                 n_head_softmax=8, 
                 dropout=0.1, 
                 n_classes=256, 
                 conv_kernel_size=3, 
                 n_conv_layer=2, 
                 pool_size=20, 
                 beta_hat_2=150,
                 model_normalization='preLC',
                 use_ff=False,
                 softmax_attn=True, 
                 output_attn=False):
        super(MambaNet, self).__init__()
        
        self.n_conv_layer = n_conv_layer
        self.pool_size = pool_size
        self.softmax_attn = softmax_attn
        self.output_attn = output_attn
        
        # Conv1D preprocessing layers (same as original EstraNet)
        filters = [min(8*2**i, d_model) for i in range(n_conv_layer-1)] + [d_model]
        self.convs = [tf.keras.layers.Conv1D(filters[l], 11 if l==0 else conv_kernel_size) 
                      for l in range(n_conv_layer)]
        self.relus = [tf.keras.layers.ReLU() for _ in range(n_conv_layer)]
        self.pools = [tf.keras.layers.AveragePooling1D(pool_size, pool_size) 
                      for _ in range(n_conv_layer)]
        
        # Positional features (kept for compatibility, though Mamba doesn't use them like Transformers)
        self.pos_feature = PositionalFeature(d_model, beta_hat_2)
        
        # Mamba layers (replacing Transformer layers)
        self.mamba_layers = [
            MambaLayer(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
                d_inner_ff=d_inner,
                dropout=dropout,
                model_normalization=model_normalization,
                use_ff=use_ff
            ) for _ in range(n_layer)
        ]
        
        # Output layers
        self.out_drop = tf.keras.layers.Dropout(dropout)
        if softmax_attn:
            self.out_attn = SoftmaxAttention(d_model, n_head_softmax, d_head_softmax)
        self.fc = tf.keras.layers.Dense(n_classes)

    def call(self, inputs, softmax_attn_smoothing=1, training=False):
        """
        Forward pass.
        
        Args:
            inputs: Input traces (batch, trace_length)
            softmax_attn_smoothing: Smoothing factor for output attention
            training: Whether in training mode
            
        Returns:
            List containing [logits] or [logits, attention_scores] if output_attn=True
        """
        # Conv1D preprocessing
        x = tf.expand_dims(inputs, -1)
        for l in range(self.n_conv_layer):
            x = self.pools[l](self.relus[l](self.convs[l](x)))
        
        bsz, slen = shape_list(x)[:2]
        
        # Get positional features (for interface compatibility)
        p_ft, p_s = self.pos_feature(slen=slen, bsz=bsz)
        
        # Mamba layers
        core = x
        for layer in self.mamba_layers:
            core = layer([core, p_ft, p_s], training=training)[0]
        
        # Output processing
        core = self.out_drop(core, training=training)
        score = None
        if self.softmax_attn:
            core, score = self.out_attn(core, softmax_attn_smoothing=softmax_attn_smoothing, 
                                       training=training)
        
        # Classification
        out = self.fc(tf.reduce_mean(core, axis=1))
        
        return [out, score] if self.output_attn else [out]


# Legacy alias for compatibility
Mamba = MambaNet
