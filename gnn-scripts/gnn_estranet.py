"""
GNN-based EstraNet Model for Side-Channel Attacks
Lightweight graph neural network architecture targeting <400K parameters
"""

import tensorflow as tf
from gnn_layers import GraphConvLayer, TemporalGraphBuilder, GlobalGraphPooling, AttentionPoolingLayer


class GNNEstraNet(tf.keras.Model):
    """
    GNN-based EstraNet for side-channel attacks.
    Architecture:
        1. Conv1D preprocessing (local feature extraction)
        2. Temporal graph construction
        3. GCN layers (message passing) - DEEP (4 layers)
        4. Attention pooling (weighted sum)
        5. Classifier head
    """
    
    def __init__(self,
                 n_gcn_layers=4,    # UPGRADED: Deeper (was 2)
                 d_model=128,
                 k_neighbors=15,    # UPGRADED: Wider context (was 5)
                 graph_pooling='attention', # UPGRADED: Attention (was mean)
                 d_head_softmax=16,
                 n_head_softmax=8,
                 dropout=0.1,
                 n_classes=256,
                 conv_kernel_size=3,
                 n_conv_layer=2,
                 pool_size=2,
                 beta_hat_2=100,
                 model_normalization='preLC',
                 softmax_attn=True,
                 output_attn=False,
                 **kwargs):
        
        super(GNNEstraNet, self).__init__(**kwargs)
        
        self.n_gcn_layers = n_gcn_layers
        self.d_model = d_model
        self.k_neighbors = k_neighbors
        self.graph_pooling = graph_pooling
        self.d_head_softmax = d_head_softmax
        self.n_head_softmax = n_head_softmax
        self.dropout = dropout
        self.n_classes = n_classes
        self.conv_kernel_size = conv_kernel_size
        self.n_conv_layer = n_conv_layer
        self.pool_size = pool_size
        self.beta_hat_2 = beta_hat_2
        self.model_normalization = model_normalization
        self.softmax_attn = softmax_attn
        self.output_attn = output_attn
        
        # --- Convolutional preprocessing layers ---
        self.conv_layers = []
        self.pool_layers = []
        
        for i in range(n_conv_layer):
            conv = tf.keras.layers.Conv1D(
                filters=d_model,
                kernel_size=conv_kernel_size,
                padding='same',
                activation='relu',
                name=f'conv1d_{i}'
            )
            self.conv_layers.append(conv)
            
            if i == 0:
                # First layer: downsample significantly
                pool = tf.keras.layers.AveragePooling1D(pool_size=pool_size, name=f'pool_{i}')
            else:
                # Other layers: mild downsampling
                pool = tf.keras.layers.AveragePooling1D(pool_size=2, name=f'pool_{i}')
            self.pool_layers.append(pool)
        
        # --- Graph construction ---
        self.graph_builder = TemporalGraphBuilder(k_neighbors=k_neighbors)
        
        # --- GCN layers ---
        self.gcn_layers = []
        for i in range(n_gcn_layers):
            gcn = GraphConvLayer(
                d_model=d_model,
                activation='gelu',
                dropout=dropout,
                name=f'gcn_{i}'
            )
            self.gcn_layers.append(gcn)
        
        # --- Global pooling ---
        if graph_pooling == 'attention':
             self.global_pool = AttentionPoolingLayer(name='attn_pool')
        else:
             self.global_pool = GlobalGraphPooling(pooling=graph_pooling)
        
        # --- Output head: Simple MLP ---
        # Simpler than Transformer's complex attention - keeps params low
        self.output_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.output_dropout = tf.keras.layers.Dropout(dropout)
        
        # --- Classifier ---
        self.fc = tf.keras.layers.Dense(n_classes, name='classifier')
    
    def call(self, inputs, softmax_attn_smoothing=None, training=False):
        """
        Args:
            inputs: Power traces [batch, seq_len] or [batch, seq_len, 1]
            softmax_attn_smoothing: Unused (for compatibility with training script)
            training: Boolean for dropout
        Returns:
            logits [batch, n_classes]
        """
        # Ensure input is 3D: [batch, seq_len, 1]
        if len(inputs.shape) == 2:
            x = tf.expand_dims(inputs, axis=-1)
        else:
            x = inputs
        
        # --- Convolutional preprocessing ---
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = conv(x)
            x = pool(x)
        
        # Now x is [batch, reduced_seq_len, d_model]
        
        # --- Build temporal graph ---
        x, adjacency = self.graph_builder(x)
        
        # --- GCN message passing ---
        for gcn in self.gcn_layers:
            x_new = gcn(x, adjacency, training=training)
            # Residual connection
            x = x + x_new
        
        # --- Global pooling ---
        graph_repr = self.global_pool(x)  # [batch, d_model]
        
        # --- Output head ---
        graph_repr = self.output_norm(graph_repr)
        graph_repr = self.output_dropout(graph_repr, training=training)
        
        # --- Classifier ---
        logits = self.fc(graph_repr)
        
        # Return as tuple to match Transformer's interface
        # Training script expects: logits, ... = model(...)[0]
        return (logits,)
    
    def get_config(self):
        return {
            'n_gcn_layers': self.n_gcn_layers,
            'd_model': self.d_model,
            'k_neighbors': self.k_neighbors,
            'graph_pooling': self.graph_pooling,
            'd_head_softmax': self.d_head_softmax,
            'n_head_softmax': self.n_head_softmax,
            'dropout': self.dropout,
            'n_classes': self.n_classes,
            'conv_kernel_size': self.conv_kernel_size,
            'n_conv_layer': self.n_conv_layer,
            'pool_size': self.pool_size,
            'beta_hat_2': self.beta_hat_2,
            'model_normalization': self.model_normalization,
            'softmax_attn': self.softmax_attn,
            'output_attn': self.output_attn,
        }
