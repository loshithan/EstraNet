"""
TensorFlow implementation of Mamba-GNN Model for Side-Channel Analysis

This module provides a TensorFlow/Keras version of the Mamba-GNN architecture,
enabling training with EstraNet scripts and conversion to TFLite for deployment.

Architecture:
    Input → CNN Patch Embedding → Mamba Blocks → GNN Layers → Attention Pooling → Classification

Key Features:
- CNN-based patch embedding for trace segmentation
- Stacked Mamba blocks for temporal modeling
- Graph Attention Networks (GAT) for spatial modeling
- Channel attention fusion
- Attention-based pooling
- Compatible with EstraNet training pipeline
- Ready for TFLite conversion
"""

import tensorflow as tf
import numpy as np
from mamba_block_tf import OptimizedMambaBlockTF


class PatchEmbedding(tf.keras.layers.Layer):
    """
    CNN-based patch embedding for trace segmentation.
    
    Divides input trace into patches and embeds each patch using CNN.
    
    Args:
        d_model (int): Output embedding dimension
        patch_size (int): Size of each patch. Default: 50
    """
    
    def __init__(self, d_model, patch_size=50, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.patch_size = patch_size
        
        # CNN layers for patch embedding
        self.conv1 = tf.keras.layers.Conv1D(
            filters=d_model // 2,
            kernel_size=3,
            strides=patch_size,
            padding='valid',
            activation='relu',
            name='patch_conv1'
        )
        self.conv2 = tf.keras.layers.Conv1D(
            filters=d_model,
            kernel_size=1,
            activation='relu',
            name='patch_conv2'
        )
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
    
    def call(self, x, training=False):
        """
        Args:
            x (tf.Tensor): Input trace (batch, trace_length, 1)
        
        Returns:
            tf.Tensor: Patch embeddings (batch, num_patches, d_model)
        """
        x = self.conv1(x)  # (batch, num_patches, d_model//2)
        x = self.conv2(x)  # (batch, num_patches, d_model)
        x = self.norm(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'patch_size': self.patch_size,
        })
        return config


class GATLayer(tf.keras.layers.Layer):
    """
    Graph Attention Network (GAT) layer for spatial modeling.
    
    Args:
        d_model (int): Feature dimension
        num_heads (int): Number of attention heads. Default: 4
        dropout (float): Dropout rate. Default: 0.1
    """
    
    def __init__(self, d_model, num_heads=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.dropout_rate = dropout
        
        # Multi-head projections
        self.q_proj = tf.keras.layers.Dense(d_model, use_bias=False, name='q_proj')
        self.k_proj = tf.keras.layers.Dense(d_model, use_bias=False, name='k_proj')
        self.v_proj = tf.keras.layers.Dense(d_model, use_bias=False, name='v_proj')
        
        # Output projection
        self.out_proj = tf.keras.layers.Dense(d_model, name='out_proj')
        
        # Dropout
        self.dropout = tf.keras.layers.Dropout(dropout)
        
        # Layer norm
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
    
    def build_knn_graph(self, features, k=8):
        """
        Build k-NN graph adjacency matrix.
        
        Args:
            features (tf.Tensor): Node features (batch, num_nodes, d_model)
            k (int): Number of nearest neighbors
        
        Returns:
            tf.Tensor: Adjacency matrix (batch, num_nodes, num_nodes)
        """
        # Compute pairwise distances
        features_norm = tf.nn.l2_normalize(features, axis=-1)
        similarity = tf.matmul(features_norm, features_norm, transpose_b=True)
        
        # Get top-k neighbors
        _, indices = tf.nn.top_k(similarity, k=k+1)  # +1 to exclude self
        
        # Create adjacency matrix
        batch_size = tf.shape(features)[0]
        num_nodes = tf.shape(features)[1]
        
        # Create sparse adjacency
        batch_indices = tf.range(batch_size)[:, None, None]
        batch_indices = tf.tile(batch_indices, [1, num_nodes, k+1])
        
        row_indices = tf.range(num_nodes)[None, :, None]
        row_indices = tf.tile(row_indices, [batch_size, 1, k+1])
        
        sparse_indices = tf.stack([
            tf.reshape(batch_indices, [-1]),
            tf.reshape(row_indices, [-1]),
            tf.reshape(indices, [-1])
        ], axis=1)
        
        sparse_values = tf.ones([tf.shape(sparse_indices)[0]], dtype=tf.float32)
        
        adj_matrix = tf.scatter_nd(
            sparse_indices,
            sparse_values,
            [batch_size, num_nodes, num_nodes]
        )
        
        # Make symmetric
        adj_matrix = tf.maximum(adj_matrix, tf.transpose(adj_matrix, [0, 2, 1]))
        
        return adj_matrix
    
    def call(self, x, adj_matrix=None, k_neighbors=8, training=False):
        """
        Forward pass.
        
        Args:
            x (tf.Tensor): Input features (batch, num_nodes, d_model)
            adj_matrix (tf.Tensor, optional): Adjacency matrix
            k_neighbors (int): Number of neighbors for k-NN graph
            training (bool): Training mode
        
        Returns:
            tf.Tensor: Output features (batch, num_nodes, d_model)
        """
        residual = x
        
        # Build k-NN graph if adjacency not provided
        if adj_matrix is None:
            adj_matrix = self.build_knn_graph(x, k=k_neighbors)
        
        batch_size = tf.shape(x)[0]
        num_nodes = tf.shape(x)[1]
        
        # Multi-head projections
        q = self.q_proj(x)  # (batch, num_nodes, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = tf.reshape(q, [batch_size, num_nodes, self.num_heads, self.d_head])
        k = tf.reshape(k, [batch_size, num_nodes, self.num_heads, self.d_head])
        v = tf.reshape(v, [batch_size, num_nodes, self.num_heads, self.d_head])
        
        # Transpose for batch matrix multiplication
        q = tf.transpose(q, [0, 2, 1, 3])  # (batch, heads, num_nodes, d_head)
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])
        
        # Attention scores
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(float(self.d_head))
        
        # Apply graph structure (mask non-neighbors)
        adj_mask = adj_matrix[:, None, :, :]  # (batch, 1, num_nodes, num_nodes)
        scores = tf.where(adj_mask > 0, scores, -1e9)
        
        # Attention weights
        attn_weights = tf.nn.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)
        
        # Aggregate
        out = tf.matmul(attn_weights, v)  # (batch, heads, num_nodes, d_head)
        out = tf.transpose(out, [0, 2, 1, 3])  # (batch, num_nodes, heads, d_head)
        out = tf.reshape(out, [batch_size, num_nodes, self.d_model])
        
        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out, training=training)
        
        # Residual connection and normalization
        out = self.norm(residual + out)
        
        return out
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout': self.dropout_rate,
        })
        return config


class OptimizedMambaGNNTF(tf.keras.Model):
    """
    TensorFlow implementation of Optimized Mamba-GNN for side-channel analysis.
    
    Architecture matches EstraNet configuration for fair comparison.
    
    Args:
        trace_length (int): Input trace length. Default: 700
        d_model (int): Model dimension. Default: 128
        mamba_layers (int): Number of Mamba blocks. Default: 4
        gnn_layers (int): Number of GNN layers. Default: 3
        num_classes (int): Number of output classes. Default: 256
        k_neighbors (int): K for k-NN graph. Default: 8
        dropout (float): Dropout rate. Default: 0.1
        patch_size (int): Patch size for embedding. Default: 50
    """
    
    def __init__(self,
                 trace_length=700,
                 d_model=128,
                 mamba_layers=4,
                 gnn_layers=3,
                 num_classes=256,
                 k_neighbors=8,
                 dropout=0.1,
                 patch_size=50,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.trace_length = trace_length
        self.d_model = d_model
        self.mamba_layers = mamba_layers
        self.gnn_layers = gnn_layers
        self.num_classes = num_classes
        self.k_neighbors = k_neighbors
        self.dropout_rate = dropout
        self.patch_size = patch_size
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(d_model, patch_size=patch_size)
        
        # Mamba blocks for temporal modeling
        self.mamba_blocks = [
            OptimizedMambaBlockTF(
                d_model=d_model,
                d_conv=7,
                expand=2,
                dropout=dropout,
                name=f'mamba_block_{i}'
            )
            for i in range(mamba_layers)
        ]
        
        # GNN layers for spatial modeling
        self.gnn_layers_list = [
            GATLayer(
                d_model=d_model,
                num_heads=4,
                dropout=dropout,
                name=f'gat_layer_{i}'
            )
            for i in range(gnn_layers)
        ]
        
        # Channel attention fusion
        self.fusion = tf.keras.layers.Dense(d_model, name='fusion')
        self.channel_attn = tf.keras.layers.Dense(d_model, activation='sigmoid', name='channel_attn')
        
        # Attention pooling
        self.attn_pool_q = tf.keras.layers.Dense(d_model, name='attn_pool_q')
        self.attn_pool_k = tf.keras.layers.Dense(d_model, name='attn_pool_k')
        
        # Classification head
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 2, activation='relu', name='classifier_hidden'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(num_classes, name='classifier_output')
        ], name='classifier')
    
    def call(self, x, training=False):
        """
        Forward pass.
        
        Args:
            x (tf.Tensor): Input trace (batch, trace_length) or (batch, trace_length, 1)
        
        Returns:
            tf.Tensor: Logits (batch, num_classes)
        """
        # Ensure input has channel dimension
        if len(x.shape) == 2:
            x = x[:, :, None]  # (batch, trace_length, 1)
        
        # Patch embedding
        h = self.patch_embed(x, training=training)  # (batch, num_patches, d_model)
        
        # Temporal modeling with Mamba blocks
        h_temp = h
        for mamba in self.mamba_blocks:
            h_temp = mamba(h_temp, training=training)
        
        # Build k-NN graph
        adj_matrix = self.gnn_layers_list[0].build_knn_graph(h_temp, k=self.k_neighbors)
        
        # Spatial modeling with GNN
        h_graph = h_temp
        for gnn in self.gnn_layers_list:
            h_graph = gnn(h_graph, adj_matrix=adj_matrix, training=training)
        
        # Channel attention fusion
        combined = tf.concat([h_temp, h_graph], axis=-1)  # (batch, num_patches, d_model*2)
        channel_weights = self.channel_attn(combined)
        h_fused = self.fusion(combined) * channel_weights
        
        # Attention pooling
        q = self.attn_pool_q(tf.reduce_mean(h_fused, axis=1, keepdims=True))  # (batch, 1, d_model)
        k = self.attn_pool_k(h_fused)  # (batch, num_patches, d_model)
        
        attn_weights = tf.nn.softmax(
            tf.matmul(q, k, transpose_b=True) / tf.sqrt(float(self.d_model)),
            axis=-1
        )  # (batch, 1, num_patches)
        
        h_global = tf.squeeze(tf.matmul(attn_weights, h_fused), axis=1)  # (batch, d_model)
        
        # Classification
        logits = self.classifier(h_global, training=training)  # (batch, num_classes)
        
        return logits
    
    def get_config(self):
        return {
            'trace_length': self.trace_length,
            'd_model': self.d_model,
            'mamba_layers': self.mamba_layers,
            'gnn_layers': self.gnn_layers,
            'num_classes': self.num_classes,
            'k_neighbors': self.k_neighbors,
            'dropout': self.dropout_rate,
            'patch_size': self.patch_size,
        }


# Example usage and model creation
if __name__ == '__main__':
    print("Testing OptimizedMambaGNNTF...")
    
    # Create model with EstraNet-aligned configuration
    model = OptimizedMambaGNNTF(
        trace_length=700,
        d_model=128,
        mamba_layers=4,
        gnn_layers=3,
        num_classes=256,
        k_neighbors=8,
        dropout=0.1,
        patch_size=50
    )
    
    # Test input
    x = tf.random.normal([4, 700])  # batch=4, trace_length=700
    
    print(f"Input shape: {x.shape}")
    output = model(x, training=True)
    print(f"Output shape: {output.shape}")
    print(f"✓ Model test passed")
    
    # Model summary
    model.build(input_shape=(None, 700))
    print("\nModel Summary:")
    model.summary()
    
    # Count parameters
    total_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"\nTotal trainable parameters: {total_params:,}")
    
    print("\n✓ Model ready for training with EstraNet scripts")
    print("✓ Model ready for TFLite conversion")
