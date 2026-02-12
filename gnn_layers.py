"""
Graph Neural Network Layers for Side-Channel Analysis
Lightweight GNN components for temporal graph construction and message passing
"""

import tensorflow as tf
import numpy as np


class GraphConvLayer(tf.keras.layers.Layer):
    """
    Graph Convolution Layer with edge-aware message passing.
    Aggregates features from neighboring nodes in the temporal graph.
    """
    
    def __init__(self, d_model, activation='relu', use_bias=True, dropout=0.1, **kwargs):
        super(GraphConvLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.activation_name = activation
        self.use_bias = use_bias
        self.dropout_rate = dropout
        
    def build(self, input_shape):
        # Weight matrix for self features
        self.w_self = self.add_weight(
            name='w_self',
            shape=(input_shape[-1], self.d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Weight matrix for neighbor features
        self.w_neighbor = self.add_weight(
            name='w_neighbor',
            shape=(input_shape[-1], self.d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.d_model,),
                initializer='zeros',
                trainable=True
            )
        
        # Layer normalization
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        
        # Activation
        if self.activation_name == 'relu':
            self.activation = tf.nn.relu
        elif self.activation_name == 'gelu':
            self.activation = tf.nn.gelu
        else:
            self.activation = lambda x: x
        
        super(GraphConvLayer, self).build(input_shape)
    
    def call(self, x, adjacency, training=False):
        """
        Args:
            x: Node features [batch, num_nodes, d_in]
            adjacency: Adjacency matrix [num_nodes, num_nodes] (same for all batches)
            training: Boolean for dropout
        Returns:
            Updated node features [batch, num_nodes, d_model]
        """
        # Self transformation
        h_self = tf.matmul(x, self.w_self)
        
        # Neighbor aggregation
        # Need to expand adjacency for batch: [1, num_nodes, num_nodes]
        adjacency_expanded = tf.expand_dims(adjacency, axis=0)
        # adjacency @ x gives sum of neighbor features
        neighbor_features = tf.matmul(adjacency_expanded, x)  # [batch, num_nodes, d_in]
        h_neighbor = tf.matmul(neighbor_features, self.w_neighbor)
        
        # Combine self and neighbor features
        h = h_self + h_neighbor
        
        if self.use_bias:
            h = h + self.bias
        
        # Activation
        h = self.activation(h)
        
        # Layer norm
        h = self.layer_norm(h)
        
        # Dropout
        h = self.dropout(h, training=training)
        
        return h


class TemporalGraphBuilder(tf.keras.layers.Layer):
    """
    Builds temporal adjacency graph from sequential features.
    Connects each node to k nearest temporal neighbors.
    """
    
    def __init__(self, k_neighbors=5, **kwargs):
        super(TemporalGraphBuilder, self).__init__(**kwargs)
        self.k = k_neighbors
        
    def build(self, input_shape):
        self.num_nodes = input_shape[1]
        # Build adjacency matrix as a non-trainable weight
        adj_numpy = self._build_adjacency_matrix()
        self.adjacency = self.add_weight(
            name='adjacency',
            shape=(self.num_nodes, self.num_nodes),
            initializer=tf.constant_initializer(adj_numpy),
            trainable=False
        )
        super(TemporalGraphBuilder, self).build(input_shape)
    
    def _build_adjacency_matrix(self):
        """
        Create k-NN temporal adjacency matrix.
        Each node connects to k/2 neighbors on each side.
        """
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        
        k_half = self.k // 2
        
        for i in range(self.num_nodes):
            # Connect to k neighbors (k/2 on each side)
            for offset in range(-k_half, k_half + 1):
                j = i + offset
                if 0 <= j < self.num_nodes and i != j:
                    adj[i, j] = 1.0
        
        # Normalize by degree (symmetrically)
        degree = np.sum(adj, axis=1)
        degree[degree == 0] = 1  # Avoid division by zero
        degree_inv_sqrt = np.power(degree, -0.5)
        
        # D^{-1/2} A D^{-1/2} (symmetric normalization)
        adj_normalized = adj * degree_inv_sqrt[:, None] * degree_inv_sqrt[None, :]
        
        return adj_normalized  # Return numpy array
    
    def call(self, x):
        """
        Args:
            x: Input features [batch, num_nodes, d_model]
        Returns:
            x: Same features (pass-through)
            adjacency: Adjacency matrix [num_nodes, num_nodes]
        """
        return x, self.adjacency


class GlobalGraphPooling(tf.keras.layers.Layer):
    """
    Pools node features to create fixed-size graph representation.
    Supports mean and max pooling.
    """
    
    def __init__(self, pooling='mean', **kwargs):
        super(GlobalGraphPooling, self).__init__(**kwargs)
        self.pooling = pooling
    
    def call(self, x):
        """
        Args:
            x: Node features [batch, num_nodes, d_model]
        Returns:
            Pooled features [batch, d_model]
        """
        if self.pooling == 'mean':
            return tf.reduce_mean(x, axis=1)
        elif self.pooling == 'max':
            return tf.reduce_max(x, axis=1)
        elif self.pooling == 'sum':
            return tf.reduce_sum(x, axis=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
    
    def get_config(self):
        config = super(GlobalGraphPooling, self).get_config()
        config.update({'pooling': self.pooling})
        return config
