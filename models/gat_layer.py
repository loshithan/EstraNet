"""
Enhanced Graph Attention Network (GAT) Layer

This module implements an enhanced GAT layer with edge features for spatial modeling
in side-channel analysis. The layer uses multi-head attention with edge embeddings
to capture relationships between different time patches.

Key Features:
- Multi-head attention mechanism
- Edge feature embedding for weighted graphs
- Attention masking for sparse graphs
- Residual connections with learnable scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedGAT(nn.Module):
    """
    Enhanced Graph Attention Network layer with edge features.
    
    This layer performs graph-based spatial modeling using multi-head attention.
    It can incorporate edge weights from an adjacency matrix to guide the attention
    mechanism, making it particularly effective for k-NN graph structures.
    
    Args:
        d_model (int): Model dimension (input and output feature size)
        n_heads (int): Number of attention heads. Default: 8
        dropout (float): Dropout probability. Default: 0.15
    
    Shape:
        - Input: (batch_size, num_nodes, d_model)
        - Adjacency Matrix: (batch_size, num_nodes, num_nodes) [optional]
        - Output: (batch_size, num_nodes, d_model)
    
    Example:
        >>> gat = EnhancedGAT(d_model=192, n_heads=8)
        >>> x = torch.randn(32, 14, 192)  # batch=32, nodes=14, features=192
        >>> adj = torch.rand(32, 14, 14)  # adjacency matrix
        >>> output = gat(x, adj)
        >>> print(output.shape)  # torch.Size([32, 14, 192])
    """
    
    def __init__(self, d_model, n_heads=8, dropout=0.15):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)

        # Edge feature embedding
        self.edge_embed = nn.Linear(1, n_heads, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x, adj_matrix=None):
        """
        Forward pass of the GAT layer.
        
        Args:
            x (torch.Tensor): Input node features of shape (batch, num_nodes, d_model)
            adj_matrix (torch.Tensor, optional): Adjacency matrix of shape 
                (batch, num_nodes, num_nodes). If provided, edge weights will be
                incorporated into the attention mechanism.
            
        Returns:
            torch.Tensor: Output node features of shape (batch, num_nodes, d_model)
        """
        B, N, C = x.shape
        residual = x
        x = self.norm(x)

        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)

        if adj_matrix is not None:
            edge_weights = self.edge_embed(adj_matrix.unsqueeze(-1)).permute(0, 3, 1, 2)
            attn = attn + edge_weights
            mask = (adj_matrix == 0).unsqueeze(1)
            attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        return residual + self.gamma * out
