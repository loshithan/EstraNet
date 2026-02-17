"""
OPTIMIZED MAMBA-GNN MODEL FOR SIDE-CHANNEL ANALYSIS

This module implements the main OptimizedMambaGNN model that combines:
- Multi-scale CNN patch embedding for feature extraction
- Mamba blocks for temporal modeling
- Graph Attention Networks (GAT) for spatial modeling
- Attention-based pooling and classification

The model components are now organized in the `components/` package for better
code organization and reusability.

Current Status:
- Model capacity optimized for side-channel analysis
- Aggressive training with higher learning rates
- Better feature fusion with channel attention
- Data augmentation support

For component details, see:
- components/mamba_block.py - Temporal modeling
- components/gat_layer.py - Spatial/graph modeling  
- components/patch_embedding.py - Multi-scale feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import components (siblings)
from .mamba_block import OptimizedMambaBlock
from .gat_layer import EnhancedGAT
from .patch_embedding import CNNPatchEmbedding


# =============================================================================
# OPTIMIZED MAMBA-GNN MODEL
# =============================================================================

class OptimizedMambaGNN(nn.Module):
    def __init__(
        self,
        trace_length=700,
        d_model=192,  # INCREASE from 128
        mamba_layers=4,  # INCREASE back to 4
        gnn_layers=3,  # INCREASE to 3
        num_classes=256,
        k_neighbors=8,  # INCREASE
        dropout=0.15
    ):
        super().__init__()
        self.d_model = d_model
        self.num_patches = 14
        self.k_neighbors = k_neighbors
        
        # Input scaling for better numerical stability
        # ASCAD data is in range [-65, 45], scale to approximately [-6.5, 4.5]
        self.input_scale = 0.1

        print(f"Optimized Mamba-GNN:")
        print(f"  d_model: {d_model}, Mamba layers: {mamba_layers}, GNN layers: {gnn_layers}")
        print(f"  Input scaling: {self.input_scale}")


        # Strong CNN-based patch embedding
        self.patch_embed = CNNPatchEmbedding(d_model)

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        self.input_norm = nn.LayerNorm(d_model)

        # Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            OptimizedMambaBlock(d_model, dropout=dropout)
            for _ in range(mamba_layers)
        ])

        # GAT layers
        self.gnn_layers = nn.ModuleList([
            EnhancedGAT(d_model, n_heads=8, dropout=dropout)
            for _ in range(gnn_layers)
        ])

        # Channel attention for fusion
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model * 2, d_model, kernel_size=1),
            nn.Sigmoid()
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Multi-head attention pooling
        self.attn_pool_q = nn.Linear(d_model, d_model)
        self.attn_pool_k = nn.Linear(d_model, d_model)

        # Stronger classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout * 1.5),
            nn.Linear(d_model * 2, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout * 1.5),
            nn.Linear(768, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def build_knn_graph(self, features):
        features = features.detach()
        B, N, C = features.shape

        # Compute pairwise distances
        features_norm = F.normalize(features, p=2, dim=-1)
        similarity = torch.bmm(features_norm, features_norm.transpose(1, 2))

        # Top-k + self-connection
        topk_val, topk_idx = similarity.topk(min(self.k_neighbors + 1, N), dim=-1)

        # Create weighted adjacency
        adj_matrix = torch.zeros(B, N, N, device=features.device)
        for b in range(B):
            for i in range(N):
                adj_matrix[b, i, topk_idx[b, i]] = topk_val[b, i]

        return adj_matrix

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Apply input scaling for numerical stability
        x = x * self.input_scale

        # Strong patch embedding
        patches = self.patch_embed(x).transpose(1, 2)
        patches = patches + self.pos_encoding
        patches = self.input_norm(patches)

        # Temporal modeling
        h_temp = patches
        for mamba in self.mamba_blocks:
            h_temp = mamba(h_temp)

        # Build weighted graph
        adj_matrix = self.build_knn_graph(h_temp)

        # Spatial modeling
        h_graph = h_temp
        for gat in self.gnn_layers:
            h_graph = gat(h_graph, adj_matrix)

        # Channel attention fusion
        combined = torch.cat([h_temp, h_graph], dim=-1)
        channel_weights = self.channel_attn(combined.transpose(1, 2)).transpose(1, 2)
        h_fused = self.fusion(combined) * channel_weights

        # Attention pooling
        q = self.attn_pool_q(h_fused.mean(dim=1, keepdim=True))
        k = self.attn_pool_k(h_fused)
        attn_weights = F.softmax(torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.d_model), dim=-1)
        h_global = torch.bmm(attn_weights, h_fused).squeeze(1)

        # Classification
        logits = self.classifier(h_global)

        return logits
