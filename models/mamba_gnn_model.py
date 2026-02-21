"""
OPTIMIZED MAMBA-GNN MODEL FOR SIDE-CHANNEL ANALYSIS

Combines:
- Multi-scale CNN patch embedding for feature extraction
- Mamba blocks for temporal modeling
- Graph Attention Networks (GAT) for spatial modeling
- Attention-based pooling and classification

Fixes applied vs previous version:
  1. input_scale removed entirely — data is pre-normalised by StandardScaler
     in the training script (mean≈0, std≈1), so scaling by 0.1 was squashing
     the signal to std≈0.1 and preventing learning.
  2. The leftover  print(f"Input scaling: {self.input_scale}")  that crashed
     on __init__ has been removed.
  3. Default hyper-parameters corrected to match the small-model config
     (d_model=64, mamba_layers=2, gnn_layers=2, dropout=0.15).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .mamba_block import OptimizedMambaBlock, SelectiveMambaBlock
from .gat_layer import EnhancedGAT
from .patch_embedding import CNNPatchEmbedding


class OptimizedMambaGNN(nn.Module):
    def __init__(
        self,
        trace_length=700,
        d_model=64,        # corrected default (was 192/128)
        mamba_layers=2,    # corrected default (was 4)
        gnn_layers=2,      # corrected default (was 3)
        num_classes=256,
        k_neighbors=8,
        dropout=0.15,      # corrected default (was 0.3)
        use_patch_embed=True,  # False = per-step linear projection over all 700 samples
        use_ssm_mamba=False,   # True = real S6 selective SSM (O(n)); False = legacy gated CNN
        ssm_d_state=8          # SSM hidden state dim: 8=fast (0.73 GB/block), 16=slow (1.47 GB/block)
    ):
        super().__init__()
        self.d_model          = d_model
        self.k_neighbors      = k_neighbors
        self.use_patch_embed  = use_patch_embed
        self.use_ssm_mamba    = use_ssm_mamba

        block_type = 'selective-SSM/O(n)' if use_ssm_mamba else 'gated-CNN'
        print(f"Optimized Mamba-GNN:")
        print(f"  d_model: {d_model}, Mamba layers: {mamba_layers}, "
              f"GNN layers: {gnn_layers}, "
              f"embed={'patch14' if use_patch_embed else 'linear700'}, "
              f"block={block_type}")

        if use_patch_embed:
            # CNN multi-scale embedding → 14 patches
            self.patch_embed = CNNPatchEmbedding(d_model)
            self.num_patches = 14
        else:
            # Per-step linear projection: each of the 700 raw samples → d_model
            # This preserves full temporal resolution so Mamba can find the
            # narrow leakage window that the 14-patch CNN was averaging away.
            self.step_embed  = nn.Linear(1, d_model)
            self.num_patches = trace_length

        # Learnable positional encoding (size set above based on embed mode)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02
        )
        self.input_norm   = nn.LayerNorm(d_model)

        # Temporal modeling — Real S6 Selective SSM (O(n)) or legacy gated CNN
        if use_ssm_mamba:
            # d_state controls hidden state dimension; scan tensor size = [B,L,d_inner,d_state].
            # d_state=8  → 0.73 GB/block  (~2x faster than 16)
            # d_state=16 → 1.47 GB/block  (original, slow due to ~120 GB memcpy/step)
            self.mamba_blocks = nn.ModuleList([
                SelectiveMambaBlock(d_model, d_state=ssm_d_state, dropout=dropout)
                for _ in range(mamba_layers)
            ])
        else:
            self.mamba_blocks = nn.ModuleList([
                OptimizedMambaBlock(d_model, dropout=dropout)
                for _ in range(mamba_layers)
            ])

        # Spatial modeling — GAT layers
        self.gnn_layers = nn.ModuleList([
            EnhancedGAT(d_model, n_heads=8, dropout=dropout)
            for _ in range(gnn_layers)
        ])

        # Channel attention for Mamba + GNN fusion
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model * 2, d_model, kernel_size=1),
            nn.Sigmoid()
        )

        # Fusion projection
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Attention pooling (patches → single vector)
        self.attn_pool_q = nn.Linear(d_model, d_model)
        self.attn_pool_k = nn.Linear(d_model, d_model)

        # Classifier head
        # Hidden size capped at d_model*4 (=256 for d_model=64) so the head
        # does not dwarf the feature extractor for small d_model values.
        hidden = min(d_model * 4, 256)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

        self._init_weights()

    # -------------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )

    # -------------------------------------------------------------------------

    def build_knn_graph(self, features):
        """Build a k-NN weighted adjacency matrix.

        Args:
            features: Tensor [B, N, C]
        Returns:
            adj_matrix: Tensor [B, N, N]  (no gradient — topology only)
        """
        if features.dim() != 3:
            raise ValueError("features must have shape [B, N, C]")
        B, N, C = features.size()

        with torch.no_grad():
            # Cosine similarity
            feat_norm  = F.normalize(features, p=2, dim=-1)          # [B,N,C]
            similarity = torch.bmm(feat_norm, feat_norm.transpose(1, 2))  # [B,N,N]

            k = min(self.k_neighbors + 1, N)
            topk_val, topk_idx = similarity.topk(k, dim=-1)          # [B,N,k]

            # Vectorised scatter — no Python loops
            adj = torch.zeros(B, N, N,
                              device=features.device, dtype=features.dtype)
            b_idx = torch.arange(B, device=features.device)[:, None, None]
            r_idx = torch.arange(N, device=features.device)[None, :, None]
            adj[b_idx, r_idx, topk_idx] = topk_val

        return adj  # features retains its gradient path into GAT layers

    # -------------------------------------------------------------------------

    def forward(self, x):
        # Accept both [B, L] and [B, 1, L]
        if x.dim() == 2:
            x = x.unsqueeze(1)   # → [B, 1, L]

        # x is already normalised (mean≈0, std≈1) by StandardScaler

        # ── Embedding ─────────────────────────────────────────────────────
        if self.use_patch_embed:
            # CNN multi-scale → 14 patches  [B, d_model, 14] → [B, 14, d_model]
            patches = self.patch_embed(x).transpose(1, 2)
        else:
            # Per-step linear: [B, 1, L] → [B, L, 1] → [B, L, d_model]
            patches = self.step_embed(x.transpose(1, 2))

        patches = patches + self.pos_encoding
        patches = self.input_norm(patches)

        # ── Temporal modeling (Mamba) ─────────────────────────────────────
        # Gradient checkpointing: recompute SSM activations during backward
        # instead of storing 700 × [B, d_inner, d_state] per block (~5 GB).
        h_temp = patches
        for mamba in self.mamba_blocks:
            if self.use_ssm_mamba and self.training:
                h_temp = grad_checkpoint(mamba, h_temp, use_reentrant=False)
            else:
                h_temp = mamba(h_temp)

        # ── Graph construction ────────────────────────────────────────────
        adj_matrix = self.build_knn_graph(h_temp)

        # ── Spatial modeling (GAT) ────────────────────────────────────────
        h_graph = h_temp
        for gat in self.gnn_layers:
            h_graph = gat(h_graph, adj_matrix)

        # ── Channel-attention fusion ──────────────────────────────────────
        combined        = torch.cat([h_temp, h_graph], dim=-1)        # [B,N,2C]
        channel_weights = self.channel_attn(
            combined.transpose(1, 2)
        ).transpose(1, 2)                                              # [B,N,C]
        h_fused = self.fusion(combined) * channel_weights              # [B,N,C]

        # ── Attention pooling ─────────────────────────────────────────────
        q           = self.attn_pool_q(h_fused.mean(dim=1, keepdim=True))  # [B,1,C]
        k           = self.attn_pool_k(h_fused)                            # [B,N,C]
        attn_w      = F.softmax(
            torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.d_model), dim=-1
        )                                                                   # [B,1,N]
        h_global    = torch.bmm(attn_w, h_fused).squeeze(1)               # [B,C]

        # ── Classification ────────────────────────────────────────────────
        return self.classifier(h_global)