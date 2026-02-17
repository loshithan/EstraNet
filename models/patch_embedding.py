"""
CNN-based Patch Embedding for Multi-Scale Feature Extraction

This module implements a CNN-style patch embedding that extracts multi-scale features
from power traces. It uses parallel convolutional branches with different kernel sizes
to capture features at various temporal scales.

Key Features:
- Multi-scale feature extraction (4 different scales)
- Adaptive pooling to fixed output size
- Feature fusion with additional convolution
- Batch normalization for training stability
"""

import torch
import torch.nn as nn


class CNNPatchEmbedding(nn.Module):
    """
    CNN-style patch embedding with multi-scale feature extraction.
    
    This module processes 1D power traces using parallel convolutional branches
    with different kernel sizes (11, 25, 51, 101) to capture features at multiple
    temporal scales. All branches are pooled to the same size and concatenated
    before a final fusion layer.
    
    The multi-scale approach is inspired by Inception networks and helps capture
    both fine-grained and coarse-grained temporal patterns in side-channel traces.
    
    Args:
        d_model (int): Output feature dimension. Default: 128
    
    Shape:
        - Input: (batch_size, 1, trace_length)
        - Output: (batch_size, d_model, num_patches)
        
    Example:
        >>> embedding = CNNPatchEmbedding(d_model=192)
        >>> x = torch.randn(32, 1, 700)  # batch=32, channels=1, length=700
        >>> output = embedding(x)
        >>> print(output.shape)  # torch.Size([32, 192, 14])
    """
    
    def __init__(self, d_model=128):
        super().__init__()

        # Multi-scale feature extraction
        self.conv_layers = nn.ModuleList([
            # Scale 1: Fine-grained (kernel=11, stride=1)
            nn.Sequential(
                nn.Conv1d(1, d_model//4, kernel_size=11, stride=1, padding=5),
                nn.BatchNorm1d(d_model//4),
                nn.GELU(),
                nn.MaxPool1d(2)
            ),
            # Scale 2: Medium (kernel=25, stride=2)
            nn.Sequential(
                nn.Conv1d(1, d_model//4, kernel_size=25, stride=2, padding=12),
                nn.BatchNorm1d(d_model//4),
                nn.GELU(),
            ),
            # Scale 3: Coarse (kernel=51, stride=4)
            nn.Sequential(
                nn.Conv1d(1, d_model//4, kernel_size=51, stride=4, padding=25),
                nn.BatchNorm1d(d_model//4),
                nn.GELU(),
            ),
            # Scale 4: Very coarse (kernel=101, stride=8)
            nn.Sequential(
                nn.Conv1d(1, d_model//4, kernel_size=101, stride=8, padding=50),
                nn.BatchNorm1d(d_model//4),
                nn.GELU(),
            )
        ])

        # Adaptive pooling to fixed size (14 patches)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(14)

        # Fusion layer to combine multi-scale features
        self.fusion = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )

    def forward(self, x):
        """
        Forward pass of the patch embedding.
        
        Args:
            x (torch.Tensor): Input traces of shape (batch, 1, trace_length)
            
        Returns:
            torch.Tensor: Embedded patches of shape (batch, d_model, num_patches)
        """
        # Extract multi-scale features
        features = []
        for conv in self.conv_layers:
            feat = conv(x)
            feat = self.adaptive_pool(feat)
            features.append(feat)

        # Concatenate and fuse
        x = torch.cat(features, dim=1)
        x = self.fusion(x)

        return x
