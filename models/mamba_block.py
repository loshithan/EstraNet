"""
Optimized Mamba Block for Temporal Modeling

This module implements the OptimizedMambaBlock, which provides stronger temporal
modeling capabilities for side-channel analysis. The block uses depthwise convolutions
with group  normalization and pointwise convolutions for efficient feature mixing.

Key Features:
- Depthwise separable convolutions for efficiency
- Batch normalization for stable training
- Residual connections with learnable scaling
- GELU activation for smooth gradients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedMambaBlock(nn.Module):
    """
    Optimized Mamba Block for temporal sequence modeling.
    
    This block processes temporal sequences using a combination of:
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
        dropout (float): Dropout probability. Default: 0.15
    
    Shape:
        - Input: (batch_size, sequence_length, d_model)
        - Output: (batch_size, sequence_length, d_model)
    
    Example:
        >>> block = OptimizedMambaBlock(d_model=192, d_conv=7, expand=2)
        >>> x = torch.randn(32, 14, 192)  # batch=32, seq_len=14, features=192
        >>> output = block(x)
        >>> print(output.shape)  # torch.Size([32, 14, 192])
    """
    
    def __init__(self, d_model, d_conv=7, expand=2, dropout=0.15):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand

        self.norm1 = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Stronger convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv-1,
            groups=self.d_inner,
            bias=False
        )
        self.conv_norm = nn.GroupNorm(1, self.d_inner)  # was BatchNorm1d

        # Add pointwise convolution for better feature mixing
        self.pointwise = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=1)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x):
        """
        Forward pass of the Mamba block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm1(x)

        xz = self.in_proj(x)
        x_input, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_input.transpose(1, 2))[:, :, :x.size(1)]
        x_conv = self.conv_norm(x_conv)
        x_conv = self.pointwise(x_conv)  # Add pointwise
        x_conv = F.gelu(x_conv).transpose(1, 2)

        y = x_conv * torch.sigmoid(z)
        output = self.out_proj(y)
        output = self.dropout(output)

        return residual + self.gamma * output
