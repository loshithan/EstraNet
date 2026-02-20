"""
Mamba Blocks for Temporal Modeling

Two implementations:

1. SelectiveMambaBlock  — the REAL Mamba (S6) algorithm.
   - Selective State Space Model: A, B, C, Δ are input-dependent
   - Complexity: O(n · d_state) — LINEAR in sequence length  ← research novelty
   - Receptive field: GLOBAL — hidden state h(t) accumulates all x(0)...x(t-1)
   - vs Transformer self-attention: O(n²) — quadratic

2. OptimizedMambaBlock  — legacy gated depthwise-CNN (kept for ablation/compat).
   - NOT a real SSM; local receptive field ≈ kernel × layers

Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
           Gu & Dao, 2023. https://arxiv.org/abs/2312.00752
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelectiveMambaBlock(nn.Module):
    """
    Real Mamba block: S6 (Selective State Space Model).

    Architecture per token x_t ∈ ℝ^d_model:

        [x_ssm, z] = in_proj(LayerNorm(x))      # split into SSM + gate branches
        x_ssm  = SiLU(conv1d(x_ssm))            # short causal conv (local smooth)
        Δ, B, C = x_proj(x_ssm)                 # ALL selective (input-dependent)
        Δ       = softplus(dt_proj(Δ))           # positive time-step size

        # Discretise continuous A, B with ZOH:
        Ā(t) = exp(Δ(t) · A)                    # [B, L, d_inner, d_state]
        B̄(t) = Δ(t) · B(t) · x_ssm(t)

        # Recurrent scan — O(n · d_state), global receptive field:
        h(t) = Ā(t) · h(t-1) + B̄(t)
        y(t) = C(t) · h(t) + D · x_ssm(t)

        output = out_proj(y ⊙ SiLU(z))
        return  residual + dropout(output)

    Complexity vs Transformer:
        Transformer self-attention: O(n² · d)       ← quadratic
        This SSM scan:              O(n · d_inner · d_state)  ← LINEAR  ✓

    Args:
        d_model (int):   Input/output feature dimension.
        d_state (int):   SSM state size N. Default: 16
        d_conv  (int):   Short causal conv kernel size. Default: 4
        expand  (int):   d_inner = expand × d_model. Default: 2
        dt_rank       :  Rank of Δ projection. 'auto' → ceil(d_model/16).
        dt_min (float):  Min initial Δ.  Default: 0.001
        dt_max (float):  Max initial Δ.  Default: 0.1
        dropout(float):  Output dropout probability.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                 dt_rank='auto', dt_min=0.001, dt_max=0.1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == 'auto' else dt_rank

        self.norm    = nn.LayerNorm(d_model)

        # in_proj: d_model → 2 × d_inner  (SSM branch + gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Short causal depthwise conv for local neighbourhood context
        self.conv1d  = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        # x_proj: d_inner → (dt_rank + d_state + d_state)  [Δ, B, C]
        self.x_proj  = nn.Linear(self.d_inner,
                                  self.dt_rank + d_state * 2, bias=False)

        # dt_proj: dt_rank → d_inner  (expand low-rank Δ to full width)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialise dt_proj bias → Δ starts uniformly in [dt_min, dt_max]
        dt_init = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        # softplus⁻¹ so that softplus(bias) = dt_init
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        self.dt_proj.bias = nn.Parameter(inv_dt)
        self.dt_proj.bias._no_weight_decay = True

        # A: log-diagonal [d_inner, d_state], fixed structure, negative after exp
        A = (torch.arange(1, d_state + 1, dtype=torch.float32)
                  .unsqueeze(0).expand(self.d_inner, -1))
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D: per-channel skip connection weight
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout  = nn.Dropout(dropout)

    # ── Selective scan ────────────────────────────────────────────────────────

    def _selective_scan(self, u, delta, A, B, C, D):
        """
        Stable per-step selective scan.

        Runs a Python loop over L time steps. Each step allocates only
        [B, d_inner, d_state] (~2 MB), keeping peak memory small regardless
        of sequence length when combined with gradient checkpointing.

        Speed: when torch.compile() wraps SelectiveMambaBlock (activated in
        OptimizedMambaGNN if torch.compile is available), the compiler fuses
        this loop into a Triton kernel, giving speed close to the native CUDA
        mamba-ssm kernel without extra packages.

        Args:
            u     : [B, L, d_inner]
            delta : [B, L, d_inner]   positive Δ
            A     : [d_inner, d_state] negative diagonal
            B     : [B, L, d_state]
            C     : [B, L, d_state]
            D     : [d_inner]
        Returns:
            y     : [B, L, d_inner]
        """
        B_b, L, d_inner = u.shape
        h  = u.new_zeros(B_b, d_inner, self.d_state)
        ys = []
        for t in range(L):
            dt_t  = delta[:, t]                                  # [B, di]
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A[None])     # [B, di, ds]
            dBu   = (dt_t.unsqueeze(-1)                         # [B, di,  1]
                     * B[:, t, None, :]                         # [B,  1, ds]
                     * u[:, t, :, None])                        # [B, di, ds]
            h  = A_bar * h + dBu
            y  = (h * C[:, t, None, :]).sum(-1)                 # [B, di]
            ys.append(y)
        y = torch.stack(ys, dim=1)                               # [B, L, di]
        y = y + u * D[None, None, :]
        return y

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, x):
        """
        Args:
            x: [B, L, d_model]
        Returns:
            [B, L, d_model]
        """
        residual = x
        x = self.norm(x)                               # pre-norm

        # ── Input projection ──────────────────────────────────────────────
        xz    = self.in_proj(x)                        # [B, L, 2·d_inner]
        x_ssm, z = xz.chunk(2, dim=-1)                # each [B, L, d_inner]

        # ── Local causal conv ─────────────────────────────────────────────
        x_conv = self.conv1d(
            x_ssm.transpose(1, 2)                      # [B, d_inner, L]
        )[:, :, :x.size(1)]                            # causal trim
        x_conv = F.silu(x_conv).transpose(1, 2)        # [B, L, d_inner]

        # ── Selective parameters (all depend on x_conv at each position) ──
        x_dbl  = self.x_proj(x_conv)                   # [B, L, dt_rank+2·d_state]
        delta, B_mat, C_mat = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta  = F.softplus(self.dt_proj(delta))       # [B, L, d_inner] > 0

        A = -torch.exp(self.A_log)                     # [d_inner, d_state] < 0

        # ── Selective scan — O(n), global receptive field ─────────────────
        y = self._selective_scan(x_conv, delta, A, B_mat, C_mat, self.D)

        # ── Gated output ──────────────────────────────────────────────────
        y = y * F.silu(z)
        return residual + self.dropout(self.out_proj(y))


# =============================================================================
# Legacy block — kept for backward compatibility with old checkpoints
# =============================================================================

class OptimizedMambaBlock(nn.Module):
    """
    Legacy gated depthwise-CNN block. NOT a real SSM.

    ⚠  Effective receptive field ≈ d_conv × num_layers (e.g. 7×4 = 28 samples).
       Cannot learn associations between distant time steps.
       Use SelectiveMambaBlock for correct Mamba (S6) behaviour.

    Kept for backward compatibility with checkpoints trained with this block.
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
