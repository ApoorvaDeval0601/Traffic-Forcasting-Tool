"""
Transformer-based temporal encoder for traffic sequences.

Uses sinusoidal + learnable positional encoding. Captures long-range
temporal dependencies (morning rush at t-72 influences noon predictions).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal PE (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TrafficTemporalTransformer(nn.Module):
    """
    Transformer encoder over the time dimension.

    Input:  (B, N, T, hidden_dim)
    Output: (B, N, T, hidden_dim)

    Each node's time series is processed independently through
    the same Transformer, sharing weights across nodes.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 3,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.pos_encoding = SinusoidalPositionalEncoding(
            hidden_dim, max_len=max_seq_len, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,     # (B, T, D) convention
            norm_first=True,      # Pre-LN: more stable for traffic data
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_attention: bool = False,
    ):
        """
        Args:
            x:     (B, N, T, hidden_dim)
            mask:  optional causal mask (T, T)
            return_attention: store attention weights

        Returns:
            out:   (B, N, T, hidden_dim)
        """
        B, N, T, D = x.shape

        # Reshape: treat each node independently → (B*N, T, D)
        x = rearrange(x, "b n t d -> (b n) t d")

        # Positional encoding
        x = self.pos_encoding(x)

        # Transformer encoding
        out = self.transformer(x, mask=mask)  # (B*N, T, D)

        # Reshape back
        out = rearrange(out, "(b n) t d -> b n t d", b=B, n=N)
        return out


class MultiStepDecoder(nn.Module):
    """
    Decode the encoded representation into H future time steps.

    Input:  (B, N, T, hidden_dim) — encoded history
    Output: (B, N, H, out_features) — predicted future speeds
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        horizon: int = 12,  # 12 steps × 5 min = 1 hour
        out_features: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.out_features = out_features

        # Project last time step's features to H predictions
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, horizon * out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  (B, N, T, hidden_dim)
        Returns:
            (B, N, H, out_features)
        """
        # Use the last time step as summary
        last = x[:, :, -1, :]  # (B, N, hidden_dim)
        out = self.head(last)   # (B, N, H * out_features)
        out = out.view(*last.shape[:2], self.horizon, self.out_features)
        return out
