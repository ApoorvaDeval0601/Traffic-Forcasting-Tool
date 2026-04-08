"""
LSTM-based temporal encoder — baseline for comparison with Transformer.

Used in the ablation study to show where Transformer attention wins.
"""

import torch
import torch.nn as nn
from einops import rearrange


class LSTMTemporalEncoder(nn.Module):
    """
    Bidirectional LSTM over the time dimension.

    Input:  (B, N, T, hidden_dim)
    Output: (B, N, T, hidden_dim)
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,  # Causal for real-time use
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Project back if bidirectional
        if bidirectional:
            self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.proj = nn.Identity()

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, N, T, hidden_dim)
        Returns:
            out: (B, N, T, hidden_dim)
        """
        B, N, T, D = x.shape

        # Per-node LSTM
        x = rearrange(x, "b n t d -> (b n) t d")
        out, _ = self.lstm(x)         # (B*N, T, D or 2D)
        out = self.proj(out)          # (B*N, T, D)
        out = self.layer_norm(out)
        out = rearrange(out, "(b n) t d -> b n t d", b=B, n=N)
        return out


class STGNNWithLSTM(nn.Module):
    """
    ST-GNN variant using LSTM temporal encoder instead of Transformer.
    Drop-in replacement for ablation study.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        horizon: int = 12,
        num_gat_layers: int = 2,
        gat_heads: int = 8,
        lstm_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        from models.gat_spatial import GATSpatialEncoder
        from models.transformer_temporal import MultiStepDecoder

        self.spatial = GATSpatialEncoder(
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_heads=gat_heads,
            num_layers=num_gat_layers,
            dropout=dropout,
        )
        self.temporal = LSTMTemporalEncoder(
            hidden_dim=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout,
        )
        self.decoder = MultiStepDecoder(
            hidden_dim=hidden_dim,
            horizon=horizon,
            out_features=1,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          (B, N, T, F)
            edge_index: (2, E)
        Returns:
            preds:      (B, N, H, 1)
        """
        spatial_out = self.spatial(x, edge_index)   # (B, N, T, D)
        temporal_out = self.temporal(spatial_out)   # (B, N, T, D)
        preds = self.decoder(temporal_out)          # (B, N, H, 1)
        return preds
