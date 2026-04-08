"""
Full ST-GNN: GAT spatial encoder + Transformer temporal encoder + decoder.

Architecture:
  Input (B, N, T, F)
    ↓
  GAT Spatial Encoder  → captures road graph dependencies
    ↓
  Transformer Temporal → captures time-series patterns
    ↓
  Multi-Step Decoder   → predicts H future steps
    ↓
  Output (B, N, H, 1)  — predicted speed (mph)
"""

import torch
import torch.nn as nn

from models.gat_spatial import GATSpatialEncoder
from models.transformer_temporal import TrafficTemporalTransformer, MultiStepDecoder


class STGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network for traffic speed forecasting.

    Key design choices:
    - GAT encoder: adaptive, learned attention over road graph
    - Transformer temporal: parallel, long-range temporal attention
    - Multi-step decoder: simultaneous H-step prediction (not autoregressive)
    """

    def __init__(
        self,
        in_features: int = 1,          # Speed only (or + time features)
        hidden_dim: int = 64,
        horizon: int = 12,             # 12 × 5min = 1 hour ahead
        # GAT params
        gat_heads: int = 8,
        gat_layers: int = 2,
        # Transformer params
        transformer_heads: int = 8,
        transformer_layers: int = 3,
        transformer_ff_dim: int = 256,
        # Shared
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.horizon = horizon

        self.spatial_encoder = GATSpatialEncoder(
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_heads=gat_heads,
            num_layers=gat_layers,
            dropout=dropout,
        )

        self.temporal_encoder = TrafficTemporalTransformer(
            hidden_dim=hidden_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            ff_dim=transformer_ff_dim,
            dropout=dropout,
        )

        self.decoder = MultiStepDecoder(
            hidden_dim=hidden_dim,
            horizon=horizon,
            out_features=1,
            dropout=dropout,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Args:
            x:               (B, N, T, F) — input window
            edge_index:      (2, E) — graph edges
            return_attention: return GAT attention weights

        Returns:
            preds:           (B, N, H, 1) — speed predictions
            spatial_attn:    list of attention tensors (if return_attention)
        """
        if return_attention:
            spatial_out, spatial_attn = self.spatial_encoder(
                x, edge_index, return_attention=True
            )
        else:
            spatial_out = self.spatial_encoder(x, edge_index)
            spatial_attn = None

        temporal_out = self.temporal_encoder(spatial_out)
        preds = self.decoder(temporal_out)

        if return_attention:
            return preds, spatial_attn
        return preds

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        params = self.count_parameters()
        return (
            f"STGNN(\n"
            f"  in_features={self.in_features}, hidden_dim={self.hidden_dim},\n"
            f"  horizon={self.horizon}\n"
            f"  spatial=GATSpatialEncoder, temporal=TransformerEncoder\n"
            f"  parameters={params:,}\n"
            f")"
        )


def build_model(config: dict) -> STGNN:
    """Factory function from config dict."""
    return STGNN(
        in_features=config.get("in_features", 1),
        hidden_dim=config.get("hidden_dim", 64),
        horizon=config.get("horizon", 12),
        gat_heads=config.get("gat_heads", 8),
        gat_layers=config.get("gat_layers", 2),
        transformer_heads=config.get("transformer_heads", 8),
        transformer_layers=config.get("transformer_layers", 3),
        transformer_ff_dim=config.get("transformer_ff_dim", 256),
        dropout=config.get("dropout", 0.1),
    )
