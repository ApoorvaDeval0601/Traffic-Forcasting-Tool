"""
Unit tests for ST-GNN components.
Run with: pytest tests/ -v
"""

import torch
import numpy as np
import pytest


# ── Model shape tests ──────────────────────────────────────────────────────

def make_dummy_graph(N=10, E=30):
    """Create random edge_index for N nodes with E edges."""
    src = torch.randint(0, N, (E,))
    dst = torch.randint(0, N, (E,))
    edge_index = torch.stack([src, dst], dim=0)
    return edge_index


def test_gat_spatial_output_shape():
    from models.gat_spatial import GATSpatialEncoder
    B, N, T, F = 2, 10, 12, 1
    model = GATSpatialEncoder(in_features=F, hidden_dim=32, num_heads=4, num_layers=2)
    x = torch.randn(B, N, T, F)
    edge_index = make_dummy_graph(N, E=20)
    out = model(x, edge_index)
    assert out.shape == (B, N, T, 32), f"Expected (2,10,12,32), got {out.shape}"


def test_transformer_temporal_output_shape():
    from models.transformer_temporal import TrafficTemporalTransformer
    B, N, T, D = 2, 10, 12, 32
    model = TrafficTemporalTransformer(hidden_dim=D, num_heads=4, num_layers=2, ff_dim=64)
    x = torch.randn(B, N, T, D)
    out = model(x)
    assert out.shape == (B, N, T, D), f"Expected {(B,N,T,D)}, got {out.shape}"


def test_decoder_output_shape():
    from models.transformer_temporal import MultiStepDecoder
    B, N, T, D = 2, 10, 12, 32
    H = 12
    model = MultiStepDecoder(hidden_dim=D, horizon=H, out_features=1)
    x = torch.randn(B, N, T, D)
    out = model(x)
    assert out.shape == (B, N, H, 1), f"Expected {(B,N,H,1)}, got {out.shape}"


def test_full_stgnn_forward():
    from models.stgnn import STGNN
    B, N, T, F = 2, 10, 12, 1
    model = STGNN(
        in_features=F, hidden_dim=32, horizon=12,
        gat_heads=4, gat_layers=1,
        transformer_heads=4, transformer_layers=1, transformer_ff_dim=64,
    )
    x = torch.randn(B, N, T, F)
    edge_index = make_dummy_graph(N, E=20)
    out = model(x, edge_index)
    assert out.shape == (B, N, 12, 1)


def test_stgnn_with_lstm():
    from models.lstm_temporal import STGNNWithLSTM
    B, N, T, F = 2, 10, 12, 1
    model = STGNNWithLSTM(
        in_features=F, hidden_dim=32, horizon=12,
        num_gat_layers=1, gat_heads=4, lstm_layers=1,
    )
    x = torch.randn(B, N, T, F)
    edge_index = make_dummy_graph(N, E=20)
    out = model(x, edge_index)
    assert out.shape == (B, N, 12, 1)


def test_attention_return():
    from models.stgnn import STGNN
    B, N, T, F = 1, 6, 12, 1
    model = STGNN(
        in_features=F, hidden_dim=16, horizon=3,
        gat_heads=2, gat_layers=1,
        transformer_heads=2, transformer_layers=1, transformer_ff_dim=32,
    )
    x = torch.randn(B, N, T, F)
    edge_index = make_dummy_graph(N, E=12)
    preds, attn = model(x, edge_index, return_attention=True)
    assert preds.shape == (B, N, 3, 1)
    assert isinstance(attn, list) and len(attn) > 0


# ── Metric tests ────────────────────────────────────────────────────────────

def test_masked_mae_ignores_zeros():
    from evaluation.metrics import masked_mae
    pred   = np.array([5.0, 10.0, 0.0])
    target = np.array([4.0,  0.0, 0.0])  # second is zero (missing)
    mae = masked_mae(pred, target, null_val=0.0)
    # Only non-zero target contributes: |5-4| = 1.0
    assert abs(mae - 1.0) < 1e-5


def test_metrics_per_horizon():
    from evaluation.metrics import compute_metrics
    B, N, H = 4, 5, 12
    pred   = np.random.randn(B, N, H) + 50
    target = np.random.randn(B, N, H) + 50
    metrics = compute_metrics(pred, target, horizons=[2, 5, 11])
    assert "mae" in metrics
    assert "mae_15min" in metrics
    assert "rmse_60min" in metrics


# ── Loss test ────────────────────────────────────────────────────────────────

def test_masked_mae_loss():
    from training.trainer import MaskedMAELoss
    loss_fn = MaskedMAELoss(mask_threshold=10.0)
    pred   = torch.ones(2, 5, 12) * 50
    target = torch.ones(2, 5, 12) * 55
    target[:, :, :2] = 0.0   # First 2 steps "missing"
    loss = loss_fn(pred, target)
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_parameter_count():
    from models.stgnn import STGNN
    model = STGNN(in_features=1, hidden_dim=64, horizon=12)
    params = model.count_parameters()
    assert 100_000 < params < 2_000_000, f"Unexpected param count: {params}"
