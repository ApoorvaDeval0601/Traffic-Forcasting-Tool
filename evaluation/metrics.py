"""
Traffic forecasting metrics: MAE, RMSE, MAPE.

Computed per-horizon (15min, 30min, 60min) and averaged.
Masks missing/zero values following DCRNN convention.
"""

import numpy as np
from typing import Dict, Optional


def masked_mae(pred: np.ndarray, target: np.ndarray, null_val: float = 0.0) -> float:
    mask = target != null_val
    if mask.sum() == 0:
        return 0.0
    return np.abs(pred[mask] - target[mask]).mean()


def masked_rmse(pred: np.ndarray, target: np.ndarray, null_val: float = 0.0) -> float:
    mask = target != null_val
    if mask.sum() == 0:
        return 0.0
    return np.sqrt(((pred[mask] - target[mask]) ** 2).mean())


def masked_mape(
    pred: np.ndarray, target: np.ndarray, null_val: float = 0.0, eps: float = 1e-4
) -> float:
    mask = target != null_val
    if mask.sum() == 0:
        return 0.0
    return (np.abs(pred[mask] - target[mask]) / (np.abs(target[mask]) + eps)).mean()


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    null_val: float = 0.0,
    horizons: Optional[list] = None,
) -> Dict[str, float]:
    """
    Compute MAE, RMSE, MAPE across all horizons + per-horizon metrics.

    Args:
        pred:     (B, N, H) — predicted speed
        target:   (B, N, H) — actual speed
        horizons: list of horizon indices to report (e.g. [2, 5, 11] = 15/30/60 min)
    """
    results = {
        "mae":  masked_mae(pred, target, null_val),
        "rmse": masked_rmse(pred, target, null_val),
        "mape": masked_mape(pred, target, null_val),
    }

    # Per-horizon breakdown
    if horizons is None:
        H = pred.shape[-1]
        # Report at 15min (step 3), 30min (step 6), 60min (step 12)
        horizons = [2, 5, min(11, H - 1)]

    horizon_names = ["15min", "30min", "60min"][: len(horizons)]
    for name, h in zip(horizon_names, horizons):
        p_h = pred[:, :, h]
        t_h = target[:, :, h]
        results[f"mae_{name}"]  = masked_mae(p_h, t_h, null_val)
        results[f"rmse_{name}"] = masked_rmse(p_h, t_h, null_val)
        results[f"mape_{name}"] = masked_mape(p_h, t_h, null_val)

    return results


def print_metrics_table(metrics_transformer: dict, metrics_lstm: dict):
    """Pretty-print comparison table of Transformer vs LSTM."""
    print("\n" + "=" * 60)
    print(f"{'Metric':<20} {'Transformer':>18} {'LSTM':>18}")
    print("=" * 60)
    for key in ["mae_15min", "mae_30min", "mae_60min",
                "rmse_15min", "rmse_30min", "rmse_60min",
                "mape_15min", "mape_30min", "mape_60min"]:
        t_val = metrics_transformer.get(key, float("nan"))
        l_val = metrics_lstm.get(key, float("nan"))
        winner = "←" if t_val < l_val else "  "
        print(f"{key:<20} {t_val:>18.4f} {l_val:>18.4f} {winner}")
    print("=" * 60)
    print(f"{'MAE (avg)':<20} {metrics_transformer['mae']:>18.4f} {metrics_lstm['mae']:>18.4f}")
    print(f"{'RMSE (avg)':<20} {metrics_transformer['rmse']:>18.4f} {metrics_lstm['rmse']:>18.4f}")
    print()
