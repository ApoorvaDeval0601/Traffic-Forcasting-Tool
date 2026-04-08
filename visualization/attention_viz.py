"""
Visualize GAT attention weights on the road network.

Generates:
1. Heatmap: N×N attention matrix per layer
2. Graph overlay: Edge thickness proportional to attention
3. Top-k neighbors for a selected sensor
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from pathlib import Path


def plot_attention_heatmap(
    attn_weights: np.ndarray,
    top_n: int = 30,
    title: str = "GAT Layer 1 — Attention Weights",
    save_path: str = None,
):
    """
    Plot attention weight heatmap for top_n most connected sensors.

    Args:
        attn_weights: (N, N) averaged attention matrix
        top_n: show only the top_n most active sensors
    """
    # Select top_n sensors by total attention received
    total_attention = attn_weights.sum(axis=0)
    top_idx = np.argsort(total_attention)[-top_n:]
    sub_attn = attn_weights[np.ix_(top_idx, top_idx)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sub_attn, cmap="Blues", aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Attention weight")

    ax.set_title(title, fontsize=14, pad=16)
    ax.set_xlabel("Target sensor (receiving attention)", fontsize=11)
    ax.set_ylabel("Source sensor (giving attention)", fontsize=11)

    # Tick labels for top sensors
    tick_step = max(1, top_n // 10)
    ticks = list(range(0, top_n, tick_step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(top_idx[t]) for t in ticks], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([str(top_idx[t]) for t in ticks], fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def plot_congestion_propagation(
    sensor_speeds: np.ndarray,   # (T, N) — speed over time
    sensor_lats: np.ndarray,     # (N,)
    sensor_lngs: np.ndarray,     # (N,)
    incident_time: int,          # Time step of incident
    n_steps: int = 6,
    save_dir: str = None,
):
    """
    Plot a sequence of frames showing congestion propagating from an incident.

    Args:
        sensor_speeds: (T, N) speed array (mph)
        incident_time: time step where incident occurs
        n_steps: number of post-incident steps to show
    """
    fig, axes = plt.subplots(1, n_steps, figsize=(4 * n_steps, 4))
    norm = Normalize(vmin=0, vmax=80)
    cmap = plt.cm.RdYlGn  # Red (slow) → Green (fast)

    for i, ax in enumerate(axes):
        t = incident_time + i
        speeds = sensor_speeds[t]
        colors = cmap(norm(speeds))

        sc = ax.scatter(sensor_lngs, sensor_lats, c=speeds, cmap=cmap,
                        norm=norm, s=20, alpha=0.85, linewidths=0)

        ax.set_title(f"t+{i*5}min", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("Speed (mph)", fontsize=10)

    fig.suptitle("Congestion propagation after incident", fontsize=13, y=1.02)
    plt.tight_layout()

    if save_dir:
        path = Path(save_dir) / "congestion_propagation.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()
    return fig


def plot_forecast_comparison(
    actual: np.ndarray,    # (H,) or (T,)
    predicted_transformer: np.ndarray,
    predicted_lstm: np.ndarray,
    sensor_id: int = 0,
    horizon: int = 12,
    save_path: str = None,
):
    """
    Plot actual vs Transformer vs LSTM forecast for a single sensor.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    steps = np.arange(1, horizon + 1) * 5

    ax.plot(steps, actual[:horizon], "k-", linewidth=2, label="Actual", zorder=3)
    ax.plot(steps, predicted_transformer[:horizon], "b--", linewidth=1.8,
            marker="o", markersize=4, label="Transformer", alpha=0.85)
    ax.plot(steps, predicted_lstm[:horizon], "r:", linewidth=1.8,
            marker="s", markersize=4, label="LSTM", alpha=0.75)

    # Shade error region
    ax.fill_between(steps, actual[:horizon], predicted_transformer[:horizon],
                    alpha=0.1, color="blue", label="Transformer error")

    ax.axhline(60, color="green", linestyle="--", linewidth=0.8, alpha=0.5, label="Free flow")
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("Speed (mph)")
    ax.set_title(f"Sensor {sensor_id} — 1-hour forecast comparison")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 85)
    ax.grid(alpha=0.3)

    mae_t = np.abs(predicted_transformer[:horizon] - actual[:horizon]).mean()
    mae_l = np.abs(predicted_lstm[:horizon] - actual[:horizon]).mean()
    ax.text(0.02, 0.05,
            f"MAE — Transformer: {mae_t:.2f}  LSTM: {mae_l:.2f}",
            transform=ax.transAxes, fontsize=9, color="gray")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    return fig
