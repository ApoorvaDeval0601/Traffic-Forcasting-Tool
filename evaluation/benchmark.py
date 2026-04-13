"""
Benchmark: evaluate both models on test set and save comparison JSON.

Usage:
    python evaluation/benchmark.py \
        --transformer-ckpt checkpoints/stgnn_metr_la/best_model.pt \
        --lstm-ckpt checkpoints/lstm_metr_la/best_model.pt \
        --dataset metr-la
"""

import sys
import json
import click
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.stgnn import build_model
from models.lstm_temporal import STGNNWithLSTM
from evaluation.metrics import compute_metrics, print_metrics_table


def load_model_from_checkpoint(ckpt_path: str, device: str):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]["model"]
    model_type = "lstm" if "lstm" in str(ckpt_path).lower() else "stgnn"

    if model_type == "lstm":
        model = STGNNWithLSTM(
            in_features=cfg["in_features"],
            hidden_dim=cfg["hidden_dim"],
            horizon=cfg["horizon"],
        )
    else:
        model = build_model(cfg)

    model.load_state_dict(checkpoint["model_state"])
    model.eval().to(device)
    return model


@torch.no_grad()
def evaluate_model(model, test_X, test_Y, edge_index, mean, std, batch_size=64, device="cpu"):
    """Evaluate model and return metrics in original mph scale."""
    N_samples = test_X.shape[0]
    all_preds = []

    edge_idx = torch.from_numpy(edge_index).long().to(device)

    for i in range(0, N_samples, batch_size):
        xb   = torch.from_numpy(test_X[i:i+batch_size]).float().to(device)
        pred = model(xb, edge_idx).squeeze(-1).cpu().numpy()
        all_preds.append(pred)

    preds = np.concatenate(all_preds, axis=0)   # (N, nodes, horizon) — normalized

    # Denormalize both predictions and targets to mph
    preds_mph   = preds   * std + mean
    targets_mph = test_Y.squeeze(-1) * std + mean

    return compute_metrics(preds_mph, targets_mph)


@click.command()
@click.option("--transformer-ckpt", required=True)
@click.option("--lstm-ckpt",        required=True)
@click.option("--dataset",          default="metr-la")
@click.option("--data-dir",         default="data")
@click.option("--device",           default="cpu")
@click.option("--output",           default="evaluation/benchmark_results.json")
def main(transformer_ckpt, lstm_ckpt, dataset, data_dir, device, output):
    processed  = Path(data_dir) / "processed" / dataset
    graph_dir  = Path(data_dir) / "graphs"

    # Load test data (still normalized)
    test_X     = np.load(processed / "test_X.npy")    # (B, N, T, F) normalized
    test_Y     = np.load(processed / "test_Y.npy")    # (B, N, H, F) normalized
    edge_index = np.load(graph_dir / f"{dataset}_edge_index.npy")

    # Load normalization stats
    mean = float(np.load(processed / "mean.npy"))
    std  = float(np.load(processed / "std.npy"))

    print(f"Test samples : {test_X.shape[0]}")
    print(f"Denorm stats : mean={mean:.2f} mph, std={std:.2f} mph\n")

    print("Loading models...")
    transformer = load_model_from_checkpoint(transformer_ckpt, device)
    lstm_model  = load_model_from_checkpoint(lstm_ckpt,        device)

    print("Evaluating Transformer...")
    transformer_metrics = evaluate_model(
        transformer, test_X, test_Y, edge_index, mean, std, device=device
    )

    print("Evaluating LSTM...")
    lstm_metrics = evaluate_model(
        lstm_model, test_X, test_Y, edge_index, mean, std, device=device
    )

    # Add parameter counts
    transformer_metrics["params"] = sum(p.numel() for p in transformer.parameters())
    lstm_metrics["params"]        = sum(p.numel() for p in lstm_model.parameters())

    print_metrics_table(transformer_metrics, lstm_metrics)

    results = {
        "dataset":     dataset,
        "transformer": transformer_metrics,
        "lstm":        lstm_metrics,
    }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()