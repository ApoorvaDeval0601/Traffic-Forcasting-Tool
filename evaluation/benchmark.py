"""
Benchmark: evaluate both models on test set and save comparison JSON.

Usage:
    python evaluation/benchmark.py \
        --transformer-ckpt checkpoints/stgnn_metrla/best_model.pt \
        --lstm-ckpt checkpoints/lstm_metrla/best_model.pt \
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
    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = checkpoint["config"]["model"]
    model_type = "lstm" if "lstm" in ckpt_path.lower() else "stgnn"

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
def evaluate_model(model, test_X, test_Y, edge_index, batch_size=64, device="cpu"):
    N_samples = test_X.shape[0]
    all_preds, all_targets = [], []

    edge_idx = torch.from_numpy(edge_index).long().to(device)

    for i in range(0, N_samples, batch_size):
        xb = torch.from_numpy(test_X[i:i+batch_size]).float().to(device)
        pred = model(xb, edge_idx).squeeze(-1).cpu().numpy()
        all_preds.append(pred)
        all_targets.append(test_Y[i:i+batch_size].squeeze(-1))

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return compute_metrics(preds, targets)


@click.command()
@click.option("--transformer-ckpt", required=True)
@click.option("--lstm-ckpt", required=True)
@click.option("--dataset", default="metr-la")
@click.option("--data-dir", default="data")
@click.option("--device", default="cpu")
@click.option("--output", default="evaluation/benchmark_results.json")
def main(transformer_ckpt, lstm_ckpt, dataset, data_dir, device, output):
    processed = Path(data_dir) / "processed" / dataset
    graph_dir  = Path(data_dir) / "graphs"

    # Load test data
    test_X = np.load(processed / "test_X.npy")   # (B, N, T, F)
    test_Y = np.load(processed / "test_Y.npy")   # (B, N, H, F)
    edge_index = np.load(graph_dir / f"{dataset}_edge_index.npy")

    # Denormalize targets
    mean = np.load(processed / "mean.npy")
    std  = np.load(processed / "std.npy")
    test_Y_real = test_Y * std + mean

    print("Loading models...")
    transformer = load_model_from_checkpoint(transformer_ckpt, device)
    lstm_model  = load_model_from_checkpoint(lstm_ckpt, device)

    print("\nEvaluating Transformer...")
    transformer_metrics = evaluate_model(transformer, test_X, test_Y_real, edge_index, device=device)

    print("Evaluating LSTM...")
    lstm_metrics = evaluate_model(lstm_model, test_X, test_Y_real, edge_index, device=device)

    # Add parameter counts
    transformer_metrics["params"] = sum(p.numel() for p in transformer.parameters())
    lstm_metrics["params"] = sum(p.numel() for p in lstm_model.parameters())

    print_metrics_table(transformer_metrics, lstm_metrics)

    results = {
        "dataset": dataset,
        "transformer": transformer_metrics,
        "lstm": lstm_metrics,
    }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output}")


if __name__ == "__main__":
    main()
