"""
Preprocessor for METR-LA / PEMS-BAY from Zenodo CSV + PKL files.

BEFORE RUNNING: manually place these files in data/raw/
    METR-LA.csv
    adj_mx_METR-LA.pkl
    PEMS-BAY.csv          (optional, for pems-bay)
    adj_mx_PEMS-BAY.pkl   (optional, for pems-bay)

Download from: https://zenodo.org/records/5724362

Usage:
    python scripts/download_data.py --dataset metr-la
    python scripts/download_data.py --dataset pems-bay
"""

import pickle
import click
import numpy as np
import pandas as pd
from pathlib import Path


FILE_MAP = {
    "metr-la": {
        "speed_csv": "METR-LA.csv",
        "adj_pkl":   "adj_mx_METR-LA.pkl",
    },
    "pems-bay": {
        "speed_csv": "PEMS-BAY.csv",
        "adj_pkl":   "adj_mx_PEMS-BAY.pkl",
    },
}


def load_adj_pkl(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    if isinstance(data, list) and len(data) == 3:
        sensor_ids, sensor_id_to_ind, adj_mx = data
        return sensor_ids, sensor_id_to_ind, adj_mx.astype(np.float32)

    if isinstance(data, np.ndarray):
        N = data.shape[0]
        sensor_ids = list(range(N))
        sensor_id_to_ind = {i: i for i in range(N)}
        return sensor_ids, sensor_id_to_ind, data.astype(np.float32)

    raise ValueError(f"Unexpected pickle format: {type(data)}")


def adj_to_edge_index(adj_mx: np.ndarray, threshold: float = 0.0):
    rows, cols = np.where(adj_mx > threshold)
    edge_index   = np.stack([rows, cols], axis=0).astype(np.int64)
    edge_weights = adj_mx[rows, cols].astype(np.float32)
    return edge_index, edge_weights


def load_speed_csv(path: Path) -> np.ndarray:
    print(f"  Reading {path.name} ...")
    df = pd.read_csv(path, index_col=0)
    print(f"  Shape: {df.shape[0]} timesteps x {df.shape[1]} sensors")
    df = df.ffill().fillna(0.0)
    data = df.values.astype(np.float32)
    return data[:, :, np.newaxis]   # (T, N, 1)


def normalize(data: np.ndarray, train_end: int):
    train_slice = data[:train_end]
    nonzero = train_slice[train_slice != 0]
    mean = float(nonzero.mean())
    std  = float(nonzero.std())
    print(f"  Train stats -> mean={mean:.3f} mph,  std={std:.3f} mph")
    normalized = (data - mean) / (std + 1e-8)
    return normalized, mean, std


def split_and_window(data, train_r, val_r, in_steps, out_steps):
    T = data.shape[0]
    train_end = int(T * train_r)
    val_end   = int(T * (train_r + val_r))

    splits = {
        "train": data[:train_end],
        "val":   data[train_end:val_end],
        "test":  data[val_end:],
    }

    result = {}
    for name, split in splits.items():
        T_s = split.shape[0]
        X_list, Y_list = [], []
        for t in range(T_s - in_steps - out_steps + 1):
            X_list.append(split[t          : t + in_steps ].transpose(1, 0, 2))
            Y_list.append(split[t + in_steps : t + in_steps + out_steps].transpose(1, 0, 2))
        result[name] = (np.stack(X_list), np.stack(Y_list))
        print(f"  {name:5s}: {len(X_list)} samples  "
              f"X={result[name][0].shape}  Y={result[name][1].shape}")
    return result


@click.command()
@click.option("--dataset",     type=click.Choice(["metr-la", "pems-bay"]), default="metr-la")
@click.option("--data-dir",    default="data")
@click.option("--in-steps",    default=12)
@click.option("--out-steps",   default=12)
@click.option("--train-ratio", default=0.7)
@click.option("--val-ratio",   default=0.1)
def main(dataset, data_dir, in_steps, out_steps, train_ratio, val_ratio):
    raw_dir       = Path(data_dir) / "raw"
    processed_dir = Path(data_dir) / "processed" / dataset
    graph_dir     = Path(data_dir) / "graphs"

    processed_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    files = FILE_MAP[dataset]

    # ── 1. Adjacency matrix from .pkl ─────────────────────────────────────
    adj_path = raw_dir / files["adj_pkl"]
    if not adj_path.exists():
        print(f"\nERROR: File not found: {adj_path}")
        print(f"Please download  '{files['adj_pkl']}'  from:")
        print(f"  https://zenodo.org/records/5724362")
        print(f"and place it in:  {raw_dir.resolve()}\n")
        return

    print(f"\n[1/3] Loading adjacency from {adj_path.name} ...")
    sensor_ids, sensor_id_to_ind, adj_mx = load_adj_pkl(adj_path)
    N = adj_mx.shape[0]
    print(f"  Sensors: {N}")

    edge_index, edge_weights = adj_to_edge_index(adj_mx)
    np.save(graph_dir / f"{dataset}_adj.npy",          adj_mx)
    np.save(graph_dir / f"{dataset}_edge_index.npy",   edge_index)
    np.save(graph_dir / f"{dataset}_edge_weights.npy", edge_weights)
    print(f"  Graph saved -> {N} nodes, {edge_index.shape[1]} edges")

    # ── 2. Speed data from .csv ────────────────────────────────────────────
    csv_path = raw_dir / files["speed_csv"]
    if not csv_path.exists():
        print(f"\nERROR: File not found: {csv_path}")
        print(f"Please download  '{files['speed_csv']}'  from:")
        print(f"  https://zenodo.org/records/5724362")
        print(f"and place it in:  {raw_dir.resolve()}\n")
        return

    print(f"\n[2/3] Loading speed data from {csv_path.name} ...")
    data = load_speed_csv(csv_path)

    # ── 3. Normalise + window ──────────────────────────────────────────────
    print(f"\n[3/3] Normalising and windowing (in={in_steps}, out={out_steps}) ...")
    train_end = int(data.shape[0] * train_ratio)
    norm_data, mean, std = normalize(data, train_end)

    np.save(processed_dir / "mean.npy", np.array(mean))
    np.save(processed_dir / "std.npy",  np.array(std))

    windows = split_and_window(norm_data, train_ratio, val_ratio, in_steps, out_steps)

    for split_name, (X, Y) in windows.items():
        np.save(processed_dir / f"{split_name}_X.npy", X)
        np.save(processed_dir / f"{split_name}_Y.npy", Y)

    print(f"\nDone! Saved to: {processed_dir.resolve()}")
    print(f"mean={mean:.3f} mph   std={std:.3f} mph\n")


if __name__ == "__main__":
    main()