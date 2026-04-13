"""
precompute_cache.py — Run model on all test frames once, save to JSON cache.
The cached API then serves from this file with zero ML dependencies.

Usage:
    python scripts/precompute_cache.py
    
Output:
    data/predictions_cache.json  (~15MB compressed)
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.stgnn import build_model

def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Paths
    data_dir   = Path("data")
    processed  = data_dir / "processed" / "metr-la"
    graph_dir  = data_dir / "graphs"
    model_path = "checkpoints/stgnn_metr_la/best_model.pt"

    # Load normalization
    mean = float(np.load(processed / "mean.npy"))
    std  = float(np.load(processed / "std.npy"))
    print(f"Normalization: mean={mean:.2f}, std={std:.2f}")

    # Load test data
    test_X = np.load(processed / "test_X.npy")   # (6832, 207, 12, 1)
    test_Y = np.load(processed / "test_Y.npy")   # (6832, 207, 12, 1)
    print(f"Test samples: {len(test_X)}")

    # Load graph
    edge_index = np.load(graph_dir / "metr-la_edge_index.npy")
    edge_idx   = torch.from_numpy(edge_index).long().to(device)

    # Load sensor coords
    with open(data_dir / "raw" / "sensor_coords.json") as f:
        coords = json.load(f)[:207]

    # Load road edges
    with open(data_dir / "raw" / "road_edges.json") as f:
        road_data = json.load(f)

    # Load model
    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"]["model"])
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    print("Model loaded")

    # Pre-compute all predictions
    all_frames = []
    batch_size = 64

    with torch.no_grad():
        for start in tqdm(range(0, len(test_X), batch_size), desc="Computing predictions"):
            end  = min(start + batch_size, len(test_X))
            xb   = torch.from_numpy(test_X[start:end]).float().to(device)
            pred = model(xb, edge_idx).cpu().numpy()  # (B, 207, 12, 1)

            for i in range(end - start):
                frame_idx = start + i
                current   = test_X[frame_idx, :, -1, 0] * std + mean   # (207,)
                predicted = pred[i, :, :, 0] * std + mean               # (207, 12)
                actual    = test_Y[frame_idx, :, :, 0] * std + mean     # (207, 12)

                sensors = []
                for s in range(207):
                    sensors.append({
                        "id":  s,
                        "lat": coords[s][0],
                        "lng": coords[s][1],
                        "cs":  round(float(max(0, current[s])), 1),
                        "pr":  [round(float(v), 1) for v in predicted[s]],
                        "ac":  [round(float(v), 1) for v in actual[s]],
                    })

                all_frames.append(sensors)

    print(f"Computed {len(all_frames)} frames")

    # Save cache
    cache = {
        "total_frames": len(all_frames),
        "mean":         mean,
        "std":          std,
        "coords":       coords,
        "edges":        road_data["edges"],
        "frames":       all_frames,
    }

    out_path = data_dir / "predictions_cache.json"
    print(f"Saving to {out_path} ...")
    with open(out_path, "w") as f:
        json.dump(cache, f, separators=(",", ":"))  # compact JSON

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Done! Cache size: {size_mb:.1f} MB")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    run()