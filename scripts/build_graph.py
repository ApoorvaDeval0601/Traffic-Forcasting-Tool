"""
build_graph.py - No-op helper.

The adjacency matrix is already extracted from the .pkl file
during download_data.py and saved to data/graphs/.

This script just confirms the graph files exist.

Usage:
    python scripts/build_graph.py --dataset metr-la
"""

import click
import numpy as np
from pathlib import Path


@click.command()
@click.option("--dataset", type=click.Choice(["metr-la", "pems-bay"]), default="metr-la")
@click.option("--data-dir", default="data")
def main(dataset, data_dir):
    graph_dir = Path(data_dir) / "graphs"
    edge_index_path = graph_dir / f"{dataset}_edge_index.npy"
    adj_path        = graph_dir / f"{dataset}_adj.npy"

    if not edge_index_path.exists():
        print(f"Graph not found. Run this first:")
        print(f"  python scripts/download_data.py --dataset {dataset}")
        return

    edge_index = np.load(edge_index_path)
    adj        = np.load(adj_path)
    print(f"Graph for {dataset}:")
    print(f"  Nodes : {adj.shape[0]}")
    print(f"  Edges : {edge_index.shape[1]}")
    print(f"  Files : {graph_dir}")


if __name__ == "__main__":
    main()
