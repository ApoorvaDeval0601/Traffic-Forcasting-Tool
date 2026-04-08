"""
CLI training entry point.

Usage:
    # Train ST-GNN (Transformer)
    python scripts/train.py --model stgnn --dataset metr-la

    # Train LSTM baseline
    python scripts/train.py --model lstm --dataset metr-la

    # Resume from last checkpoint if interrupted
    python scripts/train.py --model stgnn --dataset metr-la --resume

    # Custom hyperparameters
    python scripts/train.py --model stgnn --hidden-dim 128 --epochs 200
"""

import sys
import glob
import click
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.stgnn import build_model
from models.lstm_temporal import STGNNWithLSTM
from training.trainer import Trainer
from training.config import STGNNConfig, DataConfig, ModelConfig, TrainConfig


class TrafficDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, edge_index: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.edge_index = torch.from_numpy(edge_index).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "y": self.Y[idx],
            "edge_index": self.edge_index,
        }


def collate_fn(batch):
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "edge_index": batch[0]["edge_index"],
    }


def find_latest_checkpoint(save_dir: Path) -> Path | None:
    """Find the most recent epoch checkpoint to resume from."""
    checkpoints = sorted(glob.glob(str(save_dir / "epoch_*.pt")))
    if checkpoints:
        return Path(checkpoints[-1])
    return None


@click.command()
@click.option("--model",            type=click.Choice(["stgnn", "lstm"]), default="stgnn")
@click.option("--dataset",          type=click.Choice(["metr-la", "pems-bay"]), default="metr-la")
@click.option("--hidden-dim",       default=64)
@click.option("--epochs",           default=100)
@click.option("--batch-size",       default=64)
@click.option("--lr",               default=1e-3)
@click.option("--gat-layers",       default=2)
@click.option("--transformer-layers", default=3)
@click.option("--dropout",          default=0.1)
@click.option("--device",           default="auto")
@click.option("--use-wandb",        is_flag=True)
@click.option("--data-dir",         default="data")
@click.option("--save-dir",         default="checkpoints")
@click.option("--resume",           is_flag=True, help="Resume from latest checkpoint")
def main(
    model, dataset, hidden_dim, epochs, batch_size, lr,
    gat_layers, transformer_layers, dropout, device, use_wandb,
    data_dir, save_dir, resume
):
    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

    print(f"\n{'='*50}")
    print(f"  Training {model.upper()} on {dataset}")
    print(f"  Device: {device}")
    print(f"{'='*50}\n")

    # Load data
    processed   = Path(data_dir) / "processed" / dataset
    graph_dir   = Path(data_dir) / "graphs"

    train_X    = np.load(processed / "train_X.npy")
    train_Y    = np.load(processed / "train_Y.npy")
    val_X      = np.load(processed / "val_X.npy")
    val_Y      = np.load(processed / "val_Y.npy")
    edge_index = np.load(graph_dir / f"{dataset}_edge_index.npy")

    in_features = train_X.shape[-1]
    horizon     = train_Y.shape[2]

    print(f"Train: {train_X.shape}, Val: {val_X.shape}")
    print(f"Graph: {edge_index.shape[1]} edges, in_features={in_features}, horizon={horizon}\n")

    train_ds = TrafficDataset(train_X, train_Y, edge_index)
    val_ds   = TrafficDataset(val_X,   val_Y,   edge_index)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda"), collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )

    # Build model
    cfg_dict = {
        "in_features":        in_features,
        "hidden_dim":         hidden_dim,
        "horizon":            horizon,
        "gat_layers":         gat_layers,
        "gat_heads":          8,
        "transformer_layers": transformer_layers,
        "transformer_heads":  8,
        "transformer_ff_dim": hidden_dim * 4,
        "dropout":            dropout,
    }

    exp_name = f"stgnn_{dataset.replace('-', '_')}" if model == "stgnn" \
               else f"lstm_{dataset.replace('-', '_')}"

    if model == "stgnn":
        net = build_model(cfg_dict)
    else:
        net = STGNNWithLSTM(
            in_features=in_features,
            hidden_dim=hidden_dim,
            horizon=horizon,
            num_gat_layers=gat_layers,
            gat_heads=8,
            lstm_layers=2,
            dropout=dropout,
        )

    # Config
    config = STGNNConfig(
        data=DataConfig(dataset=dataset),
        model=ModelConfig(
            in_features=in_features,
            hidden_dim=hidden_dim,
            horizon=horizon,
            gat_layers=gat_layers,
            gat_heads=8,
            transformer_layers=transformer_layers,
            transformer_heads=8,
            transformer_ff_dim=hidden_dim * 4,
            dropout=dropout,
        ),
        train=TrainConfig(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            use_wandb=use_wandb,
            experiment_name=exp_name,
            save_dir=save_dir,
        ),
    )

    # Resume from checkpoint if requested
    start_epoch = 0
    if resume:
        ckpt_dir = Path(save_dir) / exp_name
        latest   = find_latest_checkpoint(ckpt_dir)
        if latest:
            print(f"Resuming from: {latest}")
            ckpt = torch.load(latest, map_location=device)
            net.load_state_dict(ckpt["model_state"])
            start_epoch = ckpt["epoch"] + 1
            print(f"Resuming from epoch {start_epoch}\n")
        else:
            print("No checkpoint found, starting from scratch.\n")

    print(net)

    # Train
    trainer = Trainer(
        model=net,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        start_epoch=start_epoch,
    )
    trainer.train()


if __name__ == "__main__":
    main()