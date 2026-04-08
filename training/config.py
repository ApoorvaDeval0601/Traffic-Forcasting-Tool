"""
Training configuration for ST-GNN.
"""

from dataclasses import dataclass, field, asdict
from typing import Literal


@dataclass
class DataConfig:
    dataset: Literal["metr-la", "pems-bay"] = "metr-la"
    data_dir: str = "data"
    in_steps: int = 12          # 12 × 5min = 1 hour input window
    out_steps: int = 12         # 12 × 5min = 1 hour prediction horizon
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    # test_ratio = 0.2 (remainder)
    normalize: bool = True


@dataclass
class ModelConfig:
    in_features: int = 1        # Speed only
    hidden_dim: int = 64
    horizon: int = 12
    gat_heads: int = 8
    gat_layers: int = 2
    transformer_heads: int = 8
    transformer_layers: int = 3
    transformer_ff_dim: int = 256
    dropout: float = 0.1


@dataclass
class TrainConfig:
    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"

    # Scheduler
    scheduler: str = "cosine"       # cosine | step | plateau
    warmup_epochs: int = 5
    min_lr: float = 1e-5

    # Training
    epochs: int = 100
    batch_size: int = 64
    grad_clip: float = 5.0
    early_stop_patience: int = 15

    # Loss
    loss: str = "masked_mae"        # masked_mae | mse | huber
    mask_threshold: float = 10.0    # Ignore speeds below threshold (stopped/missing)

    # Logging
    log_interval: int = 50          # steps
    eval_interval: int = 1          # epochs
    save_dir: str = "checkpoints"
    use_wandb: bool = False
    experiment_name: str = "stgnn_metrla"

    # Hardware
    device: str = "cuda"            # cuda | cpu | mps
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class STGNNConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return asdict(self)


# Preset configs

METR_LA_CONFIG = STGNNConfig(
    data=DataConfig(dataset="metr-la"),
    model=ModelConfig(hidden_dim=64, gat_layers=2, transformer_layers=3),
    train=TrainConfig(epochs=100, batch_size=64, lr=1e-3),
)

PEMS_BAY_CONFIG = STGNNConfig(
    data=DataConfig(dataset="pems-bay"),
    model=ModelConfig(hidden_dim=64, gat_layers=2, transformer_layers=3),
    train=TrainConfig(epochs=100, batch_size=64, lr=1e-3),
)

# Ablation: LSTM vs Transformer
LSTM_BASELINE_CONFIG = STGNNConfig(
    data=DataConfig(dataset="metr-la"),
    model=ModelConfig(hidden_dim=64, gat_layers=2),
    train=TrainConfig(epochs=100, batch_size=64, lr=1e-3, experiment_name="lstm_baseline_metrla"),
)
