"""
Training loop for ST-GNN.
Fixed: mask_threshold=0.0 for normalized data.
"""

import time
import math
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from training.config import STGNNConfig
from evaluation.metrics import compute_metrics


class MaskedMAELoss(nn.Module):
    """MAE loss — mask_threshold=0.0 only masks exact zeros (missing sensors)."""

    def __init__(self, mask_threshold: float = 0.0, eps: float = 1e-4):
        super().__init__()
        self.mask_threshold = mask_threshold
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.mask_threshold == 0.0:
            mask = (target != 0.0).float()
        else:
            mask = (target.abs() > self.mask_threshold).float()
        loss = (pred - target).abs() * mask
        return loss.sum() / (mask.sum() + self.eps)


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-5):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        self.base_lrs      = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            lr_scale = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            lr_scale = max(
                self.min_lr / self.base_lrs[0],
                0.5 * (1 + math.cos(math.pi * progress)),
            )
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * lr_scale
        return self.optimizer.param_groups[0]["lr"]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: STGNNConfig,
        train_loader,
        val_loader,
        scaler=None,
        device: Optional[str] = None,
        start_epoch: int = 0,
    ):
        self.model        = model
        self.config       = config
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.scaler       = scaler
        self.start_epoch  = start_epoch

        self.device = device or config.train.device
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        self.model = self.model.to(self.device)

        self.criterion = MaskedMAELoss(mask_threshold=0.0)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )

        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=config.train.warmup_epochs,
            total_epochs=config.train.epochs,
            min_lr=config.train.min_lr,
        )

        self.save_dir = Path(config.train.save_dir) / config.train.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_mae     = float("inf")
        self.patience_counter = 0

        self.use_wandb = config.train.use_wandb
        if self.use_wandb:
            import wandb
            wandb.init(project="stgnn-traffic", name=config.train.experiment_name, config=config.to_dict())

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        steps = 0
        t0 = time.time()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in pbar:
            x          = batch["x"].to(self.device)
            edge_index = batch["edge_index"].to(self.device)
            y          = batch["y"].to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(x, edge_index)
            loss = self.criterion(pred.squeeze(-1), y.squeeze(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return {"train_loss": total_loss / steps, "epoch_time": time.time() - t0}

    @torch.no_grad()
    def eval_epoch(self) -> dict:
        self.model.eval()
        all_preds, all_targets = [], []

        for batch in self.val_loader:
            x          = batch["x"].to(self.device)
            edge_index = batch["edge_index"].to(self.device)
            y          = batch["y"].to(self.device)
            pred = self.model(x, edge_index)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

        preds   = np.concatenate(all_preds,   axis=0)
        targets = np.concatenate(all_targets, axis=0)

        if self.scaler is not None:
            preds   = self.scaler.inverse_transform(preds)
            targets = self.scaler.inverse_transform(targets)

        return compute_metrics(preds.squeeze(-1), targets.squeeze(-1))

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        state = {
            "epoch":           epoch,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics":         metrics,
            "config":          self.config.to_dict(),
        }
        torch.save(state, self.save_dir / f"epoch_{epoch:03d}.pt")
        if is_best:
            torch.save(state, self.save_dir / "best_model.pt")
            print(f"  ✓ New best model saved (val MAE: {metrics['mae']:.4f})")

    def train(self):
        cfg          = self.config.train
        total_epochs = cfg.epochs

        print(f"\nTraining on {self.device} | {total_epochs} epochs")
        if self.start_epoch > 0:
            print(f"Resuming from epoch {self.start_epoch}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}\n")

        for epoch in range(self.start_epoch, total_epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics   = self.eval_epoch()
            lr            = self.scheduler.step(epoch)

            print(
                f"Epoch {epoch+1:3d}/{total_epochs} | "
                f"loss={train_metrics['train_loss']:.4f} | "
                f"val_mae={val_metrics['mae']:.4f} | "
                f"val_rmse={val_metrics['rmse']:.4f} | "
                f"lr={lr:.6f}"
            )

            if self.use_wandb:
                import wandb
                wandb.log({"epoch": epoch+1, "lr": lr,
                           **{f"train/{k}": v for k, v in train_metrics.items()},
                           **{f"val/{k}":   v for k, v in val_metrics.items()}})

            is_best = val_metrics["mae"] < self.best_val_mae
            if is_best:
                self.best_val_mae     = val_metrics["mae"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best=is_best)

            if self.patience_counter >= cfg.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        print(f"\nTraining complete. Best val MAE: {self.best_val_mae:.4f}")
        if self.use_wandb:
            import wandb
            wandb.finish()