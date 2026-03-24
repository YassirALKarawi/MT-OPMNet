"""Training loop with early stopping, cosine annealing, and history tracking."""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from .model import MTOPMNet
from .losses import FocalLoss, UncertaintyWeightedLoss


class Trainer:
    """Handles model training, validation, and checkpointing.

    Args:
        model: MTOPMNet instance.
        cfg: Full configuration dictionary.
        device: Torch device.
        osnr_stats: Dict with 'mean' and 'std' for OSNR denormalisation.
    """

    def __init__(self, model: MTOPMNet, cfg: dict,
                 device: torch.device, osnr_stats: dict = None):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.osnr_stats = osnr_stats or {"mean": 17.5, "std": 7.36}
        tcfg = cfg["training"]

        # Loss functions
        self.mse = nn.MSELoss()
        self.focal = FocalLoss(gamma=tcfg["focal_gamma"])

        # Uncertainty weighting
        self.use_uw = tcfg["use_uncertainty_weighting"]
        if self.use_uw:
            self.uw = UncertaintyWeightedLoss().to(device)
            params = list(model.parameters()) + list(self.uw.parameters())
        else:
            self.uw = None
            params = model.parameters()

        # Optimiser with weight decay and scheduler
        self.optimizer = torch.optim.AdamW(
            params, lr=tcfg["lr"], weight_decay=1e-4
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=tcfg["max_epochs"],
            eta_min=tcfg["lr_min"],
        )

        self.max_epochs = tcfg["max_epochs"]
        self.patience = tcfg["patience"]

    def _compute_osnr_mae_db(self, pred_norm, true_norm):
        """Compute OSNR MAE in original dB scale."""
        s = self.osnr_stats
        pred_db = pred_norm * s["std"] + s["mean"]
        true_db = true_norm * s["std"] + s["mean"]
        return torch.mean(torch.abs(pred_db - true_db)).item()

    def _run_epoch(self, loader, train: bool = True):
        """Run one epoch of training or validation."""
        self.model.train() if train else self.model.eval()
        total_loss = 0.0
        total_osnr_loss = 0.0
        total_mfi_loss = 0.0
        total_osnr_mae = 0.0
        correct = 0
        n_samples = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for hist, osnr, mfi in loader:
                hist = hist.to(self.device)
                osnr = osnr.to(self.device)
                mfi = mfi.to(self.device)

                osnr_pred, mfi_logits = self.model(hist)
                loss_osnr = self.mse(osnr_pred, osnr)
                loss_mfi = self.focal(mfi_logits, mfi)

                if self.use_uw:
                    loss = self.uw(loss_osnr, loss_mfi)
                else:
                    loss = loss_osnr + loss_mfi

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=5.0
                    )
                    self.optimizer.step()

                batch_size = hist.size(0)
                total_loss += loss.item() * batch_size
                total_osnr_loss += loss_osnr.item() * batch_size
                total_mfi_loss += loss_mfi.item() * batch_size
                total_osnr_mae += (
                    self._compute_osnr_mae_db(osnr_pred, osnr) * batch_size
                )
                correct += (mfi_logits.argmax(1) == mfi).sum().item()
                n_samples += batch_size

        return {
            "loss": total_loss / n_samples,
            "osnr_loss": total_osnr_loss / n_samples,
            "mfi_loss": total_mfi_loss / n_samples,
            "osnr_mae_db": total_osnr_mae / n_samples,
            "mfi_acc": correct / n_samples,
        }

    def train(self, train_loader, val_loader, save_dir: str = "results"):
        """Full training loop with early stopping.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            save_dir: Directory for saving checkpoints.

        Returns:
            Dictionary of training history.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train": [], "val": []}

        print(f"\n{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>11} | "
              f"{'OSNR MAE':>9} | {'Val Acc':>8} | {'LR':>10}")
        print("-" * 72)

        start = time.time()
        for epoch in range(1, self.max_epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False)
            self.scheduler.step()

            history["train"].append(train_metrics)
            history["val"].append(val_metrics)

            lr = self.optimizer.param_groups[0]["lr"]
            print(f"{epoch:>6} | {train_metrics['loss']:>11.5f} | "
                  f"{val_metrics['loss']:>11.5f} | "
                  f"{val_metrics['osnr_mae_db']:>8.4f} | "
                  f"{val_metrics['mfi_acc']:>7.2%} | {lr:>10.2e}")

            # Early stopping on validation loss
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                checkpoint = {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "osnr_stats": self.osnr_stats,
                    "config": self.cfg,
                }
                if self.use_uw:
                    checkpoint["uw_state"] = self.uw.state_dict()
                torch.save(checkpoint, save_path / "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(no improvement for {self.patience} epochs)")
                    break

        elapsed = time.time() - start
        print(f"\nTraining complete in {elapsed:.1f}s | "
              f"Best val loss: {best_val_loss:.5f}")

        # Save training history
        self._save_history(history, save_path)

        # Log uncertainty weights if used
        if self.use_uw:
            w_osnr = torch.exp(-self.uw.log_var_osnr).item()
            w_mfi = torch.exp(-self.uw.log_var_mfi).item()
            print(f"Uncertainty weights — OSNR: {w_osnr:.4f}, MFI: {w_mfi:.4f}")

        return history

    def _save_history(self, history: dict, save_path: Path):
        """Save training history to JSON."""
        serialisable = {
            "train": [
                {k: float(v) for k, v in epoch.items()}
                for epoch in history["train"]
            ],
            "val": [
                {k: float(v) for k, v in epoch.items()}
                for epoch in history["val"]
            ],
        }
        with open(save_path / "training_history.json", "w") as f:
            json.dump(serialisable, f, indent=2)
