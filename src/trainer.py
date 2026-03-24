"""Training loop with early stopping and cosine annealing."""

import time
from pathlib import Path

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
    """

    def __init__(self, model: MTOPMNet, cfg: dict,
                 device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
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

        # Optimiser and scheduler
        self.optimizer = torch.optim.Adam(params, lr=tcfg["lr"])
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=tcfg["max_epochs"],
            eta_min=tcfg["lr_min"],
        )

        self.max_epochs = tcfg["max_epochs"]
        self.patience = tcfg["patience"]

    def _run_epoch(self, loader, train: bool = True):
        """Run one epoch of training or validation."""
        self.model.train() if train else self.model.eval()
        total_loss = 0.0
        total_osnr_loss = 0.0
        total_mfi_loss = 0.0
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
                correct += (mfi_logits.argmax(1) == mfi).sum().item()
                n_samples += batch_size

        return {
            "loss": total_loss / n_samples,
            "osnr_loss": total_osnr_loss / n_samples,
            "mfi_loss": total_mfi_loss / n_samples,
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

        print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>11} | "
              f"{'Val Acc':>8} | {'LR':>10}")
        print("-" * 62)

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
                  f"{val_metrics['mfi_acc']:>7.2%} | {lr:>10.2e}")

            # Early stopping
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                checkpoint = {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "val_loss": best_val_loss,
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

        return history
