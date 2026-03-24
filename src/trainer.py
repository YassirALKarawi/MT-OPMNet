"""Training loop matching paper Algorithm 1 and Section IV-E.

Uses Smooth-L1 (Huber) loss for OSNR, focal loss for MFI, Adam optimiser,
cosine annealing schedule, and early stopping with patience=20.
"""

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
    """Training with Smooth-L1 + Focal + Uncertainty Weighting.

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
        self.osnr_stats = osnr_stats or {"mean": 20.0, "std": 5.77}
        tcfg = cfg["training"]

        # Loss functions (paper Eq. 8: Smooth-L1, Eq. 10: Focal)
        self.huber = nn.SmoothL1Loss()
        self.focal = FocalLoss(gamma=tcfg["focal_gamma"])

        # Uncertainty weighting (paper Eq. 7)
        self.use_uw = tcfg["use_uncertainty_weighting"]
        if self.use_uw:
            self.uw = UncertaintyWeightedLoss().to(device)
            params = list(model.parameters()) + list(self.uw.parameters())
        else:
            self.uw = None
            params = model.parameters()

        # Adam optimiser (paper Section IV-E)
        self.optimizer = torch.optim.Adam(params, lr=tcfg["lr"])
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=tcfg["max_epochs"],
            eta_min=tcfg["lr_min"],
        )

        self.max_epochs = tcfg["max_epochs"]
        self.patience = tcfg["patience"]

    def _compute_osnr_mae_db(self, pred_norm, true_norm):
        s = self.osnr_stats
        pred_db = pred_norm * s["std"] + s["mean"]
        true_db = true_norm * s["std"] + s["mean"]
        return torch.mean(torch.abs(pred_db - true_db)).item()

    def _run_epoch(self, loader, train: bool = True):
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

                # Smooth-L1 for OSNR (Eq. 8)
                loss_osnr = self.huber(osnr_pred, osnr)
                # Focal for MFI (Eq. 10)
                loss_mfi = self.focal(mfi_logits, mfi)

                if self.use_uw:
                    loss = self.uw(loss_osnr, loss_mfi)
                else:
                    loss = loss_osnr + loss_mfi

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    all_params = list(self.model.parameters())
                    if self.use_uw:
                        all_params += list(self.uw.parameters())
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
                    self.optimizer.step()

                bs = hist.size(0)
                total_loss += loss.item() * bs
                total_osnr_loss += loss_osnr.item() * bs
                total_mfi_loss += loss_mfi.item() * bs
                total_osnr_mae += self._compute_osnr_mae_db(osnr_pred, osnr) * bs
                correct += (mfi_logits.argmax(1) == mfi).sum().item()
                n_samples += bs

        return {
            "loss": total_loss / n_samples,
            "osnr_loss": total_osnr_loss / n_samples,
            "mfi_loss": total_mfi_loss / n_samples,
            "osnr_mae_db": total_osnr_mae / n_samples,
            "mfi_acc": correct / n_samples,
        }

    def train(self, train_loader, val_loader, save_dir: str = "results"):
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
            train_m = self._run_epoch(train_loader, train=True)
            val_m = self._run_epoch(val_loader, train=False)
            self.scheduler.step()

            history["train"].append(train_m)
            history["val"].append(val_m)

            lr = self.optimizer.param_groups[0]["lr"]
            print(f"{epoch:>6} | {train_m['loss']:>11.5f} | "
                  f"{val_m['loss']:>11.5f} | "
                  f"{val_m['osnr_mae_db']:>8.4f} | "
                  f"{val_m['mfi_acc']:>7.2%} | {lr:>10.2e}")

            if val_m["loss"] < best_val_loss:
                best_val_loss = val_m["loss"]
                patience_counter = 0
                ckpt = {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "osnr_stats": self.osnr_stats,
                    "config": self.cfg,
                }
                if self.use_uw:
                    ckpt["uw_state"] = self.uw.state_dict()
                torch.save(ckpt, save_path / "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(no improvement for {self.patience} epochs)")
                    break

        elapsed = time.time() - start
        print(f"\nTraining complete in {elapsed:.1f}s | "
              f"Best val loss: {best_val_loss:.5f}")

        self._save_history(history, save_path)

        if self.use_uw:
            w1 = torch.exp(-self.uw.log_var_osnr).item()
            w2 = torch.exp(-self.uw.log_var_mfi).item()
            print(f"Uncertainty weights — OSNR: {w1:.4f}, MFI: {w2:.4f}")

        return history

    def _save_history(self, history, save_path):
        serialisable = {
            split: [{k: float(v) for k, v in ep.items()} for ep in epochs]
            for split, epochs in history.items()
        }
        with open(save_path / "training_history.json", "w") as f:
            json.dump(serialisable, f, indent=2)
