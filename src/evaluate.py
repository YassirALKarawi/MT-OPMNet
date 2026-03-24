"""Evaluation metrics, reporting, and visualisation for MT-OPMNet."""

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .dataset import MODULATION_FORMATS


def evaluate_model(model, test_loader, device: torch.device,
                   osnr_stats: dict = None):
    """Run inference on the test set and collect predictions.

    Args:
        model: Trained MTOPMNet.
        test_loader: Test DataLoader.
        device: Torch device.
        osnr_stats: Dict with 'mean' and 'std' for denormalisation.

    Returns:
        Dictionary with ground truth and predictions for both tasks (in dB).
    """
    if osnr_stats is None:
        osnr_stats = {"mean": 0.0, "std": 1.0}

    model.eval()
    all_osnr_true, all_osnr_pred = [], []
    all_mfi_true, all_mfi_pred = [], []

    with torch.no_grad():
        for hist, osnr_norm, mfi in test_loader:
            hist = hist.to(device)
            osnr_pred_norm, mfi_logits = model(hist)

            # Denormalise OSNR to dB
            osnr_true_db = osnr_norm.numpy() * osnr_stats["std"] + osnr_stats["mean"]
            osnr_pred_db = (osnr_pred_norm.cpu().numpy() * osnr_stats["std"]
                           + osnr_stats["mean"])

            all_osnr_true.append(osnr_true_db)
            all_osnr_pred.append(osnr_pred_db)
            all_mfi_true.append(mfi.numpy())
            all_mfi_pred.append(mfi_logits.argmax(1).cpu().numpy())

    return {
        "osnr_true": np.concatenate(all_osnr_true).flatten(),
        "osnr_pred": np.concatenate(all_osnr_pred).flatten(),
        "mfi_true": np.concatenate(all_mfi_true),
        "mfi_pred": np.concatenate(all_mfi_pred),
    }


def compute_metrics(results: dict) -> dict:
    """Compute evaluation metrics for both tasks.

    Args:
        results: Output from evaluate_model().

    Returns:
        Dictionary of metric names to values.
    """
    osnr_t, osnr_p = results["osnr_true"], results["osnr_pred"]
    mfi_t, mfi_p = results["mfi_true"], results["mfi_pred"]

    # Per-modulation OSNR MAE
    per_mod_mae = {}
    for idx, name in enumerate(MODULATION_FORMATS):
        mask = mfi_t == idx
        if mask.sum() > 0:
            per_mod_mae[name] = float(mean_absolute_error(
                osnr_t[mask], osnr_p[mask]
            ))

    metrics = {
        "osnr_mae": float(mean_absolute_error(osnr_t, osnr_p)),
        "osnr_rmse": float(np.sqrt(mean_squared_error(osnr_t, osnr_p))),
        "osnr_r2": float(r2_score(osnr_t, osnr_p)),
        "osnr_max_error": float(np.max(np.abs(osnr_t - osnr_p))),
        "osnr_per_mod_mae": per_mod_mae,
        "mfi_accuracy": float(np.mean(mfi_t == mfi_p)),
        "mfi_report": classification_report(
            mfi_t, mfi_p,
            target_names=MODULATION_FORMATS,
            output_dict=True,
        ),
    }
    return metrics


def print_metrics(metrics: dict):
    """Print a formatted summary of evaluation metrics."""
    print("\n" + "=" * 55)
    print("  MT-OPMNet Evaluation Results")
    print("=" * 55)

    print("\n  OSNR Estimation:")
    print(f"    MAE       : {metrics['osnr_mae']:.4f} dB")
    print(f"    RMSE      : {metrics['osnr_rmse']:.4f} dB")
    print(f"    R²        : {metrics['osnr_r2']:.4f}")
    print(f"    Max Error : {metrics['osnr_max_error']:.4f} dB")

    print("\n  OSNR MAE per Modulation Format:")
    for name, mae in metrics["osnr_per_mod_mae"].items():
        print(f"    {name:8s} : {mae:.4f} dB")

    print("\n  Modulation Format Identification:")
    print(f"    Accuracy : {metrics['mfi_accuracy']:.2%}")
    report = metrics["mfi_report"]
    print(f"    Macro F1 : {report['macro avg']['f1-score']:.4f}")
    print(f"    W-Avg F1 : {report['weighted avg']['f1-score']:.4f}")

    print("\n  Per-Class MFI Performance:")
    print(f"    {'Format':8s} | {'Precision':>9s} | {'Recall':>6s} | {'F1':>6s}")
    print(f"    {'-'*38}")
    for name in MODULATION_FORMATS:
        if name in report:
            r = report[name]
            print(f"    {name:8s} | {r['precision']:>9.4f} | "
                  f"{r['recall']:>6.4f} | {r['f1-score']:>6.4f}")

    print("=" * 55)


def save_results(metrics: dict, results: dict, save_dir: str = "results"):
    """Save metrics to JSON and evaluation data to npz."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save serialisable metrics
    serialisable = {k: v for k, v in metrics.items()
                    if k not in ("mfi_report",)}
    serialisable["mfi_macro_f1"] = metrics["mfi_report"]["macro avg"]["f1-score"]
    serialisable["mfi_weighted_f1"] = metrics["mfi_report"]["weighted avg"]["f1-score"]
    with open(save_path / "metrics.json", "w") as f:
        json.dump(serialisable, f, indent=2)

    # Save raw predictions
    np.savez(
        save_path / "predictions.npz",
        osnr_true=results["osnr_true"],
        osnr_pred=results["osnr_pred"],
        mfi_true=results["mfi_true"],
        mfi_pred=results["mfi_pred"],
    )
    print(f"Results saved to {save_path}/")


def plot_results(results: dict, metrics: dict,
                 history: dict = None, save_dir: str = "figures"):
    """Generate and save all evaluation plots.

    Creates:
        - OSNR scatter plot (true vs predicted)
        - MFI confusion matrix
        - OSNR error distribution histogram
        - Per-modulation OSNR error boxplot
        - OSNR error vs true OSNR
        - Training curves (if history provided)
    """
    fig_dir = Path(save_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    _plot_osnr_scatter(results, metrics, fig_dir)
    _plot_confusion_matrix(results, metrics, fig_dir)
    _plot_osnr_error_dist(results, fig_dir)
    _plot_osnr_per_modulation(results, fig_dir)
    _plot_osnr_vs_error(results, fig_dir)

    if history:
        _plot_training_curves(history, fig_dir)

    print(f"Figures saved to {fig_dir}/")


def _plot_osnr_scatter(results, metrics, fig_dir):
    """OSNR true vs predicted scatter plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.cm.Set2(results["mfi_true"] / 4)
    ax.scatter(results["osnr_true"], results["osnr_pred"],
               alpha=0.5, s=12, c=colors)
    lims = [results["osnr_true"].min(), results["osnr_true"].max()]
    ax.plot(lims, lims, "k--", linewidth=1, label="Ideal")
    ax.set_xlabel("True OSNR (dB)")
    ax.set_ylabel("Predicted OSNR (dB)")
    ax.set_title(f"OSNR Estimation (MAE = {metrics['osnr_mae']:.3f} dB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "osnr_scatter.png", dpi=150)
    plt.close(fig)


def _plot_confusion_matrix(results, metrics, fig_dir):
    """MFI confusion matrix."""
    cm = confusion_matrix(results["mfi_true"], results["mfi_pred"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(MODULATION_FORMATS)))
    ax.set_yticks(range(len(MODULATION_FORMATS)))
    ax.set_xticklabels(MODULATION_FORMATS, rotation=45, ha="right")
    ax.set_yticklabels(MODULATION_FORMATS)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"MFI Confusion Matrix (Acc = {metrics['mfi_accuracy']:.2%})")
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def _plot_osnr_error_dist(results, fig_dir):
    """OSNR prediction error distribution."""
    errors = results["osnr_pred"] - results["osnr_true"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(errors, bins=50, color="#2563eb", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("OSNR Prediction Error (dB)")
    ax.set_ylabel("Count")
    ax.set_title(f"OSNR Error Distribution (std = {errors.std():.3f} dB)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "osnr_error_dist.png", dpi=150)
    plt.close(fig)


def _plot_osnr_per_modulation(results, fig_dir):
    """Boxplot of OSNR error per modulation format."""
    errors = results["osnr_pred"] - results["osnr_true"]
    mfi = results["mfi_true"]

    data = []
    labels = []
    for idx, name in enumerate(MODULATION_FORMATS):
        mask = mfi == idx
        if mask.sum() > 0:
            data.append(np.abs(errors[mask]))
            labels.append(name)

    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ["#6366f1", "#2563eb", "#0891b2", "#059669", "#dc2626"]
    for patch, color in zip(bp["boxes"], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("|OSNR Error| (dB)")
    ax.set_title("OSNR Absolute Error per Modulation Format")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "osnr_per_modulation.png", dpi=150)
    plt.close(fig)


def _plot_osnr_vs_error(results, fig_dir):
    """OSNR MAE as a function of true OSNR value."""
    osnr_true = results["osnr_true"]
    errors = np.abs(results["osnr_pred"] - osnr_true)

    # Bin by OSNR value
    osnr_bins = np.arange(10.0, 31.0, 1.0)
    bin_centers = []
    bin_maes = []
    for i in range(len(osnr_bins) - 1):
        mask = (osnr_true >= osnr_bins[i]) & (osnr_true < osnr_bins[i + 1])
        if mask.sum() > 0:
            bin_centers.append((osnr_bins[i] + osnr_bins[i + 1]) / 2)
            bin_maes.append(errors[mask].mean())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(bin_centers, bin_maes, "o-", color="#2563eb", linewidth=2,
            markersize=5)
    ax.set_xlabel("True OSNR (dB)")
    ax.set_ylabel("MAE (dB)")
    ax.set_title("OSNR Estimation Error vs OSNR Level")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "osnr_vs_error.png", dpi=150)
    plt.close(fig)


def _plot_training_curves(history, fig_dir):
    """Plot training and validation loss/accuracy curves."""
    epochs = range(1, len(history["train"]) + 1)

    train_loss = [e["loss"] for e in history["train"]]
    val_loss = [e["loss"] for e in history["val"]]
    train_acc = [e["mfi_acc"] for e in history["train"]]
    val_acc = [e["mfi_acc"] for e in history["val"]]
    val_mae = [e["osnr_mae_db"] for e in history["val"]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Loss curves
    axes[0].plot(epochs, train_loss, label="Train", color="#2563eb")
    axes[0].plot(epochs, val_loss, label="Val", color="#dc2626")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MFI accuracy
    axes[1].plot(epochs, train_acc, label="Train", color="#2563eb")
    axes[1].plot(epochs, val_acc, label="Val", color="#dc2626")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("MFI Classification Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # OSNR MAE
    axes[2].plot(epochs, val_mae, color="#059669", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("MAE (dB)")
    axes[2].set_title("Validation OSNR MAE")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "training_curves.png", dpi=150)
    plt.close(fig)
