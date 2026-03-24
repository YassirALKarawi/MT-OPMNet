"""Evaluation metrics and reporting for MT-OPMNet."""

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


def evaluate_model(model, test_loader, device: torch.device):
    """Run inference on the test set and collect predictions.

    Args:
        model: Trained MTOPMNet.
        test_loader: Test DataLoader.
        device: Torch device.

    Returns:
        Dictionary with ground truth and predictions for both tasks.
    """
    model.eval()
    all_osnr_true, all_osnr_pred = [], []
    all_mfi_true, all_mfi_pred = [], []

    with torch.no_grad():
        for hist, osnr, mfi in test_loader:
            hist = hist.to(device)
            osnr_pred, mfi_logits = model(hist)

            all_osnr_true.append(osnr.numpy())
            all_osnr_pred.append(osnr_pred.cpu().numpy())
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

    metrics = {
        "osnr_mae": float(mean_absolute_error(osnr_t, osnr_p)),
        "osnr_rmse": float(np.sqrt(mean_squared_error(osnr_t, osnr_p))),
        "osnr_r2": float(r2_score(osnr_t, osnr_p)),
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
    print("\n" + "=" * 50)
    print("  MT-OPMNet Evaluation Results")
    print("=" * 50)

    print("\n  OSNR Estimation:")
    print(f"    MAE  : {metrics['osnr_mae']:.4f} dB")
    print(f"    RMSE : {metrics['osnr_rmse']:.4f} dB")
    print(f"    R²   : {metrics['osnr_r2']:.4f}")

    print("\n  Modulation Format Identification:")
    print(f"    Accuracy : {metrics['mfi_accuracy']:.2%}")
    report = metrics["mfi_report"]
    print(f"    Macro F1 : {report['macro avg']['f1-score']:.4f}")
    print(f"    W-Avg F1 : {report['weighted avg']['f1-score']:.4f}")
    print("=" * 50)


def save_results(metrics: dict, results: dict, save_dir: str = "results"):
    """Save metrics to JSON and evaluation data to npz."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save serialisable metrics
    serialisable = {k: v for k, v in metrics.items() if k != "mfi_report"}
    serialisable["mfi_macro_f1"] = metrics["mfi_report"]["macro avg"]["f1-score"]
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


def plot_results(results: dict, metrics: dict,
                 save_dir: str = "figures"):
    """Generate and save evaluation plots.

    Creates:
        - OSNR scatter plot (true vs predicted)
        - MFI confusion matrix
        - OSNR error distribution histogram
    """
    fig_dir = Path(save_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. OSNR scatter plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(results["osnr_true"], results["osnr_pred"],
               alpha=0.4, s=10, c="#2563eb")
    lims = [results["osnr_true"].min(), results["osnr_true"].max()]
    ax.plot(lims, lims, "k--", linewidth=1, label="Ideal")
    ax.set_xlabel("True OSNR (dB)")
    ax.set_ylabel("Predicted OSNR (dB)")
    ax.set_title(f"OSNR Estimation (MAE={metrics['osnr_mae']:.3f} dB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "osnr_scatter.png", dpi=150)
    plt.close(fig)

    # 2. Confusion matrix
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
    ax.set_title(f"MFI Confusion Matrix (Acc={metrics['mfi_accuracy']:.2%})")
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 3. OSNR error distribution
    errors = results["osnr_pred"] - results["osnr_true"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(errors, bins=50, color="#2563eb", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("OSNR Prediction Error (dB)")
    ax.set_ylabel("Count")
    ax.set_title("OSNR Error Distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "osnr_error_dist.png", dpi=150)
    plt.close(fig)

    print(f"Figures saved to {fig_dir}/")
