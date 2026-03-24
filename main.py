"""Main entry point for MT-OPMNet training and evaluation."""

import argparse

import torch

from src.model import MTOPMNet
from src.dataset import create_dataloaders
from src.trainer import Trainer
from src.evaluate import (
    evaluate_model, compute_metrics, print_metrics,
    save_results, plot_results,
)
from src.utils import (
    load_config, apply_fast_overrides, set_seed,
    get_device, model_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MT-OPMNet: Multi-Task Optical Performance Monitoring"
    )
    parser.add_argument(
        "--mode", choices=["train", "eval", "full"], default="full",
        help="Run mode: train, eval, or full (train+eval)",
    )
    parser.add_argument(
        "--config", default="configs/default.json",
        help="Path to configuration JSON file",
    )
    parser.add_argument(
        "--checkpoint", default="results/best_model.pt",
        help="Path to model checkpoint (for eval mode)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Use fast training settings for quick experiments",
    )
    return parser.parse_args()


def train(cfg, device):
    """Train the MT-OPMNet model."""
    print("Building dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    print(f"  Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}")

    model = MTOPMNet(
        n_bins=cfg["dataset"]["n_bins"],
        n_classes=cfg["model"]["n_classes"],
        use_caam=cfg["model"]["use_caam"],
    )
    model_summary(model)

    trainer = Trainer(model, cfg, device)
    trainer.train(train_loader, val_loader)

    return test_loader


def evaluate(cfg, device, checkpoint_path, test_loader=None):
    """Evaluate a trained MT-OPMNet model."""
    if test_loader is None:
        _, _, test_loader = create_dataloaders(cfg)

    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = MTOPMNet(
        n_bins=cfg["dataset"]["n_bins"],
        n_classes=cfg["model"]["n_classes"],
        use_caam=cfg["model"]["use_caam"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    results = evaluate_model(model, test_loader, device)
    metrics = compute_metrics(results)
    print_metrics(metrics)
    save_results(metrics, results)
    plot_results(results, metrics)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.fast:
        cfg = apply_fast_overrides(cfg)
        print("Fast mode enabled.")

    set_seed(cfg["dataset"]["seed"])
    device = get_device()
    print(f"Device: {device}")

    if args.mode == "train":
        train(cfg, device)
    elif args.mode == "eval":
        evaluate(cfg, device, args.checkpoint)
    else:  # full
        test_loader = train(cfg, device)
        evaluate(cfg, device, "results/best_model.pt", test_loader)


if __name__ == "__main__":
    main()
