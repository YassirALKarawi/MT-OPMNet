"""Utility functions for configuration and reproducibility."""

import json
import random
from pathlib import Path

import numpy as np
import torch


def load_config(path: str = "configs/default.json") -> dict:
    """Load and return JSON configuration."""
    with open(path) as f:
        return json.load(f)


def apply_fast_overrides(cfg: dict) -> dict:
    """Apply fast-mode overrides for quick experiments."""
    fast = cfg.get("fast", {})
    if "max_epochs" in fast:
        cfg["training"]["max_epochs"] = fast["max_epochs"]
    if "patience" in fast:
        cfg["training"]["patience"] = fast["patience"]
    if "n_realisations" in fast:
        cfg["dataset"]["n_realisations"] = fast["n_realisations"]
    return cfg


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: torch.nn.Module):
    """Print a summary of model architecture and parameters."""
    total = count_parameters(model)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Trainable parameters: {total:,}")
    print()
    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:20s} -> {n_params:>8,} params")
    print()
