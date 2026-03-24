"""Synthetic amplitude histogram dataset for OPM tasks."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Modulation format labels and OSNR range
MODULATION_FORMATS = ["OOK", "DPSK", "DQPSK", "8QAM", "16QAM"]
OSNR_RANGE = (5.0, 30.0)  # dB


def generate_amplitude_histogram(modulation_idx: int, osnr_db: float,
                                 n_symbols: int, n_bins: int,
                                 rng: np.random.Generator) -> np.ndarray:
    """Generate a synthetic amplitude histogram for given modulation and OSNR.

    Simulates received signal amplitudes by combining ideal constellation
    amplitudes with AWGN noise at the specified OSNR level.

    Args:
        modulation_idx: Index into MODULATION_FORMATS.
        osnr_db: Optical signal-to-noise ratio in dB.
        n_symbols: Number of symbols to simulate.
        n_bins: Number of histogram bins.
        rng: NumPy random generator.

    Returns:
        Normalised amplitude histogram of shape (n_bins,).
    """
    # Ideal amplitude levels for each modulation format
    amplitude_levels = {
        0: np.array([0.0, 1.0]),                          # OOK
        1: np.array([1.0]),                                # DPSK
        2: np.array([1.0]),                                # DQPSK
        3: np.array([0.5, 1.0, 1.5]),                     # 8QAM
        4: np.array([0.33, 0.67, 1.0, 1.33]),             # 16QAM
    }

    levels = amplitude_levels[modulation_idx]
    symbols = rng.choice(levels, size=n_symbols)

    # Add AWGN noise based on OSNR
    osnr_linear = 10 ** (osnr_db / 10)
    signal_power = np.mean(symbols ** 2)
    noise_std = np.sqrt(signal_power / (2 * osnr_linear))
    noisy = np.abs(symbols + rng.normal(0, noise_std, n_symbols))

    # Build normalised histogram
    hist, _ = np.histogram(noisy, bins=n_bins, range=(0, 2.0))
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total

    return hist


def build_dataset(n_bins: int = 100, n_symbols: int = 16384,
                  n_realisations: int = 5, seed: int = 42):
    """Build the full synthetic OPM dataset.

    Generates amplitude histograms for all combinations of modulation
    formats and OSNR values, with multiple noise realisations per point.

    Args:
        n_bins: Number of histogram bins.
        n_symbols: Symbols per histogram.
        n_realisations: Independent noise realisations per (format, OSNR).
        seed: Random seed for reproducibility.

    Returns:
        histograms: Array of shape (N, n_bins).
        osnr_labels: Array of shape (N,).
        mfi_labels: Array of shape (N,) with integer class indices.
    """
    rng = np.random.default_rng(seed)
    osnr_values = np.arange(OSNR_RANGE[0], OSNR_RANGE[1] + 0.5, 0.5)

    histograms = []
    osnr_labels = []
    mfi_labels = []

    for mod_idx in range(len(MODULATION_FORMATS)):
        for osnr in osnr_values:
            for _ in range(n_realisations):
                hist = generate_amplitude_histogram(
                    mod_idx, osnr, n_symbols, n_bins, rng
                )
                histograms.append(hist)
                osnr_labels.append(osnr)
                mfi_labels.append(mod_idx)

    return (
        np.array(histograms, dtype=np.float32),
        np.array(osnr_labels, dtype=np.float32),
        np.array(mfi_labels, dtype=np.int64),
    )


class OPMDataset(Dataset):
    """PyTorch dataset wrapping amplitude histograms with OPM labels."""

    def __init__(self, histograms: np.ndarray, osnr: np.ndarray,
                 mfi: np.ndarray):
        self.histograms = torch.from_numpy(histograms).unsqueeze(1)  # (N,1,B)
        self.osnr = torch.from_numpy(osnr).unsqueeze(1)              # (N,1)
        self.mfi = torch.from_numpy(mfi)                             # (N,)

    def __len__(self):
        return len(self.osnr)

    def __getitem__(self, idx):
        return self.histograms[idx], self.osnr[idx], self.mfi[idx]


def create_dataloaders(cfg: dict):
    """Create train/val/test DataLoaders from configuration.

    Args:
        cfg: Full configuration dictionary.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    ds = cfg["dataset"]
    histograms, osnr, mfi = build_dataset(
        n_bins=ds["n_bins"],
        n_symbols=ds["n_symbols"],
        n_realisations=ds["n_realisations"],
        seed=ds["seed"],
    )

    # Shuffle deterministically
    rng = np.random.default_rng(ds["seed"])
    perm = rng.permutation(len(histograms))
    histograms, osnr, mfi = histograms[perm], osnr[perm], mfi[perm]

    # Split
    n = len(histograms)
    n_train = int(n * ds["train_ratio"])
    n_val = int(n * ds["val_ratio"])

    splits = {
        "train": (histograms[:n_train], osnr[:n_train], mfi[:n_train]),
        "val": (histograms[n_train:n_train + n_val],
                osnr[n_train:n_train + n_val],
                mfi[n_train:n_train + n_val]),
        "test": (histograms[n_train + n_val:],
                 osnr[n_train + n_val:],
                 mfi[n_train + n_val:]),
    }

    batch_size = cfg["training"]["batch_size"]
    loaders = {}
    for name, (h, o, m) in splits.items():
        ds_obj = OPMDataset(h, o, m)
        loaders[name] = DataLoader(
            ds_obj,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=0,
            pin_memory=True,
        )

    return loaders["train"], loaders["val"], loaders["test"]
