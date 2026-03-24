"""Synthetic amplitude histogram dataset for OPM tasks.

Generates realistic amplitude histograms (AAH) for optical signals by
simulating complex (I/Q) baseband signals with proper constellation
geometries and AWGN noise based on OSNR.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Modulation format labels and OSNR range
MODULATION_FORMATS = ["OOK", "DPSK", "DQPSK", "8QAM", "16QAM"]
OSNR_RANGE = (5.0, 30.0)  # dB


def _ook_constellation():
    """OOK: On-Off Keying — two amplitude levels."""
    return np.array([0.0 + 0j, 1.0 + 0j])


def _dpsk_constellation():
    """DPSK: 2 symbols on unit circle (0 and pi)."""
    return np.exp(1j * np.array([0, np.pi]))


def _dqpsk_constellation():
    """DQPSK: 4 symbols on unit circle (pi/4 spacing)."""
    return np.exp(1j * np.array([np.pi / 4, 3 * np.pi / 4,
                                  5 * np.pi / 4, 7 * np.pi / 4]))


def _8qam_constellation():
    """8QAM: star-8QAM with two amplitude rings."""
    inner = 0.5 * np.exp(1j * np.array([np.pi / 4, 3 * np.pi / 4,
                                         5 * np.pi / 4, 7 * np.pi / 4]))
    outer = 1.0 * np.exp(1j * np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2]))
    return np.concatenate([inner, outer])


def _16qam_constellation():
    """16QAM: square 4x4 grid constellation."""
    levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)  # normalised
    grid = np.array([i + 1j * q for i in levels for q in levels])
    return grid


CONSTELLATIONS = {
    0: _ook_constellation,
    1: _dpsk_constellation,
    2: _dqpsk_constellation,
    3: _8qam_constellation,
    4: _16qam_constellation,
}


def generate_amplitude_histogram(modulation_idx: int, osnr_db: float,
                                 n_symbols: int, n_bins: int,
                                 rng: np.random.Generator) -> np.ndarray:
    """Generate a synthetic amplitude histogram for given modulation and OSNR.

    Uses complex baseband signal model:
        r = s + n,  where n ~ CN(0, sigma^2)
    Then computes |r| (amplitude) and builds a normalised histogram.

    Args:
        modulation_idx: Index into MODULATION_FORMATS.
        osnr_db: Optical signal-to-noise ratio in dB.
        n_symbols: Number of symbols to simulate.
        n_bins: Number of histogram bins.
        rng: NumPy random generator.

    Returns:
        Normalised amplitude histogram of shape (n_bins,).
    """
    constellation = CONSTELLATIONS[modulation_idx]()
    symbols = rng.choice(constellation, size=n_symbols)

    # Signal power
    signal_power = np.mean(np.abs(symbols) ** 2)

    # AWGN noise: complex Gaussian with variance = signal_power / OSNR
    osnr_linear = 10 ** (osnr_db / 10)
    if signal_power > 0:
        noise_var = signal_power / osnr_linear
    else:
        # OOK case: half symbols are zero, use average power
        noise_var = 0.5 / osnr_linear

    noise = (rng.normal(0, np.sqrt(noise_var / 2), n_symbols)
             + 1j * rng.normal(0, np.sqrt(noise_var / 2), n_symbols))

    received = symbols + noise
    amplitudes = np.abs(received)

    # Dynamic histogram range based on constellation
    max_amp = np.max(np.abs(constellation)) * 2.0
    max_amp = max(max_amp, 0.5)  # minimum range

    # Build normalised histogram
    hist, _ = np.histogram(amplitudes, bins=n_bins, range=(0, max_amp))
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
        osnr_labels: Array of shape (N,) in dB.
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
                 mfi: np.ndarray, osnr_mean: float = 0.0,
                 osnr_std: float = 1.0):
        self.histograms = torch.from_numpy(histograms).unsqueeze(1)  # (N,1,B)
        # Normalise OSNR for regression stability
        osnr_norm = (osnr - osnr_mean) / osnr_std
        self.osnr = torch.from_numpy(osnr_norm.astype(np.float32)).unsqueeze(1)
        self.mfi = torch.from_numpy(mfi)
        self.osnr_mean = osnr_mean
        self.osnr_std = osnr_std

    def __len__(self):
        return len(self.osnr)

    def __getitem__(self, idx):
        return self.histograms[idx], self.osnr[idx], self.mfi[idx]


def create_dataloaders(cfg: dict):
    """Create train/val/test DataLoaders from configuration.

    OSNR values are z-score normalised using training set statistics
    for stable regression training.

    Args:
        cfg: Full configuration dictionary.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, osnr_stats).
        osnr_stats is a dict with 'mean' and 'std' for denormalisation.
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

    # Split indices
    n = len(histograms)
    n_train = int(n * ds["train_ratio"])
    n_val = int(n * ds["val_ratio"])

    # Compute normalisation from training set only
    osnr_mean = float(osnr[:n_train].mean())
    osnr_std = float(osnr[:n_train].std())
    if osnr_std < 1e-6:
        osnr_std = 1.0

    osnr_stats = {"mean": osnr_mean, "std": osnr_std}

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
        ds_obj = OPMDataset(h, o, m, osnr_mean, osnr_std)
        loaders[name] = DataLoader(
            ds_obj,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=0,
            pin_memory=True,
        )

    return loaders["train"], loaders["val"], loaders["test"], osnr_stats
