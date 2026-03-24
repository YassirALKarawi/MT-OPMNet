"""Analytical WDM channel model and AAH dataset generation for OPM tasks.

Implements a 9-channel WDM analytical fibre-channel model with ASE noise,
self-phase modulation (SPM), cross-phase modulation (XPM), residual
chromatic dispersion, and laser phase noise, as described in the paper.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ── Modulation formats (paper Section III) ──────────────────────────
MODULATION_FORMATS = ["QPSK", "8QAM", "16QAM", "32QAM", "64QAM"]

# Physical constants
PLANCK = 6.626e-34          # J·s
SPEED_OF_LIGHT = 3e8        # m/s


# ── Constellation definitions ───────────────────────────────────────

def _qpsk_constellation():
    """QPSK: 4 symbols on unit circle at π/4 + k·π/2."""
    return np.exp(1j * np.array([np.pi / 4, 3 * np.pi / 4,
                                  5 * np.pi / 4, 7 * np.pi / 4]))


def _8qam_constellation():
    """Star-8QAM: inner ring (r=0.5) and outer ring (r=1.0)."""
    inner = 0.5 * np.exp(1j * np.array([np.pi / 4, 3 * np.pi / 4,
                                         5 * np.pi / 4, 7 * np.pi / 4]))
    outer = 1.0 * np.exp(1j * np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2]))
    c = np.concatenate([inner, outer])
    return c / np.sqrt(np.mean(np.abs(c) ** 2))


def _16qam_constellation():
    """Square 16-QAM: 4×4 grid, normalised to unit average power."""
    levels = np.array([-3, -1, 1, 3], dtype=np.float64)
    c = np.array([i + 1j * q for i in levels for q in levels])
    return c / np.sqrt(np.mean(np.abs(c) ** 2))


def _32qam_constellation():
    """Cross-32QAM constellation, normalised to unit average power."""
    points = []
    for i in range(-3, 4, 2):
        for q in range(-3, 4, 2):
            if abs(i) + abs(q) <= 6:
                points.append(i + 1j * q)
    # Pad to 32 with outer corners if needed
    extra = [(-5 + 1j), (-5 - 1j), (5 + 1j), (5 - 1j),
             (-1 + 5j), (1 + 5j), (-1 - 5j), (1 - 5j)]
    for p in extra:
        if len(points) >= 32:
            break
        points.append(p)
    c = np.array(points[:32])
    return c / np.sqrt(np.mean(np.abs(c) ** 2))


def _64qam_constellation():
    """Square 64-QAM: 8×8 grid, normalised to unit average power."""
    levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float64)
    c = np.array([i + 1j * q for i in levels for q in levels])
    return c / np.sqrt(np.mean(np.abs(c) ** 2))


CONSTELLATIONS = {
    0: _qpsk_constellation,
    1: _8qam_constellation,
    2: _16qam_constellation,
    3: _32qam_constellation,
    4: _64qam_constellation,
}


# ── WDM analytical channel model ───────────────────────────────────

def _compute_ase_power(n_spans, span_loss_db, nf_db, freq_hz, bw_hz):
    """Compute accumulated ASE noise power (Eq. 3 in paper).

    P_ASE = N_span · h · ν · NF_lin · (G - 1) · B_ref
    """
    nf_lin = 10 ** (nf_db / 10)
    gain_lin = 10 ** (span_loss_db / 10)   # G = span loss
    p_ase = n_spans * PLANCK * freq_hz * nf_lin * (gain_lin - 1) * bw_hz
    return p_ase


def _compute_nli_power(launch_power_w, n_spans, span_length_km,
                       alpha_lin, beta2, gamma, symbol_rate_hz,
                       n_channels, channel_spacing_hz):
    """Estimate nonlinear interference power via simplified GN-model.

    SPM + XPM contribution using the Gaussian-noise model approximation.
    """
    l_eff = (1 - np.exp(-alpha_lin * span_length_km * 1e3)) / alpha_lin
    l_eff_a = 1.0 / alpha_lin

    # SPM contribution
    p_nli_spm = (8 / 27) * gamma ** 2 * launch_power_w ** 3 * l_eff ** 2
    p_nli_spm *= np.log(np.pi ** 2 * abs(beta2) * l_eff_a
                        * symbol_rate_hz ** 2 + 1e-30)
    p_nli_spm /= (np.pi * abs(beta2) * symbol_rate_hz ** 2 + 1e-30)

    # XPM from neighbouring channels (approximate)
    if n_channels > 1:
        xpm_factor = 0.0
        for k in range(1, (n_channels - 1) // 2 + 1):
            delta_f = k * channel_spacing_hz
            xpm_factor += np.log(1 + np.pi ** 2 * abs(beta2) * l_eff_a
                                 * delta_f * symbol_rate_hz + 1e-30)
        p_nli_xpm = (16 / 27) * gamma ** 2 * launch_power_w ** 3 * l_eff ** 2
        p_nli_xpm *= xpm_factor / (np.pi * abs(beta2)
                                    * symbol_rate_hz ** 2 + 1e-30)
    else:
        p_nli_xpm = 0.0

    # Scale by number of spans (incoherent accumulation)
    return (p_nli_spm + p_nli_xpm) * n_spans


def _apply_phase_noise(symbols, linewidth_hz, symbol_rate_hz, rng):
    """Apply Wiener laser phase noise process."""
    n = len(symbols)
    t_sym = 1.0 / symbol_rate_hz
    phase_var = 2 * np.pi * linewidth_hz * t_sym
    phase_increments = rng.normal(0, np.sqrt(phase_var), n)
    phase_walk = np.cumsum(phase_increments)
    return symbols * np.exp(1j * phase_walk)


def _apply_residual_cd(symbols, beta2, residual_km, symbol_rate_hz):
    """Apply residual chromatic dispersion in frequency domain."""
    n = len(symbols)
    freqs = np.fft.fftfreq(n, d=1.0 / symbol_rate_hz)
    # β₂ in s²/m, residual in m
    h_cd = np.exp(-1j * 0.5 * beta2 * (2 * np.pi * freqs) ** 2
                  * residual_km * 1e3)
    return np.fft.ifft(np.fft.fft(symbols) * h_cd)


def generate_amplitude_histogram(mod_idx, osnr_db, n_symbols, n_bins,
                                 symbol_rate_gbd, distance_km,
                                 launch_power_dbm, wdm_cfg, rng):
    """Generate a synthetic AAH using the analytical WDM channel model.

    Implements the signal model from paper Sections III and V:
    complex symbols → SPM/XPM + ASE noise → phase noise → residual CD
    → coherent detection → |r| → normalised histogram.
    """
    # ── Constellation ──
    constellation = CONSTELLATIONS[mod_idx]()
    symbols = rng.choice(constellation, size=n_symbols)

    # ── Physical parameters ──
    symbol_rate_hz = symbol_rate_gbd * 1e9
    wavelength = wdm_cfg["centre_wavelength_nm"] * 1e-9
    freq_hz = SPEED_OF_LIGHT / wavelength
    alpha_db_km = wdm_cfg["fibre_attenuation_db_per_km"]
    alpha_lin = alpha_db_km / (10 * np.log10(np.e)) / 1e3  # Np/m
    span_km = wdm_cfg["span_length_km"]
    n_spans = max(1, int(round(distance_km / span_km)))
    span_loss_db = alpha_db_km * span_km
    beta2 = -(wavelength ** 2 * wdm_cfg["fibre_dispersion_ps_per_nm_per_km"]
              * 1e-3) / (2 * np.pi * SPEED_OF_LIGHT)  # s²/m
    gamma = wdm_cfg["nonlinear_coefficient_per_w_per_km"] * 1e-3  # 1/(W·m)
    n_channels = wdm_cfg["n_channels"]
    ch_spacing_hz = wdm_cfg["channel_spacing_ghz"] * 1e9
    nf_db = wdm_cfg["edfa_noise_figure_db"]
    bw_ref_hz = wdm_cfg["reference_bandwidth_nm"] * 1e-9 * freq_hz ** 2 / SPEED_OF_LIGHT

    # ── Signal power ──
    p_tx = 10 ** ((launch_power_dbm - 30) / 10)  # W
    signal_power = np.mean(np.abs(symbols) ** 2)
    symbols_scaled = symbols * np.sqrt(p_tx / signal_power)

    # ── ASE noise (Eq. 3) ──
    p_ase = _compute_ase_power(n_spans, span_loss_db, nf_db, freq_hz, bw_ref_hz)

    # ── NLI noise (GN-model) ──
    p_nli = _compute_nli_power(p_tx, n_spans, span_km, alpha_lin, beta2,
                               gamma, symbol_rate_hz, n_channels, ch_spacing_hz)

    # ── Target OSNR: scale noise to match desired OSNR ──
    osnr_lin = 10 ** (osnr_db / 10)
    total_noise_power = p_tx / osnr_lin
    # Blend ASE + NLI proportionally
    p_noise_total = p_ase + p_nli
    if p_noise_total > 0:
        noise_scale = np.sqrt(total_noise_power / p_noise_total)
    else:
        noise_scale = np.sqrt(total_noise_power / (p_ase + 1e-30))

    ase_std = np.sqrt(p_ase / 2) * noise_scale
    nli_std = np.sqrt(max(p_nli, 0) / 2) * noise_scale

    # Complex AWGN (ASE)
    ase_noise = (rng.normal(0, ase_std, n_symbols)
                 + 1j * rng.normal(0, ase_std, n_symbols))
    # NLI noise (modelled as Gaussian)
    nli_noise = (rng.normal(0, nli_std, n_symbols)
                 + 1j * rng.normal(0, nli_std, n_symbols))

    received = symbols_scaled + ase_noise + nli_noise

    # ── Laser phase noise ──
    linewidth_hz = wdm_cfg["laser_linewidth_khz"] * 1e3
    received = _apply_phase_noise(received, linewidth_hz,
                                  symbol_rate_hz, rng)

    # ── Residual CD (small random residual) ──
    residual_cd_km = rng.uniform(0, 2.0)
    if residual_cd_km > 0:
        received = _apply_residual_cd(received, beta2, residual_cd_km,
                                      symbol_rate_hz)

    # ── Amplitude histogram ──
    amplitudes = np.abs(received)
    max_amp = np.percentile(amplitudes, 99.5) * 1.2
    max_amp = max(max_amp, 0.5)

    hist, _ = np.histogram(amplitudes, bins=n_bins, range=(0, max_amp))
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total

    return hist


# ── Dataset builder ─────────────────────────────────────────────────

def build_dataset(cfg):
    """Build the full synthetic OPM dataset matching paper Section V.

    Iterates over modulation formats, OSNR values, distances, symbol
    rates, and launch powers with multiple noise realisations.

    Returns:
        histograms, osnr_labels, mfi_labels (numpy arrays).
    """
    ds = cfg["dataset"]
    wdm = cfg["wdm"]

    rng = np.random.default_rng(ds["seed"])

    osnr_lo, osnr_hi = ds["osnr_range"]
    osnr_step = ds.get("osnr_step", 2.0)
    osnr_values = np.arange(osnr_lo, osnr_hi + osnr_step / 2, osnr_step)
    distances = ds["distances_km"]
    symbol_rates = ds["symbol_rates_gbd"]
    launch_powers = ds["launch_powers_dbm"]
    n_realisations = ds["n_realisations"]
    n_symbols = ds["n_symbols"]
    n_bins = ds["n_bins"]

    histograms = []
    osnr_labels = []
    mfi_labels = []

    for mod_idx in range(len(MODULATION_FORMATS)):
        for osnr in osnr_values:
            for sr in symbol_rates:
                for dist in distances:
                    for pwr in launch_powers:
                        for _ in range(n_realisations):
                            hist = generate_amplitude_histogram(
                                mod_idx, osnr, n_symbols, n_bins,
                                sr, dist, pwr, wdm, rng,
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
    """PyTorch dataset with optional Gaussian augmentation (paper Sec. IV-E)."""

    def __init__(self, histograms, osnr, mfi, osnr_mean=0.0, osnr_std=1.0,
                 training=False, augmentation_std=0.0):
        self.histograms = torch.from_numpy(histograms).unsqueeze(1)
        osnr_norm = (osnr - osnr_mean) / osnr_std
        self.osnr = torch.from_numpy(osnr_norm.astype(np.float32)).unsqueeze(1)
        self.mfi = torch.from_numpy(mfi)
        self.osnr_mean = osnr_mean
        self.osnr_std = osnr_std
        self.training = training
        self.augmentation_std = augmentation_std

    def __len__(self):
        return len(self.osnr)

    def __getitem__(self, idx):
        hist = self.histograms[idx]
        if self.training and self.augmentation_std > 0:
            noise = torch.randn_like(hist) * self.augmentation_std
            hist = torch.clamp(hist + noise, min=0)
            hist = hist / (hist.sum() + 1e-8)
        return hist, self.osnr[idx], self.mfi[idx]


def create_dataloaders(cfg):
    """Create train/val/test DataLoaders from configuration.

    Returns:
        (train_loader, val_loader, test_loader, osnr_stats).
    """
    ds_cfg = cfg["dataset"]
    histograms, osnr, mfi = build_dataset(cfg)

    # Shuffle deterministically
    rng = np.random.default_rng(ds_cfg["seed"])
    perm = rng.permutation(len(histograms))
    histograms, osnr, mfi = histograms[perm], osnr[perm], mfi[perm]

    n = len(histograms)
    n_train = int(n * ds_cfg["train_ratio"])
    n_val = int(n * ds_cfg["val_ratio"])

    osnr_mean = float(osnr[:n_train].mean())
    osnr_std = float(osnr[:n_train].std())
    if osnr_std < 1e-6:
        osnr_std = 1.0
    osnr_stats = {"mean": osnr_mean, "std": osnr_std}

    aug_std = ds_cfg.get("augmentation_std", 0.0)

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
        ds_obj = OPMDataset(
            h, o, m, osnr_mean, osnr_std,
            training=(name == "train"),
            augmentation_std=aug_std if name == "train" else 0.0,
        )
        loaders[name] = DataLoader(
            ds_obj,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=0,
            pin_memory=True,
        )

    return loaders["train"], loaders["val"], loaders["test"], osnr_stats
