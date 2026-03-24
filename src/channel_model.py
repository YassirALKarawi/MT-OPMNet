"""
Analytical optical fibre channel model for MT-OPMNet.

Generates normalised asynchronous amplitude histograms (AAHs) for various
modulation formats transmitted through a fibre-optic channel with:
  - ASE noise (OSNR-referenced, 0.1 nm bandwidth)
  - Self-phase modulation (SPM) via nonlinear phase rotation
  - Cross-phase modulation (XPM) from neighbouring WDM channels
  - Residual chromatic dispersion (CD) after DSP compensation
  - Laser phase noise (100 kHz linewidth)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
C_LIGHT = 3e8                   # speed of light  [m/s]
H_PLANCK = 6.626e-34            # Planck constant  [J s]
LAMBDA_C = 1550e-9              # centre wavelength [m]
REF_BW = 12.5e9                 # 0.1 nm reference bandwidth [Hz]

# Fibre parameters (standard SMF)
ALPHA_DB_KM = 0.2               # attenuation [dB/km]
ALPHA_LIN = ALPHA_DB_KM / (10 * np.log10(np.e)) / 1e3   # [1/m]
GAMMA_NL = 1.3e-3               # nonlinear coefficient [1/(W m)]
BETA2 = -21.7e-27               # GVD parameter [s^2/m]
SPAN_LENGTH_KM = 80             # EDFA span length [km]
NF_DB = 5.0                     # amplifier noise figure [dB]

# Number of WDM neighbours for XPM (each side)
N_WDM_NEIGHBOURS = 4
CH_SPACING_HZ = 50e9            # WDM channel spacing [Hz]

# DSP residual CD fraction after compensation
RESIDUAL_CD_FRACTION = 0.02     # 2 % residual CD

# Laser phase noise
LINEWIDTH_HZ = 100e3            # laser linewidth [Hz]

# AAH settings
N_BINS = 100                    # histogram bins
N_SYMBOLS = 2**14               # symbols per realisation
SAMPLES_PER_SYMBOL = 2          # oversampling factor


# ---------------------------------------------------------------------------
# QAM constellation points
# ---------------------------------------------------------------------------

def _qpsk_constellation():
    """Return normalised QPSK constellation (unit average power)."""
    pts = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    return pts


def _8qam_constellation():
    """Return normalised 8-QAM (star-8QAM) constellation."""
    inner = np.exp(1j * np.pi * np.arange(4) / 2 + 1j * np.pi / 4)
    outer = np.sqrt(3) * np.exp(1j * np.pi * np.arange(4) / 2)
    pts = np.concatenate([inner, outer])
    pts /= np.sqrt(np.mean(np.abs(pts) ** 2))
    return pts


def _16qam_constellation():
    """Return normalised 16-QAM constellation."""
    x = np.array([-3, -1, 1, 3])
    grid = (x[:, None] + 1j * x[None, :]).ravel()
    grid /= np.sqrt(np.mean(np.abs(grid) ** 2))
    return grid


def _32qam_constellation():
    """Return normalised cross-32QAM constellation.

    Uses the 6x6 grid {-5,-3,-1,1,3,5}^2 minus the 4 corner points,
    giving exactly 32 points with four-fold symmetry.
    """
    x = np.array([-5, -3, -1, 1, 3, 5])
    grid = (x[:, None] + 1j * x[None, :]).ravel()  # 36 points
    # Remove the 4 corners: (+-5, +-5)
    corners = {5+5j, 5-5j, -5+5j, -5-5j}
    pts = np.array([p for p in grid if p not in corners])  # 32 points
    pts /= np.sqrt(np.mean(np.abs(pts) ** 2))
    return pts


def _64qam_constellation():
    """Return normalised 64-QAM constellation."""
    x = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    grid = (x[:, None] + 1j * x[None, :]).ravel()
    grid /= np.sqrt(np.mean(np.abs(grid) ** 2))
    return grid


CONSTELLATIONS = {
    "QPSK": _qpsk_constellation,
    "8QAM": _8qam_constellation,
    "16QAM": _16qam_constellation,
    "32QAM": _32qam_constellation,
    "64QAM": _64qam_constellation,
}

MOD_FORMATS = list(CONSTELLATIONS.keys())


# ---------------------------------------------------------------------------
# Channel impairments
# ---------------------------------------------------------------------------

def _add_ase_noise(signal, osnr_db, symbol_rate, rng):
    """Add circular-symmetric ASE noise to achieve target OSNR.

    OSNR is defined in a 0.1 nm (12.5 GHz) reference bandwidth.
    """
    sig_power = np.mean(np.abs(signal) ** 2)
    osnr_lin = 10 ** (osnr_db / 10)
    # Noise in signal bandwidth = P_sig / OSNR * (Rs / B_ref)
    noise_power = sig_power / osnr_lin * (symbol_rate / REF_BW)
    noise_power = max(noise_power, 0.0)
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
    )
    return signal + noise


def _apply_spm(signal, launch_power_dbm, distance_km):
    """Apply nonlinear phase rotation from self-phase modulation."""
    p_launch = 1e-3 * 10 ** (launch_power_dbm / 10)  # [W]
    n_spans = max(1, int(np.round(distance_km / SPAN_LENGTH_KM)))
    l_eff = (1 - np.exp(-ALPHA_LIN * SPAN_LENGTH_KM * 1e3)) / ALPHA_LIN  # [m]
    # Normalise signal to have mean power = launch power
    sig_power = np.mean(np.abs(signal) ** 2)
    if sig_power > 0:
        signal = signal * np.sqrt(p_launch / sig_power)
    phi_nl = GAMMA_NL * l_eff * np.abs(signal) ** 2 * n_spans
    return signal * np.exp(1j * phi_nl)


def _apply_xpm(signal, launch_power_dbm, distance_km, symbol_rate, rng):
    """Approximate XPM from neighbouring WDM channels as additive phase noise."""
    p_launch = 1e-3 * 10 ** (launch_power_dbm / 10)
    n_spans = max(1, int(np.round(distance_km / SPAN_LENGTH_KM)))
    l_eff = (1 - np.exp(-ALPHA_LIN * SPAN_LENGTH_KM * 1e3)) / ALPHA_LIN

    # XPM phase variance from N_WDM_NEIGHBOURS on each side
    # Walk-off reduces XPM efficiency for distant channels
    phi_xpm_var = 0.0
    for k in range(1, N_WDM_NEIGHBOURS + 1):
        delta_f = k * CH_SPACING_HZ
        # L_walk = T_symbol / (|beta2| * 2*pi * delta_f)
        walk_off_length = 1.0 / (abs(BETA2) * 2 * np.pi * delta_f * symbol_rate)
        l_eff_xpm = min(l_eff, walk_off_length)
        phi_xpm_var += (2 * GAMMA_NL * p_launch * l_eff_xpm) ** 2
    phi_xpm_var *= 2 * n_spans  # both sides, accumulated over spans

    phi_xpm = rng.normal(0, np.sqrt(phi_xpm_var), len(signal))
    return signal * np.exp(1j * phi_xpm)


def _apply_residual_cd(signal, distance_km, symbol_rate):
    """Apply residual chromatic dispersion after DSP compensation.

    Models the frequency-domain phase from residual (uncompensated) GVD.
    """
    n_samples = len(signal)
    dt = 1.0 / (symbol_rate * SAMPLES_PER_SYMBOL)
    freqs = np.fft.fftfreq(n_samples, d=dt)

    residual_beta2_length = BETA2 * distance_km * 1e3 * RESIDUAL_CD_FRACTION
    H_cd = np.exp(-1j * 0.5 * residual_beta2_length * (2 * np.pi * freqs) ** 2)

    return np.fft.ifft(np.fft.fft(signal) * H_cd)


def _apply_phase_noise(signal, symbol_rate, rng):
    """Apply laser phase noise (Wiener process) to the signal."""
    dt = 1.0 / (symbol_rate * SAMPLES_PER_SYMBOL)
    phase_var = 2 * np.pi * LINEWIDTH_HZ * dt
    phase_increments = rng.normal(0, np.sqrt(phase_var), len(signal))
    phase_walk = np.cumsum(phase_increments)
    return signal * np.exp(1j * phase_walk)


# ---------------------------------------------------------------------------
# AAH generation
# ---------------------------------------------------------------------------

def generate_signal(mod_format, n_symbols=N_SYMBOLS, rng=None):
    """Generate a random baseband QAM signal with oversampling.

    Parameters
    ----------
    mod_format : str
        One of 'QPSK', '8QAM', '16QAM', '32QAM', '64QAM'.
    n_symbols : int
        Number of symbols.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    signal : ndarray, complex
        Oversampled baseband signal (length = n_symbols * SAMPLES_PER_SYMBOL).
    """
    if rng is None:
        rng = np.random.default_rng()
    constellation = CONSTELLATIONS[mod_format]()
    indices = rng.integers(0, len(constellation), size=n_symbols)
    symbols = constellation[indices]

    # Simple rectangular-pulse oversampling
    signal = np.repeat(symbols, SAMPLES_PER_SYMBOL)
    return signal


def apply_channel(signal, osnr_db, symbol_rate, distance_km,
                  launch_power_dbm, rng=None):
    """Pass signal through the analytical fibre channel model.

    Parameters
    ----------
    signal : ndarray, complex
        Baseband signal.
    osnr_db : float
        Target OSNR in dB (0.1 nm reference bandwidth).
    symbol_rate : float
        Symbol rate in Hz (e.g. 28e9).
    distance_km : float
        Transmission distance in km.
    launch_power_dbm : float
        Per-channel launch power in dBm.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    signal : ndarray, complex
        Channel-impaired signal.
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1) SPM (nonlinear phase rotation)
    signal = _apply_spm(signal, launch_power_dbm, distance_km)

    # 2) XPM from WDM neighbours
    signal = _apply_xpm(signal, launch_power_dbm, distance_km, symbol_rate, rng)

    # 3) Residual chromatic dispersion
    signal = _apply_residual_cd(signal, distance_km, symbol_rate)

    # 4) Laser phase noise
    signal = _apply_phase_noise(signal, symbol_rate, rng)

    # 5) ASE noise (added last to match OSNR definition)
    signal = _add_ase_noise(signal, osnr_db, symbol_rate, rng)

    return signal


def compute_aah(signal, n_bins=N_BINS):
    """Compute normalised asynchronous amplitude histogram.

    Parameters
    ----------
    signal : ndarray, complex
        Channel-impaired signal.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    hist : ndarray, float32, shape (n_bins,)
        Normalised amplitude histogram (sums to 1).
    """
    amplitudes = np.abs(signal)

    # Fixed bin range [0, max_amp] to keep histograms comparable
    max_amp = np.percentile(amplitudes, 99.5)
    if max_amp < 1e-12:
        max_amp = 1.0

    hist, _ = np.histogram(amplitudes, bins=n_bins, range=(0, max_amp))
    hist = hist.astype(np.float32)

    # Normalise to unit area
    total = hist.sum()
    if total > 0:
        hist /= total

    return hist


def generate_aah(mod_format, osnr_db, symbol_rate, distance_km,
                 launch_power_dbm, n_symbols=N_SYMBOLS, n_bins=N_BINS,
                 seed=None):
    """End-to-end: generate signal, apply channel, return normalised AAH.

    Parameters
    ----------
    mod_format : str
        Modulation format name.
    osnr_db : float
        Target OSNR [dB].
    symbol_rate : float
        Symbol rate [Hz].
    distance_km : float
        Distance [km].
    launch_power_dbm : float
        Launch power [dBm].
    n_symbols : int
        Number of symbols per realisation.
    n_bins : int
        AAH bins.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    aah : ndarray, float32, shape (n_bins,)
    """
    rng = np.random.default_rng(seed)
    signal = generate_signal(mod_format, n_symbols, rng)
    signal = apply_channel(signal, osnr_db, symbol_rate, distance_km,
                           launch_power_dbm, rng)
    return compute_aah(signal, n_bins)
