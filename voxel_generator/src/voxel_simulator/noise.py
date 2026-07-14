"""Gaussian noise for signed T1-T2 signals.

The signal we work with is **real-valued (signed)**: the inversion-recovery curve genuinely
goes negative right after the inversion pulse, and the data we're matching keeps those
negatives (it's not a magnitude image). So the correct noise model here is plain **additive
Gaussian** — it leaves the sign alone.

Two ways to set the noise level, and each function returns `(noisy_signal, sigma)`:
  - **SNR** (relative): `sigma = max(|S_clean|) / SNR`.
  - **sigma** (absolute): pass `sigma=` directly (e.g. 0.1, 0.2). Since the clean signal has
    amplitude ~1 (M0=1), a direct sigma is the intuitive "how much noise" knob. When `sigma`
    is given it wins and `snr` is ignored.
"""
from __future__ import annotations

import numpy as np


def _sigma_from_snr(signal_clean: np.ndarray, snr: float) -> float:
    """Per-channel noise std implied by the SNR, relative to the peak clean signal."""
    if snr <= 0:
        raise ValueError("SNR must be positive.")
    peak = float(np.max(np.abs(signal_clean)))
    if peak == 0:
        raise ValueError("Clean signal is all zeros; cannot scale noise.")
    return peak / snr


def add_gaussian_noise(signal_clean, snr, rng, sigma=None):
    """Additive Gaussian straight onto the (signed) signal — the model we actually use.

    Negatives stay negative (no rectification), which is right for real-valued data. Pass
    `sigma` to set the noise std directly (e.g. 0.1), otherwise it's derived from `snr`.
    """
    if sigma is None:
        sigma = _sigma_from_snr(signal_clean, snr)
    signal_noisy = signal_clean + rng.normal(0.0, sigma, size=signal_clean.shape)
    return signal_noisy, sigma


add_noise = add_gaussian_noise
