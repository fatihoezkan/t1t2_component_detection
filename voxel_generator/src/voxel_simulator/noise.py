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

The noise is drawn as a **standardized** z and then scaled: `S_clean + sigma * z`. That split is
load-bearing, not cosmetic — it is what lets the fixed-SNR ladder hand the same voxel the same z
at every rung so that only the amplitude changes, making SNR the single controlled variable.
Writing it as `rng.normal(0, sigma)` happens to produce the same numbers today, but only because
of how NumPy internally scales a shared standard-normal draw; that is an implementation detail
NumPy does not promise across versions, and the ladder's whole design would rest on it.
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

    Draws a standardized z and scales it (see the module docstring): with the same `rng` state,
    two different sigmas reuse the same z, which is exactly what the paired fixed-SNR ladder needs.
    """
    if sigma is None:
        sigma = _sigma_from_snr(signal_clean, snr)
    z = rng.standard_normal(signal_clean.shape)
    signal_noisy = signal_clean + sigma * z
    return signal_noisy, sigma


add_noise = add_gaussian_noise
