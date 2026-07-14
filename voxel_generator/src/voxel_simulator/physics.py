"""IR-MSE forward model: clean 64-point signal from a list of compartments."""

from __future__ import annotations

import numpy as np

from .protocol import Protocol


def simulate_clean_signal(
    protocol: Protocol,
    t1: np.ndarray,
    t2: np.ndarray,
    w: np.ndarray,
    m0: float = 1.0,
) -> np.ndarray:
    """
    Compute the clean (noise-free) 64-point voxel signal.

    S_p = M0 * sum_c w_c * (1 - 2 exp(-TI_p/T1_c) + exp(-TR/T1_c)) * exp(-TE_p/T2_c)

    Parameters
    ----------
    protocol : Protocol
        Fixed scanner protocol (TI, TE, TR).
    t1, t2 : (K,) arrays
        Compartment relaxation times in ms. K = number of compartments.
    w : (K,) array
        Compartment weights. Must be nonneg and sum to ~1.
    m0 : float
        Overall amplitude (default 1.0).

    Returns
    -------
    signal : (64,) array
        Clean signed signal at each protocol point.
    """
    t1 = np.asarray(t1, dtype=np.float64).flatten()
    t2 = np.asarray(t2, dtype=np.float64).flatten()
    w = np.asarray(w, dtype=np.float64).flatten()

    if not (t1.shape == t2.shape == w.shape):
        raise ValueError(f"t1, t2, w must have same shape; got {t1.shape}, {t2.shape}, {w.shape}")
    if np.any(t1 <= 0) or np.any(t2 <= 0):
        raise ValueError("T1 and T2 must be strictly positive (ms).")
    if np.any(w < 0):
        raise ValueError("Weights must be nonnegative.")
    if not np.isclose(w.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Weights must sum to 1; got {w.sum():.6f}")

    ti = protocol.ti[:, None]   # (64, 1)
    te = protocol.te[:, None]   # (64, 1)
    tr = protocol.tr

    t1_row = t1[None, :]        # (1, K)
    t2_row = t2[None, :]        # (1, K)

    inv_recovery = 1.0 - 2.0 * np.exp(-ti / t1_row) + np.exp(-tr / t1_row)   # (64, K)
    t2_decay     = np.exp(-te / t2_row)                                      # (64, K)
    per_comp     = inv_recovery * t2_decay                                   # (64, K)

    signal = m0 * (per_comp * w[None, :]).sum(axis=1)                        # (64,)
    return signal
