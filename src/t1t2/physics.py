"""The forward model — turning compartments back into a signal.

This is the same IR-MSE physics the generator uses to make the data, re-implemented here on
the training side for two jobs:

  1.  a future **signal-consistency loss** needs a *differentiable* (torch) forward so we can
      resynthesize the signal from a prediction and backprop the mismatch;
  2.  parity tests — the numpy forward here must match the generator bit-for-bit, so training
      targets and the physics speak exactly the same language.

The equation, per protocol point p = (TI_p, TE_p):

    S_p = M0 · Σ_c  w_c · (1 − 2·exp(−TI_p/T1_c) + exp(−TR/T1_c)) · exp(−TE_p/T2_c)
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import scipy.io as sio

# The vendored protocol ships with the training folder, so this default keeps everything
# self-contained (no reach back into voxel_simulator/).
_DEFAULT_MAT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "voxel_generator", "data", "ti_te_dict.mat")
)


@dataclass(frozen=True)
class Protocol:
    """Frozen scanner protocol: 64 ordered (TI, TE) pairs + a single TR (ms)."""

    ti: np.ndarray
    te: np.ndarray
    tr: float

    @property
    def n_points(self) -> int:
        return int(self.ti.shape[0])


def load_protocol(mat_path: str | None = None) -> Protocol:
    """Load TI, TE, TR from ti_te_dict.mat, preserving the scanner's acquisition order."""
    d = sio.loadmat(mat_path or _DEFAULT_MAT)
    ti = np.asarray(d["ti"], dtype=np.float64).flatten()
    te = np.asarray(d["te"], dtype=np.float64).flatten()
    tr = float(np.asarray(d["tr"]).flatten()[0])
    if ti.shape != te.shape:
        raise ValueError(f"TI/TE shape mismatch: {ti.shape} vs {te.shape}")
    return Protocol(ti=ti, te=te, tr=tr)


def forward_numpy(protocol: Protocol, t1, t2, w, m0: float = 1.0) -> np.ndarray:
    """Clean 64-point signal for one voxel's compartments (numpy). Matches the generator."""
    t1 = np.asarray(t1, np.float64).ravel()
    t2 = np.asarray(t2, np.float64).ravel()
    w = np.asarray(w, np.float64).ravel()
    ti = protocol.ti[:, None]                                  # (P, 1)
    te = protocol.te[:, None]
    inv = 1.0 - 2.0 * np.exp(-ti / t1[None, :]) + np.exp(-protocol.tr / t1[None, :])   # (P, K)
    dec = np.exp(-te / t2[None, :])                            # (P, K)
    return m0 * ((inv * dec) * w[None, :]).sum(axis=1)         # (P,)


def forward_torch(protocol: Protocol, params, mask=None, m0: float = 1.0):
    """Differentiable batched forward: params (B, K, 3) = [T1_ms, T2_ms, weight] -> (B, P).

    `mask` (B, K) optionally zeroes padded/absent compartments. Written for the later
    signal-consistency loss; unused by the baseline pipeline for now.
    """
    import torch

    dev = params.device
    ti = torch.as_tensor(protocol.ti, dtype=params.dtype, device=dev)   # (P,)
    te = torch.as_tensor(protocol.te, dtype=params.dtype, device=dev)
    tr = float(protocol.tr)
    t1 = params[..., 0].clamp(min=1e-6).unsqueeze(1)           # (B, 1, K)
    t2 = params[..., 1].clamp(min=1e-6).unsqueeze(1)
    w = params[..., 2].unsqueeze(1)                            # (B, 1, K)
    if mask is not None:
        w = w * mask.unsqueeze(1)
    ti = ti.view(1, -1, 1)                                     # (1, P, 1)
    te = te.view(1, -1, 1)
    inv = 1.0 - 2.0 * torch.exp(-ti / t1) + torch.exp(-tr / t1)   # (B, P, K)
    dec = torch.exp(-te / t2)
    return m0 * (inv * dec * w).sum(dim=-1)                    # (B, P)
