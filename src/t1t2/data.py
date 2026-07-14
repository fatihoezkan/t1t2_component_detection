"""From voxel_generator Parquet files to the exact tensors the model consumes.

The generator writes one voxel per row: the 64-point signal (S_1..S_64), up to three
compartments (T1_i, T2_i, w_i, NaN-padded when a voxel has fewer), and n_comp. Training,
on the other hand, wants three tensors per batch — the input signal, a flat target vector,
and the compartment count. This module is the glue, and it holds two decisions that are
easy to get subtly wrong:

1.  Relaxation times are mapped into [0, 1] before the model sees them, because the model's
    T1/T2/weight heads end in a sigmoid and can only ever emit [0, 1]. `TargetNormalizer`
    does that mapping (and its exact inverse, so predictions can be read back in ms).

2.  Empty compartment slots are padded with **zeros, not NaN**. NaN is the honest thing to
    store on disk, but it would poison gradients. Zero-filling is safe *because* we also pass
    n_comp: the Hungarian loss only ever looks at the first n_comp targets, so the zeros are
    never matched against anything — inert filler, not fake compartments.

The untouched millisecond targets are kept on the dataset object too, so evaluation can
report real-unit errors ("40 ms" means something; "0.03" doesn't).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

_NORM_MODES = ("identity", "linear_minmax", "log_minmax")


class TargetNormalizer:
    """Converts (T1, T2) between milliseconds and the model's [0, 1] range.

    Why log by default: brain relaxation times run from ~20 ms (myelin water) to a few
    thousand ms (CSF) — over a decade. On a linear scale the big values dominate and the
    short-T2 pools get almost no resolution; log-space spreads them evenly, matches the
    generator's log-uniform sampling, and puts the T1 and T2 loss terms on the same footing
    so the loss weights can all just be ~1.0.
    """

    def __init__(self, mode="log_minmax", t1_min=100.0, t1_max=7000.0, t2_min=5.0, t2_max=4000.0):
        if mode not in _NORM_MODES:
            raise ValueError(f"unknown normalization {mode!r}; expected one of {_NORM_MODES}")
        self.mode = mode
        self.t1_min, self.t1_max = float(t1_min), float(t1_max)
        self.t2_min, self.t2_max = float(t2_min), float(t2_max)

    @classmethod
    def from_config(cls, data_cfg) -> "TargetNormalizer":
        return cls(data_cfg.normalization, data_cfg.t1_min, data_cfg.t1_max,
                   data_cfg.t2_min, data_cfg.t2_max)

    def _fwd(self, x, lo, hi, clip):
        """ms -> [0, 1]. One implementation for T1 and T2; the caller passes the bounds."""
        x = np.asarray(x, dtype=np.float64)
        if self.mode == "identity":
            out = x
        elif self.mode == "linear_minmax":
            out = (x - lo) / (hi - lo)
        else:  # log_minmax
            out = (np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo))
        # Clamp only when building targets: a target the sigmoid could never reach is worse
        # than a clamp. Never clamp a prediction we're about to invert — that corrupts it.
        if clip and self.mode != "identity":
            out = np.clip(out, 0.0, 1.0)
        return out

    def _inv(self, y, lo, hi):
        """[0, 1] -> ms. Exact inverse of _fwd, for reading predictions back in real units."""
        y = np.asarray(y, dtype=np.float64)
        if self.mode == "identity":
            return y
        if self.mode == "linear_minmax":
            return y * (hi - lo) + lo
        return np.exp(y * (np.log(hi) - np.log(lo)) + np.log(lo))

    def normalize_t1(self, t1, clip=True):
        return self._fwd(t1, self.t1_min, self.t1_max, clip)

    def normalize_t2(self, t2, clip=True):
        return self._fwd(t2, self.t2_min, self.t2_max, clip)

    def denormalize_t1(self, x):
        return self._inv(x, self.t1_min, self.t1_max)

    def denormalize_t2(self, x):
        return self._inv(x, self.t2_min, self.t2_max)


def _signal_columns(n_inputs: int) -> list[str]:
    """Signal column names in order. Order matters — column p must always mean the same
    (TI_p, TE_p), at training time and later at inference on a real scan."""
    return [f"S_{i + 1}" for i in range(n_inputs)]


def _apply_signal_norm(X: np.ndarray, mode: str) -> np.ndarray:
    """Optional per-voxel rescaling of the input signal.

    Real scans come out at an arbitrary overall scale (receiver gain, coil), so eventually the
    network should see something scale-invariant — and it must be applied identically to
    synthetic and real data. "none" leaves the signal alone (fine for synthetic-only work),
    "max" divides each voxel by its own peak magnitude, "first" by its first sample.
    """
    if mode == "none":
        return X
    if mode == "max":
        m = np.max(np.abs(X), axis=1, keepdims=True)
        m[m == 0] = 1.0
        return (X / m).astype(np.float32)
    if mode == "first":
        f = X[:, :1].copy()
        f[f == 0] = 1.0
        return (X / f).astype(np.float32)
    raise ValueError(f"unknown signal_norm {mode!r}; expected none|max|first")


class VoxelDataset(Dataset):
    """One Parquet split loaded fully into memory as tensors.

    Loaded eagerly rather than streamed per-item — the files fit in RAM and it keeps the
    training loop fast. `limit` grabs just the first few hundred voxels for tests and smoke
    runs without needing a separate tiny file.
    """

    def __init__(self, path, cfg, normalizer: TargetNormalizer | None = None, limit: int | None = None):
        self.cfg = cfg
        self.normalizer = normalizer or TargetNormalizer.from_config(cfg)
        n_in, max_c = cfg.n_inputs, cfg.max_comp

        df = pd.read_parquet(path)
        if limit is not None:
            df = df.iloc[:limit].reset_index(drop=True)

        # --- input signal ---
        # copy=True because pandas can hand back a read-only view, and torch.from_numpy on a
        # non-writable buffer is a footgun (silent UB if anything ever writes to it).
        X = df[_signal_columns(n_in)].to_numpy(np.float32, copy=True)
        X = _apply_signal_norm(X, cfg.signal_norm)

        # --- targets ---
        n_comp = df["n_comp"].to_numpy(np.int64)
        t1 = np.stack([df[f"T1_{i + 1}"].to_numpy(np.float64) for i in range(max_c)], axis=1)
        t2 = np.stack([df[f"T2_{i + 1}"].to_numpy(np.float64) for i in range(max_c)], axis=1)
        w = np.stack([df[f"w_{i + 1}"].to_numpy(np.float64) for i in range(max_c)], axis=1)

        t1n = self.normalizer.normalize_t1(t1)
        t2n = self.normalizer.normalize_t2(t2)
        target = np.stack([t1n, t2n, w], axis=2)            # (N, max_comp, 3)
        target = np.nan_to_num(target, nan=0.0)             # empty slots -> inert zeros
        y = target.reshape(len(df), max_c * 3).astype(np.float32)

        # keep raw ms targets untouched so eval can report real-unit errors
        self.raw_t1, self.raw_t2, self.raw_w = t1, t2, w

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.n_comp = torch.from_numpy(n_comp.astype(np.int16))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.n_comp[i]


def make_dataloader(path, cfg, batch_size, shuffle, normalizer=None, num_workers=0, limit=None):
    """Build a DataLoader and return it alongside its dataset.

    The eval code reaches into the dataset's raw-ms targets, so handing back both saves the
    caller from constructing the dataset twice.
    """
    ds = VoxelDataset(path, cfg, normalizer, limit=limit)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader, ds
