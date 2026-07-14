"""Per-voxel random sampling: how many compartments, their random (T1, T2), weights, SNR.

Tissue-agnostic on purpose. A compartment is just a random (T1, T2) point with T1 > T2 —
no tissue prototypes, no realism. The job is compartment detection under noise, not tissue
simulation, so the data spreads compartments across the whole (T1, T2) plane.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Per-voxel compartment-count distribution. The length is the MAX number of compartments,
# and each entry is the probability of that count: (0.2, 0.6, 0.2) => P(1)=0.2, P(2)=0.6,
# P(3)=0.2. Pass a longer tuple to allow more, e.g. (0.2, 0.5, 0.2, 0.1) allows up to 4.
N_COMP_PROBS = (0.20, 0.60, 0.20)

# SNR window (uniform).
SNR_MIN = 30.0
SNR_MAX = 150.0

# Smallest weight worth calling a compartment (below this it's invisible in the signal).
MIN_WEIGHT = 0.05

# Random (T1, T2) ranges in ms. t1_min stays above t2_min so T2 < T1 always has room.
T1_RANGE = (50.0, 4000.0)
T2_RANGE = (5.0, 3000.0)


@dataclass
class VoxelSpec:
    """Ground-truth description of one voxel."""
    n_comp: int
    snr: float
    t1: np.ndarray          # (n_comp,)
    t2: np.ndarray          # (n_comp,)
    w: np.ndarray           # (n_comp,)


def voxel_seed(master_seed: int, voxel_id: int) -> int:
    """Deterministic per-voxel seed. We don't store it — it just keeps the train/val/test
    splits disjoint (each split uses a different master seed → non-overlapping voxels)."""
    return int((np.uint64(master_seed) * np.uint64(1_000_003) + np.uint64(voxel_id)) % np.uint64(2**32 - 1))


def sample_weights(n_comp: int, rng: np.random.Generator, min_weight: float = MIN_WEIGHT) -> np.ndarray:
    """Weights that sum to 1, each at least min_weight (symmetric Dirichlet, rescaled)."""
    if n_comp * min_weight >= 1.0:
        raise ValueError(f"n_comp * min_weight = {n_comp * min_weight} >= 1.")
    raw = rng.dirichlet(np.ones(n_comp))
    return raw * (1.0 - n_comp * min_weight) + min_weight


def sample_random_compartment(
    rng: np.random.Generator,
    t1_range: tuple[float, float] = T1_RANGE,
    t2_range: tuple[float, float] = T2_RANGE,
) -> tuple[float, float]:
    """One compartment as a random (T1, T2) with the single physical tie T1 > T2.

    Log-uniform in both (relaxation times span a decade, so linear would waste resolution).
    T2's upper bound is capped at T1, which is what makes T2 < T1 hold for free.
    """
    lo1, hi1 = t1_range
    t1 = float(np.exp(rng.uniform(np.log(lo1), np.log(hi1))))
    lo2, hi2 = t2_range
    hi2 = min(hi2, t1)                       # enforce T2 < T1
    t2 = float(np.exp(rng.uniform(np.log(lo2), np.log(hi2))))
    return t1, t2


def sample_voxel_spec(
    voxel_id: int,
    master_seed: int = 0,
    snr_min: float = SNR_MIN,
    snr_max: float = SNR_MAX,
    t1_range: tuple[float, float] = T1_RANGE,
    t2_range: tuple[float, float] = T2_RANGE,
    n_comp_probs=N_COMP_PROBS,
) -> VoxelSpec:
    """Draw a full ground-truth spec for one voxel: n_comp random (T1, T2) compartments
    (each T1 > T2), their weights, and an SNR.

    n_comp_probs is the compartment-count distribution (see the constant): its length caps
    the number of compartments and its values are the per-count probabilities.
    """
    rng = np.random.default_rng(voxel_seed(master_seed, voxel_id))

    probs = np.asarray(n_comp_probs, dtype=float)
    if not np.isclose(probs.sum(), 1.0):
        raise ValueError(f"n_comp_probs must sum to 1; got {probs.sum():.4f}")
    n_comp = int(rng.choice(np.arange(1, len(probs) + 1), p=probs))
    t1 = np.empty(n_comp)
    t2 = np.empty(n_comp)
    for i in range(n_comp):
        t1[i], t2[i] = sample_random_compartment(rng, t1_range, t2_range)

    w = sample_weights(n_comp, rng)
    snr = float(rng.uniform(snr_min, snr_max))

    return VoxelSpec(n_comp=n_comp, snr=snr, t1=t1, t2=t2, w=w)
