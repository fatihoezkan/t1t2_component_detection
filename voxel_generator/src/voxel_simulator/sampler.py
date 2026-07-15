"""Per-voxel random sampling: the compartments' random (T1, T2), their weights, the SNR.

Tissue-agnostic on purpose. A compartment is just a random (T1, T2) point with T1 > T2 —
no tissue prototypes, no realism. The job is compartment detection under noise, not tissue
simulation, so the data spreads compartments across the whole feasible (T1, T2) region.

**The compartment count is an input, not a draw.** Each dataset file is generated with one fixed
`n_comp` (see `MAX_COMP`), which is what makes the per-n splits exactly balanced. It also has to
work this way: the RNG streams below are keyed *on* `n_comp`, so a voxel whose count came out of
its own stream would be circular.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The most compartments a voxel may have. This is the single source of truth: it fixes the parquet
# schema width (T1/T2/w_1..MAX_COMP) and the valid range of --n-comp.
MAX_COMP = 4

# SNR window (uniform) for the random-SNR splits.
SNR_MIN = 30.0
SNR_MAX = 150.0

# Smallest weight worth calling a compartment (below this it's invisible in the signal).
MIN_WEIGHT = 0.05

# Random (T1, T2) ranges in ms. t1_min stays above t2_min so T2 < T1 always has room.
T1_RANGE = (50.0, 4000.0)
T2_RANGE = (5.0, 3000.0)

# Which split a voxel belongs to. Part of every RNG stream key, so the splits cannot collide
# however large they get — unlike an arithmetic seed, where the sizes eventually overlap.
SPLIT_TRAIN = 0
SPLIT_VAL = 1
SPLIT_TEST = 2
SPLIT_SNR_LADDER = 3

# Independent RNG streams per voxel. Separating them is what makes the fixed-SNR ladder a
# *paired* comparison: the ladder reuses the parameter and noise streams unchanged and never
# touches the SNR stream, so every rung gets the same voxel and the same standardized noise,
# and only the amplitude differs.
STREAM_PARAMS = 1001
STREAM_NOISE = 2001
STREAM_SNR = 3001


@dataclass
class VoxelSpec:
    """Ground-truth description of one voxel."""
    n_comp: int
    snr: float
    t1: np.ndarray          # (n_comp,)
    t2: np.ndarray          # (n_comp,)
    w: np.ndarray           # (n_comp,)


def voxel_rng(
    base_seed: int,
    n_comp: int,
    split_code: int,
    voxel_id: int,
    stream_id: int,
) -> np.random.Generator:
    """The RNG for one (voxel, stream), derived from integer entropy only.

    Using SeedSequence rather than arithmetic means two different keys cannot land on the same
    state by construction — no matter how many voxels a split has. Reproducible for a pinned
    environment; NumPy does not promise Generator streams across versions.
    """
    return np.random.default_rng(
        np.random.SeedSequence([int(base_seed), int(n_comp), int(split_code), int(voxel_id), int(stream_id)])
    )


def validate_ranges(
    t1_range: tuple[float, float] = T1_RANGE,
    t2_range: tuple[float, float] = T2_RANGE,
) -> None:
    """Reject impossible (T1, T2) ranges up front.

    Called before generation starts so a bad range fails in the first second rather than partway
    through a million voxels.
    """
    lo1, hi1 = t1_range
    lo2, hi2 = t2_range
    if not (0 < lo1 < hi1):
        raise ValueError(f"t1_range must satisfy 0 < min < max; got {t1_range}")
    if not (0 < lo2 < hi2):
        raise ValueError(f"t2_range must satisfy 0 < min < max; got {t2_range}")
    if lo2 >= hi1:
        raise ValueError(
            f"t2_min ({lo2}) >= t1_max ({hi1}): no (T1, T2) with T2 < T1 exists in these ranges."
        )


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
    max_tries: int = 1000,
) -> tuple[float, float]:
    """One compartment as a random (T1, T2) with the single physical tie T1 > T2.

    T1 and T2 are drawn log-uniform from their *own* ranges and the pair is kept only if T2 < T1
    (relaxation times span decades, so linear would waste resolution). The result is uniform over
    the feasible log-space region {(log T1, log T2) : T2 < T1} — so note that **neither marginal
    is log-uniform**: the log-T1 density is proportional to the feasible log-T2 width at that T1.

    Capping T2 at T1 instead (the obvious shortcut) would be uniform in log-T1 with a uniform
    conditional log-T2, which oversamples short T1 — at these ranges T1 ∈ [50,100] came out 1.7×
    too dense. Rejection costs ~1.4 draws per compartment and is negligible next to the forward
    model and the parquet write.
    """
    validate_ranges(t1_range, t2_range)
    lo1, hi1 = t1_range
    lo2, hi2 = t2_range
    log1, log2 = (np.log(lo1), np.log(hi1)), (np.log(lo2), np.log(hi2))
    for _ in range(max_tries):
        t1 = float(np.exp(rng.uniform(*log1)))
        t2 = float(np.exp(rng.uniform(*log2)))
        if t2 < t1:
            return t1, t2
    raise RuntimeError(
        f"rejection sampling found no T2 < T1 in {max_tries} tries for T1{t1_range}, T2{t2_range}; "
        "the feasible region is too small for these ranges."
    )


def sample_voxel_spec(
    voxel_id: int,
    n_comp: int,
    base_seed: int = 0,
    split_code: int = SPLIT_TRAIN,
    snr_min: float = SNR_MIN,
    snr_max: float = SNR_MAX,
    snr: float | None = None,
    t1_range: tuple[float, float] = T1_RANGE,
    t2_range: tuple[float, float] = T2_RANGE,
) -> VoxelSpec:
    """Draw the ground truth for one voxel: `n_comp` random (T1, T2) compartments, weights, SNR.

    `n_comp` is given, never drawn (see the module docstring). Pass `snr` to pin it — the fixed-SNR
    ladder does that, and because the SNR lives in its own stream, pinning it leaves the parameter
    draw byte-for-byte identical to the un-pinned one.
    """
    if not 1 <= n_comp <= MAX_COMP:
        raise ValueError(f"n_comp must be in 1..{MAX_COMP}; got {n_comp}")

    rng = voxel_rng(base_seed, n_comp, split_code, voxel_id, STREAM_PARAMS)

    t1 = np.empty(n_comp)
    t2 = np.empty(n_comp)
    for i in range(n_comp):
        t1[i], t2[i] = sample_random_compartment(rng, t1_range, t2_range)

    w = sample_weights(n_comp, rng)

    if snr is None:
        snr_rng = voxel_rng(base_seed, n_comp, split_code, voxel_id, STREAM_SNR)
        snr = float(snr_rng.uniform(snr_min, snr_max))

    return VoxelSpec(n_comp=n_comp, snr=float(snr), t1=t1, t2=t2, w=w)
