"""Synthetic voxel data generation — the single generation module.

The API layers cleanly, smallest unit first:

    generate_voxel         one voxel   -> target spec + observed signal
    generate_one            one voxel   -> one dict (a dataset row)
    generate_dataset        N voxels    -> a DataFrame
    generate_dataset_family a config    -> train/val/test + fixed-SNR parquet files

There is no CLI in here on purpose: the one command-line entry point lives in
`run_generator.py` at the repo root, so the library stays import-only and easy to test.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .noise import add_noise
from .physics import simulate_clean_signal
from .protocol import Protocol, load_protocol
from .sampler import SNR_MAX, SNR_MIN, T1_RANGE, T2_RANGE, VoxelSpec, sample_voxel_spec

MAX_COMP = 3   # fixed table width for the ground-truth columns


@dataclass(frozen=True)
class GeneratedVoxel:
    """A fully simulated voxel before it is flattened into a dataset row."""

    voxel_id: int
    spec: VoxelSpec
    signal: np.ndarray
    sigma: float


def _noise_rng(master_seed: int, voxel_id: int) -> np.random.Generator:
    """Independent deterministic RNG stream for measurement noise."""
    noise_ss = np.random.SeedSequence([int(master_seed), int(voxel_id), 7919])
    return np.random.default_rng(noise_ss)


def simulate_voxel_signal(
    protocol: Protocol,
    spec: VoxelSpec,
    master_seed: int,
    voxel_id: int,
    noise_sigma: float | None = None,
) -> tuple[np.ndarray, float]:
    """Forward + measurement model: target compartments -> model input signal."""
    clean = simulate_clean_signal(protocol, spec.t1, spec.t2, spec.w)
    return add_noise(clean, spec.snr, _noise_rng(master_seed, voxel_id), sigma=noise_sigma)


def generate_voxel(
    voxel_id: int,
    master_seed: int,
    protocol: Protocol,
    snr_min: float = SNR_MIN,
    snr_max: float = SNR_MAX,
    t1_range: tuple[float, float] = T1_RANGE,
    t2_range: tuple[float, float] = T2_RANGE,
    noise_sigma: float | None = None,
) -> GeneratedVoxel:
    """Generate the physical contents and measured signal for one voxel.

    The readable pipeline is:

        sample (T1, T2, w) targets
        -> simulate the 64-point TI/TE measurement
        -> return target + model input signal
    """
    spec = sample_voxel_spec(voxel_id, master_seed, snr_min, snr_max, t1_range, t2_range)
    signal, sigma = simulate_voxel_signal(protocol, spec, master_seed, voxel_id, noise_sigma)
    return GeneratedVoxel(voxel_id=voxel_id, spec=spec, signal=signal, sigma=sigma)


def voxel_to_row(voxel: GeneratedVoxel, protocol: Protocol, noise_sigma: float | None = None) -> dict:
    """Flatten a generated voxel into the stable tabular dataset schema."""
    spec = voxel.spec
    row: dict = {
        "voxel_id": voxel.voxel_id,
        "snr": np.nan if noise_sigma is not None else spec.snr,
        "sigma": voxel.sigma,
        "n_comp": spec.n_comp,
    }
    for i in range(MAX_COMP):
        if i < spec.n_comp:
            row[f"T1_{i+1}"] = float(spec.t1[i])
            row[f"T2_{i+1}"] = float(spec.t2[i])
            row[f"w_{i+1}"]  = float(spec.w[i])
        else:
            row[f"T1_{i+1}"] = np.nan
            row[f"T2_{i+1}"] = np.nan
            row[f"w_{i+1}"]  = np.nan
    for p in range(protocol.n_points):
        row[f"S_{p+1}"] = np.float32(voxel.signal[p])
    return row


def generate_one(
    voxel_id: int,
    master_seed: int,
    protocol: Protocol,
    snr_min: float = SNR_MIN,
    snr_max: float = SNR_MAX,
    t1_range: tuple[float, float] = T1_RANGE,
    t2_range: tuple[float, float] = T2_RANGE,
    noise_sigma: float | None = None,
) -> dict:
    """Build one full row of the dataset.

    noise_sigma sets the absolute Gaussian noise std directly (e.g. 0.1); when given it wins
    and the per-voxel SNR is not used (the `snr` column is left NaN to say so).
    """
    voxel = generate_voxel(
        voxel_id,
        master_seed,
        protocol,
        snr_min,
        snr_max,
        t1_range,
        t2_range,
        noise_sigma,
    )
    return voxel_to_row(voxel, protocol, noise_sigma)

def generate_dataset(
    n_voxels: int,
    master_seed: int = 0,
    protocol: Optional[Protocol] = None,
    snr_min: float = SNR_MIN,
    snr_max: float = SNR_MAX,
    t1_range: tuple[float, float] = T1_RANGE,
    t2_range: tuple[float, float] = T2_RANGE,
    noise_sigma: float | None = None,
) -> pd.DataFrame:
    """Generate a DataFrame of `n_voxels` rows.

    Noise level is set either by SNR (snr_min/snr_max) or, if `noise_sigma` is given, by that
    absolute Gaussian std directly.
    """
    if protocol is None:
        protocol = load_protocol()
    rows = [
        generate_one(i, master_seed, protocol, snr_min, snr_max, t1_range, t2_range, noise_sigma)
        for i in range(n_voxels)
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Dataset family: turn one config into the full set of train/val/test + fixed-SNR files.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetFamilyConfig:
    """Settings for train/val/test plus fixed-SNR evaluation datasets."""

    out_dir: str | Path = "output/data"
    n_train: int = 1_000_000
    n_val: int = 100_000
    n_test: int = 100_000
    n_per_snr: int = 50_000
    snr_min: float = 20.0
    snr_max: float = 150.0
    snr_ladder: tuple[float, ...] = (20, 40, 60, 100, 150)
    t1_range: tuple[float, float] = T1_RANGE
    t2_range: tuple[float, float] = T2_RANGE
    # Absolute-noise mode: if set, the whole family uses Gaussian noise of this std (SNR is
    # ignored) and the robustness test sets come from sigma_ladder instead of snr_ladder.
    noise_sigma: float | None = None
    sigma_ladder: tuple[float, ...] = (0.05, 0.1, 0.2)

    def __post_init__(self) -> None:
        if min(self.n_train, self.n_val, self.n_test, self.n_per_snr) < 0:
            raise ValueError("dataset sizes must be nonnegative")
        if self.noise_sigma is not None:
            if self.noise_sigma <= 0 or any(s <= 0 for s in self.sigma_ladder):
                raise ValueError("sigma values must be positive")
        else:
            if self.snr_min <= 0 or self.snr_max <= 0:
                raise ValueError("SNR bounds must be positive")
            if self.snr_min > self.snr_max:
                raise ValueError("snr_min must be <= snr_max")
            if any(snr <= 0 for snr in self.snr_ladder):
                raise ValueError("all fixed SNR values must be positive")


@dataclass(frozen=True)
class DatasetJob:
    name: str
    n_voxels: int
    master_seed: int
    snr_min: float = SNR_MIN
    snr_max: float = SNR_MAX
    noise_sigma: float | None = None


def smoke_config(config: DatasetFamilyConfig) -> DatasetFamilyConfig:
    """Return a small version of a config for quick local checks."""
    return replace(config, n_train=2_000, n_val=500, n_test=500, n_per_snr=300)


def build_dataset_jobs(config: DatasetFamilyConfig) -> list[DatasetJob]:
    """Create the deterministic split jobs from one config.

    The disjoint master seeds (0, 10, 20, 30, 40, ...) are what make the splits
    leakage-free: every split draws voxels from a non-overlapping seed range.
    """
    if config.noise_sigma is not None:      # absolute-noise mode
        s = config.noise_sigma
        jobs = [
            DatasetJob("train", config.n_train, 0, noise_sigma=s),
            DatasetJob("val", config.n_val, 10, noise_sigma=s),
            DatasetJob("test", config.n_test, 20, noise_sigma=s),
        ]
        for k, sig in enumerate(config.sigma_ladder):
            jobs.append(DatasetJob(f"test_sigma{sig:g}", config.n_per_snr, 30 + 10 * k, noise_sigma=float(sig)))
        return jobs

    jobs = [
        DatasetJob("train", config.n_train, 0, config.snr_min, config.snr_max),
        DatasetJob("val", config.n_val, 10, config.snr_min, config.snr_max),
        DatasetJob("test", config.n_test, 20, config.snr_min, config.snr_max),
    ]
    for k, snr in enumerate(config.snr_ladder):
        jobs.append(DatasetJob(f"test_snr{int(snr)}", config.n_per_snr, 30 + 10 * k, float(snr), float(snr)))
    return jobs


def generate_dataset_family(config: DatasetFamilyConfig, *, verbose: bool = True) -> list[Path]:
    """Generate train/val/test and fixed-SNR test parquet files."""
    protocol = load_protocol()
    out_dir = Path(config.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    level = (f"sigma={config.noise_sigma:g}" if config.noise_sigma is not None
             else f"snr=[{config.snr_min:g},{config.snr_max:g}]")
    if verbose:
        print(protocol.summary())
        print(
            f"noise=gaussian | {level} | T1{config.t1_range} T2{config.t2_range}"
        )

    written: list[Path] = []
    total, t0 = 0, time.time()
    for job in build_dataset_jobs(config):
        t = time.time()
        df = generate_dataset(
            job.n_voxels,
            master_seed=job.master_seed,
            protocol=protocol,
            snr_min=job.snr_min,
            snr_max=job.snr_max,
            t1_range=config.t1_range,
            t2_range=config.t2_range,
            noise_sigma=job.noise_sigma,
        )

        path = out_dir / f"{job.name}.parquet"
        df.to_parquet(path, index=False)
        written.append(path)
        total += job.n_voxels

        if verbose:
            mb = os.path.getsize(path) / 1e6
            job_level = (f"sigma={job.noise_sigma:g}" if job.noise_sigma is not None
                         else f"snr=[{job.snr_min:g},{job.snr_max:g}]")
            print(
                f"  {job.name:<14} n={job.n_voxels:>9,} seed={job.master_seed:<3} "
                f"{job_level} -> {path} ({mb:.0f} MB, {time.time() - t:.1f}s)"
            )

    if verbose:
        print(f"done: {total:,} voxels in {len(written)} files, {time.time() - t0:.0f}s total -> {out_dir}")
    return written
