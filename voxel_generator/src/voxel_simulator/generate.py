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

import hashlib
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .noise import add_noise
from .physics import simulate_clean_signal
from .protocol import DEFAULT_MAT_PATH, Protocol, load_protocol
from .sampler import (
    MAX_COMP,
    MIN_WEIGHT,
    SNR_MAX,
    SNR_MIN,
    SPLIT_SNR_LADDER,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    STREAM_NOISE,
    STREAM_PARAMS,
    STREAM_SNR,
    T1_RANGE,
    T2_RANGE,
    VoxelSpec,
    sample_voxel_spec,
    validate_ranges,
    voxel_rng,
)


def _check_unique_names(names: list[str], what: str) -> None:
    """Ladder filenames are derived from their level, so two levels can collide into one name."""
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(
            f"{what} produces duplicate output names {sorted(dupes)}; each level must map to a "
            "distinct filename or one would silently overwrite the other."
        )


@dataclass(frozen=True)
class GeneratedVoxel:
    """A fully simulated voxel before it is flattened into a dataset row."""

    voxel_id: int
    spec: VoxelSpec
    signal: np.ndarray
    sigma: float


def simulate_voxel_signal(
    protocol: Protocol,
    spec: VoxelSpec,
    base_seed: int,
    split_code: int,
    voxel_id: int,
    noise_sigma: float | None = None,
) -> tuple[np.ndarray, float]:
    """Forward + measurement model: target compartments -> model input signal.

    The noise stream is keyed the same way as the parameter stream but with STREAM_NOISE, so the
    two are independent yet both reproducible from (base_seed, n_comp, split, voxel_id).
    """
    clean = simulate_clean_signal(protocol, spec.t1, spec.t2, spec.w)
    rng = voxel_rng(base_seed, spec.n_comp, split_code, voxel_id, STREAM_NOISE)
    return add_noise(clean, spec.snr, rng, sigma=noise_sigma)


def generate_voxel(
    voxel_id: int,
    n_comp: int,
    protocol: Protocol,
    base_seed: int = 0,
    split_code: int = SPLIT_TRAIN,
    snr_min: float = SNR_MIN,
    snr_max: float = SNR_MAX,
    snr: float | None = None,
    t1_range: tuple[float, float] = T1_RANGE,
    t2_range: tuple[float, float] = T2_RANGE,
    noise_sigma: float | None = None,
) -> GeneratedVoxel:
    """Generate the physical contents and measured signal for one voxel.

    The readable pipeline is:

        sample (T1, T2, w) targets
        -> simulate the 64-point TI/TE measurement
        -> return target + model input signal

    Every argument is passed on by keyword. That is deliberate: this call used to be positional,
    which silently dropped the compartment count and made --n-comp a no-op that looked like it
    worked.
    """
    spec = sample_voxel_spec(
        voxel_id=voxel_id,
        n_comp=n_comp,
        base_seed=base_seed,
        split_code=split_code,
        snr_min=snr_min,
        snr_max=snr_max,
        snr=snr,
        t1_range=t1_range,
        t2_range=t2_range,
    )
    signal, sigma = simulate_voxel_signal(
        protocol, spec, base_seed=base_seed, split_code=split_code, voxel_id=voxel_id,
        noise_sigma=noise_sigma,
    )
    return GeneratedVoxel(voxel_id=voxel_id, spec=spec, signal=signal, sigma=sigma)


def voxel_to_row(voxel: GeneratedVoxel, protocol: Protocol, noise_sigma: float | None = None) -> dict:
    """Flatten a generated voxel into the stable tabular dataset schema.

    The ground-truth table is always MAX_COMP wide regardless of this file's n_comp, with unused
    slots NaN. That fixed width is what lets the per-n files be concatenated into one dataset.
    """
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
    n_comp: int,
    protocol: Protocol,
    base_seed: int = 0,
    split_code: int = SPLIT_TRAIN,
    snr_min: float = SNR_MIN,
    snr_max: float = SNR_MAX,
    snr: float | None = None,
    t1_range: tuple[float, float] = T1_RANGE,
    t2_range: tuple[float, float] = T2_RANGE,
    noise_sigma: float | None = None,
) -> dict:
    """Build one full row of the dataset.

    noise_sigma sets the absolute Gaussian noise std directly (e.g. 0.1); when given it wins
    and the per-voxel SNR is not used (the `snr` column is left NaN to say so).
    """
    voxel = generate_voxel(
        voxel_id=voxel_id,
        n_comp=n_comp,
        protocol=protocol,
        base_seed=base_seed,
        split_code=split_code,
        snr_min=snr_min,
        snr_max=snr_max,
        snr=snr,
        t1_range=t1_range,
        t2_range=t2_range,
        noise_sigma=noise_sigma,
    )
    return voxel_to_row(voxel, protocol, noise_sigma)


def generate_dataset(
    n_voxels: int,
    n_comp: int,
    base_seed: int = 0,
    split_code: int = SPLIT_TRAIN,
    protocol: Optional[Protocol] = None,
    snr_min: float = SNR_MIN,
    snr_max: float = SNR_MAX,
    snr: float | None = None,
    t1_range: tuple[float, float] = T1_RANGE,
    t2_range: tuple[float, float] = T2_RANGE,
    noise_sigma: float | None = None,
) -> pd.DataFrame:
    """Generate a DataFrame of `n_voxels` rows, all with exactly `n_comp` compartments.

    Noise level is set either by SNR (snr_min/snr_max, or a pinned `snr`) or, if `noise_sigma` is
    given, by that absolute Gaussian std directly.
    """
    if protocol is None:
        protocol = load_protocol()
    rows = [
        generate_one(
            voxel_id=i,
            n_comp=n_comp,
            protocol=protocol,
            base_seed=base_seed,
            split_code=split_code,
            snr_min=snr_min,
            snr_max=snr_max,
            snr=snr,
            t1_range=t1_range,
            t2_range=t2_range,
            noise_sigma=noise_sigma,
        )
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
    # Every voxel in this family has exactly this many compartments. Files are per-n so the
    # per-count splits come out exactly balanced instead of merely balanced in expectation.
    n_comp: int = 1
    # Base seed for the whole family. Different base seeds give independent datasets.
    seed: int = 0
    overwrite: bool = False
    n_train: int = 250_000
    n_val: int = 25_000
    n_test: int = 25_000
    n_per_snr: int = 12_500
    # Train-SNR bounds come from the sampler constants — never a second literal, or the two
    # drift apart (they did: this used to default to 20 while SNR_MIN said 30, and this won).
    snr_min: float = SNR_MIN
    snr_max: float = SNR_MAX
    # The ladder deliberately reaches below snr_min: SNR 20 is an extrapolation test.
    snr_ladder: tuple[float, ...] = (20, 40, 60, 100, 150)
    t1_range: tuple[float, float] = T1_RANGE
    t2_range: tuple[float, float] = T2_RANGE
    # Absolute-noise mode: if set, the whole family uses Gaussian noise of this std (SNR is
    # ignored) and the robustness test sets come from sigma_ladder instead of snr_ladder.
    noise_sigma: float | None = None
    sigma_ladder: tuple[float, ...] = (0.05, 0.1, 0.2)

    def __post_init__(self) -> None:
        # Everything here fails before a single voxel is generated. A million-voxel cluster job
        # should die in the first second on a bad config, not forty minutes in.
        if min(self.n_train, self.n_val, self.n_test, self.n_per_snr) < 0:
            raise ValueError("dataset sizes must be nonnegative")
        if not 1 <= self.n_comp <= MAX_COMP:
            raise ValueError(f"n_comp must be in 1..{MAX_COMP}; got {self.n_comp}")
        if self.n_comp * MIN_WEIGHT >= 1.0:
            raise ValueError(
                f"n_comp={self.n_comp} x MIN_WEIGHT={MIN_WEIGHT} >= 1: no valid weights exist."
            )
        validate_ranges(self.t1_range, self.t2_range)
        if self.noise_sigma is not None:
            if self.noise_sigma <= 0 or any(s <= 0 for s in self.sigma_ladder):
                raise ValueError("sigma values must be positive")
            _check_unique_names([f"test_sigma{s:g}" for s in self.sigma_ladder], "sigma_ladder")
        else:
            if self.snr_min <= 0 or self.snr_max <= 0:
                raise ValueError("SNR bounds must be positive")
            if self.snr_min > self.snr_max:
                raise ValueError("snr_min must be <= snr_max")
            if any(snr <= 0 for snr in self.snr_ladder):
                raise ValueError("all fixed SNR values must be positive")
            # Filenames use int(snr), so 20.5 and 20.7 would both become test_snr20 and the
            # second would silently overwrite the first.
            _check_unique_names([f"test_snr{int(s)}" for s in self.snr_ladder], "snr_ladder")


@dataclass(frozen=True)
class DatasetJob:
    name: str
    n_voxels: int
    split_code: int
    snr_min: float = SNR_MIN
    snr_max: float = SNR_MAX
    snr: float | None = None            # pinned SNR (fixed-SNR ladder); None => drawn per voxel
    noise_sigma: float | None = None


def smoke_config(config: DatasetFamilyConfig) -> DatasetFamilyConfig:
    """Return a small version of a config for quick local checks."""
    return replace(config, n_train=2_000, n_val=500, n_test=500, n_per_snr=300)


def build_dataset_jobs(config: DatasetFamilyConfig) -> list[DatasetJob]:
    """Create the deterministic split jobs from one config.

    Splits are separated by their *split code*, which is part of every RNG stream key — so they
    cannot overlap however big they get. (The previous scheme spaced arithmetic master seeds by a
    fixed stride, which silently collided once a split grew past it.)

    Every rung of the fixed-SNR ladder shares SPLIT_SNR_LADDER and pins its SNR rather than
    drawing one. Since SNR has its own stream, that leaves the parameter and noise streams
    untouched: all rungs get the *same* voxels with the *same* standardized noise, and only the
    amplitude changes. That is what makes the ladder a paired comparison instead of five
    unrelated samples.

    Checking that property from the written files needs a tolerance, not equality. Ground truth is
    stored float64 and does match exactly, but the signal is stored float32, so recovering
    z = (S - S_clean)/sigma amplifies the storage rounding by 1/sigma — about 2e-5 at SNR 150.
    Compare recovered z with atol ~1e-4.
    """
    if config.noise_sigma is not None:      # absolute-noise mode
        s = config.noise_sigma
        jobs = [
            DatasetJob("train", config.n_train, SPLIT_TRAIN, noise_sigma=s),
            DatasetJob("val", config.n_val, SPLIT_VAL, noise_sigma=s),
            DatasetJob("test", config.n_test, SPLIT_TEST, noise_sigma=s),
        ]
        for sig in config.sigma_ladder:
            jobs.append(
                DatasetJob(f"test_sigma{sig:g}", config.n_per_snr, SPLIT_SNR_LADDER,
                           noise_sigma=float(sig))
            )
        return jobs

    jobs = [
        DatasetJob("train", config.n_train, SPLIT_TRAIN, config.snr_min, config.snr_max),
        DatasetJob("val", config.n_val, SPLIT_VAL, config.snr_min, config.snr_max),
        DatasetJob("test", config.n_test, SPLIT_TEST, config.snr_min, config.snr_max),
    ]
    for snr in config.snr_ladder:
        jobs.append(
            DatasetJob(f"test_snr{int(snr)}", config.n_per_snr, SPLIT_SNR_LADDER,
                       config.snr_min, config.snr_max, snr=float(snr))
        )
    return jobs


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write via a temp file + rename, so a crash never leaves a half file under the real name.

    A truncated parquet that *looks* like a finished dataset is the worst outcome here: it would
    sail through a glob and only surface as strange results much later.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)           # atomic within a filesystem
    finally:
        if tmp.exists():
            tmp.unlink()


def _git_state(repo_dir: Path) -> dict:
    """Best-effort git commit + dirty flag for provenance. Never fails generation."""
    import subprocess

    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(args, cwd=repo_dir, capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    commit = _run("git", "rev-parse", "HEAD")
    status = _run("git", "status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def _dependency_versions() -> dict:
    """Record what actually produced the numbers.

    NumPy does not guarantee Generator bit streams across versions, so "reproducible" is only a
    claim about a recorded environment — this is that record.
    """
    import platform

    import pyarrow

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
    }


def build_manifest(config: DatasetFamilyConfig, jobs: list[DatasetJob], rows: dict[str, int]) -> dict:
    """Everything needed to say exactly what this dataset is and where it came from."""
    return {
        "n_comp": config.n_comp,
        "base_seed": config.seed,
        "max_comp": MAX_COMP,
        "splits": {
            j.name: {
                "rows": rows[j.name],
                "split_code": j.split_code,
                "snr": j.snr,
                "snr_min": None if j.noise_sigma is not None else j.snr_min,
                "snr_max": None if j.noise_sigma is not None else j.snr_max,
                "noise_sigma": j.noise_sigma,
            }
            for j in jobs
        },
        "streams": {"params": STREAM_PARAMS, "noise": STREAM_NOISE, "snr": STREAM_SNR},
        "physics": {
            "t1_range": list(config.t1_range),
            "t2_range": list(config.t2_range),
            "min_weight": MIN_WEIGHT,
            "noise": "additive_gaussian_signed",
        },
        "protocol_sha256": _sha256(Path(DEFAULT_MAT_PATH)),
        "git": _git_state(Path(__file__).resolve().parents[3]),
        "dependencies": _dependency_versions(),
    }


def _sha256(path: Path) -> str:
    """Checksum of the protocol file, so the cluster can prove it has the same one."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_dataset_family(config: DatasetFamilyConfig, *, verbose: bool = True) -> list[Path]:
    """Generate train/val/test and fixed-SNR test parquet files for one fixed n_comp."""
    protocol = load_protocol()
    out_dir = Path(config.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    jobs = build_dataset_jobs(config)

    # Refuse to clobber before generating anything — finding out after an hour is worse.
    if not config.overwrite:
        existing = [out_dir / f"{j.name}.parquet" for j in jobs]
        existing = [p for p in existing if p.exists()]
        if existing:
            raise FileExistsError(
                f"{len(existing)} output file(s) already exist in {out_dir} "
                f"(e.g. {existing[0].name}). Pass overwrite=True / --overwrite to replace them."
            )

    level = (f"sigma={config.noise_sigma:g}" if config.noise_sigma is not None
             else f"snr=[{config.snr_min:g},{config.snr_max:g}]")
    if verbose:
        print(protocol.summary())
        print(
            f"n_comp={config.n_comp} | base_seed={config.seed} | noise=gaussian | {level} | "
            f"T1{config.t1_range} T2{config.t2_range}"
        )

    written: list[Path] = []
    rows: dict[str, int] = {}
    total, t0 = 0, time.time()
    for job in jobs:
        t = time.time()
        df = generate_dataset(
            job.n_voxels,
            n_comp=config.n_comp,
            base_seed=config.seed,
            split_code=job.split_code,
            protocol=protocol,
            snr_min=job.snr_min,
            snr_max=job.snr_max,
            snr=job.snr,
            t1_range=config.t1_range,
            t2_range=config.t2_range,
            noise_sigma=job.noise_sigma,
        )

        path = out_dir / f"{job.name}.parquet"
        _write_parquet_atomic(df, path)
        written.append(path)
        rows[job.name] = len(df)
        total += job.n_voxels

        if verbose:
            mb = os.path.getsize(path) / 1e6
            job_level = (f"sigma={job.noise_sigma:g}" if job.noise_sigma is not None
                         else (f"snr={job.snr:g}" if job.snr is not None
                               else f"snr=[{job.snr_min:g},{job.snr_max:g}]"))
            print(
                f"  {job.name:<14} n={job.n_voxels:>9,} split={job.split_code} "
                f"{job_level} -> {path} ({mb:.0f} MB, {time.time() - t:.1f}s)"
            )

    # Last, and only on full success: the manifest is the family's "this is complete" marker.
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(build_manifest(config, jobs, rows), f, indent=2)
    written.append(manifest_path)

    if verbose:
        print(f"done: {total:,} voxels in {len(written) - 1} files, "
              f"{time.time() - t0:.0f}s total -> {out_dir}")
    return written
