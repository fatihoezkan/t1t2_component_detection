"""The single entry point for generating synthetic voxel datasets.

Quick dry run (tiny files, seconds):
    python run_generator.py --smoke

Full training family (train/val/test + fixed-SNR test sets):
    python run_generator.py --out-dir output/data

Every setting has a flag; run `python run_generator.py --help` to see them. For use inside
a larger Python pipeline, import from `voxel_simulator.generate` instead of shelling out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from voxel_simulator.generate import (  # noqa: E402
    DatasetFamilyConfig,
    generate_dataset_family,
    smoke_config,
)
from voxel_simulator.sampler import MAX_COMP, SNR_MAX, SNR_MIN, T1_RANGE, T2_RANGE  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate the synthetic T1–T2 voxel dataset family.")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "output" / "data"),
                    help="Directory for the parquet files.")
    ap.add_argument("--n-comp", type=int, required=True, choices=range(1, MAX_COMP + 1),
                    help="Compartments per voxel. Every voxel in this family gets exactly this "
                         "many; generate one family per count to get an exactly balanced dataset.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Base seed for the whole family. A different seed gives an independent "
                         "dataset (splits stay disjoint either way).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace existing output files. Off by default so a re-run cannot quietly "
                         "destroy a dataset that took hours to make.")
    ap.add_argument("--n-train", type=int, default=250_000)
    ap.add_argument("--n-val", type=int, default=25_000)
    ap.add_argument("--n-test", type=int, default=25_000)
    ap.add_argument("--n-per-snr", type=int, default=12_500, help="Voxels per fixed-SNR test set.")
    ap.add_argument("--snr-min", type=float, default=SNR_MIN,
                    help="Lower train-SNR bound (defaults to the sampler's SNR_MIN).")
    ap.add_argument("--snr-max", type=float, default=SNR_MAX,
                    help="Upper train-SNR bound (defaults to the sampler's SNR_MAX).")
    ap.add_argument("--snr-ladder", type=float, nargs="+", default=[20, 40, 60, 100, 150],
                    help="Fixed SNR values to build separate test sets for. Values below --snr-min "
                         "(e.g. 20) are extrapolation tests and must be reported as such.")
    ap.add_argument("--t1-min", type=float, default=T1_RANGE[0])
    ap.add_argument("--t1-max", type=float, default=T1_RANGE[1])
    ap.add_argument("--t2-min", type=float, default=T2_RANGE[0])
    ap.add_argument("--t2-max", type=float, default=T2_RANGE[1])
    ap.add_argument("--noise-sigma", type=float, default=None,
                    help="Absolute Gaussian noise std (e.g. 0.1). If set, SNR is ignored and the "
                         "robustness test sets use --sigma-ladder instead of --snr-ladder.")
    ap.add_argument("--sigma-ladder", type=float, nargs="+", default=[0.05, 0.1, 0.2],
                    help="Fixed sigma values for the test sets (used only with --noise-sigma).")
    ap.add_argument("--smoke", action="store_true", help="Tiny sizes for a quick dry run.")
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    config = DatasetFamilyConfig(
        out_dir=a.out_dir,
        n_comp=a.n_comp,
        seed=a.seed,
        overwrite=a.overwrite,
        n_train=a.n_train,
        n_val=a.n_val,
        n_test=a.n_test,
        n_per_snr=a.n_per_snr,
        snr_min=a.snr_min,
        snr_max=a.snr_max,
        snr_ladder=tuple(a.snr_ladder),
        t1_range=(a.t1_min, a.t1_max),
        t2_range=(a.t2_min, a.t2_max),
        noise_sigma=a.noise_sigma,
        sigma_ladder=tuple(a.sigma_ladder),
    )
    if a.smoke:
        config = smoke_config(config)
    generate_dataset_family(config)


if __name__ == "__main__":
    main()
