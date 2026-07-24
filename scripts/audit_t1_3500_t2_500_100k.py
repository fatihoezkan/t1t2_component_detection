"""Hard-gate audit for the new T1<=3500 ms, T2<=500 ms dataset.

This is a new standalone artifact and does not execute or overwrite the historical audit
notebook. It validates every generated row, split separation, paired SNR targets, manifests, and
the requested physical bounds, then writes a JSON verdict and two coverage figures.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_ROWS = {
    "train": 33333,
    "val": 3333,
    "test": 3333,
    "test_snr20": 1667,
    "test_snr40": 1667,
    "test_snr60": 1667,
    "test_snr100": 1667,
    "test_snr150": 1667,
}
EXPECTED_SPLIT_CODE = {
    "train": 0,
    "val": 1,
    "test": 2,
    "test_snr20": 3,
    "test_snr40": 3,
    "test_snr60": 3,
    "test_snr100": 3,
    "test_snr150": 3,
}
TARGET_COLUMNS = [
    "voxel_id", "n_comp",
    *[f"{kind}_{i}" for i in range(1, 5) for kind in ("T1", "T2", "w")],
]
SIGNAL_COLUMNS = [f"S_{i}" for i in range(1, 65)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/t1_3500_t2_500_100k")
    parser.add_argument("--output", default="results/data_audit/t1_3500_t2_500_100k")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    failures, checks = [], []
    coverage = []
    weight_by_n = {}
    family_records = {}

    def check(condition, name, detail=""):
        passed = bool(condition)
        checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            failures.append({"name": name, "detail": detail})

    common_identity = None
    for n_comp in (1, 2, 3):
        family = data_root / f"n{n_comp}"
        manifest_path = family / "manifest.json"
        check(manifest_path.is_file(), f"n{n_comp}: manifest exists", str(manifest_path))
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        identity = {
            "base_seed": manifest.get("base_seed"),
            "streams": manifest.get("streams"),
            "physics": manifest.get("physics"),
            "protocol_sha256": manifest.get("protocol_sha256"),
        }
        if common_identity is None:
            common_identity = identity
        check(identity == common_identity, f"n{n_comp}: common family identity")
        check(manifest.get("n_comp") == n_comp, f"n{n_comp}: manifest count")
        check(manifest.get("base_seed") == 3500500, f"n{n_comp}: base seed")
        physics = manifest.get("physics", {})
        check(physics.get("t1_range") == [50.0, 3500.0], f"n{n_comp}: T1 manifest range")
        check(physics.get("t2_range") == [5.0, 500.0], f"n{n_comp}: T2 manifest range")
        check(
            physics.get("noise") == "additive_gaussian_signed",
            f"n{n_comp}: signed Gaussian noise",
        )

        split_hashes = {}
        ladder_targets = None
        family_records[f"n{n_comp}"] = {}
        for split, expected_rows in EXPECTED_ROWS.items():
            path = family / f"{split}.parquet"
            check(path.is_file(), f"n{n_comp}/{split}: file exists", str(path))
            if not path.is_file():
                continue
            df = pd.read_parquet(path)
            family_records[f"n{n_comp}"][split] = len(df)
            check(len(df) == expected_rows, f"n{n_comp}/{split}: exact row count", str(len(df)))
            check(
                int(manifest["splits"][split]["split_code"]) == EXPECTED_SPLIT_CODE[split],
                f"n{n_comp}/{split}: split code",
            )
            check(
                np.array_equal(df["voxel_id"].to_numpy(), np.arange(expected_rows)),
                f"n{n_comp}/{split}: sequential voxel IDs",
            )
            check((df["n_comp"] == n_comp).all(), f"n{n_comp}/{split}: fixed n_comp")

            t1 = df[[f"T1_{i}" for i in range(1, n_comp + 1)]].to_numpy(float)
            t2 = df[[f"T2_{i}" for i in range(1, n_comp + 1)]].to_numpy(float)
            weight = df[[f"w_{i}" for i in range(1, n_comp + 1)]].to_numpy(float)
            signal = df[SIGNAL_COLUMNS].to_numpy(np.float32)

            check(np.isfinite(t1).all(), f"n{n_comp}/{split}: finite active T1")
            check(np.isfinite(t2).all(), f"n{n_comp}/{split}: finite active T2")
            check(np.isfinite(weight).all(), f"n{n_comp}/{split}: finite active weights")
            check(np.isfinite(signal).all(), f"n{n_comp}/{split}: finite signals")
            check(
                ((t1 >= 50.0) & (t1 <= 3500.0)).all(),
                f"n{n_comp}/{split}: T1 in [50,3500]",
                f"observed [{t1.min():.4g},{t1.max():.4g}]",
            )
            check(
                ((t2 >= 5.0) & (t2 <= 500.0)).all(),
                f"n{n_comp}/{split}: T2 in [5,500]",
                f"observed [{t2.min():.4g},{t2.max():.4g}]",
            )
            check((t1 > t2).all(), f"n{n_comp}/{split}: T1>T2")
            check((weight >= 0.05 - 1e-12).all(), f"n{n_comp}/{split}: minimum weight")
            check(
                np.allclose(weight.sum(axis=1), 1.0, atol=1e-10),
                f"n{n_comp}/{split}: weights sum to one",
            )
            check((signal < 0).any(), f"n{n_comp}/{split}: signed negative samples present")

            if n_comp < 4:
                padded = df[
                    [f"{kind}_{i}" for i in range(n_comp + 1, 5)
                     for kind in ("T1", "T2", "w")]
                ]
                check(padded.isna().all().all(), f"n{n_comp}/{split}: padding remains NaN")

            target_frame = df[TARGET_COLUMNS]
            if split in ("train", "val", "test"):
                split_hashes[split] = set(
                    pd.util.hash_pandas_object(target_frame, index=False).astype("uint64")
                )
            else:
                if ladder_targets is None:
                    ladder_targets = target_frame
                else:
                    check(
                        target_frame.equals(ladder_targets),
                        f"n{n_comp}/{split}: paired ladder targets",
                    )

            if split == "train":
                coverage.append(
                    pd.DataFrame({
                        "T1": t1.ravel(),
                        "T2": t2.ravel(),
                        "weight": weight.ravel(),
                        "n_comp": n_comp,
                    })
                )
                weight_by_n[n_comp] = weight.ravel()

        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            if left in split_hashes and right in split_hashes:
                overlap = split_hashes[left] & split_hashes[right]
                check(
                    not overlap,
                    f"n{n_comp}: no {left}/{right} target overlap",
                    f"{len(overlap)} overlapping hashes",
                )

    passed = not failures
    summary = {
        "data_root": str(data_root),
        "passed": passed,
        "checks_total": len(checks),
        "checks_failed": len(failures),
        "failures": failures,
        "common_identity": common_identity,
        "rows": family_records,
        "requested_bounds": {
            "t1": [50.0, 3500.0],
            "t2": [5.0, 500.0],
            "constraint": "T1>T2",
        },
    }
    tmp = output / "audit_summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2))
    os.replace(tmp, output / "audit_summary.json")

    if coverage:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cov = pd.concat(coverage, ignore_index=True)
        sample = cov.iloc[::max(1, len(cov) // 40000)]
        fig, ax = plt.subplots(figsize=(7.5, 6))
        points = ax.scatter(
            sample["T1"], sample["T2"], c=sample["weight"],
            s=5, alpha=.25, cmap="viridis", rasterized=True,
        )
        line = np.geomspace(50, 500, 200)
        ax.plot(line, line, "k--", lw=1, label="T1 = T2 boundary")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set(
            xlabel="T1 (ms)", ylabel="T2 (ms)",
            title="New training-family coverage: T1≤3500 ms, T2≤500 ms",
        )
        ax.grid(which="both", alpha=.2)
        ax.legend()
        fig.colorbar(points, ax=ax, label="true signal fraction")
        fig.tight_layout()
        fig.savefig(output / "coverage.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        for n_comp, values in weight_by_n.items():
            ax.hist(values, bins=np.linspace(.05, 1, 40), histtype="step", lw=2,
                    density=True, label=f"n={n_comp}")
        ax.set(
            xlabel="true signal fraction", ylabel="density",
            title="Signal-fraction distribution by compartment count",
        )
        ax.grid(alpha=.2); ax.legend()
        fig.tight_layout()
        fig.savefig(output / "signal_fraction_distribution.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
