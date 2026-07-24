"""Finalize a trained baseline checkpoint without retraining it.

Usage:

    PYTHONPATH=src python -m t1t2.baseline \
        --config configs/baseline_final_100k.yaml

The command selects the existence threshold on validation data, freezes it, evaluates the test
and paired SNR splits, and creates a self-contained result package with metrics, query analysis,
figures, and provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
import yaml

from .baseline_analysis import (
    plot_count_confusion,
    plot_error_by_count,
    plot_examples,
    plot_query_specialization,
    plot_snr_robustness,
    plot_threshold_calibration,
    plot_training_curves,
    query_specialization,
    save_json,
    select_examples,
    signal_reconstruction_metrics,
)
from .config import load_config
from .data import TargetNormalizer, VoxelDataset
from .device import device_info, get_device
from .eval import (
    calibrate_existence_threshold,
    compute_metrics,
    detr_query_outputs,
    predictions_from_query_outputs,
    scatter_figure,
    true_compartments,
)
from .model import build_model


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_state(repo: Path) -> dict:
    def run(*args):
        result = subprocess.run(
            args, cwd=repo, capture_output=True, text=True, timeout=15, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "changed_paths": [] if not status else [line[3:] for line in status.splitlines()],
    }


def _paths(data_cfg: dict, split: str) -> list[str]:
    root = Path(data_cfg["family_root"])
    return [str(root / f"n{n}" / f"{split}.parquet") for n in data_cfg["compartment_counts"]]


def _manifest_validation(final_cfg: dict, run_cfg) -> dict:
    """Validate the deterministic-prefix claim before using the larger local families."""
    data_cfg = final_cfg["data"]
    root = Path(data_cfg["family_root"])
    counts = [int(n) for n in data_cfg["compartment_counts"]]
    expected_seed = int(data_cfg["expected_base_seed"])
    rows_by_split = {
        "val": int(data_cfg["val_rows_per_count"]),
        "test": int(data_cfg["test_rows_per_count"]),
        **{
            f"test_snr{int(snr)}": int(data_cfg["snr_rows_per_count"])
            for snr in data_cfg["snr_levels"]
        },
    }
    expected_split_code = {"val": 1, "test": 2}
    expected_split_code.update({f"test_snr{int(snr)}": 3 for snr in data_cfg["snr_levels"]})

    manifests, common = {}, None
    for n in counts:
        manifest_path = root / f"n{n}" / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing dataset manifest: {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)
        if int(manifest["n_comp"]) != n:
            raise ValueError(f"{manifest_path} says n_comp={manifest['n_comp']}, expected {n}")
        if int(manifest["base_seed"]) != expected_seed:
            raise ValueError(
                f"{manifest_path} has base_seed={manifest['base_seed']}, expected {expected_seed}"
            )
        identity = {
            "base_seed": manifest["base_seed"],
            "streams": manifest["streams"],
            "physics": manifest["physics"],
            "protocol_sha256": manifest["protocol_sha256"],
        }
        if common is None:
            common = identity
        elif identity != common:
            raise ValueError(f"{manifest_path} does not match the other compartment families")
        if tuple(manifest["physics"]["t1_range"]) != (run_cfg.data.t1_min, run_cfg.data.t1_max):
            raise ValueError("dataset T1 range disagrees with the trained model configuration")
        if tuple(manifest["physics"]["t2_range"]) != (run_cfg.data.t2_min, run_cfg.data.t2_max):
            raise ValueError("dataset T2 range disagrees with the trained model configuration")

        split_record = {}
        for split, rows in rows_by_split.items():
            meta = manifest["splits"].get(split)
            if meta is None:
                raise ValueError(f"{manifest_path} has no {split} split")
            if int(meta["rows"]) < rows:
                raise ValueError(f"{manifest_path}: {split} has {meta['rows']} rows, need {rows}")
            if int(meta["split_code"]) != expected_split_code[split]:
                raise ValueError(
                    f"{manifest_path}: {split} has split code {meta['split_code']}, "
                    f"expected {expected_split_code[split]}"
                )
            parquet = root / f"n{n}" / f"{split}.parquet"
            if not parquet.exists():
                raise FileNotFoundError(f"missing dataset split: {parquet}")
            voxel_ids = pd.read_parquet(parquet, columns=["voxel_id"]).iloc[:rows, 0].to_numpy()
            if not np.array_equal(voxel_ids, np.arange(rows)):
                raise ValueError(
                    f"{parquet}: requested prefix is not voxel_id 0..{rows - 1}"
                )
            split_record[split] = {
                "path": str(parquet),
                "available_rows": int(meta["rows"]),
                "used_prefix_rows": rows,
                "split_code": int(meta["split_code"]),
            }
        manifests[f"n{n}"] = {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "splits": split_record,
        }
    return {"common_identity": common, "families": manifests}


def _dataset(paths, rows_per_count, cfg, normalizer):
    return VoxelDataset(
        paths,
        cfg.data,
        normalizer,
        limit=int(rows_per_count) * len(paths),
    )


def _load_model(run_cfg, checkpoint, device):
    model = build_model(run_cfg.model).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model, state


def _runtime_versions() -> dict:
    import matplotlib
    import pyarrow

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pyarrow.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _prepare_output(path: Path, allow_existing: bool) -> None:
    if path.exists() and not allow_existing:
        raise FileExistsError(
            f"{path} already exists; use a new output directory or pass --allow-existing"
        )
    path.mkdir(parents=True, exist_ok=True)


def finalize_baseline(config_path, output_dir=None, allow_existing=False, log=print):
    """Run the complete offline baseline evaluation and return its summary dictionary."""
    config_path = Path(config_path)
    with open(config_path) as f:
        final_cfg = yaml.safe_load(f)

    source = final_cfg["source"]
    run_config_path = Path(source["run_config"])
    checkpoint_path = Path(source["checkpoint"])
    history_path = Path(source["history"])
    audit_path = Path(source["data_audit"])
    for required in (run_config_path, checkpoint_path, history_path, audit_path):
        if not required.exists():
            raise FileNotFoundError(f"required source artifact is missing: {required}")
    with open(audit_path) as f:
        data_audit = json.load(f)
    if not data_audit.get("passed", False):
        raise ValueError(f"dataset audit did not pass: {audit_path}")

    run_cfg = load_config(run_config_path)
    output = Path(output_dir or final_cfg["output_dir"])
    _prepare_output(output, allow_existing)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)

    log(f"[baseline-finalize] validating deterministic data prefixes")
    data_provenance = _manifest_validation(final_cfg, run_cfg)
    data_provenance["audit"] = {
        "path": str(audit_path),
        "sha256": _sha256(audit_path),
        "passed": True,
        "checks_total": data_audit.get("checks_total"),
        "checks_failed": data_audit.get("checks_failed"),
        "warnings": data_audit.get("warnings"),
    }
    normalizer = TargetNormalizer.from_config(run_cfg.data)
    device = get_device(final_cfg.get("device", run_cfg.train.device))
    model, checkpoint = _load_model(run_cfg, checkpoint_path, device)
    batch_size = int(final_cfg.get("batch_size", 512))
    counts = final_cfg["data"]["compartment_counts"]
    n_queries = run_cfg.model.n_queries
    started = time.perf_counter()

    # 1. Validation-only threshold selection.
    val_paths = _paths(final_cfg["data"], "val")
    val_ds = _dataset(
        val_paths, final_cfg["data"]["val_rows_per_count"], run_cfg, normalizer
    )
    log(f"[baseline-finalize] validation inference: {len(val_ds):,} voxels on {device}")
    val_queries = detr_query_outputs(model, val_ds, device, normalizer, batch_size=batch_size)
    val_trues = true_compartments(val_ds)
    threshold_cfg = final_cfg.get("threshold", {})
    thresholds = np.linspace(
        float(threshold_cfg.get("min", 0.05)),
        float(threshold_cfg.get("max", 0.95)),
        int(threshold_cfg.get("steps", 91)),
    )
    calibration = calibrate_existence_threshold(val_queries, val_trues, thresholds)
    threshold = float(calibration["selected_threshold"])
    save_json(calibration, output / "threshold_calibration.json")
    np.savez_compressed(output / "validation_query_outputs.npz", **val_queries)
    plot_threshold_calibration(calibration, figures / "threshold_calibration.png")
    log(
        f"[baseline-finalize] selected threshold={threshold:.3f} on validation "
        f"(count accuracy={calibration['selected']['count_accuracy']:.3f})"
    )

    # 2. Test once with the frozen threshold.
    test_paths = _paths(final_cfg["data"], "test")
    test_ds = _dataset(
        test_paths, final_cfg["data"]["test_rows_per_count"], run_cfg, normalizer
    )
    log(f"[baseline-finalize] frozen-threshold test inference: {len(test_ds):,} voxels")
    test_queries = detr_query_outputs(model, test_ds, device, normalizer, batch_size=batch_size)
    test_trues = true_compartments(test_ds)
    test_preds = predictions_from_query_outputs(test_queries, threshold)
    test_metrics = compute_metrics(test_preds, test_trues, n_queries=n_queries)
    test_metrics["exist_thresh"] = threshold
    test_metrics["threshold_selected_on"] = "validation"
    test_metrics |= signal_reconstruction_metrics(
        test_preds, test_trues, test_ds, run_cfg.data.signal_norm
    )
    save_json(test_metrics, output / "metrics_test.json")
    np.savez_compressed(output / "test_query_outputs.npz", **test_queries)

    # 3. Query responsibility and deterministic error analysis.
    query_report, matched_records = query_specialization(test_queries, test_trues, threshold)
    examples = select_examples(test_queries, test_trues, threshold)
    save_json(query_report, output / "query_analysis.json")
    save_json(examples, output / "prediction_examples.json")

    # 4. Paired SNR ladder under the same frozen threshold.
    ladder = {}
    train_snr_min = float(final_cfg["data"]["train_snr_min"])
    for snr in final_cfg["data"]["snr_levels"]:
        label = f"test_snr{int(snr)}"
        paths = _paths(final_cfg["data"], label)
        ds = _dataset(
            paths, final_cfg["data"]["snr_rows_per_count"], run_cfg, normalizer
        )
        queries = detr_query_outputs(model, ds, device, normalizer, batch_size=batch_size)
        trues = true_compartments(ds)
        preds = predictions_from_query_outputs(queries, threshold)
        metrics = compute_metrics(preds, trues, n_queries=n_queries)
        metrics["snr"] = float(snr)
        metrics["exist_thresh"] = threshold
        metrics["extrapolation"] = float(snr) < train_snr_min
        metrics |= signal_reconstruction_metrics(
            preds, trues, ds, run_cfg.data.signal_norm
        )
        ladder[label] = metrics
        log(f"[baseline-finalize] SNR {snr:g}: count accuracy={metrics['count_accuracy']:.3f}")
    save_json(ladder, output / "metrics_snr_ladder.json")

    # 5. All thesis-ready baseline figures are scripted and reproducible.
    with open(history_path) as f:
        history = json.load(f)
    plot_training_curves(history, figures / "training_curves.png")
    scatter_figure(
        test_preds, test_trues, figures / "matched_t1_t2.png",
        title=f"Final baseline test (existence threshold {threshold:.2f})",
    )
    plot_count_confusion(test_metrics, figures / "count_confusion.png")
    plot_error_by_count(test_metrics, figures / "errors_by_true_count.png")
    plot_snr_robustness(ladder, figures / "snr_robustness.png")
    plot_query_specialization(
        query_report, matched_records, figures / "query_specialization.png"
    )
    plot_examples(examples, figures / "success_failure_examples.png")

    elapsed = time.perf_counter() - started
    repo = Path(__file__).resolve().parents[2]
    provenance = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "finalization_config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
            "contents": final_cfg,
        },
        "trained_run": {
            "run_config": str(run_config_path),
            "run_config_sha256": _sha256(run_config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_epoch_zero_based": checkpoint.get("epoch"),
            "checkpoint_validation_loss": checkpoint.get("val"),
            "history": str(history_path),
            "history_sha256": _sha256(history_path),
            "model_parameters": int(sum(p.numel() for p in model.parameters())),
            "train_seed": int(run_cfg.train.seed),
        },
        "data": data_provenance,
        "evaluation": {
            "validation_voxels": len(val_ds),
            "test_voxels": len(test_ds),
            "compartment_counts": counts,
            "selected_threshold": threshold,
            "selection_split": "validation",
            "threshold_grid": [float(thresholds[0]), float(thresholds[-1]), len(thresholds)],
            "device": str(device),
            "device_info": device_info(),
            "wall_seconds": float(elapsed),
        },
        "runtime_versions": _runtime_versions(),
        "git": _git_state(repo),
    }
    save_json(provenance, output / "provenance.json")

    summary = {
        "name": final_cfg["name"],
        "status": "complete",
        "source_checkpoint": str(checkpoint_path),
        "selected_existence_threshold": threshold,
        "validation_count_accuracy_at_selected_threshold": calibration["selected"]["count_accuracy"],
        "test": {
            key: test_metrics[key]
            for key in (
                "n_voxels",
                "count_accuracy",
                "count_mae",
                "existence_precision",
                "existence_recall",
                "existence_f1",
                "t1_rel_median",
                "t2_rel_median",
                "w_mae",
                "t1_abs_mean_ms",
                "t2_abs_mean_ms",
                "w_abs_mean",
                "t2_ge_t1_rate",
                "weight_sum_dev_median",
                "signal_rmse_prediction_vs_observed_median",
                "signal_rmse_truth_vs_observed_median",
            )
        },
        "output_dir": str(output),
        "wall_seconds": float(elapsed),
    }
    save_json(summary, output / "baseline_summary.json")
    log(f"[baseline-finalize] complete -> {output} ({elapsed:.1f}s)")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Finalize a trained T1T2-DETR baseline.")
    parser.add_argument("--config", required=True, help="Baseline finalization YAML.")
    parser.add_argument("--output-dir", default=None, help="Override the YAML output directory.")
    parser.add_argument(
        "--allow-existing", action="store_true",
        help="Allow writing into an existing output directory (files with the same names change).",
    )
    args = parser.parse_args()
    summary = finalize_baseline(
        args.config, output_dir=args.output_dir, allow_existing=args.allow_existing
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
