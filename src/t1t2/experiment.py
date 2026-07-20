"""One command: config in, trained model + evaluation + artifacts out.

This is the thin orchestrator the cluster job actually calls. It trains from a YAML, evaluates
the trained model on the held-out test split, and drops everything into results/<name>/ so a
run is self-describing after the fact.

    PYTHONPATH=src python -m t1t2.experiment --config configs/baseline.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import load_config
from .data import TargetNormalizer, VoxelDataset
from .device import get_device
from .eval import evaluate_detr, evaluate_snr_ladder
from .train import train


def run_experiment(config_path, results_dir=None, max_epochs=None, limit=None,
                   resume=True, log=print):
    """Train, evaluate, and summarize one experiment end to end."""
    cfg = load_config(config_path)
    results_dir = Path(results_dir) if results_dir else Path("results") / cfg.name

    history, results_dir, model = train(
        cfg, results_dir=results_dir, max_epochs=max_epochs, resume=resume, limit=limit, log=log,
    )
    # train() hands back the best model, not the final epoch — nothing to reload here.

    # best.pt is the authority: train() selects it using early_stopping_min_delta. Recomputing a
    # strict minimum from history can report a different epoch from the model actually evaluated.
    best_path = Path(results_dir) / "checkpoints" / "best.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=True) if best_path.exists() else None
    summary = {
        "name": cfg.name,
        "epochs_run": len(history),
        "epoch_budget": max_epochs if max_epochs is not None else cfg.train.epochs,
        "best_epoch": None if best is None else int(best["epoch"]) + 1,
        "best_val": None if best is None else float(best["val"]),
        "early_stopped": len(history) < (max_epochs if max_epochs is not None else cfg.train.epochs),
        # Reported alongside best_val because the data-scaling arms see very different numbers of
        # updates per epoch; comparing them at equal epochs would compare different things.
        "total_steps": history[-1].get("cum_steps") if history else 0,
        "wall_seconds": round(sum(h.get("seconds", 0) for h in history), 1),
    }

    # Evaluate on the test split (fall back to val, then train, whichever exists).
    test_path = cfg.data.test_path or cfg.data.val_path or cfg.data.train_path
    normalizer = TargetNormalizer.from_config(cfg.data)
    test_ds = VoxelDataset(test_path, cfg.data, normalizer, limit=limit)
    device = get_device(cfg.train.device)

    log(f"[{cfg.name}] evaluating DETR on {test_path} ({len(test_ds)} voxels)")
    summary["detr"] = evaluate_detr(model, test_ds, device, normalizer, results_dir,
                                    n_queries=cfg.model.n_queries)

    # Per-SNR, if the fixed-SNR rungs sit next to the test files. They are a paired set (same
    # voxels, same noise pattern, only the amplitude differs), so differences across rungs are
    # the SNR effect rather than sampling variation.
    ladder = _snr_ladder_paths(test_path)
    if ladder:
        log(f"[{cfg.name}] scoring the fixed-SNR ladder: {', '.join(sorted(ladder))}")
        summary["snr_ladder"] = evaluate_snr_ladder(
            model, ladder, cfg.data, device, normalizer, results_dir,
            train_snr_min=_train_snr_min(cfg),
            n_queries=cfg.model.n_queries, limit=limit,
        )

    with open(Path(results_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[{cfg.name}] done -> {results_dir}")
    return summary


def _train_snr_min(cfg) -> float | None:
    """Read the training-SNR floor recorded beside the generated data."""
    try:
        paths = cfg.data.train_path
        first = paths if isinstance(paths, str) else paths[0]
        manifest = Path(first).with_name("manifest.json")
        with open(manifest) as f:
            return float(json.load(f)["splits"]["train"]["snr_min"])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _snr_ladder_paths(test_path) -> dict:
    """Find test_snr*.parquet siblings of the test split, grouped by rung.

    Returns {rung_label: [path per compartment count]} so each rung is scored across all n at
    once, mirroring how the test split itself is assembled.
    """
    paths = [test_path] if isinstance(test_path, str) else list(test_path)
    ladder: dict[str, list[str]] = {}
    for p in paths:
        for rung in sorted(Path(p).parent.glob("test_snr*.parquet")):
            ladder.setdefault(rung.stem, []).append(str(rung))
    return ladder


def main():
    ap = argparse.ArgumentParser(description="Train + evaluate one T1T2-DETR experiment.")
    ap.add_argument("--config", required=True, help="Path to the experiment YAML.")
    ap.add_argument("--results-dir", default=None, help="Override results/<name>/.")
    ap.add_argument("--max-epochs", type=int, default=None, help="Cap epochs (smoke runs).")
    ap.add_argument("--limit", type=int, default=None, help="Cap voxels per split (smoke runs).")
    ap.add_argument("--no-resume", action="store_true", help="Ignore any existing checkpoint.")
    a = ap.parse_args()

    summary = run_experiment(
        a.config, results_dir=a.results_dir, max_epochs=a.max_epochs, limit=a.limit,
        resume=not a.no_resume,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
