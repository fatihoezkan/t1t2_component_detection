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

from .config import load_config
from .data import TargetNormalizer, VoxelDataset
from .device import get_device
from .eval import evaluate_detr
from .train import train


def run_experiment(config_path, results_dir=None, max_epochs=None, limit=None,
                   resume=True, log=print):
    """Train, evaluate, and summarize one experiment end to end."""
    cfg = load_config(config_path)
    results_dir = Path(results_dir) if results_dir else Path("results") / cfg.name

    history, results_dir, model = train(
        cfg, results_dir=results_dir, max_epochs=max_epochs, resume=resume, limit=limit, log=log,
    )

    summary = {"name": cfg.name, "epochs_run": len(history)}

    # Evaluate on the test split (fall back to val, then train, whichever exists).
    test_path = cfg.data.test_path or cfg.data.val_path or cfg.data.train_path
    normalizer = TargetNormalizer.from_config(cfg.data)
    test_ds = VoxelDataset(test_path, cfg.data, normalizer, limit=limit)
    device = get_device(cfg.train.device)

    log(f"[{cfg.name}] evaluating DETR on {test_path} ({len(test_ds)} voxels)")
    summary["detr"] = evaluate_detr(model, test_ds, device, normalizer, results_dir)

    with open(Path(results_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[{cfg.name}] done -> {results_dir}")
    return summary


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
