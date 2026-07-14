"""Pipeline tests — physics parity, training loop, evaluation. Smoke-scale.

These exercise the M3 pieces (physics/train/eval) end to end on tiny data. They are
smoke tests, not the thesis training run — that happens on the cluster.

    cd t1t2_training && PYTHONPATH=src python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from t1t2.config import DataConfig, ExperimentConfig, LossConfig, ModelConfig, TrainConfig
from t1t2.eval import evaluate_detr
from t1t2.data import TargetNormalizer, VoxelDataset
from t1t2.device import get_device
from t1t2.physics import forward_numpy, forward_torch, load_protocol
from t1t2.train import train

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "data" / "dev"


def _cfg(**train_kw) -> ExperimentConfig:
    return ExperimentConfig(
        name="pipe",
        data=DataConfig(
            train_path=str(DEV / "train.parquet"),
            val_path=str(DEV / "val.parquet"),
            test_path=str(DEV / "test.parquet"),
        ),
        model=ModelConfig(),
        loss=LossConfig(),
        train=TrainConfig(**train_kw),
    )


def test_physics_parity_with_generator():
    """Our numpy forward must match the vendored generator bit-for-bit."""
    sys.path.insert(0, str(ROOT / "voxel_generator" / "src"))
    from voxel_simulator.physics import simulate_clean_signal
    from voxel_simulator.protocol import load_protocol as gen_load

    gp = gen_load()
    proto = load_protocol()
    t1, t2, w = np.array([1200.0, 300.0]), np.array([80.0, 40.0]), np.array([0.7, 0.3])
    ref = simulate_clean_signal(gp, t1, t2, w)
    ours = forward_numpy(proto, t1, t2, w)
    assert np.allclose(ref, ours, rtol=1e-12, atol=1e-12)


def test_forward_torch_matches_numpy():
    proto = load_protocol()
    t1, t2, w = [1200.0, 300.0], [80.0, 40.0], [0.7, 0.3]
    ours = forward_numpy(proto, t1, t2, w)
    params = torch.tensor([[[1200.0, 80.0, 0.7], [300.0, 40.0, 0.3]]])   # (1, 2, 3)
    tout = forward_torch(proto, params)[0].numpy()
    assert np.allclose(ours, tout, rtol=1e-5, atol=1e-5)


def test_train_smoke_and_resume(tmp_path):
    cfg = _cfg(epochs=2, batch_size=128, ckpt_every=1)
    hist, rd, model = train(cfg, results_dir=tmp_path / "run", limit=256, log=lambda *a: None)
    assert len(hist) == 2
    assert np.isfinite(hist[-1]["train"]["loss"])
    assert (tmp_path / "run" / "checkpoints" / "last.pt").exists()
    # resume: continue to epoch 4 from the checkpoint
    hist2, _, _ = train(cfg, results_dir=tmp_path / "run", max_epochs=4, limit=256, log=lambda *a: None)
    assert len(hist2) == 4


def test_eval_produces_metrics(tmp_path):
    cfg = _cfg(epochs=1, batch_size=128)
    _, rd, model = train(cfg, results_dir=tmp_path / "run", limit=256, log=lambda *a: None)
    ds = VoxelDataset(cfg.data.test_path, cfg.data, limit=256)
    m = evaluate_detr(model, ds, get_device(None), TargetNormalizer.from_config(cfg.data), tmp_path / "run")
    for key in ("count_accuracy", "t1_rel_median", "t2_rel_median_csf", "t2_rel_median_noncsf"):
        assert key in m
    assert (tmp_path / "run" / "figures" / "scatter_detr.png").exists()
