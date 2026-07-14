"""Architecture smoke tests — shapes and gradients fit together. No training here.

These prove the pieces built in this milestone actually connect: config loads, the dataset
yields the tensors the model expects, a forward pass has the right shape, and the Hungarian
loss produces a finite scalar with a gradient. Run with:

    cd t1t2_training && PYTHONPATH=src python -m pytest tests/ -q
"""
from __future__ import annotations

from pathlib import Path

import torch

from t1t2.config import DataConfig, ExperimentConfig, LossConfig, ModelConfig, TrainConfig, load_config
from t1t2.data import TargetNormalizer, VoxelDataset, make_dataloader
from t1t2.loss import HungarianLoss
from t1t2.model import build_model

ROOT = Path(__file__).resolve().parents[1]
DEV_TRAIN = ROOT / "data" / "dev" / "train.parquet"


def _cfg() -> ExperimentConfig:
    return ExperimentConfig(
        name="smoke",
        data=DataConfig(train_path=str(DEV_TRAIN)),
        model=ModelConfig(),
        loss=LossConfig(),
        train=TrainConfig(),
    )


def test_config_roundtrip(tmp_path):
    cfg = load_config(ROOT / "configs" / "baseline.yaml")
    assert cfg.data.n_inputs == 64 and cfg.model.n_queries == 10
    p = tmp_path / "c.yaml"
    cfg.save(p)
    assert load_config(p).name == cfg.name


def test_normalizer_roundtrip():
    import numpy as np
    nz = TargetNormalizer(mode="log_minmax")
    x = np.array([50.0, 800.0, 3000.0])
    back = nz.denormalize_t1(nz.normalize_t1(x, clip=False))
    assert np.allclose(back, x, rtol=1e-9)


def test_dataset_shapes():
    cfg = _cfg()
    ds = VoxelDataset(cfg.data.train_path, cfg.data, limit=128)
    X, y, nc = ds[0]
    assert X.shape == (64,)
    assert y.shape == (cfg.data.max_comp * 3,)
    assert 1 <= int(nc) <= cfg.data.max_comp
    # normalized targets live in [0, 1]; padding is inert zeros
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_forward_and_loss_shapes():
    cfg = _cfg()
    loader, _ = make_dataloader(cfg.data.train_path, cfg.data, batch_size=16, shuffle=False, limit=64)
    model = build_model(cfg.model)
    crit = HungarianLoss(cfg.loss)
    X, y, nc = next(iter(loader))
    out = model(X)
    assert out.shape == (16, cfg.model.n_queries, 4)
    loss, l1, l2, lw, le = crit(out, y, nc)
    assert loss.dim() == 0 and torch.isfinite(loss)
    # gradient flows
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_aux_loss_path():
    cfg = _cfg()
    cfg.model.aux_loss = True
    model = build_model(cfg.model)
    X = torch.randn(8, 64)
    out = model(X)
    assert isinstance(out, dict) and "pred" in out and "aux" in out
    assert out["pred"].shape == (8, cfg.model.n_queries, 4)
    assert len(out["aux"]) == cfg.model.n_dlayers
