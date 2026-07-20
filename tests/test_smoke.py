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
DEV_DIR = ROOT / "data" / "dev_1to4"
DEV_TRAIN = [str(DEV_DIR / f"n{n}" / "train.parquet") for n in range(1, 5)]
MAX_COMP = 4        # what the dev data carries; the loader reads it off the columns


def _cfg() -> ExperimentConfig:
    return ExperimentConfig(
        name="smoke",
        data=DataConfig(train_path=DEV_TRAIN),
        model=ModelConfig(),
        loss=LossConfig(),
        train=TrainConfig(),
    )


def test_config_roundtrip(tmp_path):
    cfg = load_config(ROOT / "configs" / "baseline.yaml")
    assert cfg.data.n_inputs == 64 and cfg.model.n_queries == 10
    assert len(cfg.data.train_path) == 3
    assert all(f"/n{n}/" in p for n, p in enumerate(cfg.data.train_path, start=1))
    p = tmp_path / "c.yaml"
    cfg.save(p)
    assert load_config(p).name == cfg.name


def test_cluster_config_is_the_exact_100k_baseline():
    cfg = load_config(ROOT / "configs" / "cluster.yaml")
    assert cfg.data.train_limit_per_path is None
    assert len(cfg.data.train_path) == len(cfg.data.val_path) == len(cfg.data.test_path) == 3
    assert all("data/baseline_100k/" in p for p in cfg.data.train_path)
    assert cfg.model.input_dim == cfg.data.n_inputs == 64
    assert cfg.model.aux_loss is False and cfg.loss.signal_consistency is False


def test_normalizer_roundtrip():
    import numpy as np
    nz = TargetNormalizer(mode="log_minmax")
    x = np.array([50.0, 800.0, 3000.0])
    back = nz.denormalize_t1(nz.normalize_t1(x, clip=False))
    assert np.allclose(back, x, rtol=1e-9)


def test_dataset_shapes():
    cfg = _cfg()
    ds = VoxelDataset(cfg.data.train_path, cfg.data, limit=128)
    assert ds.max_comp == MAX_COMP                  # inferred from the columns, not configured
    X, y, nc = ds[0]
    assert X.shape == (64,)
    assert y.shape == (MAX_COMP * 3,)
    assert 1 <= int(nc) <= MAX_COMP
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


# --------------------------------------------------------------------------------------
# Schema inference: the compartment width comes from the data, and cannot be misconfigured.
# --------------------------------------------------------------------------------------

def test_max_comp_is_not_configurable():
    """A config copy of the table width could go stale against the data; there must not be one."""
    assert not hasattr(DataConfig(train_path="x"), "max_comp")


def test_infer_max_comp_reads_the_columns():
    import pandas as pd

    from t1t2.data import infer_max_comp
    cols = {"n_comp": [1]}
    for i in (1, 2, 3):
        cols |= {f"T1_{i}": [1.0], f"T2_{i}": [1.0], f"w_{i}": [1.0]}
    assert infer_max_comp(pd.DataFrame(cols)) == 3


def _toy_frame(k, n_comp=1, skip=None):
    import pandas as pd
    d = {"n_comp": [n_comp]} | {f"S_{i+1}": [0.1] for i in range(64)}
    for i in range(1, k + 1):
        if i == skip:
            continue
        d |= {f"T1_{i}": [500.0], f"T2_{i}": [50.0], f"w_{i}": [1.0]}
    return pd.DataFrame(d)


def test_infer_max_comp_rejects_malformed_schemas():
    import pytest

    from t1t2.data import infer_max_comp

    with pytest.raises(ValueError, match="contiguous"):          # T1_1, T1_3 but no T1_2
        infer_max_comp(_toy_frame(3, skip=2))

    with pytest.raises(ValueError, match="families disagree"):   # w_2 missing
        infer_max_comp(_toy_frame(2).drop(columns=["w_2"]))


def test_loader_rejects_n_comp_beyond_the_available_slots(tmp_path):
    """The silent-corruption path, closed.

    With a configured width this combination trained a model that structurally could not count
    past the configured number, while reporting perfectly plausible metrics: the cost matrix was
    built at the config width and numpy happily returns 2 columns when asked for 4.
    """
    import pytest

    p = tmp_path / "over.parquet"
    _toy_frame(2, n_comp=4).to_parquet(p, index=False)
    with pytest.raises(ValueError, match="only 2 ground-truth column slots"):
        VoxelDataset(str(p), DataConfig(train_path=str(p)))


def test_multi_path_limit_splits_evenly_across_counts():
    """Generic multi-file loading still reaches every supplied count, including stress-test n=4."""
    cfg = _cfg()
    ds = VoxelDataset(cfg.data.train_path, cfg.data, limit=40)
    assert len(ds) == 40
    counts = sorted(set(int(c) for c in ds.n_comp))
    assert counts == [1, 2, 3, 4], f"limit did not reach every per-n file: {counts}"


def test_signal_norm_max_puts_every_peak_at_one():
    cfg = _cfg()
    cfg.data.signal_norm = "max"
    ds = VoxelDataset(cfg.data.train_path, cfg.data, limit=64)
    peaks = ds.X.abs().max(dim=1).values
    assert torch.allclose(peaks, torch.ones_like(peaks), atol=1e-5)


def test_normalizer_bounds_match_the_generator():
    """A mismatch clamps real targets, so the model could never reach the edges of the space."""
    from voxel_simulator.sampler import T1_RANGE, T2_RANGE

    cfg = load_config(ROOT / "configs" / "baseline.yaml").data
    assert (cfg.t1_min, cfg.t1_max) == T1_RANGE
    assert (cfg.t2_min, cfg.t2_max) == T2_RANGE
