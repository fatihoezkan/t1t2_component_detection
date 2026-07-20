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
from t1t2.experiment import run_experiment
from t1t2.physics import forward_numpy, forward_torch, load_protocol
from t1t2.train import train

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "data" / "dev_1to4"


def _split(name: str) -> list[str]:
    return [str(DEV / f"n{n}" / f"{name}.parquet") for n in range(1, 5)]


def _cfg(**train_kw) -> ExperimentConfig:
    train_kw.setdefault("early_stopping", False)     # most tests want a fixed epoch count
    return ExperimentConfig(
        name="pipe",
        data=DataConfig(
            train_path=_split("train"),
            val_path=_split("val"),
            test_path=_split("test"),
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
    assert [h["epoch"] for h in hist2] == [0, 1, 2, 3]       # history not truncated by the resume


def test_early_stopping_fires_and_returns_the_best_model(tmp_path):
    """Early stopping must stop, and the returned model must be the best one — not the last.

    Evaluating the final epoch was the old behaviour and it silently reports the wrong model
    whenever the last epoch is not the best, which is most of the time.
    """
    cfg = _cfg(epochs=20, batch_size=128, ckpt_every=1,
               early_stopping=True, early_stopping_patience=2,
               early_stopping_min_delta=10.0)      # nothing counts as an improvement
    hist, rd, model = train(cfg, results_dir=tmp_path / "run", limit=256, resume=False,
                            log=lambda *a: None)

    assert len(hist) == 3, f"expected 1 attempt + patience 2, got {len(hist)} epochs"
    best = tmp_path / "run" / "checkpoints" / "best.pt"
    assert best.exists()

    saved = torch.load(best, map_location="cpu")["model"]
    for k, v in model.state_dict().items():
        assert torch.equal(saved[k], v.cpu()), f"returned model differs from best.pt at {k}"


def test_experiment_reports_the_checkpoint_it_evaluates(tmp_path):
    """The summary must name best.pt, not a smaller loss ignored by min_delta."""
    cfg = _cfg(epochs=3, batch_size=64, early_stopping=True,
               early_stopping_patience=2, early_stopping_min_delta=10.0)
    config_path = tmp_path / "config.yaml"
    cfg.save(config_path)
    result_dir = tmp_path / "run"
    summary = run_experiment(config_path, results_dir=result_dir, limit=64,
                             resume=False, log=lambda *a: None)
    selected = torch.load(result_dir / "checkpoints" / "best.pt", map_location="cpu")
    assert summary["best_epoch"] == int(selected["epoch"]) + 1
    assert summary["best_val"] == float(selected["val"])
    assert summary["snr_ladder"]["test_snr20"]["extrapolation"] is True


def test_early_stopping_needs_a_val_split(tmp_path):
    """Without validation there is nothing to select on, so it must disable itself, not crash."""
    cfg = _cfg(epochs=2, batch_size=128, early_stopping=True, early_stopping_patience=1)
    cfg.data.val_path = None
    hist, _, _ = train(cfg, results_dir=tmp_path / "run", limit=128, resume=False,
                       log=lambda *a: None)
    assert len(hist) == 2 and all(h["val"] == {} for h in hist)


def test_resume_refuses_a_different_config(tmp_path):
    """Checkpoints are keyed only by directory, so a changed config would blend two experiments."""
    import pytest

    cfg = _cfg(epochs=1, batch_size=128, ckpt_every=1)
    train(cfg, results_dir=tmp_path / "run", limit=128, log=lambda *a: None)

    changed = _cfg(epochs=1, batch_size=128, ckpt_every=1)
    changed.model.n_queries = 12
    with pytest.raises(ValueError, match="different config"):
        train(changed, results_dir=tmp_path / "run", limit=128, log=lambda *a: None)


def test_checkpoint_metadata_is_plain_python(tmp_path):
    """torch>=2.6 loads with weights_only=True; a numpy scalar in here breaks resume on the
    cluster and nowhere else."""
    cfg = _cfg(epochs=1, batch_size=128, ckpt_every=1)
    train(cfg, results_dir=tmp_path / "run", limit=128, log=lambda *a: None)
    state = torch.load(tmp_path / "run" / "checkpoints" / "last.pt", map_location="cpu",
                       weights_only=True)
    assert type(state["best_val"]) is float
    assert type(state["best_epoch"]) is int and type(state["bad_epochs"]) is int


def test_train_limit_per_path_leaves_val_alone(tmp_path):
    """The data-scaling arms must all be scored on the same validation set."""
    from t1t2.data import VoxelDataset

    cfg = _cfg(epochs=1, batch_size=64)
    cfg.data.train_limit_per_path = 16
    train(cfg, results_dir=tmp_path / "run", log=lambda *a: None)

    val_full = VoxelDataset(cfg.data.val_path, cfg.data)
    assert len(val_full) == sum(
        len(VoxelDataset(p, cfg.data)) for p in cfg.data.val_path
    ), "val was capped by the train-only limit"


def test_eval_produces_metrics(tmp_path):
    cfg = _cfg(epochs=1, batch_size=128)
    _, rd, model = train(cfg, results_dir=tmp_path / "run", limit=256, log=lambda *a: None)
    ds = VoxelDataset(cfg.data.test_path, cfg.data, limit=256)
    m = evaluate_detr(model, ds, get_device(None), TargetNormalizer.from_config(cfg.data), tmp_path / "run")
    for key in ("count_accuracy", "t1_rel_median", "t2_rel_median_csf", "t2_rel_median_noncsf"):
        assert key in m
    assert (tmp_path / "run" / "figures" / "scatter_detr.png").exists()

    # Per-n is mandatory: an aggregate across counts averages easy and hard regimes.
    for n in (1, 2, 3, 4):
        assert f"count_accuracy_n{n}" in m and f"n_voxels_n{n}" in m
    assert sum(m[f"n_voxels_n{n}"] for n in (1, 2, 3, 4)) == m["n_voxels"]

    # the count-correct conditional block, and the physics checks
    assert "cc_t1_rel_median" in m and "cc_n_voxels" in m
    assert "t2_ge_t1_rate" in m and "weight_sum_dev_median" in m
    assert m["exist_thresh"] == 0.5          # never tuned on the split being reported


def test_confusion_matrix_shape_and_totals():
    """The matrix is deliberately not square: 10 queries means predicted count can be 0..10."""
    from t1t2.eval import count_confusion

    trues = [[(500.0, 50.0, 1.0)], [(500.0, 50.0, 0.5), (900.0, 90.0, 0.5)]]
    preds = [[(500.0, 50.0, 1.0)], []]                       # right, then a total miss
    c = count_confusion(preds, trues, n_queries=10)

    assert c["true_counts"] == [1, 2] and c["predicted_range"] == [0, 10]
    assert len(c["matrix"]["1"]) == 11
    assert c["matrix"]["1"][1] == 1                          # true 1 -> predicted 1
    assert c["matrix"]["2"][0] == 1                          # true 2 -> predicted 0
    assert sum(sum(row) for row in c["matrix"].values()) == len(trues)


def test_physics_violations_are_measured_not_fixed():
    """Independent sigmoid heads can emit T2 >= T1; we report that rather than clamp it."""
    from t1t2.eval import physics_violations

    preds = [[(500.0, 900.0, 0.5)],                          # T2 > T1: unphysical; sum(w)=0.5
             [(500.0, 50.0, 0.4), (900.0, 90.0, 0.4)]]       # fine, but sum(w)=0.8
    v = physics_violations(preds)
    assert v["t2_ge_t1_rate"] == 1 / 3                        # 1 of 3 compartments
    assert abs(v["weight_sum_dev_median"] - 0.35) < 1e-9      # median(|0.5-1|, |0.8-1|) = median(.5,.2)


def test_snr_ladder_is_scored_per_rung_and_flags_extrapolation(tmp_path):
    from t1t2.eval import evaluate_snr_ladder

    cfg = _cfg(epochs=1, batch_size=128)
    _, _, model = train(cfg, results_dir=tmp_path / "run", limit=128, log=lambda *a: None)

    paths = {f"test_snr{s}": [str(DEV / f"n{n}" / f"test_snr{s}.parquet") for n in range(1, 5)]
             for s in (20, 150)}
    out = evaluate_snr_ladder(model, paths, cfg.data, get_device(None),
                              TargetNormalizer.from_config(cfg.data), tmp_path / "run",
                              train_snr_min=30.0, limit=64)

    assert out["test_snr20"]["extrapolation"] is True         # below the training range
    assert out["test_snr150"]["extrapolation"] is False
    assert out["test_snr20"]["snr"] == 20.0
    assert (tmp_path / "run" / "metrics_snr_ladder.json").exists()
