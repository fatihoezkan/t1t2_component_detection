from pathlib import Path

import numpy as np
import torch

from t1t2.config import (
    DataConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    TrainConfig,
    load_config,
)
from t1t2.eval import (
    calibrate_existence_threshold,
    parameter_recovery_analysis,
)
from t1t2.loss import HungarianLoss
from t1t2.train import train


ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "data" / "dev_1to4"


def _two_compartment_loss(mode):
    cfg = LossConfig(
        t1_weight=1.0,
        t2_weight=0.0,
        w_weight=0.0,
        exist_weight=0.0,
        t1_t2_weighting=mode,
    )
    # Correct assignment: q0 -> target0 has squared T1 error .25 and true weight .9;
    # q1 -> target1 is exact. T2 and weights are exact and disabled in the total.
    pred = torch.tensor([[
        [.5, 0.0, .9, 10.0],
        [1.0, 0.0, .1, 10.0],
    ]])
    target = torch.tensor([[0.0, 0.0, .9, 1.0, 0.0, .1]])
    _, t1, *_ = HungarianLoss(cfg)(pred, target, torch.tensor([2]))
    return float(t1)


def test_signal_fraction_loss_uses_weight_sum_not_compartment_count():
    assert np.isclose(_two_compartment_loss("legacy"), .25 * .9 / 2)
    assert np.isclose(_two_compartment_loss("signal_fraction"), .25 * .9)
    assert np.isclose(_two_compartment_loss("uniform"), .25 / 2)


def test_old_configs_keep_legacy_training_and_evaluation_defaults():
    cfg = load_config(ROOT / "configs" / "cluster.yaml")
    assert cfg.loss.t1_t2_weighting == "legacy"
    assert cfg.train.selection_metric == "total_loss"
    assert cfg.train.lr_scheduler == "constant"
    assert cfg.train.gradient_clip_norm is None
    assert cfg.evaluation.calibrate_threshold is False
    assert cfg.evaluation.fixed_threshold == .5


def test_parameter_recovery_leads_with_closeness_and_recovered_fraction():
    trues = [[(500.0, 50.0, .9), (1000.0, 100.0, .1)]]
    preds = [[(500.0, 50.0, .9)]]  # exact dominant pool, weak pool missed
    analysis = parameter_recovery_analysis(preds, trues)
    summary = analysis["summary"]
    assert np.isclose(summary["recovered_signal_fraction"], .9)
    assert summary["t1_fraction_weighted_relative_error_matched"] == 0
    assert summary["t2_fraction_weighted_relative_error_matched"] == 0
    assert np.isclose(summary["weight_set_l1_error_mean"], .1)
    weak_bin = next(b for b in analysis["bins"] if b["weight_min"] == .1)
    assert weak_bin["match_rate"] == 0


def test_parameter_threshold_can_prefer_closeness_over_exact_count():
    outputs = {
        "params": np.array([[
            [1000.0, 100.0, 1.0],  # bad but very confident
            [100.0, 10.0, 1.0],    # exact but less confident
        ]]),
        "exist_prob": np.array([[.9, .6]]),
    }
    trues = [[(100.0, 10.0, 1.0)]]
    parameter_choice = calibrate_existence_threshold(
        outputs, trues, thresholds=[.5, .7], objective="parameter_set_error"
    )
    count_choice = calibrate_existence_threshold(
        outputs, trues, thresholds=[.5, .7], objective="count_accuracy"
    )
    assert parameter_choice["selected_threshold"] == .5  # exact parameters + one extra
    assert count_choice["selected_threshold"] == .7      # correct count but wrong parameters


def test_new_long_config_is_isolated_and_parameter_first():
    cfg = load_config(ROOT / "configs" / "t1_3500_t2_500_weighted_long.yaml")
    assert cfg.name == "t1_3500_t2_500_weighted_long"
    assert all("data/t1_3500_t2_500_100k/" in p for p in cfg.data.train_path)
    assert (cfg.data.t1_min, cfg.data.t1_max) == (50.0, 3500.0)
    assert (cfg.data.t2_min, cfg.data.t2_max) == (5.0, 500.0)
    assert cfg.loss.t1_t2_weighting == "signal_fraction"
    assert cfg.train.epochs == 500
    assert cfg.train.selection_metric == "parameter_loss"
    assert cfg.train.lr_scheduler == "reduce_on_plateau"
    assert cfg.evaluation.threshold_objective == "parameter_set_error"


def test_parameter_selection_scheduler_and_resume_metadata(tmp_path):
    split = lambda name: [str(DEV / f"n{n}" / f"{name}.parquet") for n in (1, 2, 3)]
    cfg = ExperimentConfig(
        name="parameter_smoke",
        data=DataConfig(train_path=split("train"), val_path=split("val")),
        model=ModelConfig(hidden_dim=16, fs_dim=8, n_queries=4, n_dlayers=1, n_heads=2),
        loss=LossConfig(t1_t2_weighting="signal_fraction"),
        train=TrainConfig(
            epochs=2,
            batch_size=16,
            early_stopping=False,
            device="cpu",
            selection_metric="parameter_loss",
            lr_scheduler="reduce_on_plateau",
            scheduler_patience=1,
            gradient_clip_norm=1.0,
        ),
    )
    history, _, _ = train(
        cfg, results_dir=tmp_path / "new_run", limit=48, resume=False, log=lambda *_: None
    )
    assert all("parameter_loss" in h["train"] and "parameter_loss" in h["val"] for h in history)
    best = torch.load(
        tmp_path / "new_run" / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=True,
    )
    last = torch.load(
        tmp_path / "new_run" / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert best["selection_metric"] == "parameter_loss"
    assert "parameter_loss" in best and "val_loss" in best
    assert last["scheduler"] is not None
