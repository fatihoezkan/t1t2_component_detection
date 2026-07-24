import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from t1t2.baseline import finalize_baseline
from t1t2.baseline_analysis import (
    query_specialization,
    signal_reconstruction_metrics,
)
from t1t2.config import load_config
from t1t2.eval import (
    calibrate_existence_threshold,
    compute_metrics,
)
from t1t2.model import build_model
from t1t2.physics import forward_numpy, load_protocol


ROOT = Path(__file__).resolve().parents[1]


def _query_outputs(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    n, q = probabilities.shape
    params = np.ones((n, q, 3), dtype=float)
    params[..., 0] = np.linspace(200, 1200, q)
    params[..., 1] = np.linspace(20, 120, q)
    params[..., 2] = 1 / q
    return {"params": params, "exist_prob": probabilities}


def test_threshold_is_selected_by_validation_count_accuracy():
    trues = [
        [(200.0, 20.0, 1.0)],
        [(200.0, 20.0, .5), (600.0, 60.0, .5)],
        [(200.0, 20.0, .3), (600.0, 60.0, .3), (1200.0, 120.0, .4)],
    ]
    outputs = _query_outputs([
        [.9, .4, .2, .1],
        [.9, .8, .4, .1],
        [.9, .8, .7, .2],
    ])
    calibrated = calibrate_existence_threshold(outputs, trues, thresholds=[.3, .5, .75])
    assert calibrated["selected_threshold"] == .5
    assert calibrated["selected"]["count_accuracy"] == 1.0


def test_metrics_report_existence_precision_and_recall():
    trues = [
        [(200.0, 20.0, 1.0)],
        [(300.0, 30.0, .5), (800.0, 80.0, .5)],
    ]
    preds = [
        [(200.0, 20.0, 1.0), (900.0, 90.0, .1)],  # one surplus
        [(300.0, 30.0, .5)],                        # one missed
    ]
    metrics = compute_metrics(preds, trues, n_queries=4)
    assert metrics["existence_tp"] == 2
    assert metrics["existence_fp"] == 1
    assert metrics["existence_fn"] == 1
    assert metrics["existence_precision"] == 2 / 3
    assert metrics["existence_recall"] == 2 / 3
    assert "t1_abs_mean_ms" in metrics
    assert "t1_abs_median_ms" in metrics


def test_signal_reconstruction_is_zero_for_exact_clean_prediction():
    protocol = load_protocol()
    true = [(900.0, 80.0, 1.0)]
    signal = forward_numpy(protocol, [900.0], [80.0], [1.0])
    signal = signal / np.max(np.abs(signal))
    ds = SimpleNamespace(
        X=torch.tensor(signal[None], dtype=torch.float32),
        cfg=SimpleNamespace(n_inputs=64),
        __len__=lambda self: 1,
    )
    # Special methods are resolved on the type, so make a tiny real class for len(ds).
    ds = type("SignalDataset", (), {
        "__len__": lambda self: 1,
        "X": ds.X,
        "cfg": ds.cfg,
    })()
    metrics = signal_reconstruction_metrics([true], [true], ds, "max", protocol)
    assert metrics["signal_rmse_prediction_vs_truth_median"] < 1e-12
    assert metrics["signal_rmse_prediction_vs_observed_median"] < 1e-7


def test_query_analysis_measures_activity_without_assuming_roles():
    trues = [[(200.0, 20.0, 1.0)], [(600.0, 60.0, 1.0)]]
    outputs = _query_outputs([[.9, .1], [.2, .8]])
    report, matched = query_specialization(outputs, trues, threshold=.5)
    assert report["n_queries"] == 2
    assert [q["active_count"] for q in report["queries"]] == [1, 1]
    assert {r["query"] for r in matched} == {0, 1}


def test_offline_finalizer_smoke_writes_complete_artifact_set(tmp_path):
    run_cfg = load_config(ROOT / "configs" / "baseline.yaml")
    run_cfg.name = "baseline_finalize_test"
    run_cfg.model.hidden_dim = 16
    run_cfg.model.fs_dim = 8
    run_cfg.model.n_queries = 4
    run_cfg.model.n_dlayers = 1
    run_cfg.model.n_heads = 2
    run_cfg.train.device = "cpu"
    run_cfg.train.num_workers = 0
    run_config_path = tmp_path / "trained_config.yaml"
    run_cfg.save(run_config_path)

    model = build_model(run_cfg.model)
    checkpoint_path = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "epoch": 0, "val": .1}, checkpoint_path)
    history_path = tmp_path / "history.json"
    history_path.write_text(json.dumps([{
        "epoch": 0,
        "train": {"loss": 1, "t1": .2, "t2": .2, "wt": .2, "ex": .4},
        "val": {"loss": .8, "t1": .2, "t2": .2, "wt": .2, "ex": .2},
        "seconds": .1,
        "steps": 1,
        "cum_steps": 1,
    }]))

    output = tmp_path / "final"
    final_config = {
        "name": "baseline_finalize_test",
        "source": {
            "run_config": str(run_config_path),
            "checkpoint": str(checkpoint_path),
            "history": str(history_path),
            "data_audit": str(ROOT / "results" / "data_audit" / "dev_1to4" / "audit_summary.json"),
        },
        "output_dir": str(output),
        "device": "cpu",
        "batch_size": 4,
        "data": {
            "family_root": str(ROOT / "data" / "dev_1to4"),
            "compartment_counts": [1, 2, 3],
            "expected_base_seed": 0,
            "train_snr_min": 30.0,
            "val_rows_per_count": 2,
            "test_rows_per_count": 2,
            "snr_rows_per_count": 2,
            "snr_levels": [20],
        },
        "threshold": {"min": .3, "max": .7, "steps": 5},
    }
    config_path = tmp_path / "finalize.yaml"
    config_path.write_text(yaml.safe_dump(final_config, sort_keys=False))

    summary = finalize_baseline(config_path, log=lambda *_: None)
    assert summary["status"] == "complete"
    assert summary["test"]["n_voxels"] == 6
    for artifact in (
        "baseline_summary.json",
        "metrics_test.json",
        "metrics_snr_ladder.json",
        "threshold_calibration.json",
        "query_analysis.json",
        "prediction_examples.json",
        "provenance.json",
        "validation_query_outputs.npz",
        "test_query_outputs.npz",
        "figures/training_curves.png",
        "figures/threshold_calibration.png",
        "figures/count_confusion.png",
        "figures/errors_by_true_count.png",
        "figures/snr_robustness.png",
        "figures/query_specialization.png",
        "figures/success_failure_examples.png",
        "figures/matched_t1_t2.png",
    ):
        assert (output / artifact).is_file(), artifact
