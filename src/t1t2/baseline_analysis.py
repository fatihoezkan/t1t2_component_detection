"""Reusable analysis and figure generation for a finalized DETR baseline.

The training loop deliberately stays small. This module contains the heavier offline work needed
for a thesis-quality result: signal reconstruction, query specialization, threshold curves, count
confusion, SNR plots, and deterministic success/failure examples.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .eval import _match
from .physics import Protocol, forward_numpy, load_protocol


def _finite_summary(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
    }


def _normalize_signal(signal: np.ndarray, mode: str) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if mode == "none":
        return signal
    if mode == "max":
        scale = float(np.max(np.abs(signal)))
    elif mode == "first":
        scale = float(signal[0])
    else:
        raise ValueError(f"unknown signal normalization {mode!r}")
    return signal if scale == 0 else signal / scale


def protocol_for_inputs(n_inputs: int) -> Protocol:
    """Load the protocol and retain its first ``n_inputs`` positions without reordering."""
    protocol = load_protocol()
    if n_inputs > protocol.n_points:
        raise ValueError(
            f"model requests {n_inputs} inputs but the stored protocol has {protocol.n_points}"
        )
    return Protocol(protocol.ti[:n_inputs], protocol.te[:n_inputs], protocol.tr)


def signal_reconstruction_metrics(preds, trues, ds, signal_norm: str, protocol=None) -> dict:
    """Compare predicted and true forward signals with the observed normalized input.

    ``truth_vs_observed`` is the empirical noise floor: even exact parameters cannot reconstruct
    the added noise. ``prediction_vs_truth`` isolates parameter error from that noise. All values
    are per-voxel RMSE on the same normalized scale seen by the network.
    """
    if not (len(preds) == len(trues) == len(ds)):
        raise ValueError("prediction, truth, and dataset lengths must agree")
    protocol = protocol or protocol_for_inputs(ds.cfg.n_inputs)
    observed = ds.X.detach().cpu().numpy().astype(np.float64)
    pred_obs, truth_obs, pred_truth = [], [], []
    for obs, pred, true in zip(observed, preds, trues):
        if pred:
            pp = np.asarray(pred, dtype=np.float64)
            pred_signal = forward_numpy(protocol, pp[:, 0], pp[:, 1], pp[:, 2])
        else:
            pred_signal = np.zeros(protocol.n_points, dtype=np.float64)
        tt = np.asarray(true, dtype=np.float64)
        true_signal = forward_numpy(protocol, tt[:, 0], tt[:, 1], tt[:, 2])
        pred_signal = _normalize_signal(pred_signal, signal_norm)
        true_signal = _normalize_signal(true_signal, signal_norm)
        pred_obs.append(np.sqrt(np.mean((pred_signal - obs) ** 2)))
        truth_obs.append(np.sqrt(np.mean((true_signal - obs) ** 2)))
        pred_truth.append(np.sqrt(np.mean((pred_signal - true_signal) ** 2)))
    p = _finite_summary(pred_obs)
    t = _finite_summary(truth_obs)
    pt = _finite_summary(pred_truth)
    return {
        "signal_rmse_prediction_vs_observed_mean": p["mean"],
        "signal_rmse_prediction_vs_observed_median": p["median"],
        "signal_rmse_prediction_vs_observed_p90": p["p90"],
        "signal_rmse_truth_vs_observed_mean": t["mean"],
        "signal_rmse_truth_vs_observed_median": t["median"],
        "signal_rmse_truth_vs_observed_p90": t["p90"],
        "signal_rmse_prediction_vs_truth_mean": pt["mean"],
        "signal_rmse_prediction_vs_truth_median": pt["median"],
        "signal_rmse_prediction_vs_truth_p90": pt["p90"],
        "signal_normalization": signal_norm,
    }


def query_specialization(query_outputs, trues, threshold: float):
    """Summarize which learned query is active and which true compartments it matches.

    The model is a set predictor, so this function does not assume query identities have a stable
    tissue meaning. It measures that question empirically and returns both JSON-ready summaries
    and matched records used by the visualization.
    """
    params = np.asarray(query_outputs["params"], dtype=np.float64)
    probs = np.asarray(query_outputs["exist_prob"], dtype=np.float64)
    if params.shape[:2] != probs.shape or len(params) != len(trues):
        raise ValueError("query outputs and truths have incompatible shapes")
    n_voxels, n_queries = probs.shape
    true_counts = np.asarray([len(t) for t in trues], dtype=int)
    active = probs > threshold
    matched_records = []
    per_query = []

    for i, true in enumerate(trues):
        qidx = np.flatnonzero(active[i])
        if not len(qidx) or not true:
            continue
        truth = np.asarray(true, dtype=np.float64)
        pred = params[i, qidx]
        cost = (
            (np.log(pred[:, None, 0]) - np.log(truth[None, :, 0])) ** 2
            + (np.log(pred[:, None, 1]) - np.log(truth[None, :, 1])) ** 2
        )
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            matched_records.append({
                "voxel": int(i),
                "query": int(qidx[r]),
                "true_count": int(len(true)),
                "true_t1": float(truth[c, 0]),
                "true_t2": float(truth[c, 1]),
                "true_w": float(truth[c, 2]),
                "pred_t1": float(pred[r, 0]),
                "pred_t2": float(pred[r, 1]),
                "pred_w": float(pred[r, 2]),
                "exist_prob": float(probs[i, qidx[r]]),
            })

    for q in range(n_queries):
        is_active = active[:, q]
        rec = [r for r in matched_records if r["query"] == q]
        matched_t1 = [r["true_t1"] for r in rec]
        matched_t2 = [r["true_t2"] for r in rec]
        matched_w = [r["true_w"] for r in rec]
        pred_active = params[is_active, q]
        row = {
            "query": q,
            "mean_existence_probability": float(np.mean(probs[:, q])) if n_voxels else float("nan"),
            "active_count": int(is_active.sum()),
            "active_rate": float(np.mean(is_active)) if n_voxels else float("nan"),
            "matched_count": len(rec),
            "matched_fraction_of_active": (
                float(len(rec) / is_active.sum()) if is_active.sum() else float("nan")
            ),
            "matched_true_t1_median_ms": (
                float(np.median(matched_t1)) if matched_t1 else float("nan")
            ),
            "matched_true_t2_median_ms": (
                float(np.median(matched_t2)) if matched_t2 else float("nan")
            ),
            "matched_true_w_median": (
                float(np.median(matched_w)) if matched_w else float("nan")
            ),
            "active_pred_t1_median_ms": (
                float(np.median(pred_active[:, 0])) if len(pred_active) else float("nan")
            ),
            "active_pred_t2_median_ms": (
                float(np.median(pred_active[:, 1])) if len(pred_active) else float("nan")
            ),
            "active_pred_w_median": (
                float(np.median(pred_active[:, 2])) if len(pred_active) else float("nan")
            ),
            "active_rate_by_true_count": {
                str(k): float(np.mean(is_active[true_counts == k]))
                for k in sorted(set(true_counts.tolist()))
            },
        }
        per_query.append(row)

    return {
        "threshold": float(threshold),
        "n_voxels": int(n_voxels),
        "n_queries": int(n_queries),
        "queries": per_query,
    }, matched_records


def select_examples(query_outputs, trues, threshold: float):
    """Choose one deterministic success and failure per true count.

    A success has the right count and the smallest median matched log-parameter error. A failure
    prioritizes wrong counts and then the largest error. This avoids hand-picking flattering or
    dramatic cases for the thesis.
    """
    params = np.asarray(query_outputs["params"], dtype=np.float64)
    probs = np.asarray(query_outputs["exist_prob"], dtype=np.float64)
    rows = []
    for i, true in enumerate(trues):
        qidx = np.flatnonzero(probs[i] > threshold)
        pred = [tuple(params[i, q]) for q in qidx]
        pairs = _match(pred, true)
        errors = [
            abs(np.log(p[0] / t[0])) + abs(np.log(p[1] / t[1]))
            for p, t in pairs
        ]
        score = float(np.median(errors)) if errors else float("inf")
        rows.append({
            "index": i,
            "true_count": len(true),
            "pred_count": len(pred),
            "count_correct": len(pred) == len(true),
            "error_score": score,
            "queries": [int(q) for q in qidx],
            "predictions": [[float(v) for v in params[i, q]] for q in qidx],
            "truth": [[float(v) for v in t] for t in true],
        })

    chosen = {"successes": [], "failures": []}
    for count in sorted({r["true_count"] for r in rows}):
        group = [r for r in rows if r["true_count"] == count]
        correct = [r for r in group if r["count_correct"]]
        if correct:
            chosen["successes"].append(min(correct, key=lambda r: (r["error_score"], r["index"])))
        wrong = [r for r in group if not r["count_correct"]]
        pool = wrong if wrong else group
        chosen["failures"].append(max(
            pool,
            key=lambda r: (
                abs(r["pred_count"] - r["true_count"]),
                r["error_score"],
                -r["index"],
            ),
        ))
    return chosen


def _plot_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_training_curves(history, path) -> str:
    plt = _plot_setup()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epoch = np.asarray([h["epoch"] + 1 for h in history])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ax[0].plot(epoch, [h["train"]["loss"] for h in history], label="train", lw=2)
    if history and history[0].get("val"):
        ax[0].plot(epoch, [h["val"]["loss"] for h in history], label="validation", lw=2)
    ax[0].set(title="Total Hungarian loss", xlabel="epoch", ylabel="loss")
    ax[0].grid(alpha=.25); ax[0].legend()
    colors = {"t1": "#0072B2", "t2": "#D55E00", "wt": "#009E73", "ex": "#CC79A7"}
    for key, color in colors.items():
        ax[1].plot(epoch, [h["val"][key] for h in history], label=key, color=color, lw=2)
    ax[1].set(title="Validation loss components", xlabel="epoch", ylabel="weighted loss")
    ax[1].grid(alpha=.25); ax[1].legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_threshold_calibration(calibration, path) -> str:
    plt = _plot_setup()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    curve = calibration["curve"]
    x = [r["threshold"] for r in curve]
    selected = calibration["selected_threshold"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, [r["count_accuracy"] for r in curve], lw=2.4, label="exact count accuracy")
    ax.plot(x, [r["existence_f1"] for r in curve], lw=2, label="existence F1")
    ax.plot(x, [r["existence_precision"] for r in curve], lw=1.5, label="precision", alpha=.8)
    ax.plot(x, [r["existence_recall"] for r in curve], lw=1.5, label="recall", alpha=.8)
    ax.axvline(selected, color="black", ls="--", lw=1.5, label=f"selected = {selected:.2f}")
    ax.set(title="Validation-only existence-threshold selection",
           xlabel="existence threshold", ylabel="score", ylim=(0, 1.02))
    ax.grid(alpha=.25); ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_count_confusion(metrics, path) -> str:
    plt = _plot_setup()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    confusion = metrics["confusion"]
    rows = confusion["true_counts"]
    matrix = np.asarray([confusion["matrix"][str(k)] for k in rows], dtype=float)
    nonzero_cols = np.flatnonzero(matrix.sum(axis=0))
    last = int(nonzero_cols[-1]) if len(nonzero_cols) else 0
    matrix = matrix[:, :last + 1]
    denom = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(matrix, denom, out=np.zeros_like(matrix), where=denom != 0)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    image = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            if matrix[r, c]:
                color = "white" if norm[r, c] > .55 else "black"
                ax.text(c, r, f"{int(matrix[r,c])}\n{norm[r,c]:.1%}",
                        ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(len(rows)), rows)
    ax.set(xlabel="predicted compartment count", ylabel="true compartment count",
           title="Test count confusion (row-normalized)")
    fig.colorbar(image, ax=ax, label="fraction within true count")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_error_by_count(metrics, path) -> str:
    plt = _plot_setup()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = metrics["confusion"]["true_counts"]
    x = np.arange(len(counts))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    acc = [metrics[f"count_accuracy_n{k}"] for k in counts]
    ax[0].bar(x, acc, color="#0072B2")
    ax[0].set_xticks(x, counts)
    ax[0].set(title="Exact count accuracy", xlabel="true compartment count",
              ylabel="accuracy", ylim=(0, 1))
    width = .25
    for offset, key, label, color in [
        (-width, "t1_rel_median", "T1", "#0072B2"),
        (0, "t2_rel_median", "T2", "#D55E00"),
        (width, "w_mae", "weight", "#009E73"),
    ]:
        ax[1].bar(x + offset, [metrics[f"n{k}_{key}"] for k in counts],
                  width=width, label=label, color=color)
    ax[1].set_xticks(x, counts)
    ax[1].set(title="Matched median errors", xlabel="true compartment count",
              ylabel="relative error / absolute weight error")
    ax[1].legend()
    for a in ax:
        a.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_snr_robustness(ladder, path) -> str:
    plt = _plot_setup()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(ladder.values(), key=lambda x: x["snr"])
    snr = [r["snr"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].plot(snr, [r["count_accuracy"] for r in rows], "o-", lw=2, label="count accuracy")
    ax[0].plot(snr, [r["existence_f1"] for r in rows], "s-", lw=2, label="existence F1")
    ax[0].set(ylabel="score", ylim=(0, 1), title="Counting robustness")
    ax[0].legend()
    ax[1].plot(snr, [r["t1_rel_median"] for r in rows], "o-", lw=2, label="T1")
    ax[1].plot(snr, [r["t2_rel_median"] for r in rows], "s-", lw=2, label="T2")
    ax[1].plot(snr, [r["w_mae"] for r in rows], "^-", lw=2, label="weight")
    ax[1].set(ylabel="matched median error", title="Parameter robustness")
    ax[1].legend()
    for a in ax:
        a.axvspan(min(snr), 30, color="#D55E00", alpha=.09, label="below train range")
        a.set_xlabel("SNR")
        a.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_query_specialization(analysis, matched_records, path) -> str:
    plt = _plot_setup()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    queries = analysis["queries"]
    q = np.arange(len(queries))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    ax[0].bar(q - .18, [r["active_rate"] for r in queries], width=.36, label="active")
    ax[0].bar(q + .18, [
        r["matched_count"] / analysis["n_voxels"] for r in queries
    ], width=.36, label="matched")
    ax[0].set(title="Query use on test voxels", xlabel="query index", ylabel="fraction of voxels")
    ax[0].set_xticks(q); ax[0].legend(); ax[0].grid(axis="y", alpha=.25)
    if matched_records:
        # Deterministic thinning keeps a large test set readable without changing its support.
        rec = matched_records[::max(1, len(matched_records) // 6000)]
        for query in q:
            rq = [r for r in rec if r["query"] == query]
            if rq:
                ax[1].scatter([r["true_t1"] for r in rq], [r["true_t2"] for r in rq],
                              s=7, alpha=.35, label=f"Q{query}")
        ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set(title="Ground-truth compartments matched to each query",
              xlabel="true T1 (ms)", ylabel="true T2 (ms)")
    ax[1].grid(which="both", alpha=.2)
    if len(q) <= 12:
        ax[1].legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_examples(examples, path) -> str:
    plt = _plot_setup()
    from matplotlib.ticker import NullFormatter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    successes, failures = examples["successes"], examples["failures"]
    ncol = max(len(successes), len(failures), 1)
    fig, axes = plt.subplots(2, ncol, figsize=(4.2 * ncol, 8), squeeze=False)
    for row_idx, (label, records) in enumerate([("success", successes), ("failure", failures)]):
        for col in range(ncol):
            ax = axes[row_idx, col]
            if col >= len(records):
                ax.axis("off")
                continue
            record = records[col]
            truth = np.asarray(record["truth"])
            pred = np.asarray(record["predictions"])
            if len(truth):
                ax.scatter(truth[:, 0], truth[:, 1], s=400 * truth[:, 2] + 30,
                           marker="x", linewidths=2.2, color="black", label="truth")
                for j, point in enumerate(truth):
                    ax.annotate(f"T{j + 1} ({point[2]:.2f})", point[:2],
                                xytext=(5, 5), textcoords="offset points", fontsize=8)
            if len(pred):
                ax.scatter(pred[:, 0], pred[:, 1], s=400 * pred[:, 2] + 30,
                           facecolors="none", edgecolors="#D55E00", linewidths=2, label="prediction")
                for query, point in zip(record["queries"], pred):
                    ax.annotate(f"Q{query} ({point[2]:.2f})", point[:2],
                                xytext=(5, -12), textcoords="offset points",
                                color="#D55E00", fontsize=8)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            points = np.concatenate(
                [a for a in (truth, pred) if len(a)],
                axis=0,
            )
            # A nearly exact one-compartment success otherwise gets microscopic limits and
            # unreadable scientific-notation ticks. Give every panel a useful log-space frame.
            for axis, values in ((ax.set_xlim, points[:, 0]), (ax.set_ylim, points[:, 1])):
                lo, hi = float(values.min()) / 1.35, float(values.max()) * 1.35
                if hi / lo < 2:
                    center = float(np.sqrt(lo * hi))
                    lo, hi = center / np.sqrt(2), center * np.sqrt(2)
                axis(lo, hi)
            ax.set(title=f"{label}: voxel {record['index']} | "
                         f"{record['true_count']}→{record['pred_count']}",
                   xlabel="T1 (ms)", ylabel="T2 (ms)")
            ax.grid(which="both", alpha=.2)
            legend_loc = "upper left" if row_idx == 0 and col != 1 else "upper right"
            ax.legend(fontsize=8, loc=legend_loc)
    fig.suptitle("Deterministically selected test successes and failures", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def save_json(value, path) -> str:
    def json_safe(item):
        if isinstance(item, dict):
            return {str(k): json_safe(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [json_safe(v) for v in item]
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_safe(value), f, indent=2, allow_nan=False)
    return str(path)
