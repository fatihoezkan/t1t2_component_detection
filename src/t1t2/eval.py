"""Turning raw predictions into numbers (and a picture) you can put in the thesis.

A trained model outputs (T1, T2, weight, existence) per query. That is not directly a
*result*. This module boils it down to the same questions a reviewer will ask:

  - Did it find the right *number* of compartments? (count accuracy)
  - Are the T1/T2 values close, in real milliseconds?
  - How does it do on CSF specifically — the pool we know is barely measurable with this
    protocol (TE_max ≈ 150 ms) — versus everything else?

To score any of that we first line predictions up with the truth (the same matching problem
as the loss, in a different setting). CSF gets its own line so its built-in unmeasurability
neither drags down nor flatters the headline numbers. Predictions are a list of
(T1_ms, T2_ms, weight) per voxel, and everything below scores them against the ground truth.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# T2 above this (ms) is treated as CSF: its T2 is essentially unconstrained here (longest TE is
# 150 ms, so exp(-TE/T2) barely moves for a 2000 ms pool), so we always break it out separately.
CSF_T2_MS = 1000.0


def detr_query_outputs(model, ds, device, normalizer, batch_size=512):
    """Return every query before thresholding, in physical units.

    Keeping the raw query table is important for two baseline jobs that cannot be done from the
    filtered compartment lists: selecting an existence threshold on validation data and asking
    whether individual queries specialize. The returned dictionary contains:

    ``params``
        ``(N, Q, 3)`` float array with T1 ms, T2 ms, and weight.
    ``exist_prob``
        ``(N, Q)`` float array with sigmoid-transformed existence probabilities.
    """
    model.eval()
    params, probs = [], []
    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            X = ds.X[i:i + batch_size].to(device)
            out = model(X)
            out = out["pred"] if isinstance(out, dict) else out
            row = out.detach().cpu().numpy()
            physical = np.empty(row[..., :3].shape, dtype=np.float64)
            physical[..., 0] = normalizer.denormalize_t1(row[..., 0])
            physical[..., 1] = normalizer.denormalize_t2(row[..., 1])
            physical[..., 2] = row[..., 2]
            params.append(physical)
            # Stable sigmoid for the moderate logits emitted here. Clipping also keeps unusual
            # checkpoints from overflowing exp during offline analysis.
            probs.append(1.0 / (1.0 + np.exp(-np.clip(row[..., 3], -80.0, 80.0))))
    n_queries = getattr(model, "n_queries", 0)
    return {
        "params": np.concatenate(params, axis=0)
        if params else np.empty((0, n_queries, 3), dtype=np.float64),
        "exist_prob": np.concatenate(probs, axis=0)
        if probs else np.empty((0, n_queries), dtype=np.float64),
    }


def predictions_from_query_outputs(query_outputs, exist_thresh=0.5):
    """Convert unfiltered query arrays into the list-of-compartments evaluation format."""
    params = np.asarray(query_outputs["params"])
    probs = np.asarray(query_outputs["exist_prob"])
    if params.shape[:2] != probs.shape:
        raise ValueError(
            f"query parameter/probability shapes disagree: {params.shape} vs {probs.shape}"
        )
    preds = []
    for row, score in zip(params, probs):
        keep = score > exist_thresh
        preds.append([tuple(float(v) for v in p) for p in row[keep]])
    return preds


def detr_predictions(model, ds, device, normalizer, exist_thresh=0.5, batch_size=512):
    """Run the model and keep only the queries that claim a compartment exists.

    The model always emits n_queries guesses; the existence score separates the real ones from
    the "nothing here" ones. Survivors get their T1/T2 converted back to ms (undoing the
    training normalization). Runs under no_grad in chunks — this is inference.
    """
    query_outputs = detr_query_outputs(model, ds, device, normalizer, batch_size=batch_size)
    return predictions_from_query_outputs(query_outputs, exist_thresh)


def true_compartments(ds):
    """Pull ground-truth compartments back out of the dataset in the same list format.

    Reads the raw (unnormalized) ms values stashed at load time, taking only the first n_comp of
    each voxel's slots — the rest are padding.
    """
    trues, nc = [], ds.n_comp.numpy()
    for i in range(len(ds)):
        k = int(nc[i])
        trues.append([(float(ds.raw_t1[i, j]), float(ds.raw_t2[i, j]), float(ds.raw_w[i, j]))
                      for j in range(k)])
    return trues


def _match(pred, true):
    """Pair predicted and true compartments before scoring. Matches on distance in
    (log T1, log T2) space, so a 50 ms miss on a short-T2 pool counts as heavily as a
    proportionally similar miss on a long one. Empty on either side -> nothing to pair."""
    if not pred or not true:
        return []
    P = np.array([[p[0], p[1]] for p in pred], np.float64)
    T = np.array([[t[0], t[1]] for t in true], np.float64)
    cost = ((np.log(P[:, None, 0]) - np.log(T[None, :, 0])) ** 2
            + (np.log(P[:, None, 1]) - np.log(T[None, :, 1])) ** 2)
    r, c = linear_sum_assignment(cost)
    return [(pred[i], true[j]) for i, j in zip(r, c)]


def _parameter_match_indices(pred, true):
    """Hungarian assignment using all three primary outputs: T1, T2, and weight."""
    if not pred or not true:
        return np.array([], dtype=int), np.array([], dtype=int)
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    cost = (
        (np.log(p[:, None, 0]) - np.log(t[None, :, 0])) ** 2
        + (np.log(p[:, None, 1]) - np.log(t[None, :, 1])) ** 2
        + (p[:, None, 2] - t[None, :, 2]) ** 2
    )
    return linear_sum_assignment(cost)


def parameter_recovery_analysis(preds, trues, weight_bins=None):
    """Parameter-first set recovery, including errors versus true signal fraction.

    Count accuracy is intentionally not the headline here. The core outputs are:

    - signal-fraction-weighted T1/T2 relative error on matched compartments;
    - recovered true signal fraction (so missing weak/strong pools remain visible);
    - set-level L1 weight error, including unmatched true and predicted weights;
    - a bounded parameter-set error used only for validation threshold selection.

    The bounded score assigns a 100% T1/T2 error to a missed true compartment, clips a matched
    relative error at 100%, includes set-level weight L1 clipped at one, and averages those three
    terms. Its complement is reported as ``parameter_set_accuracy``. The separate unbounded
    physical errors remain the quantities to interpret scientifically.
    """
    if len(preds) != len(trues):
        raise ValueError("prediction and truth lengths must agree")
    if weight_bins is None:
        weight_bins = (0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.000001)
    weight_bins = np.asarray(weight_bins, dtype=np.float64)
    if np.any(np.diff(weight_bins) <= 0):
        raise ValueError("weight bins must be strictly increasing")

    records, extras = [], []
    total_true_weight = recovered_true_weight = 0.0
    matched_weighted_t1 = matched_weighted_t2 = matched_weight_mass = 0.0
    matched_weight_abs = []
    per_voxel_set_error, per_voxel_weight_l1, per_voxel_extra_weight = [], [], []

    for voxel, (pred, true) in enumerate(zip(preds, trues)):
        p = np.asarray(pred, dtype=np.float64).reshape(-1, 3)
        t = np.asarray(true, dtype=np.float64).reshape(-1, 3)
        rows, cols = _parameter_match_indices(pred, true)
        pred_to_true = {int(r): int(c) for r, c in zip(rows, cols)}
        true_to_pred = {int(c): int(r) for r, c in zip(rows, cols)}

        t1_penalty = t2_penalty = 0.0
        weight_l1 = 0.0
        for j, target in enumerate(t):
            true_weight = float(target[2])
            total_true_weight += true_weight
            if j in true_to_pred:
                i = true_to_pred[j]
                estimate = p[i]
                t1_rel = float(abs(estimate[0] - target[0]) / target[0])
                t2_rel = float(abs(estimate[1] - target[1]) / target[1])
                w_abs = float(abs(estimate[2] - true_weight))
                recovered_true_weight += true_weight
                matched_weighted_t1 += true_weight * t1_rel
                matched_weighted_t2 += true_weight * t2_rel
                matched_weight_mass += true_weight
                matched_weight_abs.append(w_abs)
                t1_penalty += true_weight * min(t1_rel, 1.0)
                t2_penalty += true_weight * min(t2_rel, 1.0)
                weight_l1 += w_abs
                records.append({
                    "voxel": voxel,
                    "matched": True,
                    "true_t1": float(target[0]),
                    "true_t2": float(target[1]),
                    "true_weight": true_weight,
                    "pred_t1": float(estimate[0]),
                    "pred_t2": float(estimate[1]),
                    "pred_weight": float(estimate[2]),
                    "t1_rel_error": t1_rel,
                    "t2_rel_error": t2_rel,
                    "weight_abs_error": w_abs,
                })
            else:
                # Missing a compartment is a full error in the bounded validation score.
                t1_penalty += true_weight
                t2_penalty += true_weight
                weight_l1 += true_weight
                records.append({
                    "voxel": voxel,
                    "matched": False,
                    "true_t1": float(target[0]),
                    "true_t2": float(target[1]),
                    "true_weight": true_weight,
                    "pred_t1": None,
                    "pred_t2": None,
                    "pred_weight": None,
                    "t1_rel_error": None,
                    "t2_rel_error": None,
                    "weight_abs_error": None,
                })

        extra_weight = 0.0
        for i, estimate in enumerate(p):
            if i not in pred_to_true:
                extra_weight += float(estimate[2])
                weight_l1 += float(estimate[2])
                extras.append({
                    "voxel": voxel,
                    "pred_t1": float(estimate[0]),
                    "pred_t2": float(estimate[1]),
                    "pred_weight": float(estimate[2]),
                })

        true_mass = float(t[:, 2].sum()) if len(t) else 1.0
        t1_term = t1_penalty / max(true_mass, 1e-12)
        t2_term = t2_penalty / max(true_mass, 1e-12)
        set_error = (t1_term + t2_term + min(weight_l1, 1.0)) / 3.0
        per_voxel_set_error.append(set_error)
        per_voxel_weight_l1.append(weight_l1)
        per_voxel_extra_weight.append(extra_weight)

    matched = [r for r in records if r["matched"]]
    summary = {
        "n_voxels": len(preds),
        "n_true_compartments": len(records),
        "n_matched_compartments": len(matched),
        "n_extra_predictions": len(extras),
        "matched_true_compartment_rate": (
            float(len(matched) / len(records)) if records else float("nan")
        ),
        "recovered_signal_fraction": (
            float(recovered_true_weight / total_true_weight)
            if total_true_weight else float("nan")
        ),
        "t1_fraction_weighted_relative_error_matched": (
            float(matched_weighted_t1 / matched_weight_mass)
            if matched_weight_mass else float("nan")
        ),
        "t2_fraction_weighted_relative_error_matched": (
            float(matched_weighted_t2 / matched_weight_mass)
            if matched_weight_mass else float("nan")
        ),
        "weight_absolute_error_matched_mean": _mean(matched_weight_abs),
        "weight_set_l1_error_mean": _mean(per_voxel_weight_l1),
        "extra_predicted_weight_per_voxel_mean": _mean(per_voxel_extra_weight),
        "parameter_set_error": _mean(per_voxel_set_error),
        "parameter_set_accuracy": (
            float(1.0 - np.mean(per_voxel_set_error))
            if per_voxel_set_error else float("nan")
        ),
    }

    bins = []
    for i, (lo, hi) in enumerate(zip(weight_bins[:-1], weight_bins[1:])):
        bucket = [
            r for r in records
            if r["true_weight"] >= lo
            and (r["true_weight"] < hi or (i == len(weight_bins) - 2 and r["true_weight"] <= hi))
        ]
        bucket_matched = [r for r in bucket if r["matched"]]
        bucket_mass = sum(r["true_weight"] for r in bucket)
        matched_mass = sum(r["true_weight"] for r in bucket_matched)
        bins.append({
            "weight_min": float(lo),
            "weight_max": float(min(hi, 1.0)),
            "label": f"{lo:.2f}–{min(hi, 1.0):.2f}",
            "n_true": len(bucket),
            "n_matched": len(bucket_matched),
            "match_rate": (
                float(len(bucket_matched) / len(bucket)) if bucket else float("nan")
            ),
            "recovered_signal_fraction_within_bin": (
                float(matched_mass / bucket_mass) if bucket_mass else float("nan")
            ),
            "t1_relative_error_median": _median(
                [r["t1_rel_error"] for r in bucket_matched]
            ),
            "t2_relative_error_median": _median(
                [r["t2_rel_error"] for r in bucket_matched]
            ),
            "weight_absolute_error_median": _median(
                [r["weight_abs_error"] for r in bucket_matched]
            ),
        })
    return {"summary": summary, "bins": bins, "records": records, "extras": extras}


def _median(a):
    return float(np.median(a)) if len(a) else float("nan")


def _mean(a):
    return float(np.mean(a)) if len(a) else float("nan")


def _regression_block(preds, trues, prefix=""):
    """Matched-pair errors over whatever subset of voxels it is handed."""
    t1_rel, t2_rel, t1_abs, t2_abs, w_abs = [], [], [], [], []
    t2_rel_csf, t2_rel_noncsf = [], []
    for pred, true in zip(preds, trues):
        for p, t in _match(pred, true):
            t1_abs.append(abs(p[0] - t[0]))
            t2_abs.append(abs(p[1] - t[1]))
            t1_rel.append(abs(p[0] - t[0]) / t[0])
            rel2 = abs(p[1] - t[1]) / t[1]
            t2_rel.append(rel2)
            w_abs.append(abs(p[2] - t[2]))
            (t2_rel_csf if t[1] > CSF_T2_MS else t2_rel_noncsf).append(rel2)
    return {
        f"{prefix}n_matched": int(len(t1_rel)),
        # Backwards-compatible headline keys. They were historically documented as medians and
        # are retained so existing notebooks keep reading old and new runs consistently.
        f"{prefix}t1_rel_median": _median(t1_rel),
        f"{prefix}t2_rel_median": _median(t2_rel),
        f"{prefix}t1_mae_ms": _median(t1_abs),
        f"{prefix}t2_mae_ms": _median(t2_abs),
        f"{prefix}w_mae": _median(w_abs),
        # Explicit names remove the old ambiguity around "MAE": the completion report can now
        # provide both the robust median and the conventional mean absolute error.
        f"{prefix}t1_abs_median_ms": _median(t1_abs),
        f"{prefix}t2_abs_median_ms": _median(t2_abs),
        f"{prefix}w_abs_median": _median(w_abs),
        f"{prefix}t1_abs_mean_ms": _mean(t1_abs),
        f"{prefix}t2_abs_mean_ms": _mean(t2_abs),
        f"{prefix}w_abs_mean": _mean(w_abs),
        f"{prefix}t1_rel_mean": _mean(t1_rel),
        f"{prefix}t2_rel_mean": _mean(t2_rel),
        f"{prefix}t2_rel_median_noncsf": _median(t2_rel_noncsf),
        f"{prefix}t2_rel_median_csf": _median(t2_rel_csf),
    }


def count_detection_metrics(preds, trues):
    """Count-based existence precision/recall over a set-prediction dataset.

    After optimal assignment, each voxel contributes ``min(predicted, true)`` detected
    compartments; surplus predictions are false positives and missing predictions are false
    negatives. This intentionally evaluates *existence/counting*. Parameter quality is scored
    separately on the matched pairs.
    """
    pc = np.asarray([len(p) for p in preds], dtype=int)
    tc = np.asarray([len(t) for t in trues], dtype=int)
    tp = int(np.minimum(pc, tc).sum())
    fp = int(np.maximum(pc - tc, 0).sum())
    fn = int(np.maximum(tc - pc, 0).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall and np.isfinite(precision + recall) else float("nan"))
    n = len(pc)
    return {
        "existence_tp": tp,
        "existence_fp": fp,
        "existence_fn": fn,
        "existence_precision": float(precision),
        "existence_recall": float(recall),
        "existence_f1": float(f1),
        "false_positive_compartments_per_voxel": float(fp / n) if n else float("nan"),
        "missed_compartments_per_voxel": float(fn / n) if n else float("nan"),
    }


def count_confusion(preds, trues, n_queries=10):
    """True count (rows) x predicted count (columns) — how it misses, not just how often.

    Predicted counts run 0..n_queries because the model can keep any number of its queries, so
    the table is deliberately not square. Under- and over-counting are different failures with
    different causes, and a single accuracy number hides which one is happening.
    """
    tc = np.array([len(t) for t in trues], dtype=int)
    pc = np.array([len(p) for p in preds], dtype=int)
    rows = sorted(set(tc.tolist()))
    mat = {}
    for r in rows:
        counts = np.bincount(pc[tc == r], minlength=n_queries + 1)[: n_queries + 1]
        mat[str(r)] = [int(c) for c in counts]
    return {"true_counts": rows, "predicted_range": [0, n_queries], "matrix": mat}


def physics_violations(preds):
    """Do the predictions obey the physics the data was built from?

    The heads are independent sigmoids, so nothing stops the model emitting T2 >= T1 or weights
    that miss 1. Neither is post-processed away, so measuring them says whether the model has
    actually learned the structure of the problem or is just fitting numbers. This is the
    concrete form of "outputs make physical sense".
    """
    viol, wsum = [], []
    for pred in preds:
        if not pred:
            continue
        viol.extend(1.0 if p[1] >= p[0] else 0.0 for p in pred)
        wsum.append(abs(sum(p[2] for p in pred) - 1.0))
    return {
        "t2_ge_t1_rate": float(np.mean(viol)) if viol else float("nan"),
        "weight_sum_dev_median": _median(wsum),
        "weight_sum_dev_mean": float(np.mean(wsum)) if wsum else float("nan"),
    }


def compute_metrics(preds, trues, n_queries=10):
    """Score a whole split: counting, per-parameter error, per-n, physics, CSF breakout.

    Count accuracy is judged per voxel on the *number* of compartments — no matching needed,
    just len(pred) vs len(true). Regression errors are gathered over matched pairs only, each T2
    error filed into the CSF or non-CSF bucket by the true T2. Medians for the relative errors,
    because a few badly-matched outliers would wreck a mean and misrepresent typical behaviour.

    Per-n metrics are not optional decoration. The per-count files deliberately balance easy and
    hard regimes, so an aggregate can describe neither of them well. Any claim about quality has
    to be read per-n; the separately generated n=4 family is an even harder stress test.
    """
    pc = np.array([len(p) for p in preds])
    tc = np.array([len(t) for t in trues])

    m = {
        "n_voxels": int(len(preds)),
        "count_accuracy": float((pc == tc).mean()) if len(pc) else float("nan"),
        "count_mae": float(np.abs(pc - tc).mean()) if len(pc) else float("nan"),
    }
    m |= count_detection_metrics(preds, trues)
    m |= _regression_block(preds, trues)

    # Restricted to voxels whose count is right. An undercounting model only ever reports the
    # compartments it did find, which flatters its matched errors; this brackets that bias from
    # the other side. It is conditioned on success, so it is NOT a random-voxel estimate.
    ok = [i for i in range(len(preds)) if len(preds[i]) == len(trues[i])]
    m |= _regression_block([preds[i] for i in ok], [trues[i] for i in ok], prefix="cc_")
    m["cc_n_voxels"] = len(ok)

    for k in sorted(set(tc.tolist())):
        idx = np.flatnonzero(tc == k)
        m[f"count_accuracy_n{k}"] = float((pc[idx] == k).mean())
        m[f"count_mae_n{k}"] = float(np.abs(pc[idx] - k).mean())
        m[f"n_voxels_n{k}"] = int(len(idx))
        m |= {
            f"existence_precision_n{k}": count_detection_metrics(
                [preds[i] for i in idx], [trues[i] for i in idx]
            )["existence_precision"],
            f"existence_recall_n{k}": count_detection_metrics(
                [preds[i] for i in idx], [trues[i] for i in idx]
            )["existence_recall"],
        }
        sub = _regression_block([preds[i] for i in idx], [trues[i] for i in idx], prefix=f"n{k}_")
        keep = (
            f"n{k}_t1_rel_median", f"n{k}_t2_rel_median", f"n{k}_w_mae",
            f"n{k}_t1_abs_mean_ms", f"n{k}_t2_abs_mean_ms", f"n{k}_w_abs_mean",
        )
        m |= {key: sub[key] for key in keep}

    m["confusion"] = count_confusion(preds, trues, n_queries)
    m |= physics_violations(preds)
    recovery = parameter_recovery_analysis(preds, trues)
    m["parameter_recovery"] = {
        "summary": recovery["summary"],
        "bins": recovery["bins"],
    }
    return m


def calibrate_existence_threshold(query_outputs, trues, thresholds=None,
                                  objective="count_accuracy"):
    """Select the existence threshold using validation data only.

    ``count_accuracy`` exactly preserves the finalized baseline rule. ``parameter_set_error``
    prioritizes closeness of T1/T2/weight and penalizes missed/extra compartments through the
    bounded parameter-set score. The test split never participates in either rule.
    """
    if objective not in {"count_accuracy", "parameter_set_error"}:
        raise ValueError(
            "threshold objective must be count_accuracy|parameter_set_error; "
            f"got {objective!r}"
        )
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    curve = []
    for threshold in thresholds:
        threshold = float(threshold)
        preds = predictions_from_query_outputs(query_outputs, threshold)
        pc = np.asarray([len(p) for p in preds], dtype=int)
        tc = np.asarray([len(t) for t in trues], dtype=int)
        det = count_detection_metrics(preds, trues)
        recovery = parameter_recovery_analysis(preds, trues)["summary"]
        curve.append({
            "threshold": threshold,
            "count_accuracy": float(np.mean(pc == tc)) if len(pc) else float("nan"),
            "count_mae": float(np.mean(np.abs(pc - tc))) if len(pc) else float("nan"),
            "existence_precision": det["existence_precision"],
            "existence_recall": det["existence_recall"],
            "existence_f1": det["existence_f1"],
            "parameter_set_error": recovery["parameter_set_error"],
            "parameter_set_accuracy": recovery["parameter_set_accuracy"],
            "recovered_signal_fraction": recovery["recovered_signal_fraction"],
            "t1_fraction_weighted_relative_error_matched":
                recovery["t1_fraction_weighted_relative_error_matched"],
            "t2_fraction_weighted_relative_error_matched":
                recovery["t2_fraction_weighted_relative_error_matched"],
            "weight_set_l1_error_mean": recovery["weight_set_l1_error_mean"],
        })
    if not curve:
        raise ValueError("threshold grid is empty")
    if objective == "count_accuracy":
        key = lambda x: (
            -x["count_accuracy"],
            x["count_mae"],
            -x["existence_f1"],
            abs(x["threshold"] - 0.5),
            x["threshold"],
        )
    else:
        key = lambda x: (
            x["parameter_set_error"],
            -x["recovered_signal_fraction"],
            x["weight_set_l1_error_mean"],
            abs(x["threshold"] - 0.5),
            x["threshold"],
        )
    selected = min(curve, key=key)
    return {
        "objective": objective,
        "selected_threshold": selected["threshold"],
        "selected": selected,
        "curve": curve,
    }


def scatter_figure(preds, trues, path, title=""):
    """The predicted-vs-true scatter — the one plot that tells you at a glance if it works.

    Two panels (T1 and T2), each point a matched pair, log axes, with the y=x line: points
    hugging the diagonal mean good recovery, systematic offsets mean bias. matplotlib is
    imported lazily on the Agg backend so this runs on a headless cluster node.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = [(p[0], t[0], p[1], t[1]) for pred, true in zip(preds, trues) for p, t in _match(pred, true)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
    if pairs:
        a = np.array(pairs)
        for k, (ip, it, name) in enumerate([(0, 1, "T1"), (2, 3, "T2")]):
            true_v, pred_v = a[:, it], a[:, ip]
            # Shared, equal, square axis limits derived from the data (both true and pred), so
            # the y=x line is a true 45° corner-to-corner diagonal instead of a skewed segment.
            both = np.concatenate([true_v, pred_v])
            both = both[both > 0]
            lo, hi = float(both.min()) / 1.3, float(both.max()) * 1.3
            ax[k].plot([lo, hi], [lo, hi], "k--", lw=1, zorder=1, label="y = x")
            ax[k].scatter(true_v, pred_v, s=6, alpha=0.35, zorder=2)
            ax[k].set_xscale("log"); ax[k].set_yscale("log")
            ax[k].set_xlim(lo, hi); ax[k].set_ylim(lo, hi)
            ax[k].set_aspect("equal", "box")                    # square panel -> honest 45° diagonal
            ax[k].grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
            ax[k].set_xlabel(f"true {name} (ms)"); ax[k].set_ylabel(f"pred {name} (ms)")
            ax[k].set_title(name); ax[k].legend(loc="upper left", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def parameter_scatter_figure(analysis, path, title=""):
    """Predicted-versus-true T1, T2, and weight using the three-parameter assignment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = [r for r in analysis["records"] if r["matched"]]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    specs = [
        ("true_t1", "pred_t1", "T1 (ms)", True),
        ("true_t2", "pred_t2", "T2 (ms)", True),
        ("true_weight", "pred_weight", "signal fraction", False),
    ]
    for ax, (true_key, pred_key, label, use_log) in zip(axes, specs):
        if records:
            true_v = np.asarray([r[true_key] for r in records])
            pred_v = np.asarray([r[pred_key] for r in records])
            lo = float(min(true_v.min(), pred_v.min()))
            hi = float(max(true_v.max(), pred_v.max()))
            if use_log:
                lo, hi = lo / 1.2, hi * 1.2
                ax.set_xscale("log"); ax.set_yscale("log")
            else:
                lo, hi = min(0.0, lo), max(1.0, hi)
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
            ax.scatter(true_v, pred_v, s=5, alpha=.25)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set(xlabel=f"true {label}", ylabel=f"predicted {label}", title=label)
        ax.grid(which="both", alpha=.2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def error_vs_signal_fraction_figure(analysis, path, title=""):
    """T1/T2 relative error against the true compartment's signal fraction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = [r for r in analysis["records"] if r["matched"]]
    # Deterministic thinning keeps large test sets readable without changing their support.
    records = records[::max(1, len(records) // 12000)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    colors = ("#0072B2", "#D55E00")
    for ax, key, label, color in zip(
        axes,
        ("t1_rel_error", "t2_rel_error"),
        ("T1", "T2"),
        colors,
    ):
        if records:
            x = np.asarray([r["true_weight"] for r in records])
            y = np.maximum(np.asarray([r[key] for r in records]), 1e-4)
            ax.scatter(x, y, s=5, alpha=.18, color=color, rasterized=True)
        bins = analysis["bins"]
        centers = [(b["weight_min"] + b["weight_max"]) / 2 for b in bins]
        medians = [
            b[f"{label.lower()}_relative_error_median"] for b in bins
        ]
        ax.plot(centers, medians, "o-", color="black", lw=2, label="bin median")
        ax.set_yscale("log")
        ax.set(
            xlabel="true signal fraction",
            ylabel="absolute relative error",
            title=f"{label} error vs signal fraction",
            xlim=(0.045, 1.01),
        )
        ax.grid(which="both", alpha=.2)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def recovery_vs_signal_fraction_figure(analysis, path, title=""):
    """Detection coverage and weight error stratified by true signal fraction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = analysis["bins"]
    labels = [b["label"] for b in bins]
    x = np.arange(len(bins))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(x, [b["match_rate"] for b in bins], "o-", lw=2, label="compartment match rate")
    axes[0].plot(
        x,
        [b["recovered_signal_fraction_within_bin"] for b in bins],
        "s-",
        lw=2,
        label="recovered signal-fraction mass",
    )
    axes[0].set(ylabel="recovery", ylim=(0, 1.03), title="Recovery by true signal fraction")
    axes[0].legend()
    axes[1].bar(
        x,
        [b["weight_absolute_error_median"] for b in bins],
        color="#009E73",
        alpha=.85,
    )
    axes[1].set(ylabel="median absolute weight error", title="Weight error by true signal fraction")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.set_xlabel("true signal-fraction bin")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def threshold_calibration_figure(calibration, path):
    """Show both the legacy count objective and parameter-first threshold objective."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = calibration["curve"]
    x = [r["threshold"] for r in curve]
    selected = calibration["selected_threshold"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(x, [r["parameter_set_accuracy"] for r in curve],
            lw=2.4, label="parameter-set accuracy")
    ax.plot(x, [r["recovered_signal_fraction"] for r in curve],
            lw=2, label="recovered signal fraction")
    ax.plot(x, [r["count_accuracy"] for r in curve],
            lw=1.7, label="exact count accuracy", alpha=.8)
    ax.axvline(selected, color="black", ls="--", lw=1.5,
               label=f"selected = {selected:.2f}")
    ax.set(
        xlabel="existence threshold",
        ylabel="score",
        ylim=(0, 1.02),
        title=f"Validation-only threshold selection: {calibration['objective']}",
    )
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def evaluate_detr(model, ds, device, normalizer, results_dir, exist_thresh=0.5, tag="detr",
                  n_queries=10):
    """End-to-end for the model: predict, score, drop metrics + figure into results_dir.

    exist_thresh stays at its default. Tuning it on the split being reported would be scoring
    the model on data it was tuned against; if the count looks miscalibrated, sweep it on
    validation and freeze the choice before test is opened.
    """
    preds = detr_predictions(model, ds, device, normalizer, exist_thresh)
    trues = true_compartments(ds)
    metrics = compute_metrics(preds, trues, n_queries=n_queries)
    metrics["exist_thresh"] = exist_thresh
    _save(metrics, preds, trues, results_dir, tag)
    analysis = parameter_recovery_analysis(preds, trues)
    rd = Path(results_dir)
    compact = {"summary": analysis["summary"], "bins": analysis["bins"]}
    with open(rd / f"parameter_recovery_{tag}.json", "w") as f:
        json.dump(compact, f, indent=2)
    parameter_scatter_figure(
        analysis,
        rd / "figures" / f"parameter_scatter_{tag}.png",
        title=f"{tag}: T1/T2/weight recovery",
    )
    error_vs_signal_fraction_figure(
        analysis,
        rd / "figures" / f"error_vs_signal_fraction_{tag}.png",
        title=f"{tag}: relaxation-time error versus compartment strength",
    )
    recovery_vs_signal_fraction_figure(
        analysis,
        rd / "figures" / f"recovery_vs_signal_fraction_{tag}.png",
        title=f"{tag}: weak versus dominant compartments",
    )
    return metrics


def evaluate_snr_ladder(model, paths, cfg, device, normalizer, results_dir, train_snr_min=None,
                        exist_thresh=0.5, n_queries=10, limit=None):
    """Score each fixed-SNR rung separately: performance as a function of noise.

    The rungs are generated as a paired set — same voxels, same standardized noise, only the
    amplitude differs — so differences across rungs are the SNR effect and not sampling noise.

    `paths` maps a label to the parquet path(s) for that rung. Rungs below the training SNR
    range are flagged `extrapolation`: they are a robustness probe, and folding them into
    in-distribution numbers would misreport both.
    """
    from .data import VoxelDataset

    out = {}
    for label, path in paths.items():
        ds = VoxelDataset(path, cfg, normalizer, limit=limit)
        preds = detr_predictions(model, ds, device, normalizer, exist_thresh)
        m = compute_metrics(preds, true_compartments(ds), n_queries=n_queries)
        snr = _snr_of(label)
        m["snr"] = snr
        m["extrapolation"] = bool(train_snr_min is not None and snr is not None and snr < train_snr_min)
        out[label] = m

    rd = Path(results_dir)
    rd.mkdir(parents=True, exist_ok=True)
    with open(rd / "metrics_snr_ladder.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def _snr_of(label):
    """Pull the SNR out of a rung label like 'test_snr20'."""
    m = re.search(r"snr(\d+)", str(label))
    return float(m.group(1)) if m else None


def _save(metrics, preds, trues, results_dir, tag):
    """Write metrics_<tag>.json and the scatter into results_dir."""
    rd = Path(results_dir)
    (rd / "figures").mkdir(parents=True, exist_ok=True)
    with open(rd / f"metrics_{tag}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    scatter_figure(preds, trues, rd / "figures" / f"scatter_{tag}.png", title=tag)
