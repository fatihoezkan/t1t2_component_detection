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


def detr_predictions(model, ds, device, normalizer, exist_thresh=0.5, batch_size=512):
    """Run the model and keep only the queries that claim a compartment exists.

    The model always emits n_queries guesses; the existence score separates the real ones from
    the "nothing here" ones. Survivors get their T1/T2 converted back to ms (undoing the
    training normalization). Runs under no_grad in chunks — this is inference.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            X = ds.X[i:i + batch_size].to(device)
            out = model(X)
            out = out["pred"] if isinstance(out, dict) else out
            out = out.cpu().numpy()
            for row in out:                                    # (n_queries, 4)
                exist = 1.0 / (1.0 + np.exp(-row[:, 3]))
                keep = exist > exist_thresh
                t1 = normalizer.denormalize_t1(row[keep, 0])
                t2 = normalizer.denormalize_t2(row[keep, 1])
                w = row[keep, 2]
                preds.append([(float(a), float(b), float(c)) for a, b, c in zip(t1, t2, w)])
    return preds


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


def _median(a):
    return float(np.median(a)) if len(a) else float("nan")


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
        f"{prefix}t1_rel_median": _median(t1_rel),
        f"{prefix}t2_rel_median": _median(t2_rel),
        f"{prefix}t1_mae_ms": _median(t1_abs),
        f"{prefix}t2_mae_ms": _median(t2_abs),
        f"{prefix}w_mae": _median(w_abs),
        f"{prefix}t2_rel_median_noncsf": _median(t2_rel_noncsf),
        f"{prefix}t2_rel_median_csf": _median(t2_rel_csf),
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

    Per-n metrics are not optional decoration. The data is uniform over n_comp=1..4, and the
    smallest compartment of a 4-compartment voxel sits below the noise floor in most voxels at
    low SNR — so an aggregate averages an easy regime with a near-impossible one and describes
    neither. Any claim about quality has to be read per-n.
    """
    pc = np.array([len(p) for p in preds])
    tc = np.array([len(t) for t in trues])

    m = {
        "n_voxels": int(len(preds)),
        "count_accuracy": float((pc == tc).mean()) if len(pc) else float("nan"),
        "count_mae": float(np.abs(pc - tc).mean()) if len(pc) else float("nan"),
    }
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
        sub = _regression_block([preds[i] for i in idx], [trues[i] for i in idx], prefix=f"n{k}_")
        m |= {key: sub[key] for key in (f"n{k}_t1_rel_median", f"n{k}_t2_rel_median", f"n{k}_w_mae")}

    m["confusion"] = count_confusion(preds, trues, n_queries)
    m |= physics_violations(preds)
    return m


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
