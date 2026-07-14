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


def compute_metrics(preds, trues):
    """Score a whole split: counting, per-parameter error, and the CSF breakout.

    Count accuracy is judged per voxel on the *number* of compartments — no matching needed,
    just len(pred) vs len(true). Regression errors are gathered over matched pairs only, each T2
    error filed into the CSF or non-CSF bucket by the true T2. Medians for the relative errors,
    because a few badly-matched outliers would wreck a mean and misrepresent typical behaviour.
    """
    pc = np.array([len(p) for p in preds])
    tc = np.array([len(t) for t in trues])
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
        "n_voxels": int(len(preds)),
        "n_matched": int(len(t1_rel)),
        "count_accuracy": float((pc == tc).mean()) if len(pc) else float("nan"),
        "count_mae": float(np.abs(pc - tc).mean()) if len(pc) else float("nan"),
        "t1_rel_median": _median(t1_rel),
        "t2_rel_median": _median(t2_rel),
        "t1_mae_ms": _median(t1_abs),
        "t2_mae_ms": _median(t2_abs),
        "w_mae": _median(w_abs),
        "t2_rel_median_noncsf": _median(t2_rel_noncsf),
        "t2_rel_median_csf": _median(t2_rel_csf),
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


def evaluate_detr(model, ds, device, normalizer, results_dir, exist_thresh=0.5, tag="detr"):
    """End-to-end for the model: predict, score, drop metrics + figure into results_dir."""
    preds = detr_predictions(model, ds, device, normalizer, exist_thresh)
    trues = true_compartments(ds)
    metrics = compute_metrics(preds, trues)
    _save(metrics, preds, trues, results_dir, tag)
    return metrics


def _save(metrics, preds, trues, results_dir, tag):
    """Write metrics_<tag>.json and the scatter into results_dir."""
    rd = Path(results_dir)
    (rd / "figures").mkdir(parents=True, exist_ok=True)
    with open(rd / f"metrics_{tag}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    scatter_figure(preds, trues, rd / "figures" / f"scatter_{tag}.png", title=tag)
