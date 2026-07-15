"""The training loop, plus everything needed to stop and pick up again later.

Nothing exotic here — the interesting design lives elsewhere (the loss, the normalization).
What this file cares about is being *unattended-friendly*: it runs on whatever hardware it
finds, writes progress to disk every few epochs, and if the job dies it resumes from the last
checkpoint instead of starting over. That matters because the real training happens on a
remote cluster and you don't want a dropped connection to cost you hours.

Everything is driven by an ExperimentConfig, so a run is fully described by its YAML and the
artifacts it leaves in results/<name>/.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig, load_config
from .data import TargetNormalizer, make_dataloader
from .device import device_info, get_device
from .loss import HungarianLoss
from .model import build_model

# The five numbers the loss returns, in order — kept as a constant so training and eval can't
# drift on what "position 2" means.
_LOSS_KEYS = ("loss", "t1", "t2", "wt", "ex")


def _total_limit(data_cfg) -> int | None:
    """Turn the per-path train cap into the total the loader expects.

    The loader splits a total evenly across the paths, so with the per-n files a cap of "k per
    path" is k * n_paths. Expressed per-path in the config because that is what keeps the arms
    of the data-scaling ablation balanced across compartment counts.
    """
    per = getattr(data_cfg, "train_limit_per_path", None)
    if per is None:
        return None
    paths = data_cfg.train_path
    n = 1 if isinstance(paths, str) else len(paths)
    return per * n


def _fingerprint(cfg: ExperimentConfig) -> dict:
    """What a resume must agree on: the data and the optimization it is continuing."""
    return {"data": asdict(cfg.data), "model": asdict(cfg.model),
            "loss": asdict(cfg.loss), "train": asdict(cfg.train)}


def _check_resume_compatible(cfg: ExperimentConfig, results_dir: Path, resume: bool, log) -> None:
    """Refuse to resume into a results directory that belonged to a different run.

    Checkpoints are keyed only by directory, so pointing a changed config at an existing one
    would load those weights and carry on — producing a model that is half one experiment and
    half another, with a config.yaml claiming it was all the second. Fail instead.
    """
    prev_path = results_dir / "config.yaml"
    if not (resume and prev_path.exists() and (results_dir / "checkpoints" / "last.pt").exists()):
        return

    prev = load_config(prev_path)
    old, new = _fingerprint(prev), _fingerprint(cfg)
    changed = {k: (old[k], new[k]) for k in new if old[k] != new[k]}
    if changed:
        detail = ", ".join(sorted(changed))
        raise ValueError(
            f"{results_dir} holds a checkpoint from a different config (differs in: {detail}). "
            "Resuming would blend two experiments. Use a new results dir / cfg.name, or pass "
            "resume=False to start over."
        )
    log(f"[{cfg.name}] resume fingerprint matches the existing run")


def set_seed(seed: int) -> None:
    """Pin every RNG we touch so a run is reproducible. Not bulletproof across GPUs, but enough
    to make two local runs match and keep debugging sane."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_epoch(model, loader, crit, device, opt=None, aux_weight=1.0) -> dict:
    """One pass over a loader — training if given an optimizer, evaluating if not.

    Folding both modes into one function keeps train and validation honestly identical except
    for the backward pass. `opt is None` is the switch. aux_weight ramps the per-decoder-layer
    auxiliary losses in — later layers count fully, earlier ones are dialled down (the usual
    DETR deep-supervision recipe).
    """
    train = opt is not None
    model.train() if train else model.eval()
    agg = {k: [] for k in _LOSS_KEYS}
    with torch.enable_grad() if train else torch.no_grad():
        for X, y, nc in loader:
            X, y, nc = X.to(device), y.to(device), nc.to(device)
            out = model(X)
            if isinstance(out, dict):                          # aux_loss on
                loss, l1, l2, lw, le = crit(out["pred"], y, nc)
                for i, aux in enumerate(out["aux"]):
                    al, *_ = crit(aux, y, nc)
                    loss = loss + al * min((i + 1) * aux_weight, 1.0)
            else:
                loss, l1, l2, lw, le = crit(out, y, nc)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            for k, v in zip(_LOSS_KEYS, (loss, l1, l2, lw, le)):
                agg[k].append(float(v.item()))
    return {k: float(np.mean(v)) for k, v in agg.items()}


def train(cfg: ExperimentConfig, results_dir=None, max_epochs=None, resume=True, limit=None, log=print):
    """Train from a config and leave a fully resumable trail behind.

    max_epochs overrides the config epoch count (handy for smoke runs); limit caps how many
    voxels get loaded (ditto). `log` is injectable so tests can silence it. Returns the loss
    history, where the artifacts landed, and the trained model (so the caller can hand it
    straight to evaluation without reloading a checkpoint).
    """
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    log(f"[{cfg.name}] device={device} | {device_info()}")

    results_dir = Path(results_dir) if results_dir else Path("results") / cfg.name
    ckpt_dir = results_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Compare against the previous run's config *before* overwriting it: resuming into a results
    # directory that belonged to a different config or dataset would silently blend two runs.
    _check_resume_compatible(cfg, results_dir, resume, log)
    cfg.save(results_dir / "config.yaml")                      # snapshot the exact config next to outputs

    # One normalizer shared by train and val so both splits are transformed identically.
    normalizer = TargetNormalizer.from_config(cfg.data)
    # train_limit_per_path is the data-scaling knob and applies to train only; `limit` is the
    # smoke-run knob and caps everything. Validation must stay identical across arms or their
    # best_val numbers are not comparable.
    train_loader, _ = make_dataloader(
        cfg.data.train_path, cfg.data, cfg.train.batch_size, True,
        normalizer, cfg.train.num_workers,
        limit=limit if limit is not None else _total_limit(cfg.data),
    )
    val_loader = None
    if cfg.data.val_path:
        val_loader, _ = make_dataloader(
            cfg.data.val_path, cfg.data, cfg.train.batch_size, False,
            normalizer, cfg.train.num_workers, limit=limit,
        )

    model = build_model(cfg.model).to(device)
    crit = HungarianLoss(cfg.loss)
    # filter(requires_grad) so a frozen pretrained encoder isn't handed to the optimizer.
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay, betas=tuple(cfg.train.opt_betas),
    )
    # Resume: a checkpoint carries model + optimizer + epoch + the selection state, so resuming
    # picks the run back up rather than restarting momentum and the patience counter from
    # scratch. It is a *stateful* resume, not a bit-identical replay — the dataloader's shuffle
    # order is not restored, so the trajectory after a resume differs from an uninterrupted run.
    start_epoch, history = 0, []
    best_val, best_epoch, bad_epochs = float("inf"), -1, 0
    last_ckpt, best_ckpt = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    if resume and last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_epoch = state["epoch"] + 1
        history = state["history"]
        best_val = state.get("best_val", float("inf"))
        best_epoch = state.get("best_epoch", -1)
        bad_epochs = state.get("bad_epochs", 0)
        log(f"[{cfg.name}] resumed at epoch {start_epoch} (best val {best_val:.5f} @ ep {best_epoch + 1})")

    # Early stopping needs a validation signal; without one there is nothing to select on.
    early_stop = cfg.train.early_stopping and val_loader is not None
    if cfg.train.early_stopping and val_loader is None:
        log(f"[{cfg.name}] no val split -> early stopping disabled, final epoch is the result")

    epochs = max_epochs if max_epochs is not None else cfg.train.epochs
    steps = sum(h.get("steps", 0) for h in history)
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        tr = _run_epoch(model, train_loader, crit, device, opt, cfg.loss.aux_weight)
        va = _run_epoch(model, val_loader, crit, device) if val_loader else {}
        steps += len(train_loader)
        history.append({
            "epoch": epoch, "train": tr, "val": va,
            "lr": float(opt.param_groups[0]["lr"]),
            "seconds": round(time.time() - t0, 2),
            "steps": len(train_loader),
            "cum_steps": steps,
        })

        # Track the best model by validation loss. Without this, evaluation would score whatever
        # the final epoch happened to be — which is not the model you would ship.
        improved = bool(va) and va["loss"] < best_val - cfg.train.early_stopping_min_delta
        if improved:
            # Plain Python scalars only: torch>=2.6 loads with weights_only=True by default, and
            # a stray numpy scalar in here would break resume on the cluster and nowhere else.
            best_val, best_epoch, bad_epochs = float(va["loss"]), epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": best_val}, best_ckpt)
        elif va:
            bad_epochs += 1

        # Checkpoint on a cadence (and always on the final epoch). history.json is rewritten
        # every epoch so a live run's curves can be watched without waiting for a checkpoint.
        if epoch % cfg.train.ckpt_every == 0 or epoch == epochs - 1:
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                 "history": history, "best_val": best_val, "best_epoch": best_epoch,
                 "bad_epochs": bad_epochs},
                last_ckpt,
            )
        with open(results_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        msg = f"[{cfg.name}] ep {epoch + 1}/{epochs} train {tr['loss']:.5f}"
        if va:
            msg += (f" | val {va['loss']:.5f} (t1 {va['t1']:.4f} t2 {va['t2']:.4f} "
                    f"wt {va['wt']:.4f} ex {va['ex']:.4f})")
            msg += "  *best*" if improved else f"  (no gain {bad_epochs}/{cfg.train.early_stopping_patience})"
        log(msg)

        if early_stop and bad_epochs >= cfg.train.early_stopping_patience:
            log(f"[{cfg.name}] early stop at epoch {epoch + 1}: no val gain for "
                f"{bad_epochs} epochs. Best {best_val:.5f} @ epoch {best_epoch + 1}.")
            break

    # Hand back the model that should actually be evaluated, so callers cannot accidentally
    # score the last epoch instead of the best one.
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device)["model"])
        log(f"[{cfg.name}] loaded best.pt (epoch {best_epoch + 1}, val {best_val:.5f}) for evaluation")

    return history, str(results_dir), model
