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
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .data import TargetNormalizer, make_dataloader
from .device import device_info, get_device
from .loss import HungarianLoss
from .model import build_model

# The five numbers the loss returns, in order — kept as a constant so training and eval can't
# drift on what "position 2" means.
_LOSS_KEYS = ("loss", "t1", "t2", "wt", "ex")


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
    cfg.save(results_dir / "config.yaml")                      # snapshot the exact config next to outputs

    # One normalizer shared by train and val so both splits are transformed identically.
    normalizer = TargetNormalizer.from_config(cfg.data)
    train_loader, _ = make_dataloader(
        cfg.data.train_path, cfg.data, cfg.train.batch_size, True,
        normalizer, cfg.train.num_workers, limit=limit,
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
    sched = torch.optim.lr_scheduler.StepLR(opt, cfg.train.lr_step)

    # Resume: a checkpoint carries model + optimizer/scheduler + epoch, so resuming continues
    # the exact trajectory rather than restarting the LR schedule or momentum from scratch.
    start_epoch, history = 0, []
    last_ckpt = ckpt_dir / "last.pt"
    if resume and last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_epoch = state["epoch"] + 1
        history = state["history"]
        log(f"[{cfg.name}] resumed at epoch {start_epoch}")

    epochs = max_epochs if max_epochs is not None else cfg.train.epochs
    for epoch in range(start_epoch, epochs):
        tr = _run_epoch(model, train_loader, crit, device, opt, cfg.loss.aux_weight)
        va = _run_epoch(model, val_loader, crit, device) if val_loader else {}
        history.append({"epoch": epoch, "train": tr, "val": va})

        # Checkpoint on a cadence (and always on the final epoch). history.json is rewritten
        # every epoch so a live run's curves can be watched without waiting for a checkpoint.
        if epoch % cfg.train.ckpt_every == 0 or epoch == epochs - 1:
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "epoch": epoch, "history": history},
                last_ckpt,
            )
        with open(results_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        sched.step()
        msg = f"[{cfg.name}] ep {epoch + 1}/{epochs} train {tr['loss']:.5f}"
        if va:
            msg += f" | val {va['loss']:.5f} (t1 {va['t1']:.4f} t2 {va['t2']:.4f} wt {va['wt']:.4f} ex {va['ex']:.4f})"
        log(msg)

    return history, str(results_dir), model
