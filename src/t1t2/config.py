"""An experiment, described entirely as data.

One idea runs through this whole package: a run is fully defined by its config and nothing
else. Want a bigger decoder, log vs linear normalization, a different loss balance, the
signal-consistency term? Those are config edits, not code edits. If two runs differ, their
YAMLs differ — there is no hidden switch in the code. That is what makes the eventual
comparison across experiments meaningful and a run reproducible months later.

The config splits into the four things you actually tune independently — data, model, loss,
training — each its own dataclass so fields are typed and discoverable. Loading is YAML in,
dataclass out; saving is the reverse, and every run drops its own config next to its results.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    """Where the data lives and how it is transformed on the way into the model."""

    train_path: str
    val_path: Optional[str] = None
    test_path: Optional[str] = None

    n_inputs: int = 64            # the fixed thesis setting: 8 TI × 8 TE, in scanner order
    max_comp: int = 3             # widest a voxel's compartment table ever gets

    # How T1/T2 (ms) map into the model's [0, 1] output range (see data.py for why log is the
    # default). Bounds are the physical window we expect; targets outside get clamped.
    normalization: str = "log_minmax"     # identity | linear_minmax | log_minmax
    t1_min: float = 100.0
    t1_max: float = 7000.0
    t2_min: float = 5.0
    t2_max: float = 4000.0

    # Per-voxel input rescaling (none | max | first). Real scans arrive at an arbitrary
    # overall scale, so this must be applied identically to synthetic and real data later.
    signal_norm: str = "none"


@dataclass
class ModelConfig:
    """The network's shape — these sizes are the levers for the architecture sweep."""

    input_dim: int = 64
    hidden_dim: int = 512
    fs_dim: int = 256             # feature / query vector width
    n_queries: int = 10           # most compartments the model can ever propose
    n_dlayers: int = 4            # transformer decoder depth
    n_heads: int = 4              # attention heads per decoder layer
    aux_loss: bool = False        # supervise every decoder layer, not just the last
    pretrain_path: Optional[str] = None   # optional warm-start weights for the encoder
    freeze_encoder: bool = False


@dataclass
class LossConfig:
    """How the loss terms are balanced, plus the (not-yet-wired) physics term.

    With log-normalized targets these weights all sit near 1.0 — no powers-of-ten juggling.
    The signal-consistency fields are the hook for a later extension: resynthesize the signal
    from the prediction and penalize the mismatch. Note the default type is **mse**, not
    Rician: our simulated data is signed (keeps IR negatives), so plain MSE is the right
    likelihood here; the Rician-loss caveat in the literature is for magnitude data.
    """

    t1_weight: float = 1.0
    t2_weight: float = 1.0
    w_weight: float = 1.0
    exist_weight: float = 0.1
    aux_weight: float = 1.0       # how strongly earlier decoder layers' aux losses ramp in

    signal_consistency: bool = False           # later milestone; loss ignores this for now
    signal_consistency_weight: float = 0.0
    signal_consistency_type: str = "mse"       # mse | rician


@dataclass
class TrainConfig:
    """The optimization schedule and run mechanics."""

    epochs: int = 200
    batch_size: int = 256
    lr: float = 1.0e-4
    lr_step: int = 1000
    weight_decay: float = 1.0e-4
    opt_betas: tuple = (0.9, 0.98)
    device: Optional[str] = None      # None -> auto-detect (see device.py)
    seed: int = 0
    num_workers: int = 0
    ckpt_every: int = 20              # epochs between checkpoints


@dataclass
class ExperimentConfig:
    """The four sub-configs plus a name and free-text notes — the whole description."""

    name: str
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Write this config to YAML (creating parent dirs). sort_keys=False keeps the
        human-friendly section order instead of alphabetising everything."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def load_config(path: str | Path) -> ExperimentConfig:
    """Read a YAML file into a fully-typed ExperimentConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return from_dict(raw)


def from_dict(raw: dict) -> ExperimentConfig:
    """Rebuild the nested dataclasses from a plain dict.

    Explicit rather than a generic recursive helper: the nesting is shallow, and doing it by
    hand makes a missing or renamed key fail loudly instead of being silently dropped. The one
    quirk: YAML gives opt_betas back as a list, and we want a tuple.
    """
    data = DataConfig(**raw["data"])
    model = ModelConfig(**raw.get("model", {}))
    loss = LossConfig(**raw.get("loss", {}))
    train_raw = dict(raw.get("train", {}))
    if train_raw.get("opt_betas") is not None:
        train_raw["opt_betas"] = tuple(train_raw["opt_betas"])
    train = TrainConfig(**train_raw)
    return ExperimentConfig(
        name=raw["name"],
        data=data,
        model=model,
        loss=loss,
        train=train,
        notes=raw.get("notes", ""),
    )
