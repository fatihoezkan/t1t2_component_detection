"""Device selection — one place that decides where tensors live.

The rule is simple: use CUDA if there's a GPU (the cluster), fall back to Apple's MPS for
local smoke tests on the Mac, and CPU if neither is around. Everything else in the package
asks this module rather than calling torch.cuda directly, so a run is portable between the
laptop and the cluster with no code change.
"""
from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the device to run on. `prefer` (e.g. "cuda"/"cpu") wins if given; otherwise
    auto-detect cuda > mps > cpu."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info() -> str:
    """A short human-readable line for logs: what we're running on."""
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps (Apple GPU)"
    return "cpu"
