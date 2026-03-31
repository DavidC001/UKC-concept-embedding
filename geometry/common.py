"""Shared utilities for geometry analysis."""

from typing import Iterable

import torch

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def progress_iter(iterable: Iterable, enabled: bool, desc: str, total: int = None):
    if enabled and tqdm is not None:
        return tqdm(iterable, desc=desc, total=total)
    return iterable


def safe_norm(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    n = v.norm()
    return v / (n + eps)
