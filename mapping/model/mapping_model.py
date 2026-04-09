from __future__ import annotations

import torch
import torch.nn as nn


class LinearMapper(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 1024) -> None:
        super().__init__()
        # self.net = nn.Linear(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def cosine_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = x / x.norm(dim=1, keepdim=True).clamp(min=1e-12)
    y_norm = y / y.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return 1 - (x_norm * y_norm).sum(dim=1).mean()
