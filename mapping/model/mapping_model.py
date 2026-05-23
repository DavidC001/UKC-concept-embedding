from __future__ import annotations

import torch
import torch.nn as nn


class LinearMapper(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)
        # self.net = nn.Sequential(
        #     nn.Linear(in_dim, 1024),
        #     nn.SiLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(1024, out_dim),
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def contrastive_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # use info-NCE loss as a contrastive loss to align mapped glosses with rote embeddings
    x = nn.functional.normalize(x, dim=1)
    y = nn.functional.normalize(y, dim=1)
    logits = x @ y.T
    labels = torch.arange(x.size(0), device=x.device)
    
    temperature = 0.07
    logits = logits / temperature
    
    return nn.CrossEntropyLoss()(logits, labels)
