from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm

from mapping.model.mapping_model import LinearMapper, contrastive_loss


@dataclass
class TrainResult:
    model: LinearMapper
    train_loss: list[float]
    test_cosine: float
    test_mse: float


def train_mapper(
    train_set: TensorDataset,
    test_set: TensorDataset,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> TrainResult:
    # Handle both TensorDataset and Subset cases
    dataset = train_set.dataset
    in_dim = dataset.tensors[0].shape[1]
    out_dim = dataset.tensors[1].shape[1]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = LinearMapper(in_dim=in_dim, out_dim=out_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss_hist: list[float] = []

    for _ in tqdm(range(epochs), desc="Training mapper"):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = contrastive_loss(out, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_epoch = running / len(train_set)
        train_loss_hist.append(train_epoch)

    model.eval()
    if len(test_set) > 0:
        with torch.no_grad():
            total_cos = 0.0
            total_sqerr = 0.0
            total_examples = 0
            total_values = 0

            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)

                pred_norm = pred / pred.norm(dim=1, keepdim=True).clamp(min=1e-12)
                yb_norm = yb / yb.norm(dim=1, keepdim=True).clamp(min=1e-12)
                total_cos += float((pred_norm * yb_norm).sum(dim=1).sum().item())

                total_sqerr += float(((pred - yb) ** 2).sum().item())
                total_examples += xb.size(0)
                total_values += xb.size(0) * pred.size(1)

            test_cos = total_cos / max(total_examples, 1)
            test_mse = total_sqerr / max(total_values, 1)
    else:
        test_cos = float("nan")
        test_mse = float("nan")

    return TrainResult(
        model=model,
        train_loss=train_loss_hist,
        test_cosine=test_cos,
        test_mse=test_mse,
    )


def save_checkpoint(
    out_path: Path,
    model: LinearMapper,
    train_loss: list[float],
    in_dim: int,
    out_dim: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_dim": in_dim,
            "out_dim": out_dim,
            "train_loss": train_loss,
        },
        out_path,
    )
