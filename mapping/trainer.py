from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm

from mapping.model.mapping_model import LinearMapper, cosine_loss


@dataclass
class TrainResult:
    model: LinearMapper
    train_loss: list[float]
    val_loss: list[float]
    test_cosine: float
    test_mse: float


def train_mapper(
    train_set: TensorDataset,
    val_set: TensorDataset,
    test_set: TensorDataset,
    batch_size: int,
    hidden: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> TrainResult:
    in_dim = train_set.tensors[0].shape[1]
    out_dim = train_set.tensors[1].shape[1]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = LinearMapper(in_dim=in_dim, out_dim=out_dim, hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss_hist: list[float] = []
    val_loss_hist: list[float] = []

    for _ in tqdm(range(epochs), desc="Training mapper"):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = cosine_loss(out, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_epoch = running / len(train_set)
        train_loss_hist.append(train_epoch)

        if len(val_set) > 0:
            model.eval()
            v_running = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    out = model(xb)
                    loss = cosine_loss(out, yb)
                    v_running += loss.item() * xb.size(0)
            val_epoch = v_running / len(val_set)
        else:
            val_epoch = float("nan")
        val_loss_hist.append(val_epoch)

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
        val_loss=val_loss_hist,
        test_cosine=test_cos,
        test_mse=test_mse,
    )


def save_checkpoint(
    out_path: Path,
    model: LinearMapper,
    train_loss: list[float],
    val_loss: list[float],
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
            "val_loss": val_loss,
        },
        out_path,
    )
