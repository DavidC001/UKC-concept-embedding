from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def _collect_2d_tensors(obj: Any, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
    found: list[tuple[str, torch.Tensor]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                found.append((key, v))
            elif isinstance(v, dict):
                found.extend(_collect_2d_tensors(v, key))
    return found


def load_rote_embeddings(checkpoint_path: Path) -> np.ndarray:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing RotE checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, torch.Tensor) and ckpt.ndim == 2:
        return ckpt.detach().cpu().float().numpy()

    candidates = _collect_2d_tensors(ckpt)
    if not candidates and isinstance(ckpt, dict) and "state_dict" in ckpt:
        candidates = _collect_2d_tensors(ckpt["state_dict"])

    if not candidates:
        raise RuntimeError(f"Could not find 2D embedding tensor in checkpoint: {checkpoint_path}")

    def rank(item: tuple[str, torch.Tensor]) -> tuple[int, int]:
        key, tensor = item
        key_lower = key.lower()
        score = 0
        if "entity.weight" in key_lower:
            score += 100
        if "entity" in key_lower:
            score += 20
        if "concept" in key_lower:
            score += 10
        if "emb" in key_lower:
            score += 5
        return score, tensor.shape[0]

    best_key, best_tensor = sorted(candidates, key=rank, reverse=True)[0]
    print(f"Using RotE tensor '{best_key}' with shape={tuple(best_tensor.shape)}")
    return best_tensor.detach().cpu().float().numpy()


def load_entity_to_id(entity_to_id_path: Path) -> dict[str, int]:
    if not entity_to_id_path.exists():
        raise FileNotFoundError(f"Missing entity_to_id mapping: {entity_to_id_path}")
    with open(entity_to_id_path, "rb") as fh:
        raw = pickle.load(fh)

    if isinstance(raw, dict):
        return {str(k): int(v) for k, v in raw.items()}
    if isinstance(raw, list):
        return {str(k): int(v) for k, v in raw}
    raise TypeError(f"Unsupported entity_to_id type: {type(raw)!r}")


def build_concept_id_to_label(concepts_df: pd.DataFrame | None) -> dict[str, str]:
    if concepts_df is None:
        return {}

    id_candidates = "id"
    label_candidates = "label"
    
    id_col = next((col for col in concepts_df.columns if id_candidates in col.lower()), None)
    label_col = next((col for col in concepts_df.columns if label_candidates in col.lower()), None)
    
    if id_col is None or label_col is None:
        return {}

    mapping: dict[str, str] = {}
    for _, row in concepts_df.iterrows():
        concept_id = str(row[id_col])
        label = str(row[label_col])
        mapping[concept_id] = label
    return mapping


def align_concepts_to_rote(
    concept_ids: np.ndarray,
    entity_to_id: dict[str, int],
    concept_id_to_label: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    concept_id_to_label = concept_id_to_label or {}
    indices = np.full((len(concept_ids),), -1, dtype=np.int64)

    for i, concept_id in enumerate(concept_ids):
        cid = str(concept_id)
        idx = entity_to_id.get(cid)
        if idx is None:
            label = concept_id_to_label.get(cid)
            if label is not None:
                idx = entity_to_id.get(label)
        if idx is not None:
            indices[i] = int(idx)

    mask = indices >= 0
    return indices, mask
