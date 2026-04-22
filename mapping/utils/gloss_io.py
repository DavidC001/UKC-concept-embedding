from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_GLOSS_COLUMNS = ("concept_id", "gloss")


def load_concept_glosses(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing concept glosses CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_GLOSS_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {missing}. "
            f"Expected columns include {list(REQUIRED_GLOSS_COLUMNS)}"
        )

    df = df.loc[:, list(REQUIRED_GLOSS_COLUMNS)].copy()
    df["concept_id"] = df["concept_id"].astype(str)
    df["gloss"] = df["gloss"].astype(str).fillna("")
    return df


def load_concepts(concepts_csv: Path) -> pd.DataFrame | None:
    if not concepts_csv.exists():
        return None
    return pd.read_csv(concepts_csv)


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (arr / norms).astype(np.float32)


def save_embeddings_npz(
    output_path: Path,
    concept_ids: Iterable[str],
    embeddings: np.ndarray,
    concept_labels: Iterable[str] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "concept_ids": np.asarray(list(concept_ids)),
        "embeddings": embeddings.astype(np.float32),
    }
    if concept_labels is not None:
        payload["concept_labels"] = np.asarray(list(concept_labels))
    np.savez_compressed(output_path, **payload)
