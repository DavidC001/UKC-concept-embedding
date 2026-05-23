from __future__ import annotations

from typing import Iterable

from sentence_transformers import SentenceTransformer
import numpy as np
import torch

def encode_glosses(
    glosses: Iterable[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:

    model = SentenceTransformer(model_name)
    model.to(device)
    emb = model.encode(
        list(glosses),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device,
    )
    return emb.astype(np.float32)
