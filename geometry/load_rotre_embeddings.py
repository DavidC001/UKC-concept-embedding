"""Utils for loaing RotE model checkpoint and get entity embeddings"""

from typing import Tuple
import pickle
import os

import torch
import numpy as np


def load_state_dict(path: str, map_location='cpu') -> dict:
    sd = torch.load(path, map_location=map_location)

    return sd


def get_entity_embeddings_from_state_dict(sd: dict) -> torch.Tensor:
    return sd['entity.weight'].detach().cpu()


def load_embeddings(checkpoint_path: str, map_location='cpu') -> torch.Tensor:
    sd = load_state_dict(checkpoint_path, map_location=map_location)
    emb = get_entity_embeddings_from_state_dict(sd)

    return emb


def whiten_embeddings(emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Center embeddings, compute covariance and multiply by a matrix to get identity matrix as covariance. 
    Returns the whitened embeddings, the mean and the whitening matrix (for later unwhitening if needed).
    """
    if not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb)

    W, d = emb.shape
    mean = torch.mean(emb, dim=0)
    centered = emb - mean

    cov = centered.T @ centered / float(W)
    eps = 1e-12
    evals, evecs = torch.linalg.eigh(cov + eps * torch.eye(d, device=cov.device))
    inv_sqrt = evecs @ torch.diag(1.0 / torch.sqrt(evals)) @ evecs.T
    g = centered @ inv_sqrt
    return g, mean, inv_sqrt


def load_entity_to_index(entity_pickle_path: str) -> dict:
    with open(entity_pickle_path, 'rb') as f:
        mapping = pickle.load(f)
    return mapping


def build_vocab_list(num_entities: int, entity_to_id: dict, id_to_label: dict=None) -> list:
    labels = [None] * num_entities
    for orig_id_str, idx in entity_to_id.items():
        if id_to_label and orig_id_str in id_to_label:
            labels[idx] = id_to_label[orig_id_str]
        else:
            labels[idx] = f"ent_{orig_id_str}"
    return labels
