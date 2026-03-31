"""Mapping and embedding loading utilities."""

from collections import defaultdict
import pickle

import pandas as pd
import torch


def load_rote_embeddings(model_path):
    ckpt = torch.load(model_path, map_location="cpu")
    return ckpt["entity.weight"].detach().cpu().float()


def create_bidirectional_mappings(pickle_path):
    with open(pickle_path, "rb") as f:
        entity_to_idx = pickle.load(f)
    concept_id_to_index = {int(k): int(v) for k, v in entity_to_idx.items()}
    index_to_concept_id = {v: k for k, v in concept_id_to_index.items()}
    return concept_id_to_index, index_to_concept_id


def load_concept_id_to_uk_id(csv_path):
    df = pd.read_csv(csv_path)
    id_col = pd.to_numeric(df["id"], errors="coerce")
    uk_col = pd.to_numeric(df["uk_id"], errors="coerce")
    valid = id_col.notna() & uk_col.notna()
    return dict(zip(id_col[valid].astype("int64"), uk_col[valid].astype("int64")))


def load_uk_id_to_concept_id(csv_path):
    df = pd.read_csv(csv_path)
    id_col = pd.to_numeric(df["id"], errors="coerce")
    uk_col = pd.to_numeric(df["uk_id"], errors="coerce")
    valid = id_col.notna() & uk_col.notna()
    return dict(zip(uk_col[valid].astype("int64"), id_col[valid].astype("int64")))

def load_is_a_edges_from_csv(csv_path, relation_type=20):
    """Load IS_A edges from CSV as (parent_concept_id, child_concept_id)."""
    edges = []
    in_relations_copy = False

    relations_df = pd.read_csv(csv_path, usecols=['relation_type', 'src_con_id', 'trg_con_id'])
    
    for _, row in relations_df.iterrows():
        if row['relation_type'] == relation_type:
            parent_id = row['src_con_id']
            child_id = row['trg_con_id']
            if parent_id != child_id:
                edges.append((parent_id, child_id))
    
    return edges


def build_parent_index_from_is_a(concept_id_to_index, num_concepts, csv_path):
    """Build parent index vector in embedding-index space from IS_A relations."""
    parent_candidates = defaultdict(list)
    seen_parent = defaultdict(set)
    edges = load_is_a_edges_from_csv(csv_path=csv_path, relation_type=20)

    for parent_cid, child_cid in edges:
        parent_idx = concept_id_to_index.get(parent_cid)
        child_idx = concept_id_to_index.get(child_cid)
        if parent_idx is None or child_idx is None or parent_idx == child_idx:
            continue
        if parent_idx not in seen_parent[child_idx]:
            seen_parent[child_idx].add(parent_idx)
            parent_candidates[child_idx].append(parent_idx)

    parent_index = torch.full((num_concepts,), -1, dtype=torch.long)
    num_multi_parent = 0

    for child_idx, parents in parent_candidates.items():
        if len(parents) > 1:
            num_multi_parent += 1
        parent_index[child_idx] = parents[0]

    num_roots = int((parent_index < 0).sum().item())
    return parent_index, num_roots, num_multi_parent
