#!/usr/bin/env python3
from __future__ import annotations

import os
import random
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from mapping.config import parse_args
from mapping.evaluation import run_geodesic_analysis, save_topk_report, topk_neighbors
from mapping.model.gloss_encoder import encode_glosses
from mapping.trainer import save_checkpoint, train_mapper
from mapping.utils.gloss_io import l2_normalize, load_concept_glosses, load_concepts, save_embeddings_npz
from mapping.utils.load_embeddings import (
    align_concepts_to_rote,
    build_concept_id_to_label,
    load_entity_to_id,
    load_rote_embeddings,
)

from torch.utils.data import TensorDataset, random_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def _map_in_batches(
    model: torch.nn.Module,
    gloss_emb: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("mapping inference batch size must be > 0")

    mapped_chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, gloss_emb.shape[0], batch_size):
            end = min(start + batch_size, gloss_emb.shape[0])
            batch = gloss_emb[start:end].to(device)
            mapped_chunks.append(model(batch).cpu().numpy())
    return np.concatenate(mapped_chunks, axis=0).astype(np.float32, copy=False)


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.random_seed)
        
    gloss_df = load_concept_glosses(cfg.concept_glosses_csv, cfg.concept_relations_csv)
    concept_ids = gloss_df["concept_id"].to_numpy(dtype=str)
    
    if os.path.exists(cfg.output_dir / f"{cfg.model_name}_raw.npz") and os.path.exists(cfg.output_dir / f"{cfg.model_name}_norm.npz"):
        print("Found existing embeddings, skipping encoding")
        raw_emb = np.load(cfg.output_dir / f"{cfg.model_name}_raw.npz")["embeddings"]
        norm_emb = np.load(cfg.output_dir / f"{cfg.model_name}_norm.npz")["embeddings"]
    else:
        glosses = gloss_df["gloss"].tolist()

        print(f"Loaded gloss rows: {len(gloss_df)}")
        raw_emb = encode_glosses(
            glosses=glosses,
            model_name=cfg.model_name,
            batch_size=cfg.embedding_batch_size,
            device=cfg.device,
        )
        norm_emb = l2_normalize(raw_emb)

        save_embeddings_npz(cfg.output_dir / f"{cfg.model_name}_raw.npz", concept_ids, raw_emb)
        save_embeddings_npz(cfg.output_dir / f"{cfg.model_name}_norm.npz", concept_ids, norm_emb)

    rote_emb = load_rote_embeddings(cfg.rote_checkpoint)
    rote_norm = l2_normalize(rote_emb)

    # load concept_id to matrix idx mapping from entity_to_id.pickle
    entity_to_id = load_entity_to_id(cfg.entity_to_id)
    idx_to_entity = {idx: ent for ent, idx in entity_to_id.items()}

    concepts_df = load_concepts(cfg.concepts_csv)
    # build concept_id to label mapping from concepts.csv
    id_to_label = build_concept_id_to_label(concepts_df)
    entity_id_to_label = {ent_id: id_to_label.get(ent_id, "NONE") for ent_id in entity_to_id.keys()}
    
    paired_idx, aligned_mask = align_concepts_to_rote(concept_ids, entity_to_id, id_to_label)
    matched = int(aligned_mask.sum())
    print(f"Aligned concepts: {matched}/{len(concept_ids)}")
    if matched == 0:
        raise RuntimeError("No concepts from concept_glosses.csv aligned to RotE entities")

    X = norm_emb[aligned_mask]
    Y = rote_norm[paired_idx[aligned_mask]]
    aligned_ids = concept_ids[aligned_mask]

    # split into train/val/test sets
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    generator = torch.Generator().manual_seed(cfg.random_seed)
    train_split = 1.0  - cfg.test_ratio
    train_set, _, test_set = random_split(
        dataset,
        [train_split, 0.0, cfg.test_ratio],
        generator=generator,
    )

    result = train_mapper(
        train_set=train_set,
        test_set=test_set,        
        batch_size=cfg.mapper_batch_size,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        device=cfg.device,
    )

    print(f"Test cosine: {result.test_cosine:.6f}")
    print(f"Test MSE: {result.test_mse:.6e}")

    save_checkpoint(
        out_path=cfg.output_dir / "linear_mapper_qwen_to_rote.pt",
        model=result.model,
        train_loss=result.train_loss,
        in_dim=X.shape[1],
        out_dim=Y.shape[1],
    )

    test_indices = np.asarray(test_set.indices, dtype=np.int64)
    test_ids = aligned_ids[test_indices]
    test_gloss_emb = torch.from_numpy(X[test_indices])

    result.model.eval()
    mapped = _map_in_batches(
        model=result.model,
        gloss_emb=test_gloss_emb,
        device=cfg.device,
        batch_size=cfg.mapper_batch_size,
    )
    mapped_norm = l2_normalize(mapped)

    top_idx, top_scores = topk_neighbors(
        mapped_norm=mapped_norm,
        rote_norm=rote_norm,
        top_k=cfg.top_k,
    )
    save_topk_report(
        out_path=cfg.output_dir / "topk_neighbors.csv",
        concept_ids=test_ids,
        top_idx=top_idx,
        top_scores=top_scores,
        idx_to_entity=idx_to_entity,
        concept_id_to_label=id_to_label,
        entity_id_to_label=entity_id_to_label,
    )

    print(f"Saved top-k report: {cfg.output_dir / 'topk_neighbors.csv'}")

    
    if cfg.run_geodesic:
        run_geodesic_analysis(
            concept_ids=test_ids,
            top1_indices=top_idx[:, 0],
            idx_to_entity=idx_to_entity,
            concept_id_to_label=id_to_label,
            entity_id_to_label=entity_id_to_label,
            concept_relations_csv=cfg.concept_relations_csv,
            output_dir=cfg.output_dir,
            seed=cfg.random_seed,
        )

    np.savez_compressed(
        cfg.output_dir / "mapped_embeddings.npz",
        concept_ids=test_ids,
        mapped_embeddings=mapped_norm.astype(np.float32),
    )
    print(f"Pipeline completed. Outputs in: {cfg.output_dir}")


if __name__ == "__main__":
    main()
