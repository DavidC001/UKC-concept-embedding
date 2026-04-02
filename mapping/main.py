#!/usr/bin/env python3
from __future__ import annotations

import random
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from mapping.config import parse_args
from mapping.evaluation import run_geodesic_analysis, save_topk_report, topk_neighbors
from mapping.model.gloss_encoder import choose_device, encode_glosses
from mapping.trainer import save_checkpoint, train_mapper
from mapping.utils.gloss_io import l2_normalize, load_concept_glosses, load_optional_concepts, save_embeddings_npz
from mapping.utils.load_embeddings import (
    align_concepts_to_rote,
    build_concept_id_to_label,
    load_entity_to_id,
    load_rote_embeddings,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _to_device(device_str: str) -> torch.device:
    if device_str == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.random_seed)

    encoder_device = choose_device(cfg.device)
    train_device = _to_device(encoder_device)
    print(f"Encoder device: {encoder_device}")
    print(f"Training device: {train_device}")

    gloss_df = load_concept_glosses(cfg.concept_glosses_csv)
    concept_ids = gloss_df["concept_id"].to_numpy(dtype=str)
    glosses = gloss_df["gloss"].tolist()

    print(f"Loaded gloss rows: {len(gloss_df)}")
    raw_emb = encode_glosses(
        glosses=glosses,
        model_name=cfg.model_name,
        batch_size=cfg.embedding_batch_size,
        device=encoder_device,
    )
    norm_emb = l2_normalize(raw_emb)

    save_embeddings_npz(cfg.output_dir / "qwen_concept_embeddings_raw.npz", concept_ids, raw_emb)
    save_embeddings_npz(cfg.output_dir / "qwen_concept_embeddings.npz", concept_ids, norm_emb)

    rote_emb = load_rote_embeddings(cfg.rote_checkpoint)
    rote_norm = l2_normalize(rote_emb)

    entity_to_id = load_entity_to_id(cfg.entity_to_id)
    idx_to_entity = {idx: ent for ent, idx in entity_to_id.items()}

    concepts_df = load_optional_concepts(cfg.concepts_csv)
    id_to_label = build_concept_id_to_label(concepts_df)

    paired_idx, aligned_mask = align_concepts_to_rote(concept_ids, entity_to_id, id_to_label)
    matched = int(aligned_mask.sum())
    print(f"Aligned concepts: {matched}/{len(concept_ids)}")
    if matched == 0:
        raise RuntimeError("No concepts from concept_glosses.csv aligned to RotE entities")

    X = norm_emb[aligned_mask]
    Y = rote_norm[paired_idx[aligned_mask]]
    aligned_ids = concept_ids[aligned_mask]

    result = train_mapper(
        X=X,
        Y=Y,
        batch_size=cfg.mapper_batch_size,
        hidden=cfg.mapper_hidden,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.random_seed,
        device=train_device,
    )

    print(f"Test cosine: {result.test_cosine:.6f}")
    print(f"Test MSE: {result.test_mse:.6e}")

    save_checkpoint(
        out_path=cfg.output_dir / "linear_mapper_qwen_to_rote.pt",
        model=result.model,
        train_loss=result.train_loss,
        val_loss=result.val_loss,
        in_dim=X.shape[1],
        out_dim=Y.shape[1],
    )

    result.model.eval()
    with torch.no_grad():
        mapped = result.model(torch.from_numpy(X).float().to(train_device)).cpu().numpy()
    mapped_norm = l2_normalize(mapped)

    top_idx, top_scores = topk_neighbors(mapped_norm=mapped_norm, rote_norm=rote_norm, top_k=cfg.top_k)
    save_topk_report(
        out_path=cfg.output_dir / "topk_neighbors.csv",
        concept_ids=aligned_ids,
        top_idx=top_idx,
        top_scores=top_scores,
        idx_to_entity=idx_to_entity,
    )

    print(f"Saved top-k report: {cfg.output_dir / 'topk_neighbors.csv'}")

    if cfg.run_geodesic:
        run_geodesic_analysis(
            concept_ids=aligned_ids,
            top1_indices=top_idx[:, 0],
            idx_to_entity=idx_to_entity,
            concept_relations_csv=cfg.concept_relations_csv,
            output_dir=cfg.output_dir,
            sample_size=cfg.geodesic_sample_size,
            seed=cfg.random_seed,
        )

    np.savez_compressed(
        cfg.output_dir / "mapped_embeddings.npz",
        concept_ids=aligned_ids,
        mapped_embeddings=mapped_norm.astype(np.float32),
    )
    print(f"Pipeline completed. Outputs in: {cfg.output_dir}")


if __name__ == "__main__":
    main()
