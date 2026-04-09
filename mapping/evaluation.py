from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


def topk_neighbors(
    mapped_norm: np.ndarray,
    rote_norm: np.ndarray,
    top_k: int,
    query_chunk_size: int = 512,
    corpus_chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if query_chunk_size <= 0 or corpus_chunk_size <= 0:
        raise ValueError("chunk sizes must be > 0")
    if rote_norm.shape[0] == 0:
        raise ValueError("rote_norm is empty")

    k = min(top_k, rote_norm.shape[0])
    n_queries = mapped_norm.shape[0]

    out_idx = np.full((n_queries, k), -1, dtype=np.int64)
    out_scores = np.full((n_queries, k), -np.inf, dtype=np.float32)

    for q_start in range(0, n_queries, query_chunk_size):
        q_end = min(q_start + query_chunk_size, n_queries)
        q_block = mapped_norm[q_start:q_end]

        best_scores = np.full((q_block.shape[0], k), -np.inf, dtype=np.float32)
        best_idx = np.full((q_block.shape[0], k), -1, dtype=np.int64)

        for c_start in range(0, rote_norm.shape[0], corpus_chunk_size):
            c_end = min(c_start + corpus_chunk_size, rote_norm.shape[0])
            c_block = rote_norm[c_start:c_end]

            sim_block = q_block @ c_block.T
            local_k = min(k, sim_block.shape[1])

            local_pos = np.argpartition(-sim_block, kth=local_k - 1, axis=1)[:, :local_k]
            local_scores = np.take_along_axis(sim_block, local_pos, axis=1)
            local_idx = local_pos.astype(np.int64, copy=False) + c_start

            merged_scores = np.concatenate([best_scores, local_scores], axis=1)
            merged_idx = np.concatenate([best_idx, local_idx], axis=1)

            keep_pos = np.argpartition(-merged_scores, kth=k - 1, axis=1)[:, :k]
            best_scores = np.take_along_axis(merged_scores, keep_pos, axis=1)
            best_idx = np.take_along_axis(merged_idx, keep_pos, axis=1)

        order = np.argsort(-best_scores, axis=1)
        out_scores[q_start:q_end] = np.take_along_axis(best_scores, order, axis=1)
        out_idx[q_start:q_end] = np.take_along_axis(best_idx, order, axis=1)

    return out_idx, out_scores


def _resolve_label(raw_id: str, id_to_label: dict[str, str] | None) -> str:
    if id_to_label is None:
        return str(raw_id)
    label = id_to_label.get(str(raw_id))
    if label is None or not str(label).strip():
        return str(raw_id)
    return str(label)


def save_topk_report(
    out_path: Path,
    concept_ids: np.ndarray,
    top_idx: np.ndarray,
    top_scores: np.ndarray,
    idx_to_entity: dict[int, str],
    concept_id_to_label: dict[str, str] | None = None,
    entity_id_to_label: dict[str, str] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "concept_id",
            "concept_label",
            "rank",
            "rote_index",
            "rote_entity",
            "rote_entity_label",
            "cosine",
        ])
        for i, concept_id in enumerate(concept_ids):
            concept_id_str = str(concept_id)
            concept_label = _resolve_label(concept_id_str, concept_id_to_label)
            for rank in range(top_idx.shape[1]):
                rote_idx = int(top_idx[i, rank])
                rote_entity = idx_to_entity.get(rote_idx, "")
                rote_entity_label = _resolve_label(rote_entity, entity_id_to_label)
                writer.writerow(
                    [
                        concept_id_str,
                        concept_label,
                        rank + 1,
                        rote_idx,
                        rote_entity,
                        rote_entity_label,
                        float(top_scores[i, rank]),
                    ]
                )


def run_geodesic_analysis(
    concept_ids: np.ndarray,
    top1_indices: np.ndarray,
    idx_to_entity: dict[int, str],
    concept_id_to_label: dict[str, str] | None,
    entity_id_to_label: dict[str, str] | None,
    concept_relations_csv: Path,
    output_dir: Path,
    sample_size: int,
    seed: int,
) -> None:
    try:
        import networkx as nx
    except ImportError:
        print("Skipping geodesic analysis: networkx is not installed")
        return

    if not concept_relations_csv.exists():
        print(f"Skipping geodesic analysis: relation file not found at {concept_relations_csv}")
        return

    rels = pd.read_csv(concept_relations_csv)
    lower_cols = [c.lower() for c in rels.columns]
    if "src_con_id" in lower_cols and "trg_con_id" in lower_cols:
        src_col = rels.columns[lower_cols.index("src_con_id")]
        trg_col = rels.columns[lower_cols.index("trg_con_id")]
    else:
        src_col = rels.columns[0]
        trg_col = rels.columns[1]

    graph = nx.Graph()
    graph.add_edges_from(zip(rels[src_col].astype(str), rels[trg_col].astype(str)))

    np.random.seed(seed)
    n = len(concept_ids)
    if n == 0:
        print("Skipping geodesic analysis: no aligned concept ids")
        return

    sample_n = min(sample_size, n)
    sample_idx = np.random.choice(n, sample_n, replace=False)

    distances: list[int] = []
    random_distances: list[int] = []
    entity_values = np.array(list(idx_to_entity.values()), dtype=object)
    if entity_values.size == 0:
        print("Skipping random baseline geodesic analysis: idx_to_entity is empty")

    distance_rows: list[dict[str, object]] = []
    random_rows: list[dict[str, object]] = []

    for i in sample_idx:
        src = str(concept_ids[i])
        src_label = _resolve_label(src, concept_id_to_label)
        pred_idx = int(top1_indices[i])
        tgt = idx_to_entity.get(pred_idx)
        tgt_label = _resolve_label(tgt or "", entity_id_to_label) if tgt is not None else ""
        if tgt is None:
            continue
        if src not in graph or tgt not in graph:
            continue
        try:
            d = nx.shortest_path_length(graph, source=src, target=tgt)
            distances.append(int(d))
            distance_rows.append(
                {
                    "concept_id": src,
                    "concept_label": src_label,
                    "mapped_entity": tgt,
                    "mapped_entity_label": tgt_label,
                    "geodesic_distance": int(d),
                }
            )
        except nx.NetworkXNoPath:
            continue

        if entity_values.size > 0:
            rand_tgt = str(entity_values[np.random.randint(0, entity_values.size)])
            rand_tgt_label = _resolve_label(rand_tgt, entity_id_to_label)
            if rand_tgt in graph:
                try:
                    rd = nx.shortest_path_length(graph, source=src, target=rand_tgt)
                    random_distances.append(int(rd))
                    random_rows.append(
                        {
                            "concept_id": src,
                            "concept_label": src_label,
                            "random_entity": rand_tgt,
                            "random_entity_label": rand_tgt_label,
                            "geodesic_distance": int(rd),
                        }
                    )
                except nx.NetworkXNoPath:
                    pass

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(distance_rows if distance_rows else {"geodesic_distance": distances}).to_csv(
        output_dir / "geodesic_distances.csv",
        index=False,
    )
    pd.DataFrame(random_rows if random_rows else {"geodesic_distance": random_distances}).to_csv(
        output_dir / "geodesic_distances_random.csv",
        index=False,
    )

    if not distances:
        print("No geodesic distances found for sampled pairs")
        return

    try:
        import matplotlib.pyplot as plt

        bins = list(range(0, max(distances) + 2))
        plt.figure(figsize=(7, 4))
        plt.hist(distances, bins=bins, color="#2f7f7f", edgecolor="black")
        plt.title("Geodesic distance between concept and top-1 mapped entity")
        plt.xlabel("Geodesic distance")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / "geodesic_distance.png", dpi=160)
        plt.close()

        if random_distances:
            bins_random = list(range(0, max(random_distances) + 2))
            plt.figure(figsize=(7, 4))
            plt.hist(random_distances, bins=bins_random, color="#ad6a3d", edgecolor="black")
            plt.title("Geodesic distance for random baseline entity")
            plt.xlabel("Geodesic distance")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(output_dir / "geodesic_distance_random.png", dpi=160)
            plt.close()

            bins_max = max(max(distances), max(random_distances))
            bins_cmp = list(range(0, bins_max + 2))
            plt.figure(figsize=(8, 4.5))
            plt.hist(distances, bins=bins_cmp, alpha=0.55, color="#2f7f7f", edgecolor="black", label="Mapped top-1")
            plt.hist(random_distances, bins=bins_cmp, alpha=0.55, color="#ad6a3d", edgecolor="black", label="Random baseline")
            plt.title("Geodesic distance: mapped top-1 vs random baseline")
            plt.xlabel("Geodesic distance")
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "geodesic_distance_comparison.png", dpi=160)
            plt.close()
        else:
            print("No random baseline geodesic distances found")
    except ImportError:
        print("Skipping geodesic plot: matplotlib is not installed")
