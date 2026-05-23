from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

import networkx as nx


def topk_neighbors(
    mapped_norm: np.ndarray,
    rote_norm: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute top-k nearest neighbors based on cosine similarity, processing in chunks for memory efficiency."""
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    
    k = min(top_k, rote_norm.shape[0])
    
    sim_matrix = mapped_norm @ rote_norm.T
    
    out_idx = np.zeros((mapped_norm.shape[0], k), dtype=np.int32)
    out_scores = np.zeros((mapped_norm.shape[0], k), dtype=np.float32)
    
    for start in range(0, sim_matrix.shape[0], 1000):
        end = min(start + 1000, sim_matrix.shape[0])
        batch_sim = sim_matrix[start:end]
        batch_idx = np.argpartition(batch_sim, -k, axis=1)[:, -k:]
        batch_scores = np.take_along_axis(batch_sim, batch_idx, axis=1)
        
        sorted_idx = np.argsort(-batch_scores, axis=1)
        out_idx[start:end] = np.take_along_axis(batch_idx, sorted_idx, axis=1)
        out_scores[start:end] = np.take_along_axis(batch_scores, sorted_idx, axis=1)
    
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
    seed: int,
) -> None:

    if not concept_relations_csv.exists():
        print(f"Skipping geodesic analysis: relation file not found at {concept_relations_csv}")
        return

    rels = pd.read_csv(concept_relations_csv)
    
    # filter only relation_type 20
    rels = rels[rels["relation_type"] == 20]
    
    # remove syntetic roots 114316 114317 and 114318 as they create shortcuts in the graph 
    # that distort geodesic distance analysis
    # rels = rels[~rels["src_con_id"].isin([114316, 114317, 114318])]
    # rels = rels[~rels["trg_con_id"].isin([114316, 114317, 114318])]
    
    src_col = rels["src_con_id"]
    trg_col = rels["trg_con_id"]
    graph = nx.Graph()
    graph.add_edges_from(zip(src_col.astype(str), trg_col.astype(str)))

    np.random.seed(seed)
    n = len(concept_ids)
    if n == 0:
        print("Skipping geodesic analysis: no aligned concept ids")
        return

    distances: list[int] = []
    random_distances: list[int] = []
    entity_values = np.array(list(idx_to_entity.values()), dtype=object)
    if entity_values.size == 0:
        print("Skipping random baseline geodesic analysis: idx_to_entity is empty")

    distance_rows: list[dict[str, object]] = []
    random_rows: list[dict[str, object]] = []

    for i in range(n):
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
        except nx.NetworkXNoPath:
            d = -2
        
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
                    rd = -2
                    random_distances.append(int(rd))

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

        bins = list(range(-2, max(distances) + 2))
        plt.figure(figsize=(7, 4))
        plt.hist(distances, bins=bins, color="#2f7f7f", edgecolor="black")
        plt.title("Geodesic distance between concept and top-1 mapped entity")
        plt.xlabel("Geodesic distance")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / "geodesic_distance.png", dpi=160)
        plt.close()

        if random_distances:
            bins_random = list(range(-2, max(random_distances) + 2))
            plt.figure(figsize=(7, 4))
            plt.hist(random_distances, bins=bins_random, color="#ad6a3d", edgecolor="black")
            plt.title("Geodesic distance for random baseline entity")
            plt.xlabel("Geodesic distance")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(output_dir / "geodesic_distance_random.png", dpi=160)
            plt.close()

            bins_max = max(max(distances), max(random_distances))
            bins_cmp = list(range(-2, bins_max + 2))
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
