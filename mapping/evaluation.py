from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def topk_neighbors(mapped_norm: np.ndarray, rote_norm: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    sim = mapped_norm @ rote_norm.T
    idx = np.argsort(-sim, axis=1)[:, :top_k]
    scores = np.take_along_axis(sim, idx, axis=1)
    return idx, scores


def save_topk_report(
    out_path: Path,
    concept_ids: np.ndarray,
    top_idx: np.ndarray,
    top_scores: np.ndarray,
    idx_to_entity: dict[int, str],
) -> None:
    rows: list[dict[str, object]] = []
    for i, concept_id in enumerate(concept_ids):
        for rank in range(top_idx.shape[1]):
            rote_idx = int(top_idx[i, rank])
            rows.append(
                {
                    "concept_id": str(concept_id),
                    "rank": rank + 1,
                    "rote_index": rote_idx,
                    "rote_entity": idx_to_entity.get(rote_idx, ""),
                    "cosine": float(top_scores[i, rank]),
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def run_geodesic_analysis(
    concept_ids: np.ndarray,
    top1_indices: np.ndarray,
    idx_to_entity: dict[int, str],
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
    for i in sample_idx:
        src = str(concept_ids[i])
        pred_idx = int(top1_indices[i])
        tgt = idx_to_entity.get(pred_idx)
        if tgt is None:
            continue
        if src not in graph or tgt not in graph:
            continue
        try:
            d = nx.shortest_path_length(graph, source=src, target=tgt)
            distances.append(int(d))
        except nx.NetworkXNoPath:
            continue

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"geodesic_distance": distances}).to_csv(
        output_dir / "geodesic_distances.csv",
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
    except ImportError:
        print("Skipping geodesic plot: matplotlib is not installed")
