"""Direction estimation and hierarchy metric computations."""

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np
import torch

from category import estimate_single_dir_from_embeddings
from common import progress_iter, safe_norm


def estimate_dirs(
    g_whitened: torch.Tensor,
    ent2idx: Dict[str, int],
    node_to_members: Dict[int, List[int]],
    show_progress: bool = False,
) -> Dict[int, Dict[str, torch.Tensor]]:
    dirs: Dict[int, Dict[str, torch.Tensor]] = {}

    items = progress_iter(
        node_to_members.items(),
        enabled=show_progress,
        desc="Estimating category directions",
        total=len(node_to_members),
    )

    for node, members in items:
        mapped = [ent2idx[str(c)] for c in members if str(c) in ent2idx]
        if len(mapped) < 3:
            continue

        cat_emb = g_whitened[mapped]
        lda_dir, mean_dir = estimate_single_dir_from_embeddings(cat_emb)
        dirs[node] = {"lda": lda_dir, "mean": mean_dir}

    return dirs


def cosine_matrix_from_dirs(sorted_nodes: List[int], dirs: Dict[int, Dict[str, torch.Tensor]], version: str):
    vecs = []
    kept_nodes = []
    for n in sorted_nodes:
        if n in dirs:
            vecs.append(safe_norm(dirs[n][version]).unsqueeze(0))
            kept_nodes.append(n)

    if not vecs:
        return np.zeros((0, 0), dtype=np.float32), []

    mat = torch.cat(vecs, dim=0)
    cos = (mat @ mat.T).cpu().numpy()
    return cos, kept_nodes


def shortest_path_matrix(g: nx.DiGraph, nodes: List[int]) -> np.ndarray:
    if not nodes:
        return np.zeros((0, 0), dtype=np.float32)

    und = g.to_undirected()
    ordered_nodes = list(dict.fromkeys(nodes))
    und_sub = und.subgraph(ordered_nodes).copy()
    dist = nx.floyd_warshall_numpy(und_sub, nodelist=ordered_nodes)
    dist = np.asarray(dist, dtype=np.float32)

    prox = np.zeros_like(dist, dtype=np.float32)
    mask = dist > 0
    prox[mask] = 1.0 / dist[mask] - 1.0
    np.fill_diagonal(prox, 1.0)
    return prox


def compute_orthogonality_metrics(
    hgraph: nx.DiGraph,
    sorted_nodes: List[int],
    dirs_original: Dict[int, Dict[str, torch.Tensor]],
    dirs_shuffled: Dict[int, Dict[str, torch.Tensor]],
    version: str,
    seed: int,
    show_progress: bool = False,
) -> Dict[str, Dict[str, List[float]]]:
    random.seed(seed)

    available = [n for n in sorted_nodes if n in dirs_original and n in dirs_shuffled]

    roots = [n for n in hgraph.nodes if hgraph.in_degree(n) == 0]
    depth = {n: 10**9 for n in hgraph.nodes}
    for r in roots:
        lens = nx.single_source_shortest_path_length(hgraph, r)
        for n, d in lens.items():
            if d < depth[n]:
                depth[n] = d
    depth = {n: d for n, d in depth.items() if d < 10**9 and n in available}

    nodes_by_depth = defaultdict(list)
    for n, d in depth.items():
        nodes_by_depth[d].append(n)

    metrics = {
        "b": {
            "original_parent": [],
            "original_random_parent": [],
            "shuffled_parent": [],
            "shuffled_random_parent": [],
        },
        "e": {
            "original_parent": [],
            "original_random_parent": [],
            "shuffled_parent": [],
            "shuffled_random_parent": [],
        },
    }

    nodes_iter = progress_iter(
        available,
        enabled=show_progress,
        desc="Computing orthogonality metrics",
        total=len(available),
    )

    for node in nodes_iter:
        preds = [p for p in hgraph.predecessors(node) if p in available]
        if not preds:
            continue
        parent = preds[0]

        for space_name, dmap in [("original", dirs_original), ("shuffled", dirs_shuffled)]:
            child_dir = dmap[node][version]
            parent_dir = dmap[parent][version]

            child_parent = safe_norm(child_dir - parent_dir)
            pnorm = safe_norm(parent_dir)
            metrics["b"][f"{space_name}_parent"].append(float((child_parent @ pnorm).cpu().item()))

            parent_depth = depth.get(parent, None)
            if parent_depth is not None:
                candidates = [c for c in nodes_by_depth[parent_depth] if c != parent]
            else:
                candidates = [c for c in available if c != parent]
            rand_parent = random.choice(candidates) if candidates else random.choice(available)
            rand_parent_dir = dmap[rand_parent][version]
            child_rand = safe_norm(child_dir - rand_parent_dir)
            rand_parent_norm = safe_norm(rand_parent_dir)
            metrics["b"][f"{space_name}_random_parent"].append(
                float((child_rand @ rand_parent_norm).cpu().item())
            )

        parent_preds = [gp for gp in hgraph.predecessors(parent) if gp in available]
        if not parent_preds:
            continue
        grandparent = parent_preds[0]

        for space_name, dmap in [("original", dirs_original), ("shuffled", dirs_shuffled)]:
            child_dir = dmap[node][version]
            parent_dir = dmap[parent][version]
            grandparent_dir = dmap[grandparent][version]

            child_parent = safe_norm(child_dir - parent_dir)
            parent_grandparent = safe_norm(parent_dir - grandparent_dir)
            metrics["e"][f"{space_name}_parent"].append(
                float((child_parent @ parent_grandparent).cpu().item())
            )

            parent_depth = depth.get(parent, None)
            grandparent_depth = depth.get(grandparent, None)
            if parent_depth is not None:
                p_candidates = [c for c in nodes_by_depth[parent_depth] if c != parent]
            else:
                p_candidates = [c for c in available if c != parent]
            if grandparent_depth is not None:
                gp_candidates = [c for c in nodes_by_depth[grandparent_depth] if c != grandparent]
            else:
                gp_candidates = [c for c in available if c != grandparent]

            rand_parent = random.choice(p_candidates) if p_candidates else random.choice(available)
            rand_grandparent = random.choice(gp_candidates) if gp_candidates else random.choice(available)
            rand_parent_dir = dmap[rand_parent][version]
            rand_grandparent_dir = dmap[rand_grandparent][version]
            child_rand = safe_norm(child_dir - rand_parent_dir)
            parent_grand_rand = safe_norm(rand_parent_dir - rand_grandparent_dir)
            metrics["e"][f"{space_name}_random_parent"].append(
                float((child_rand @ parent_grand_rand).cpu().item())
            )

    return metrics


def save_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        import json

        json.dump(obj, f, indent=2)


def write_text_log(path: Path, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def compute_projection_feature_stats(
    g_whitened: torch.Tensor,
    g_shuffled: torch.Tensor,
    ent2idx: Dict[str, int],
    node_members: Dict[int, List[int]],
    feature_nodes: List[int],
    seed: int,
    train_ratio: float = 0.8,
    random_sample_size: int = 20000,
) -> Dict[str, dict]:
    rng = random.Random(seed)

    out = {
        "nodes": [],
        "original": {"train_mean": [], "train_std": [], "test_mean": [], "test_std": [], "random_mean": [], "random_std": []},
        "shuffled": {"train_mean": [], "train_std": [], "test_mean": [], "test_std": [], "random_mean": [], "random_std": []},
    }

    n_total = g_whitened.shape[0]
    rand_k = min(random_sample_size, n_total)

    for node in feature_nodes:
        members = node_members.get(node, [])
        mapped = [ent2idx[str(c)] for c in members if str(c) in ent2idx]
        mapped = list(dict.fromkeys(mapped))
        if len(mapped) < 8:
            continue

        rng.shuffle(mapped)
        split = max(3, int(len(mapped) * train_ratio))
        if split >= len(mapped):
            split = len(mapped) - 2

        train_idx = mapped[:split]
        test_idx = mapped[split:]
        if len(test_idx) < 2:
            continue

        # Estimate category direction from train subset, as in paper Figure 3.
        train_emb_orig = g_whitened[train_idx]
        train_emb_shuf = g_shuffled[train_idx]
        lda_orig, _ = estimate_single_dir_from_embeddings(train_emb_orig)
        lda_shuf, _ = estimate_single_dir_from_embeddings(train_emb_shuf)

        dir_orig = lda_orig / (lda_orig.norm() ** 2 + 1e-12)
        dir_shuf = lda_shuf / (lda_shuf.norm() ** 2 + 1e-12)

        rand_idx = rng.sample(range(n_total), rand_k)

        vals_orig_train = (g_whitened[train_idx] @ dir_orig).cpu().numpy()
        vals_orig_test = (g_whitened[test_idx] @ dir_orig).cpu().numpy()
        vals_orig_rand = (g_whitened[rand_idx] @ dir_orig).cpu().numpy()

        vals_shuf_train = (g_shuffled[train_idx] @ dir_shuf).cpu().numpy()
        vals_shuf_test = (g_shuffled[test_idx] @ dir_shuf).cpu().numpy()
        vals_shuf_rand = (g_shuffled[rand_idx] @ dir_shuf).cpu().numpy()

        out["nodes"].append(int(node))

        out["original"]["train_mean"].append(float(np.mean(vals_orig_train)))
        out["original"]["train_std"].append(float(np.std(vals_orig_train)))
        out["original"]["test_mean"].append(float(np.mean(vals_orig_test)))
        out["original"]["test_std"].append(float(np.std(vals_orig_test)))
        out["original"]["random_mean"].append(float(np.mean(vals_orig_rand)))
        out["original"]["random_std"].append(float(np.std(vals_orig_rand)))

        out["shuffled"]["train_mean"].append(float(np.mean(vals_shuf_train)))
        out["shuffled"]["train_std"].append(float(np.std(vals_shuf_train)))
        out["shuffled"]["test_mean"].append(float(np.mean(vals_shuf_test)))
        out["shuffled"]["test_std"].append(float(np.std(vals_shuf_test)))
        out["shuffled"]["random_mean"].append(float(np.mean(vals_shuf_rand)))
        out["shuffled"]["random_std"].append(float(np.std(vals_shuf_rand)))

    return out


