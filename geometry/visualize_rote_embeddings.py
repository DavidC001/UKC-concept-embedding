"""Create a 2D visualization from RoTE entity embeddings.

This script loads a checkpoint produced by the ukc_embedding training code,
extracts entity embeddings, applies dimensionality reduction, and saves a 2D plot.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import os
import pickle
import re
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load checkpoint and return a state dict."""
    checkpoint = torch.load(path, map_location=map_location)

    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format: expected a dict-like object")

    if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
        return checkpoint["model_state_dict"]

    return checkpoint


def extract_entity_embeddings(state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
    """Extract entity embeddings from known key patterns."""
    candidate_keys = (
        "entity.weight",      # ukc_embedding Euclidean/Hyperbolic models (including RotE)
        "embeddings.0.weight" # complex-valued model family
    )

    for key in candidate_keys:
        if key in state_dict:
            tensor = state_dict[key]
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Found key '{key}' but value is not a torch.Tensor")
            return tensor.detach().cpu().numpy()

    available = ", ".join(list(state_dict.keys())[:20])
    raise KeyError(
        "Unable to find entity embeddings in checkpoint. "
        f"Tried keys {candidate_keys}. Sample available keys: {available}"
    )


def load_entity_labels(entity_map_path: str, n_entities: int, concepts_csv_path: str = "") -> np.ndarray:
    """Load labels from concepts.csv (preferred) and entity map (fallback)."""
    labels = np.array([f"entity_{i}" for i in range(n_entities)], dtype=object)
    id_by_index: Dict[int, str] = {}

    if entity_map_path:
        with open(entity_map_path, "rb") as f:
            entity_to_id = pickle.load(f)

        if not isinstance(entity_to_id, dict):
            raise ValueError("entity map must be a dict(name -> id)")

        for name, idx in entity_to_id.items():
            if isinstance(idx, (int, np.integer)) and 0 <= idx < n_entities:
                name_str = str(name)
                id_by_index[int(idx)] = name_str
                labels[idx] = name_str

    if concepts_csv_path:
        if not os.path.exists(concepts_csv_path):
            print(f"Warning: concepts CSV not found at '{concepts_csv_path}'. Using ID labels.")
            return labels

        concepts_df = pd.read_csv(concepts_csv_path, usecols=["id", "label"])
        concept_label_by_id = {
            str(concept_id): str(label)
            for concept_id, label in zip(concepts_df["id"], concepts_df["label"])
        }

        replaced = 0
        for idx, concept_id in id_by_index.items():
            concept_label = concept_label_by_id.get(concept_id)
            if concept_label:
                labels[idx] = concept_label
                replaced += 1

        print(f"Loaded concept labels from {concepts_csv_path}: mapped {replaced}/{n_entities} entities")

    return labels


def load_index_to_concept_id(entity_map_path: str, n_entities: int) -> Dict[int, int]:
    """Load embedding index -> concept_id mapping from entity_to_id.pickle."""
    if not entity_map_path:
        raise ValueError("Hierarchy subtree filtering requires --entity-map")

    with open(entity_map_path, "rb") as f:
        entity_to_id = pickle.load(f)

    if not isinstance(entity_to_id, dict):
        raise ValueError("entity map must be a dict(name -> id)")

    index_to_concept_id: Dict[int, int] = {}
    for concept_id_raw, idx in entity_to_id.items():
        if not isinstance(idx, (int, np.integer)):
            continue
        if idx < 0 or idx >= n_entities:
            continue
        try:
            concept_id = int(concept_id_raw)
        except (TypeError, ValueError):
            continue
        index_to_concept_id[int(idx)] = concept_id

    return index_to_concept_id


def resolve_root_concept_id(subtree_root: str, concepts_csv_path: str) -> int:
    """Resolve subtree root string to a concept id (numeric id or label)."""
    if not subtree_root:
        raise ValueError("subtree_root cannot be empty")

    try:
        return int(subtree_root)
    except ValueError:
        pass

    if not concepts_csv_path or not os.path.exists(concepts_csv_path):
        raise ValueError(
            "Non-numeric subtree root requires a valid --concepts-csv path to resolve label -> id"
        )

    concepts_df = pd.read_csv(concepts_csv_path, usecols=["id", "label"])
    labels = concepts_df["label"].astype(str).str.lower()
    matches = concepts_df.loc[labels == subtree_root.lower(), "id"].tolist()

    if not matches:
        raise ValueError(f"No concept id found for label '{subtree_root}'")
    if len(matches) > 1:
        raise ValueError(
            f"Label '{subtree_root}' is ambiguous ({len(matches)} matches). "
            "Use a numeric concept id for --subtree-root."
        )

    return int(matches[0])


def descendants_from_is_a(
    concept_relations_csv_path: str,
    root_concept_id: int,
    relation_type: int = 20,
    max_depth: int = -1,
) -> set[int]:
    """Return root + descendants following relation_type edges parent->child.

    max_depth semantics:
    - -1: no depth limit
    - 0: only root
    - d>0: include nodes up to d edges from root
    """
    if not os.path.exists(concept_relations_csv_path):
        raise FileNotFoundError(f"concept relations CSV not found: {concept_relations_csv_path}")

    rel_df = pd.read_csv(
        concept_relations_csv_path,
        usecols=["relation_type", "src_con_id", "trg_con_id"],
    )
    rel_type_col = pd.to_numeric(rel_df["relation_type"], errors="coerce")
    src_col = pd.to_numeric(rel_df["src_con_id"], errors="coerce")
    trg_col = pd.to_numeric(rel_df["trg_con_id"], errors="coerce")
    valid = rel_type_col.notna() & src_col.notna() & trg_col.notna() & (rel_type_col == relation_type)

    children_by_parent: Dict[int, list[int]] = defaultdict(list)
    for parent_raw, child_raw in zip(src_col[valid], trg_col[valid]):
        parent = int(parent_raw)
        child = int(child_raw)
        if parent != child:
            children_by_parent[parent].append(child)

    visited: set[int] = set()
    queue: deque[tuple[int, int]] = deque([(root_concept_id, 0)])
    while queue:
        node, depth = queue.popleft()
        if node in visited:
            continue
        visited.add(node)

        if max_depth >= 0 and depth >= max_depth:
            continue

        for child in children_by_parent.get(node, []):
            if child not in visited:
                queue.append((child, depth + 1))

    return visited


def load_is_a_edges(
    concept_relations_csv_path: str,
    relation_type: int = 20,
) -> list[tuple[int, int]]:
    """Load IS_A edges as (parent_concept_id, child_concept_id)."""
    rel_df = pd.read_csv(
        concept_relations_csv_path,
        usecols=["relation_type", "src_con_id", "trg_con_id"],
    )
    rel_type_col = pd.to_numeric(rel_df["relation_type"], errors="coerce")
    src_col = pd.to_numeric(rel_df["src_con_id"], errors="coerce")
    trg_col = pd.to_numeric(rel_df["trg_con_id"], errors="coerce")
    valid = rel_type_col.notna() & src_col.notna() & trg_col.notna() & (rel_type_col == relation_type)

    edges: list[tuple[int, int]] = []
    for parent_raw, child_raw in zip(src_col[valid], trg_col[valid]):
        parent = int(parent_raw)
        child = int(child_raw)
        if parent != child:
            edges.append((parent, child))
    return edges


def build_hierarchy_segments(
    coords: np.ndarray,
    original_indices: np.ndarray,
    index_to_concept_id: Dict[int, int],
    concept_relations_csv_path: str,
    relation_type: int,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Build line segments between parent and child points in plotted subset."""
    concept_to_row: Dict[int, int] = {}
    for row_idx, emb_idx in enumerate(original_indices):
        concept_id = index_to_concept_id.get(int(emb_idx))
        if concept_id is not None and concept_id not in concept_to_row:
            concept_to_row[concept_id] = row_idx

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for parent_id, child_id in load_is_a_edges(concept_relations_csv_path, relation_type=relation_type):
        parent_row = concept_to_row.get(parent_id)
        child_row = concept_to_row.get(child_id)
        if parent_row is None or child_row is None:
            continue
        segments.append(
            (
                (float(coords[parent_row, 0]), float(coords[parent_row, 1])),
                (float(coords[child_row, 0]), float(coords[child_row, 1])),
            )
        )
    return segments


def _safe_perplexity(n_samples: int, requested: float) -> float:
    max_valid = max(1.0, (n_samples - 1) / 3.0)
    return float(min(requested, max_valid))


def filter_subtree_hierarchy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    index_to_concept_id: Dict[int, int],
    subtree_root: str,
    concept_relations_csv_path: str,
    concepts_csv_path: str,
    relation_type: int,
    max_depth: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter embeddings/labels to descendants of subtree root in IS_A hierarchy."""
    original_indices = np.arange(embeddings.shape[0])

    if not subtree_root:
        return embeddings, labels, original_indices

    root_concept_id = resolve_root_concept_id(subtree_root, concepts_csv_path)
    subtree_ids = descendants_from_is_a(
        concept_relations_csv_path=concept_relations_csv_path,
        root_concept_id=root_concept_id,
        relation_type=relation_type,
        max_depth=max_depth,
    )

    mask = np.array(
        [index_to_concept_id.get(int(i), -1) in subtree_ids for i in original_indices],
        dtype=bool,
    )

    filtered_embeddings = embeddings[mask]
    filtered_labels = labels[mask]
    filtered_indices = original_indices[mask]
    return filtered_embeddings, filtered_labels, filtered_indices


def _slugify_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text)


def reduce_to_2d(embeddings: np.ndarray, method: str, random_state: int, perplexity: float) -> Tuple[np.ndarray, str]:
    """Reduce high-dimensional embeddings to two dimensions."""
    n_samples, n_features = embeddings.shape

    if n_samples < 3:
        raise ValueError("Need at least 3 points for meaningful 2D projection")

    if method == "auto":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(30, n_samples - 1),
                min_dist=0.1,
                metric="euclidean",
                random_state=random_state,
            )
            return reducer.fit_transform(embeddings), "umap"
        except Exception:
            method = "pca_tsne"

    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        return reducer.fit_transform(embeddings), "pca"

    if method == "tsne":
        tsne_perplexity = _safe_perplexity(n_samples, perplexity)
        reducer = TSNE(
            n_components=2,
            perplexity=tsne_perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
        return reducer.fit_transform(embeddings), "tsne"

    if method == "pca_tsne":
        pre_components = min(50, n_features, n_samples - 1)
        pca50 = PCA(n_components=pre_components, random_state=random_state)
        reduced = pca50.fit_transform(embeddings)

        tsne_perplexity = _safe_perplexity(n_samples, perplexity)
        tsne = TSNE(
            n_components=2,
            perplexity=tsne_perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
        return tsne.fit_transform(reduced), "pca_tsne"

    if method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as exc:
            raise ImportError("Method 'umap' requires package 'umap-learn'") from exc

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(30, n_samples - 1),
            min_dist=0.1,
            metric="euclidean",
            random_state=random_state,
        )
        return reducer.fit_transform(embeddings), "umap"

    raise ValueError(f"Unknown method: {method}")


def _annotation_indices(n_samples: int, max_labels: int) -> Iterable[int]:
    if max_labels <= 0:
        return []
    if n_samples <= max_labels:
        return range(n_samples)
    return np.linspace(0, n_samples - 1, max_labels, dtype=int)


def save_plot(
    coords: np.ndarray,
    labels: np.ndarray,
    plot_path: str,
    title: str,
    max_labels: int,
    hierarchy_segments: Optional[list[tuple[tuple[float, float], tuple[float, float]]]] = None,
    root_point: Optional[tuple[float, float]] = None,
) -> None:
    """Save a 2D scatter plot for reduced embeddings."""
    plt.figure(figsize=(11, 9))

    if hierarchy_segments:
        line_collection = LineCollection(
            hierarchy_segments,
            colors="#505050",
            linewidths=0.35,
            alpha=0.15,
            zorder=1,
        )
        plt.gca().add_collection(line_collection)

    plt.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.65, linewidths=0)

    if root_point is not None:
        plt.scatter(
            [root_point[0]],
            [root_point[1]],
            s=120,
            c="red",
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
            label="subtree root",
        )
        plt.legend(loc="best")

    for i in _annotation_indices(coords.shape[0], max_labels):
        plt.annotate(str(labels[i]), (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.8)

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize RoTE entity embeddings in 2D")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument(
        "--entity-map",
        default="",
        help="Path to entity_to_id.pickle (optional, for labels)",
    )
    parser.add_argument(
        "--concepts-csv",
        default="../../dataset/concepts.csv",
        help="Path to concepts.csv used to map concept IDs to human-readable labels",
    )
    parser.add_argument(
        "--method",
        default="pca_tsne",
        choices=["auto", "pca", "tsne", "pca_tsne", "umap"],
        help="Dimensionality reduction method",
    )
    parser.add_argument("--perplexity", default=30.0, type=float, help="t-SNE perplexity")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument(
        "--output-dir",
        default="geometry/rote_2d_visualization/output",
        help="Directory where outputs are stored",
    )
    parser.add_argument(
        "--max-labels",
        default=80,
        type=int,
        help="Maximum number of point labels to draw on the plot",
    )
    parser.add_argument(
        "--subtree-root",
        default="",
        help="Subtree root concept id (or exact concept label from concepts.csv)",
    )
    parser.add_argument(
        "--concept-relations-csv",
        default="../../dataset/concept_relations.csv",
        help="Path to concept_relations.csv used to build hierarchy edges",
    )
    parser.add_argument(
        "--subtree-relation-type",
        default=20,
        type=int,
        help="Relation type used as IS_A when building subtree (default: 20)",
    )
    parser.add_argument(
        "--subtree-max-depth",
        default=-1,
        type=int,
        help="Maximum depth from root in hierarchy (-1 means no limit)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)

    state_dict = load_checkpoint(args.checkpoint, map_location="cpu")
    embeddings = extract_entity_embeddings(state_dict)
    labels = load_entity_labels(args.entity_map, embeddings.shape[0], args.concepts_csv)
    index_to_concept_id: Optional[Dict[int, int]] = None
    root_concept_id: Optional[int] = None
    if args.subtree_root:
        index_to_concept_id = load_index_to_concept_id(args.entity_map, embeddings.shape[0])
        root_concept_id = resolve_root_concept_id(args.subtree_root, args.concepts_csv)

    if args.subtree_root:
        if index_to_concept_id is None:
            raise ValueError("Missing index_to_concept_id for subtree filtering")
        embeddings, labels, original_indices = filter_subtree_hierarchy(
            embeddings=embeddings,
            labels=labels,
            index_to_concept_id=index_to_concept_id,
            subtree_root=args.subtree_root,
            concept_relations_csv_path=args.concept_relations_csv,
            concepts_csv_path=args.concepts_csv,
            relation_type=args.subtree_relation_type,
            max_depth=args.subtree_max_depth,
        )
    else:
        original_indices = np.arange(embeddings.shape[0])

    if embeddings.shape[0] < 3:
        raise ValueError(
            "Subtree filter returned fewer than 3 entities. "
            "Use a broader subtree or disable subtree filtering."
        )

    print(f"Loaded {embeddings.shape[0]} entity embeddings with dimension {embeddings.shape[1]}")
    if args.subtree_root:
        print(
            f"Applied hierarchy subtree filter: root='{args.subtree_root}', "
            f"relation_type={args.subtree_relation_type}, max_depth={args.subtree_max_depth}"
        )
    coords_2d, used_method = reduce_to_2d(
        embeddings=embeddings,
        method=args.method,
        random_state=args.seed,
        perplexity=args.perplexity,
    )

    base_name = f"rote_2d_{used_method}"
    if args.subtree_root:
        base_name = f"{base_name}_subtree_{_slugify_filename(args.subtree_root)}"
    csv_path = os.path.join(args.output_dir, f"{base_name}.csv")
    plot_path = os.path.join(args.output_dir, f"{base_name}.png")

    out_df = pd.DataFrame(
        {
            "entity_index": original_indices,
            "label": labels,
            "x": coords_2d[:, 0],
            "y": coords_2d[:, 1],
        }
    )
    out_df.to_csv(csv_path, index=False)

    title = f"RoTE Entity Embeddings in 2D ({used_method})"
    if args.subtree_root:
        title = f"{title} | subtree={args.subtree_root}"

    hierarchy_segments = None
    root_point = None
    if args.subtree_root and index_to_concept_id is not None:
        hierarchy_segments = build_hierarchy_segments(
            coords=coords_2d,
            original_indices=original_indices,
            index_to_concept_id=index_to_concept_id,
            concept_relations_csv_path=args.concept_relations_csv,
            relation_type=args.subtree_relation_type,
        )

        if root_concept_id is not None:
            root_row = None
            for row_idx, emb_idx in enumerate(original_indices):
                if index_to_concept_id.get(int(emb_idx)) == root_concept_id:
                    root_row = row_idx
                    break
            if root_row is not None:
                root_point = (float(coords_2d[root_row, 0]), float(coords_2d[root_row, 1]))

        print(f"Hierarchy edges drawn: {len(hierarchy_segments)}")

    save_plot(
        coords_2d,
        labels,
        plot_path,
        title=title,
        max_labels=args.max_labels,
        hierarchy_segments=hierarchy_segments,
        root_point=root_point,
    )

    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Reduction method: {used_method}")
    print(f"Saved coordinates: {csv_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
