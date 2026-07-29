"""Entry point for hierarchy geometry analysis on RoTE embeddings."""

import argparse
import os
from pathlib import Path

import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm

from hierarchy import (
    build_hierarchy_graph,
    build_node_sets_for_directions,
    descendant_indices,
    find_child_by_label,
    graph_roots,
    load_concepts,
)
from load_rotre_embeddings import (
    build_vocab_list,
    load_embeddings,
    load_entity_to_index,
    whiten_embeddings,
)
from metrics import (
    compute_orthogonality_metrics,
    compute_projection_feature_stats,
    cosine_matrix_from_dirs,
    estimate_dirs,
    save_json,
    shortest_path_matrix,
)
from visualizations import (
    plot_heatmaps,
    plot_orthogonality_curves,
    plot_projection_feature_figure,
    run_visual_2d,
    run_visual_3d,
)


sns.set_theme(context="paper", style="white", palette="colorblind", font="DejaVu Sans")


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = Path(args.output_dir)
    experiments = {
        experiment.strip().lower()
        for experiment in args.experiments.split(",")
        if experiment.strip()
    }
    run_all = "all" in experiments
    plot_tag = "_inverse" if args.inverse_relation else ""

    ent2idx = load_entity_to_index(args.entity_to_id)
    emb = load_embeddings(args.checkpoint, map_location="cpu")
    g_whitened, _, _ = whiten_embeddings(emb)

    # compute permutated version to have a baseline where no correlation structure is preserved, but the distribution of values is the same
    perm = torch.randperm(g_whitened.shape[0], generator=torch.Generator().manual_seed(args.seed))
    g_shuffled = g_whitened[perm]

    concepts_df = load_concepts(args.concepts_csv)
    hgraph = build_hierarchy_graph(args.concept_relations_csv, relation_type=args.hierarchy_relation_type, inverse=args.inverse_relation)
    roots = graph_roots(hgraph)
    
    # breakpoint()
    
    # print number of roots
    print(f"Found {len(roots)} roots in the hierarchy graph.")
    print("Root IDs and labels:")
    for r in roots:
        label = concepts_df.loc[concepts_df.id == r, "label"].values
        label_str = label[0] if len(label) > 0 else "Unknown"
        print(f" - ID: {r}, Label: {label_str}")

    # breakpoint()

    # take subtrees of the animal and plant roots for focused analysis and visualization
    animal_root =  args.animal_root_id
    plant_root = args.plant_root_id


    node_members = {}
    roots_iter = tqdm(
        roots,
        desc="Building hierarchy node sets",
        total=len(roots),
    )
    for r in roots_iter:
        node_members.update(build_node_sets_for_directions(hgraph, r, min_size=args.min_category_size))

    dirs_original = estimate_dirs(g_whitened, ent2idx, node_members)
    dirs_shuffled = estimate_dirs(g_shuffled, ent2idx, node_members)

    kept_nodes = sorted([n for n in node_members.keys() if n in dirs_original and n in dirs_shuffled])

    """
    Compute the heatmap of cosine similarities between category directions, and compare it to the proximity in the hierarchy (shortest path distance).
    """
    if run_all or "heatmap" in experiments:
        cos_lda_original = cosine_matrix_from_dirs(kept_nodes, dirs_original, version="lda")
        cos_lda_shuffled = cosine_matrix_from_dirs(kept_nodes, dirs_shuffled, version="lda")
        dist_prox = shortest_path_matrix(hgraph, kept_nodes)

        plot_heatmaps(
            out_dir / f"heatmap_hierarchy_lda{plot_tag}.png",
            dist_prox,
            cos_lda_original,
            cos_lda_shuffled,
            title_prefix="Hierarchy",
        )


    """
    Compute orthogonality metrics for category directions, comparing original vs shuffled embeddings, and parent vs random parent.
    """
    if run_all or "orthogonality" in experiments:
        metrics = compute_orthogonality_metrics(
            hgraph=hgraph,
            sorted_nodes=kept_nodes,
            dirs_original=dirs_original,
            dirs_shuffled=dirs_shuffled,
            version="lda",
            seed=args.seed,
        )

        plot_orthogonality_curves(
            out_dir / f"hier_orthogonality_b{plot_tag}.png",
            metrics,
            key="b",
            title=r"cos(l_w - l_parent, l_parent)",
        )
        plot_orthogonality_curves(
            out_dir / f"hier_orthogonality_e{plot_tag}.png",
            metrics,
            key="e",
            title=r"cos(l_w - l_parent, l_parent - l_grandparent)",
        )


    """
    Compute and plot metrics on binary features
    """
    if run_all or "feature_projection" in experiments:
        proj_stats = compute_projection_feature_stats(
            g_whitened=g_whitened,
            g_shuffled=g_shuffled,
            ent2idx=ent2idx,
            node_members=node_members,
            feature_nodes=kept_nodes,
            seed=args.seed,
            train_ratio=args.feature_train_ratio,
            random_sample_size=args.feature_random_sample_size,
        )
        plot_projection_feature_figure(out_dir / f"feature_projection{plot_tag}.png", proj_stats)


    """
    Plot 2D and 3D visualizations of the animal and plant subtrees, highlighting the positions of the category directions for the main categories 
    (animal, plant, mammal, bird, fish, reptile).
    """
    if run_all or "visualization" in experiments:
        print("Running 2D and 3D visualizations of the animal and plant subtrees...")
        id_to_label = {str(int(r.id)): str(r.label).lower() for r in concepts_df.itertuples(index=False)}
        vocab_list = build_vocab_list(emb.shape[0], ent2idx, id_to_label=id_to_label)

        mammals_id = find_child_by_label(hgraph, concepts_df, animal_root, "mammal")
        birds_id = find_child_by_label(hgraph, concepts_df, animal_root, "bird")
        fish_id = find_child_by_label(hgraph, concepts_df, animal_root, "fish")
        reptile_id = find_child_by_label(hgraph, concepts_df, animal_root, "reptile")

        ids = {
            "animal": animal_root,
            "plant": plant_root,
            "mammal": mammals_id,
            "bird": birds_id,
            "fish": fish_id,
            "reptile": reptile_id,
        }

        missing = [k for k, v in ids.items() if v not in dirs_original]
        if missing:
            raise RuntimeError(
                "Missing directional estimates for: " + ", ".join(missing) +
                ". Try lowering --min_category_size."
            )

        idx_sets = {
            "animal": descendant_indices(animal_root, hgraph, ent2idx, max_depth=5),
            "plant": descendant_indices(plant_root, hgraph, ent2idx, max_depth=5),
            "mammal": descendant_indices(mammals_id, hgraph, ent2idx, max_depth=5),
            "bird": descendant_indices(birds_id, hgraph, ent2idx, max_depth=5),
            "fish": descendant_indices(fish_id, hgraph, ent2idx, max_depth=5),
            "reptile": descendant_indices(reptile_id, hgraph, ent2idx, max_depth=5),
        }

        run_visual_2d(
            out_dir / f"three_2d_plots_rotre_hierarchy{plot_tag}.png",
            g_whitened,
            vocab_list,
            idx_sets,
            dirs_original,
            ids,
        )

        run_visual_3d(
            out_dir / f"two_3d_plots_rotre_hierarchy{plot_tag}.png",
            g_whitened,
            idx_sets,
            dirs_original,
            ids,
        )


    """
    Save a summary of the main metrics in a JSON file, for easier experiment tracking and comparison with future runs. Also save a text log with the main results and stats.
    """
    metrics_summary = {
        "num_total_nodes_with_members": len(node_members),
        "num_nodes_with_dirs_original": len(dirs_original),
        "num_nodes_used_for_metrics": len(kept_nodes),
        "animal_root": int(animal_root),
        "plant_root": int(plant_root),
        "inverse_relation": bool(args.inverse_relation),
    }
    if run_all or "visualization" in experiments:
        metrics_summary.update({
            "mammal_node": int(mammals_id),
            "bird_node": int(birds_id),
            "fish_node": int(fish_id),
            "reptile_node": int(reptile_id),
        })
    if run_all or "orthogonality" in experiments:
        metrics_summary.update({
            "orthogonality": {
                "b": {
                    "original_parent": float(np.mean(metrics["b"]["original_parent"])) if metrics["b"]["original_parent"] else None,
                    "original_random_parent": float(np.mean(metrics["b"]["original_random_parent"])) if metrics["b"]["original_random_parent"] else None,
                    "shuffled_parent": float(np.mean(metrics["b"]["shuffled_parent"])) if metrics["b"]["shuffled_parent"] else None,
                },
                "e": {
                    "original_parent": float(np.mean(metrics["e"]["original_parent"])) if metrics["e"]["original_parent"] else None,
                    "original_random_parent": float(np.mean(metrics["e"]["original_random_parent"])) if metrics["e"]["original_random_parent"] else None,
                    "shuffled_parent": float(np.mean(metrics["e"]["shuffled_parent"])) if metrics["e"]["shuffled_parent"] else None,
                },
            }
        })
    if run_all or "feature_projection" in experiments:
        metrics_summary.update({
            "feature_projection": {
                "num_features": len(proj_stats.get("nodes", [])),
                "original_train_mean_global": float(np.mean(proj_stats["original"]["train_mean"])) if proj_stats["original"]["train_mean"] else None,
                "original_test_mean_global": float(np.mean(proj_stats["original"]["test_mean"])) if proj_stats["original"]["test_mean"] else None,
                "original_random_mean_global": float(np.mean(proj_stats["original"]["random_mean"])) if proj_stats["original"]["random_mean"] else None,
                "shuffled_train_mean_global": float(np.mean(proj_stats["shuffled"]["train_mean"])) if proj_stats["shuffled"]["train_mean"] else None,
                "shuffled_test_mean_global": float(np.mean(proj_stats["shuffled"]["test_mean"])) if proj_stats["shuffled"]["test_mean"] else None,
                "shuffled_random_mean_global": float(np.mean(proj_stats["shuffled"]["random_mean"])) if proj_stats["shuffled"]["random_mean"] else None,
            },
        })
        
    save_json(out_dir / "metrics_summary.json", metrics_summary)

    print("Saved outputs to:", out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="dataset/RotE/model.pt")
    parser.add_argument("--entity_to_id", type=str, default="dataset/RotE/entity_to_id.pickle")
    
    parser.add_argument("--concepts_csv", type=str, default="dataset/concepts.csv")
    parser.add_argument("--concept_relations_csv", type=str, default="dataset/concept_relations.csv")
    
    parser.add_argument("--hierarchy_relation_type", type=int, default=20)
    parser.add_argument("--inverse_relation", action="store_true", help="If set, will invert the hierarchy relation type for analysis.")
    
    parser.add_argument("--animal_root_id", type=int, default=37)
    parser.add_argument("--plant_root_id", type=int, default=38)
    
    parser.add_argument("--min_category_size", type=int, default=25)
    parser.add_argument("--feature_train_ratio", type=float, default=0.7)
    parser.add_argument("--feature_random_sample_size", type=int, default=20000)
    
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="geometry/outputs")
    
    # what experiments to run, for example, "all" or "heatmap,orthogonality,feature_projection,visualization"
    parser.add_argument("--experiments", type=str, default="all")
    
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
