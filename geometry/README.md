# Geometry Analysis for RoTE (UKC) in the Style of arXiv:2406.01506v3

This folder implements a paper-inspired geometry analysis pipeline that adapts
the methodology of:

The Geometry of Categorical and Hierarchical Concepts in Large Language Models
(arXiv:2406.01506v3)

to the RoTE/UKC concept embedding space used in this repository.

Main entrypoint: geometry/run.py

## 1) What this reproduces from the paper

The implementation follows the same high-level logic as Sections 5.1-5.2 and
Appendix A/F of the paper:

1. Build a canonical space by centering and whitening embeddings.
2. Estimate one vector per hierarchy concept (LDA estimator, plus mean vector).
3. Compare original vs shuffled controls.
4. Evaluate hierarchy geometry with:
   - cosine similarity matrices between estimated concept vectors,
   - orthogonality diagnostics for parent/child and parent/child/grandparent,
   - train/test/random projection diagnostics (Figure-3 style),
   - 2D and 3D visualizations for animal/plant subhierarchies.

Core formulas mirrored from the paper:

- Figure-3 projection diagnostic:
  (g(y)^T l_w) / ||l_w||^2
- Theorem-8 style orthogonality checks:
  cos(l_w - l_parent, l_parent)
  cos(l_w - l_parent, l_parent - l_grandparent)

## 2) Important adaptations and differences

This code is not a byte-for-byte reproduction of the original Gemma/LLaMA
WordNet experiments. It is a structurally faithful adaptation to this project.

### Data/model domain

- Paper: token unembeddings from Gemma/LLaMA and WordNet synset word sets.
- Here: RoTE entity embeddings and UKC concept hierarchy from CSV files.

### Feature construction

- Paper: vocabulary sets Y(w) from WordNet words/inflections.
- Here: descendants in the concept graph define each concept member set.

### Minimum concept size

- Paper noun setup mentions minimum 50 words/synset.
- Here default is min_category_size=25.

### Train/test split ratio

- Paper Figure-3 split: 70/30.
- Here default feature_train_ratio=0.8 (80/20), configurable.

### Shortest-path heatmap normalization

- Paper text describes proximity as (1 + distance)^(-1).
- Current code uses 1/distance - 1 (with diagonal set to 1), which changes scale
  and sign interpretation for non-neighbors.

### Parent selection in DAG

- As in the paper appendix note, when a node has multiple parents, one parent is
  selected for edge-wise metrics.
- Current code uses the first predecessor returned by networkx iteration.

## 3) Code structure

- run.py
  - Pipeline orchestration, argument parsing, and output saving.
- load_rotre_embeddings.py
  - Loads checkpoint embeddings and applies centering + whitening.
- hierarchy.py
  - Reads concept tables, builds directed hierarchy, and derives descendant sets.
- category.py
  - LDA-style vector estimator using Ledoit-Wolf covariance shrinkage.
- metrics.py
  - Direction estimation, cosine/proximity matrices, orthogonality metrics,
    Figure-3 style projection statistics, JSON/TXT summaries.
- visualizations.py
  - Heatmaps, orthogonality curves, Figure-3 style plot, 2D/3D geometry plots,
    and subtree graphs.
- plotting.py
  - Low-level projection plotting utilities used by visualizations.py.
- common.py
  - Utility helpers (safe normalization, optional progress bars).

## 4) Input files and assumptions

Expected defaults:

- checkpoint: dataset/RotE/model.pt
- entity_to_id: dataset/RotE/entity_to_id.pickle
- concepts table: dataset/concepts.csv
- hierarchy edges: dataset/concept_relations.csv
- hierarchy relation_type: 20

Assumptions:

- relation_type=20 corresponds to parent -> child (has_hyponym style edge).
- Concept IDs present in hierarchy are mapped in entity_to_id for projection.
- animal and plant roots default to concept IDs 37 and 38.

## 5) Output files and how to interpret them

All outputs go to --output_dir (default geometry/figures).

### heatmap_hierarchy_lda.png

- Left: hierarchy proximity matrix.
- Middle: cosine similarity between LDA vectors in original space.
- Right: cosine similarity after shuffled control.

Paper connection:
- Corresponds to the Figure-4 style analysis of hierarchy distance vs
  representation geometry.

### hier_orthogonality_b.png

- Curves for cos(l_w - l_parent, l_parent):
  - Original
  - Original + random parent baseline
  - Shuffled baseline

Paper connection:
- Corresponds to Figure-5 style statement-(a) Theorem-8 check.

### hier_orthogonality_e.png

- Curves for cos(l_w - l_parent, l_parent - l_grandparent):
  - Original
  - Original + random parent/grandparent baseline
  - Shuffled baseline

Paper connection:
- Corresponds to Appendix-F Figure-10 style statement-(d) Theorem-8 check.

### feature_projection_figure3_style.png

- Two panels: Original Unembeddings vs Shuffled Unembeddings.
- For each feature, plots train/test/random projection mean with error bars.

Paper connection:
- Directly mirrors Figure 3 diagnostic logic.

Expected pattern:
- Original: train/test near 1, random near 0.
- Shuffled: no strong separation.

### three_2d_plots_rotre_hierarchy.png

- 2D projections for:
  - animal vs mammal
  - animal vs (bird - mammal)
  - (animal - plant) vs (bird - mammal)

Paper connection:
- Mirrors Figure 2 style geometric demonstrations of hierarchical orthogonality.

### two_3d_plots_rotre_hierarchy.png

- Left panel:
  - span{mammal, bird, fish} with animal vector shown relative to polytope plane.
- Right panel:
  - simplex/tetrahedron style view for mammal, bird, fish, reptile contrasts.

Paper connection:
- Mirrors Figure 6 and Appendix-A style categorical polytope geometry.

### animal_plant_subtrees.png

- Side-by-side graph views of local hierarchy under animal and plant roots.

### metrics_summary.json

- Compact machine-readable summary:
  - node counts,
  - selected concept IDs,
  - aggregate means of orthogonality and projection diagnostics.

### results_log.txt

- Human-readable run summary with the same key statistics.

## 6) CLI usage

Basic run:

python geometry/run.py \
  --checkpoint dataset/RotE/model.pt \
  --entity_to_id dataset/RotE/entity_to_id.pickle \
  --output_dir geometry/figures

Disable progress bars:

python geometry/run.py --no_progress

Useful knobs:

- --min_category_size 25
- --feature_train_ratio 0.8
- --feature_random_sample_size 20000
- --animal_root_id 37
- --plant_root_id 38
- --subtree_depth 3
- --seed 100

## 7) Practical reading guide

If you want a quick sanity pass that the geometry matches paper expectations:

1. Check feature_projection_figure3_style.png first.
2. Then inspect hier_orthogonality_b.png and hier_orthogonality_e.png.
3. Use heatmap_hierarchy_lda.png for global structure.
4. Use 2D/3D plots for interpretable local examples.

## 8) Summary

This geometry module is best viewed as a RoTE/UKC adaptation of the paper's
experimental geometry protocol. The conceptual structure and diagnostics are
aligned with the paper, while dataset/model-specific details differ by design.
