## Geometry visualizations

This folder contains intrinsic evaluations and diagnostic plots for the concept embeddings learned by RotE. The plots are created by [run.py](run.py) and saved under the output directory configured in the script.

### 1. `heatmap_hierarchy_lda.png`

This figure compares three matrices over the same set of hierarchy nodes: the shortest-path proximity in the hierarchy, the cosine similarity of the learned category directions on the original embeddings, and the cosine similarity of the same directions on shuffled embeddings.

The hierarchy matrix is computed from graph distance on the concept graph and converted into a proximity score. The cosine matrices are built by estimating one direction per node and then taking pairwise dot products between the normalized directions. The original embeddings should show stronger block structure than the shuffled baseline if the hierarchy is encoded geometrically.

### 2. `hier_orthogonality_b.png`

This plot measures the angle between a child-to-parent difference vector and the parent direction for every valid hierarchy edge. For each edge, the code computes `cos(l_w - l_parent, l_parent)` for the original embeddings, for a random-parent baseline, and for shuffled embeddings.

Values near zero indicate that the child-parent displacement is close to orthogonal to the parent direction. The random-parent and shuffled curves are controls: they should not preserve the same regularity as the original curve if the hierarchy is genuinely reflected in the embedding space.

### 3. `hier_orthogonality_e.png`

This plot measures a second-order consistency relation along two consecutive hierarchy edges. For each child-parent-grandparent chain, the code computes `cos(l_w - l_parent, l_parent - l_grandparent)` and compares it against random-parent and shuffled baselines.

This checks whether the direction from grandparent to parent is aligned with the direction from parent to child. Again, the random and shuffled curves are the controls.

### 4. `feature_projection.png`

This figure reproduces the feature-projection experiment for binary hierarchy features. For each feature node, the members are split into train and test subsets, an LDA direction is fitted on the train subset, and the embeddings are projected onto that direction.

The plot shows the mean and standard deviation of the projections for train, test, and random samples, for both the original and shuffled embeddings. The y-axis is the normalized projection `g(y)^T \bar{\ell}_w / ||\bar{\ell}_w||^2`. If a feature is well represented, the test projections should stay close to the train projections, while the random baseline should be near zero.

### 5. `three_2d_plots_rotre_hierarchy.png`

This figure contains three 2D geometric views of selected animal/plant subtrees. It projects the whitened embeddings into low-dimensional subspaces spanned by learned LDA directions and shows both the concept directions and the descendant embeddings.

The panels visualize: `animal vs mammal`, `animal vs mammal -> bird`, and `plant -> animal vs mammal -> bird`. The purpose is to make the local hierarchy geometry interpretable, not to compute a new metric.

### 6. `two_3d_plots_rotre_hierarchy.png`

This figure shows two 3D views of the same hierarchy region. The first panel uses the directions for `mammal`, `bird`, and `fish` with `animal` as the higher-level direction; the second panel adds `reptile` and visualizes the simplex-like relation among the four directions.

The 3D views are a geometric illustration of how the learned concept directions are arranged relative to one another and how the corresponding embedding clouds sit around those directions.

### Output summary

The script also saves a JSON summary with the averaged metric values used in the figures. That file is intended for quick comparison across runs without reopening the plots.
