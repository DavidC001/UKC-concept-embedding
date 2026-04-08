"""Build and load hierarchy cache for YOLO9000-style probabilities."""
import torch
from WSD.concepts import build_parent_index_from_is_a, create_bidirectional_mappings


def build_hierarchy_group_tensors(parent_index):
    """Precompute group/path tensors used by factorized hierarchy probabilities."""
    parent_index = parent_index.long().cpu()
    num_concepts = int(parent_index.shape[0])

    roots = (parent_index < 0).nonzero(as_tuple=False).flatten().tolist()
    if not roots:
        roots = list(range(num_concepts))

    children_by_parent = [[] for _ in range(num_concepts)]
    for child_idx in range(num_concepts):
        p = int(parent_index[child_idx].item())
        if 0 <= p < num_concepts and p != child_idx:
            children_by_parent[p].append(child_idx)

    child_group_offsets = [0]
    flat_children = []
    child_pos_in_parent = torch.full((num_concepts,), -1, dtype=torch.long)
    softmax_parent_groups = []

    for parent_idx in range(num_concepts):
        children = children_by_parent[parent_idx]
        if children:
            softmax_parent_groups.append(torch.tensor(children, dtype=torch.long))
        for local_pos, child_idx in enumerate(children):
            flat_children.append(child_idx)
            child_pos_in_parent[child_idx] = local_pos
        child_group_offsets.append(len(flat_children))

    root_to_pos = {root_idx: pos for pos, root_idx in enumerate(roots)}
    all_paths = []
    root_pos_by_concept = torch.full((num_concepts,), 0, dtype=torch.long)
    edge_counts = torch.zeros((num_concepts,), dtype=torch.long)

    for concept_idx in range(num_concepts):
        cur = concept_idx
        chain = [cur]
        seen = {cur}

        while True:
            parent = int(parent_index[cur].item())
            if parent < 0 or parent >= num_concepts or parent == cur:
                break
            if parent in seen:
                break
            chain.append(parent)
            seen.add(parent)
            cur = parent

        chain = list(reversed(chain))
        root = chain[0]
        if root not in root_to_pos:
            root_to_pos[root] = len(root_to_pos)
            roots.append(root)
        root_pos_by_concept[concept_idx] = root_to_pos[root]

        edges = []
        for i in range(len(chain) - 1):
            parent_idx = chain[i]
            child_idx = chain[i + 1]
            edges.append((parent_idx, child_idx))

        edge_counts[concept_idx] = len(edges)
        all_paths.append(edges)

    max_edges = int(edge_counts.max().item()) if num_concepts > 0 else 0
    if max_edges == 0:
        max_edges = 1

    path_parents = torch.full((num_concepts, max_edges), -1, dtype=torch.long)
    path_children = torch.full((num_concepts, max_edges), -1, dtype=torch.long)

    for concept_idx, edges in enumerate(all_paths):
        for step_idx, (parent_idx, child_idx) in enumerate(edges):
            path_parents[concept_idx, step_idx] = parent_idx
            path_children[concept_idx, step_idx] = child_idx

    return {
        "root_indices": torch.tensor(roots, dtype=torch.long),
        "child_group_offsets": torch.tensor(child_group_offsets, dtype=torch.long),
        "child_group_children": torch.tensor(flat_children, dtype=torch.long),
        "child_pos_in_parent": child_pos_in_parent,
        "path_root_pos": root_pos_by_concept,
        "path_edge_counts": edge_counts,
        "path_parents": path_parents,
        "path_children": path_children,
        "softmax_parent_groups": softmax_parent_groups,
    }


def build_hierarchy(entity_to_id_csv, concept_rel_csv):

    concept_id_to_index, _ = create_bidirectional_mappings(entity_to_id_csv)
    num_concepts = len(concept_id_to_index)

    parent_index, num_roots, num_multi_parent = build_parent_index_from_is_a(
        concept_id_to_index=concept_id_to_index,
        num_concepts=num_concepts,
        csv_path=concept_rel_csv,
    )
    hierarchy_tensors = build_hierarchy_group_tensors(parent_index)

    hierarchy_data = {
        "parent_index": parent_index,
        "num_roots": num_roots,
        "num_multi_parent": num_multi_parent,
        "concept_id_to_index": concept_id_to_index,
        "num_concepts": num_concepts,
        **hierarchy_tensors,
    }

    return hierarchy_data

