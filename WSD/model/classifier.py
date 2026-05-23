"""Classifier and hierarchy scoring components."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, output_dim=500, dropout=0.1, project_num_layers=1):
        super().__init__()

        layers = []
        current_dim = input_dim
        
        # Add multiple layers if project_num_layers > 1
        for i in range(project_num_layers-1):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
            
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UnifiedConceptClassifier(nn.Module):
    def __init__(
        self,
        concept_embeddings,
        input_dim,
        hidden_dim=1024,
        output_dim=500,
        dropout=0.1,
        temperature=0.1,
        encoder_backbone=None,
        parent_index=None,
        hierarchy_data=None,
        weighted_hierarchy_alpha=0.5,
        freeze_concept_embeddings=True,
        baseline=False,
        baseline_type="linear",
        similarity_metric="cosine",
        project_num_layers=1,
    ):
        super().__init__()
        self.encoder_backbone = encoder_backbone
        self.projector = Projector(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout, project_num_layers=project_num_layers)
        self.baseline = bool(baseline)
        self.baseline_type = str(baseline_type)
        self.similarity_metric = str(similarity_metric)
        self.weighted_hierarchy_alpha = float(weighted_hierarchy_alpha)

        if self.similarity_metric not in ("cosine", "dot_product", "l2_distance"):
            raise ValueError("similarity_metric must be one of: cosine, dot_product, l2_distance")

        num_concepts = int(concept_embeddings.shape[0])
        if self.baseline:
            if self.baseline_type == "linear":
                self.baseline_head = nn.Linear(output_dim, num_concepts, bias=False)
            elif self.baseline_type == "random":
                self.register_buffer("random_classifier", torch.randn(num_concepts, output_dim))
            else:
                raise ValueError("baseline_type must be one of: linear, random")
            self.concept_embeddings = None
        else:
            self.concept_embeddings = nn.Parameter(concept_embeddings, requires_grad=not freeze_concept_embeddings)

        # learnable temperature scaling for similarity-based scores use log for better gradient
        self.temperature = nn.Parameter(torch.tensor(1/temperature).log(), requires_grad=True)
    
        if parent_index is not None:
            parent_index = parent_index.long()
            if parent_index.shape[0] != concept_embeddings.shape[0]:
                raise ValueError("parent_index size must match number of concept embeddings")
            self.register_buffer("parent_index", parent_index)
            self._validate_precomputed_hierarchy(hierarchy_data)
            self._load_precomputed_hierarchy_buffers(hierarchy_data)
            self._build_weighted_hierarchy_buffers()
        else:
            self.parent_index = None

    def _build_weighted_hierarchy_buffers(self):
        """Precompute ancestor indices/weights for weighted hierarchical logit aggregation.

        Clamped to 2 hops (node + 2 ancestors) to prevent deep paths from dominating.
        """
        if self.parent_index is None:
            return

        num_concepts = int(self.parent_index.shape[0])
        device = self.parent_index.device
        current = torch.arange(num_concepts, dtype=torch.long, device=device)
        levels = []
        max_hops = 2

        for hop in range(max_hops + 1):
            levels.append(current.clone())
            if hop >= max_hops:
                break
            parent_next = torch.full_like(current, -1)
            valid = current >= 0
            if valid.any():
                parent_next[valid] = self.parent_index[current[valid]]

            if not (parent_next >= 0).any():
                break
            current = parent_next

        ancestors = torch.stack(levels, dim=0)
        valid_mask = (ancestors >= 0).float()
        depth = torch.arange(ancestors.shape[0], dtype=torch.float32, device=device).unsqueeze(1)
        weights = (self.weighted_hierarchy_alpha ** depth) * valid_mask

        self.register_buffer("weighted_ancestor_index", ancestors)
        self.register_buffer("weighted_ancestor_weights", weights)

    def _validate_precomputed_hierarchy(self, hierarchy_data):
        if hierarchy_data is None:
            raise ValueError("hierarchy_data is required when parent_index is provided")

        required = {
            "root_indices",
            "child_group_offsets",
            "child_group_children",
            "child_pos_in_parent",
            "path_root_pos",
            "path_edge_counts",
            "path_parents",
            "path_children",
        }
        missing = sorted(k for k in required if k not in hierarchy_data)
        if missing:
            raise ValueError(f"Missing precomputed hierarchy tensors: {missing}")

    def _load_precomputed_hierarchy_buffers(self, hierarchy_data):
        self.register_buffer("root_indices", hierarchy_data["root_indices"].long())
        self.register_buffer("child_group_offsets", hierarchy_data["child_group_offsets"].long())
        self.register_buffer("child_group_children", hierarchy_data["child_group_children"].long())
        self.register_buffer("child_pos_in_parent", hierarchy_data["child_pos_in_parent"].long())
        self.register_buffer("path_root_pos", hierarchy_data["path_root_pos"].long())
        self.register_buffer("path_edge_counts", hierarchy_data["path_edge_counts"].long())
        self.register_buffer("path_parents", hierarchy_data["path_parents"].long())
        self.register_buffer("path_children", hierarchy_data["path_children"].long())

        group_sizes = self.child_group_offsets[1:] - self.child_group_offsets[:-1]
        child_group_parent_ids = torch.repeat_interleave(
            torch.arange(group_sizes.shape[0], dtype=torch.long),
            group_sizes,
        )
        self.register_buffer("child_group_parent_ids", child_group_parent_ids)

    def encode(self, batch):
        if "embedding" in batch:
            return batch["embedding"]
        if "input_ids" in batch:
            if self.encoder_backbone is None:
                raise RuntimeError("input_ids provided but encoder_backbone is None")
            return self.encoder_backbone(
                batch["input_ids"],
                batch["attention_mask"],
                batch.get("target_token_idx"),
            )
        raise ValueError("Batch must contain either embedding or input_ids")

    def score(self, batch):
        x = self.encode(batch)
        projected = self.projector(x)

        if self.baseline:
            if self.baseline_type == "linear":
                return self.baseline_head(projected)

            if self.similarity_metric == "cosine":
                projected_norm = F.normalize(projected, p=2, dim=-1)
                random_classifier_norm = F.normalize(self.random_classifier, p=2, dim=-1)
                return torch.matmul(projected_norm, random_classifier_norm.t())
            
            if self.similarity_metric == "dot_product":
                return torch.matmul(projected, self.random_classifier.t())

            diff = projected.unsqueeze(1) - self.random_classifier.unsqueeze(0)
            l2_dist = torch.norm(diff, p=2, dim=-1)
            return -l2_dist 

        if self.similarity_metric == "cosine":
            projected = F.normalize(projected, p=2, dim=-1)
            concept_norm = F.normalize(self.concept_embeddings, p=2, dim=-1)
            return torch.matmul(projected, concept_norm.t())
        
        if self.similarity_metric == "dot_product":
            return torch.matmul(projected, self.concept_embeddings.t())

        diff = projected.unsqueeze(1) - self.concept_embeddings.unsqueeze(0)
        l2_dist = torch.norm(diff, p=2, dim=-1)
        return -l2_dist

    def weighted_aggregate_logits(self, scores):
        """Aggregate each concept logit with its ancestors using precomputed depth weights."""
        if self.parent_index is None:
            raise RuntimeError("weighted_aggregate_logits requires hierarchy buffers")

        ancestor_index = self.weighted_ancestor_index
        safe_index = ancestor_index.clamp(min=0)
        ancestor_scores = scores[:, safe_index]
        weights = self.weighted_ancestor_weights.to(device=scores.device, dtype=scores.dtype).unsqueeze(0)
        return (ancestor_scores * weights).sum(dim=1)

    def _collect_active_hierarchy_from_targets(self, targets):
        """Collect active roots and per-parent active children from target paths."""
        if targets.numel() == 0:
            return {}

        edges_per_target = self.path_edge_counts[targets]
        max_depth = int(edges_per_target.max().item()) if edges_per_target.numel() > 0 else 0
        if max_depth == 0:
            return {}

        path_parents = self.path_parents[targets]
        path_children = self.path_children[targets]
        active_children_by_parent = {}

        for depth in range(max_depth):
            has_edge = edges_per_target > depth
            if not has_edge.any():
                continue

            active_rows = has_edge.nonzero(as_tuple=False).flatten()
            parents_at_depth = path_parents[active_rows, depth]
            children_at_depth = path_children[active_rows, depth]

            valid = (parents_at_depth >= 0) & (children_at_depth >= 0)
            if not valid.any():
                continue

            parents_list = parents_at_depth[valid].tolist()
            children_list = children_at_depth[valid].tolist()
            for parent_idx, child_idx in zip(parents_list, children_list):
                active_children_by_parent.setdefault(int(parent_idx), set()).add(int(child_idx))

        return active_children_by_parent

    def gold_log_probs(self, scores, targets, active_targets=None):
        """Compute log p(target) with YOLO9000 path decomposition from raw scores.
        """
        if targets.numel() == 0:
            return scores.new_empty((0,))

        if self.parent_index is None:
            raise RuntimeError("gold_log_probs requires hierarchy buffers")

        batch_idx = torch.arange(targets.shape[0], device=scores.device)

        active_children_by_parent = {}
        if active_targets is None:
            active_targets = targets
        active_children_by_parent = self._collect_active_hierarchy_from_targets(active_targets)
        
        target_root_positions = self.path_root_pos[targets]
        
        root_scores = scores[:, self.root_indices]
        root_log_probs = F.log_softmax(root_scores, dim=1)
        log_prob = root_log_probs[batch_idx, target_root_positions]

        edges_per_target = self.path_edge_counts[targets]
        max_depth = int(edges_per_target.max().item())
        if max_depth == 0:
            return log_prob

        path_parents = self.path_parents[targets]
        path_children = self.path_children[targets]

        for depth in range(max_depth):
            has_edge = edges_per_target > depth
            if not has_edge.any():
                continue

            active_batch_indices = has_edge.nonzero(as_tuple=False).flatten()
            parents_at_depth = path_parents[active_batch_indices, depth]
            children_at_depth = path_children[active_batch_indices, depth]

            unique_parents = parents_at_depth.unique().tolist()
            for parent_idx in unique_parents:
                is_this_parent = parents_at_depth == parent_idx
                batch_rows = active_batch_indices[is_this_parent]
                expected_children = children_at_depth[is_this_parent]

                start = int(self.child_group_offsets[parent_idx].item())
                end = int(self.child_group_offsets[parent_idx + 1].item())
                if end <= start:
                    continue

                sibling_indices = self.child_group_children[start:end]
                
                active_children = active_children_by_parent.get(int(parent_idx))
                if active_children:
                    active_children_tensor = torch.tensor(
                        sorted(active_children),
                        dtype=sibling_indices.dtype,
                        device=sibling_indices.device,
                    )
                    active_mask = torch.isin(sibling_indices, active_children_tensor)
                    sibling_indices = sibling_indices[active_mask]
                if sibling_indices.numel() == 0:
                    continue

                sibling_scores = scores[batch_rows][:, sibling_indices]
                sibling_log_probs = F.log_softmax(sibling_scores, dim=1)

                child_matches = expected_children.unsqueeze(1) == sibling_indices.unsqueeze(0)
                if not child_matches.any(dim=1).all():
                    raise RuntimeError("Expected child missing from active sibling subset")
                child_positions = child_matches.long().argmax(dim=1)

                local_batch_idx = torch.arange(batch_rows.shape[0], device=scores.device)
                log_prob[batch_rows] += sibling_log_probs[local_batch_idx, child_positions]

        return log_prob

    def forward(self, batch):
        scores = self.score(batch)
        return scores
