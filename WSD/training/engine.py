"""Training loop internals and optimization helpers."""

import torch
import torch.nn.functional as F
from tqdm import tqdm


def validate_precomputed_alignment(npz_data, split_map):
    """Fail fast when NPZ embeddings/labels are not aligned with current TSV splits."""
    split_to_keys = {
        "train": ("train_embeddings", "train_labels"),
        "eval": ("eval_embeddings", "eval_labels"),
        "test": ("test_embeddings", "test_labels"),
    }

    errors = []
    for split, (emb_key, lbl_key) in split_to_keys.items():
        emb_n = int(len(npz_data[emb_key]))
        lbl_n = int(len(npz_data[lbl_key]))
        split_n = int(len(split_map[split].ids))

        if emb_n != lbl_n:
            errors.append(f"[{split}] NPZ mismatch: {emb_key}={emb_n}, {lbl_key}={lbl_n}")
        if emb_n != split_n or lbl_n != split_n:
            errors.append(
                f"[{split}] NPZ/TSV mismatch: embeddings={emb_n}, labels={lbl_n}, split_rows={split_n}"
            )

        if npz_data[lbl_key].ndim != 1:
            errors.append(f"[{split}] expected 1D scalar labels in NPZ, got shape={npz_data[lbl_key].shape}")

        if split_map[split].labels.ndim != 1:
            errors.append(f"[{split}] expected 1D scalar labels in split map, got shape={split_map[split].labels.shape}")

        if npz_data[lbl_key].shape != split_map[split].labels.shape:
            errors.append(
                f"[{split}] label shape mismatch: npz={npz_data[lbl_key].shape}, split={split_map[split].labels.shape}"
            )

    if errors:
        detail = "\n  - " + "\n  - ".join(errors)
        raise ValueError(
            "Precomputed data alignment check failed. "
            "Regenerate NPZ embeddings from the same current train/eval/test TSV files."
            f"\nDetails:{detail}"
        )


def candidate_ce_loss(
    model,
    selected_scores,
    gold_targets,
    sample_idx,
    candidates,
    concept_id_to_index,
    use_extra_negative_sampling=False,
    num_extra_negatives=64,
):
    per_item_losses = []
    num_concepts = int(selected_scores.shape[1])

    for local_i, batch_i in enumerate(sample_idx.tolist()):
        gold_raw = int(gold_targets[local_i].item())
        mapped_gold = concept_id_to_index.get(gold_raw)
        gold_idx = int(mapped_gold) if mapped_gold is not None else gold_raw
        if not (0 <= gold_idx < num_concepts):
            continue

        cand_ids = candidates[batch_i]

        cand_indices = []
        for cid in cand_ids:
            idx = concept_id_to_index.get(int(cid))
            if idx is not None and 0 <= idx < num_concepts:
                cand_indices.append(int(idx))

        if gold_idx not in cand_indices:
            cand_indices.append(gold_idx)

        if use_extra_negative_sampling and num_extra_negatives > 0:
            blocked = set(cand_indices)
            available = max(0, num_concepts - len(blocked))
            to_sample = min(int(num_extra_negatives), available)
            while to_sample > 0:
                neg = int(torch.randint(0, num_concepts, (1,), device=selected_scores.device).item())
                if neg in blocked:
                    continue
                blocked.add(neg)
                cand_indices.append(neg)
                to_sample -= 1

        if not cand_indices:
            continue

        cand_tensor = torch.tensor(cand_indices, dtype=torch.long, device=selected_scores.device)
        row_scores = selected_scores[local_i]
        repeated_scores = row_scores.unsqueeze(0).expand(cand_tensor.shape[0], -1)
        cand_path_log_probs = model.gold_log_probs(
            repeated_scores,
            cand_tensor,
            active_targets=cand_tensor,
        )

        target_pos = (cand_tensor == gold_idx).nonzero(as_tuple=False)
        if target_pos.numel() == 0:
            continue

        target_pos = target_pos.flatten()[0]
        per_item_losses.append(F.cross_entropy(cand_path_log_probs.unsqueeze(0), target_pos.unsqueeze(0)))

    if not per_item_losses:
        return None

    return torch.stack(per_item_losses).mean()


def hierarchy_is_compatible(hierarchy_data, concept_id_to_index, num_concepts):
    """Check hierarchy cache compatibility with current embedding index space."""
    parent_index = hierarchy_data.get("parent_index")
    if parent_index is None:
        return False
    if int(parent_index.shape[0]) != int(num_concepts):
        return False

    for key in ("path_edge_counts", "path_root_pos", "child_pos_in_parent"):
        tensor = hierarchy_data.get(key)
        if tensor is None or int(tensor.shape[0]) != int(num_concepts):
            return False

    cached_n = hierarchy_data.get("num_concepts")
    if cached_n is not None and int(cached_n) != int(num_concepts):
        return False

    cached_map = hierarchy_data.get("concept_id_to_index")
    if cached_map is not None and cached_map != concept_id_to_index:
        return False

    return True


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    concept_id_to_index,
    use_hierarchy_loss,
    hierarchy_loss_type,
    use_extra_negative_sampling,
    num_extra_negatives,
    move_batch_to_device,
):
    model.train()
    total_loss = 0.0
    total_steps = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        batch = move_batch_to_device(batch, device)
        raw_scores = model(batch, apply_hierarchy_softmax=False)
        labels = batch["labels"]
        valid_mask = labels != -100
        if int(valid_mask.sum().item()) == 0:
            continue

        sample_idx = valid_mask.nonzero(as_tuple=False).flatten()
        gold_targets = labels[sample_idx]
        selected_scores = raw_scores[sample_idx]

        if use_hierarchy_loss:
            if hierarchy_loss_type == "factorized":
                loss = candidate_ce_loss(
                    model=model,
                    selected_scores=selected_scores,
                    gold_targets=gold_targets,
                    sample_idx=sample_idx,
                    candidates=batch["candidates"],
                    concept_id_to_index=concept_id_to_index,
                    use_extra_negative_sampling=use_extra_negative_sampling,
                    num_extra_negatives=num_extra_negatives,
                )
                if loss is None:
                    gold_log_probs = model.gold_log_probs(selected_scores, gold_targets)
                    loss = -gold_log_probs.mean()
            elif hierarchy_loss_type == "weighted_agg":
                aggregated_scores = model.weighted_aggregate_logits(selected_scores)
                loss = F.nll_loss(F.log_softmax(aggregated_scores, dim=1), gold_targets)
            else:
                raise ValueError("HIERARCHY_LOSS_TYPE must be one of: factorized, weighted_agg")
        else:
            loss = F.cross_entropy(selected_scores, gold_targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_steps += 1

    if total_steps == 0:
        return 0.0
    return total_loss / total_steps


def run_validation_loss(
    model,
    loader,
    device,
    use_hierarchy_loss,
    hierarchy_loss_type,
    move_batch_to_device,
):
    model.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            raw_scores = model(batch, apply_hierarchy_softmax=False)
            labels = batch["labels"]
            valid_mask = labels != -100
            if int(valid_mask.sum().item()) == 0:
                continue

            sample_idx = valid_mask.nonzero(as_tuple=False).flatten()
            gold_targets = labels[sample_idx]
            selected_scores = raw_scores[sample_idx]

            if use_hierarchy_loss:
                if hierarchy_loss_type == "factorized":
                    gold_log_probs = model.gold_log_probs(selected_scores, gold_targets)
                    loss = -gold_log_probs.mean()
                elif hierarchy_loss_type == "weighted_agg":
                    aggregated_scores = model.weighted_aggregate_logits(selected_scores)
                    loss = F.nll_loss(F.log_softmax(aggregated_scores, dim=1), gold_targets)
                else:
                    raise ValueError("HIERARCHY_LOSS_TYPE must be one of: factorized, weighted_agg")
            else:
                loss = F.cross_entropy(selected_scores, gold_targets)

            total_loss += float(loss.item())
            total_steps += 1

    if total_steps == 0:
        return float("inf")
    return total_loss / total_steps


def is_improved(current_value, best_value, mode, min_delta):
    if best_value is None:
        return True
    if mode == "max":
        return current_value > (best_value + min_delta)
    if mode == "min":
        return current_value < (best_value - min_delta)
    raise ValueError("EARLY_STOPPING_MODE must be one of: min, max")


def resolve_monitor_value(monitor, eval_scores, eval_loss):
    if monitor == "eval_f1":
        all_scores = eval_scores.get("ALL", {}) if eval_scores else {}
        value = all_scores.get("f1")
        if value is not None:
            return float(value), "eval_f1"
        return -float(eval_loss), "-eval_loss_fallback"

    if monitor == "eval_loss":
        return float(eval_loss), "eval_loss"

    raise ValueError("EARLY_STOPPING_MONITOR must be one of: eval_f1, eval_loss")
