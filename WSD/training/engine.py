"""Training loop internals and optimization helpers."""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from WSD.common import move_batch_to_device

def _compute_loss(model, selected_scores, gold_targets, loss_type):
    if loss_type == "hierarchy_factorized":
        gold_log_probs = model.gold_log_probs(selected_scores * model.temperature.exp(), gold_targets)
        loss = -gold_log_probs.mean()
    elif loss_type == "hierarchy_weighted_agg":
        aggregated_scores = model.weighted_aggregate_logits(selected_scores * model.temperature.exp())
        loss = F.nll_loss(F.log_softmax(aggregated_scores, dim=1), gold_targets)
    elif loss_type == "cosine":
        gold_scores = selected_scores[torch.arange(gold_targets.shape[0], device=gold_targets.device), gold_targets]
        loss = 1.0 - gold_scores.mean()
    elif loss_type == "standard":
        loss = F.cross_entropy(selected_scores * model.temperature.exp(), gold_targets)
    else:
        raise ValueError("LOSS_TYPE must be one of: standard, cosine, hierarchy_factorized, hierarchy_weighted_agg")
    
    return loss

def run_epoch(
    model,
    loader,
    optimizer,
    device,
    loss_type,
):
    model.train()
    total_loss = 0.0
    total_steps = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        batch = move_batch_to_device(batch, device)
        raw_scores = model(batch) # (B, num_concepts) unnormalized similarity scores for each class
        labels = batch["labels"] # (B) gold concept indices, aligned with model output space, with -100 for ignored samples
        
        valid_mask = labels != -100
        if int(valid_mask.sum().item()) == 0:
            continue

        sample_idx = valid_mask.nonzero(as_tuple=False).flatten()
        gold_targets = labels[sample_idx]        # (S) gold concept indices for valid samples
        selected_scores = raw_scores[sample_idx] # (S, num_concepts) similarity scores for valid samples

        loss = _compute_loss(model, selected_scores, gold_targets, loss_type)

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
    loss_type
):
    model.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            raw_scores = model(batch)
            labels = batch["labels"]
            valid_mask = labels != -100
            if int(valid_mask.sum().item()) == 0:
                continue

            sample_idx = valid_mask.nonzero(as_tuple=False).flatten()
            gold_targets = labels[sample_idx]
            selected_scores = raw_scores[sample_idx]

            loss = _compute_loss(model, selected_scores, gold_targets, loss_type)

            total_loss += float(loss.item())
            total_steps += 1

    if total_steps == 0:
        return float("inf")
    return total_loss / total_steps


def is_improved(current_value, best_value, mode):
    if best_value is None:
        return True
    if mode == "max":
        return current_value > best_value
    if mode == "min":
        return current_value < best_value
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
