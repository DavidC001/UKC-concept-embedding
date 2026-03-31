"""Tensor and label utilities shared across training/evaluation."""

import torch


def move_batch_to_device(batch, device):
    out = dict(batch)
    for key in ["embedding", "labels", "input_ids", "attention_mask", "target_token_idx"]:
        if key in out and isinstance(out[key], torch.Tensor):
            out[key] = out[key].to(device)
    return out


def remap_labels_to_index_space(labels, concept_id_to_index, num_concepts):
    """Map labels to contiguous model indices; unknown/out-of-range labels become -100."""
    out = labels.copy()
    flat = out.reshape(-1)

    for i in range(flat.shape[0]):
        value = int(flat[i])
        if value == -100:
            continue

        mapped = concept_id_to_index.get(value)
        if mapped is not None:
            idx = int(mapped)
            flat[i] = idx if 0 <= idx < num_concepts else -100
        else:
            flat[i] = -100

    return out
