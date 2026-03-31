"""Prediction extraction and serialization helpers."""

import torch


def evaluate_with_candidates(scores, candidates, labels, ids, concept_id_to_index, index_to_concept_id):
    preds = []
    for i in range(len(ids)):
        label_value = labels[i]
        if isinstance(label_value, torch.Tensor):
            if label_value.dim() == 0:
                if int(label_value.item()) == -100:
                    continue
            else:
                label_seq = label_value.cpu().numpy()
                valid = (label_seq != -100).nonzero()[0]
                if len(valid) == 0:
                    continue
        elif int(label_value) == -100:
            continue

        if candidates is None:
            cand_ids = []
        else:
            cand_ids = candidates[i]
        
        cand_indices = [concept_id_to_index[int(c)] for c in cand_ids if int(c) in concept_id_to_index]
        if not cand_indices:
            pred_idx = int(scores[i].argmax().item())
        else:
            cand_tensor = torch.tensor(cand_indices, device=scores.device, dtype=torch.long)
            cand_scores = scores[i, cand_tensor]
            pred_idx = int(cand_tensor[int(cand_scores.argmax().item())].item())

        preds.append((ids[i], index_to_concept_id.get(pred_idx, 1)))
    return preds

def save_predictions(predictions, filepath):
    with open(filepath, "w") as f:
        for sample_id, concept_id in predictions:
            out_id = concept_id
            f.write(f"{sample_id} {out_id}\n")
