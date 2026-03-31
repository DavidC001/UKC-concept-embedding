"""Dataset classes and data loading utilities."""

import torch
import numpy as np
import pandas as pd
import ast
from torch.utils.data import Dataset


class WSDDataset(Dataset):
    """Word Sense Disambiguation dataset with candidates."""
    
    def __init__(self, embeddings, labels, candidates, ids, answers=None):
        """
        Args:
            embeddings: (N, 768) numpy array
            labels: (N,) numpy array of scalar target concept IDs
            candidates: list of lists, variable-length candidate concept IDs
            ids: list of sample IDs
            answers: list of answer concept IDs (optional)
        """
        self.embeddings = torch.from_numpy(embeddings).float()
        self.labels = torch.from_numpy(labels).long()
        self.candidates = [np.array(c, dtype=np.int64) for c in candidates]
        self.ids = ids
        self.answers = answers
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'labels': self.labels[idx],
            'candidates': self.candidates[idx],
            'id': self.ids[idx],
            'answer': self.answers[idx] if self.answers else None
        }


def collate_wsd_batch(batch):
    """Custom collate for variable-length candidates."""
    return {
        'embedding': torch.stack([b['embedding'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
        'candidates': [b['candidates'] for b in batch],
        'id': [b['id'] for b in batch],
        'answer': [b['answer'] for b in batch]
    }


def load_npz_data(npz_file):
    """Load embeddings and labels from NPZ file."""
    def to_scalar_targets(arr):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return arr.astype(np.int64)
        if arr.ndim == 2:
            out = np.full((arr.shape[0],), -100, dtype=np.int64)
            valid = arr != -100
            has_valid = valid.any(axis=1)
            first_pos = np.argmax(valid, axis=1)
            row_idx = np.arange(arr.shape[0])
            out[has_valid] = arr[row_idx[has_valid], first_pos[has_valid]]
            return out
        raise ValueError(f"Unsupported label tensor rank in NPZ: shape={arr.shape}")

    data = np.load(npz_file)
    return {
        'train_embeddings': data['train_embeddings'],
        'train_labels': to_scalar_targets(data['train_labels']),
        'eval_embeddings': data['eval_embeddings'],
        'eval_labels': to_scalar_targets(data['eval_labels']),
        'test_embeddings': data['test_embeddings'],
        'test_labels': to_scalar_targets(data['test_labels'])
    }


def load_metadata(train_tsv, eval_tsv, test_tsv):
    """Load IDs, candidates, and answers from TSV files."""
    def parse_tsv(filepath):
        df = pd.read_csv(filepath, sep='\t')
        ids = df['id'].tolist()
        candidates = [ast.literal_eval(c) for c in df['candidates_id']]
        answers = df['answer_id'].tolist() if 'answer_id' in df.columns else None
        return ids, candidates, answers
    
    train_ids, train_cands, train_ans = parse_tsv(train_tsv)
    eval_ids, eval_cands, eval_ans = parse_tsv(eval_tsv)
    test_ids, test_cands, test_ans = parse_tsv(test_tsv)
    
    return {
        'train': (train_ids, train_cands, train_ans),
        'eval': (eval_ids, eval_cands, eval_ans),
        'test': (test_ids, test_cands, test_ans)
    }
