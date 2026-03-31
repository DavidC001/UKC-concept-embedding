"""Standalone datasets and dataloaders for WSD."""

import ast
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class SplitData:
    ids: list
    candidates: list
    answers: list
    labels: np.ndarray
    records: list


def _parse_labels(label_value):
    if isinstance(label_value, str):
        return ast.literal_eval(label_value)
    if isinstance(label_value, list):
        return label_value
    return []


def _extract_target_label_from_sequence(label_seq):
    for value in label_seq:
        iv = int(value)
        if iv != -100:
            return iv
    return -100


def _build_scalar_targets(df):
    labels = np.full((len(df),), -100, dtype=np.int64)

    if "answer_id" in df.columns:
        answer_ids = pd.to_numeric(df["answer_id"], errors="coerce")
        for i, value in enumerate(answer_ids.tolist()):
            if pd.notna(value):
                labels[i] = int(value)
        return labels

    for i, v in enumerate(df["label"].tolist()):
        seq = _parse_labels(v)
        labels[i] = _extract_target_label_from_sequence(seq)
    return labels


def load_tsv_split(tsv_path, max_label_len=128):
    del max_label_len

    df = pd.read_csv(tsv_path, sep="\t")

    ids = df["id"].tolist()
    candidates = [ast.literal_eval(c) for c in df["candidates_id"]]
    answers = df["answer_id"].tolist() if "answer_id" in df.columns else None
    labels = _build_scalar_targets(df)
    records = df.to_dict(orient="records")

    return SplitData(
        ids=ids,
        candidates=[np.array(c, dtype=np.int64) for c in candidates],
        answers=answers,
        labels=labels,
        records=records,
    )


def load_all_splits(train_tsv, eval_tsv, test_tsv, max_label_len=128):
    return {
        "train": load_tsv_split(train_tsv, max_label_len=max_label_len),
        "eval": load_tsv_split(eval_tsv, max_label_len=max_label_len),
        "test": load_tsv_split(test_tsv, max_label_len=max_label_len),
    }


class PrecomputedEmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels, split_data: SplitData):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.labels = torch.from_numpy(labels).long()
        self.ids = split_data.ids
        self.candidates = split_data.candidates
        self.answers = split_data.answers

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            "embedding": self.embeddings[idx],
            "labels": self.labels[idx],
            "id": self.ids[idx],
            "candidates": self.candidates[idx],
            "answer": self.answers[idx] if self.answers else None,
        }


class EncoderTextDataset(Dataset):
    def __init__(self, split_data: SplitData):
        self.records = split_data.records
        self.labels = torch.from_numpy(split_data.labels).long()
        self.ids = split_data.ids
        self.candidates = split_data.candidates
        self.answers = split_data.answers

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        return {
            "record": rec,
            "labels": self.labels[idx],
            "id": self.ids[idx],
            "candidates": self.candidates[idx],
            "answer": self.answers[idx] if self.answers else None,
        }


def collate_precomputed(batch):
    return {
        "embedding": torch.stack([b["embedding"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "id": [b["id"] for b in batch],
        "candidates": [b["candidates"] for b in batch],
        "answer": [b["answer"] for b in batch],
    }


def _extract_word_location(record):
    for key in ("loc", "location", "word_loc", "word_index"):
        if key in record and record[key] is not None:
            return int(record[key])
    raise ValueError("Record is missing target word location (expected one of: loc, location, word_loc, word_index)")


def _last_subword_positions(encodings, target_word_locs):
    positions = []
    found = []

    for i, enc in enumerate(encodings.encodings):
        target_word_loc = int(target_word_locs[i])
        word_ids = enc.word_ids

        token_positions = [j for j, wid in enumerate(word_ids) if wid == target_word_loc]
        if token_positions:
            positions.append(token_positions[-1])
            found.append(True)
        else:
            positions.append(0)
            found.append(False)

    return torch.tensor(positions, dtype=torch.long), torch.tensor(found, dtype=torch.bool)


def collate_encoder(batch, tokenizer, formatter, max_length, token_pooling):
    records = [b["record"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])

    if token_pooling == "target_last_subword":
        split_words = [str(rec.get("sentence_text", "")).split() for rec in records]
        tokens = tokenizer(
            split_words,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            is_split_into_words=True,
        )
        target_word_locs = [_extract_word_location(rec) for rec in records]
        target_token_idx, target_found = _last_subword_positions(tokens, target_word_locs)

        labels = labels.clone()
        labels[~target_found] = -100
        texts = [" ".join(words) for words in split_words]
    else:
        texts = [formatter.format(rec) for rec in records]
        tokens = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        target_token_idx = None

    out = {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels": labels,
        "id": [b["id"] for b in batch],
        "candidates": [b["candidates"] for b in batch],
        "answer": [b["answer"] for b in batch],
        "texts": texts,
    }
    if target_token_idx is not None:
        out["target_token_idx"] = target_token_idx
    return out


def load_npz_data(npz_file):
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
        "train_embeddings": data["train_embeddings"],
        "train_labels": to_scalar_targets(data["train_labels"]),
        "eval_embeddings": data["eval_embeddings"],
        "eval_labels": to_scalar_targets(data["eval_labels"]),
        "test_embeddings": data["test_embeddings"],
        "test_labels": to_scalar_targets(data["test_labels"]),
    }


def build_dataloaders_precomputed(npz_data, split_map, batch_size, eval_batch_size):
    train_ds = PrecomputedEmbeddingDataset(npz_data["train_embeddings"], npz_data["train_labels"], split_map["train"])
    eval_ds = PrecomputedEmbeddingDataset(npz_data["eval_embeddings"], npz_data["eval_labels"], split_map["eval"])
    test_ds = PrecomputedEmbeddingDataset(npz_data["test_embeddings"], npz_data["test_labels"], split_map["test"])

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_precomputed),
        "eval": DataLoader(eval_ds, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_precomputed),
        "test": DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_precomputed),
    }


def build_dataloaders_encoder(split_map, tokenizer, formatter, batch_size, eval_batch_size, max_length, token_pooling):
    train_ds = EncoderTextDataset(split_map["train"])
    eval_ds = EncoderTextDataset(split_map["eval"])
    test_ds = EncoderTextDataset(split_map["test"])

    collate_fn = partial(
        collate_encoder,
        tokenizer=tokenizer,
        formatter=formatter,
        max_length=max_length,
        token_pooling=token_pooling,
    )

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
        "eval": DataLoader(eval_ds, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn),
        "test": DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn),
    }
