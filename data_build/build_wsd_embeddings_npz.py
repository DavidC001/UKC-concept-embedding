#!/usr/bin/env python3
"""Build WSD NPZ embeddings from current TSV splits.

This script reads train/eval/test TSV files and writes a single NPZ with:
  - train_embeddings, train_labels
  - eval_embeddings, eval_labels
  - test_embeddings, test_labels

The output is aligned row-by-row with the input TSV files.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build aligned NPZ embeddings from WSD TSV splits")
    parser.add_argument("--dataset-dir", type=Path, default=Path("../dataset"), help="Directory containing train/eval/test TSV files")
    parser.add_argument("--model-name", type=str, default="FacebookAI/xlm-roberta-large", help="HF model name")
    parser.add_argument("--output", type=Path, default=None, help="Output NPZ path")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer max length")
    parser.add_argument("--pooling", choices=["target_last_subword", "cls"], default="target_last_subword")
    parser.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization")
    parser.add_argument("--device", type=str, default="auto", help="cuda, mps, cpu, or auto")
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split(tsv_path: Path) -> pd.DataFrame:
    return pd.read_csv(tsv_path, sep="\t")


def _extract_target_label_from_sequence(raw_label) -> int:
    seq = ast.literal_eval(str(raw_label))
    for value in seq:
        iv = int(value)
        if iv != -100:
            return iv
    return -100


def build_target_labels(df: pd.DataFrame) -> np.ndarray:
    """Build scalar labels directly from answer_id when available.

    Fallback to parsing the legacy sequence label only if answer_id is missing.
    """
    labels = np.full((len(df),), -100, dtype=np.int64)

    if "answer_id" in df.columns:
        answer_ids = pd.to_numeric(df["answer_id"], errors="coerce")
        for i, value in enumerate(answer_ids.tolist()):
            if pd.notna(value):
                labels[i] = int(value)
        return labels

    for i, raw in enumerate(df["label"].tolist()):
        labels[i] = _extract_target_label_from_sequence(raw)
    return labels


def _load_tokenizer(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True, add_prefix_space=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def encode_split(
    df: pd.DataFrame,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    pooling: str,
) -> np.ndarray:
    sentence_texts = df["sentence_text"].astype(str).tolist()
    target_locs = df["loc"].astype(int).tolist()

    all_embeddings = []
    all_target_found = []
    model.eval()

    with torch.no_grad():
        for start in tqdm(range(0, len(df), batch_size), desc="Encode", leave=False):
            end = min(start + batch_size, len(df))
            batch_texts = sentence_texts[start:end]
            batch_locs = target_locs[start:end]

            if pooling == "target_last_subword":
                batch_words = [t.split() for t in batch_texts]
                encodings = tokenizer(
                    batch_words,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                    is_split_into_words=True,
                )
            else:
                encodings = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )

            inputs = {k: v.to(device) for k, v in encodings.items()}
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state

            for i in range(hidden.shape[0]):
                if pooling == "cls":
                    emb = hidden[i, 0]
                    target_found = True
                else:
                    word_ids = encodings.encodings[i].word_ids
                    target_word_id = batch_locs[i]
                    token_idxs = [j for j, wid in enumerate(word_ids) if wid == target_word_id]
                    target_found = len(token_idxs) > 0
                    emb = hidden[i, token_idxs[-1]] if target_found else hidden[i, 0]
                all_embeddings.append(emb.detach().cpu().numpy())
                all_target_found.append(target_found)

    if not all_embeddings:
        empty_embeddings = np.zeros((0, int(model.config.hidden_size)), dtype=np.float32)
        empty_mask = np.zeros((0,), dtype=bool)
        return empty_embeddings, empty_mask
    return np.vstack(all_embeddings).astype(np.float32), np.asarray(all_target_found, dtype=bool)


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (embeddings / norms).astype(np.float32)


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()

    if args.output is None:
        out_name = f"sentence_embeddings_labels_loc_{sanitize_model_name(args.model_name)}.npz"
        output_path = (dataset_dir.parent / out_name).resolve()
    else:
        output_path = args.output.resolve()

    device = choose_device(args.device)
    print(f"Using device: {device}")
    print(f"Model: {args.model_name}")

    tokenizer = _load_tokenizer(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    split_paths = {
        "train": dataset_dir / "train",
        "eval": dataset_dir / "eval",
        "test": dataset_dir / "test",
    }

    data = {}
    for split, path in split_paths.items():
        print(f"Loading {split}: {path}")
        df = load_split(path)

        labels = build_target_labels(df)
        embeddings, target_found_mask = encode_split(
            df=df,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pooling=args.pooling,
        )

        # If target token is truncated away, keep row alignment but ignore in loss/eval.
        labels[~target_found_mask] = -100

        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError(
                f"{split} mismatch after encoding: embeddings={embeddings.shape[0]}, labels={labels.shape[0]}"
            )

        if not args.no_normalize:
            embeddings = l2_normalize(embeddings)

        data[f"{split}_embeddings"] = embeddings
        data[f"{split}_labels"] = labels

        print(
            f"  {split}: rows={len(df)} emb_shape={embeddings.shape} label_shape={labels.shape} "
            f"target_missing={int((~target_found_mask).sum())}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **data)
    print(f"Saved NPZ: {output_path}")


if __name__ == "__main__":
    main()
