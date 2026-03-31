"""
Build precomputed embeddings for WSD datasets.

This script encodes WSD dataset sentences using a pretrained model and saves
the embeddings in NPZ format compatible with PrecomputedEmbeddingDataset.
"""

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class SentenceDataset(Dataset):
    """Dataset wrapper for sentences with location information."""

    def __init__(self, sentences, labels, locations):
        self.sentences = sentences
        self.labels = labels
        self.locations = locations

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx], self.locations[idx]


def collate_fn(batch):
    """Collate function for sentence dataset."""
    sentences, labels, locs = zip(*batch)
    return list(sentences), list(labels), list(locs)


def load_tsv_file(tsv_path):
    """Load a single TSV file and extract sentences, labels, and locations."""
    df = pd.read_csv(tsv_path, sep="\t")

    sentences = df["sentence_text"].tolist()
    
    # Parse labels
    labels = []
    for label_val in df["label"].tolist():
        if isinstance(label_val, str):
            labels.append(ast.literal_eval(label_val))
        elif isinstance(label_val, list):
            labels.append(label_val)
        else:
            labels.append([])
    
    # Load locations
    locations = df["loc"].astype(int).tolist()

    return sentences, labels, locations


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings to unit vectors."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normalized = embeddings / norms
    return normalized


def encode_dataset(
    sentences,
    labels,
    locations,
    tokenizer,
    model,
    device,
    batch_size=32,
    max_length=128,
    show_progress=True,
):
    """
    Encode sentences using a pretrained model.

    Args:
        sentences: List of sentence strings
        labels: List of label sequences for each sentence
        locations: List of target word locations in each sentence
        tokenizer: Tokenizer to use
        model: Model to use for encoding
        device: Device to use (cuda or cpu)
        batch_size: Batch size for encoding
        max_length: Maximum sequence length
        show_progress: Whether to show progress bar

    Returns:
        embeddings: (N, D) array of sentence embeddings
        token_labels: (N, T) array of token labels (padded to max_length)
    """
    dataset = SentenceDataset(sentences, labels, locations)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0,
    )

    all_embeddings = []
    all_token_labels = []

    iterator = tqdm(dataloader, desc="Encoding", disable=not show_progress)

    for batch in iterator:
        batch_sentences, batch_word_labels, batch_locs = batch
        batch_words = [s.split() for s in batch_sentences]
        batch_size_actual = len(batch_words)

        # Tokenize
        encodings = tokenizer(
            batch_words,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            is_split_into_words=True,
        )

        inputs = {k: v.to(device) for k, v in encodings.items()}

        # Align labels: map word labels to tokens
        batch_labels = []
        for i in range(batch_size_actual):
            word_ids = encodings.encodings[i].word_ids
            token_labels = [-100] * len(word_ids)

            for j, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                if word_id >= len(batch_word_labels[i]):
                    continue

                # Detect last token of a word
                is_last_token = (
                    j == len(word_ids) - 1 or word_ids[j + 1] != word_id
                )

                if is_last_token:
                    token_labels[j] = batch_word_labels[i][word_id]

            batch_labels.append(token_labels)

        # Pad labels
        padded_labels = []
        for lbl in batch_labels:
            if len(lbl) < max_length:
                lbl = lbl + [-100] * (max_length - len(lbl))
            else:
                lbl = lbl[:max_length]
            padded_labels.append(lbl)

        batch_labels = np.array(padded_labels, dtype=np.int64)

        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states[-1]  # (B, T, H)
        batch_embeddings = []

        # Extract target word embeddings (last subword token)
        for i in range(batch_size_actual):
            word_ids = encodings.encodings[i].word_ids
            target_word_id = batch_locs[i]

            token_idxs = [
                j for j, w_id in enumerate(word_ids) if w_id == target_word_id
            ]

            if not token_idxs:
                # Fallback to CLS token if target word not found
                emb = hidden_states[i, 0]
            else:
                # Use last subword token
                emb = hidden_states[i, token_idxs[-1]]

            batch_embeddings.append(emb.cpu().numpy())

        all_embeddings.extend(batch_embeddings)
        all_token_labels.append(batch_labels)

    final_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
    final_token_labels = np.vstack(all_token_labels) if all_token_labels else np.array([])

    if len(final_embeddings) != len(final_token_labels):
        raise ValueError(
            "Embedding/label size mismatch after encoding: "
            f"embeddings={len(final_embeddings)}, labels={len(final_token_labels)}"
        )

    return final_embeddings, final_token_labels


def main():
    parser = argparse.ArgumentParser(
        description="Build precomputed embeddings for WSD datasets"
    )
    parser.add_argument(
        "--train_tsv",
        type=str,
        default="dataset/train",
        help="Path to training TSV file",
    )
    parser.add_argument(
        "--eval_tsv",
        type=str,
        default="dataset/eval",
        help="Path to evaluation TSV file",
    )
    parser.add_argument(
        "--test_tsv",
        type=str,
        default="dataset/test",
        help="Path to test TSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sentence_embeddings.npz",
        help="Output NPZ file path",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-base",
        help="Pretrained model name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="L2-normalize embeddings",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu, mps). Auto-detect if not specified",
    )

    args = parser.parse_args()

    # Set device
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"✓ Using device: {device}")

    # Load model and tokenizer
    print(f"✓ Loading model: {args.model_name}")
    model = AutoModel.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = model.to(device)
    model.eval()

    print(f"  - Model hidden size: {model.config.hidden_size}")
    print(f"  - Tokenizer vocab size: {tokenizer.vocab_size}")

    # Load datasets
    print("\n✓ Loading datasets...")
    train_sentences, train_labels, train_locations = load_tsv_file(args.train_tsv)
    eval_sentences, eval_labels, eval_locations = load_tsv_file(args.eval_tsv)
    test_sentences, test_labels, test_locations = load_tsv_file(args.test_tsv)

    print(f"  - Train: {len(train_sentences)} samples")
    print(f"  - Eval: {len(eval_sentences)} samples")
    print(f"  - Test: {len(test_sentences)} samples")

    # Encode datasets
    print("\n✓ Encoding datasets...")

    train_embeddings, train_token_labels = encode_dataset(
        train_sentences,
        train_labels,
        train_locations,
        tokenizer,
        model,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        show_progress=True,
    )

    eval_embeddings, eval_token_labels = encode_dataset(
        eval_sentences,
        eval_labels,
        eval_locations,
        tokenizer,
        model,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        show_progress=True,
    )

    test_embeddings, test_token_labels = encode_dataset(
        test_sentences,
        test_labels,
        test_locations,
        tokenizer,
        model,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        show_progress=True,
    )

    if len(train_embeddings) != len(train_token_labels):
        raise ValueError(
            f"Train mismatch before save: embeddings={len(train_embeddings)}, labels={len(train_token_labels)}"
        )
    if len(eval_embeddings) != len(eval_token_labels):
        raise ValueError(
            f"Eval mismatch before save: embeddings={len(eval_embeddings)}, labels={len(eval_token_labels)}"
        )
    if len(test_embeddings) != len(test_token_labels):
        raise ValueError(
            f"Test mismatch before save: embeddings={len(test_embeddings)}, labels={len(test_token_labels)}"
        )

    # Normalize embeddings
    if args.normalize:
        print("\n✓ L2-normalizing embeddings...")
        train_embeddings = l2_normalize(train_embeddings)
        eval_embeddings = l2_normalize(eval_embeddings)
        test_embeddings = l2_normalize(test_embeddings)

    # Save embeddings
    print(f"\n✓ Saving embeddings to: {args.output}")
    np.savez_compressed(
        args.output,
        train_embeddings=train_embeddings,
        train_labels=train_token_labels,
        eval_embeddings=eval_embeddings,
        eval_labels=eval_token_labels,
        test_embeddings=test_embeddings,
        test_labels=test_token_labels,
    )

    print(f"✓ Done!")
    print(f"  - Train embeddings shape: {train_embeddings.shape}")
    print(f"  - Eval embeddings shape: {eval_embeddings.shape}")
    print(f"  - Test embeddings shape: {test_embeddings.shape}")


if __name__ == "__main__":
    main()
