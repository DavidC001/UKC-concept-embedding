#!/usr/bin/env python3
"""Rebuild WSD split TSV files from *.complete.data.json.

This script regenerates train/eval/test TSV files while keeping legacy-compatible
columns used by current training code:
  - answers, answer_id
  - candidates, candidates_id
  - label

For eval/test, gold labels are read from split gold key files (uk_id space) and
mapped to id space via concepts.csv. The script chooses a primary answer_id for
the scalar columns and uses it to build the sequence label field.

When overwriting an existing split file, a timestamped backup is created first.
"""

from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "id",
    "lemma",
    "pos",
    "loc",
    "sentence",
    "candidates",
    "sentence_text",
    "candidates_id",
    "answers",
    "answer_id",
    "label",
]


def load_uk_to_id(concepts_csv: Path) -> dict[int, int]:
    df = pd.read_csv(concepts_csv)
    id_col = pd.to_numeric(df["id"], errors="coerce")
    uk_col = pd.to_numeric(df["uk_id"], errors="coerce")
    valid = id_col.notna() & uk_col.notna()
    return dict(zip(uk_col[valid].astype("int64"), id_col[valid].astype("int64")))


def load_gold(gold_path: Path) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    with gold_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            sid = parts[0]
            vals: list[int] = []
            for tok in parts[1:]:
                try:
                    vals.append(int(tok))
                except Exception:
                    continue
            out[sid] = vals
    return out


def to_sentence_text(sentence_field) -> str:
    if isinstance(sentence_field, list):
        return " ".join(str(x) for x in sentence_field)
    if isinstance(sentence_field, str):
        # Many rows store sentence as a Python-list string.
        try:
            parsed = ast.literal_eval(sentence_field)
            if isinstance(parsed, list):
                return " ".join(str(x) for x in parsed)
        except Exception:
            pass
        return sentence_field
    return ""


def normalize_sentence_list(sentence_field) -> list[str]:
    if isinstance(sentence_field, list):
        return [str(x) for x in sentence_field]
    if isinstance(sentence_field, str):
        try:
            parsed = ast.literal_eval(sentence_field)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
        return sentence_field.split()
    return []


def safe_int(v):
    try:
        return int(v)
    except Exception:
        return None


def choose_primary_answer_id(gold_ids: list[int], candidate_ids: list[int]) -> int | None:
    if not gold_ids:
        return None
    cand_set = set(candidate_ids)
    for gid in gold_ids:
        if gid in cand_set:
            return gid
    return gold_ids[0]


def build_label(loc: int | None, token_count: int, answer_id: int | None) -> list[int]:
    labels = [-100] * max(token_count, 0)
    if answer_id is None:
        return labels
    if loc is None:
        return labels
    if 0 <= loc < len(labels):
        labels[loc] = int(answer_id)
    return labels


def maybe_backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}.bak.{ts}")
    backup.write_bytes(path.read_bytes())
    return backup


def rebuild_split(
    split: str,
    dataset_dir: Path,
    uk_to_id: dict[int, int],
    gold_uk: dict[str, list[int]] | None,
) -> pd.DataFrame:
    complete_path = dataset_dir / f"{split}.complete.data.json"
    with complete_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    rows = []
    for rec in records:
        sid = str(rec.get("id", ""))
        lemma = rec.get("lemma")
        pos = rec.get("pos")
        loc = safe_int(rec.get("loc"))
        sentence_raw = rec.get("sentence")
        sentence_list = normalize_sentence_list(sentence_raw)
        sentence_text = to_sentence_text(sentence_raw)

        if split == "train":
            ans_uk = safe_int(rec.get("answers"))
            cand_uk = [ans_uk] if ans_uk is not None else []
            cand_id = [uk_to_id[u] for u in cand_uk if u in uk_to_id]
            primary_id = cand_id[0] if cand_id else None
            primary_uk = ans_uk
        else:
            cand_uk = [safe_int(x) for x in (rec.get("candidates") or [])]
            cand_uk = [x for x in cand_uk if x is not None]
            cand_id = [uk_to_id[u] for u in cand_uk if u in uk_to_id]

            g_uk = (gold_uk or {}).get(sid, [])
            g_id = [uk_to_id[u] for u in g_uk if u in uk_to_id]
            primary_id = choose_primary_answer_id(g_id, cand_id)

            if primary_id is None:
                primary_uk = None
            else:
                # Prefer gold order when mapping back to UK id for scalar answers.
                primary_uk = None
                for u in g_uk:
                    mapped = uk_to_id.get(u)
                    if mapped == primary_id:
                        primary_uk = u
                        break
                if primary_uk is None:
                    # Fallback: first candidate UK that maps to primary id.
                    for u in cand_uk:
                        if uk_to_id.get(u) == primary_id:
                            primary_uk = u
                            break

        labels = build_label(loc=loc, token_count=len(sentence_list), answer_id=primary_id)

        row = {
            "id": sid,
            "lemma": lemma,
            "pos": pos,
            "loc": loc,
            "sentence": str(sentence_list),
            "answers": primary_uk,
            "sentence_text": sentence_text,
            "answer_id": primary_id,
            "candidates": str(cand_uk),
            "candidates_id": str(cand_id),
            "label": str(labels),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Preserve integer semantics while allowing missing values.
    for col in ("loc", "answers", "answer_id"):
        df[col] = pd.array(df[col], dtype="Int64")

    # Keep exact legacy column order expected by current code.
    df = df[OUTPUT_COLUMNS]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild split TSVs from complete data JSON files")
    parser.add_argument("--dataset-dir", type=Path, default=Path("../dataset"), help="Path to dataset directory")
    parser.add_argument("--write", action="store_true", help="Overwrite split files in dataset-dir")
    parser.add_argument("--no-backup", action="store_true", help="Do not save backups before overwrite")
    parser.add_argument("--out-dir", type=Path, default=None, help="Optional output dir for rebuilt files")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    concepts_csv = dataset_dir / "concepts.csv"

    uk_to_id = load_uk_to_id(concepts_csv)

    eval_gold = load_gold(dataset_dir / "eval-gold-standard" / "UKC.gold.key.txt")
    test_gold = load_gold(dataset_dir / "test-gold-standard" / "UKC.gold.key.txt")

    split_to_gold = {
        "train": None,
        "eval": eval_gold,
        "test": test_gold,
    }

    out_dir = args.out_dir.resolve() if args.out_dir else dataset_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "eval", "test"):
        df = rebuild_split(split, dataset_dir, uk_to_id, split_to_gold[split])
        out_path = out_dir / split

        if args.write and out_dir == dataset_dir and not args.no_backup:
            backup = maybe_backup(out_path)
            if backup is not None:
                print(f"[{split}] backup: {backup}")

        df.to_csv(out_path, sep="\t", index=False, na_rep="")
        print(f"[{split}] wrote {len(df)} rows to {out_path}")

    if not args.write:
        print("Dry run complete. Re-run with --write to overwrite split files.")


if __name__ == "__main__":
    main()
