"""
Normalize UKC concept files by pruning selected hyponym trees and adding POS roots.

This script:
1. Prunes subtrees that are reachable via relation_type=20 (hyponym)
   from a predefined set of root concept IDs listed by the user.
2. Removes all relations (any relation_type) touching removed concepts.
3. Ensures verb/adjective concepts are attached to dedicated synthetic roots;
   adverbs are handled as well when present.

Input files expected:
- concepts.csv
- concept_glosses.csv
- concept_relations.csv
- concept_pos.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

import pandas as pd


HYPONYM_RELATION_TYPE = 20
ENTITY_CONCEPT_ID = 1

# IDs listed before 'entity' in the user-provided root list.
PRUNE_ROOT_IDS_BEFORE_ENTITY: Tuple[int, ...] = (
    110413,
    110154,
    110444,
    110149,
    110112,
    111316,
    111318,
    111320,
    111322,
    111348,
    111350,
    111356,
    111358,
    111362,
    111364,
    111374,
    111376,
    111380,
    111382,
    111384,
    111386,
    111388,
    111391,
    111396,
    111401,
    111410,
    111412,
    111418,
    111419,
    111421,
    111423,
    111426,
    111428,
    111430,
    111432,
    111436,
    111438,
    111440,
    111442,
    111444,
    111446,
    111448,
    111450,
    111459,
    111469,
    111471,
    111472,
    111474,
    111476,
    111492,
    111494,
    111502,
    111503,
    111504,
    111506,
    111507,
    111508,
    111510,
    111511,
    111513,
    111514,
    111516,
    111517,
    111518,
    111520,
    111521,
    111522,
    111524,
    111525,
    111527,
    111528,
    111530,
    111531,
    111532,
    111534,
    111535,
    111536,
    111538,
    111539,
    111541,
    111542,
    111544,
    111545,
    111547,
    111548,
    111550,
    111552,
    111554,
    111556,
    111560,
    111562,
    111566,
    111568,
)

# POS IDs from concept_pos.csv:
# 1=NOUN, 2=ADJECTIVE, 3=VERB, 4=ADVERB
POS_ROOT_SPECS: Tuple[Tuple[int, str, str], ...] = (
    (3, "verb root", "Words used to describe actions, processes, states of being, and occurrences."),
    (2, "adjective root", "Words used to describe properties and attributes of things."),
    (4, "adverb root", "Words used to describe the manner, place, time, frequency, certainty, or other circumstances of an action or event."),
)


def build_children_map(relations_df: pd.DataFrame) -> Dict[int, Set[int]]:
    """Build a mapping from parent concept ID to set of child concept IDs for hyponym relations."""
    children: Dict[int, Set[int]] = defaultdict(set)
    hyponym_df = relations_df[relations_df["relation_type"] == HYPONYM_RELATION_TYPE]
    for row in hyponym_df.itertuples(index=False):
        children[int(row.src_con_id)].add(int(row.trg_con_id))
    return children


def collect_descendants(children_map: Dict[int, Set[int]], root_ids: Iterable[int]) -> Set[int]:
    """Collect all descendant concept IDs reachable from the given root IDs using the children map."""
    removed: Set[int] = set(int(x) for x in root_ids)
    queue: deque[int] = deque(int(x) for x in root_ids)

    while queue:
        current = queue.popleft()
        for child in children_map.get(current, set()):
            if child not in removed:
                removed.add(child)
                queue.append(child)

    return removed


def last_used_concept_ids(
    concepts_df: pd.DataFrame,
    glosses_df: pd.DataFrame,
    relations_df: pd.DataFrame,
    concept_pos_df: pd.DataFrame,
) -> int:
    """Collect biggest concept ID referenced anywhere to avoid collisions."""
    max_concept_id = max(
        concepts_df["id"].astype(int).max(),
        glosses_df["concept_id"].astype(int).max(),
        concept_pos_df["concept_id"].astype(int).max(),
        relations_df["src_con_id"].astype(int).max(),
        relations_df["trg_con_id"].astype(int).max(),
    )
    return max_concept_id

def add_pos_roots(
    concepts_df: pd.DataFrame,
    glosses_df: pd.DataFrame,
    relations_df: pd.DataFrame,
    concept_pos_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    concept_ids = set(concepts_df["id"].astype(int))
    root_labels = {label for _, label, _ in POS_ROOT_SPECS}

    last_concept_id = last_used_concept_ids(concepts_df, glosses_df, relations_df, concept_pos_df)
    next_concept_id = last_concept_id + 1
    next_relation_id = int(relations_df["id"].astype(int).max()) + 1

    # Keep optional provenance/kb convention aligned with the existing file.
    default_kb_id = 1
    default_provenance_id = 1

    added_edges = []
    added_concepts = []
    added_glosses = []
    added_pos = []

    created_roots: Dict[str, int] = {}

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    hyponym_df = relations_df[relations_df["relation_type"] == HYPONYM_RELATION_TYPE]
    hyponym_edges = [
        (int(src), int(trg))
        for src, trg in zip(hyponym_df["src_con_id"], hyponym_df["trg_con_id"])
    ]
    existing_edges = set(hyponym_edges)

    for pos_id, root_label, root_gloss in POS_ROOT_SPECS:
        raw_pos_concepts = {
            int(cid)
            for cid in concept_pos_df[concept_pos_df["pos"] == pos_id]["concept_id"].astype(int)
            if int(cid) in concept_ids
        }

        if not raw_pos_concepts:
            # skip when no concepts exist.
            continue

        root_id = next_concept_id
        next_concept_id += 1
        added_concepts.append(
            {
                "id": root_id,
                "creation_date": ts,
                "modification_date": ts,
                "is_local": "t",
                "label": root_label,
                "uk_id": root_id,
                "kb_id": 1,
            }
        )
        added_glosses.append({"concept_id": root_id, "gloss": root_gloss})
        added_pos.append({"pos": pos_id, "concept_id": root_id})
        concept_ids.add(root_id)

        pos_concepts = set(raw_pos_concepts)
        # Preserve noun hierarchy root: never attach entity under synthetic POS roots.
        pos_concepts.discard(ENTITY_CONCEPT_ID)

        # Attach only roots in the POS-internal hyponym graph so descendants keep
        # their original hierarchy and are not directly connected to the synthetic root.
        incoming_from_same_pos: Dict[int, int] = defaultdict(int)
        for src_id, trg_id in hyponym_edges:
            if src_id in pos_concepts and trg_id in pos_concepts:
                incoming_from_same_pos[trg_id] += 1

        attach_ids = sorted(
            cid for cid in pos_concepts if incoming_from_same_pos.get(cid, 0) == 0
        )

        created_roots[root_label] = root_id

        for cid in attach_ids:
            edge = (root_id, int(cid))
            if edge in existing_edges:
                continue
            added_edges.append(
                {
                    "id": next_relation_id,
                    "creation_date": ts,
                    "modification_date": ts,
                    "relation_type": HYPONYM_RELATION_TYPE,
                    "kb_id": default_kb_id,
                    "src_con_id": root_id,
                    "trg_con_id": int(cid),
                    "provenance_id": default_provenance_id,
                }
            )
            existing_edges.add(edge)
            next_relation_id += 1

    if added_concepts:
        concepts_df = pd.concat([concepts_df, pd.DataFrame(added_concepts)], ignore_index=True)
    if added_glosses:
        glosses_df = pd.concat([glosses_df, pd.DataFrame(added_glosses)], ignore_index=True)
    if added_pos:
        concept_pos_df = pd.concat([concept_pos_df, pd.DataFrame(added_pos)], ignore_index=True)
    if added_edges:
        relations_df = pd.concat([relations_df, pd.DataFrame(added_edges)], ignore_index=True)

    return concepts_df, glosses_df, relations_df, concept_pos_df, created_roots

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize concept hierarchy files.")
    parser.add_argument("--concepts", default="dataset/original/concepts.csv", help="Path to concepts.csv")
    parser.add_argument("--concept-glosses", default="dataset/original/concept_glosses.csv", help="Path to concept_glosses.csv")
    parser.add_argument("--concept-relations", default="dataset/original/concept_relations.csv", help="Path to concept_relations.csv")
    parser.add_argument("--concept-pos", default="dataset/original/concept_pos.csv", help="Path to concept_pos.csv")
    parser.add_argument(
        "--output-dir",
        default="dataset",
        help="Directory where normalized CSV files are written",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    concepts_path = Path(args.concepts)
    glosses_path = Path(args.concept_glosses)
    relations_path = Path(args.concept_relations)
    pos_path = Path(args.concept_pos)

    concepts_df = pd.read_csv(concepts_path)
    glosses_df = pd.read_csv(glosses_path)
    relations_df = pd.read_csv(relations_path)
    concept_pos_df = pd.read_csv(pos_path)

    required_rel_cols = {
        "id",
        "creation_date",
        "modification_date",
        "relation_type",
        "kb_id",
        "src_con_id",
        "trg_con_id",
        "provenance_id",
    }
    missing_rel = required_rel_cols - set(relations_df.columns)
    if missing_rel:
        raise ValueError(f"concept_relations.csv missing columns: {sorted(missing_rel)}")

    children_map = build_children_map(relations_df)
    remove_ids = collect_descendants(children_map, PRUNE_ROOT_IDS_BEFORE_ENTITY)

    # Remove concepts and all rows depending on them.
    concepts_df = concepts_df[~concepts_df["id"].astype(int).isin(remove_ids)].copy()
    glosses_df = glosses_df[~glosses_df["concept_id"].astype(int).isin(remove_ids)].copy()
    concept_pos_df = concept_pos_df[~concept_pos_df["concept_id"].astype(int).isin(remove_ids)].copy()
    relations_df = relations_df[
        ~relations_df["src_con_id"].astype(int).isin(remove_ids)
        & ~relations_df["trg_con_id"].astype(int).isin(remove_ids)
    ].copy()

    concepts_df, glosses_df, relations_df, concept_pos_df, created_roots = add_pos_roots(
        concepts_df=concepts_df,
        glosses_df=glosses_df,
        relations_df=relations_df,
        concept_pos_df=concept_pos_df,
    )

    # Reorder for cleaner diffs and predictable output.
    concepts_df = concepts_df.sort_values("id").reset_index(drop=True)
    glosses_df = glosses_df.sort_values("concept_id").reset_index(drop=True)
    concept_pos_df = concept_pos_df.sort_values(["concept_id", "pos"]).reset_index(drop=True)
    relations_df = relations_df.sort_values("id").reset_index(drop=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_concepts = out_dir / "concepts.csv"
    out_glosses = out_dir / "concept_glosses.csv"
    out_relations = out_dir / "concept_relations.csv"
    out_pos = out_dir / "concept_pos.csv"

    concepts_df.to_csv(out_concepts, index=False)
    glosses_df.to_csv(out_glosses, index=False)
    relations_df.to_csv(out_relations, index=False)
    concept_pos_df.to_csv(out_pos, index=False)

    print(f"Removed concepts (hyponym subtrees from listed roots): {len(remove_ids)}")
    print("Created/used POS roots:")
    for label, cid in created_roots.items():
        print(f"  - {label}: {cid}")
    print("Output files:")
    print(f"  - {out_concepts}")
    print(f"  - {out_glosses}")
    print(f"  - {out_relations}")
    print(f"  - {out_pos}")


if __name__ == "__main__":
    main()
