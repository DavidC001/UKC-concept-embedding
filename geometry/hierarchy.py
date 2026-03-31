"""Hierarchy construction and concept selection helpers."""

from typing import Dict, List

import networkx as nx
import pandas as pd


def load_concepts(concepts_csv: str) -> pd.DataFrame:
    return pd.read_csv(concepts_csv, usecols=["id", "label"])


def build_hierarchy_graph(relations_csv: str, relation_type: int = 20) -> nx.DiGraph:
    rel = pd.read_csv(relations_csv, usecols=["relation_type", "src_con_id", "trg_con_id"])
    rel = rel[rel["relation_type"] == relation_type]

    # relation_type=20 is has_hyponym: src is parent, trg is child.
    g = nx.DiGraph()
    g.add_edges_from(rel[["src_con_id", "trg_con_id"]].itertuples(index=False, name=None))
    return g


def graph_roots(g: nx.DiGraph) -> List[int]:
    return [n for n in g.nodes if g.in_degree(n) == 0]


def find_root_by_label(concepts_df: pd.DataFrame, label: str) -> int:
    mask = concepts_df["label"].str.lower() == label.lower()
    rows = concepts_df[mask]
    if rows.empty:
        raise ValueError(f"Could not find concept label '{label}' in concepts.csv")
    return int(rows.sort_values("id").iloc[0]["id"])


def validate_concept_id(concepts_df: pd.DataFrame, concept_id: int, expected_label: str = "") -> int:
    rows = concepts_df[concepts_df["id"].astype(int) == int(concept_id)]
    if rows.empty:
        raise ValueError(f"Concept id {concept_id} not found in concepts.csv")

    if expected_label:
        label = str(rows.iloc[0]["label"]).lower()
        if label != expected_label.lower():
            raise ValueError(
                f"Concept id {concept_id} has label '{label}', expected '{expected_label}'."
            )

    return int(concept_id)


def build_node_sets_for_directions(
    g: nx.DiGraph,
    node: int,
    min_size: int,
) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for n in nx.descendants(g, node) | {node}:
        members = list(nx.descendants(g, n) | {n})
        if len(members) >= min_size:
            out[n] = members
    return out


def descendant_indices(root: int, hgraph: nx.DiGraph, ent2idx: Dict[str, int]) -> List[int]:
    nodes = list(nx.descendants(hgraph, root) | {root})
    return [ent2idx[str(n)] for n in nodes if str(n) in ent2idx]


def find_child_by_label(hgraph: nx.DiGraph, concepts_df: pd.DataFrame, parent_id: int, label: str) -> int:
    id_to_label = dict(zip(concepts_df["id"].astype(int), concepts_df["label"].str.lower()))
    for child in hgraph.successors(parent_id):
        if id_to_label.get(int(child), "") == label.lower():
            return int(child)

    rows = concepts_df[concepts_df["label"].str.lower() == label.lower()]
    if rows.empty:
        raise ValueError(f"Could not find concept '{label}'")
    return int(rows.sort_values("id").iloc[0]["id"])
