"""Evaluation utilities."""

from .predictions import evaluate_with_candidates, save_predictions
from .scorer import (
    append_aggregate_report,
    append_evaluation_report,
    compile_scorer_if_needed,
    evaluate_with_gold_standard,
    load_id_to_pos,
    log_aggregate_scores_to_wandb,
    log_scores_to_wandb,
    print_aggregate_scores_table,
    print_scores_table,
)
from .pipeline import evaluate


__all__ = [
    "evaluate_with_candidates",
    "save_predictions",
    "compile_scorer_if_needed",
    "evaluate_with_gold_standard",
    "print_scores_table",
    "print_aggregate_scores_table",
    "append_evaluation_report",
    "append_aggregate_report",
    "log_scores_to_wandb",
    "log_aggregate_scores_to_wandb",
    "load_id_to_pos",
    "evaluate",
]
