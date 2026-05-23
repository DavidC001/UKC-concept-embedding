"""Configuration and CLI parsing for standalone WSD training."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Optional, Sequence

project_root = Path(__file__).resolve().parent.parent

@dataclass
class WSDConfig:
    PROJECT_ROOT: Path = project_root
    """Root directory of the project. Other paths are defined relative to this."""

    TRAIN_TSV: str = str(project_root / "dataset/train")
    """Path to the training data TSV file"""
    EVAL_TSV: str = str(project_root / "dataset/eval")
    """Path to the evaluation data TSV file"""
    TEST_TSV: str = str(project_root / "dataset/test")
    """Path to the test data TSV file"""

    NUM_RUNS: int = 1
    """Number of runs to perform in order to get average and standard deviation of results."""

    SEED: int = 42
    """Base random seed for multi-run experiments. Each run uses SEED + run_index."""

    ROTE_MODEL: str = str(project_root / "dataset/RotE/model.pt")
    """Path to the pre-trained RotE model checkpoint."""
    ENTITY_TO_ID: str = str(project_root / "dataset/RotE/entity_to_id.pickle")
    """Path to the entity_to_id mapping used by the RotE model."""
    CONCEPTS_CSV: str = str(project_root / "dataset/concepts.csv")
    """Path to the concepts.csv file containing concept information."""
    CONCEPT_REL_CSV: str = str(project_root / "dataset/concept_relations.csv")
    """Path to the concept_relations.csv file containing concept relation information."""

    TRAINING_MODE: str = "encoder" # can be "precomputed" or "encoder"
    """precomputed" means using precomputed sentence embeddings, while "encoder" means using a transformer encoder to compute them on the fly."""
    DATA_FILE: str = str(project_root / "dataset/embs/sentence_embeddings_FacebookAI_xlm-roberta-large.npz")
    """Path to the data file containing precomputed sentence embeddings (used if TRAINING_MODE is "precomputed")."""

    ENCODER_MODEL: str = "xlm-roberta-base"
    """Name or path of the transformer model to use as encoder (used if TRAINING_MODE is "encoder")."""
    ENCODER_TRAIN_MODE: str = "lora" # can be "full", "lora", or "frozen"
    """Determines which parameters of the encoder are trainable. "full" means all parameters are trainable, "lora" means only LoRA adapter parameters are trainable, and "frozen" means all encoder parameters are frozen."""
    MAX_SEQUENCE_LENGTH: int = 128
    """The maximum length of input sequences."""

    LORA_R: int = 8
    """The LoRA rank (r) hyperparameter"""
    LORA_ALPHA: int = 16
    """The LoRA alpha hyperparameter"""
    DROPOUT: float = 0.1
    """Dropout rate for the model."""

    NUM_EPOCHS: int = 1
    """Number of training epochs."""
    BATCH_SIZE: int = 32
    """Batch size for training."""
    EVAL_BATCH_SIZE: int = 32
    """Batch size for evaluation."""
    LEARNING_RATE: float = 1e-4
    """Learning rate for the optimizer."""
    WEIGHT_DECAY: float = 1e-5
    """Weight decay for the optimizer."""
    DEVICE: str = "cuda"
    """Device to use for training and evaluation (e.g., "cuda" or "cpu")."""

    FREEZE_CONCEPT_EMBEDDINGS: bool = True
    """Whether to freeze the concept embeddings during training."""
    
    LOSS_TYPE: str = "standard"
    """The type of loss function to use. Can be "standard", "cosine", "hierarchy_factorized", or "hierarchy_weighted_agg"."""
    WEIGHTED_HIERARCHY_ALPHA: float = 0.5
    """The alpha hyperparameter for weighted hierarchy loss."""
    EVAL_USE_CANDIDATES: bool = True
    """Whether to use candidates during evaluation."""

    USE_EARLY_STOPPING: bool = True
    """Whether to use early stopping during training."""
    EARLY_STOPPING_PATIENCE: int = 3
    """Number of epochs to wait before stopping training."""
    EARLY_STOPPING_MONITOR: str = "eval_f1"
    """Quantity to be monitored for early stopping."""
    EARLY_STOPPING_MODE: str = "max"
    """Either 'min' or 'max'."""

    BASELINE: bool = False
    """Whether to use a baseline model."""
    BASELINE_TYPE: str = "linear"
    """Type of baseline model to use."""
    SIMILARITY_METRIC: str = "cosine"
    """Metric to use for computing similarity."""

    PROJECT_NUM_LAYERS: int = 1
    """Number of layers in the projection head (if using a projection head)."""
    HIDDEN_DIM: int = 1024
    """Hidden dimension size for the projection head (if using a projection head)."""

    OUTPUT_DIR: str = str(Path(__file__).resolve().parent.parent / "WSD/outputs")
    """Directory where model checkpoints and logs will be saved."""
    WANDB_PROJECT: str = "wsd-unified"
    """Weights & Biases project name for logging."""
    USE_WANDB: bool = True
    """Whether to use Weights & Biases for logging."""

    EVAL_GOLD_STANDARD_DIR: str = str(Path(__file__).resolve().parent.parent / "dataset/eval-gold-standard")
    """Directory containing the evaluation gold standard data."""
    TEST_GOLD_STANDARD_DIR: str = str(Path(__file__).resolve().parent.parent / "dataset/test-gold-standard")
    """Directory containing the test gold standard data."""
    SCORER_JAVA_PATH: str = str(Path(__file__).resolve().parent.parent / "WSD/Scorer.java")
    """Path to the Java scorer script."""

    MODE: str = "train"
    """Execution mode. Can be "train", "experiments", "quick" or "eval"."""
    
    EXPERIMENTS_LIMIT: Optional[int] = None
    """Optional limit on the number of experiment configurations to run (used in experiments mode)."""


def build_parser() -> argparse.ArgumentParser:
    defaults : WSDConfig = WSDConfig()
    
    parser = argparse.ArgumentParser(description="Standalone WSD trainer")
    parser.add_argument(
        "--mode",
        choices=["train", "experiments", "quick"],
        default=defaults.MODE,
        help="Execution mode",
    )
    
    parser.add_argument("--num-runs", type=int, default=defaults.NUM_RUNS)
    parser.add_argument("--seed", type=int, default=defaults.SEED)

    parser.add_argument("--training-mode", choices=["precomputed", "encoder"], default=defaults.TRAINING_MODE)
    parser.add_argument("--data-file", default=defaults.DATA_FILE)
    parser.add_argument("--encoder-model", default=defaults.ENCODER_MODEL)
    parser.add_argument("--encoder-train-mode", choices=["full", "lora", "frozen"], default=defaults.ENCODER_TRAIN_MODE)
    
    parser.add_argument("--max-length", type=int, default=defaults.MAX_SEQUENCE_LENGTH)
    parser.add_argument("--lora-r", type=int, default=defaults.LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=defaults.LORA_ALPHA)
    parser.add_argument("--dropout", type=float, default=defaults.DROPOUT)
    parser.add_argument("--epochs", type=int, default=defaults.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=defaults.BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=defaults.EVAL_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=defaults.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=defaults.WEIGHT_DECAY)
    parser.add_argument("--device", default=defaults.DEVICE)

    parser.add_argument("--loss-type", choices=["standard", "cosine", "hierarchy_factorized", "hierarchy_weighted_agg"], default=defaults.LOSS_TYPE)
    parser.add_argument("--weighted-hierarchy-alpha", type=float, default=defaults.WEIGHTED_HIERARCHY_ALPHA)
    parser.add_argument("--eval-use-candidates", action="store_true", default=defaults.EVAL_USE_CANDIDATES)

    parser.add_argument("--use-early-stopping", action="store_true", dest="use_early_stopping")
    parser.add_argument("--no-early-stopping", action="store_false", dest="use_early_stopping")
    parser.set_defaults(use_early_stopping=defaults.USE_EARLY_STOPPING)
    parser.add_argument("--early-stopping-patience", type=int, default=defaults.EARLY_STOPPING_PATIENCE)
    parser.add_argument(
        "--early-stopping-monitor",
        choices=["eval_f1", "eval_loss"],
        default=defaults.EARLY_STOPPING_MONITOR,
    )
    parser.add_argument(
        "--early-stopping-mode",
        choices=["max", "min"],
        default=defaults.EARLY_STOPPING_MODE,
    )

    parser.add_argument("--baseline", action="store_true", default=defaults.BASELINE)
    parser.add_argument("--baseline-type", choices=["linear", "random"], default=defaults.BASELINE_TYPE)
    parser.add_argument("--similarity-metric", choices=["cosine", "dot_product", "l2_distance"], default=defaults.SIMILARITY_METRIC)

    parser.add_argument("--output-dir", default=defaults.OUTPUT_DIR)
    parser.add_argument("--experiments-limit", type=int, default=defaults.EXPERIMENTS_LIMIT)

    return parser


def config_from_parsed_args(args: argparse.Namespace) -> WSDConfig:
    defaults : WSDConfig = WSDConfig()
    
    return replace(
        defaults,
        NUM_RUNS=args.num_runs,
        SEED=args.seed,
        TRAINING_MODE=args.training_mode,
        DATA_FILE=args.data_file,
        ENCODER_MODEL=args.encoder_model,
        ENCODER_TRAIN_MODE=args.encoder_train_mode,
        MAX_SEQUENCE_LENGTH=args.max_length,
        LORA_R=args.lora_r,
        LORA_ALPHA=args.lora_alpha,
        DROPOUT=args.dropout,
        NUM_EPOCHS=args.epochs,
        BATCH_SIZE=args.batch_size,
        EVAL_BATCH_SIZE=args.eval_batch_size,
        LEARNING_RATE=args.lr,
        WEIGHT_DECAY=args.weight_decay,
        DEVICE=args.device,
        LOSS_TYPE=args.loss_type,
        WEIGHTED_HIERARCHY_ALPHA=args.weighted_hierarchy_alpha,
        EVAL_USE_CANDIDATES=args.eval_use_candidates,
        USE_EARLY_STOPPING=args.use_early_stopping,
        EARLY_STOPPING_PATIENCE=args.early_stopping_patience,
        EARLY_STOPPING_MONITOR=args.early_stopping_monitor,
        EARLY_STOPPING_MODE=args.early_stopping_mode,
        BASELINE=args.baseline,
        BASELINE_TYPE=args.baseline_type,
        SIMILARITY_METRIC=args.similarity_metric,
        OUTPUT_DIR=args.output_dir,
        MODE=args.mode,
        EXPERIMENTS_LIMIT=args.experiments_limit,
    )


def parse_cli_config(argv: Optional[Sequence[str]] = None) -> WSDConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    return config_from_parsed_args(args)


def cfg_replace(cfg: WSDConfig, **overrides: Any) -> WSDConfig:
    return replace(cfg, **overrides)


def cfg_asdict(cfg: WSDConfig) -> dict[str, Any]:
    dict = asdict(cfg)
    # convert Paths to strings for easier logging and serialization
    for k, v in dict.items():
        if isinstance(v, Path):
            dict[k] = str(v)
    return dict
