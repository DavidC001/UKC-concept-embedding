"""Main entry point for standalone WSD training."""

import argparse
from types import SimpleNamespace

import WSD.config as config
from WSD.experiments import run_baseline_experiments, run_experiments
from WSD.scripts.build_hierarchy import build_and_save_hierarchy
from WSD.training import train_unified


def _build_cfg_from_args(args):
    cfg = {
        "TRAIN_TSV": config.TRAIN_TSV,
        "EVAL_TSV": config.EVAL_TSV,
        "TEST_TSV": config.TEST_TSV,
        "ROTE_MODEL": config.ROTE_MODEL,
        "ENTITY_TO_ID": config.ENTITY_TO_ID,
        "CONCEPTS_CSV": config.CONCEPTS_CSV,
        "CONCEPT_REL_CSV": config.CONCEPT_REL_CSV,
        "HIERARCHY_CACHE_PATH": args.hierarchy_cache_path,
        "TRAINING_MODE": args.training_mode,
        "DATA_FILE": args.data_file,
        "ENCODER_MODEL": args.encoder_model,
        "ENCODER_TRAIN_MODE": args.encoder_train_mode,
        "ENCODER_TOKEN_POOLING": args.encoder_token_pooling,
        "MAX_SEQUENCE_LENGTH": args.max_length,
        "LORA_R": args.lora_r,
        "LORA_ALPHA": args.lora_alpha,
        "DROPOUT": args.dropout,
        "FORMATTER_PRESET": args.formatter_preset,
        "FORMATTER_TEMPLATE": args.formatter_template,
        "NUM_EPOCHS": args.epochs,
        "BATCH_SIZE": args.batch_size,
        "EVAL_BATCH_SIZE": args.eval_batch_size,
        "LEARNING_RATE": args.lr,
        "WEIGHT_DECAY": args.weight_decay,
        "DEVICE": args.device,
        "USE_EXTRA_NEGATIVE_SAMPLING": args.use_extra_negative_sampling,
        "NUM_EXTRA_NEGATIVES": args.num_extra_negatives,
        "USE_HIERARCHY_LOSS": args.use_hierarchy_loss,
        "HIERARCHY_LOSS_TYPE": args.hierarchy_loss_type,
        "WEIGHTED_HIERARCHY_ALPHA": args.weighted_hierarchy_alpha,
        "EVAL_USE_CANDIDATES": args.eval_use_candidates,
        "USE_EARLY_STOPPING": args.use_early_stopping,
        "EARLY_STOPPING_PATIENCE": args.early_stopping_patience,
        "EARLY_STOPPING_MIN_DELTA": args.early_stopping_min_delta,
        "EARLY_STOPPING_MONITOR": args.early_stopping_monitor,
        "EARLY_STOPPING_MODE": args.early_stopping_mode,
        "BASELINE": args.baseline,
        "BASELINE_TYPE": args.baseline_type,
        "SIMILARITY_METRIC": args.similarity_metric,
        "OUTPUT_DIR": args.output_dir,
        "WANDB_PROJECT": config.WANDB_PROJECT,
        "USE_WANDB": config.USE_WANDB,
        "EVAL_GOLD_STANDARD_DIR": config.EVAL_GOLD_STANDARD_DIR,
        "TEST_GOLD_STANDARD_DIR": config.TEST_GOLD_STANDARD_DIR,
        "SCORER_JAVA_PATH": config.SCORER_JAVA_PATH,
    }
    return SimpleNamespace(**cfg)


def build_parser():
    p = argparse.ArgumentParser(description="Standalone WSD trainer")
    p.add_argument(
        "--mode",
        choices=["train", "experiments", "baseline-experiments", "quick"],
        default="train",
        help="Execution mode",
    )

    p.add_argument("--training-mode", choices=["precomputed", "encoder"], default=config.TRAINING_MODE)
    p.add_argument("--data-file", default=config.DATA_FILE)
    p.add_argument("--encoder-model", default=config.ENCODER_MODEL)
    p.add_argument("--encoder-train-mode", choices=["full", "lora", "frozen"], default=config.ENCODER_TRAIN_MODE)
    p.add_argument(
        "--encoder-token-pooling",
        choices=["cls", "target_last_subword"],
        default=config.ENCODER_TOKEN_POOLING,
    )
    p.add_argument("--formatter-preset", default=config.FORMATTER_PRESET)
    p.add_argument("--formatter-template", default=config.FORMATTER_TEMPLATE)
    p.add_argument("--max-length", type=int, default=config.MAX_SEQUENCE_LENGTH)
    p.add_argument("--lora-r", type=int, default=config.LORA_R)
    p.add_argument("--lora-alpha", type=int, default=config.LORA_ALPHA)
    p.add_argument("--dropout", type=float, default=config.DROPOUT)
    p.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--eval-batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--use-extra-negative-sampling", action="store_true")
    p.add_argument("--num-extra-negatives", type=int, default=config.NUM_EXTRA_NEGATIVES)
    p.add_argument("--use-hierarchy-loss", action="store_true", dest="use_hierarchy_loss")
    p.add_argument("--no-hierarchy-loss", action="store_false", dest="use_hierarchy_loss")
    p.set_defaults(use_hierarchy_loss=config.USE_HIERARCHY_LOSS)
    p.add_argument(
        "--hierarchy-loss-type",
        choices=["factorized", "weighted_agg"],
        default=config.HIERARCHY_LOSS_TYPE,
    )
    p.add_argument("--weighted-hierarchy-alpha", type=float, default=config.WEIGHTED_HIERARCHY_ALPHA)
    p.add_argument("--eval-use-candidates", action="store_true", default=config.EVAL_USE_CANDIDATES)
    p.add_argument("--use-early-stopping", action="store_true", dest="use_early_stopping")
    p.add_argument("--no-early-stopping", action="store_false", dest="use_early_stopping")
    p.set_defaults(use_early_stopping=config.USE_EARLY_STOPPING)
    p.add_argument("--early-stopping-patience", type=int, default=config.EARLY_STOPPING_PATIENCE)
    p.add_argument("--early-stopping-min-delta", type=float, default=config.EARLY_STOPPING_MIN_DELTA)
    p.add_argument(
        "--early-stopping-monitor",
        choices=["eval_f1", "eval_loss"],
        default=config.EARLY_STOPPING_MONITOR,
    )
    p.add_argument(
        "--early-stopping-mode",
        choices=["max", "min"],
        default=config.EARLY_STOPPING_MODE,
    )
    p.add_argument("--baseline", action="store_true", default=config.BASELINE)
    p.add_argument("--baseline-type", choices=["linear", "random"], default=config.BASELINE_TYPE)
    p.add_argument("--similarity-metric", choices=["cosine", "dot_product", "l2_distance"], default=config.SIMILARITY_METRIC)
    p.add_argument("--hierarchy-cache-path", default=config.HIERARCHY_CACHE_PATH)
    p.add_argument("--output-dir", default=config.OUTPUT_DIR)
    p.add_argument("--skip-build-hierarchy", action="store_true", help="Skip automatic hierarchy cache build")
    p.add_argument("--experiments-limit", type=int, default=None, help="Limit number of experiment combinations")
    return p


def _cmd_train(args):
    cfg = _build_cfg_from_args(args)
    train_unified(cfg)


def _cmd_experiments(args):
    run_experiments(limit=args.experiments_limit)


def _cmd_baseline_experiments(args):
    run_baseline_experiments(limit=args.experiments_limit)


def _cmd_quick(args):
    args.epochs = 1
    args.batch_size = min(args.batch_size, 8)
    args.eval_batch_size = min(args.eval_batch_size, 8)
    _cmd_train(args)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.skip_build_hierarchy:
        build_and_save_hierarchy(args.hierarchy_cache_path)

    if args.mode == "train":
        _cmd_train(args)
    elif args.mode == "experiments":
        _cmd_experiments(args)
    elif args.mode == "baseline-experiments":
        _cmd_baseline_experiments(args)
    elif args.mode == "quick":
        _cmd_quick(args)


if __name__ == "__main__":
    main()
