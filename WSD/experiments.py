"""Grid search runner for unified WSD."""

import itertools
from types import SimpleNamespace

import WSD.config as config
from WSD.training import train_unified


HYPERPARAMETERS = {
    "NUM_EPOCHS": [10],
    "ENCODER_TRAIN_MODE": ["lora"],
    "ENCODER_MODEL": ["xlm-roberta-large"],
    "DATA_FILE": [str(config.PROJECT_ROOT / "dataset/embs/sentence_embeddings_FacebookAI_xlm-roberta-large.npz")],
    "ROTE_MODEL": [str(config.PROJECT_ROOT / "dataset/RotE/model.pt")],
    # not being used as now hier loss uses weighted agg 
    "USE_EXTRA_NEGATIVE_SAMPLING": [True],
    "NUM_EXTRA_NEGATIVES": [256],
    "DROPOUT": [0.1],
    "EARLY_STOPPING_PATIENCE": [3],
    "WEIGHTED_HIERARCHY_ALPHA": [0.5],
    
    
    "PROJECT_NUM_LAYERS": [1],
    "WEIGHT_DECAY": [1e-4],
    "LEARNING_RATE": [1e-4],
    "BATCH_SIZE": [16],
    "BASELINE": [False,True],
    "BASELINE_TYPE": ["linear"],
    # "TRAINING_MODE": ["precomputed", "encoder"],
    "TRAINING_MODE": ["precomputed"],
    "HIERARCHY_LOSS_TYPE": ["factorized"],    
    "USE_HIERARCHY_LOSS": [True],
    "SIMILARITY_METRIC": ["cosine"],
}


def _base_config_dict():
    return {
        "TRAIN_TSV": config.TRAIN_TSV,
        "EVAL_TSV": config.EVAL_TSV,
        "TEST_TSV": config.TEST_TSV,
        "ROTE_MODEL": config.ROTE_MODEL,
        "ENTITY_TO_ID": config.ENTITY_TO_ID,
        "CONCEPTS_CSV": config.CONCEPTS_CSV,
        "CONCEPT_REL_CSV": config.CONCEPT_REL_CSV,
        "HIERARCHY_CACHE_PATH": config.HIERARCHY_CACHE_PATH,
        "TRAINING_MODE": config.TRAINING_MODE,
        "DATA_FILE": config.DATA_FILE,
        "ENCODER_MODEL": config.ENCODER_MODEL,
        "ENCODER_TRAIN_MODE": config.ENCODER_TRAIN_MODE,
        "ENCODER_TOKEN_POOLING": config.ENCODER_TOKEN_POOLING,
        "MAX_SEQUENCE_LENGTH": config.MAX_SEQUENCE_LENGTH,
        "LORA_R": config.LORA_R,
        "LORA_ALPHA": config.LORA_ALPHA,
        "DROPOUT": config.DROPOUT,
        "FORMATTER_PRESET": config.FORMATTER_PRESET,
        "FORMATTER_TEMPLATE": config.FORMATTER_TEMPLATE,
        "NUM_EPOCHS": config.NUM_EPOCHS,
        "BATCH_SIZE": config.BATCH_SIZE,
        "EVAL_BATCH_SIZE": config.EVAL_BATCH_SIZE,
        "LEARNING_RATE": config.LEARNING_RATE,
        "WEIGHT_DECAY": config.WEIGHT_DECAY,
        "DEVICE": config.DEVICE,
        "USE_EXTRA_NEGATIVE_SAMPLING": config.USE_EXTRA_NEGATIVE_SAMPLING,
        "NUM_EXTRA_NEGATIVES": config.NUM_EXTRA_NEGATIVES,
        "USE_HIERARCHY_LOSS": config.USE_HIERARCHY_LOSS,
        "HIERARCHY_LOSS_TYPE": config.HIERARCHY_LOSS_TYPE,
        "WEIGHTED_HIERARCHY_ALPHA": config.WEIGHTED_HIERARCHY_ALPHA,
        "USE_EARLY_STOPPING": config.USE_EARLY_STOPPING,
        "EARLY_STOPPING_PATIENCE": config.EARLY_STOPPING_PATIENCE,
        "EARLY_STOPPING_MIN_DELTA": config.EARLY_STOPPING_MIN_DELTA,
        "EARLY_STOPPING_MONITOR": config.EARLY_STOPPING_MONITOR,
        "EARLY_STOPPING_MODE": config.EARLY_STOPPING_MODE,
        "BASELINE": config.BASELINE,
        "BASELINE_TYPE": config.BASELINE_TYPE,
        "SIMILARITY_METRIC": config.SIMILARITY_METRIC,
        "OUTPUT_DIR": config.OUTPUT_DIR,
        "WANDB_PROJECT": config.WANDB_PROJECT,
        "USE_WANDB": config.USE_WANDB,
        "EVAL_GOLD_STANDARD_DIR": config.EVAL_GOLD_STANDARD_DIR,
        "TEST_GOLD_STANDARD_DIR": config.TEST_GOLD_STANDARD_DIR,
        "SCORER_JAVA_PATH": config.SCORER_JAVA_PATH,
        "PROJECT_NUM_LAYERS": config.PROJECT_NUM_LAYERS,
    }


def _apply_experiment_overrides(base_cfg, params):
    cfg = dict(base_cfg)
    cfg.update(params)
    return cfg


def _iter_param_grid(param_grid):
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _canonicalize_combo(params, base_cfg):
    """Canonicalize irrelevant knobs so equivalent configs can be deduplicated."""
    cfg = dict(base_cfg)
    cfg.update(params)

    training_mode = str(cfg["TRAINING_MODE"])
    baseline = bool(cfg["BASELINE"])
    baseline_type = str(cfg.get("BASELINE_TYPE", "linear"))
    use_hierarchy_loss = bool(cfg["USE_HIERARCHY_LOSS"])
    hierarchy_loss_type = str(cfg["HIERARCHY_LOSS_TYPE"])

    if not use_hierarchy_loss or hierarchy_loss_type != "weighted_agg":
        cfg["USE_EXTRA_NEGATIVE_SAMPLING"] = False
        cfg["HIERARCHY_LOSS_TYPE"] = "off"
        cfg["WEIGHTED_HIERARCHY_ALPHA"] = "off"

    if not baseline:
        cfg["BASELINE_TYPE"] = "off"
    elif baseline_type == "linear":
        cfg["SIMILARITY_METRIC"] = "canonical"

    if training_mode == "precomputed":
        cfg["ENCODER_TRAIN_MODE"] = "n/a"
        cfg["ENCODER_MODEL"] = "n/a"
        cfg["ENCODER_TOKEN_POOLING"] = "n/a"
    elif training_mode == "encoder":
        cfg["DATA_FILE"] = "n/a"

    return cfg


def run_experiments(param_grid=None, limit=None, base_overrides=None):
    if param_grid is None:
        param_grid = HYPERPARAMETERS

    base_cfg = _base_config_dict()
    if base_overrides:
        base_cfg.update(base_overrides)

    all_combos = list(_iter_param_grid(param_grid))
    combos = []
    seen = set()
    for p in all_combos:
        canonical = _canonicalize_combo(p, base_cfg)
        signature = tuple(sorted(canonical.items()))
        if signature in seen:
            continue
        seen.add(signature)
        combos.append(p)

    skipped = len(all_combos) - len(combos)
    if skipped > 0:
        print(f"Skipping {skipped} redundant combinations based on constraints")

    if limit is not None:
        combos = combos[: int(limit)]

    print(f"Running {len(combos)} experiments")
    for i, params in enumerate(combos, start=1):
        print(f"\n[{i}/{len(combos)}] params={params}")
        cfg_dict = _apply_experiment_overrides(base_cfg, params)
        cfg = SimpleNamespace(**cfg_dict)
        train_unified(cfg)


def run_limited_grid():
    param_grid = {
        "TRAINING_MODE": [config.TRAINING_MODE],
        "ENCODER_TRAIN_MODE": [config.ENCODER_TRAIN_MODE],
        "LEARNING_RATE": [1e-4],
        "WEIGHT_DECAY": [1e-3],
        "BATCH_SIZE": [32],
        "USE_HIERARCHY_LOSS": [True],
        "BASELINE": [False],
        "BASELINE_TYPE": ["linear"],
    }
    run_experiments(param_grid=param_grid)


def run_baseline_experiments(limit=None):
    """Run baseline variants requested for precomputed and LoRA modes."""
    param_grid = {
        "TRAINING_MODE": ["precomputed", "encoder"],
        "ENCODER_TRAIN_MODE": ["lora"],
        "BASELINE": [True],
        "BASELINE_TYPE": ["linear", "random"],
        "USE_HIERARCHY_LOSS": [False],
        "SIMILARITY_METRIC": ["cosine", "dot_product", "l2_distance"],
        "LEARNING_RATE": [config.LEARNING_RATE],
        "WEIGHT_DECAY": [config.WEIGHT_DECAY],
        "BATCH_SIZE": [config.BATCH_SIZE],
        "NUM_EPOCHS": [config.NUM_EPOCHS],
    }
    run_experiments(param_grid=param_grid, limit=limit)
