"""Grid search runner for unified WSD."""

import itertools
from pathlib import Path

import WSD.config as config
from WSD.training import train

project_root = Path(__file__).resolve().parent.parent

HYPERPARAMETERS = {
    "NUM_EPOCHS": [10],
    "ENCODER_TRAIN_MODE": ["lora"],
    "ENCODER_MODEL": ["xlm-roberta-large"],
    "DATA_FILE": [str(project_root / "dataset/embs/sentence_embeddings_FacebookAI_xlm-roberta-large.npz")],
    "ROTE_MODEL": [str(project_root / "dataset/RotE/model.pt")],
    "DROPOUT": [0.1],
    "EARLY_STOPPING_PATIENCE": [3],
    "WEIGHTED_HIERARCHY_ALPHA": [0.5],
    
    "LORA_R": [8],
    "LORA_ALPHA": [16],
    
    "ENCODER_MODEL": ["xlm-roberta-large"],
    "FREEZE_CONCEPT_EMBEDDINGS": [True],
    "TRAINING_MODE": ["precomputed", "encoder"],
    
    "PROJECT_NUM_LAYERS": [1],
    "WEIGHT_DECAY": [1e-4],
    "LEARNING_RATE": [1e-4],
    "BATCH_SIZE": [16],
    "BASELINE": [False, True],
    "BASELINE_TYPE": ["random", "linear"],
    
    # "LOSS_TYPE": ["standard", "cosine", "hierarchy_factorized", "hierarchy_weighted_agg"],
    "LOSS_TYPE": ["standard"],
    
    "SIMILARITY_METRIC": ["cosine"]
}

def _apply_experiment_overrides(base_cfg, params):
    return config.cfg_replace(base_cfg, **params)


def _iter_param_grid(param_grid):
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _canonicalize_combo(params, base_cfg):
    """Canonicalize irrelevant knobs so equivalent configs can be deduplicated."""
    cfg_obj = _apply_experiment_overrides(base_cfg, params)
    cfg = config.cfg_asdict(cfg_obj)

    training_mode = str(cfg["TRAINING_MODE"])
    baseline = bool(cfg["BASELINE"])
    baseline_type = str(cfg.get("BASELINE_TYPE", "linear"))
    
    loss_type = str(cfg["LOSS_TYPE"])
    
    if loss_type != "hierarchy_weighted_agg":
        cfg["WEIGHTED_HIERARCHY_ALPHA"] = 0.5

    if loss_type == "cosine":
        cfg["SIMILARITY_METRIC"] = "cosine"

    if not baseline:
        cfg["BASELINE_TYPE"] = "off"
    elif baseline_type == "linear":
        cfg["SIMILARITY_METRIC"] = "canonical"
        cfg["FREEZE_CONCEPT_EMBEDDINGS"] = True

    if training_mode == "precomputed":
        cfg["ENCODER_TRAIN_MODE"] = "n/a"
        cfg["ENCODER_MODEL"] = "n/a"
    elif training_mode == "encoder":
        cfg["DATA_FILE"] = "n/a"

    return cfg


def run_experiments(param_grid=None, limit=None, base_overrides=None):
    if param_grid is None:
        param_grid = HYPERPARAMETERS

    base_cfg = config.WSDConfig()
    if base_overrides:
        base_cfg = config.cfg_replace(base_cfg, **base_overrides)

    # create the combinations from the grid and remove redundant cases
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
        cfg = _apply_experiment_overrides(base_cfg, params)
        train(cfg)

