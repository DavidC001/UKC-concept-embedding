import json
import os
import shutil
from datetime import datetime

import torch
from tqdm import tqdm


from WSD.datasets.dataset import (
    build_dataloaders_encoder,
    build_dataloaders_precomputed,
    load_all_splits,
    load_npz_data_or_build,
)
from WSD.utils.build_hierarchy import build_hierarchy
from WSD.utils.build_precomputed_embeddings import validate_precomputed_alignment
from WSD.common import move_batch_to_device, remap_labels_to_index_space
from WSD.concepts import create_bidirectional_mappings, load_rote_embeddings, load_uk_id_to_concept_id
from WSD.evaluation import (
    append_evaluation_report,
    compile_scorer_if_needed,
    evaluate_with_candidates,
    evaluate_with_gold_standard,
    log_scores_to_wandb,
    print_scores_table,
    save_predictions,
)
from WSD.model import EncoderBackbone, UnifiedConceptClassifier
from WSD.training.engine import (
    is_improved,
    resolve_monitor_value,
    run_epoch,
    run_validation_loss,
)
from WSD.config import WSDConfig, cfg_replace

def run_evaluate(cfg: WSDConfig):

    # Load all necessary data for evaluation
    concept_id_to_index, index_to_concept_id = create_bidirectional_mappings(cfg.ENTITY_TO_ID)
    uk_id_to_concept_id = load_uk_id_to_concept_id(cfg.CONCEPTS_CSV)
    concept_embeddings = load_rote_embeddings(cfg.ROTE_MODEL)
    num_concepts = int(concept_embeddings.shape[0])
    hierarchy_data = build_hierarchy(entity_to_id_csv=cfg.ENTITY_TO_ID, concept_rel_csv=cfg.CONCEPT_REL_CSV)

    split_map = load_all_splits(
        cfg.TRAIN_TSV,
        cfg.EVAL_TSV,
        cfg.TEST_TSV,
    )
    for split_name in ("train", "eval", "test"):
        split_map[split_name].labels = remap_labels_to_index_space(
            split_map[split_name].labels,
            concept_id_to_index=concept_id_to_index,
            num_concepts=num_concepts,
        )
    
    if cfg.TRAINING_MODE == "precomputed":
        npz_data = load_npz_data_or_build(
            cfg.DATA_FILE,
            train_tsv=cfg.TRAIN_TSV,
            eval_tsv=cfg.EVAL_TSV,
            test_tsv=cfg.TEST_TSV,
            model_name=getattr(cfg, "ENCODER_MODEL", "roberta-base"),
            max_length=cfg.MAX_SEQUENCE_LENGTH,
            device=cfg.DEVICE,
        )
        npz_data["train_labels"] = remap_labels_to_index_space(
            npz_data["train_labels"],
            concept_id_to_index=concept_id_to_index,
            num_concepts=num_concepts,
        )
        npz_data["eval_labels"] = remap_labels_to_index_space(
            npz_data["eval_labels"],
            concept_id_to_index=concept_id_to_index,
            num_concepts=num_concepts,
        )
        npz_data["test_labels"] = remap_labels_to_index_space(
            npz_data["test_labels"],
            concept_id_to_index=concept_id_to_index,
            num_concepts=num_concepts,
        )
        
        validate_precomputed_alignment(npz_data, split_map)
        loaders = build_dataloaders_precomputed(
            npz_data=npz_data,
            split_map=split_map,
            batch_size=cfg.BATCH_SIZE,
            eval_batch_size=cfg.EVAL_BATCH_SIZE,
        )
        input_dim = int(npz_data["train_embeddings"].shape[1])
        
    elif cfg.TRAINING_MODE == "encoder":
        encoder_backbone = EncoderBackbone(
            model_name=cfg.ENCODER_MODEL,
            train_mode=cfg.ENCODER_TRAIN_MODE,
            lora_r=cfg.LORA_R,
            lora_alpha=cfg.LORA_ALPHA,
            lora_dropout=cfg.DROPOUT,
        )
        loaders = build_dataloaders_encoder(
            split_map=split_map,
            tokenizer=encoder_backbone.tokenizer,
            batch_size=cfg.BATCH_SIZE,
            eval_batch_size=cfg.EVAL_BATCH_SIZE,
            max_length=cfg.MAX_SEQUENCE_LENGTH,
        )
        input_dim = encoder_backbone.hidden_size
        
    else:
        raise ValueError("TRAINING_MODE must be one of: precomputed, encoder")


    weighted_hierarchy_alpha = float(getattr(cfg, "WEIGHTED_HIERARCHY_ALPHA", 0.5))

    model = UnifiedConceptClassifier(
        concept_embeddings=concept_embeddings,
        input_dim=input_dim,
        hidden_dim=1024,
        output_dim=concept_embeddings.shape[1],
        dropout=cfg.DROPOUT,
        temperature=0.1,
        encoder_backbone=encoder_backbone,
        parent_index=hierarchy_data["parent_index"].to(cfg.DEVICE),
        hierarchy_data=hierarchy_data,
        weighted_hierarchy_alpha=weighted_hierarchy_alpha,
        freeze_concept_embeddings=getattr(cfg, "FREEZE_CONCEPT_EMBEDDINGS", True),
        baseline=getattr(cfg, "BASELINE", False),
        baseline_type=getattr(cfg, "BASELINE_TYPE", "linear"),
        similarity_metric=getattr(cfg, "SIMILARITY_METRIC", "cosine"),
        project_num_layers=getattr(cfg, "PROJECT_NUM_LAYERS", 1),
    ).to(cfg.DEVICE)

    if encoder_backbone is not None:
        model.encoder_backbone = model.encoder_backbone.to(cfg.DEVICE)

    # Load the trained model weights
    best_model_path = cfg.OUTPUT_DIR / "BEST" / cfg.RUN_NAME / "best_model.pt"
    if not best_model_path.exists():
        raise ValueError(f"Best model not found at {best_model_path}")
    state_dict = torch.load(best_model_path, map_location=cfg.DEVICE)
    model.load_state_dict(state_dict)
    
    # Evaluate on the eval and test sets
    scores = {}
    for split_name in ("eval", "test"):
        print(f"Evaluating on {split_name} set...")
        dataloader = loaders[split_name]
        scores[split_name] = evaluate_with_gold_standard(
            model=model,
            dataloader=dataloader,
            index_to_concept_id=index_to_concept_id,
            uk_id_to_concept_id=uk_id_to_concept_id,
            output_dir=cfg.OUTPUT_DIR / "BEST" / cfg.RUN_NAME,
            split_name=split_name,
        )

    return scores
    
    
def evaluate(cfg: WSDConfig):
    best_folder = cfg.OUTPUT_DIR / "BEST"
    # list all subdirs
    subdirs = [d for d in best_folder.iterdir() if d.is_dir()]
    if not subdirs:
        raise ValueError(f"No subdirectories found in {best_folder}")
    
    for subdir in subdirs:
        config_path = subdir / "config.json"
        
        cfg_dict = json.loads(config_path.read_text(encoding="utf-8"))
        run_cfg = cfg_replace(WSDConfig(), **cfg_dict)
        
        print(f"Evaluating run: {subdir.name}")
        scores = run_evaluate(run_cfg)
        
        # Save scores to a JSON file in the run directory
        scores_file = subdir / "evaluation_scores.json"
        with open(scores_file, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2)
        