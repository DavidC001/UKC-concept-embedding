"""Top-level unified training pipeline."""

import json
import os
import shutil
from datetime import datetime

import torch
from tqdm import tqdm

try:
    import wandb

    HAS_WANDB = True
except Exception:
    HAS_WANDB = False

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
from WSD.config import WSDConfig, cfg_asdict

def _evaluate(
    model,
    loader,
    device,
    concept_id_to_index,
    index_to_concept_id,
    output_file
):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            batch = move_batch_to_device(batch, device)
            scores = model(batch)
            
            candidates = batch["candidates"]
            
            preds = evaluate_with_candidates(
                scores=scores,
                candidates=candidates,
                labels=batch["labels"],
                ids=batch["id"],
                concept_id_to_index=concept_id_to_index,
                index_to_concept_id=index_to_concept_id,
            )
            
            all_preds.extend(preds)

    save_predictions(all_preds, output_file)
    return len(all_preds)


def train(cfg : WSDConfig):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_suffix = f"_baseline_{cfg.BASELINE_TYPE}" if getattr(cfg, "BASELINE", False) else ""
    run_name = f"{timestamp}_{cfg.TRAINING_MODE}_{cfg.LOSS_TYPE}{baseline_suffix}"
    run_dir = os.path.join(cfg.OUTPUT_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    report_file = os.path.join(run_dir, "evaluation_report.txt")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Training mode: {cfg.TRAINING_MODE}\n")
        f.write(f"Loss type: {cfg.LOSS_TYPE}\n")
        f.write(f"Baseline: {getattr(cfg, 'BASELINE', False)}\n")
        f.write(f"Baseline type: {getattr(cfg, 'BASELINE_TYPE', 'linear')}\n")
        f.write("\n")

    # save config for reproducibility
    config_save_path = os.path.join(run_dir, "config.json")
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(cfg_asdict(cfg), f, indent=4)
    print(f"Saved config: {config_save_path}")

    wandb_run = None
    if getattr(cfg, "USE_WANDB", False):
        if not HAS_WANDB:
            print("Warning: USE_WANDB=True but wandb is not installed. Install with: pip install wandb")
        else:
            wandb_run = wandb.init(
                project=getattr(cfg, "WANDB_PROJECT", "wsd-unified"),
                name=run_name,
                config=dict(vars(cfg)),
            )

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

    encoder_backbone = None
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


    loss_type = str(getattr(cfg, "LOSS_TYPE", "standard"))
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scorer_dir = compile_scorer_if_needed(cfg.SCORER_JAVA_PATH)

    early_stopping_enabled = bool(getattr(cfg, "USE_EARLY_STOPPING", False))
    early_stopping_patience = int(getattr(cfg, "EARLY_STOPPING_PATIENCE", 3))
    early_stopping_monitor = str(getattr(cfg, "EARLY_STOPPING_MONITOR", "eval_f1"))
    early_stopping_mode = str(getattr(cfg, "EARLY_STOPPING_MODE", "max"))

    best_monitor_value = None
    best_epoch = 0
    epochs_without_improvement = 0
    best_state_dict = None

    print("Training parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape}")

    n_eval = 0
    n_test = 0
    eval_file = ""
    test_file = ""

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        train_loss = run_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            device=cfg.DEVICE,
            concept_id_to_index=concept_id_to_index,
            loss_type=loss_type,
            move_batch_to_device=move_batch_to_device,
        )
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}")
        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": float(train_loss),
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                }
            )

        eval_file = os.path.join(run_dir, "eval_predictions.txt")
        test_file = os.path.join(run_dir, "test_predictions.txt")

        n_eval = _evaluate(
            model,
            loaders["eval"],
            cfg.DEVICE,
            concept_id_to_index,
            index_to_concept_id,
            eval_file,
        )
        n_test = _evaluate(
            model,
            loaders["test"],
            cfg.DEVICE,
            concept_id_to_index,
            index_to_concept_id,
            test_file,
        )

        eval_loss = run_validation_loss(
            model=model,
            loader=loaders["eval"],
            device=cfg.DEVICE,
            loss_type=loss_type,
            move_batch_to_device=move_batch_to_device,
        )

        eval_scores = evaluate_with_gold_standard(
            predictions_file=eval_file,
            gold_dir=cfg.EVAL_GOLD_STANDARD_DIR,
            scorer_dir=scorer_dir,
            concept_id_to_index=concept_id_to_index,
            uk_id_to_concept_id=uk_id_to_concept_id,
            tsv_file=cfg.EVAL_TSV,
        )
        test_scores = evaluate_with_gold_standard(
            predictions_file=test_file,
            gold_dir=cfg.TEST_GOLD_STANDARD_DIR,
            scorer_dir=scorer_dir,
            concept_id_to_index=concept_id_to_index,
            uk_id_to_concept_id=uk_id_to_concept_id,
            tsv_file=cfg.TEST_TSV,
        )

        if eval_scores:
            print_scores_table("Eval Set Coverage & Scores", eval_scores)
        if test_scores:
            print_scores_table("Test Set Coverage & Scores", test_scores)
        append_evaluation_report(report_file, epoch, eval_scores, test_scores)

        if wandb_run is not None:
            wandb.log(
                {
                    "eval/num_predictions": int(n_eval),
                    "test/num_predictions": int(n_test),
                    "eval/loss": float(eval_loss),
                }
            )
            log_scores_to_wandb("eval", eval_scores, wandb)
            log_scores_to_wandb("test", test_scores, wandb)

        monitor_value, monitor_name = resolve_monitor_value(
            monitor=early_stopping_monitor,
            eval_scores=eval_scores,
            eval_loss=eval_loss,
        )

        if is_improved(monitor_value, best_monitor_value, early_stopping_mode):
            best_monitor_value = monitor_value
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_model_path = os.path.join(run_dir, "best_model.pt")
            torch.save(best_state_dict, best_model_path)
            
            # save current evaluation report table in a separate file for the best epoch
            best_report_file = os.path.join(run_dir, "best_evaluation_report.txt")
            shutil.copy(report_file, best_report_file)
            
            print(f"New best checkpoint at epoch {epoch} ({monitor_name}={monitor_value:.4f})")
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement in {monitor_name} for {epochs_without_improvement} epoch(s) "
                f"(best epoch={best_epoch}, best={best_monitor_value:.4f})"
            )

        if wandb_run is not None:
            wandb.log(
                {
                    "early_stopping/monitor": float(monitor_value),
                    "early_stopping/best_monitor": float(best_monitor_value),
                    "early_stopping/best_epoch": int(best_epoch),
                    "early_stopping/epochs_without_improvement": int(epochs_without_improvement),
                }
            )

        if early_stopping_enabled and epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(patience={early_stopping_patience}, monitor={monitor_name})"
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model_path = os.path.join(run_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    print(f"Saved model: {model_path}")
    print(f"Saved eval predictions ({n_eval}): {eval_file}")
    print(f"Saved test predictions ({n_test}): {test_file}")
    print(f"Saved evaluation report: {report_file}")

    if wandb_run is not None:
        wandb.log(
            {
                "artifacts/model_path": model_path,
                "artifacts/eval_predictions": eval_file,
                "artifacts/test_predictions": test_file,
                "artifacts/best_epoch": int(best_epoch),
            }
        )
        wandb.finish()

    return run_dir
