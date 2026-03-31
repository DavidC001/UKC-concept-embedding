"""Main training script with wandb logging."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import subprocess
from datetime import datetime
from tqdm import tqdm
import re
import random

import config
from dataset import WSDDataset, load_npz_data, load_metadata, collate_wsd_batch
from models import ConceptClassifier
from utils import (load_rote_embeddings, create_bidirectional_mappings, 
                   load_uk_id_to_concept_id, evaluate_with_candidates, 
                   save_predictions, compute_metrics, create_concept_mask, map_labels_to_indices)
from types import SimpleNamespace

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed. Install with: pip install wandb")


def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=True)
    for batch in pbar:
        embeddings = batch['embedding'].to(device)
        labels = batch['labels'].to(device)
        valid_mask = labels != -100
        if int(valid_mask.sum().item()) == 0:
            continue

        logits = model(embeddings)
        loss = loss_fn(logits[valid_mask], labels[valid_mask])
        
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # for name, p in model.named_parameters():
        #     if p.requires_grad:
        #         print(name, p.grad.abs().mean().item())
        #         break
                
        total_loss += loss.item() * int(valid_mask.sum().item())
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader.dataset)


def evaluate(model, eval_loader, loss_fn, device, concept_id_to_index, index_to_concept_id):
    """Evaluate on eval/test set."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    
    pbar = tqdm(eval_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in pbar:
            embeddings = batch['embedding'].to(device)
            labels = batch['labels'].to(device)
            candidates = batch['candidates']
            ids = batch['id']
            valid_mask = labels != -100
            
            logits = model(embeddings)
            if int(valid_mask.sum().item()) > 0:
                loss = loss_fn(logits[valid_mask], labels[valid_mask])
                total_loss += loss.item() * int(valid_mask.sum().item())
            
            # Get predictions with candidate limiting
            preds = evaluate_with_candidates(logits, candidates, labels, ids, 
                                            concept_id_to_index, index_to_concept_id, device)
            all_predictions.extend(preds)
    
    avg_loss = total_loss / len(eval_loader.dataset) if len(eval_loader.dataset) > 0 else 0
    return avg_loss, all_predictions


def compile_and_run_scorer():
    """Compile the Scorer.java file if needed."""
    scorer_java = 'Scorer.java'
    scorer_class = 'Scorer.class'
    
    if not os.path.exists(scorer_class):
        print("Compiling Scorer.java...")
        result = subprocess.run(['javac', scorer_java], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: Failed to compile Scorer.java: {result.stderr}")
            return False
    return True


def evaluate_with_gold_standard(predictions_file, eval_dir, run_name, uk_id_to_concept_id):
    """Evaluate predictions against gold standard using Scorer.java.
    
    Args:
        predictions_file: path to predictions file
        eval_dir: directory containing gold standard files
        run_name: name for logging purposes
    
    Returns:
        dict with evaluation metrics
    """
    gold_standards = [
        ('ALL', 'UKC.gold.key.txt'),
        ('Seen Only', 'UKC.in.test.gold.key.txt'),
        ('Unseen Only', 'UKC.out.test.gold.key.txt'),
        ('Single Candidate Only', 'UKC.single.test.gold.key.txt'),
        ('Multiple Candidates Only', 'UKC.multi.test.gold.key.txt'),
        ('Multiple Candidates Seen Only', 'UKC.multi.in.test.gold.key.txt'),
        ('Multiple Candidates Unseen Only', 'UKC.multi.out.test.gold.key.txt')
    ]
    
    # Read predicted IDs
    predicted_ids = set()
    with open(predictions_file, 'r') as pf:
        for line in pf:
            parts = line.strip().split()
            if not parts:
                continue
            predicted_ids.add(parts[0])

    # load concept mapping to check gold concept existence
    concept_id_to_index, _ = create_bidirectional_mappings()

    # prepare filtered gold output directory
    filtered_dir = os.path.join(os.path.dirname(predictions_file), 'filtered_gold')
    os.makedirs(filtered_dir, exist_ok=True)

    results = {}

    for title, gold_file in gold_standards:
        gold_path = os.path.join(eval_dir, gold_file)
        if not os.path.exists(gold_path):
            continue

        filtered_lines = []
        total_gold_lines = 0
        with open(gold_path, 'r') as gf:
            for line in gf:
                line = line.strip()
                if not line:
                    continue
                total_gold_lines += 1
                parts = line.split()
                gid = parts[0]
                gold_concepts = []
                for tok in parts[1:]:
                    try:
                        uk_id = int(tok)
                        concept_id = uk_id_to_concept_id.get(uk_id)
                        if concept_id is not None:
                            gold_concepts.append(int(concept_id))
                    except Exception:
                        # skip non-int tokens
                        pass

                # keep only if we predicted for this id and at least one gold concept exists in embedding
                if gid in predicted_ids and any((gc in concept_id_to_index) for gc in gold_concepts):
                    converted_line = " ".join([gid] + [str(gc) for gc in gold_concepts])
                    filtered_lines.append(converted_line)

        filtered_path = os.path.join(filtered_dir, gold_file)
        with open(filtered_path, 'w') as outf:
            for l in filtered_lines:
                outf.write(l + '\n')

        effective_size = len(filtered_lines)
        coverage_gold = (effective_size / total_gold_lines * 100.0) if total_gold_lines > 0 else 0.0
        coverage_pred = (effective_size / len(predicted_ids) * 100.0) if len(predicted_ids) > 0 else 0.0

        # run scorer on filtered gold
        try:
            result = subprocess.run(
                ['java', 'Scorer', filtered_path, predictions_file],
                capture_output=True, text=True, timeout=30
            )

            metrics = {}
            if result.returncode == 0:
                output = result.stdout
                p_match = re.search(r'P=\s*([\d.]+)%', output)
                r_match = re.search(r'R=\s*([\d.]+)%', output)
                f1_match = re.search(r'F1=\s*([\d.]+)%', output)
                # print("p_match:", p_match)
                # print("r_match:", r_match)
                # print("f1_match:", f1_match)
                if p_match and r_match and f1_match:
                    metrics = {
                        'precision': float(p_match.group(1)),
                        'recall': float(r_match.group(1)),
                        'f1': float(f1_match.group(1)),
                        'effective_size': effective_size,
                        'coverage_gold_pct': coverage_gold,
                        'coverage_predicted_pct': coverage_pred,
                        'gold_total': total_gold_lines
                    }
            results[title] = metrics
        except Exception as e:
            print(f"Warning: Failed to score {title}: {e}")

    return results


def set_all_seeds(seed):
    """Set all relevant random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic flags improve reproducibility across repeated runs.
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate_data_alignment(data, metadata):
    """Fail fast if NPZ arrays and TSV metadata are not aligned."""
    split_to_keys = {
        'train': ('train_embeddings', 'train_labels'),
        'eval': ('eval_embeddings', 'eval_labels'),
        'test': ('test_embeddings', 'test_labels'),
    }

    errors = []
    for split, (emb_k, lbl_k) in split_to_keys.items():
        emb_n = len(data[emb_k])
        lbl_n = len(data[lbl_k])
        ids, cands, _ = metadata[split]
        meta_n = len(ids)

        if emb_n != lbl_n:
            errors.append(
                f"[{split}] NPZ mismatch: {emb_k}={emb_n} but {lbl_k}={lbl_n}"
            )

        if emb_n != meta_n or lbl_n != meta_n:
            errors.append(
                f"[{split}] NPZ/TSV mismatch: embeddings={emb_n}, labels={lbl_n}, metadata={meta_n}"
            )

        if getattr(data[lbl_k], 'ndim', 1) != 1:
            errors.append(
                f"[{split}] expected 1D scalar labels, got shape={data[lbl_k].shape}"
            )

        if len(cands) != meta_n:
            errors.append(
                f"[{split}] metadata mismatch: candidates={len(cands)} but ids={meta_n}"
            )

    if errors:
        joined = "\n  - " + "\n  - ".join(errors)
        raise ValueError(
            "Data alignment check failed. Your embeddings/labels NPZ is not aligned with TSV splits.\n"
            "Regenerate embeddings from the same split files before training.\n"
            f"Details:{joined}"
        )


def train(config_dict=None, seed=None, return_metrics=False):
    """Main training function."""
    
    # Create timestamp for this run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure all required keys have defaults
    if config_dict is None:
        config_dict = {}
    
    default_config = {
        'epochs': config.NUM_EPOCHS,
        'batch_size': config.BATCH_SIZE,
        'lr': config.LEARNING_RATE,
        'weight_decay': config.WEIGHT_DECAY,
        'temperature': config.TEMPERATURE,
        'momentum': config.MOMENTUM,
        'optim': config.OPTIM,
        'do': config.DROPOUT,
        'baseline': False,
        'baseline_type': 'linear',
        'encoder': 'mmbert',
        'seed': None,
    }
    
    # Merge with provided config
    for key in default_config:
        if key not in config_dict:
            config_dict[key] = default_config[key]

    if seed is not None:
        config_dict['seed'] = seed
    
    cf = SimpleNamespace(**config_dict)

    if cf.seed is not None:
        set_all_seeds(cf.seed)

    run_name = f"{run_timestamp}_lr{cf.lr}_wd{cf.weight_decay}_bt{cf.batch_size}_t{cf.temperature}_m{cf.momentum}_{cf.optim}_{cf.encoder}"

    if cf.seed is not None:
        run_name += f"_seed{cf.seed}"
    
    run_name += "_1LAYER_cos"
    
    if cf.baseline:
        run_name += "_baseline_" + cf.baseline_type
    
    # Create timestamped output directory
    output_base = config.OUTPUT_DIR
    run_output_dir = os.path.join(output_base, run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    
    print(f"Output directory: {run_output_dir}")
    
    # Initialize wandb with timestamp in run name
    if HAS_WANDB and config.USE_WANDB:
        wandb.init(project=config.WANDB_PROJECT, name=run_name, config=cf.__dict__)
    
    device = torch.device(config.DEVICE)
    
    # Compile Scorer if needed
    compile_and_run_scorer()
    
    # Load mappings
    print("Loading concept mappings...")
    concept_id_to_index, index_to_concept_id = create_bidirectional_mappings()
    uk_id_to_concept_id = load_uk_id_to_concept_id()
    
    print("Loading data...")
    data = load_npz_data(config.DATA_FILE)
    metadata = load_metadata(config.TRAIN_TSV, config.EVAL_TSV, config.TEST_TSV)
    validate_data_alignment(data, metadata)
    
    # Map labels from concept IDs to embedding matrix indices
    print("Mapping labels to embedding indices...")
    data['train_labels'] = map_labels_to_indices(data['train_labels'], concept_id_to_index)
    data['eval_labels'] = map_labels_to_indices(data['eval_labels'], concept_id_to_index)
    data['test_labels'] = map_labels_to_indices(data['test_labels'], concept_id_to_index)
    
    train_ids, train_cands, train_ans = metadata['train']
    eval_ids, eval_cands, eval_ans = metadata['eval']
    test_ids, test_cands, test_ans = metadata['test']
    
    # Limit training samples if specified
    if config.LIMIT_TRAIN_SAMPLES:
        data['train_embeddings'] = data['train_embeddings'][:config.LIMIT_TRAIN_SAMPLES]
        data['train_labels'] = data['train_labels'][:config.LIMIT_TRAIN_SAMPLES]
        train_ids = train_ids[:config.LIMIT_TRAIN_SAMPLES]
        train_cands = train_cands[:config.LIMIT_TRAIN_SAMPLES]
        train_ans = train_ans[:config.LIMIT_TRAIN_SAMPLES] if train_ans else None
    
    print(f"Data shapes: Train {data['train_embeddings'].shape}, "
          f"Eval {data['eval_embeddings'].shape}, Test {data['test_embeddings'].shape}")
    
    # Create datasets
    train_ds = WSDDataset(data['train_embeddings'], data['train_labels'], 
                          train_cands, train_ids, train_ans)
    eval_ds = WSDDataset(data['eval_embeddings'], data['eval_labels'], 
                         eval_cands, eval_ids, eval_ans)
    test_ds = WSDDataset(data['test_embeddings'], data['test_labels'], 
                         test_cands, test_ids, test_ans)
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=cf.batch_size, 
                             shuffle=True, collate_fn=collate_wsd_batch)
    eval_loader = DataLoader(eval_ds, batch_size=config.EVAL_BATCH_SIZE, 
                            shuffle=False, collate_fn=collate_wsd_batch)
    test_loader = DataLoader(test_ds, batch_size=config.EVAL_BATCH_SIZE, 
                            shuffle=False, collate_fn=collate_wsd_batch)
    
    print(f"Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}, "
          f"Test batches: {len(test_loader)}")
    
    # Load model and RotE embeddings
    print("Loading RotE embeddings...")
    rote_emb = load_rote_embeddings(config.ROTE_MODEL)
    
    # Create concept mask: 1 for concepts that appear as answers in training set
    print("Creating concept mask for training set concepts...")
    concept_mask = create_concept_mask(train_ans, config.NUM_CONCEPTS, concept_id_to_index)
    concept_mask = concept_mask.to(device)
    print(f"  Concept mask: {concept_mask.sum().item():.0f} / {concept_mask.shape[0]} concepts in training set")
        
    model = ConceptClassifier(rote_emb, input_dim=config.INPUT_DIM, dropout=cf.do, 
                             temperature=cf.temperature, concept_mask=concept_mask, baseline=cf.baseline, baseline_type=cf.baseline_type).to(device)

    print("model:")
    print(model)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
    
    print("Config\n")
    print(cf)

    if cf.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    elif cf.optim == 'Adafactor':
        optimizer = optim.Adafactor(model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    elif cf.optim == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    elif cf.optim == 'RAdam':
        optimizer = optim.RAdam(model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    elif cf.optim == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    elif cf.optim == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay, momentum=cf.momentum)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)


    print(optimizer)


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,        # reduce LR by half
        patience=1,        # epochs to wait before reducing
        min_lr=1e-6
    )
    
    print(f"Model on: {device}, RotE shape: {rote_emb.shape}")
    
    # Training loop
    print("\nStarting training...")
    last_train_loss = None
    last_eval_loss = None
    last_test_loss = None
    final_eval_scores = {}
    final_test_scores = {}

    for epoch in range(1, cf.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        eval_loss, eval_preds = evaluate(model, eval_loader, loss_fn, device, 
                                        concept_id_to_index, index_to_concept_id)
        test_loss, test_preds = evaluate(model, test_loader, loss_fn, device,
                                        concept_id_to_index, index_to_concept_id)

        last_train_loss = train_loss
        last_eval_loss = eval_loss
        last_test_loss = test_loss
        
        
        print(f"Epoch {epoch}/{cf.epochs}: "
              f"Train Loss={train_loss:.4f}, Eval Loss={eval_loss:.4f}, "
              f"Test Loss={test_loss:.4f}")
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to wandb
        if HAS_WANDB and config.USE_WANDB:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'eval_loss': eval_loss,
                'test_loss': test_loss,
                'lr': current_lr,
            })
    
        # Save model and predictions in concept id space
        eval_pred_file = os.path.join(run_output_dir, config.EVAL_PREDICTIONS_FILE)
        test_pred_file = os.path.join(run_output_dir, config.TEST_PREDICTIONS_FILE)
        model_file = os.path.join(run_output_dir, config.MODEL_SAVE_PATH)
        
        # torch.save(model.state_dict(), model_file)
        save_predictions(eval_preds, eval_pred_file, concept_id_to_uk_id=None)
        save_predictions(test_preds, test_pred_file, concept_id_to_uk_id=None)
        
        print(f"\nTraining complete!")
        print(f"Model saved to {model_file}")
        print(f"Eval predictions: {len(eval_preds)}, Test predictions: {len(test_preds)}")
        
        # Evaluate against gold standards
        print("\nEvaluating against gold standards...")
        eval_scores = evaluate_with_gold_standard(
            eval_pred_file,
            config.EVAL_GOLD_STANDARD_DIR,
            'Eval Set',
            uk_id_to_concept_id,
        )
        test_scores = evaluate_with_gold_standard(
            test_pred_file,
            config.TEST_GOLD_STANDARD_DIR,
            'Test Set',
            uk_id_to_concept_id,
        )

        final_eval_scores = eval_scores
        final_test_scores = test_scores
        
        eval_f1 = eval_scores.get('ALL', {}).get('f1', 0.0)
        scheduler.step(-eval_f1)
        
        # Log evaluation scores to wandb
        if HAS_WANDB and config.USE_WANDB:
            for category, metrics in eval_scores.items():
                if category == 'Single Candidate Only':
                    continue
                if not metrics:
                    continue
                wandb.log({
                    f'eval_precision/{category}': metrics.get('precision', 0.0),
                    f'eval_recall/{category}': metrics.get('recall', 0.0),
                    f'eval_f1/{category}': metrics.get('f1', 0.0),
                })

            for category, metrics in test_scores.items():
                if not metrics:
                    continue
                wandb.log({
                    f'test_precision/{category}': metrics.get('precision', 0.0),
                    f'test_recall/{category}': metrics.get('recall', 0.0),
                    f'test_f1/{category}': metrics.get('f1', 0.0),
                })

            # Log eval scores summary
            if eval_scores:
                all_metrics = eval_scores.get('ALL', {})
                print(f"\nEval Set (ALL): P={all_metrics.get('precision', 0):.1f}%, R={all_metrics.get('recall', 0):.1f}%, F1={all_metrics.get('f1', 0):.1f}%")
            if test_scores:
                all_metrics = test_scores.get('ALL', {})
                print(f"Test Set (ALL): P={all_metrics.get('precision', 0):.1f}%, R={all_metrics.get('recall', 0):.1f}%, F1={all_metrics.get('f1', 0):.1f}%")

###
        # Write coverage/evaluation report file for this run
        report_file = os.path.join(run_output_dir, 'evaluation_report.txt')
        with open(report_file, 'w') as rf:
            def write_table(title, scores):
                cols = [
                    ("Subset", 35),
                    ("Gold", 8),
                    ("Evaluated", 10),
                    ("Coverage (%)", 14),
                    ("Precision", 10),
                    ("Recall", 8),
                    ("F1", 6),
                ]

                def sep():
                    return "+" + "+".join("-" * w for _, w in cols) + "+\n"

                def row(values):
                    return "|" + "|".join(
                        f"{str(v):<{w}}" for v, (_, w) in zip(values, cols)
                    ) + "|\n"

                rf.write(title + "\n")
                rf.write(sep())
                rf.write(row([c[0] for c in cols]))
                rf.write(sep())

                print("\n" + title)
                print(sep().strip())
                print(row([c[0] for c in cols]).strip())
                print(sep().strip())

                subsets = [
                    "ALL",
                    "Seen Only",
                    "Unseen Only",
                    "Single Candidate Only",
                    "Multiple Candidates Only",
                    "Multiple Candidates Seen Only",
                    "Multiple Candidates Unseen Only",
                ]

                for subset in subsets:
                    metrics = scores.get(subset, {})
                    row_vals = [
                        subset,
                        int(metrics.get("gold_total", 0)),
                        int(metrics.get("effective_size", 0)),
                        f"{float(metrics.get('coverage_gold_pct', 0.0)):.1f}",
                        f"{float(metrics.get('precision', 0.0)):.1f}",
                        f"{float(metrics.get('recall', 0.0)):.1f}",
                        f"{float(metrics.get('f1', 0.0)):.1f}",
                    ]
                    rf.write(row(row_vals))
                    print(row(row_vals).strip())

                rf.write(sep())
                print(sep().strip())


            write_table('Eval Set Coverage & Scores', eval_scores)
            rf.write('\n')
            write_table('Test Set Coverage & Scores', test_scores)

        

    print(f"\nEvaluation report saved to {report_file}")

    if HAS_WANDB and config.USE_WANDB:
        wandb.finish()

    metrics = {
        'run_name': run_name,
        'run_output_dir': run_output_dir,
        'seed': cf.seed,
        'train_loss': last_train_loss,
        'eval_loss': last_eval_loss,
        'test_loss': last_test_loss,
        'eval_scores': final_eval_scores,
        'test_scores': final_test_scores,
    }

    if return_metrics:
        return model, eval_preds, test_preds, metrics
    
    return model, eval_preds, test_preds


if __name__ == '__main__':
    train()
