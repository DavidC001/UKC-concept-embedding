"""Train Knowledge Graph embeddings for link prediction."""

import argparse
import json
import logging
import os

import torch
import torch.optim

import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from models import all_models
from optimizers.kg_optimizer import KGOptimizer
from utils.train import get_savedir, avg_both, format_metrics, count_params
import time

# --- wandb ---
import wandb

parser = argparse.ArgumentParser(description="Knowledge Graph Embedding")
# parser.add_argument("--dataset", default="UKC", choices=["UKC", "UKC_NEW", "UKC_NEW_7", "UKC_CUTDYN", "UKC_AUG"], help="Knowledge Graph dataset")
parser.add_argument("--dataset", default="UKC", help="Knowledge Graph dataset")
parser.add_argument("--model", default="RotE", choices=all_models, help="Knowledge Graph embedding model")
parser.add_argument("--regularizer", choices=["N3", "F2"], default="N3", help="Regularizer")
parser.add_argument("--reg", default=0, type=float, help="Regularization weight")
parser.add_argument("--optimizer", choices=["Adagrad", "Adam", "SparseAdam", "AdamW"], default="AdamW", help="Optimizer")
parser.add_argument("--max_epochs", default=50, type=int, help="Maximum number of epochs to train for")
parser.add_argument("--patience", default=2, type=int, help="Number of validation checks (at validation frequency) with no improvement in validation loss before early stopping")
parser.add_argument("--lr_reduce_factor", default=0.5, type=float, help="Factor to multiply learning rate by when validation loss doesn't improve (set to 1.0 to disable)")
parser.add_argument("--valid", default=3, type=float, help="Number of epochs before validation")
parser.add_argument("--rank", default=500, type=int, help="Embedding dimension")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
parser.add_argument("--neg_sample_size", default=10000, type=int, help="Negative sample size, -1 to not use negative sampling")
parser.add_argument("--dropout", default=0, type=float, help="Dropout rate")
parser.add_argument("--init_size", default=1e-3, type=float, help="Initial embeddings' scale")
parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--gamma", default=0, type=float, help="Margin for distance-based losses")
parser.add_argument("--bias", default="learn", type=str, choices=["constant", "learn", "none"], help="Bias type (none for no bias)")
parser.add_argument("--dtype", default="double", type=str, choices=["single", "double"], help="Machine precision")
parser.add_argument("--double_neg", action="store_true", help="Whether to negative sample both head and tail entities")
parser.add_argument("--debug", action="store_true", help="Only use 1000 examples for debugging")
parser.add_argument("--multi_c", action="store_true", help="Multiple curvatures per relation")

def metrics_to_python(metrics):
    return {k: to_python(v) for k, v in metrics.items()}


def ranks_to_python(all_ranks):
    """Convert {lhs: tensor, rhs: tensor} to python lists"""
    return {
        "lhs": all_ranks["lhs"].tolist(),
        "rhs": all_ranks["rhs"].tolist(),
    }

def to_python(x):
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()  # scalar
        return x.tolist()   # multi-element tensor
    elif isinstance(x, dict):
        return {k: to_python(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [to_python(v) for v in x]
    else:
        return x

def metrics_to_json(metrics):
    return to_python(metrics)
def log_outliers(model, examples, all_ranks, dataset, epoch, results_dir, timestamp, rank_threshold=1000):
    """Diagnoses and logs outlier triples with ranks > rank_threshold."""
    
    # Use the new method added to the KGModel class
    outlier_list = model._get_outlier_info(examples, all_ranks, rank_threshold, dataset)
    
    if outlier_list:
        file_path = os.path.join(results_dir, f"outliers_epoch_{epoch}_{timestamp}.txt")
        with open(file_path, "w") as f:
            f.write(f"--- Outliers (Rank > {rank_threshold}) at Epoch {epoch} ---\n")
            f.write(f"Total Outliers Found: {len(outlier_list)}\n\n")
            f.write("Format: [Prediction Side] Outlier: Rank [rank] for triple (h_id, r_id, t_id) | Frequencies (in training set): h=[h_freq], r=[r_freq], t=[t_freq]\n\n")
            for detail in outlier_list:
                f.write(detail + "\n")
        logging.info(f"\t Outlier diagnosis: Found {len(outlier_list)} triples with rank > {rank_threshold}. Saved to {file_path}")
    else:
        logging.info(f"\t Outlier diagnosis: No triples found with rank > {rank_threshold}.")
        
def train(args):
    save_dir = get_savedir(args.model, args.dataset, args.rank)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # --- wandb init ---
    wandb.init(
        project="thesis",
        name=f"{args.model}_{args.dataset}_{args.rank}_{timestamp}",
        config=vars(args),
    )

    # ----- JSON arrays for saving metric history -----
    valid_json = []
    test_json = []


    # file logger
    log_dir = os.path.join(save_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(log_dir, "train.log")
    )

    # stdout logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Saving logs in: {}".format(save_dir))

    # create dataset
    dataset_path = os.path.join(os.environ["DATA_PATH"], args.dataset)
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()

    # load data
    logging.info("\t " + str(dataset.get_shape()))
    train_examples = dataset.get_examples("train")
    valid_examples = dataset.get_examples("valid")
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()

    # save config
    with open(os.path.join(log_dir, f"config_{timestamp}.json"), "w") as fjson:
        json.dump(vars(args), fjson)

    # create model
    model = getattr(models, args.model)(args)
    total = count_params(model)
    logging.info("Total number of parameters {}".format(total))
    ### device = "cuda"
    device = "cuda" if torch.cuda.is_available() else "mps"
    model.to(device)

    # get optimizer
    regularizer = getattr(regularizers, args.regularizer)(args.reg)
    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    optimizer = KGOptimizer(model, regularizer, optim_method, args.batch_size, args.neg_sample_size,
                            bool(args.double_neg))
    counter = 0
    # Track best validation loss (lower is better). We will stop training if
    # validation loss does not improve for `args.patience` validation checks.
    best_valid_loss = None
    best_epoch = None
    best_epoch = None
    best_mrr = None
    logging.info("\t Start training")

    rank_history = []  # store distributions over time
    results_dir = os.path.join(save_dir, "results")
    os.makedirs(results_dir, exist_ok=True) 
    for step in range(args.max_epochs):

        # Train step
        model.train()
        train_loss = optimizer.epoch(train_examples)
        logging.info("\t Epoch {} | average train loss: {:.4f}".format(step, train_loss))

        # --- wandb: log train loss ---
        wandb.log({"train/loss": train_loss, "epoch": step, "iteration": step})

        # Valid step
        model.eval()
        valid_loss = optimizer.calculate_valid_loss(valid_examples)
        logging.info("\t Epoch {} | average valid loss: {:.4f}".format(step, valid_loss))

        # --- wandb: log valid loss ---
        wandb.log({"valid/loss": valid_loss, "epoch": step, "iteration": step})


        if (step + 1) % args.valid == 0:
            valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
            # --- ADDED CALL TO DIAGNOSE OUTLIERS ---
         
            # --- START: ADDED CALL TO DIAGNOSE OUTLIERS ---
            log_outliers(
                model, 
                valid_examples, 
                valid_metrics['all_ranks'], 
                dataset, 
                step, 
                results_dir, 
                timestamp,
                rank_threshold=1000 # Your specified threshold
            )

            ## Save rank distribution
            # rank_history.append({
            #     "epoch": step,
            #     "ranks": ranks_to_python(valid_metrics['all_ranks'])
            # })


            # with open(os.path.join(results_dir,
            # f"rank_history_{args.model}_{args.dataset}_{args.rank}_{timestamp}.json"), "w") as f: json.dump(rank_history, f, indent=4)


            logging.info(format_metrics(valid_metrics, split="valid"))

            # --- wandb: log valid metrics ---
            wandb.log({f"{k}": v for k, v in valid_metrics.items()})

            # ===== append validation result to JSON array =====
            valid_json.append({
                "epoch": step,
                **valid_metrics
            })


            # Use validation loss for early stopping (lower is better).
            # valid_loss is computed each epoch, but metrics are computed only
            # at the `args.valid` frequency. We check early stopping here
            # (at the same frequency as metrics are computed).
            valid_mrr = valid_metrics["mrr"]
            model_dir = os.path.join(save_dir, "model_checkpoint")
            os.makedirs(model_dir, exist_ok=True)

            if not best_mrr or valid_mrr > best_mrr:
                best_mrr = valid_mrr

            best_path_current = os.path.join(model_dir, f"model_{timestamp}_best.pt")

            # Save model when validation loss improves
            if best_valid_loss is None or valid_loss < best_valid_loss:
                if os.path.exists(best_path_current):
                    os.remove(best_path_current)

                torch.save(model.state_dict(), best_path_current)
                # wandb.save(best_path_current)

                best_valid_loss = valid_loss
                counter = 0
                best_epoch = step
                logging.info(f"\t New best VALID LOSS={best_valid_loss:.6f}. current best MRR={valid_mrr:.6f} Saved model to {best_path_current}")

                ### model.cuda()
                model.to(device)
            else:
                counter += 1

                # Reduce learning rate when validation loss doesn't improve.
                # If lr_reduce_factor is 1.0 (default disabled behavior) it won't change.
                if args.lr_reduce_factor and args.lr_reduce_factor != 1.0:
                    try:
                        optimizer.reduce_lr(factor=args.lr_reduce_factor)
                        # log current lr(s)
                        lrs = [pg.get('lr', None) for pg in optimizer.optimizer.param_groups]
                        logging.info(f"\t Validation did not improve — reduced LR by factor {args.lr_reduce_factor}. Current lr(s): {lrs}")
                        wandb.log({"lr/reduced_to": lrs, "epoch": step, "iteration": step})
                    except Exception:
                        # If optim doesn't expose param_groups in expected place
                        logging.info("\t Could not reduce LR — optimizer does not support param_groups access")

                if counter >= args.patience:
                    logging.info("\t Early stopping (no improvement in validation loss for {} checks)".format(args.patience))
                    break
                elif counter == args.patience // 2:
                    pass
                    # logging.info("\t Reducing learning rate")
                    # optimizer.reduce_lr()


    # with open(os.path.join(results_dir,
    # f"rank_history_{args.model}_{args.dataset}_{args.rank}_{timestamp}.json"), "w") as f: json.dump(rank_history, f, indent=4)

    logging.info("\t Optimization finished")
    model_dir = os.path.join(save_dir, "model_checkpoint")
    os.makedirs(model_dir, exist_ok=True) 

    best_path = os.path.join(model_dir, f"model_{timestamp}_best.pt")
    logging.info(f"\t Loading best model saved at epoch {best_epoch}")
    model.load_state_dict(torch.load(best_path, map_location=device))

    # if not best_mrr:
    #     torch.save(model.state_dict(), os.path.join(model_dir, f"model_{timestamp}.pt"))
    # else:
    #     logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
    #     torch.save(model.state_dict(), os.path.join(model_dir, f"model_{timestamp}_best.pt"))
    #     model.load_state_dict(torch.load(os.path.join(model_dir, f"model_{timestamp}.pt")))

    # --- wandb: save final or best model ---
    # wandb.save(os.path.join(model_dir, f"model_{timestamp}_best.pt"))

    ### model.cuda()
    model.to(device)
    model.eval()

    # Validation metrics
    valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))

    # valid_metrics = to_python(valid_metrics_raw)
    logging.info(format_metrics(valid_metrics, split="valid"))


    # --- wandb: log final valid metrics ---
    wandb.log({f"final_valid/{k}": v for k, v in valid_metrics.items()})


    # ===== append final validation metrics too (optional) =====
    
    valid_json.append({
        "epoch": best_epoch if best_epoch is not None else args.max_epochs,
        **metrics_to_json(valid_metrics)
    })


    # Test metrics
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
    logging.info(format_metrics(test_metrics, split="test"))
    
    # --- wandb: log test metrics ---
    wandb.log({f"test/{k}": v for k, v in test_metrics.items()})

    # ===== append test metrics to JSON =====
    
    test_json.append({
        "epoch": best_epoch if best_epoch is not None else args.max_epochs,
        **metrics_to_json(test_metrics)
    })

    os.makedirs(results_dir, exist_ok=True)
    

    # valid_json_serializable = [to_python(m) for m in valid_json]
    # test_json_serializable  = [to_python(m) for m in test_json]
    # ===== write JSON files =====
    # with open(os.path.join(results_dir, f"valid_results_{args.model}_{args.dataset}_{args.rank}_{timestamp}.json"), "w") as f:
    #     json.dump(valid_json, f, indent=4)

    # with open(os.path.join(results_dir, f"test_results_{args.model}_{args.dataset}_{args.rank}_{timestamp}.json"), "w") as f:
    #     json.dump(test_json, f, indent=4)

    logging.info("\t Saved valid_results.json and test_results.json")



if __name__ == "__main__":
    train(parser.parse_args())
