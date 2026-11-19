"""Train Knowledge Graph embeddings for link prediction."""

import argparse
import json
import logging
import os
import time

import wandb
import torch

import models
from datasets.kg_dataset import KGDataset
from models import all_models
from optimizers import KGOptimizer, build_regularizers
from utils.train import get_savedir, avg_both, format_metrics, count_params, scalar_metrics
from utils.config import build_parser


def train(args):
    # set seed for all random number generators (Python, NumPy, PyTorch)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.save_dir:
        save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = get_savedir(args.model, args.dataset, args.rank)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # --- wandb init ---
    wandb.init(
        project="thesis",
        name=f"{args.model}_{args.dataset}_{args.rank}_{timestamp}",
        config=vars(args),
    )
    # every wandb.log call advances wandb's own step; plot everything against the epoch instead
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

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
    args.sym = [int(r) for r in dataset.symmetric]  # symmetric relation ids (list, so vars(args) stays JSON-serialisable)

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
    if hasattr(model, "describe"):  # RelGeoE: relation -> family table actually used
        logging.info(f"\t {args.model} families: {model.describe()}")
    
    total = count_params(model)
    logging.info("Total number of parameters {}".format(total))
    ### device = "cuda"
    device = args.device
    model.to(device)

    # regularizers: --regularizer N3 RootAnchor --reg 0.001 1.0 -> {"N3": N3(0.001), "RootAnchor": RootAnchor(1.0, ...)}
    regs = build_regularizers(args.regularizer, args.reg,
                              entity_to_id=getattr(dataset, "entity_to_id", None), root_target=args.root_target)
    logging.info("\t Regularizers: " + ", ".join(f"{n} (w={r.weight})" for n, r in regs.items()))

    # get optimizer
    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    optimizer = KGOptimizer(
        model, regs, optim_method,
        args.batch_size, args.neg_sample_size, bool(args.double_neg))
    
    # early stopping
    counter = 0
    best_valid_loss = None
    best_epoch = None
    best_sel_mrr = None
    logging.info("\t Start training")

    results_dir = os.path.join(save_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    for step in range(args.max_epochs):

        # Train step
        model.train()
        train_loss, train_regs = optimizer.epoch(train_examples)
        logging.info("\t Epoch {} | average train loss: {:.4f}".format(step, train_loss)
                     + "".join(f" | {n}: {v:.4f}" for n, v in train_regs.items()))

        # --- wandb: log train loss ---
        wandb.log({"train/loss": train_loss, **{f"train/reg/{n}": v for n, v in train_regs.items()},
                   "epoch": step, "iteration": step})

        # Valid step
        model.eval()
        valid_loss, valid_regs = optimizer.calculate_valid_loss(valid_examples)
        logging.info("\t Epoch {} | average valid loss: {:.4f}".format(step, valid_loss)
                     + "".join(f" | {n}: {v:.4f}" for n, v in valid_regs.items()))

        # --- wandb: log valid loss ---
        wandb.log({"valid/loss": valid_loss, **{f"valid/reg/{n}": v for n, v in valid_regs.items()},
                   "epoch": step, "iteration": step})


        if (step + 1) % args.valid == 0:
            valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
            valid_mrr = valid_metrics["mrr"]

            logging.info(format_metrics(valid_metrics, split="valid"))

            # LOG METRICS
            wandb.log({f"valid/{k}": v for k, v in scalar_metrics(valid_metrics).items()} | {"epoch": step})

            valid_json.append({
                "epoch": step,
                **scalar_metrics(valid_metrics)
            })

            # Save model when the selection metric improves
            if args.early_stop_metric == "mrr":
                improved = best_sel_mrr is None or valid_mrr > best_sel_mrr
            else:
                improved = best_valid_loss is None or valid_loss < best_valid_loss
                
            if improved:
                best_sel_mrr = valid_mrr
                best_valid_loss = valid_loss
                counter = 0
                best_epoch = step
                
                model_dir = os.path.join(save_dir, "model_checkpoint")
                os.makedirs(model_dir, exist_ok=True)
                best_path_current = os.path.join(model_dir, f"model_{timestamp}_best.pt")

                torch.save(model.state_dict(), best_path_current)
                # wandb.save(best_path_current)

                logging.info(f"\t New best ({args.early_stop_metric}) VALID LOSS={valid_loss:.6f}. MRR={valid_mrr:.6f} Saved model to {best_path_current}")

                model.to(device)
            else:
                counter += 1

                # Reduce learning rate when validation loss doesn't improve.
                if args.lr_reduce_factor and args.lr_reduce_factor != 1.0:
                    optimizer.reduce_lr(factor=args.lr_reduce_factor)
                    logging.info(f"\t Validation did not improve — reduced LR by factor {args.lr_reduce_factor}.")
                    
                    lrs = [pg.get('lr', None) for pg in optimizer.optimizer.param_groups]
                    logging.info(f"\t\t Current lr(s): {lrs}")
                    # wandb can't plot a list: one scalar per param group (plain train/lr when there is only one)
                    lr_log = {"train/lr": lrs[0]} if len(lrs) == 1 else {f"train/lr_{i}": lr for i, lr in enumerate(lrs)}
                    wandb.log(lr_log | {"epoch": step, "iteration": step})

                if counter >= args.patience:
                    logging.info("\t Early stopping (no improvement in validation {} for {} checks)".format(args.early_stop_metric, args.patience))
                    break

    logging.info("\t Optimization finished")
    model_dir = os.path.join(save_dir, "model_checkpoint")
    os.makedirs(model_dir, exist_ok=True) 

    best_path = os.path.join(model_dir, f"model_{timestamp}_best.pt")
    logging.info(f"\t Loading best model saved at epoch {best_epoch}")
    model.load_state_dict(torch.load(best_path, map_location=device))

    model.to(device)
    model.eval()

    # Validation metrics
    valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))

    logging.info(format_metrics(valid_metrics, split="valid"))
    
    final_valid = {f"final_valid/{k}": v for k, v in scalar_metrics(valid_metrics).items()}


    # ===== append final validation metrics too (optional) =====
    
    valid_json.append({
        "epoch": best_epoch if best_epoch is not None else args.max_epochs,
        **scalar_metrics(valid_metrics)
    })


    # Test metrics
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
    logging.info(format_metrics(test_metrics, split="test"))
    
    # --- wandb: log test metrics ---
    final_test = {f"test/{k}": v for k, v in scalar_metrics(test_metrics).items()}
    # final numbers go in the run summary (runs table, bar/scalar panels); logged at the best epoch too
    wandb.log(final_valid | final_test | {"epoch": best_epoch if best_epoch is not None else step})
    wandb.run.summary.update(final_valid | final_test | {"best_epoch": best_epoch})

    # ===== append test metrics to JSON =====
    
    test_json.append({
        "epoch": best_epoch if best_epoch is not None else args.max_epochs,
        **scalar_metrics(test_metrics)
    })

    os.makedirs(results_dir, exist_ok=True)
    

    logging.info("\t Saved valid_results.json and test_results.json")

    if args.results_file:
        # per-relation test metrics (queries in both directions), keyed by raw relation name
        test_per_relation = {}
        for r in range(args.sizes[1] // 2):
            rel_examples = dataset.get_examples("test", rel_idx=r)
            if len(rel_examples) == 0:
                continue
            m = avg_both(*model.compute_metrics(rel_examples, filters))
            test_per_relation[str(dataset.id_to_relation.get(r, r))] = {
                "n": int(len(rel_examples)),
                **scalar_metrics(m),
            }
        summary = {
            "args": {k: v for k, v in vars(args).items() if k != "sizes"},
            "best_epoch": best_epoch,
            "checkpoint": best_path,
            "save_dir": save_dir,
            "valid": scalar_metrics(valid_metrics),
            "test": scalar_metrics(test_metrics),
            "test_per_relation": test_per_relation,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.results_file)), exist_ok=True)
        with open(args.results_file, "w") as f:
            json.dump(summary, f, indent=2)
        logging.info(f"\t Wrote summary to {args.results_file}")
    wandb.finish()



if __name__ == "__main__":
    train(build_parser().parse_args())
