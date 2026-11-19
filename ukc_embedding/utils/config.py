import argparse
from models import all_models
from optimizers.regularizers import REGULARIZERS



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Knowledge Graph Embedding")
    # parser.add_argument("--dataset", default="UKC", choices=["UKC", "UKC_NEW", "UKC_NEW_7", "UKC_CUTDYN", "UKC_AUG"], help="Knowledge Graph dataset")

    # === MODEL AND DATASET ===
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument("--device", default="cuda", type=str, help="Device to use for training (e.g., 'cuda', 'cpu')")
    parser.add_argument("--debug", action="store_true", help="Only use 1000 examples for debugging")
    parser.add_argument("--dataset", default="UKC", help="Knowledge Graph dataset")
    parser.add_argument("--model", default="RotE", choices=all_models, help="Knowledge Graph embedding model")

    # === HYPERPARAMETERS ===
    parser.add_argument("--rank", default=100, type=int, help="Embedding dimension")
    parser.add_argument("--neg_sample_size", default=250, type=int, help="Negative sample size, -1 to not use negative sampling")
    parser.add_argument("--bias", default="learn", type=str, choices=["constant", "learn", "none"], help="Bias type (none for no bias)")
    parser.add_argument("--multi_c", action="store_true", help="Multiple curvatures per relation")

    parser.add_argument("--relgeo_families", default=None, type=str, help="RelGeoE: overrides of the relation->family table, e.g. '22:offset,101:near' (families: hier, offset, affine, reflect, attr, near)")
    parser.add_argument("--relgeo_affine_rank", default=8, type=int, help="RelGeoE: rank of the relation-specific part of the affine maps")
    parser.add_argument("--relgeo_affine", default="lowrank", choices=["lowrank", "tucker"], help="RelGeoE: linear part of the affine family, s*I + U V^T (lowrank) or s*I + TuckER core x relation vector (tucker)")
    parser.add_argument("--relgeo_tucker_dim", default=32, type=int, help="RelGeoE: relation-vector size d_r of the TuckER core (d x d_r x d)")
    parser.add_argument("--relgeo_attr_rank", default=8, type=int, help="RelGeoE: dimension of the attribute subspace (ANTONYM reflection, VALUE_OF sphere)")
    parser.add_argument("--relgeo_attr_separate", action="store_true", help="RelGeoE: one attribute subspace per relation instead of one shared by ANTONYM and VALUE_OF")

    # === TRAINING & REGULARIZATION ===
    parser.add_argument("--optimizer", choices=["Adagrad", "Adam", "SparseAdam", "AdamW"], default="AdamW", help="Optimizer")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--max_epochs", default=50, type=int, help="Maximum number of epochs to train for")
    parser.add_argument("--batch_size", default=1024, type=int, help="Batch size")
    parser.add_argument("--early_stop_metric", choices=["loss", "mrr"], default="loss", help="Validation quantity used for checkpoint selection, LR reduction and early stopping (loss: lower is better, mrr: higher is better)")
    parser.add_argument("--valid", default=3, type=float, help="Number of epochs before validation")
    parser.add_argument("--patience", default=2, type=int, help="Number of validation checks (at validation frequency) with no improvement in validation loss before early stopping")

    parser.add_argument("--regularizer", nargs="+", choices=REGULARIZERS, default=["N3"],
                        help="One or more regularizers added to the loss, e.g. '--regularizer N3 RootAnchor --reg 0.001 1.0' "
                             "(N3/F2: norm of the batch embeddings; RootAnchor: w*||e_root - target||^2)")
    parser.add_argument("--reg", nargs="+", default=[0.0], type=float, help="Weight of each --regularizer, in the same order")
    parser.add_argument("--root_target", default="origin", choices=["origin", "centroid"], help="Where RootAnchor pulls the root")
    
    parser.add_argument("--lr_reduce_factor", default=0.5, type=float, help="Factor to multiply learning rate by when validation loss doesn't improve (set to 1.0 to disable)")
    parser.add_argument("--dropout", default=0, type=float, help="Dropout rate")
    parser.add_argument("--init_size", default=1e-3, type=float, help="Initial embeddings' scale")
    parser.add_argument("--gamma", default=0, type=float, help="Margin for distance-based losses")
    parser.add_argument("--double_neg", action="store_true", help="Whether to negative sample both head and tail entities")

    # === OUTPUT ===
    parser.add_argument("--save_dir", default=None, type=str, help="Fixed output directory (default: logs/<date>/<dataset>/<model>_<time>)")
    parser.add_argument("--results_file", default=None, type=str, help="If set, write final valid/test metrics and the best checkpoint path to this JSON file")

    return parser
