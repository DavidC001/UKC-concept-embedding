from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

@dataclass
class PipelineConfig:
    project_root: Path = project_root
    """Root directory of the project, used to resolve relative paths"""
    concept_glosses_csv: Path = Path(project_root / "dataset/concept_glosses.csv")
    """CSV with columns: concept_id, gloss"""
    concepts_csv: Path = Path(project_root / "dataset/concepts.csv")
    """Optional concept metadata CSV"""
    concept_relations_csv: Path = Path(project_root / "dataset/concept_relations.csv")
    """Concept graph CSV used for geodesic analysis"""

    rote_checkpoint: Path = Path(project_root / "dataset/RotE/model.pt")
    """RotE checkpoint path"""
    entity_to_id: Path = Path(project_root / "dataset/RotE/entity_to_id.pickle")
    """Entity-to-index pickle used to align concept ids"""
    model_name: str = "Qwen/Qwen3-Embedding-8B"
    """SentenceTransformer model name for gloss encoding"""
    output_dir: Path = Path(project_root / "mapping/outputs")
    """Output directory"""

    embedding_batch_size: int = 512
    """Batch size for encoding glosses into embeddings"""
    mapper_batch_size: int = 40000
    """Batch size for training the mapper"""
    epochs: int = 100
    """Number of training epochs"""
    lr: float = 1e-3
    """Learning rate"""

    weight_decay: float = 1e-2
    """Weight decay for mapper optimizer"""
    test_ratio: float = 0.3
    """Ratio of training data to use for testing"""
    top_k: int = 10
    """Number of nearest neighbors to retrieve during evaluation"""
    run_geodesic: bool = True
    """Whether to run geodesic distance analysis after mapping evaluation"""
    
    random_seed: int = 42
    """Random seed for reproducibility"""
    device: str = "cuda"
    """Device to run training and inference on"""

def parse_args() -> PipelineConfig:
    defaults = PipelineConfig()

    parser = argparse.ArgumentParser(
        description="Encode concept glosses and train a mapper into RotE embedding space",
    )
    parser.add_argument(
        "--concept-glosses-csv",
        type=Path,
        default=defaults.concept_glosses_csv,
        help="CSV with columns: concept_id, gloss",
    )
    parser.add_argument(
        "--concepts-csv",
        type=Path,
        default=defaults.concepts_csv,
        help="Optional concept metadata CSV",
    )
    parser.add_argument(
        "--concept-relations-csv",
        type=Path,
        default=defaults.concept_relations_csv,
        help="Concept graph CSV used for geodesic analysis",
    )
    parser.add_argument(
        "--rote-checkpoint",
        type=Path,
        default=defaults.rote_checkpoint,
        help="RotE checkpoint path",
    )
    parser.add_argument(
        "--entity-to-id",
        type=Path,
        default=defaults.entity_to_id,
        help="Entity-to-index pickle used to align concept ids",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=defaults.model_name,
        help="SentenceTransformer model name for gloss encoding",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=defaults.output_dir,
        help="Output directory",
    )
    
    parser.add_argument("--embedding-batch-size", type=int, default=defaults.embedding_batch_size)
    parser.add_argument("--mapper-batch-size", type=int, default=defaults.mapper_batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--test-ratio", type=float, default=defaults.test_ratio)
    parser.add_argument("--top-k", type=int, default=defaults.top_k)

    parser.add_argument("--random-seed", type=int, default=defaults.random_seed)
    parser.add_argument(
        "--device",
        type=str,
        default=defaults.device,
        help="auto, cuda, mps, cpu",
    )
    parser.add_argument(
        "--no-geodesic",
        action="store_true",
        help="Disable geodesic distance analysis",
    )

    args = parser.parse_args()

    return PipelineConfig(
        # data paths
        concept_glosses_csv=args.concept_glosses_csv,
        concepts_csv=args.concepts_csv,
        concept_relations_csv=args.concept_relations_csv,
        rote_checkpoint=args.rote_checkpoint,
        entity_to_id=args.entity_to_id,
        
        model_name=args.model_name,
        output_dir=args.output_dir,
        embedding_batch_size=args.embedding_batch_size,
        mapper_batch_size=args.mapper_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        test_ratio=args.test_ratio,
        top_k=args.top_k,
        run_geodesic=not args.no_geodesic,
        random_seed=args.random_seed,
        device=args.device,
    )
