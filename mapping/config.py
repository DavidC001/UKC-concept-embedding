from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    project_root: Path
    concept_glosses_csv: Path
    concepts_csv: Path
    concept_relations_csv: Path
    rote_checkpoint: Path
    entity_to_id: Path
    model_name: str
    output_dir: Path
    embedding_batch_size: int
    mapper_batch_size: int
    mapper_hidden: int
    epochs: int
    lr: float
    weight_decay: float
    val_ratio: float
    test_ratio: float
    top_k: int
    show_examples: int
    geodesic_sample_size: int
    run_geodesic: bool
    random_seed: int
    device: str

def parse_args() -> PipelineConfig:
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Encode concept glosses and train a mapper into RotE embedding space",
    )
    parser.add_argument(
        "--concept-glosses-csv",
        type=Path,
        default=project_root / "dataset" / "concept_glosses.csv",
        help="CSV with columns: concept_id, gloss",
    )
    parser.add_argument(
        "--concepts-csv",
        type=Path,
        default=project_root / "dataset" / "concepts.csv",
        help="Optional concept metadata CSV",
    )
    parser.add_argument(
        "--concept-relations-csv",
        type=Path,
        default=project_root / "dataset" / "concept_relations.csv",
        help="Concept graph CSV used for geodesic analysis",
    )
    parser.add_argument(
        "--rote-checkpoint",
        type=Path,
        default=project_root / "dataset" / "RotE" / "model.pt",
        help="RotE checkpoint path",
    )
    parser.add_argument(
        "--entity-to-id",
        type=Path,
        default=project_root / "dataset" / "RotE" / "entity_to_id.pickle",
        help="Entity-to-index pickle used to align concept ids",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-Embedding-8B",
        help="SentenceTransformer model name for gloss encoding",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "mapping" / "outputs",
        help="Output directory",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--mapper-batch-size", type=int, default=64)
    parser.add_argument("--mapper-hidden", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--show-examples", type=int, default=10)
    parser.add_argument("--geodesic-sample-size", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cuda, mps, cpu",
    )
    parser.add_argument(
        "--no-geodesic",
        action="store_true",
        help="Disable geodesic distance analysis",
    )

    args = parser.parse_args()

    return PipelineConfig(
        project_root=project_root,
        concept_glosses_csv=args.concept_glosses_csv,
        concepts_csv=args.concepts_csv,
        concept_relations_csv=args.concept_relations_csv,
        rote_checkpoint=args.rote_checkpoint,
        entity_to_id=args.entity_to_id,
        model_name=args.model_name,
        output_dir=args.output_dir,
        embedding_batch_size=args.embedding_batch_size,
        mapper_batch_size=args.mapper_batch_size,
        mapper_hidden=args.mapper_hidden,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        top_k=args.top_k,
        show_examples=args.show_examples,
        geodesic_sample_size=args.geodesic_sample_size,
        run_geodesic=not args.no_geodesic,
        random_seed=args.random_seed,
        device=args.device,
    )
