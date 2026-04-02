# Gloss-to-RotE Mapping Pipeline

This directory contains the runnable pipeline that encodes concept glosses, aligns them to RotE entity embeddings, trains a mapper, and produces evaluation artifacts.

The pipeline is designed to run in the `ANLP` conda environment.

## Overview

The current workflow is:

1. Load concept glosses from `dataset/concept_glosses.csv`.
2. Encode each gloss with a SentenceTransformer model.
3. Normalize and save the resulting embeddings.
4. Load RotE embeddings and the `entity_to_id` mapping.
5. Align concept ids to RotE entity indices.
6. Train a neural mapper from gloss space to RotE space.
7. Save the mapper checkpoint, top-k neighbor report, and optional geodesic analysis outputs.

## Components

### `mapping/main.py`

The main entrypoint. It orchestrates the full pipeline end to end:

- reads configuration from CLI arguments,
- loads and encodes glosses,
- aligns concepts to RotE embeddings,
- trains the mapper,
- writes evaluation outputs.

You can run it either as a module or as a script.

### `mapping/config.py`

Defines the pipeline configuration and CLI arguments:

- input CSV paths,
- RotE checkpoint path,
- `entity_to_id` path,
- model name,
- batch sizes,
- training hyperparameters,
- output directory,
- device selection,
- geodesic-analysis toggle.

### `mapping/model/gloss_encoder.py`

Encodes gloss text into dense vectors using `sentence-transformers`.

It also provides device selection for CUDA, MPS, or CPU.

### `mapping/utils/gloss_io.py`

Handles gloss input and embedding serialization.

Responsibilities:

- load `dataset/concept_glosses.csv`,
- validate that the file has `concept_id` and `gloss` columns,
- normalize embeddings,
- save `.npz` files for raw and normalized embeddings.

### `mapping/utils/load_embeddings.py`

Loads the RotE checkpoint and the concept-to-index mapping.

It also provides the alignment helper that maps `concept_id` values to RotE indices, with optional label fallback from `dataset/concepts.csv`.

### `mapping/model/mapping_model.py`

Contains the mapper network and the cosine-based loss:

- `LinearMapper`
- `cosine_loss`

The mapper is intentionally simple so the pipeline stays easy to inspect and reproduce.

### `mapping/trainer.py`

Contains the training loop and checkpoint save logic.

Responsibilities:

- split the aligned data into train, validation, and test sets,
- train the mapper,
- compute test cosine similarity and MSE,
- save the trained model checkpoint.

### `mapping/evaluation.py`

Produces post-training analysis artifacts:

- top-k nearest-neighbor report,
- optional geodesic distance analysis,
- optional geodesic histogram plot.

## Input Files

The default inputs are:

- `dataset/concept_glosses.csv` with columns `concept_id` and `gloss`,
- `dataset/concepts.csv` for optional label enrichment,
- `dataset/concept_relations.csv` for geodesic analysis,
- `dataset/entity_to_id.pickle` or `dataset/RotE/entity_to_id.pickle`,
- `RotE_model_20251201_144211_best.pt` or `dataset/RotE/model.pt`.

## Outputs

The default output directory is `mapping/outputs/`.

Expected artifacts:

- `qwen_concept_embeddings_raw.npz`
- `qwen_concept_embeddings.npz`
- `mapped_embeddings.npz`
- `linear_mapper_qwen_to_rote.pt`
- `topk_neighbors.csv`
- `geodesic_distances.csv` if geodesic analysis is enabled and usable
- `geodesic_distance.png` if plotting dependencies are available

## How to Run

Recommended command:

```bash
conda run -n ANLP python -m mapping.main
```

If you prefer direct script execution, this also works:

```bash
conda run -n ANLP python mapping/main.py
```

## Common Options

You can override paths and hyperparameters from the CLI.

Examples:

```bash
conda run -n ANLP python -m mapping.main \
  --output-dir mapping/outputs/run_01 \
  --model-name Qwen/Qwen3-Embedding-8B \
  --embedding-batch-size 16 \
  --mapper-batch-size 64 \
  --epochs 50
```

Disable geodesic analysis:

```bash
conda run -n ANLP python -m mapping.main --no-geodesic
```

Override the gloss CSV:

```bash
conda run -n ANLP python -m mapping.main \
  --concept-glosses-csv dataset/concept_glosses.csv
```

## Procedure Details

### 1. Gloss encoding

The pipeline loads the gloss CSV, converts each gloss to a sentence embedding, normalizes the vectors, and writes both raw and normalized `.npz` files.

### 2. RotE loading and alignment

The RotE checkpoint is inspected for the entity embedding tensor. Concept ids are matched against the `entity_to_id` mapping. If a concept id is not found directly, the pipeline can fall back to a label from `dataset/concepts.csv` when available.

### 3. Mapper training

The aligned gloss embeddings and RotE embeddings are split into train/validation/test sets. A small feed-forward mapper is then trained with cosine loss.

### 4. Evaluation

After training, the pipeline computes nearest-neighbor rankings in RotE space and stores them in `topk_neighbors.csv`. If geodesic analysis is enabled, it also computes graph distances on `dataset/concept_relations.csv`.

## Notes

- The pipeline expects `dataset/concept_glosses.csv` to be the source of truth for glosses.
- The project uses a unified implementation path rather than separate notebook branches.
- If `sentence-transformers` or `networkx` is missing, the relevant stage will fail or be skipped depending on the feature.

## Suggested Validation Steps

1. Run `conda run -n ANLP python -m py_compile mapping/*.py mapping/model/*.py mapping/utils/*.py`.
2. Run the pipeline on the default inputs.
3. Confirm the output directory contains the NPZ, checkpoint, and CSV artifacts.
4. Inspect `topk_neighbors.csv` and, if enabled, `geodesic_distances.csv` for expected sizes.