"""Default configuration for standalone WSD training."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_TSV = str(PROJECT_ROOT / "dataset/train")
EVAL_TSV = str(PROJECT_ROOT / "dataset/eval")
TEST_TSV = str(PROJECT_ROOT / "dataset/test")

ROTE_MODEL = str(PROJECT_ROOT / "RotE_model_20251201_144211_best.pt")
ENTITY_TO_ID = str(PROJECT_ROOT / "dataset/RotE/entity_to_id.pickle")
CONCEPTS_CSV = str(PROJECT_ROOT / "dataset/concepts.csv")
CONCEPT_REL_CSV = str(PROJECT_ROOT / "dataset/concept_relations.csv")
HIERARCHY_CACHE_PATH = str(PROJECT_ROOT / "WSD/hierarchy_cache.pt")

TRAINING_MODE = "encoder"
DATA_FILE = str(PROJECT_ROOT / "dataset/embs/sentence_embeddings_FacebookAI_xlm-roberta-large.npz")

ENCODER_MODEL = "xlm-roberta-base"
ENCODER_TRAIN_MODE = "lora"
ENCODER_TOKEN_POOLING = "target_last_subword"
MAX_SEQUENCE_LENGTH = 128

LORA_R = 8
LORA_ALPHA = 16
DROPOUT = 0.1

FORMATTER_PRESET = "plain_sentence"
FORMATTER_TEMPLATE = None

NUM_EPOCHS = 1
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = "cuda"

USE_EXTRA_NEGATIVE_SAMPLING = False
NUM_EXTRA_NEGATIVES = 64

USE_HIERARCHY_LOSS = True
HIERARCHY_LOSS_TYPE = "factorized"
WEIGHTED_HIERARCHY_ALPHA = 0.5
EVAL_USE_CANDIDATES = True

USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.0
EARLY_STOPPING_MONITOR = "eval_f1"
EARLY_STOPPING_MODE = "max"

BASELINE = False
BASELINE_TYPE = "linear"
SIMILARITY_METRIC = "cosine"

PROJECT_NUM_LAYERS = 1

OUTPUT_DIR = str(PROJECT_ROOT / "WSD/outputs")
WANDB_PROJECT = "wsd-unified"
USE_WANDB = True

EVAL_GOLD_STANDARD_DIR = str(PROJECT_ROOT / "dataset/eval-gold-standard")
TEST_GOLD_STANDARD_DIR = str(PROJECT_ROOT / "dataset/test-gold-standard")
SCORER_JAVA_PATH = str(PROJECT_ROOT / "WSD/Scorer.java")
