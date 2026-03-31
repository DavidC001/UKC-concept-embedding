"""Configuration and hyperparameters for WSD training."""

# Data
DATA_FILE = '../sentence_embeddings_labels_loc_FacebookAI_xlm_roberta_large.npz'
ROTE_MODEL = '../RotE_model_20251201_144211_best.pt'


# ROTE_MODEL = 'RotH_model_20251130_144024_best.pt'
TRAIN_TSV = '../dataset/train'
EVAL_TSV = '../dataset/eval'
TEST_TSV = '../dataset/test'
DATASET_DIR = '../dataset'

# Training
NUM_EPOCHS = 15
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9
OPTIM = 'Adam'
DEVICE = 'cuda'  # Use CPU by default, change to 'cuda' if available
TEMPERATURE = 0.1

# Model
INPUT_DIM = 1024
HIDDEN_DIM = 1024
OUTPUT_DIM = 500
DROPOUT = 0.1
NUM_CONCEPTS = 37123

# Data limits (for testing)
LIMIT_TRAIN_SAMPLES = None  # Set to number for testing (e.g., 32)
MAX_SEQUENCE_LENGTH = 128

# Logging
LOG_INTERVAL = 10  # Print metrics every N batches
USE_WANDB = True
WANDB_PROJECT = 'wsd-experiments'

# Output
OUTPUT_DIR = 'outputs'
MODEL_SAVE_PATH = 'concept_classifier.pth'
EVAL_PREDICTIONS_FILE = 'eval_predictions.txt'
TEST_PREDICTIONS_FILE = 'test_predictions.txt'

EVAL_GOLD_STANDARD_DIR = '../dataset/eval-gold-standard'
TEST_GOLD_STANDARD_DIR = '../dataset/test-gold-standard'
