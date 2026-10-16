import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "final"
TRAIN_CSVS = ["stt/train.csv", "hf/train.csv"]
VALIDATION_CSVS = ["stt/validation.csv"]
TEST_CSVS = {"stt": "stt/test.csv", "hf": "hf/test.csv"}

SAMPLING_RATE = 16000
WORD_SEPARATOR = "|"
UNKNOWN_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"


BASE_MODEL = "facebook/wav2vec2-xls-r-300m"
OUTPUT_DIR = ROOT / "outputs" / "wav2vec2-lora-kaa"

EPOCHS = 4
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 5e-4
WARMUP_RATIO = 0.1
GRADIENT_CHECKPOINTING = False

LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

NUM_WORKERS = 8
SEED = 42

TRAIN_SUBSET_SIZE = None
EVAL_SUBSET_SIZE = None
SUBSET_MAX_SECONDS = None

# RUN_ON_MAC = bool(os.environ.get("LOCAL"))
# if RUN_ON_MAC:
#     OUTPUT_DIR = ROOT / "outputs" / "wav2vec2-lora-kaa-local"
#     TRAIN_SUBSET_SIZE = 3000
#     EVAL_SUBSET_SIZE = 300
#     SUBSET_MAX_SECONDS = 20
#     EPOCHS = 5
#     BATCH_SIZE = 4
#     GRADIENT_ACCUMULATION = 4
#     NUM_WORKERS = 0
#     GRADIENT_CHECKPOINTING = True

VOCAB_PATH = OUTPUT_DIR / "vocab.json"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FINAL_MODEL_DIR = OUTPUT_DIR / "final_model"
RESULTS_DIR = OUTPUT_DIR / "results"
BENCHMARK_RESULTS_DIR = RESULTS_DIR / "benchmarks"

KENLM_BIN = ROOT / "kenlm" / "build" / "bin"
LMPLZ_MEMORY_LIMIT = "1G"
LM_ORDER = 4
LM_DIR = OUTPUT_DIR / "lm"
LM_TEXT_PATH = LM_DIR / "lm_text.txt"
LM_PATH = LM_DIR / f"kaa_{LM_ORDER}gram.arpa"

BEAM_WIDTH = 100
LM_WEIGHT_CANDIDATES = [0.3, 0.5, 0.7, 1.0]
WORD_BONUS_CANDIDATES = [0.0, 1.0, 2.0]
TUNING_FILES = 500
DECODE_PROCESSES = 16
