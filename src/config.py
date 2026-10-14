from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "final"

# Split CSVs inside DATA_DIR. Each CSV's audio_path is relative to the CSV's own folder.
TRAIN_CSVS = ["stt/train.csv",]
VALIDATION_CSVS = ["stt/validation.csv"]
TEST_CSVS = {"stt": "stt/test.csv", "hf": "hf/test.csv"}

BASE_MODEL = "facebook/wav2vec2-xls-r-300m"
OUTPUT_DIR = ROOT / "outputs" / "wav2vec2-lora-kaa"
VOCAB_PATH = OUTPUT_DIR / "vocab.json"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FINAL_MODEL_DIR = OUTPUT_DIR / "final_model"
RESULTS_DIR = OUTPUT_DIR / "results"

EPOCHS = 4
BATCH_SIZE = 16  # x GRADIENT_ACCUMULATION = 32 files per optimizer step
GRADIENT_ACCUMULATION = 2
LEARNING_RATE = 5e-4
WARMUP_RATIO = 0.1

LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

NUM_WORKERS = 8
SEED = 42

# KenLM n-gram language model, built from the train transcripts by lm.py.
KENLM_BIN = ROOT / "kenlm" / "build" / "bin"
LM_ORDER = 4
LM_DIR = OUTPUT_DIR / "lm"
LM_PATH = LM_DIR / f"kaa_{LM_ORDER}gram.arpa"

# Decoders compared by decode.py: greedy, beam search, beam search + KenLM shallow fusion.
BEAM_WIDTH = 100
ALPHAS = [0.3, 0.5, 0.7, 1.0]
BETAS = [0.0, 1.0, 2.0]
TUNE_FILES = 500
DECODE_PROCESSES = 16
