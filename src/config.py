from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "karakalpak-combined"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

VOCAB_PATH = ROOT / "vocab.json"

MODEL_NAME = "facebook/wav2vec2-xls-r-300m"

OUTPUT_DIR = ROOT / "outputs" / "qkvo_r16"

LM_TEXT_PATH = ROOT / "lm_text.txt"
KENLM_PATH = ROOT / "kaa.arpa"

SAMPLE_RATE = 16000

BATCH_SIZE = 1
EPOCHS = 3
LEARNING_RATE = 1e-4

GRAD_ACCUMULATION = 4
WARMUP_RATIO = 0.1

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
]

BEAM_WIDTH = 50

LM_ALPHA = 0.5
LM_BETA = 1.0

SEED = 42
LOG_EVERY = 50