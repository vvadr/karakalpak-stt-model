from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

VOCAB_SIZE = 1000

SPECIAL_TOKENS = ["<blank>", "<unk>", "<pad>"]

def build_tokenizer():
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    return tokenizer

def get_text(csv_path):
    dataframe = pd.read_csv(csv_path)

    return dataframe["text"].astype(str).tolist()

def train_tokenizer(csv_path):
    tokenizer = build_tokenizer()

    texts = get_text(csv_path)

    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)

    tokenizer.train_from_iterator(texts, trainer=trainer)

    return tokenizer

def save_tokenizer(tokenizer, path):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    tokenizer.save(str(path))