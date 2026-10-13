import pandas as pd
import json

TRAIN_CSV = "data/karakalpak-combined/train.csv"
VOCAB_PATH = "vocab.json"


def build_vocab(csv_path):
    df = pd.read_csv(csv_path)

    characters = set()

    for text in df["text"].astype(str):
        text = text.strip().lower()

        for char in text:
            characters.add(char)

    characters.discard(" ")

    vocab = {
        char: i
        for i, char in enumerate(sorted(characters))
    }

    vocab["|"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)

    return vocab


vocab = build_vocab(TRAIN_CSV)

with open(VOCAB_PATH, "w", encoding="utf-8") as f:
    json.dump(
        vocab,
        f,
        ensure_ascii=False,
        indent=2
    )

print(vocab)
print("Vocabulary size:", len(vocab))