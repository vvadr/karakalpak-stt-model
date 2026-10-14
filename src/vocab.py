import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "karakalpak-combined" / "train.csv"
VOCAB_PATH = ROOT / "vocab.json"


def build_vocab():
    df = pd.read_csv(CSV_PATH)
    characters = set()

    for text in df["text"].dropna().astype(str):
        text = text.strip().lower()
        characters.update(text)

    characters.discard(" ")

    vocab = {char: index for index, char in enumerate(sorted(characters))}
    vocab["|"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)

    return vocab


def save_vocab(vocab):
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    vocab = build_vocab()
    save_vocab(vocab)

    print(vocab)
    print("Vocabulary size:", len(vocab))
    print("Saved to:", VOCAB_PATH)
