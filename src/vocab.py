import json
from pathlib import Path

TRANSCRIPTS_PATH = "data/text/transcripts.txt"
VOCAB_PATH = "vocab.json"

def build_vocab(transcripts_path):
    characters = set()

    with open(transcripts_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip().lower()

            for char in text:
                characters.add(char)

        characters.discard(" ")

        vocab = {
            char: index
            for index, char in enumerate(sorted(characters))
        }

        vocab["|"] = len(vocab)
        vocab["[UNK]"] = len(vocab)
        vocab["[PAD]"] = len(vocab)

        return vocab

def save_vocab(vocab, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            vocab,
            f,
            ensure_ascii=False,
            indent=2
        )


if __name__ == "__main__":
    vocab = build_vocab(TRANSCRIPTS_PATH)
    save_vocab(vocab, VOCAB_PATH)

    print(vocab)
    print("Vocabulary size:", len(vocab))