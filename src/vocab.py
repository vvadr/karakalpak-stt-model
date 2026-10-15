import json

import config
from dataset import load_split


def letters_in(csv_names):
    text = "".join(load_split(csv_names)["text"])
    return sorted(set(text) - {" "})


def build_vocab(letters):
    tokens = letters + [config.WORD_SEPARATOR, config.UNKNOWN_TOKEN, config.PAD_TOKEN]
    return {token: index for index, token in enumerate(tokens)}


def save_vocab(vocab):
    config.VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.VOCAB_PATH.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    letters = letters_in(config.TRAIN_CSVS)
    vocab = build_vocab(letters)
    save_vocab(vocab)

    evaluation_letters = letters_in(config.VALIDATION_CSVS + list(config.TEST_CSVS.values()))
    missing = [letter for letter in evaluation_letters if letter not in vocab]

    print("Letters:", " ".join(letters))
    print("Vocabulary size:", len(vocab))
    print("Validation/test letters missing from the vocabulary:", " ".join(missing) or "none")
    print("Saved to:", config.VOCAB_PATH)


if __name__ == "__main__":
    main()
