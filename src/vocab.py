import json

import config
from dataset import load_split


def build_vocab():
    text = "".join(load_split(config.TRAIN_CSVS)["text"])
    characters = sorted(set(text) - {" "})

    vocab = {character: index for index, character in enumerate(characters)}
    vocab["|"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)

    return vocab


def main():
    vocab = build_vocab()

    config.VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.VOCAB_PATH.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    other_text = "".join(load_split(config.VALIDATION_CSVS + list(config.TEST_CSVS.values()))["text"])
    missing = sorted(set(other_text) - set(vocab) - {" "})

    print("Letters:", " ".join(character for character in vocab if len(character) == 1 and character != "|"))
    print("Vocabulary size:", len(vocab))
    print("Validation/test letters missing from the vocabulary:", " ".join(missing) or "none")
    print("Saved to:", config.VOCAB_PATH)


if __name__ == "__main__":
    main()
