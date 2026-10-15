import subprocess

import config
from dataset import load_split


def lm_sentences():
    held_out = set(load_split(config.VALIDATION_CSVS + list(config.TEST_CSVS.values()))["text"])
    train_text = load_split(config.TRAIN_CSVS)["text"]
    sentences = [text for text in train_text if text not in held_out]
    print(f"LM text: {len(sentences)} sentences ({len(train_text) - len(sentences)} removed because they are in validation/test)")
    return sentences


def run_lmplz(text_path, arpa_path):
    lmplz = config.KENLM_BIN / "lmplz"
    command = [str(lmplz), "-o", str(config.LM_ORDER), "--discount_fallback", "-S", config.LMPLZ_MEMORY_LIMIT]
    unfinished = arpa_path.with_suffix(".partial")
    with open(text_path, "rb") as source, open(unfinished, "wb") as target:
        subprocess.run(command, stdin=source, stdout=target, check=True)
    unfinished.replace(arpa_path)


def main():
    config.LM_DIR.mkdir(parents=True, exist_ok=True)
    config.LM_TEXT_PATH.write_text("\n".join(lm_sentences()) + "\n", encoding="utf-8")
    run_lmplz(config.LM_TEXT_PATH, config.LM_PATH)
    print("Saved to:", config.LM_PATH)


if __name__ == "__main__":
    main()
