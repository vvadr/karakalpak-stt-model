import subprocess

import config
from dataset import load_split


def main():
    held_out = set(load_split(config.VALIDATION_CSVS + list(config.TEST_CSVS.values()))["text"])
    train_text = load_split(config.TRAIN_CSVS)["text"]
    sentences = [text for text in train_text if text not in held_out]

    config.LM_DIR.mkdir(parents=True, exist_ok=True)
    text_path = config.LM_DIR / "lm_text.txt"
    text_path.write_text("\n".join(sentences) + "\n", encoding="utf-8")
    print(f"LM text: {len(sentences)} sentences ({len(train_text) - len(sentences)} removed because they are in validation/test)")

    lmplz = config.KENLM_BIN / "lmplz"
    with open(text_path, "rb") as source, open(config.LM_PATH, "wb") as target:
        subprocess.run([str(lmplz), "-o", str(config.LM_ORDER), "--discount_fallback"], stdin=source, stdout=target, check=True)

    print("Saved to:", config.LM_PATH)


if __name__ == "__main__":
    main()
