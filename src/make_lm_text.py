import pandas as pd

from config import (
    TRAIN_CSV,
    LM_TEXT_PATH,
)


df = pd.read_csv(TRAIN_CSV)

with open(
    LM_TEXT_PATH,
    "w",
    encoding="utf-8",
) as f:

    for text in df["text"].dropna().astype(str):
        text = text.strip().lower()

        if text:
            f.write(text + "\n")


print(
    "Saved:",
    LM_TEXT_PATH
)