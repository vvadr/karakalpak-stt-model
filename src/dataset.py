import pandas as pd
from datasets import Dataset

import config


def read_split_csv(csv_path):
    frame = pd.read_csv(csv_path, dtype={"text": str}, keep_default_na=False)
    frame["audio_path"] = [str(csv_path.parent / path) for path in frame["audio_path"]]
    return frame


def random_subset(frame, size):
    if config.SUBSET_MAX_SECONDS:
        frame = frame[frame["duration"] <= config.SUBSET_MAX_SECONDS]
    return frame.sample(n=min(size, len(frame)), random_state=config.SEED)


def load_split(csv_names, subset_size=None):
    frame = pd.concat([read_split_csv(config.DATA_DIR / name) for name in csv_names], ignore_index=True)
    if subset_size:
        frame = random_subset(frame, subset_size)
    return Dataset.from_pandas(frame, preserve_index=False)


def print_split_sizes():
    splits = {
        "train": (config.TRAIN_CSVS, config.TRAIN_SUBSET_SIZE),
        "validation": (config.VALIDATION_CSVS, config.EVAL_SUBSET_SIZE),
        "test": (list(config.TEST_CSVS.values()), config.EVAL_SUBSET_SIZE),
    }
    for split, (csv_names, subset_size) in splits.items():
        data = load_split(csv_names, subset_size)
        print(f"{split}: {len(data)} files, {sum(data['duration']) / 3600:.2f} hours")


if __name__ == "__main__":
    print_split_sizes()
