import pandas as pd
from datasets import Dataset

import config


def load_split(csv_names, limit=None):
    frames = []

    for name in csv_names:
        csv_path = config.DATA_DIR / name
        frame = pd.read_csv(csv_path, dtype={"text": str}, keep_default_na=False)
        frame["audio_path"] = [str(csv_path.parent / path) for path in frame["audio_path"]]
        frames.append(frame)

    frame = pd.concat(frames, ignore_index=True)

    if limit:
        if config.MAX_SECONDS:
            frame = frame[frame["duration"] <= config.MAX_SECONDS]
        frame = frame.sample(n=min(limit, len(frame)), random_state=config.SEED)

    return Dataset.from_pandas(frame, preserve_index=False)


if __name__ == "__main__":
    for split, names, limit in [("train", config.TRAIN_CSVS, config.TRAIN_FILES), ("validation", config.VALIDATION_CSVS, config.EVAL_FILES), ("test", list(config.TEST_CSVS.values()), config.EVAL_FILES)]:
        data = load_split(names, limit)
        print(f"{split}: {len(data)} files, {sum(data['duration']) / 3600:.2f} hours")

    print("Example:", load_split(config.TRAIN_CSVS)[0])
