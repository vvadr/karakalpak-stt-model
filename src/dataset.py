import pandas as pd
from datasets import Dataset

import config


def load_split(csv_names):
    frames = []

    for name in csv_names:
        csv_path = config.DATA_DIR / name
        frame = pd.read_csv(csv_path, dtype={"text": str}, keep_default_na=False)
        frame["audio_path"] = [str(csv_path.parent / path) for path in frame["audio_path"]]
        frames.append(frame)

    return Dataset.from_pandas(pd.concat(frames, ignore_index=True), preserve_index=False)


if __name__ == "__main__":
    for split, names in [("train", config.TRAIN_CSVS), ("validation", config.VALIDATION_CSVS), ("test", list(config.TEST_CSVS.values()))]:
        data = load_split(names)
        print(f"{split}: {len(data)} files, {sum(data['duration']) / 3600:.2f} hours")

    print("Example:", load_split(config.TRAIN_CSVS)[0])
