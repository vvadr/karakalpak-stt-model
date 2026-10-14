from pathlib import Path

import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset

from processor import build_processor


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "karakalpak-combined"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

TRAIN_AUDIO_DIR = DATA_DIR / "train"
TEST_AUDIO_DIR = DATA_DIR / "test"


class SpeechDataset(Dataset):
    def __init__(self, csv_path, audio_dir, processor):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = Path(audio_dir)
        self.processor = processor
        self.audio_column = self.find_column(["audio", "path", "file", "filename", "audio_path", "wav", "wav_path"])
        self.text_column = self.find_column(["text", "transcript", "sentence"])

    def find_column(self, candidates):
        for column in candidates:
            if column in self.df.columns:
                return column

        raise ValueError(f"Could not find required column. Available columns: {list(self.df.columns)}")

    def resolve_audio_path(self, value):
        raw_path = Path(str(value))

        candidates = [
            raw_path,
            DATA_DIR / raw_path,
            self.audio_dir / raw_path,
            self.audio_dir / raw_path.name
        ]

        for path in candidates:
            if path.is_file():
                return path

        raise FileNotFoundError(f"Could not find audio file for: {value}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        audio_path = self.resolve_audio_path(row[self.audio_column])
        audio, sampling_rate = sf.read(audio_path)

        if sampling_rate != 16000:
            raise ValueError(f"{audio_path} has sampling rate {sampling_rate}, expected 16000")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        text = str(row[self.text_column]).strip().lower()

        input_values = self.processor(audio, sampling_rate=16000).input_values[0]
        labels = self.processor.tokenizer(text).input_ids

        return {"input_values": input_values, "labels": labels}


def build_datasets():
    processor = build_processor()
    train_dataset = SpeechDataset(TRAIN_CSV, TRAIN_AUDIO_DIR, processor)
    test_dataset = SpeechDataset(TEST_CSV, TEST_AUDIO_DIR, processor)

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = build_datasets()

    print("Train samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))

    sample = train_dataset[0]

    print("Input length:", len(sample["input_values"]))
    print("Label IDs:", sample["labels"])
