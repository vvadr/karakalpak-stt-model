from pathlib import Path

import pandas as pd
import soundfile as sf

from torch.utils.data import Dataset

from config import ROOT, SAMPLE_RATE


class SpeechDataset(Dataset):
    def __init__(self, csv_path, processor):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(csv_path)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        audio_path = Path(str(row["audio_path"]))

        if not audio_path.is_absolute():
            local_path = self.csv_path.parent / audio_path

            if local_path.exists():
                audio_path = local_path
            else:
                audio_path = ROOT / audio_path

        audio, sample_rate = sf.read(
            audio_path,
            dtype="float32"
        )

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"{audio_path}: expected {SAMPLE_RATE} Hz, got {sample_rate}"
            )

        text = str(row["text"]).strip().lower()

        input_values = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE
        ).input_values[0]

        labels = self.processor.tokenizer(
            text
        ).input_ids

        return {
            "input_values": input_values,
            "labels": labels,
        }


class DataCollatorCTC:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [
            {
                "input_values": item["input_values"]
            }
            for item in features
        ]

        label_features = [
            {
                "input_ids": item["labels"]
            }
            for item in features
        ]

        batch = self.processor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=True,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1),
            -100,
        )

        batch["labels"] = labels

        return batch