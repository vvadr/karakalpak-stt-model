from pathlib import Path

import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset


class SpeechDataset(Dataset):
    def __init__(self, csv_path, processor):
        self.csv_path = Path(csv_path)
        self.data_root = self.csv_path.parent

        self.df = pd.read_csv(csv_path)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        audio_path = Path(row["audio_path"])

        if not audio_path.is_absolute():
            audio_path = self.data_root / audio_path

        audio, sample_rate = sf.read(audio_path)

        if sample_rate != 16000:
            raise ValueError(
                f"{audio_path} is {sample_rate} Hz, expected 16000 Hz"
            )

        text = row["text"]

        input_values = self.processor(
            audio,
            sampling_rate=16000
        ).input_values[0]

        labels = self.processor.tokenizer(
            text
        ).input_ids

        return {
            "input_values": input_values,
            "labels": labels
        }