from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from .audio import load_audio

class ASRDataset(Dataset):
    def __init__(self, csv_path: str | Path, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.dataframe = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        audio_path = self.data_dir / row["audio_path"]

        audio = load_audio(audio_path)

        return {
            "audio": audio,
            "text": row["text"],
            "duration": float(row["duration"])
        }