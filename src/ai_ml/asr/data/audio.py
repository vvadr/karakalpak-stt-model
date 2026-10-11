from pathlib import Path

import soundfile as sf
import torch

SAMPLE_RATE = 16000

def load_audio(path: str | Path) -> torch.Tensor:
    audio, sample_rate = sf.read(path, dtype="float16")

    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sample_rate} Hz")
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {audio.shape}")

    audio = torch.from_numpy(audio)

    return audio