import hashlib
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset, Sampler
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor


def normalize_text(text):
    # Preserve the corpus alphabet, including accents and dotless i.
    return " ".join(unicodedata.normalize("NFC", str(text)).lower().split())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_manifest(root, split):
    df = pd.read_csv(Path(root) / f"{split}.csv", keep_default_na=False)
    required = {"audio_path", "text", "duration"}
    if not required <= set(df.columns):
        raise ValueError(f"{split}.csv needs {required}")
    df["text"] = df["text"].map(normalize_text)
    if (df.text.str.len() == 0).any() or df.audio_path.duplicated().any():
        raise ValueError(f"Empty transcripts or duplicate paths in {split}.csv")
    return df


def audit_dataset(root, train, test):
    root = Path(root).resolve()
    report = {"dataset": str(root), "normalization": "NFC, lowercase, single spaces"}
    hashes = {}
    for name, df in (("train", train), ("test", test)):
        hashes[name] = set()
        peaks, rms = [], []
        for row in df.itertuples():
            path = (root / row.audio_path).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Audio path outside dataset: {path}")
            audio, rate = sf.read(path, dtype="float32")
            if rate != 16000 or audio.ndim != 1 or not np.isfinite(audio).all():
                raise ValueError(f"Invalid mono 16 kHz audio: {path}")
            if not len(audio) or abs(len(audio) / rate - row.duration) > 0.02:
                raise ValueError(f"Empty audio or wrong duration: {path}")
            peak = float(np.max(np.abs(audio)))
            if peak < 1e-4:
                raise ValueError(f"Silent audio: {path}")
            peaks.append(peak)
            rms.append(float(np.sqrt(np.mean(audio ** 2))))
            hashes[name].add(hashlib.sha256(audio.tobytes()).hexdigest())
        report[name] = {
            "recordings": len(df), "hours": float(df.duration.sum() / 3600),
            "duration_seconds": df.duration.describe().to_dict(),
            "unique_audio_hashes": len(hashes[name]),
            "peak_range": [min(peaks), max(peaks)], "rms_range": [min(rms), max(rms)],
            "manifest_sha256": hashlib.sha256((root / f"{name}.csv").read_bytes()).hexdigest(),
        }
    report["audio_overlap"] = len(hashes["train"] & hashes["test"])
    report["transcript_overlap"] = len(set(train.text) & set(test.text))
    report["train_character_counts"] = dict(sorted(Counter("".join(train.text)).items()))
    report["test_only_characters"] = sorted(set("".join(test.text)) - set("".join(train.text)))
    if report["audio_overlap"] or report["transcript_overlap"] or report["test_only_characters"]:
        raise ValueError(f"Split leakage or unknown test characters: {report}")
    return report


def split_training(train, fraction, seed):
    groups = sorted(set(train.text))
    random.Random(seed).shuffle(groups)
    validation_groups = set(groups[:max(1, round(len(groups) * fraction))])
    # Keep at least one training example for every character, even rare letters.
    for character in set("".join(train.text)):
        if not any(character in text for text in groups if text not in validation_groups):
            validation_groups.remove(next(text for text in groups if character in text))
    validation = train[train.text.isin(validation_groups)].reset_index(drop=True)
    training = train[~train.text.isin(validation_groups)].reset_index(drop=True)
    return training, validation


def create_processor(texts, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # set(texts) would collect sentences. Join first to collect individual letters.
    letters = sorted(set("".join(texts)) - {" "})
    vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2}
    vocab.update({letter: index + 3 for index, letter in enumerate(letters)})
    write_json(directory / "vocab.json", vocab)
    tokenizer = Wav2Vec2CTCTokenizer(
        str(directory / "vocab.json"), unk_token="[UNK]", pad_token="[PAD]",
        word_delimiter_token="|", bos_token=None, eos_token=None, do_lower_case=False,
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=True, return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(feature_extractor, tokenizer)
    processor.save_pretrained(directory)
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if tokenizer.unk_token_id in ids or tokenizer.decode(ids, group_tokens=False) != text:
            raise ValueError(f"Tokenizer does not round trip: {text}")
    return processor


class SpeechDataset(Dataset):
    def __init__(self, root, frame, processor):
        self.root = Path(root)
        self.rows = frame.to_dict("records")
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        audio, rate = sf.read(self.root / row["audio_path"], dtype="float32")
        values = self.processor.feature_extractor(audio, sampling_rate=rate).input_values[0]
        labels = self.processor.tokenizer(row["text"], add_special_tokens=False).input_ids
        return {"input_values": values, "labels": labels, "text": row["text"],
                "audio_path": row["audio_path"]}


class Collator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        batch = self.processor.feature_extractor.pad(
            [{"input_values": x["input_values"]} for x in examples],
            padding=True, return_tensors="pt", return_attention_mask=True,
        )
        labels = self.processor.tokenizer.pad(
            [{"input_ids": x["labels"]} for x in examples], padding=True, return_tensors="pt",
        )
        batch["labels"] = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        batch["texts"] = [x["text"] for x in examples]
        batch["paths"] = [x["audio_path"] for x in examples]
        return batch


class DurationBatches(Sampler):
    """Randomize batches while keeping similarly long recordings together."""
    def __init__(self, frame, batch_size, seed, shuffle=True):
        self.durations = frame.duration.tolist()
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def __len__(self):
        return (len(self.durations) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.durations)))
        if self.shuffle:
            rng.shuffle(order)
        # Local sorting keeps randomness without padding every clip to the longest.
        window = self.batch_size * 50
        order = [i for start in range(0, len(order), window)
                 for i in sorted(order[start:start + window], key=lambda i: self.durations[i])]
        batches = [order[i:i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches
