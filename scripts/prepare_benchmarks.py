from __future__ import annotations

import csv
import io
import re
import unicodedata
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

from build_final_datasets import ALPHABET, clean_text

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "data" / "benchmarks"
ASR_CORPUS = Path.home() / "asr_project" / "data" / "karakalpak_speech_corpus" / "parquet_files"
WHISPER_PROJECT = Path.home() / "whisper_kaa"

FOREIGN_LETTERS = str.maketrans({"é": "e", "ü": "ú", "ö": "ó"})
HYPHENATED_WORD_PAIR = re.compile(r"([^\W\d_]+)-(?=([^\W\d_]+))")
MINUS_WORD = "alıw"

CSV_FIELDS = ["audio_path", "text", "duration"]
METADATA_FIELDS = [*CSV_FIELDS, "source_id", "text_original", "text_rules", "excluded", "other_prediction"]


def without_punctuation(text):
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in text).split())


def normalize(text):
    text = unicodedata.normalize("NFC", str(text)).lower().translate(FOREIGN_LETTERS)
    return clean_text(without_punctuation(text))


def restore_hyphens_written_as_minus(raw_text, text):
    restored = 0
    for left, right in HYPHENATED_WORD_PAIR.findall(raw_text.lower()):
        fixed = text.replace(f"{left} {MINUS_WORD} {right}", f"{left} {right}", 1)
        restored += fixed != text
        text = fixed
    still_doubtful = text.count(MINUS_WORD) > raw_text.lower().count(MINUS_WORD)
    return text, restored, still_doubtful


def exclusion_reason(text, still_doubtful):
    if still_doubtful:
        return "possible false alıw from a hyphen"
    outside = sorted(set(text) - ALPHABET)
    if outside or not text:
        return f"characters outside the alphabet: {' '.join(outside)}"
    return ""


def benchmark_row(audio_path, text_original, duration, source_id, reference=None, still_doubtful=False,
                  extra_rule="", other_prediction=""):
    text, rules = normalize(text_original if reference is None else reference)
    return {
        "audio_path": audio_path,
        "text": text,
        "duration": duration,
        "source_id": source_id,
        "text_original": text_original,
        "text_rules": ";".join(rule for rule in (extra_rule, rules) if rule),
        "excluded": exclusion_reason(text, still_doubtful),
        "other_prediction": normalize(other_prediction)[0] if other_prediction else "",
    }


def write_rows(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_benchmark(folder, name, rows):
    kept = [row for row in rows if not row["excluded"]]
    folder.mkdir(parents=True, exist_ok=True)
    write_rows(folder / f"{name}.csv", kept, CSV_FIELDS)
    write_rows(folder / f"{name}_metadata.csv", rows, METADATA_FIELDS)
    hours = sum(row["duration"] for row in kept) / 3600
    print(f"{folder.name}/{name}.csv: {len(kept)} clips ({len(rows) - len(kept)} excluded), {hours:.2f} h")


def save_audio(folder, split, record):
    data = record["cleaned_audio"]["bytes"]
    suffix = Path(record["cleaned_audio"]["path"]).suffix or ".ogg"
    audio_path = f"{split}/audio/{record['id']}{suffix}"
    (folder / audio_path).parent.mkdir(parents=True, exist_ok=True)
    (folder / audio_path).write_bytes(data)
    return audio_path, sf.info(io.BytesIO(data)).duration


def asr_corpus():
    folder = BENCHMARKS / "asr_corpus"
    for split in ("test", "validation"):
        rows = []
        for shard in sorted(ASR_CORPUS.glob(f"{split}-*.parquet")):
            for record in pq.read_table(shard, columns=["id", "cleaned_audio", "raw_text", "text"]).to_pylist():
                audio_path, duration = save_audio(folder, split, record)
                reference, restored, still_doubtful = restore_hyphens_written_as_minus(record["raw_text"], record["text"])
                extra_rule = f"hyphen-alıw×{restored}" if restored else ""
                rows.append(benchmark_row(audio_path, record["text"], duration, record["id"], reference,
                                          still_doubtful, extra_rule))
        write_benchmark(folder, split, rows)


def whisper_test():
    manifest = pd.read_csv(WHISPER_PROJECT / "manifest_tayyor.csv", keep_default_na=False)
    predictions = pd.read_csv(WHISPER_PROJECT / "07_eval" / "test_finetune_beam5.csv", keep_default_na=False)
    whisper_prediction = dict(zip(predictions["idx"], predictions["hypothesis"]))
    rows = []
    for item in manifest[manifest["split"] == "test"].itertuples():
        path = WHISPER_PROJECT / item.fayl.replace("\\", "/")
        if path.exists():
            rows.append(benchmark_row(str(path), item.matn_model, float(item.yangi_s), item.idx,
                                      other_prediction=whisper_prediction.get(item.idx, "")))
    write_benchmark(BENCHMARKS / "whisper_test", "test", rows)


if __name__ == "__main__":
    asr_corpus()
    whisper_test()
