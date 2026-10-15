from __future__ import annotations

import csv
import hashlib
import io
import random
import sys
import wave
from collections import defaultdict

import pyarrow.parquet as pq

from build_final_datasets import (ABBREVIATION_GLUED_TO_NUMBER, ALPHABET, CHANGE_FIELDS, COMBINED, DATA, DATASETS, FINAL,
                                  METADATA_FIELDS, NUMBER_SPELLINGS, NUMBER_WORDS, REVIEW_FIELDS, ROOT, STT, STT_SPLITS,
                                  SUFFIXES_AFTER_ANY_WORD, SUFFIXES_AFTER_NUMBERS, TRAIN_FIELDS, clean_text)

CLEANED = DATA / "datasets" / "TOZALANGAN_TRAIN_FINAL"
failures = []


def check(condition, message):
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        failures.append(message)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def wav(data):
    with wave.open(io.BytesIO(data), "rb") as audio:
        fmt = (audio.getframerate(), audio.getnchannels(), audio.getsampwidth())
        frames = audio.readframes(audio.getnframes())
        complete = len(frames) == audio.getnframes() * 2
    return fmt, frames, complete


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def check_folders():
    present = {p.name for p in FINAL.iterdir() if p.is_dir()}
    check(present == set(DATASETS), f"data/final holds exactly {sorted(DATASETS)}")


def check_layout(source):
    folder = FINAL / source
    fields, metadata = read_csv(folder / "metadata.csv")
    check(fields == METADATA_FIELDS, f"{source}: metadata.csv has the expected columns")
    split_folders = {p.name for p in folder.iterdir() if (p / "audio").is_dir()}
    check(split_folders == set(DATASETS[source]), f"{source}: split folders are {DATASETS[source]}")

    union = []
    for split in DATASETS[source]:
        fields, rows = read_csv(folder / f"{split}.csv")
        on_disk = {f"{split}/audio/{p.name}" for p in (folder / split / "audio").iterdir()}
        listed = {row["audio_path"] for row in rows}
        check(fields == TRAIN_FIELDS and on_disk == listed and len(listed) == len(rows),
              f"{source}/{split}: {len(rows)} CSV rows match the WAV files on disk exactly")
        union += rows

    simplified = [{k: row[k] for k in TRAIN_FIELDS} for row in metadata]
    check(sorted(tuple(r.values()) for r in union) == sorted(tuple(r.values()) for r in simplified),
          f"{source}: split CSVs together equal metadata.csv")
    return metadata


def check_audio(source, metadata):
    bad_format = bad_duration = bad_hash = 0
    for row in metadata:
        fmt, frames, complete = wav((FINAL / source / row["audio_path"]).read_bytes())
        bad_format += fmt != (16000, 1, 2) or not complete or not frames
        bad_duration += abs(len(frames) / 32000 - float(row["duration"])) > 1e-6
        bad_hash += sha256(frames) != row["pcm_sha256"]
    check(bad_format == 0 and bad_duration == 0 and bad_hash == 0,
          f"{source}: all {len(metadata)} WAVs are complete mono 16 kHz PCM16 with matching duration and PCM SHA256")


def has_leftover_number_words(words):
    split_suffix = any(a in NUMBER_WORDS and b in SUFFIXES_AFTER_NUMBERS for a, b in zip(words, words[1:]))
    leftover_word = any(word in SUFFIXES_AFTER_ANY_WORD or word in NUMBER_SPELLINGS
                        or ABBREVIATION_GLUED_TO_NUMBER.fullmatch(word) for word in words)
    return split_suffix or leftover_word


def check_text(source, metadata):
    empty = spacing = undocumented_edits = leftovers = 0
    outside = defaultdict(int)
    for row in metadata:
        text = row["text"]
        words = text.split()
        empty += not text
        spacing += text != " ".join(words)
        undocumented_edits += (text, row["text_rules"]) != clean_text(row["text_original"])
        leftovers += has_leftover_number_words(words)
        for char in set(text) - ALPHABET:
            outside[char] += 1

    check(empty == 0 and spacing == 0, f"{source}: no empty transcripts or irregular spaces")
    check(not outside, f"{source}: every character is in the 2016 Karakalpak alphabet {dict(outside)}")
    check(not any(c.isdigit() for row in metadata for c in row["text"]), f"{source}: no digits")
    check(undocumented_edits == 0, f"{source}: transcripts equal the originals with only the documented fixes")
    check(leftovers == 0, f"{source}: no split suffixes after numbers, no 'haqqında ǵı', no glued 'covidon', no jıyırma/elli/mın")
    fields, _ = read_csv(FINAL / source / "review.csv")
    check(fields == REVIEW_FIELDS, f"{source}: review.csv present")


def check_change_log(source, metadata):
    fields, changes = read_csv(FINAL / source / "changes.csv")
    expected = {row["audio_path"]: row for row in metadata if row["text"] != row["text_original"]}
    matches = all(expected[c["audio_path"]]["text"] == c["text"] and
                  expected[c["audio_path"]]["text_original"] == c["text_original"] for c in changes)
    check(fields == CHANGE_FIELDS and len(changes) == len(expected) and matches,
          f"{source}: changes.csv lists exactly the {len(expected)} edited transcripts")


def stt_split_assignments():
    split_of, split_text = {}, {}
    for split in DATASETS["stt"]:
        for row in read_csv(STT_SPLITS / f"{split}.csv")[1]:
            split_of[row["id"]] = split
            split_text[row["id"]] = row["text_normalized"]
    return split_of, split_text


def check_stt_sources(metadata):
    rows = {row["source_id"]: row for row in metadata}
    split_of, split_text = stt_split_assignments()
    byte_mismatch = text_mismatch = 0
    seen = set()
    normalized = {}
    for shard in sorted(STT.glob("*.parquet")):
        columns = ["id", "audio", "text_normalized", "cleaned_text"]
        for record in pq.read_table(shard, columns=columns).to_pylist():
            key = str(record["id"])
            seen.add(key)
            row = rows[key]
            digest = sha256(record["audio"])
            normalized[key] = (digest, record["cleaned_text"])
            byte_mismatch += sha256((FINAL / "stt" / row["audio_path"]).read_bytes()) != digest
            text_mismatch += row["text_original"] != record["text_normalized"]
            text_mismatch += row["split"] != split_of[key] or split_text[key] != record["text_normalized"]
    check(seen == set(rows) == set(split_of), f"stt: all {len(seen)} parquet ids present once")
    check(byte_mismatch == 0, "stt: every WAV is byte-identical to its parquet audio")
    check(text_mismatch == 0, "stt: original text and split match STT_SPLITS and parquet")

    same = total = 0
    for shard in sorted(CLEANED.glob("*.parquet")):
        for record in pq.read_table(shard, columns=["id", "audio", "cleaned_text"]).to_pylist():
            total += 1
            same += normalized.get(str(record["id"])) == (sha256(record["audio"]), record["cleaned_text"])
    check(total == len(normalized) == same,
          f"TOZALANGAN_TRAIN_FINAL equals STT_NORMALIZED_FINAL audio and cleaned_text ({same}/{total})")


def check_hf_sources(metadata):
    source_of = {row["audio_path"]: row["source"] for row in read_csv(COMBINED / "provenance/metadata.csv")[1]}
    rows = {row["source_location"]: row for row in metadata}
    source_rows = [item for item in read_csv(COMBINED / "metadata.csv")[1] if source_of[item["audio_path"]] == "hf"]
    mismatch = 0
    for item in source_rows:
        row = rows[f"karakalpak-combined/{item['audio_path']}"]
        mismatch += (FINAL / "hf" / row["audio_path"]).read_bytes() != (COMBINED / item["audio_path"]).read_bytes()
        mismatch += row["text_original"] != item["text"] or row["split"] != item["split"]
    check(len(rows) == len(source_rows), f"hf: all {len(source_rows)} HF recordings present, nothing else")
    check(mismatch == 0, "hf: audio bytes, original text and train/test split unchanged")


def check_leakage(final):
    rows = [row for metadata in final.values() for row in metadata]
    for key in ("pcm_sha256", "text"):
        splits = defaultdict(set)
        for row in rows:
            splits[row[key]].add(row["split"])
        check(all(len(v) == 1 for v in splits.values()), f"no {key} value crosses splits, within or across datasets")


def check_loader():
    sys.path.insert(0, str(ROOT / "src"))
    import soundfile as sf
    import config
    import dataset

    loaded = expected = 0
    for name in config.TRAIN_CSVS + config.VALIDATION_CSVS + list(config.TEST_CSVS.values()):
        data = dataset.load_split([name])
        for index in random.Random(0).sample(range(len(data)), 20):
            row = data[index]
            audio, rate = sf.read(row["audio_path"])
            expected += 1
            loaded += rate == 16000 and len(audio) > 0 and bool(row["text"])
    check(loaded == expected, f"src/dataset.py load_split reads {expected} random samples from every training CSV")


def main():
    check_folders()
    final = {}
    for source in DATASETS:
        final[source] = check_layout(source)
        check_audio(source, final[source])
        check_text(source, final[source])
        check_change_log(source, final[source])
    check_stt_sources(final["stt"])
    check_hf_sources(final["hf"])
    check_leakage(final)
    check_loader()
    print(f"\n{len(failures)} failed checks")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
