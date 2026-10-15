from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STT = DATA / "datasets" / "STT_NORMALIZED_FINAL"
STT_SPLITS = DATA / "datasets" / "STT_SPLITS"
COMBINED = DATA / "karakalpak-combined"
FINAL = DATA / "final"

DATASETS = {"stt": ("train", "validation", "test"), "hf": ("train", "test")}
ALPHABET = set("aábcdefgǵhiıjklmnńoópqrstuúvwxyz ")
LETTER_REPLACEMENTS = (("í", "ı"),)

NUMBER_SPELLINGS = {"jıyırma": "jigirma", "jıyırmanshı": "jigirmanshı", "elli": "eliw", "mın": "mıń"}
NUMBER_WORDS = set("bir eki úsh tórt bes altı jeti segiz toǵız on jigirma otız qırıq qırq eliw alpıs jetpis "
                   "seksen toqsan júz mıń million milliard".split())
SUFFIXES_AFTER_NUMBERS = set("ǵa ge qa ke ga ka sı si ın in nshı nshi ınshı inshi nıń niń dıń diń tıń tiń "
                             "dı di tı ti dan den tan ten nan nen".split())
SUFFIXES_AFTER_ANY_WORD = {"ǵı", "gi"}
DATIVE_SUFFIXES = {"ǵa", "ge", "qa", "ke", "ga", "ka"}
FRONT_VOWELS = "áeióú"
ALL_VOWELS = "aáeıioóuú"
VOICELESS_CONSONANTS = "ptkqsfxhc"

NUMBER_WORD_PATTERN = "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
ABBREVIATION_GLUED_TO_NUMBER = re.compile(rf"(covid|top|cop)({NUMBER_WORD_PATTERN})")
REPEATED_NUMBER = re.compile(rf"(?<!\S)((?:{NUMBER_WORD_PATTERN})(?: (?:{NUMBER_WORD_PATTERN}))*) \1(?!\S)")

TRAIN_FIELDS = ["audio_path", "text", "duration"]
METADATA_FIELDS = [*TRAIN_FIELDS, "split", "source_id", "source_location", "text_original",
                   "text_rules", "pcm_sha256"]
CHANGE_FIELDS = ["audio_path", "split", "source_id", "source_location", "text_rules",
                 "text_original", "text"]
REVIEW_FIELDS = ["audio_path", "split", "issue", "text"]
QUALITY_FIELDS = ["audio_path", "duration", "peak_dbfs", "rms_dbfs", "clipped_fraction",
                  "chars_per_second"]

FINAL_README = """# final

Two separate, train-ready datasets built by `scripts/build_final_datasets.py` and checked by
`scripts/validate_final_datasets.py`. They differ in domain, clip length and splits, so they are
kept apart; concatenate split CSVs to train on both.

| Dataset | Source | Splits |
| --- | --- | --- |
| `stt/` | `datasets/STT_NORMALIZED_FINAL` + `datasets/STT_SPLITS` | train / validation / test |
| `hf/` | HF part of `karakalpak-combined` | train / test |

Transcripts use only the current (2016) Karakalpak Latin alphabet
(`a á b c d e f g ǵ h i ı j k l m n ń o ó p q r s t u ú v w x y z`, digraphs `sh`, `ch`).
Edits: `í -> ı` (`Í` is the capital of `ı`), split-off suffixes re-attached (`júz den` -> `júzden`,
`haqqında ǵı` -> `haqqındaǵı`), glued abbreviations split (`covidon` -> `covid on`), and one spelling
for 20 and 50 (`jigirma`, `eliw`). `review.csv` lists rows to check by ear.

Each folder: `<split>/audio/*.wav` (mono 16 kHz PCM16, byte-identical to the source),
`<split>.csv` (`audio_path`, `text`, `duration`; paths relative to that folder), `metadata.csv`,
`changes.csv` (every edited transcript with its original), `review.csv` (rows to check by ear),
`provenance/quality.csv`, `split_summary.json`. `summary.json` here totals both. See `docs/datasets.md`.
"""

DATASET_README = {
    "stt": """# stt

News and official-speech corpus from `datasets/STT_NORMALIZED_FINAL` (`text_normalized`),
split by `datasets/STT_SPLITS`. Clips 5.9-38.5 s; 198 are longer than 30 s.
Transcripts are the original text with `í -> ı` and number-word leftovers repaired; see `changes.csv`
and `review.csv`. `source_location` is `<parquet shard>#id=<id>`.
""",
    "hf": """# hf

HF recordings from `karakalpak-combined` (seed-42 80/20 split, published HF test kept in test).
Clips 2.0-11.6 s. Transcripts are the original text with `í -> ı` and number-word fixes; see `changes.csv`.
`source_location` is the file in `karakalpak-combined`.
""",
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def last_vowel_is_front(word):
    vowels = [letter for letter in word if letter in ALL_VOWELS]
    return bool(vowels) and vowels[-1] in FRONT_VOWELS


def dative(word):
    ends_voiceless = word[-1] in VOICELESS_CONSONANTS
    if last_vowel_is_front(word):
        return "ke" if ends_voiceless else "ge"
    return "qa" if ends_voiceless else "ǵa"


def is_split_off_suffix(word, previous_word):
    return word in SUFFIXES_AFTER_ANY_WORD or (word in SUFFIXES_AFTER_NUMBERS and previous_word in NUMBER_WORDS)


def attach_suffix(word, suffix):
    if word.endswith(suffix):
        return word
    if suffix in DATIVE_SUFFIXES and word in NUMBER_WORDS:
        return word + dative(word)
    return word + suffix


def fix_number_words(text):
    words, repairs = [], set()
    for word in text.split():
        glued = ABBREVIATION_GLUED_TO_NUMBER.fullmatch(word)
        if glued:
            words.append(glued.group(1))
            word = glued.group(2)
            repairs.add("split")
        if word in NUMBER_SPELLINGS:
            word = NUMBER_SPELLINGS[word]
            repairs.add("spelling")
        if words and is_split_off_suffix(word, words[-1]):
            words[-1] = attach_suffix(words[-1], word)
            repairs.add("suffix")
        else:
            words.append(word)
    return " ".join(words), sorted(repairs)


def clean_text(text):
    applied = []
    for old, new in LETTER_REPLACEMENTS:
        if old in text:
            text = text.replace(old, new)
            applied.append(f"{old}->{new}")
    text, repairs = fix_number_words(text)
    return text, ";".join(applied + repairs)


def pcm_frames(data, label):
    with wave.open(io.BytesIO(data), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"Expected mono 16 kHz PCM16: {label}")
        frames = audio.readframes(audio.getnframes())
    if len(frames) != audio.getnframes() * 2 or not frames:
        raise ValueError(f"Truncated or empty audio: {label}")
    return frames


def decibels(value):
    return 20 * np.log10(value / 32768)


def quality(audio_path, frames, text):
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float64)
    duration = len(samples) / 16000
    peak = np.abs(samples).max()
    rms = np.sqrt(np.mean(samples ** 2))
    return {
        "audio_path": audio_path,
        "duration": f"{duration:.4f}",
        "peak_dbfs": f"{decibels(max(peak, 1)):.2f}",
        "rms_dbfs": f"{decibels(max(rms, 1e-9)):.2f}",
        "clipped_fraction": f"{np.mean(np.abs(samples) >= 32767):.6f}",
        "chars_per_second": f"{len(text.replace(' ', '')) / duration:.2f}",
    }


def add_record(stage, records, source, split, name, data, text_original, source_id, location):
    frames = pcm_frames(data, location)
    text, rules = clean_text(text_original)
    outside = set(text) - ALPHABET
    if outside:
        raise ValueError(f"{location}: characters outside the alphabet {sorted(outside)}")

    audio_path = f"{split}/audio/{name}"
    target = stage / source / audio_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)

    records[source].append({
        "audio_path": audio_path,
        "text": text,
        "duration": len(frames) / 32000,
        "split": split,
        "source_id": source_id,
        "source_location": location,
        "text_original": text_original,
        "text_rules": rules,
        "pcm_sha256": hashlib.sha256(frames).hexdigest(),
        "quality": quality(audio_path, frames, text),
    })


def stt_split_assignments():
    assignments = {}
    for split in DATASETS["stt"]:
        for row in read_csv(STT_SPLITS / f"{split}.csv"):
            if row["id"] in assignments:
                raise ValueError(f"STT id {row['id']} appears in more than one split")
            assignments[row["id"]] = (split, row["text_normalized"])
    return assignments


def build_stt(stage, records):
    assignments = stt_split_assignments()
    seen = set()
    for shard in sorted(STT.glob("*.parquet")):
        for record in pq.read_table(shard, columns=["id", "audio", "text_normalized"]).to_pylist():
            key = str(record["id"])
            split, split_text = assignments[key]
            if split_text != record["text_normalized"]:
                raise ValueError(f"STT id {key}: split CSV text differs from parquet")
            add_record(stage, records, "stt", split, f"stt_{record['id']:06d}.wav", record["audio"],
                       record["text_normalized"], key, f"datasets/STT_NORMALIZED_FINAL/{shard.name}#id={key}")
            seen.add(key)
    if seen != set(assignments):
        raise ValueError("STT parquet ids and STT_SPLITS ids differ")


def build_hf(stage, records):
    detailed = {row["audio_path"]: row for row in read_csv(COMBINED / "provenance/metadata.csv")}
    for row in read_csv(COMBINED / "metadata.csv"):
        source_row = detailed[row["audio_path"]]
        if source_row["source"] != "hf":
            continue
        path = COMBINED / row["audio_path"]
        add_record(stage, records, "hf", row["split"], path.name, path.read_bytes(), row["text"],
                   source_row["sample_id"], f"karakalpak-combined/{row['audio_path']}")


def check_splits(records):
    for source, rows in records.items():
        unexpected = {row["split"] for row in rows} - set(DATASETS[source])
        if unexpected:
            raise ValueError(f"{source} has unexpected splits {unexpected}")


def check_leakage(records):
    rows = [row for part in records.values() for row in part]
    for key in ("pcm_sha256", "text"):
        splits = defaultdict(set)
        for row in rows:
            splits[row[key]].add(row["split"])
        crossing = sum(len(value) > 1 for value in splits.values())
        if crossing:
            raise ValueError(f"{crossing} {key} groups cross split boundaries")


def hours(rows):
    return round(sum(row["duration"] for row in rows) / 3600, 3)


def summarize(rows, splits):
    by_split = {split: [row for row in rows if row["split"] == split] for split in splits}
    return {
        "splits": {split: {"files": len(part), "hours": hours(part)} for split, part in by_split.items()},
        "files": len(rows),
        "hours": hours(rows),
        "transcripts_changed": sum(row["text"] != row["text_original"] for row in rows),
        "rules": dict(Counter(rule for row in rows for rule in row["text_rules"].split(";") if rule)),
        "characters": "".join(sorted({char for row in rows for char in row["text"]} - {" "})),
    }


def review_rows(rows):
    review = []
    for row in rows:
        if REPEATED_NUMBER.search(row["text"]):
            review.append({**row, "issue": "repeated number"})
        if any(word in SUFFIXES_AFTER_NUMBERS for word in row["text"].split()[1:]):
            review.append({**row, "issue": "standalone suffix"})
    return review


def write_dataset(folder, source, rows):
    rows.sort(key=lambda row: row["audio_path"])
    for split in DATASETS[source]:
        write_csv(folder / f"{split}.csv", [row for row in rows if row["split"] == split], TRAIN_FIELDS)
    write_csv(folder / "metadata.csv", rows, METADATA_FIELDS)
    write_csv(folder / "changes.csv", [row for row in rows if row["text"] != row["text_original"]], CHANGE_FIELDS)
    write_csv(folder / "provenance/quality.csv", [row["quality"] for row in rows], QUALITY_FIELDS)
    write_csv(folder / "review.csv", review_rows(rows), REVIEW_FIELDS)
    result = summarize(rows, DATASETS[source])
    write_json(folder / "split_summary.json", result)
    (folder / "README.md").write_text(DATASET_README[source], encoding="utf-8")
    return result


def fresh_folder(path):
    if path.exists():
        shutil.rmtree(path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Build data/final/stt and data/final/hf from the raw corpora.")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing data/final")
    args = parser.parse_args()
    if FINAL.exists() and not args.overwrite:
        raise SystemExit(f"{FINAL} exists; pass --overwrite to rebuild it")

    stage = fresh_folder(FINAL.with_name(FINAL.name + ".partial"))
    records = {source: [] for source in DATASETS}
    build_stt(stage, records)
    build_hf(stage, records)
    check_splits(records)
    check_leakage(records)

    result = {
        "rules": [f"{old}->{new}" for old, new in LETTER_REPLACEMENTS],
        "alphabet": "".join(sorted(ALPHABET - {" "})),
        "datasets": {source: write_dataset(stage / source, source, rows) for source, rows in records.items()},
    }
    write_json(stage / "summary.json", result)
    (stage / "README.md").write_text(FINAL_README, encoding="utf-8")

    fresh_folder(FINAL)
    stage.rename(FINAL)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
