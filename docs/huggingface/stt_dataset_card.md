---
language:
- kaa
pretty_name: Karakalpak STT News
size_categories:
- 10K<n<100K
task_categories:
- automatic-speech-recognition
tags:
- audio
- karakalpak
- speech
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: validation
    path: data/validation-*.parquet
  - split: test
    path: data/test-*.parquet
---

# Karakalpak STT News

19,495 Karakalpak news and official-speech recordings, approximately **93.132 hours**, with normalized Latin transcripts and fixed train/validation/test splits.

Release maintainer: [Abdulaziz Yusupaliev](https://huggingface.co/iamabdulaziz). Preparation code, recorded experiments, and publication tools: [https://github.com/vvadr/karakalpak-stt-model](https://github.com/vvadr/karakalpak-stt-model).

## Splits

| Split | Clips | Hours |
| --- | ---: | ---: |
| Train | 15,595 | 74.676 |
| Validation | 1,950 | 9.281 |
| Test | 1,950 | 9.175 |

Audio is mono 16 kHz PCM16 WAV, embedded unchanged from the prepared dataset in Parquet shards. Clips are approximately 5.92–38.49 seconds. Each row includes `audio`, `text`, `duration`, a stable recording `id`, original text, source identifiers/locations, normalization rules, and a PCM SHA256.

## Load

```python
from datasets import load_dataset

data = load_dataset("iamabdulaziz/karakalpak-stt-news")
print(data["train"][0]["text"])
```

See the [fine-tuned model](https://huggingface.co/iamabdulaziz/wav2vec2-xls-r-300m-karakalpak) for inference and decoder-specific evaluation results.

## Source and preparation

The supplied source was `STT_NORMALIZED_FINAL`: 20 Parquet shards containing recording IDs, audio bytes, cleaned text and normalized text. Original split assignments came from `STT_SPLITS`. The original publisher, collection URL, and recording creators were not recorded in the supplied files. The release maintainer is not asserted to be the creator of the source recordings.

The build preserves the supplied split IDs and audio bytes. It converts `í` to `ı` and applies the documented number-word and suffix repairs, including `júz den` → `júzden`, `covidon` → `covid on`, and `elli` → `eliw`. Transcripts use the Karakalpak Latin alphabet and omit punctuation/digits. The original transcript and every change remain available.

Recipe: [build_final_datasets.py](https://github.com/vvadr/karakalpak-stt-model/blob/main/scripts/build_final_datasets.py). Validation: [validate_final_datasets.py](https://github.com/vvadr/karakalpak-stt-model/blob/main/scripts/validate_final_datasets.py). Export: [prepare_huggingface.py](https://github.com/vvadr/karakalpak-stt-model/blob/main/scripts/release/prepare_huggingface.py).

`provenance/` contains the original prepared metadata, 2,115 changed-transcript records, quality measurements, split summary and 284 listening-review candidates. Review flags identify possible issues; they are not 284 confirmed errors. Of those candidates, 227 are train, 32 validation and 25 test.

## Validation and limitations

The joint STT/literary collection passed all 37 existing file/consistency checks. All final WAVs were checked for format, completeness, duration, PCM hash, source fidelity and documented transformations. No exact PCM hash or prepared transcript crosses the split boundaries within or across these two released datasets.

Speaker identities are unknown. Exact-duplicate separation does not establish speaker independence or rule out acoustically similar excerpts. Automated consistency checks are not independent linguistic verification. Joined hyphenated words and upstream normalization can affect WER comparisons. This corpus primarily represents news/official speech.

## Release terms

The supplied source material did not specify a redistribution license or original publisher. **No additional license has been assigned by this release.** The metadata deliberately does not claim MIT, CC BY, or another unverified license. Original source credits and terms will be updated when established.

## Citation and integrity

Cite the repository and the exact Hugging Face dataset revision used in your experiment. `export_manifest.json` records source-manifest hashes, counts, exporter identity and export time. `checksums.sha256` records the published file hashes.
