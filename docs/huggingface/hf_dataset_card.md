---
language:
- kaa
license: mit
pretty_name: Karakalpak Literary Speech - Cleaned
size_categories:
- 1K<n<10K
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
  - split: test
    path: data/test-*.parquet
---

# Karakalpak Literary Speech — Cleaned

A prepared derivative of [kaa-ml/karakalpak-audio-dataset](https://huggingface.co/datasets/kaa-ml/karakalpak-audio-dataset), containing **2,613 recordings / approximately 3.481 hours** of Karakalpak literary speech and normalized Latin transcripts.

Original dataset credit: **kaa-ml / Kaa ML and the original recording/transcript contributors**. Release maintainer and preparation: [Abdulaziz Yusupaliev](https://huggingface.co/iamabdulaziz). [Code and experiment records](https://github.com/vvadr/karakalpak-stt-model).

## Splits

| Split | Clips | Hours |
| --- | ---: | ---: |
| Train | 2,090 | 2.769 |
| Test | 523 | 0.712 |

There is no validation split. The prepared source uses a seed-42 grouped 80/20 split and retains the published source test recordings in test. Recordings are approximately 2.01–11.64 seconds, mono 16 kHz PCM16 WAV, embedded in Parquet.

## Load

```python
from datasets import load_dataset

data = load_dataset("iamabdulaziz/karakalpak-literary-cleaned")
print(data["test"][0]["text"])
```

Rows contain `audio`, `text`, `duration`, stable recording IDs, original text, source identifiers/locations, text rules and PCM SHA256. The [trained model](https://huggingface.co/iamabdulaziz/wav2vec2-xls-r-300m-karakalpak) uses this corpus with the [STT news corpus](https://huggingface.co/datasets/iamabdulaziz/karakalpak-stt-news).

## Upstream and preparation

Upstream dataset: [https://huggingface.co/datasets/kaa-ml/karakalpak-audio-dataset](https://huggingface.co/datasets/kaa-ml/karakalpak-audio-dataset). Exact source revision: `0f64c03be3a9b97fbdfc64934d2a81cddff174a8`. Its card declares **MIT** and its source repository uses automatic access gating. The original card is preserved at `provenance/upstream_README.md`, alongside a source metadata record. This is a separately prepared derivative, not the original publisher's repository.

The local preparation resampled original 44.1 kHz recordings to 16 kHz. The final export preserves those prepared WAV bytes and split identities. The final text build applies `í` → `ı` and the documented number-word fixes; one transcript changed in that final build. The earlier preparation's retained collection and split metadata are preserved in provenance.

The final build selects only the HF-origin part of `karakalpak-combined`; Mendeley-origin recordings are excluded. See [build_final_datasets.py](https://github.com/vvadr/karakalpak-stt-model/blob/main/scripts/build_final_datasets.py), [organize_datasets.py](https://github.com/vvadr/karakalpak-stt-model/blob/main/scripts/organize_datasets.py), and the [publication guide](https://github.com/vvadr/karakalpak-stt-model/blob/main/docs/publication_guide.md).

## Quality and limitations

The joint collection passed all 37 existing audio/file/consistency checks, including PCM hashes and exact split separation. No exact PCM or prepared transcript crosses splits within or across the released STT and literary collections.

Seven clips remain listening-review candidates: four train and three test. Some repeated-number flags are valid arithmetic speech; flags are not confirmed errors. Speaker identities are unavailable, so these are not speaker-independent splits. Text normalization and domain coverage limit generalization claims.

## License and credit

License metadata follows the **MIT license declared by the pinned upstream card**. Preserve the original publisher/contributor attribution and that source record when redistributing this derivative. The upstream source did not contain a separate LICENSE file or a narrative description beyond its metadata; the preserved source card records exactly the supplied terms.

## Citation and integrity

Credit the original kaa-ml dataset and this preparation repository. Cite the exact Hugging Face revision used. `export_manifest.json` records source-manifest hashes and exporter identity; `checksums.sha256` records the released file hashes.
