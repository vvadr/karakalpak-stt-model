# Karakalpak speech recognition

A research pipeline for **Karakalpak speech-to-text** using **wav2vec2 XLS-R 300M**, **LoRA fine-tuning**, and **KenLM decoding**. This repository contains the data preparation recipes, training and evaluation code, recorded experiment results, and tools for publishing datasets and models.

**Recorded results:** 18.61% WER on news speech and 18.30% WER on literary speech with beam search and a KenLM 4-gram language model.

[Results](#results) · [Datasets](#datasets) · [Workflow](#workflow) · [Publication](#publishing-datasets-and-models) · [Documentation](#documentation)

## Project status

This is a research project with a completed, documented server experiment. The scores below come from the saved server logs; they have not been independently reproduced from the server weights during the latest local review.

Datasets, model weights, training checkpoints, and prepared upload folders are stored outside Git. Public Hugging Face downloads are pending publication. The current source has a known missing `IGNORED_LABEL_ID` setting affecting training and batch evaluation, and the FastAPI application is a stub. See the [readiness review](docs/release_readiness.md) before running or deploying the pipeline.

## Results

The documented run used `facebook/wav2vec2-xls-r-300m`, LoRA rank 32, and a custom 35-token character vocabulary. It trained for four epochs on **17,685 clips / 77.45 hours** using an NVIDIA GB10 GPU, taking approximately 2 hours 31 minutes.

All values are **WER / CER**; lower is better.

| Test set | Clips | Greedy | Beam search | Beam + KenLM |
| --- | ---: | --- | --- | --- |
| STT news | 1,950 | 38.37% / 8.90% | 37.75% / 8.73% | **18.61% / 5.93%** |
| HF literary | 523 | 22.32% / 3.65% | 21.68% / 3.53% | **18.30% / 3.08%** |

Additional recorded evaluations on other projects' datasets:

| Benchmark | Clips | Greedy | Beam search | Beam + KenLM |
| --- | ---: | --- | --- | --- |
| asr_corpus test | 1,047 | 39.59% / 9.06% | 38.71% / 8.88% | **21.32% / 6.25%** |
| asr_corpus validation | 1,048 | 38.16% / 8.69% | 37.34% / 8.51% | **20.27% / 5.88%** |
| whisper_test | 1,018 | 38.56% / 8.80% | 37.65% / 8.60% | **20.64% / 6.04%** |

Beam search uses width 100. KenLM uses a 4-gram language model built from the training transcripts, with alpha 0.7 and beta 2.0 selected on 500 validation clips. These scores describe the complete acoustic model and decoder combination; greedy inference has different results.

Evidence is retained in [train.log](train.log), [decode.log](decode.log), and [benchmark.log](benchmark.log). The quick Trainer evaluation includes padded frames; the decoding results above use the output-length trimming in `src/decode.py`. External benchmark overlap and speaker independence were not independently established in the local review.

## Datasets

The final training collection keeps news and literary speech separate:

| Dataset | Local folder | Train | Validation | Test | Total hours |
| --- | --- | ---: | ---: | ---: | ---: |
| STT news and official speech | `data/final/stt/` | 15,595 | 1,950 | 1,950 | 93.132 |
| HF literary speech | `data/final/hf/` | 2,090 | — | 523 | 3.481 |

Together, the full train/validation/test collection contains **22,108 recordings / 96.613 hours**. Mendeley recordings are excluded from the documented server run.

- Audio: mono, 16 kHz, PCM16 WAV.
- Training CSVs: `audio_path,text,duration`, with audio paths relative to each dataset folder.
- Provenance: original transcripts, source identifiers, PCM hashes, text changes, quality measurements, and listening-review flags.
- Text preparation: the Karakalpak Latin alphabet, `í → ı`, and documented number-word/suffix repairs.
- Splits: no exact PCM or prepared-transcript value crosses a split boundary. Speaker identities are unavailable, so these are not speaker-disjoint splits.

The existing validator passed all **37 checks** in the 2026-10-22 local audit. There are 284 STT and 7 literary recordings flagged for listening review; flags are candidates for review rather than confirmed errors.

The literary source is [kaa-ml/karakalpak-audio-dataset](https://huggingface.co/datasets/kaa-ml/karakalpak-audio-dataset). Source attribution and redistribution terms must be completed in the dataset cards before public release. See the [publication guide](docs/publication_guide.md).

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) are used for the recorded workflow.

```bash
git clone --recurse-submodules https://github.com/vvadr/karakalpak-stt-model.git
cd karakalpak-stt-model
uv sync --no-install-project
source .venv/bin/activate
uv pip install peft kenlm
uv pip install --no-deps pyctcdecode pygtrie
```

The unfinished `src/ai_ml/` package is bypassed with `--no-install-project`. Decoder dependencies are installed separately in the current recipe; the lock file does not yet capture the complete runtime environment. The `--no-deps` command preserves the existing NumPy environment and requires a compatibility check on the target machine.

Building a language model also requires KenLM's `lmplz`, CMake, Boost, and zlib:

```bash
git submodule update --init kenlm
cmake -S kenlm -B kenlm/build
cmake --build kenlm/build -j 8
```

## Workflow

```text
Source recordings and transcripts
    → prepare final datasets
    → validate audio, text, provenance, and splits
    → build character vocabulary
    → fine-tune wav2vec2 with LoRA
    → build KenLM from training transcripts
    → tune the decoder on validation data
    → evaluate fixed test sets
    → export the model and decoder together
```

The commands below describe the recorded pipeline. Address the current runtime issue described in [project status](#project-status) before training or batch evaluation. Source datasets or a prepared dataset download must be available first.

```bash
# Data preparation; an existing data/final directory is preserved by default.
python scripts/build_final_datasets.py
python scripts/validate_final_datasets.py

# Training and decoding.
python src/vocab.py
python src/train.py
python src/lm.py
python src/decode.py

# One recording, once the trained model and matching decoder assets are present.
python src/transcribe.py clip.wav

# Additional benchmarks require the source projects on the training server.
python scripts/prepare_benchmarks.py
python src/benchmark.py data/benchmarks/asr_corpus/test.csv
```

Settings are in `src/config.py`. The documented server run writes to `outputs/wav2vec2-lora-kaa/`: `final_model/` holds the merged model and processor, `lm/` the language model, and `results/` the decoder settings and predictions. Current single-file inference and benchmarks read the decoder settings saved by `decode.py`.

## Publishing datasets and models

GitHub holds the code, documentation, small experiment assets, and curated result logs. Hugging Face is the planned home for the two prepared datasets and the complete model/decoder release.

```bash
# Check which files are eligible for GitHub.
python3 scripts/release/check_github.py

# Build dataset drafts with embedded audio and preserved splits.
.venv/bin/python scripts/release/prepare_huggingface.py dataset stt
.venv/bin/python scripts/release/prepare_huggingface.py dataset hf

# On the machine holding the full server run, assemble its model bundle.
python3 scripts/release/prepare_huggingface.py model --run-dir outputs/wav2vec2-lora-kaa
```

Exports are written under the ignored `publication/` folder. Existing exports are never overwritten. The helpers prepare local files and checksums; they do not upload anything or choose a dataset license.

Follow the [step-by-step upload instructions](docs/upload_steps.md) for the exact folders, Hugging Face commands, card requirements, and verification steps. A model release must include the matching tokenizer/processor, KenLM, and decoder configuration to reproduce the KenLM results.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | Active wav2vec2, LoRA, decoding, evaluation, and transcription code |
| `scripts/` | Dataset preparation, source recovery, validation, and benchmark tools |
| `scripts/release/` | GitHub file checker and Hugging Face export helper |
| `docs/` | Readiness audit, architecture, publication instructions, and pipeline PDFs |
| `kenlm/` | Pinned upstream Git submodule |
| `train.log`, `decode.log`, `benchmark.log` | Curated evidence from the documented server experiment |
| `kaa.arpa`, `lm_text.txt`, `vocab.json` | Tracked historical experiment assets |
| `data/`, `outputs/`, `checkpoints/`, `publication/` | Local data, trained artifacts, and exports; excluded from Git |

The root `kaa.arpa`, `lm_text.txt`, and `vocab.json` belong to the older `karakalpak-combined` experiment. They are retained for reference and are not the newer server run's 4-gram LM and vocabulary. Root entry-point placeholders, `configs/`, the CTC exercise, and `src/ai_ml/` are outside the active pipeline.

## Limitations

- Speaker IDs are unavailable; unseen-speaker performance is not established.
- Validation uses news speech only; the literary corpus has no validation split.
- Text conventions, including joined hyphenated words, affect comparisons between corpora.
- The current training path automatically resumes its shared checkpoint folder; experiment identity and explicit resume behavior need improvement.
- The API is a stub. Production serving, operational limits, and latency/capacity measurements are not implemented.

## Documentation

- [Release readiness review](docs/release_readiness.md): verified findings and remaining blockers.
- [Project architecture](docs/project_architecture.md): current dependencies and proposed organization.
- [Publication guide](docs/publication_guide.md): release contents, provenance, and hosting choices.
- [Upload instructions](docs/upload_steps.md): exact preparation, upload, and verification commands.
- [Pipeline map](docs/pipeline/asr_pipeline_map.pdf) and [walkthrough](docs/pipeline/asr_pipeline_walkthrough.pdf): historical diagrams and annotated code snapshots.
