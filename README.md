# Karakalpak speech recognition

Speech-to-text for Karakalpak. The main model is **wav2vec2 XLS-R 300M fine-tuned with LoRA**,
decoded with beam search and a **KenLM 4-gram language model**. On our two test sets it reaches
**18.61% WER / 5.93% CER** (STT, news speech) and **18.30% WER / 3.08% CER** (HF, literary speech).

## Results

Model: `facebook/wav2vec2-xls-r-300m` + LoRA (rank 32), trained on 2026-10-14 on the server
(NVIDIA GB10, 4 epochs, 2 h 31 min) with 17,685 clips (77.45 h). WER / CER:

| Test set | Clips | Greedy | Beam search | Beam search + KenLM |
| --- | ---: | --- | --- | --- |
| STT test | 1,950 | 38.37% / 8.90% | 37.75% / 8.73% | **18.61% / 5.93%** |
| HF test | 523 | 22.32% / 3.65% | 21.68% / 3.53% | **18.30% / 3.08%** |
| asr_corpus test (outside data) | 1,047 | 39.59% / 9.06% | 38.71% / 8.88% | **21.32% / 6.25%** |
| asr_corpus validation (outside data) | 1,048 | 38.16% / 8.69% | 37.34% / 8.51% | **20.27% / 5.88%** |
| whisper_test (outside data) | 1,018 | 38.56% / 8.80% | 37.65% / 8.60% | **20.64% / 6.04%** |

- **Greedy** takes the most likely letter at each step. **Beam search** keeps the 100 best
  candidates. **+ KenLM** adds a word-level language model (weights α 0.7, β 2.0, tuned on 500
  validation clips).
- The outside test sets come from other projects on the server and were not used for training
  (only 3–4 of their sentences also appear in our training text).
  `whisper_test` is the test split of the Whisper project; on it, that project's fine-tuned
  Whisper-small scores 20.16% / 7.11%.
- Earlier experiments used different data and are not comparable: an older LoRA script on
  `karakalpak-combined` (55.44% WER on its 927 test clips) and Whisper-small on the Mac (32.45% WER /
  7.31% CER on 370 validation clips, see `docs/whisper_training.md`).

## Data

Training uses `data/final/`, two datasets kept apart because their domain and clip length differ:

| Dataset | Train | Validation | Test | Hours | Speech |
| --- | ---: | ---: | ---: | ---: | --- |
| STT (`data/final/stt`) | 15,595 | 1,950 | 1,950 | 93.13 | news, 5.9–38.5 s clips |
| HF (`data/final/hf`) | 2,090 | – | 523 | 3.48 | literary, 2.0–11.6 s clips |

All audio is mono 16 kHz WAV. Transcripts use only the 2016 Karakalpak Latin alphabet. Mendeley
recordings are not used (they write `ś`/`ć` and often follow Kazakh spelling). The audio is not in
Git (`data/` is ignored): `scripts/build_final_datasets.py` builds `data/final` from the source
datasets and `scripts/validate_final_datasets.py` checks it (37 checks). Details, every transcript
fix and the outside test sets are in `docs/datasets.md`.

## Setup

Python 3.12. From the repo root:

```bash
uv sync --no-install-project
source .venv/bin/activate
uv pip install peft kenlm
uv pip install --no-deps pyctcdecode pygtrie
```

- `--no-install-project` is needed because the project's own package (`src/ai_ml`) is only an
  empty template and does not build.
- `pyctcdecode` asks for `numpy<2` but works with numpy 2; `--no-deps` keeps numpy untouched.
- Without uv, use the same package names with `pip install` inside a virtual environment.

`src/lm.py` needs KenLM's `lmplz` program, built from the `kenlm` submodule (needs `cmake`, Boost
and zlib, see `docs/wav2vec2_training.md`):

```bash
git submodule update --init kenlm
mkdir -p kenlm/build && cd kenlm/build && cmake .. && make -j 8 && cd ../..
```

## Run the pipeline

```bash
mkdir -p outputs/raw_logs
python src/vocab.py      # 35-token character vocabulary
nohup python src/train.py > outputs/raw_logs/train_$(date +%F).log 2>&1 &   # about 2.5 h on the server
python src/lm.py         # 4-gram KenLM from the train transcripts
python src/decode.py     # after training: greedy vs beam search vs beam search + KenLM, tunes α and β
python scripts/prepare_benchmarks.py                          # server only: builds data/benchmarks/
python src/benchmark.py data/benchmarks/asr_corpus/test.csv   # or any other test CSV
```

Everything is written to `outputs/wav2vec2-lora-kaa/`. All settings are in `src/config.py`.
`LOCAL=1` switches to a small preset (3,000 clips, 5 epochs) that fits on a Mac.
`docs/wav2vec2_training.md` explains every file and setting.

## Logs

`train.log`, `decode.log` and `benchmark.log` are cleaned logs of the 2026-10-14 server run. Each
starts with a title line (model | run | status), then the model, the data, the settings and the
results as tables. The raw logs stay in `outputs/raw_logs/` on the machine that made them. A new run
writes its raw output there too, so these files are not overwritten.

`train.log` reports 38.43% WER on STT test, a little worse than the 38.37% above: its quick
end-of-training check also reads the padding after each clip, which added stray letters at the end
of 125 clips. `decode.log` cuts the padding off, so the table above uses its numbers.

## Repository layout

```text
src/                      the wav2vec2 + LoRA pipeline (config, data, model, training, LM, decoding)
scripts/                  build_final_datasets.py, validate_final_datasets.py, prepare_benchmarks.py,
                          plus older tools for the source datasets
notebooks/whisper/        the Whisper-small workflow (start with setup.ipynb)
notebooks/preprocessingAndFeatureExtraction/   first-week data notebooks
docs/                     guides, listed below
kenlm/                    KenLM source (Git submodule)
train.log, decode.log, benchmark.log           cleaned logs of the server run
data/, outputs/, checkpoints/                  local only, ignored by Git
```

Not used by the current pipeline: `kaa.arpa`, `lm_text.txt` and `vocab.json` in the root come from
the older experiment on `karakalpak-combined`, and `main.py`, `train.py`, `evaluate.py`,
`infer.py`, `configs/` and `src/ai_ml/` are empty template files.

## Docs

| File | What it covers |
| --- | --- |
| [docs/wav2vec2_training.md](docs/wav2vec2_training.md) | the `src/` pipeline file by file, settings, results, logs, how to run |
| [docs/datasets.md](docs/datasets.md) | `data/final`, the alphabet, transcript fixes, the outside test sets |
| [docs/remote_training_runbook.md](docs/remote_training_runbook.md) | Mac and server workflow, copying data, pushing to GitHub |
| [docs/whisper_training.md](docs/whisper_training.md) | the Whisper-small notebooks and their results |
| [docs/source_datasets.md](docs/source_datasets.md) | the original HF, Mendeley and combined exports (the old README) |
| [docs/notebook_workflow.md](docs/notebook_workflow.md), [docs/dataset_review.md](docs/dataset_review.md) | first-week notebooks and dataset review |
| [docs/pipeline/](docs/pipeline/) (PDFs) | a 4-page pipeline map and a 35-page step-by-step walkthrough |

## Limitations

- No speaker IDs exist, so a speaker can appear in both train and test.
- Validation is STT only (HF has no validation split), so the best epoch is chosen on news speech.
- Hyphenated words are joined in the STT text (`sociallıqekonomikalıq`); this costs about 1.5 WER
  points on asr_corpus, whose references split them.
