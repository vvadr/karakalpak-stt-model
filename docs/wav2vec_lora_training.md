# Karakalpak wav2vec2 XLS-R 300M with LoRA

This workflow fine tunes `facebook/wav2vec2-xls-r-300m` with a new character CTC
head and PEFT LoRA. It uses the pretrained speech encoder and your own corpus
letters. The base weights are pinned to revision
`1a640f32ac3e39899438a2931f9924c02f080a54`.

## Completed run: 2026-10-13

Run directory: `checkpoints/wav2vec2-xlsr300m-lora-kaa-run01`.
All eight epochs and 1,672 optimizer updates completed on Apple MPS. The model
was selected by validation WER, merged, saved, reloaded from disk, and evaluated
on all 927 held-out test recordings. Epoch 8 was selected.

| Epoch | Training loss | Validation loss | Validation WER | Validation CER |
| --- | ---: | ---: | ---: | ---: |
| 1 | 5.179 | 3.148 | 100.00% | 100.00% |
| 2 | 2.496 | 1.165 | 99.96% | 46.20% |
| 3 | 1.068 | 0.593 | 73.36% | 17.08% |
| 4 | 0.824 | 0.491 | 61.27% | 13.79% |
| 5 | 0.742 | 0.451 | 57.92% | 12.81% |
| 6 | 0.702 | 0.432 | 55.87% | 12.25% |
| 7 | 0.682 | 0.427 | 55.22% | 11.99% |
| 8 | 0.675 | 0.426 | 55.18% | 12.00% |

Final test: **WER 55.44%, CER 12.14%, CTC loss 0.427**, zero empty predictions.
Decoding is greedy without a language model. The source train/test manifests
retained their audited SHA256 checksums. All eight checkpoints include the
adapter, trained head, processor, custom configuration, optimizer, scheduler,
and RNG state. The standalone model contains 424 weight tensors and a
`[39, 1024]` CTC head, with no remaining LoRA wrapper tensors.

`epoch_metrics.csv`, `training_curves.png`, `test_metrics.json`, and
`test_predictions.csv` are in the run directory. The two-update preflight is
saved separately under `checkpoints/wav2vec2-xlsr300m-lora-kaa-preflight-20261013`.
See [wav2vec_source_review.md](wav2vec_source_review.md) for the subsequent static
review of the supplied `src` files, which were not executed or edited.

## Data and vocabulary

The input is `data/karakalpak-combined/{train,test}.csv`. Paths in the CSVs are
relative to that dataset folder. All 4,635 WAV files are audited for readability,
mono 16 kHz audio, finite samples, silence, matching durations, and split leakage.
The existing 927-recording test split is held out. A reproducible, transcript
grouped split reserves 373 of the 3,708 training recordings for validation,
leaving 3,335 for optimization. This split has no speaker information and cannot
establish speaker independence.

Normalize text to Unicode NFC, lowercase, and single spaces. Preserve the source
alphabet and accents; the rare characters `í`, `ć`, and Cyrillic `һ` remain
distinct. Review their source annotations before deciding to merge or replace
them in another experiment. Source CSVs and audio are read only.

`set(df["text"].astype(str))` creates a set of sentences. For a character CTC
vocabulary, join the transcripts first and sort the unique characters:

```python
letters = sorted(set("".join(normalized_training_texts)) - {" "})
vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2}
vocab.update({letter: index + 3 for index, letter in enumerate(letters)})
```

This corpus produces 39 tokens. Space becomes `|`. `[PAD]` is also the CTC blank;
the model and tokenizer both use ID 0. No BOS or EOS tokens are added. The
vocabulary is built from the original training manifest, never the test manifest.
Every training transcript must round trip through the tokenizer with
`group_tokens=False` and without unknown tokens.

Audio is normalized per recording and padded dynamically. Label padding is
masked to `-100`; it is excluded from CTC targets. The script checks that target
length plus repeated-character separation fits the output sequence. Evaluation
collapses CTC predictions and removes blanks, but compares with the original
normalized reference text, so repeated reference letters are preserved. Reported
WER and CER are corpus totals from JiWER, expressed as fractions (0.25 = 25%).

## Model and training

- LoRA: rank 16, alpha 32, dropout 0.05, `q_proj` and `v_proj` in all 24 layers.
- The randomly initialized character `lm_head` is trained and saved using
  `modules_to_save=["lm_head"]`.
- The pretrained encoder is frozen; 1,612,839 of 317,091,534 parameters train.
- Defaults: eight epochs, batch size 4, four gradient accumulation batches,
  effective batch size 16, adapter LR 0.0003, head LR 0.001, cosine schedule,
  10% warmup, gradient clipping 1.0, and BF16 autocast with FP32 CTC.
- Duration buckets reduce padding; gradients are checkpointed without reentrant
  execution. CUDA is preferred, then Apple MPS, then CPU. On MPS, CTC runs on CPU
  with a differentiable transfer back to the GPU.
- Best adapter selection uses validation WER, with CER breaking ties. Early
  stopping requires three unimproved validation epochs and at least five epochs.
- The test split is evaluated once after model selection, using the exported
  model reloaded from disk. Empty predictions and blank frame rates are recorded
  to make CTC blank collapse visible.

## Run

The older `src/ai_ml` package has been removed in this checkout. Install the
dependencies without building that package:

```bash
uv sync --no-install-project
```

Download the base model once (a network connection is required):

```bash
HF_HOME="$PWD/checkpoints/hf-cache" .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "facebook/wav2vec2-xls-r-300m",
    revision="1a640f32ac3e39899438a2931f9924c02f080a54",
    allow_patterns=["config.json", "preprocessor_config.json", "pytorch_model.bin", "model.safetensors"],
)
PY
```

Run the two-update gradient check in its own directory:

```bash
HF_HOME="$PWD/checkpoints/hf-cache" HF_HUB_OFFLINE=1 \
  .venv/bin/python -u train.py --preflight \
  --output checkpoints/wav2vec2-xlsr300m-lora-kaa-preflight-new
```

Train in a new directory:

```bash
HF_HOME="$PWD/checkpoints/hf-cache" HF_HUB_OFFLINE=1 \
  .venv/bin/python -u train.py \
  --output checkpoints/wav2vec2-xlsr300m-lora-kaa-run02
```

On macOS, prefix Python with `caffeinate -i` to prevent idle sleep. Use
`--precision fp32` if BF16 is unsupported. Each invocation refuses to overwrite
an existing run unless `--resume` is supplied. Resume from an epoch checkpoint
with the original command/settings and `--resume`:

```bash
HF_HOME="$PWD/checkpoints/hf-cache" HF_HUB_OFFLINE=1 \
  .venv/bin/python -u train.py \
  --output checkpoints/wav2vec2-xlsr300m-lora-kaa-run02 \
  --resume checkpoints/wav2vec2-xlsr300m-lora-kaa-run02/training_checkpoints/epoch-01
```

Resume restores adapters, the head, optimizer, scheduler, step counters, and RNG
state at an epoch boundary. Saved manifests must still match the dataset.

## Outputs

All weights and run files stay under the ignored `/checkpoints/` folder:

```text
wav2vec2-xlsr300m-lora-kaa-run01/
  processor/                    # vocabulary, tokenizer, audio preprocessing
  training_checkpoints/epoch-*/  # resumable adapters, head, optimizer, scheduler, RNG
  best_adapter/                 # validation-selected LoRA + trained CTC head
  model/                        # standalone merged XLS-R CTC model + processor
  data_audit.json
  settings.json
  environment.json
  parameters.json
  train_split.csv
  validation_split.csv
  test_split.csv
  baseline_validation_metrics.json
  baseline_validation_predictions.csv
  history.json
  steps.jsonl
  progress.json
  validation_epoch_*.csv
  test_metrics.json
  test_predictions.csv
```

`progress.json` updates after each optimizer step. Logs sit beside the run folder.
The model folder can be loaded directly with `Wav2Vec2ForCTC.from_pretrained`
and `Wav2Vec2Processor.from_pretrained`; the adapter folder requires the pinned
base model and custom CTC configuration.

Transcribe a recording:

```bash
.venv/bin/python infer.py \
  --model checkpoints/wav2vec2-xlsr300m-lora-kaa-run01/model \
  --audio data/karakalpak-combined/test/audio/RECORDING.wav
```

Rerun evaluation explicitly:

```bash
.venv/bin/python evaluate.py \
  --model checkpoints/wav2vec2-xlsr300m-lora-kaa-run01/model \
  --output checkpoints/wav2vec2-xlsr300m-lora-kaa-run01/reevaluation
```

The SSH training host `192.168.1.88` was unreachable during setup, so its original
scripts and the exact cause of the previous 100% WER/CER could not be inspected.
The vocabulary, blank ID, label padding, head freezing, and decoding contracts
are checked in this implementation. A successful gradient check establishes that
optimization works; accuracy must be judged from saved validation/test metrics.
