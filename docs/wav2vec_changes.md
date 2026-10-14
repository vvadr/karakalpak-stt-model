# Wav2vec2 changes and the second training run

The root `train.py`, `evaluate.py`, and `infer.py` were empty before this work.
I implemented them using a new package, `scripts/wav2vec_lora/`.
Your current `src/*.py` files were reviewed without changing or executing them.
The comparisons below describe the new implementation; they are not edits to `src`.

## Actual edits to existing files

| File and current lines | Before | After |
| --- | --- | --- |
| `train.py:1–5` | Empty file | Imports `main` from `scripts.wav2vec_lora.train` and runs it under the main guard. |
| `evaluate.py:1–32` | Empty file | CLI that reloads a standalone exported model and writes predictions and WER/CER. |
| `infer.py:1–27` | Empty file | CLI that loads/resamples audio to mono 16 kHz and transcribes it with the exported model. |
| `pyproject.toml:19` | PEFT absent from dependencies | Added `"peft>=0.21.0",`. |
| `uv.lock:44,67,2640–2659` | No PEFT package entry or project dependency | Added the PEFT dependency and resolved package metadata; installed version is 0.21.2. |
| `README.md:57–64` | No wav2vec2 training section | Added the training entry point, artifact locations, and documentation link. |

The entire added root training entry point is:

```python
from scripts.wav2vec_lora.train import main


if __name__ == "__main__":
    main()
```

## New files and the relevant lines

All lines in `scripts/wav2vec_lora/{__init__,config,data,model,train}.py`
are additions: these files did not previously exist.

| Behavior | Earlier `src` code, left unchanged | New implementation and current lines |
| --- | --- | --- |
| Vocabulary | `src/vocab.py:16–25` already collects characters, then adds special tokens at the end. The expression `set(df["text"].astype(str))` mentioned in the request would instead collect whole sentences. | `data.py:94–100`: joins transcripts before collecting letters; `[PAD]=0`, `[UNK]=1`, delimiter `|=2`; disables BOS/EOS. `data.py:108–111` checks every transcript for unknown tokens and correct decoding. A nonzero blank ID in the old code is not inherently invalid. |
| Text | `src/dataset.py:66` strips and lowercases. | `data.py:15–17`: NFC Unicode normalization, lowercase, and single spaces. Accents and dotless i are preserved. |
| Audio and labels | `src/dataset.py` loads audio and tokenizes labels; `src/collator.py:16` already masks padded labels with `-100`. | `data.py:39–74` audits every recording; `data.py:124–145` uses float32 raw waveforms, attention masks, and `-100` label padding. |
| Validation | `src/train.py:42` uses the test dataset for epoch evaluation and best-model selection. | `data.py:77–87`, `train.py:176`: creates validation from the training pool; 3,335 training, 373 validation, 927 final test recordings. |
| LoRA and head | `src/model.py:17` uses rank 8, alpha 16, dropout 0.1, q/v projections, and saves the trainable CTC head. | `config.py:16–17`, `model.py:36–43`: rank 16, alpha 32, dropout 0.05, q/v projections; explicitly checks the head is trainable. Saving the head was already correct in `src`. |
| Learning rate | `src/train.py:40`: one learning rate, `1e-4`, for 10 epochs. | `config.py:9–14`, `train.py:83–91`: 8 epochs; LoRA `3e-4`, CTC head `1e-3`, 10% warmup. These are training choices, not proof that the old rates were a bug. |
| CTC validity and MPS | `src/model.py:13` enables `ctc_zero_infinity=True`; the original training script relies on the model's loss. | `model.py:52–71`: checks that targets plus repeated letters fit the audio output and contain no blank IDs; computes FP32 CTC on CPU for MPS while preserving gradients; invalid losses raise errors. |
| Gradient accumulation | `src/train.py:40–42` delegates accumulation and loss scaling to Trainer. | `train.py:240–243`: explicitly divides the microbatch mean loss by the actual accumulation window size, including the final partial window. |
| Prediction padding | `src/metrics.py:12–17` decodes argmax over all returned frames, including padding. | `train.py:58–62`: derives valid convolution output lengths from audio masks and slices predictions before decoding. |
| Fresh runs and resume | `src/train.py:44–48` automatically resumes the most recent checkpoint in a shared `outputs/` folder. | `train.py:153,163–168,207–222`: resume requires `--resume`; a fresh run refuses to overwrite existing settings; resume restores state and verifies settings/splits. |
| Checkpoint contents | `src/train.py:24–28` saves the adapter state and training arguments; processor is saved only at the end. | `train.py:131–136,282–291`: each epoch gets adapter/head, model configuration, processor/vocabulary, optimizer, scheduler, and RNG state. |
| Standalone export | `src/train.py:53–56` saves a PEFT adapter under `final_model`. | `train.py:304–315`: selects the best validation adapter, merges it into XLS-R, saves it under `model/`, reloads that export, and evaluates the final test set. |

Also added `docs/wav2vec_lora_training.md` for usage/results and
`docs/wav2vec_source_review.md` for the static review of `src`.
The dataset files and existing first-run artifacts are preserved.

## Run 02

Started the root `train.py` with the same settings as the completed first run:

```sh
HF_HOME="$PWD/checkpoints/hf-cache" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  caffeinate -i .venv/bin/python -u train.py \
  --output checkpoints/wav2vec2-xlsr300m-lora-kaa-run02 \
  --epochs 8 --batch-size 4 --accumulation 4 \
  > checkpoints/wav2vec2-xlsr300m-lora-kaa-run02.log 2>&1
```

Within the run directory, epoch checkpoints go in `training_checkpoints/`,
the selected adapter in `best_adapter/`, and the merged standalone model in `model/`.
`progress.json`, `history.json`, and `steps.jsonl` record training progress;
`test_metrics.json` is written after training and export finish.

Run 02 completed successfully with exit code 0: 8 epochs and 1,672 optimizer
updates. Training, export, and final evaluation took 73.73 minutes, excluding
initial setup, the data audit, and baseline evaluation.

| Metric | Final training epoch | Selected validation checkpoint (epoch 8) | Final test (927 recordings) |
| --- | --- | --- | --- |
| CTC loss | 0.674955 | 0.426292 | 0.426599 |
| WER | — | 55.18% | 55.44% |
| CER | — | 12.00% | 12.14% |
| Empty predictions | — | 0 | 0 |

All eight epoch directories contain `training_state.pt` for resume. The exported
model contains 424 tensors, a `[39, 1024]` CTC head, and no LoRA wrapper tensors;
the script reloaded this standalone export for final test evaluation.
Read-only SHA256 comparisons confirmed that all reviewed `src/*.py` files and
both source CSV manifests are unchanged.
