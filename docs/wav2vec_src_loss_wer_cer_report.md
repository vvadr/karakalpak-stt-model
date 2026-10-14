# Wav2vec2 XLS-R 300M: source review and changes associated with lower loss, WER, and CER

**Report date:** 2026-10-14  
**Dataset:** `data/karakalpak-combined`  
**Model:** `facebook/wav2vec2-xls-r-300m`, with PEFT LoRA and a custom character vocabulary

## 1. What was actually changed

**Your current `src/*.py` files were reviewed, but were not edited or executed.**
The successful training used the new implementation in `scripts/wav2vec_lora/`,
called by the root `train.py`. This report compares your supplied `src` code
with that implementation and explains which changes affect learning, metric
calculation, and reliable model loading.

The root `train.py` was previously empty. Its added lines 1–5 are:

```python
from scripts.wav2vec_lora.train import main


if __name__ == "__main__":
    main()
```

All files under `scripts/wav2vec_lora/` are additions, rather than patches to
`src`. Suggested corrections to `src` below have **not** been applied there.
The original dataset manifests and the first training run were preserved.

## 2. Measured results

The completed second run used 3,335 training recordings, 373 validation
recordings selected from the original training pool, and 927 final test
recordings. It ran on Apple MPS with BF16 forward computation and FP32 CTC loss.
The recorded environment was PyTorch 2.14.0, Transformers 5.19.0, and PEFT 0.21.2.

| Epoch | Training loss | Validation loss | Validation WER | Validation CER | Empty validation predictions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.1793 | 3.1483 | 100.00% | 100.00% | 373 |
| 2 | 2.4959 | 1.1651 | 99.96% | 46.20% | 0 |
| 3 | 1.0680 | 0.5926 | 73.36% | 17.08% | 0 |
| 4 | 0.8236 | 0.4910 | 61.27% | 13.79% | 0 |
| 5 | 0.7422 | 0.4507 | 57.92% | 12.81% | 0 |
| 6 | 0.7022 | 0.4323 | 55.87% | 12.25% | 0 |
| 7 | 0.6822 | 0.4271 | 55.22% | 11.99% | 0 |
| 8 | 0.6750 | 0.4263 | 55.18% | 12.00% | 0 |

Epoch 8 was selected by the lowest validation WER. Its merged model was saved,
reloaded from disk, and evaluated on all 927 test recordings:

| Final test metric | Value |
| --- | ---: |
| CTC loss | 0.4266 |
| WER | 55.44% |
| CER | 12.14% |
| Empty predictions | 0 |

These measurements demonstrate improvement **within the new training run**.
They do not prove that one particular edit caused the improvement: several
implementation and hyperparameter choices changed together, and no experiments
were run to isolate their individual effects. The old remote logs, checkpoints,
and library versions were unavailable, so the original remote loss near 3 is
not a controlled comparison with these losses.

**Early blank predictions were not sufficient evidence of failure.** This run
also produced only blanks and 100% WER/CER after epoch 1, then learned useful
characters in epoch 2 and substantially improved WER in epoch 3.

## 3. Exact code comparison

Locations in the right column are under `scripts/wav2vec_lora/`.

| Original location in `src` | Implemented location | Change and purpose |
| --- | --- | --- |
| `train.py:16,44–48` | `train.py:153,161–168` | Separate run directories and explicit resume, so a new experiment does not silently restore an old optimizer and schedule. |
| `train.py:40–42`; `model.py:13` | `train.py:240–243` | Explicitly scale each microbatch mean loss by the actual accumulation window size. |
| `metrics.py:12–17` | `train.py:58–62`; `model.py:47–49` | Trim predictions to the real audio output length before decoding, excluding padded frames from metrics. |
| `vocab.py:16–25`; `processor.py:11` | `data.py:94–111` | Build a saved letter vocabulary, disable BOS/EOS, and check every transcript for unknown tokens and correct decoding. |
| `dataset.py:66` | `data.py:15–17` | Normalize Unicode to NFC, lowercase text, and use single spaces while preserving accented letters and dotless i. |
| `train.py:40`; `model.py:17` | `config.py:9–17`; `train.py:83–91`; `model.py:36–43` | Use separate head/LoRA learning rates and changed LoRA settings; check that the new head is trainable. |
| `model.py:13` | `model.py:52–71` | Check CTC alignment lengths, reject blank tokens in targets, expose invalid losses, and support differentiable CTC computation on MPS. |
| `train.py:42` | `data.py:77–87`; `train.py:176,313–315` | Select checkpoints using validation from the training pool; evaluate the original test split after selection. |
| `train.py:20–28,53–56` | `train.py:131–136,282–291,304–315` | Save processor/configuration with each adapter checkpoint, then export and reload a separate standalone model. |

## 4. Changes that affect optimization

### 4.1 A fresh run must not silently resume the previous experiment

Your [`src/train.py`](../src/train.py), lines 44–48, contains:

```python
last_checkpoint = get_last_checkpoint(str(OUTPUT_DIR)) if OUTPUT_DIR.exists() else None
if last_checkpoint:
    trainer.train(resume_from_checkpoint=last_checkpoint)
```

This reuses the same `outputs/` directory and restores training state whenever
a checkpoint exists. A changed learning rate in the new arguments can be
superseded by the restored optimizer/scheduler state. A checkpoint that already
reached the requested epoch count can also cause the rerun to skip new training.
Consequently, changing hyperparameters may appear to have no effect.

The new [`training entry point`](../scripts/wav2vec_lora/train.py), lines
153 and 161–168, requires `--resume` explicitly and refuses to overwrite an
existing run accidentally. Run 02 used a new output directory and a fresh
optimizer and schedule. This makes the experiment reproducible and ensures
that a requested fresh run actually trains.

### 4.2 Accumulated mean CTC losses need explicit scaling

Your `src/train.py:40–42` delegates four-step gradient accumulation to Trainer.
Your `src/model.py:13` requests a separate mean CTC loss for each microbatch.

The inspected local Transformers 5.19.0 code infers loss-normalization support
from the wav2vec2 forward method's `**kwargs`. Wav2vec2's CTC calculation does
not normalize that microbatch mean using `num_items_in_batch`. This can make
Trainer treat an unscaled loss as already scaled and omit division by the
accumulation count. It is a version-dependent finding, not a confirmed diagnosis
of the older remote environment.

The new loop at `scripts/wav2vec_lora/train.py:240–243` does the division directly:

```python
window_start = (index // config.accumulation) * config.accumulation
window_size = min(config.accumulation, len(train_batches) - window_start)
(loss / window_size).backward()
```

For a complete four-microbatch window, this averages the four mean losses
instead of summing them. The final partial window uses its actual size.
This changes gradient magnitudes and makes gradient clipping meaningful.
Training loss is logged from the unscaled loss, separately from the value
divided for backpropagation, so changing accumulation does not simply multiply
the displayed loss.

If the original Trainer implementation is retained, the locally identified
correction for Transformers 5.19.0 is to explicitly declare its loss scaling:

```python
# Suggested source correction; not applied to src.
class CTCTrainer(Trainer):
    loss_is_scaled_for_ga = False
    # Retain and correct the existing checkpoint-save implementation.

model.get_base_model().accepts_loss_kwargs = False
```

The flag tells Trainer to divide an unscaled loss by the accumulation count.
Its normalization contract is described in the official
[Trainer documentation](https://huggingface.co/docs/transformers/main/main_classes/trainer).
This suggestion requires checking the installed version before porting it to
the remote machine.

### 4.3 Give the new CTC head and LoRA adapters separate learning rates

| Setting | Supplied `src` code | Completed implementation |
| --- | ---: | ---: |
| LoRA learning rate | `1e-4` | `3e-4` |
| CTC head learning rate | `1e-4` | `1e-3` |
| LoRA rank | 8 | 16 |
| LoRA alpha | 16 | 32 |
| LoRA dropout | 0.10 | 0.05 |
| Epochs | 10 | 8 |
| Microbatch size | 4 | 4 |
| Accumulation | 4 | 4 |

The head learns the mapping to the new alphabet, so the completed run used a
higher learning rate for it than for the adapters. Optimizer parameter groups
are defined at `scripts/wav2vec_lora/train.py:83–91`.

These are **tuning choices**, not proof that `1e-4` or rank 8 was wrong.
The source already includes `modules_to_save=["lm_head"]`, so its head was not
incorrectly frozen by that configuration. The new implementation retains that
setting and adds a trainability check at `model.py:42–43`.

## 5. Changes that make WER and CER reflect real predictions

### 5.1 Exclude padded output frames before decoding

Your `src/metrics.py:12–17` takes argmax over all returned logits and decodes
the resulting token IDs. Trainer pads concatenated evaluation logits with
`-100` when their time dimensions differ. Argmax on a frame containing only
`-100` returns ID 0; your original vocabulary maps ID 0 to `a`. Padding can
therefore appear as a real letter in the decoded prediction.

The new evaluator at `scripts/wav2vec_lora/train.py:58–62` computes output
lengths from the audio attention masks and convolution lengths, then slices
each prediction before decoding:

```python
lengths = output_lengths(model, batch["attention_mask"]).cpu().tolist()
# Inside the per-recording loop:
prediction = prediction[:length]
hypothesis = processor.tokenizer.decode(
    prediction.tolist(), group_tokens=True, skip_special_tokens=True
)
```

This corrects metric calculation; it does not improve model weights by itself.
The padding issue can inflate errors, but it does not by itself establish why
the old remote model had 100% WER/CER.

### 5.2 Use a consistent letter vocabulary and processor

The expression from the original request:

```python
set(df["text"].astype(str))
```

collects unique **sentences**, not letters. The new vocabulary builder at
`scripts/wav2vec_lora/data.py:94–100` instead uses:

```python
letters = sorted(set("".join(texts)) - {" "})
vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2}
vocab.update({letter: index + 3 for index, letter in enumerate(letters)})
# Tokenizer uses bos_token=None and eos_token=None.
```

However, **your supplied `src/vocab.py:16–22` already collected individual
characters correctly** using `characters.update(text)`. It did not contain
the sentence-set mistake. The new code adds transcript round-trip checks,
preserves the processor per run, and avoids unused BOS/EOS tokens.

The resulting vocabulary has 36 characters plus blank/padding, unknown, and
word-delimiter tokens: 39 entries. The model's `pad_token_id` matches the
processor's blank ID. A nonzero blank ID in the original code is valid;
consistency and correct padding treatment are what matter.

Unicode normalization and spacing are handled at `data.py:15–17` without
removing accents or changing dotless `ı`. Label padding remains `-100`, and
reference decoding must preserve repeated letters.

## 6. CTC checks, data, and checkpoint reliability

- **CTC validity:** `model.py:52–71` verifies that target length plus required
  repeated-letter separation fits the available output frames and that the
  target contains no blank IDs. `zero_infinity=False` exposes invalid losses
  instead of silently replacing them with zero. These are diagnostics; the
  audited corpus did not demonstrate impossible alignments during training.
- **MPS support:** that function computes FP32 CTC on CPU when logits are on
  MPS. The tensor transfer preserves autograd, allowing gradients to update
  the adapters and head. This supports the local training device; it is not
  evidence of an equivalent problem on the remote CUDA machine.
- **Audio audit:** `data.py:39–74` checks mono 16 kHz audio, finite samples,
  readable files, duration agreement, silence, split overlap, and characters
  absent from the training alphabet. Dataset loading uses float32 waveforms.
  Saved audit results show no exact audio or normalized-transcript overlap
  between the original train and test manifests.
- **Validation:** `src/train.py:42` selects the best checkpoint using the test
  dataset. The new implementation takes validation from the training pool
  and uses the 927-recording test set after selection. This makes the final
  estimate more credible; it does not directly reduce optimization loss.
- **Checkpoint contents:** `train.py:131–136,282–291` saves adapter/head weights,
  the processor/vocabulary, model configuration, and optimizer/scheduler/RNG
  state. This prevents later loading from depending on a regenerated global
  vocabulary or a different head configuration.
- **Standalone model:** `train.py:304–315` merges the selected adapter into
  XLS-R, saves the model and processor, reloads that export, and evaluates it.
  A PEFT adapter save contains the adaptation weights rather than the complete
  base model; the distinction is explained in
  [PEFT checkpoint documentation](https://huggingface.co/docs/peft/en/developer_guides/checkpoint).

## 7. Parts of the original source that were already correct

The following were retained conceptually and should not be described as bugs:

- Character collection in `src/vocab.py:16–22`.
- Matching model blank ID to tokenizer pad ID in `src/model.py:13`.
- Saving/training `lm_head` with `modules_to_save` in `src/model.py:17`.
- Masking padded labels to `-100` in `src/collator.py:16`.
- Decoding reference labels with `group_tokens=False` in `src/metrics.py:18`.
- `warmup_steps=0.1`: valid as a 10% warmup fraction in the inspected local
  Transformers 5.19.0. The older remote version was not established. See
  [TrainingArguments](https://huggingface.co/docs/transformers/main/main_classes/trainer#transformers.TrainingArguments).

## 8. Conclusion and evidence

The strongest source-level concerns are silent checkpoint resumption,
version-dependent accumulation scaling, and decoding evaluation padding as
letters. The new implementation addresses those concerns, keeps the custom
alphabet consistent, and saves reloadable artifacts. Its measured loss and
error rates improved over eight epochs, although the individual contribution
of each change was not isolated.

The source review remains static: this report is not evidence that `src`
itself was fixed or trained. It documents the implemented alternative and the
specific source corrections identified for a later edit.

Primary saved evidence:

- [Epoch losses and validation metrics](../checkpoints/wav2vec2-xlsr300m-lora-kaa-run02/history.json).
- [Final test metrics](../checkpoints/wav2vec2-xlsr300m-lora-kaa-run02/test_metrics.json).
- [Training settings](../checkpoints/wav2vec2-xlsr300m-lora-kaa-run02/settings.json).
- [Runtime environment](../checkpoints/wav2vec2-xlsr300m-lora-kaa-run02/environment.json).
- [Data audit](../checkpoints/wav2vec2-xlsr300m-lora-kaa-run02/data_audit.json).
- [All eight resumable checkpoints](../checkpoints/wav2vec2-xlsr300m-lora-kaa-run02/training_checkpoints/).
- [Standalone exported model](../checkpoints/wav2vec2-xlsr300m-lora-kaa-run02/model/).
- [Full source review](wav2vec_source_review.md) and [file change inventory](wav2vec_changes.md).

Line numbers describe the files inspected when this report was created and
may move after later edits. Report creation did not start another training run.
