# Static review of `src`

Reviewed on 2026-10-13 after the separate XLS-R LoRA run completed. No module
under `src` was imported, executed, or edited. This review covers the supplied
local files and the installed Transformers 5.19.0 / PEFT 0.21.2 implementation.
The SSH host was unreachable, so the remote library versions, old checkpoints,
and old logs were not available. The findings below identify concrete code
problems; they do not establish a single proven cause of the old remote run.

## 1. Reruns automatically restore the previous experiment

**Location:** `src/train.py:16`, `src/train.py:44-48`.

Every experiment uses the same `outputs` directory. Whenever a checkpoint is
present, the script resumes it automatically. Trainer restores optimizer and
scheduler state and calculates which epochs/batches to skip from the saved
global step. Changing the learning rate in `TrainingArguments` therefore does
not create a fresh experiment with a fresh optimizer/schedule. If the saved run
has already completed the requested number of epochs, the rerun can perform no
new training at all. Changing LoRA rank or vocabulary while loading old weights
also creates an incompatible experiment.

**Correction:** give each experiment its own directory and require an explicit
resume argument. Resume only with compatible model, vocabulary, dataset, and
training settings. Keep previous checkpoints so the experiments remain
comparable. This is the strongest explanation in the supplied code for why
hyperparameter changes could appear to have no effect.

## 2. Gradient accumulation can use incorrectly scaled CTC loss

**Location:** `src/train.py:19`, `src/train.py:40-42`, with CTC reduction set in
`src/model.py:13`.

In the installed Transformers 5.19.0 implementation:

1. Trainer inspects the underlying wav2vec2 forward method. Its `**kwargs`
   signature makes Trainer infer that it accepts loss normalization arguments.
2. Trainer supplies `num_items_in_batch` and assumes that the returned loss has
   already been normalized over the accumulated batches.
3. Wav2vec2 CTC ignores that normalization argument and returns a separate mean
   CTC loss for each microbatch.
4. The default Trainer path consequently skips division by the four accumulation
   batches. Accelerate is configured with accumulation scaling equal to one
   because Trainer normally handles the division itself.

For complete four-batch windows, gradients and reported accumulated training
loss therefore use a sum where a mean is intended. This makes loss logs and
gradient clipping misleading and changes optimization. It does not by itself
prove why the previous remote model produced blank transcriptions; the remote
Transformers version would determine whether this particular defect applies.

**Correction for 5.19:** explicitly declare the per-microbatch mean loss:

```python
class CTCTrainer(Trainer):
    loss_is_scaled_for_ga = False
    # Keep the existing save implementation only after correcting item 4 below.
```

Also set `model.get_base_model().accepts_loss_kwargs = False` before constructing
Trainer, so it does not assume that wav2vec2 consumes `num_items_in_batch`.
These are suggested edits only; they have not been applied or run in `src`.
The normalization contract is described in the official
[Trainer documentation](https://huggingface.co/docs/transformers/main/main_classes/trainer).

## 3. Evaluation padding can become an extra letter `a`

**Location:** `src/metrics.py:12`, `src/metrics.py:17`, `src/vocab.py:22-25`.

Trainer pads variable-length evaluation logits with `-100` before concatenating
batches. `np.argmax` of a frame containing only `-100` returns token ID 0. Your
sorted vocabulary assigns ID 0 to `a`, while `[PAD]`/CTC blank is the last ID.
The metric function therefore decodes padding as a real letter. Repeated padded
frames collapse into an appended `a`, which can corrupt the final word and
inflate WER/CER. This is a metric defect, not proof that the model itself is
emitting that suffix.

**Correction:** replace IDs selected from entirely padded logit frames with the
tokenizer's blank ID before decoding. Also trim each prediction to its real
audio output length, using the attention mask and the encoder convolution
lengths. Assigning blank ID 0 avoids the particular concatenation-padding problem,
but a nonzero blank ID is mathematically valid when masking/decoding is correct.
Reference decoding with `group_tokens=False` at line 18 is already correct.

## 4. Saved checkpoints do not contain the full processor/model configuration

**Location:** `src/train.py:20-28`, `src/train.py:42`, `src/train.py:53-56`.

The overridden `_save` saves PEFT adapter weights and training arguments.
Trainer is not given `processing_class`, and the override does not save the
processor or the underlying custom CTC model configuration. Individual epoch
checkpoints therefore depend on the mutable root `vocab.json`; the vocabulary
and token IDs used to train that checkpoint are not preserved beside it.

The folder named `final_model` also contains a PEFT adapter, because
`save_pretrained` is called on a `PeftModel`. Saving the processor afterward
does not turn those adapter weights into a standalone 300M model. Rebuilding the
base model with its original output configuration can give a head size/blank ID
mismatch; ignoring such a mismatch can discard the trained head.

**Correction:** save the exact processor, custom base configuration, and pinned
base revision with every checkpoint. Keep the adapter plus trained `lm_head`
for resume. Separately merge the selected adapter into a base model and export
that model with its processor for standalone inference. The distinction between
adapter storage and merged model storage is documented in
[PEFT checkpoint format](https://huggingface.co/docs/peft/en/developer_guides/checkpoint).

## 5. The test split is used to select the best model

**Location:** `src/train.py:33`, `src/train.py:40-42`.

`eval_dataset=test_dataset` is evaluated after every epoch, and its WER selects
the best checkpoint. Those recordings are consequently validation data for this
experiment. They no longer provide an independent final test estimate. This
does not explain blank collapse, but it compromises the evaluation procedure.

**Correction:** reserve validation groups from `train.csv`; use `test.csv` only
for the final evaluation after selecting the model.

## Smaller issues and checks

- `src/metrics.py:7` constructs the processor at import time, and
  `src/processor.py:7-11` requires root `vocab.json`. That file is currently
  absent. The supplied training entrypoint would fail during import until the
  vocabulary is created. This explains a startup failure, not a run reporting
  100% WER/CER. Prefer building/passing one processor inside the training flow.
- Imports such as `from processor import ...` assume direct execution from
  `src`. They will fail for normal package invocation with `python -m src.train`.
  Relative imports provide a consistent package entrypoint.
- `src/processor.py:11` leaves default BOS/EOS tokens enabled although the
  vocabulary file does not define them. Set them to `None` for the intended
  character CTC alphabet; otherwise the tokenizer adds unused special tokens.
- `src/model.py:13` uses `ctc_zero_infinity=True`, and the data path has no check
  that label length plus repeated-letter separation fits the output frames.
  Impossible alignments can be silently zeroed. This is a missing diagnostic,
  not an observed defect in the audited combined corpus. Use explicit length
  checks and expose nonfinite losses while diagnosing.
- `src/train.py:40` uses one LR of `1e-4` for both LoRA and the new CTC head.
  That is an optimization choice, not a demonstrated code error. The completed
  run used separate rates: `3e-4` for LoRA and `1e-3` for the head.

## Parts that already satisfy the CTC contract

`src/vocab.py:14-22` collects and sorts individual characters correctly. It does
not use the sentence-level `set(df["text"].astype(str))` expression mentioned
earlier. `src/model.py:13` aligns the model's blank ID with the tokenizer's pad
ID. `src/model.py:17` includes the new head in `modules_to_save`, so it is
trainable. `src/collator.py:16` masks target padding to `-100`.
`src/metrics.py:18` preserves repeated reference letters. These should be retained.

`warmup_steps=0.1` is valid in the installed Transformers 5.19.0: it means 10%
of training steps. It should not be diagnosed as a fractional-step mistake for
this version. The older remote environment has not been inspected. See the
[current warmup API](https://huggingface.co/docs/transformers/main/main_classes/trainer#transformers.TrainingArguments).

## What the completed run demonstrates

The independent implementation under `scripts/wav2vec_lora` used the same
combined corpus and pretrained XLS-R 300M architecture, with a 39-token character
vocabulary and explicit CTC/gradient checks. It initially reached the same blank
state: epoch 1 validation loss 3.148 and both error rates 100%. At epoch 2, CER
fell to 46.20%; at epoch 3, WER fell to 73.36% and CER to 17.08%. After eight
epochs the exported model achieved test WER 55.44%, CER 12.14%, and loss 0.427.

Loss near 3 with all blank predictions can occur early in valid CTC training.
It becomes a diagnostic concern when the model never escapes that state.
Investigate actual optimizer updates, fresh versus resumed runs, vocabulary
parity, blank rates, head/adapter gradients, and real predictions before treating
every persistent 100% metric as a hyperparameter problem. The static findings
above are the specific corrections justified by the supplied files.
