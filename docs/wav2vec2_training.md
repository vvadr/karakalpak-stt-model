# wav2vec2 LoRA training for Karakalpak

The code in `src/` fine-tunes `facebook/wav2vec2-xls-r-300m` with LoRA on `data/final`, using
STT and HF train (17,685 files, 77.45 h). It reports WER and CER for three decoders: greedy,
beam search, and beam search with a KenLM language model (shallow fusion).

The pipeline runs in four commands, in this order:

1. `python src/vocab.py` builds the character vocabulary from the train transcripts.
2. `python src/train.py` trains, prints validation WER/CER every epoch, saves the best model,
   and scores both test sets with greedy decoding.
3. `python src/lm.py` builds a 4-gram KenLM model from the train transcripts.
4. `python src/decode.py` compares greedy, beam search and beam search + KenLM on validation and
   both test sets.

Everything is written under `outputs/wav2vec2-lora-kaa/`.

## The files, one by one

### 1. `config.py`: every setting in one place

All paths and hyperparameters live here. The other files read `config.X`, so changing a value
here changes it everywhere.

| Setting | Value | Why |
| --- | --- | --- |
| `TRAIN_CSVS` | `stt/train.csv`, `hf/train.csv` | Both datasets train one model |
| `VALIDATION_CSVS` | `stt/validation.csv` | Picks the best epoch; the test sets stay unseen until the end |
| `TEST_CSVS` | `stt`, `hf` | Scored separately, because STT is 96% of the audio |
| `BASE_MODEL` | `facebook/wav2vec2-xls-r-300m` | Multilingual pretrained encoder, 300M parameters |
| `EPOCHS` | 4 | 553 optimizer steps per epoch, 2,212 in total; LoRA gains most early |
| `BATCH_SIZE` × `GRADIENT_ACCUMULATION` | 8 × 4 = 32 files per step | Measured about 3 s per step on the DGX Spark; 16 × 2 was slower there |
| `LEARNING_RATE`, `WARMUP_RATIO` | 5e-4, 10% | LoRA needs a higher rate than full fine-tuning |
| `LORA_RANK`, `LORA_ALPHA`, `LORA_DROPOUT` | 32, 64, 0.05 | About 14.2M trainable parameters (4.3%) |
| `NUM_WORKERS` | 8 | Processes loading audio in parallel, so the GPU never waits for data |
| `LM_ORDER` | 4 | 4-gram KenLM model |
| `BEAM_WIDTH` | 100 | Hypotheses kept per step in beam search |
| `ALPHAS`, `BETAS` | 0.3–1.0, 0–2 | Grid searched to tune the LM weights |
| `TUNE_FILES` | 500 | Validation files used to tune α and β |
| `DECODE_PROCESSES` | 16 | CPU processes for beam search |
| `TRAIN_FILES`, `EVAL_FILES`, `MAX_SECONDS` | `None` | Optional subset: a random sample of clips up to `MAX_SECONDS` long; `None` uses every file |
| `GRADIENT_CHECKPOINTING` | `False` | `True` recomputes each layer in the backward pass instead of storing it: far less memory, about 30% slower |
| `LOCAL=1` block | 3,000 / 300 files, ≤ 20 s, 5 epochs, batch 4 × 4, gradient checkpointing | Quick run that fits the Mac's 24 GB; writes to `outputs/wav2vec2-lora-kaa-local/` |

### 2. `dataset.py`: reading the split CSVs

`load_split(csv_names)` reads one or more split CSVs, for example `stt/train.csv` and
`hf/train.csv`. It turns each relative `audio_path` into an absolute path and joins the rows into
one Hugging Face `Dataset` with three columns: `audio_path`, `text` and `duration`.

The audio itself is not loaded here; the collator loads it (step 5). This keeps the `duration`
column available, so the trainer can batch clips of similar length together
(`train_sampling_strategy="group_by_length"`). Clips run from 2 to 38 s, so random batches would
be about half padding.

`python src/dataset.py` prints the file count and hours of each split. Expect train 17,685 files
(77.45 h), validation 1,950 (9.28 h) and test 2,473 (9.89 h).

### 3. `vocab.py`: our own character vocabulary

This builds the output alphabet the model learns, using every character that appears in the
train transcripts:

- the 32 Karakalpak letters: `a … z`, `á ó ú ı ń ǵ`
- `|` as the word separator; the tokenizer turns each space into `|`
- `[UNK]` for any unknown character
- `[PAD]`, which is also the CTC blank

That gives 35 tokens, saved to `outputs/wav2vec2-lora-kaa/vocab.json`. The script also checks the
validation and test text and prints any letter that is missing from the vocabulary; it should
print `none`.

### 4. `processor.py`: text ↔ ids and audio → model input

`build_processor()` combines two parts:

- **`Wav2Vec2CTCTokenizer`** maps characters to ids using `vocab.json`. `bos_token=None` and
  `eos_token=None` stop it from adding unused `<s>`/`</s>` tokens, so the model has exactly 35
  outputs, matching the labels the decoders use.
- **`Wav2Vec2FeatureExtractor`** takes raw 16 kHz audio and normalizes each clip to zero mean and
  unit variance. `return_attention_mask=True` tells XLS-R which samples are padding.

`python src/processor.py` encodes and decodes a sample sentence as a check.

### 5. `collator.py`: building a batch

`DataCollatorCTC` receives a list of dataset rows. For each batch it:

1. reads the WAV files with `soundfile`
2. normalizes and pads the audio to the longest clip, with an attention mask
3. turns the transcripts into ids and pads them
4. sets the label padding to `-100`, so CTC loss ignores it

Because audio is read here, the 8 `NUM_WORKERS` processes load the next batches while the GPU
trains on the current one.

### 6. `model.py`: XLS-R + LoRA

`build_model(processor)` sets up the model in three steps:

1. **Load XLS-R with a new CTC head.** `Wav2Vec2ForCTC.from_pretrained` loads the pretrained
   encoder and adds a new `lm_head` with 35 outputs. The loading message lists `lm_head` as
   MISSING and the pretraining-only layers as UNEXPECTED; both are expected.
   `ctc_zero_infinity=True` stops impossible alignments from producing infinite loss.
2. **Freeze the CNN feature encoder.** It turns waveforms into frames and doesn't need retraining.
3. **Add LoRA adapters.** These are small trainable matrices, rank 32, on every transformer
   layer's attention (`q_proj`, `k_proj`, `v_proj`, `out_proj`) and feed-forward
   (`intermediate_dense`, `output_dense`). The new `lm_head` is trained in full
   (`modules_to_save`). Everything else stays frozen.

It prints `trainable params: 14,191,651 || all params: 329,666,246 || trainable%: 4.3049`.

### 7. `metrics.py`: WER and CER

This file has three functions:

- **`argmax_logits`** keeps only the most likely token per frame during evaluation, so the
  trainer doesn't hold every logit in memory.
- **`decode`** turns predicted ids into text. It collapses repeats and removes blanks, which is
  greedy CTC decoding, and also turns the label ids back into reference text.
- **`make_compute_metrics`** computes **WER** (word error rate) and **CER** (character error rate)
  with `jiwer`. Each is (substitutions + deletions + insertions) / reference length. Because
  insertions count, either can go above 100% early in training.

### 8. `train.py`: training loop

The script uses the Hugging Face `Trainer`, set up as follows:

- It trains for `EPOCHS`, grouping similar-length clips, using bf16 on the GPU.
- At the end of every epoch it evaluates on `stt/validation.csv`, prints `eval_wer` and
  `eval_cer` in the log, and saves a checkpoint. Only the newest and the best are kept.
- `load_best_model_at_end=True` with `metric_for_best_model="wer"` restores the epoch with the
  lowest validation WER.
- If `checkpoints/` already exists, it resumes from the latest checkpoint. Running the same
  command again continues an interrupted run.

After training it:

1. writes `history.csv` (epoch, step, validation loss, WER, CER) and prints that table
2. merges the LoRA weights into the base model and saves a standalone model plus processor to
   `final_model/`
3. calls `evaluate_model.py` to score both test sets

### 9. `evaluate_model.py`: test WER/CER (greedy)

`evaluate_tests(model_dir)` loads `final_model/` and runs each test set from `TEST_CSVS`. For each
set it prints WER, CER and three example reference/prediction pairs. It also writes:

- `results/predictions_<set>.csv` with `audio_path`, `reference` and `prediction`
- `results/test_results.json`

You can run it on its own, for example on a copied model:
`python src/evaluate_model.py path/to/final_model`.

### 10. `lm.py`: the KenLM language model

A language model scores word sequences, so the beam search prefers real Karakalpak words and word
orders. `lm.py` builds one in three steps:

1. **Collect the text.** It takes the train transcripts (STT + HF) and removes any sentence that
   also appears in validation or test, so test sentences can't leak in. With the current splits
   none are removed.
2. **Write it out.** The text goes to `lm/lm_text.txt`, one sentence per line.
3. **Run KenLM's `lmplz`.** It builds a 4-gram model with Kneser–Ney smoothing:
   `lmplz -o 4 --discount_fallback`. The fallback flag is needed for small corpora. The result
   is `lm/kaa_4gram.arpa`.

`lmplz` is a compiled KenLM program, expected at `kenlm/build/bin/lmplz` (`KENLM_BIN` in
`config.py`). The server steps below show how to build it.

### 11. `decode.py`: greedy vs beam search vs KenLM

The script runs the model once per set (validation, STT test, HF test) and keeps each file's
logits, without padding frames. It then decodes those logits three ways:

| Decoder | How it works |
| --- | --- |
| `greedy` | Most likely token per frame, then collapse repeats and remove blanks. Same as during training. |
| `beam` | CTC prefix beam search (`pyctcdecode`, width 100), no language model. Keeps the 100 best partial transcripts instead of one. |
| `kenlm` | The same beam search with **shallow fusion**: each hypothesis is scored as `log P_acoustic + α · log P_KenLM + β · words`. α weights the LM; β rewards each word so the LM doesn't favour short outputs. |

α and β are chosen by grid search over `ALPHAS` × `BETAS` (12 combinations) on the first 500
validation files, picking the lowest WER. The test sets are then decoded with those values, so
test scores stay unbiased. The validation `kenlm` score is slightly optimistic, because tuning
used part of it.

The script prints a table with WER and CER for each set × decoder and writes:

- `results/decoder_results.json` with the chosen α, β and every score
- `results/decoded_<set>.csv` with `audio_path`, `reference`, `greedy`, `beam` and `kenlm`

`pyctcdecode` needs the labels in the same order as the model outputs. `ctc_labels()` builds them
from the tokenizer, mapping `[PAD]` to the blank `""` and `|` to a space.

## Quick local run on the Mac

The full settings don't fit the Mac's 24 GB: a batch of 16 clips up to 38 s needs about 50 GB, and
the run stops with `MPS backend out of memory`. Setting `LOCAL=1` switches `config.py` to a subset:

- 3,000 random train clips of at most 20 s (10.9 h) and 300 clips each for validation and testing
- 5 epochs with batches of 4 clips × 4 accumulation steps (16 clips per step), 940 steps in total
- gradient checkpointing on: without it, even 4 clips of 20 s ran out of memory, because the Mac keeps
  full attention matrices for all 24 layers
- results in `outputs/wav2vec2-lora-kaa-local/`, separate from full runs

Nothing changes on the server, which never sets `LOCAL`. The local scores use a smaller, shorter
test sample, so they show whether training works but aren't comparable to the server's.

```bash
cd ~/Documents/IT/repos/"AI & ML"
export LOCAL=1 HF_HUB_CACHE="$PWD/checkpoints/hf-cache/hub"   # reuses the base model already on the Mac
.venv/bin/python src/vocab.py
nohup .venv/bin/python src/train.py > outputs/wav2vec2-lora-kaa-local/train.log 2>&1 &
```

`decode.py` also runs locally once `pyctcdecode` and `kenlm` are installed, but building the KenLM
model needs `lmplz`, which in turn needs Boost (`brew install boost`), so run `lm.py` on the server.

## Run it on the server

**On the Mac, copy the code and this doc to the server.** This overwrites the server's versions
of these files; the old ones are in git commit `d8c9cfc`.

```bash
cd ~/Documents/IT/repos/"AI & ML"
rsync -avc --exclude=__pycache__ --exclude=.DS_Store src/ "intern@192.168.1.88:/home/intern/AI\ \&\ ML/src/"
rsync -avc docs/wav2vec2_training.md "intern@192.168.1.88:/home/intern/AI\ \&\ ML/docs/"
```

**On the server, install the decoder packages (once).** `pyctcdecode` declares `numpy<2` but works
with numpy 2; `--no-deps` keeps your numpy, torch and pandas untouched.

```bash
cd "/home/intern/AI & ML" && source .venv/bin/activate
pip install --no-deps pyctcdecode pygtrie
pip install kenlm
```

If the venv has no `pip` (made by `uv`), use `uv pip install` with the same arguments.

**On the server, make sure KenLM's `lmplz` exists (once).**

```bash
cd "/home/intern/AI & ML"
find kenlm -name lmplz -type f 2>/dev/null
```

- If it prints `kenlm/build/bin/lmplz`, it's ready.
- If it prints another path, set `KENLM_BIN` in `src/config.py` to that folder.
- If it prints nothing, build it:

```bash
cd "/home/intern/AI & ML"
sudo apt-get install -y build-essential cmake libboost-program-options-dev libboost-system-dev libboost-thread-dev libboost-test-dev zlib1g-dev libbz2-dev liblzma-dev
[ -f kenlm/CMakeLists.txt ] || git clone https://github.com/kpu/kenlm.git kenlm
mkdir -p kenlm/build && cd kenlm/build && cmake .. && make -j 8
```

If `git clone` says `kenlm` already exists, rename the old folder (`mv kenlm kenlm_old`) and run
the block again.

**On the server, run the pipeline.**

```bash
cd "/home/intern/AI & ML" && source .venv/bin/activate
python src/vocab.py                                    # Vocabulary size: 35, missing: none
nohup python src/train.py > train.log 2>&1 &           # runs in the background
tail -f train.log                                      # Ctrl+C stops watching, not training
```

After training finishes (the log ends with the test WER/CER):

```bash
cd "/home/intern/AI & ML" && source .venv/bin/activate
python src/lm.py                                       # seconds
python src/decode.py 2>&1 | tee decode.log             # ~10–30 min: tuning + 3 sets x 3 decoders
```

## Where to see WER and CER

| When | Where |
| --- | --- |
| Every epoch, during training | `train.log` lines with `eval_wer` and `eval_cer` (validation) |
| After training | `outputs/wav2vec2-lora-kaa/history.csv`: validation WER/CER per epoch |
| After training | `outputs/wav2vec2-lora-kaa/results/test_results.json`: greedy test WER/CER per dataset |
| After `decode.py` | `outputs/wav2vec2-lora-kaa/results/decoder_results.json`: all decoders, α, β |
| To read actual outputs | `results/predictions_*.csv` and `results/decoded_*.csv` |

For reference, the earlier LoRA run on 8.5 h of HF + Mendeley data reached WER 55.4% / CER 12.1%
with greedy decoding. Beam search alone usually changes WER only slightly; KenLM usually lowers
WER more than CER, because it fixes word choice and spelling.

## What was tested

On the Mac (CPU/MPS), with 16 short train files, 2 epochs and a stand-in LM file, every step ran
end to end:

- the vocabulary (35 tokens, nothing missing in validation/test)
- training, per-epoch WER/CER and `history.csv`
- choosing the best epoch, merging and saving `final_model/`
- test WER/CER and prediction CSVs
- `lm.py` writing the LM text and calling `lmplz`
- `decode.py` loading the ARPA LM, tuning α/β and writing all three decoders' results

Not tested here:

- the real `lmplz` (it needs Boost, which the Mac doesn't have)
- CUDA/bf16
- full-size runs

The first server run of each step is the real test; paste any error into Claude Code.

## If something goes wrong

- **Out of GPU memory:** set `BATCH_SIZE = 4` and `GRADIENT_ACCUMULATION = 8` in `config.py`. The
  effective batch stays 32. Then run `train.py` again; it resumes.
- **Restart training from scratch:** delete `outputs/wav2vec2-lora-kaa/checkpoints/`.
- **`FileNotFoundError: .../kenlm/build/bin/lmplz`:** see "make sure KenLM's `lmplz` exists" above.
- **`decode.py` is slow:** lower `BEAM_WIDTH` (e.g. 50) or `TUNE_FILES` (e.g. 200), or raise
  `DECODE_PROCESSES` up to the server's CPU core count (`nproc`).
- **`ModuleNotFoundError: pyctcdecode` / `kenlm`:** run the install step in the activated venv.
