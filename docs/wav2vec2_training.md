# wav2vec2 LoRA training for Karakalpak

The code in `src/` fine-tunes `facebook/wav2vec2-xls-r-300m` with LoRA on `data/final`, using
STT and HF train (17,685 files, 77.45 h). It reports WER and CER for three decoders: greedy,
beam search, and beam search with a KenLM language model (shallow fusion). The code has no
comments; function and setting names say what they do, and this document explains each file.

The pipeline runs in four commands, in this order, plus an optional fifth:

1. `python src/vocab.py` builds the character vocabulary from the train transcripts.
2. `python src/train.py` trains, prints validation WER/CER every epoch, saves the best model,
   and scores both test sets with greedy decoding.
3. `python src/lm.py` builds a 4-gram KenLM model from the train transcripts.
4. `python src/decode.py` compares greedy, beam search and beam search + KenLM on validation and
   both test sets, and tunes the KenLM weights α and β.
5. `python src/benchmark.py <test.csv>` scores the trained model on any other test set.

Everything is written under `outputs/wav2vec2-lora-kaa/`.

## The files, one by one

### 1. `config.py`: every setting in one place

All paths and hyperparameters live here. The other files read `config.X`, so changing a value
here changes it everywhere.

| Setting | Value | Meaning |
| --- | --- | --- |
| `DATA_DIR`, `OUTPUT_DIR` | `data/final`, `outputs/wav2vec2-lora-kaa` | where the split CSVs are read; every output path (`VOCAB_PATH`, `CHECKPOINT_DIR`, `FINAL_MODEL_DIR`, `RESULTS_DIR`, `LM_DIR` …) is inside `OUTPUT_DIR` |
| `TRAIN_CSVS` | `stt/train.csv`, `hf/train.csv` | both datasets train one model |
| `VALIDATION_CSVS` | `stt/validation.csv` | picks the best epoch and tunes α/β |
| `TEST_CSVS` | `stt`, `hf` | scored separately at the end; STT is 96% of the audio |
| `SAMPLING_RATE` | 16000 | all audio is 16 kHz |
| `WORD_SEPARATOR`, `UNKNOWN_TOKEN`, `PAD_TOKEN` | `\|`, `[UNK]`, `[PAD]` | special tokens; `[PAD]` is also the CTC blank |
| `IGNORED_LABEL_ID` | -100 | label value CTC loss skips (padding) |
| `BASE_MODEL` | `facebook/wav2vec2-xls-r-300m` | multilingual pretrained encoder, 300M parameters |
| `EPOCHS` | 4 | 553 optimizer steps per epoch, 2,212 in total |
| `BATCH_SIZE` × `GRADIENT_ACCUMULATION` | 8 × 4 | 32 clips per optimizer step; about 3 s per step on the DGX Spark |
| `LEARNING_RATE`, `WARMUP_RATIO` | 5e-4, 10% | LoRA needs a higher rate than full fine-tuning |
| `GRADIENT_CHECKPOINTING` | `False` | `True` recomputes layers in the backward pass: far less memory, about 30% slower |
| `LORA_RANK`, `LORA_ALPHA`, `LORA_DROPOUT` | 32, 64, 0.05 | about 14.2M trainable parameters (4.3%) |
| `NUM_WORKERS` | 8 | processes loading audio in parallel |
| `SEED` | 42 | same shuffling and subsets on every run |
| `TRAIN_SUBSET_SIZE`, `EVAL_SUBSET_SIZE`, `SUBSET_MAX_SECONDS` | `None` | optional random subset of clips up to a length; `None` uses every file |
| `RUN_ON_MAC` | set by `LOCAL=1` | switches to the small preset for the Mac (below) |
| `KENLM_BIN`, `LMPLZ_MEMORY_LIMIT`, `LM_ORDER` | `kenlm/build/bin`, `1G`, 4 | where `lmplz` is, its memory cap, n-gram order |
| `BEAM_WIDTH` | 100 | hypotheses kept per step in beam search |
| `LM_WEIGHT_CANDIDATES` (α), `WORD_BONUS_CANDIDATES` (β) | 0.3–1.0, 0–2 | grid searched to tune KenLM |
| `TUNING_FILES` | 500 | validation files used to tune α and β |
| `DECODE_PROCESSES` | 16 | CPU processes for beam search |

### 2. `dataset.py`: reading the split CSVs

| Function | What it does |
| --- | --- |
| `read_split_csv(csv_path)` | reads one CSV and turns each relative `audio_path` into a full path |
| `random_subset(frame, size)` | keeps clips up to `SUBSET_MAX_SECONDS` and samples `size` of them with `SEED` |
| `load_split(csv_names, subset_size=None)` | joins one or more split CSVs into a Hugging Face `Dataset` with `audio_path`, `text`, `duration` |
| `print_split_sizes()` | what `python src/dataset.py` prints: files and hours per split |

The audio itself isn't loaded here; the collator does that. Keeping the `duration` column lets the
trainer batch clips of similar length together (`train_sampling_strategy="group_by_length"`); clips
run from 2 to 38 s, so random batches would be about half padding. Expect train 17,685 files
(77.45 h), validation 1,950 (9.28 h) and test 2,473 (9.89 h).

### 3. `vocab.py`: our own character vocabulary

| Function | What it does |
| --- | --- |
| `letters_in(csv_names)` | every character in those transcripts except space, sorted |
| `build_vocab(letters)` | ids for the letters, then `\|`, `[UNK]`, `[PAD]` |
| `save_vocab(vocab)` | writes `outputs/wav2vec2-lora-kaa/vocab.json` |
| `main()` | builds and saves the vocabulary, and checks validation/test use no other letters |

The result is 35 tokens: the 32 Karakalpak letters (`a … z`, `á ó ú ı ń ǵ`) with ids 0–31, then
`|` = 32 (word separator; the tokenizer turns spaces into it), `[UNK]` = 33 and `[PAD]` = 34, which
is also the CTC blank ("no new character in this frame").

### 4. `processor.py`: text ↔ ids and audio → model input

| Function | What it does |
| --- | --- |
| `build_tokenizer()` | `Wav2Vec2CTCTokenizer` from `vocab.json`; `bos_token=None` and `eos_token=None` keep the model at exactly 35 outputs |
| `build_feature_extractor()` | `Wav2Vec2FeatureExtractor`: 16 kHz, each clip normalised to zero mean and unit variance, with an attention mask marking padding |
| `build_processor()` | the two together as a `Wav2Vec2Processor` |
| `show_example()` | what `python src/processor.py` prints: a sentence encoded and decoded |

### 5. `collator.py`: building a batch

`DataCollatorCTC` turns a list of dataset rows into one batch:

| Method | What it does |
| --- | --- |
| `audio_batch(paths)` | reads the WAV files with `soundfile`, normalises them and pads to the longest clip |
| `label_batch(texts)` | encodes the transcripts, pads them, and sets padding to `IGNORED_LABEL_ID` (-100) so CTC skips it |
| `__call__(rows)` | the two together; the trainer calls this for every batch |

Because audio is read here, the 8 `NUM_WORKERS` processes load the next batches while the GPU trains
on the current one.

### 6. `model.py`: XLS-R + LoRA

| Function | What it does |
| --- | --- |
| `load_base_model(processor)` | XLS-R 300M with a new 35-output CTC head (`lm_head`); `ctc_zero_infinity=True` stops impossible alignments from giving infinite loss |
| `add_lora(model)` | rank-32 LoRA adapters on `LORA_TARGET_MODULES` (attention `q_proj k_proj v_proj out_proj`, feed-forward `intermediate_dense output_dense`) of all 24 layers; `lm_head` is trained in full (`TRAINED_IN_FULL`) |
| `build_model(processor)` | loads the base model, freezes the CNN feature encoder, adds LoRA, prints the parameter counts |
| `load_trained_model(model_dir=None)` | loads the saved `final_model/` and its processor (used by evaluation, decoding and benchmarks) |
| `gpu_supports_bf16()` | true on a GPU with bf16, so training and evaluation use 16-bit precision there |

When loading, the message lists `lm_head` as MISSING and pretraining-only layers as UNEXPECTED; both
are expected. `build_model` prints `trainable params: 14,191,651 || all params: 329,666,246 || trainable%: 4.3049`.

### 7. `metrics.py`: WER and CER

| Function | What it does |
| --- | --- |
| `argmax_logits(logits, labels)` | during evaluation keeps only the best token per frame, so all logits never sit in memory |
| `replace_ignored_labels(ids, pad_id)` | turns -100 padding back into the pad id before decoding |
| `decode(processor, pred_ids, label_ids)` | greedy CTC decoding (collapse repeats, drop blanks) of predictions, plain decoding of labels |
| `wer_and_cer(references, predictions)` | **WER** and **CER** with `jiwer`: (substitutions + deletions + insertions) / reference length |
| `make_compute_metrics(processor)` | the function the trainer calls after every evaluation |

### 8. `train.py`: training

| Function | What it does |
| --- | --- |
| `CTCTrainer._save(...)` | saves a checkpoint: LoRA weights + CTC head, processor, training arguments. `save_embedding_layers=False` stops PEFT from trying to save embedding layers wav2vec2 doesn't have (that crashed the first runs) |
| `training_arguments()` | epochs, batch, learning rate, bf16, grouping by length, evaluate and save every epoch, keep the best by WER |
| `latest_checkpoint()` | the newest checkpoint, if any, so an interrupted run resumes |
| `validation_history(log_history)` | the validation loss/WER/CER of every epoch from the trainer's log |
| `save_history(log_history)` | writes `history.csv` and prints the per-epoch table |
| `save_final_model(trainer, processor)` | merges LoRA into the weights and saves `final_model/` |
| `main()` | all of the above in order, then `evaluate_tests()` |

### 9. `evaluate_model.py`: test WER/CER (greedy)

| Function | What it does |
| --- | --- |
| `prediction_trainer(model, processor)` | a `Trainer` used only to predict |
| `evaluate_test_set(trainer, processor, name, csv_name)` | predicts one test set, writes `results/predictions_<set>.csv`, prints WER/CER and 3 examples |
| `write_predictions(...)`, `print_examples(...)` | small helpers for the above |
| `evaluate_tests(model_dir=None)` | all test sets in `TEST_CSVS`; writes `results/test_results.json` |

It runs automatically at the end of training; `python src/evaluate_model.py [path/to/final_model]`
runs it by hand.

### 10. `lm.py`: the KenLM language model

| Function | What it does |
| --- | --- |
| `lm_sentences()` | the train transcripts minus any sentence that is also in validation or test (none are) |
| `run_lmplz(text_path, arpa_path)` | runs `lmplz -o 4 --discount_fallback -S 1G`: a 4-gram with Kneser–Ney smoothing, memory capped at 1 GB so it can run next to training; writes to a `.partial` file and renames it only on success |
| `main()` | writes `lm/lm_text.txt` and builds `lm/kaa_4gram.arpa` |

### 11. `decode.py`: greedy vs beam search vs KenLM

| Function | What it does |
| --- | --- |
| `best_device()` | `cuda`, else `mps`, else `cpu` |
| `get_logits(model, processor, data)` | runs the model once over a set and keeps each file's logits without padding frames |
| `ctc_labels(processor)` | the 35 labels in model-output order for pyctcdecode, renamed by `PYCTCDECODE_LABELS` (`[PAD]` → blank, `\|` → space) |
| `greedy(processor, logits)` | best token per frame |
| `beam(decoder, logits)` | prefix beam search, width `BEAM_WIDTH`, in `DECODE_PROCESSES` processes |
| `decode_all_ways(...)` | greedy, beam and beam + KenLM for the same logits |
| `score(references, predictions)` | WER and CER, rounded |
| `tune(decoder, logits, references)` | tries every α in `LM_WEIGHT_CANDIDATES` × β in `WORD_BONUS_CANDIDATES` and keeps the lowest WER |
| `evaluation_sets()` | validation and both test sets |
| `write_decoded(...)`, `print_scores(...)` | output helpers, also used by `benchmark.py` |
| `main(model_dir=None)` | all of the above; writes `results/decoder_results.json` and `results/decoded_<set>.csv` |

With **shallow fusion**, each beam hypothesis is scored as
`log P_ctc(text | audio) + α · log P_KenLM(text) + β · words`: α weights the language model and β
rewards each word so the LM doesn't favour short outputs. α and β are tuned on the first 500
validation files only, so test scores stay unbiased.

### 12. `benchmark.py`: scoring other test sets

`python src/benchmark.py <test.csv> [name]` scores the trained model on any CSV with `audio_path`,
`text` and `duration` (paths relative to the CSV, or absolute), with all three decoders. It reuses α
and β from `results/decoder_results.json` (`tuned_alpha_and_beta()`), so nothing is tuned on the
benchmark. Results go to `results/benchmarks/<name>.json` and `<name>.csv`; the name defaults to the
CSV's folder and file name (`asr_corpus_test` for `data/benchmarks/asr_corpus/test.csv`). The test
sets found on the server are prepared by `scripts/prepare_benchmarks.py`; see `docs/datasets.md`.

## Results (server run, 2026-10-14)

Validation after each epoch, greedy decoding (1,950 STT clips):

| Epoch | WER | CER |
| ---: | ---: | ---: |
| 1 | 85.88% | 25.08% |
| 2 | 55.89% | 13.07% |
| 3 | 42.77% | 9.65% |
| 4 | 38.40% | 8.72% |

Decoders (WER / CER), α 0.7 and β 2.0:

| Set | Greedy | Beam | Beam + KenLM |
| --- | --- | --- | --- |
| STT test (1,950) | 38.37% / 8.90% | 37.75% / 8.73% | **18.61% / 5.93%** |
| HF test (523) | 22.32% / 3.65% | 21.68% / 3.53% | **18.30% / 3.08%** |
| asr_corpus test, new sentences (1,047) | 39.59% / 9.06% | 38.71% / 8.88% | **21.32% / 6.25%** |
| asr_corpus validation, new sentences (1,048) | 38.16% / 8.69% | 37.34% / 8.51% | **20.27% / 5.88%** |
| whisper_kaa test, new sentences (1,018) | 38.56% / 8.80% | 37.65% / 8.60% | **20.64% / 6.04%** |

On the whisper_kaa test, the colleague's fine-tuned Whisper-small scores 20.16% / 7.11% with the same
references. On the 508 HF test clips in `~/Benchmark`, the model saved there scores 32.76% / 6.89%
with KenLM, against 18.38% / 2.98% for ours.

## Quick local run on the Mac

The full settings don't fit the Mac's 24 GB. `LOCAL=1` makes `RUN_ON_MAC` true, which switches to
3,000 random train clips of at most 20 s and 300 clips per evaluation set, 5 epochs, batches of
4 × 4, gradient checkpointing and no loader processes (each would load its own PyTorch). Results go
to `outputs/wav2vec2-lora-kaa-local/`; the server never sets `LOCAL`.

```bash
cd ~/Documents/IT/repos/"AI & ML"
export LOCAL=1 HF_HUB_CACHE="$PWD/checkpoints/hf-cache/hub"
.venv/bin/python src/vocab.py
nohup .venv/bin/python src/train.py > outputs/wav2vec2-lora-kaa-local/train.log 2>&1 &
```

The local run reached HF test 74.57% / 15.82% and STT test 90.29% / 29.56% (300 clips each): it
confirms the pipeline works, not the model's quality.

## Run it on the server

**On the Mac, copy the code and docs to the server.**

```bash
cd ~/Documents/IT/repos/"AI & ML"
SERVER="intern@192.168.1.88:/home/intern/AI\ \&\ ML"
rsync -avc --exclude=__pycache__ --exclude=.DS_Store src/ "$SERVER/src/"
rsync -avc scripts/build_final_datasets.py scripts/validate_final_datasets.py scripts/prepare_benchmarks.py "$SERVER/scripts/"
rsync -avc docs/ "$SERVER/docs/"
```

**On the server, install the decoder packages (once).** `pyctcdecode` declares `numpy<2` but works
with numpy 2; `--no-deps` keeps numpy, torch and pandas untouched.

```bash
cd "/home/intern/AI & ML" && source .venv/bin/activate
pip install --no-deps pyctcdecode pygtrie
pip install kenlm
```

**On the server, KenLM's `lmplz`** is at `kenlm/build/bin/lmplz`. It was compiled against Boost 1.85
from the conda base environment, which no longer has it, so run `lm.py` with the `stt` environment's
libraries (below). If `lmplz` is ever missing, build it:

```bash
sudo apt-get install -y build-essential cmake libboost-program-options-dev libboost-system-dev libboost-thread-dev libboost-test-dev zlib1g-dev libbz2-dev liblzma-dev
mkdir -p kenlm/build && cd kenlm/build && cmake .. && make -j 8
```

**On the server, run the pipeline.**

```bash
cd "/home/intern/AI & ML" && source .venv/bin/activate
python src/vocab.py
nohup python src/train.py > train.log 2>&1 &
tail -f train.log
```

After training finishes (the log ends with the test WER/CER):

```bash
LD_LIBRARY_PATH=/home/intern/miniforge3/envs/stt/lib python src/lm.py
python src/decode.py 2>&1 | tee decode.log
python scripts/prepare_benchmarks.py
python src/benchmark.py data/benchmarks/asr_corpus/test.csv
```

## Where to see WER and CER

| When | Where |
| --- | --- |
| Every epoch, during training | `train.log` lines with `eval_wer` and `eval_cer` |
| After training | `outputs/wav2vec2-lora-kaa/history.csv` and `results/test_results.json` |
| After `decode.py` | `results/decoder_results.json` (all decoders, α, β) and `results/decoded_*.csv` |
| After `benchmark.py` | `results/benchmarks/<name>.json` and `<name>.csv` |

## What was tested

The pipeline ran end to end on the server: training, test scoring, KenLM, decoding and the
benchmarks above. On 2026-10-14 the code was then simplified: comments removed, names made
self-explanatory, long lines split into small functions, and a few settings renamed
(`TRAIN_FILES` → `TRAIN_SUBSET_SIZE`, `ALPHAS`/`BETAS` → `LM_WEIGHT_CANDIDATES`/`WORD_BONUS_CANDIDATES`,
`TUNE_FILES` → `TUNING_FILES`). The logic is unchanged, but the simplified version has only been
checked statically (syntax, pyflakes, every import and `config` setting exists), **not run yet**.
Paste any error from its first run into Claude Code.

## If something goes wrong

- **Out of GPU memory:** set `BATCH_SIZE = 4` and `GRADIENT_ACCUMULATION = 8` in `config.py`; the
  effective batch stays 32.
- **Restart training from scratch:** delete `outputs/wav2vec2-lora-kaa/checkpoints/`.
- **`libboost_program_options.so.1.85.0: cannot open shared object file`:** run `lm.py` with the
  `LD_LIBRARY_PATH` shown above.
- **`decode.py` is slow:** lower `BEAM_WIDTH` or `TUNING_FILES`, or raise `DECODE_PROCESSES` up to
  the CPU count (`nproc`).
- **`ModuleNotFoundError: pyctcdecode` / `kenlm`:** run the install step in the activated venv.
