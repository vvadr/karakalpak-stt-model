---
language:
- kaa
license: apache-2.0
library_name: transformers
pipeline_tag: automatic-speech-recognition
base_model: facebook/wav2vec2-xls-r-300m
base_model_relation: finetune
datasets:
- iamabdulaziz/karakalpak-stt-news
- iamabdulaziz/karakalpak-literary-cleaned
metrics:
- wer
- cer
tags:
- wav2vec2
- xls-r
- lora
- ctc
- kenlm
- karakalpak
---

# Karakalpak wav2vec2 XLS-R 300M + KenLM

**Karakalpak speech recognition with a completed LoRA fine-tune of Meta's wav2vec2 XLS-R 300M and a matching KenLM 4-gram decoder.** The inference model is merged; PEFT is not required for ordinary inference.

Maintainer: [Abdulaziz Yusupaliev](https://huggingface.co/iamabdulaziz). [Code, training recipe and reports](https://github.com/vvadr/karakalpak-stt-model).

## Results

The table reports **WER / CER**. These values were independently recomputed from the saved prediction files during publication; the original acoustic evaluation was not rerun in full and no new training was performed.

| Test | Clips | Greedy | Beam | Beam + KenLM |
| --- | ---: | --- | --- | --- |
| STT news | 1,950 | 38.37% / 8.90% | 37.75% / 8.73% | **18.61% / 5.93%** |
| HF literary | 523 | 22.32% / 3.65% | 21.68% / 3.53% | **18.30% / 3.08%** |
| asr_corpus test | 1,047 | 39.59% / 9.06% | 38.71% / 8.88% | **21.32% / 6.25%** |
| asr_corpus validation | 1,048 | 38.16% / 8.69% | 37.34% / 8.51% | **20.27% / 5.88%** |
| whisper_test | 1,018 | 38.56% / 8.80% | 37.65% / 8.60% | **20.64% / 6.04%** |

The approximately 18% results require **KenLM**, beam width 100, alpha 0.7 and beta 2.0. A default Transformers ASR/argmax call uses greedy decoding and corresponds to the greedy column. Decoder settings were tuned on 500 STT validation clips.

## Use the complete decoder

Clone the code and install the separate, verified inference environment:

```bash
git clone https://github.com/vvadr/karakalpak-stt-model.git
cd karakalpak-stt-model
uv venv .venv-inference --python 3.12
uv pip install --python .venv-inference/bin/python -r requirements-inference.txt
.venv-inference/bin/python scripts/release/transcribe_huggingface.py clip.wav --decoder kenlm
```

The command downloads only the inference model/processor and LM assets, not optimizer checkpoints. Input must be 16 kHz audio; stereo is averaged to mono. Use `--decoder greedy` for greedy output, `--device cpu` for CPU, `--revision COMMIT_SHA` to pin a published model revision, or `--model /local/bundle` for an offline bundle.

The standalone `transcribe.py` in this repository is the same inference client. It reads `decoder_config.json`, loads the model and the matching LM, and preserves tokenizer label ordering. The separate dependency set uses NumPy 1.26 because pyctcdecode 0.5.0 declares NumPy <2; it leaves the training/notebook environment unchanged.

An anonymous Hub download and CPU reload with KenLM were checked on the included 2.02-second literary example and returned the reference exactly: `azıraqtan soń bılay dedi`. The audio and reference are under `examples/`; this is a spot check, not a new full-test evaluation.

## Model and training

- Base: `facebook/wav2vec2-xls-r-300m`; custom 35-token character CTC output, blank/pad ID 34.
- LoRA: rank 32, alpha 64, dropout 0.05, targeting attention q/k/v/out and feed-forward dense projections; `lm_head` trained fully.
- Four epochs / 2,212 optimizer updates, batch size 8 × gradient accumulation 4, learning rate 5e-4, 10% warmup, seed 42.
- Training: 17,685 recordings / approximately 77.45 hours from the two linked dataset train splits. Validation: 1,950 STT news clips.
- Hardware: NVIDIA GB10; recorded duration approximately 2 hours 31 minutes on 2026-10-14. Selected checkpoint: epoch 4, step 2,212.
- Export: LoRA merged into the acoustic model. Matching processor, tokenizer, LM, vocabulary and decoder configuration are included.

`training/run_config.json`, `history.csv`, environment/provenance records, and the retained resumable checkpoints document the run. `training/checkpoints/` contains both saved checkpoints with adapter/head weights, optimizer/scheduler/RNG state, processor and trainer state. The default inference client deliberately does not download these training assets.

The base revision observed in the server cache is `1a640f32ac3e39899438a2931f9924c02f080a54`. The original training code did not pin or record its base revision, and the exact pre-refactor training source commit was not captured. The available post-run server source snapshot is identified separately; it is not mislabelled as a proven original training commit.

## Evidence and integrity

- `evaluation/`: predictions for validation and both test sets, recorded decoder results, recomputed metrics and external benchmark evidence.
- `lm/`: the exact matching 4-gram ARPA model and its training transcript text.
- `training/`: selected-run settings, history, provenance, environment and resumable checkpoints.
- `checksums.sha256`: hashes of the released files. Model and LM SHA256 values matched independently between the server and local transfer.

## Intended use and limitations

This is a research ASR release for Karakalpak Latin transcripts. It is suitable for experimentation and assisted transcription with review. Speaker IDs are unknown, so unseen-speaker performance is not established. Validation and decoder tuning are news-domain only; normalization conventions and joined hyphenated words affect cross-corpus WER. The model does not restore punctuation or guarantee verbatim transcription for arbitrary domains, noise, accents or long recordings.

The archived training code has known resume/configuration and padded-metric issues described in the GitHub readiness report. The standalone inference client avoids that training path. A production API, service capacity and operational guarantees are outside this release.

## License and credit

Base model and derived model weights are distributed with the original Apache 2.0 license and attribution in `LICENSE` and `NOTICE`. Credit Meta / the XLS-R authors and the original dataset contributors. The literary source's pinned card declares MIT; the supplied STT source did not record a license or original publisher, as explained in its dataset card. The model license does not assign a new license to dataset recordings or transcripts.

Cite the base [XLS-R paper](https://arxiv.org/abs/2111.09296), the project repository, the original literary source and the exact model/dataset revisions used.
