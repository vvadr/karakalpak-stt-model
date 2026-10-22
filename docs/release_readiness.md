# Release readiness review

Reviewed on 2026-10-22 against local `main`, commit `89bdbf7`.

**Decision: the project is not ready for a production service. A public research release is a reasonable next milestone after the release blockers below are addressed.**

This review covers the active wav2vec2 pipeline, all tracked Python files, data preparation and validation tools, the unfinished alternative architecture, pipeline PDF documentation, dependency declarations, local datasets, local model artifacts, and saved results. KenLM is treated as an upstream dependency rather than a project implementation. The latest server weights were not available in this checkout, so the server model's quality and reload behavior could not be independently verified.

No Python code, configuration, weights, datasets, or existing result files were changed. The accompanying architecture and publication documents describe a proposed organization and release process.

## Readiness by deliverable

| Deliverable | Assessment | Main remaining work |
| --- | --- | --- |
| Educational code repository | Useful, with release blockers | Restore a working execution path, complete dependency records, distinguish active and unfinished work, document artifact downloads |
| Public datasets | Structurally validated; publication incomplete | Document upstream rights and sources, review flagged clips, prepare Hub export and dataset cards |
| Downloadable research model | Not verifiable for the advertised run yet | Retrieve the server model, matching KenLM, decoder settings and machine-readable predictions; verify them together |
| Public demonstration | Not implemented | Build a bounded inference interface around a verified release |
| Production API | Not ready | Implement transcription serving, deployment, input/resource limits, operational measurements, monitoring and rollback |

## Findings that block a release

### 1. Current training and batch evaluation crash on a missing setting

[`src/collator.py`](../src/collator.py), line 22, and [`src/metrics.py`](../src/metrics.py), line 12, use `config.IGNORED_LABEL_ID`. [`src/config.py`](../src/config.py) does not define it.

Both failures were reproduced with the current processor and a small input:

```text
collator AttributeError module 'config' has no attribute 'IGNORED_LABEL_ID'
metrics  AttributeError module 'config' has no attribute 'IGNORED_LABEL_ID'
```

This affects training, Trainer evaluation, and batch decoding/benchmarks because their collator also builds labels. Single-file transcription does not take that path, but its required server artifacts are absent locally. The walkthrough PDF's configuration snapshot includes `IGNORED_LABEL_ID = -100`, which also demonstrates drift between that document and current source. A source correction is required before these entry points can work; it was not applied because this review preserves the code.

### 2. The advertised model is missing locally

The README and cleaned logs describe the 2026-10-14 NVIDIA GB10 run: LoRA rank 32, four epochs, 17,685 training clips, and 77.45 training hours. Its expected artifacts are:

```text
outputs/wav2vec2-lora-kaa/final_model/
outputs/wav2vec2-lora-kaa/lm/kaa_4gram.arpa
outputs/wav2vec2-lora-kaa/results/decoder_results.json
outputs/wav2vec2-lora-kaa/results/decoded_stt_test.csv
outputs/wav2vec2-lora-kaa/results/decoded_hf_test.csv
```

All five are missing in this checkout. The configured output directory contains only `vocab.json`. External benchmark data and the matching result JSON/CSV files are also absent locally.

The recorded scores are therefore **reported server results**, not a newly reproduced evaluation:

| Test | Clips | Greedy WER / CER | KenLM WER / CER |
| --- | ---: | --- | --- |
| STT | 1,950 | 38.37% / 8.90% | 18.61% / 5.93% |
| HF | 523 | 22.32% / 3.65% | 18.30% / 3.08% |
| asr_corpus test | 1,047 | 39.59% / 9.06% | 21.32% / 6.25% |
| asr_corpus validation | 1,048 | 38.16% / 8.69% | 20.27% / 5.88% |
| whisper_test | 1,018 | 38.56% / 8.80% | 20.64% / 6.04% |

Evidence: [`decode.log`](../decode.log), [`benchmark.log`](../benchmark.log), and [`README.md`](../README.md). Do not label a different local checkpoint with these scores.

### 3. The service is a stub

[`src/main.py`](../src/main.py) defines only `GET /get`, returning `{}`. It has no transcription endpoint, model startup/loading, request schema, upload handling or readiness check. There is no project deployment definition, serving dependency set, or measured service capacity.

The CLI is a useful inference prototype. Production would additionally need controlled model concurrency, audio validation, size/duration limits, timeouts, queue limits, structured errors, pinned artifact loading, latency/memory measurements, and a defined rollback procedure. Actual load and failure behavior must be measured before choosing hardware or claiming availability.

### 4. Installation does not capture the complete pipeline

[`pyproject.toml`](../pyproject.toml) does not declare `peft`, `pyctcdecode`, `kenlm`, or `pygtrie`. The current FastAPI stub also imports an undeclared dependency; an ASGI server dependency is not declared. These runtime packages are absent from `uv.lock` as well. PEFT happens to be installed locally; pyctcdecode, KenLM, pygtrie, FastAPI and Uvicorn were not installed in the inspected environment.

The README relies on manual installations and a `--no-deps` workaround for pyctcdecode. This can be an experiment setup, but it does not establish a reproducible supported environment. The `ai-ml = "ai_ml:main"` console entry point has no corresponding implementation, and the README explicitly bypasses installation of the unfinished package.

## Other findings that matter

| Finding | Evidence | Consequence and required follow-up |
| --- | --- | --- |
| Trainer metrics decode padded frames | `src/metrics.py:7–18`, `src/evaluate_model.py:46–55`; README acknowledges this | Validation used for best-epoch selection is affected too. Use true output lengths consistently and reproduce scores after correction. `src/decode.py:35–37` already trims frames correctly. |
| Training silently resumes a shared output directory | `src/train.py:55–58,113–115` | A new experiment can inherit a previous run. Define immutable run IDs and explicit resume behavior in a future implementation. |
| Base model is not pinned in the active source | `src/config.py:17`, `src/model.py:18–22` | Record exact upstream revision, code commit, dataset revisions, configuration and package versions for the server release. |
| Inference depends on evaluation output | `src/benchmark.py:14–16`, `src/transcribe.py:29–36` | A downloaded acoustic model alone cannot reproduce advertised KenLM behavior. Package decoder settings and LM with the release. Inference should eventually consume a release configuration rather than require local tuning first. |
| Older LM artifacts can be confused with the current one | Root `kaa.arpa` declares only 1-, 2-, and 3-grams; README identifies it as legacy | It is not the server's 4-gram LM. Keep lineage explicit and publish the matching LM only. |
| No speaker-disjoint evidence | README and source metadata have no speaker IDs | Exact duplicate checks pass, but unseen-speaker performance remains unknown. Do not describe the splits as speaker independent. |
| STT-only validation | `src/config.py:8` | Best epoch and LM settings are selected on news speech. Literary-domain generalization has less validation support. |
| External benchmark provenance is incomplete locally | `scripts/prepare_benchmarks.py:17–18,110–120` uses server home directories; some CSV paths are absolute | Release source identifiers/revisions, exclusions, normalization decisions and portable manifests. Exact/near-duplicate and speaker overlap with training were not independently established here. |
| Benchmark and training text conventions differ | Normalization in `scripts/prepare_benchmarks.py`; README hyphen limitation | Publish the scoring rules and exclusions. Cross-model comparisons need matching references, normalization and recording sets. |
| Licensing and attribution are incomplete | No root `LICENSE` or `CITATION.cff`; final dataset cards contain no license/source URL | Choose a code license and document separate rights for each dataset, the base model and the LM's source text before public redistribution. |
| Review flags remain | `data/final/stt/review.csv`, `data/final/hf/review.csv` | 284 STT and 7 HF clips are flagged. These are review candidates, not 291 confirmed errors. Some repeated-number cases are valid arithmetic speech. Record listening decisions. |
| Documentation links are stale | `data/README.md` and `data/final/README.md` link to missing `docs/datasets.md`; older notes reference missing notebooks | Provide an accurate editable source of documentation and link current entry points. No active notebooks exist outside autosave folders. |
| Active code and unfinished exercises share one tree | `src/ai_ml/`, root placeholders, `scripts/ConnectionistTemporalClassification/ctc.py` | Keep a clear active-pipeline map. The unused template's SoundFile loader still asks for unsupported `float16`; the CTC exercise does not implement a finished loss. These should not be presented as supported production code. |

The current dependency's Transformers 5.19 implementation accepts `warmup_steps=0.1` as a ratio and supports `train_sampling_strategy="group_by_length"`. Those settings were checked against the installed implementation and are not reported as defects.

## What was verified successfully

### Dataset structure and source fidelity

Executed the existing validator, without rebuilding or editing data:

```bash
.venv/bin/python -B scripts/validate_final_datasets.py
```

Result: **37 PASS messages, 0 failed checks**. The validation covered all 22,108 final WAVs, manifest membership, sample rate/channels/PCM format, duration, recorded PCM hashes, source-byte fidelity, documented text transformations, source split assignments, changes logs, exact cross-split leakage and 100 random samples through the real dataset loader.

| Dataset | Train | Validation | Test | Total hours | Local bytes, approximately |
| --- | ---: | ---: | ---: | ---: | ---: |
| STT news | 15,595 | 1,950 | 1,950 | 93.132 | 10.754 GB |
| HF literary | 2,090 | — | 523 | 3.481 | 0.403 GB |
| Total | 17,685 | 1,950 | 2,473 | 96.613 | 11.157 GB |

These are file and consistency checks. They do not establish linguistic correctness, redistribution rights, speaker independence or the absence of acoustically similar excerpts. The text validator shares cleaning rules with the builder, so it is not an independent linguistic reference.

An additional manifest-based CTC alignment check found zero impossible alignments in all five splits, using the XLS-R convolution lengths and the minimum frames required for adjacent repeated labels. The current 35-token tokenizer also passed a Karakalpak-character round trip.

### Available model artifacts

| Local artifact | Verification | Relevant saved result |
| --- | --- | --- |
| `outputs/wav2vec2-lora-kaa-local/final_model/` | Offline reload and real 2.02-second audio inference succeeded; finite logits; 35-token head/tokenizer consistent | Recalculated from saved predictions: STT 90.29% WER / 29.56% CER on 300 clips; HF 74.57% / 15.82% on 300 clips |
| `checkpoints/wav2vec2-xlsr300m-lora-kaa-run02/model/` | Safetensors header/size and 39-token head/tokenizer consistent; saved predictions rescored | Recalculated: 55.44% WER / 12.14% CER on 927 older combined-corpus test clips |
| `outputs/wav2vec2-lora-kaa/final_model/` | Missing locally | README's server results cannot be independently reproduced here |

The old 39-token model was not reloaded for new inference during this review. Header checks are not a full weight-integrity guarantee. Neither old local run is a substitute for the advertised server model.

All 53 tracked Python files parsed successfully. Parsing does not establish functional correctness. No training, dependency installation, full new model evaluation, upload or deployment was performed.

## Recommended order

1. Retrieve the complete server run with matching model, processor, LM, decoder settings, raw evidence and artifact hashes.
2. Resolve source rights and dataset attribution; record decisions for flagged recordings.
3. Authorize and apply the known runtime corrections, then verify an installation from the released dependency set.
4. Reload the actual server export and reproduce decoder-specific scores and representative predictions; bind them to code/data/artifact revisions.
5. Publish an explicitly labelled research release with linked datasets, model card, training recipe and limitations.
6. Build and measure a public demo, then implement and qualify the production service against its intended use.

An 18–21% reported WER is useful research evidence, but the acceptable error rate depends on the product. WER counts word edits relative to reference length; it is not a direct percentage of completely correct sentences. Production acceptance should be based on representative users/audio and measured operational behavior.

See [project architecture](project_architecture.md) for the proposed flow and [publication guide](publication_guide.md) for upload destinations and exact release contents.
