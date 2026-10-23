# Upload steps: GitHub, datasets, model

The code and publication helpers are published in [vvadr/karakalpak-stt-model](https://github.com/vvadr/karakalpak-stt-model). Dataset and model uploads are separate steps. The ASR implementation under `src/` was preserved during publication setup.

## Published artifacts

The 2026-10-23 research release is available at:

- [STT news dataset](https://huggingface.co/datasets/iamabdulaziz/karakalpak-stt-news).
- [Literary dataset](https://huggingface.co/datasets/iamabdulaziz/karakalpak-literary-cleaned).
- [Model, KenLM and training checkpoints](https://huggingface.co/iamabdulaziz/wav2vec2-xls-r-300m-karakalpak).

Use the README's standalone inference quick start to transcribe files. The remaining steps document how maintainers prepare and upload a later version; the initial release is already published.

## 1. Get the published code

On a new machine, clone the repository:

```bash
git clone --recurse-submodules https://github.com/vvadr/karakalpak-stt-model.git
cd karakalpak-stt-model
```

On the current Mac, continue from the existing checkout instead:

```bash
cd '/Users/macbook/Documents/IT/repos/AI & ML'
python3 -B scripts/release/check_github.py
```

The existing root `kaa.arpa`, `lm_text.txt` and `vocab.json` remain tracked as small historical experiment assets. They are not the newer server model's matching artifacts. Old Git history is preserved.

GitHub receives Python source, build/validation/export recipes, dependency files, the PDF walkthrough, documentation, the three small curated result logs, and the historical root `kaa.arpa`, `lm_text.txt` and `vocab.json`. KenLM stays a pinned Git submodule; people cloning the training code can use `git clone --recurse-submodules`.

GitHub excludes:

- `data/`, `outputs/`, `checkpoints/`, and `publication/`.
- Audio/video, Parquet, weights, generated language models, and archives.
- Environments, credentials, caches, notebook autosaves and raw logs.

This prepares the research repository for publication. The functional production blockers in [release_readiness.md](release_readiness.md) remain. A code license still needs to be chosen if you want readers to have explicit reuse terms; no license was invented by this setup.

## 2. Locate the prepared dataset folders

Both final corpora were exported locally as Parquet with embedded WAV bytes. Upload the prepared folders below, including their contents, to the root of separate Hugging Face dataset repositories:

| Source data | Upload this prepared folder | Hugging Face repository name |
| --- | --- | --- |
| `data/final/stt/` | `publication/datasets/karakalpak-stt-news/` | `iamabdulaziz/karakalpak-stt-news` |
| `data/final/hf/` | `publication/datasets/karakalpak-literary-cleaned/` | `iamabdulaziz/karakalpak-literary-cleaned` |

Each prepared folder contains:

```text
README.md                 editable draft dataset card and explicit split mapping
data/*.parquet            embedded audio, transcripts, duration, IDs and provenance fields
provenance/               original manifests, edits, quality measurements and review flags
export_manifest.json      exporter revision/hash, source manifest hashes and split counts
checksums.sha256          exported file hashes
```

These export folders are ignored by Git. Original source audio and CSV files are untouched.

For a fresh copy of the project, generate them with:

```bash
uv sync --no-install-project
.venv/bin/python -B scripts/release/prepare_huggingface.py dataset stt
.venv/bin/python -B scripts/release/prepare_huggingface.py dataset hf
```

Skip those two export commands on this Mac because the folders already exist. Re-running refuses to overwrite an existing export. For a later version, use a new destination such as `--output publication/datasets/karakalpak-stt-news-v2` and upload that version's folder.

## 3. Finish the dataset cards and source terms

For later versions, update each prepared folder's card with actual authors/upstream credits, source revisions, terms, review decisions, citation and intended use. The initial release preserves the original literary MIT card; the STT card explicitly records that the supplied source did not specify a publisher or license. No unverified open license is assigned. Versioned copies of the completed release cards are under `docs/huggingface/`.

The export helper creates editable drafts for new runs. Complete their factual provenance before replacing the public release, and do not silently assign a license absent from the source material. Link the exact GitHub/model/dataset revisions used.

After editing a card or adding license/provenance files, refresh and verify its checksums:

```bash
python3 -B scripts/release/prepare_huggingface.py checksums publication/datasets/karakalpak-stt-news
python3 -B scripts/release/prepare_huggingface.py verify publication/datasets/karakalpak-stt-news
python3 -B scripts/release/prepare_huggingface.py checksums publication/datasets/karakalpak-literary-cleaned
python3 -B scripts/release/prepare_huggingface.py verify publication/datasets/karakalpak-literary-cleaned
```

## 4. Upload the datasets to Hugging Face

Create a Hugging Face account, obtain a token with write access to your repositories, and log in interactively. Replace `your-hf-username` with your actual HF account or organization name:

```bash
uvx hf auth login
uvx hf auth whoami
HF_NAMESPACE='your-hf-username'
uvx hf repos create "$HF_NAMESPACE/karakalpak-stt-news" --repo-type dataset --private
uvx hf repos create "$HF_NAMESPACE/karakalpak-literary-cleaned" --repo-type dataset --private
uvx hf upload "$HF_NAMESPACE/karakalpak-stt-news" publication/datasets/karakalpak-stt-news . --repo-type dataset
uvx hf upload "$HF_NAMESPACE/karakalpak-literary-cleaned" publication/datasets/karakalpak-literary-cleaned . --repo-type dataset
```

The final `.` uploads each folder's contents to the repository root. Do not upload only a CSV: these Parquet shards contain the actual audio. The commands use the current [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli). Run `uvx hf upload --help` if your installed interface differs.

Open the resulting dataset pages while logged in. Check:

- STT: train **15,595**, validation **1,950**, test **1,950**.
- Literary: train **2,090**, test **523**; no validation split.
- Audio plays and transcripts, source fields and split selection appear correctly.

For a download-integrity check, download one repository to a fresh local folder and verify it, then repeat for the other:

```bash
uvx hf download "$HF_NAMESPACE/karakalpak-literary-cleaned" --repo-type dataset --local-dir publication/download-check/literary
python3 -B scripts/release/prepare_huggingface.py verify publication/download-check/literary
```

When cards, terms and content checks are complete, change each repository's visibility to **Public** in its Settings. Its URL will be `https://huggingface.co/datasets/iamabdulaziz/REPOSITORY_NAME`.

## 5. Prepare the actual server model

The actual server run was retrieved and published on 2026-10-23. The older `-local` output and 39-token checkpoints remain different experiments. For a future run, prepare the actual matching run rather than relabelling older weights.

On the machine holding the full training run, use the updated project and run:

```bash
python3 -B scripts/release/prepare_huggingface.py model --run-dir '/absolute/path/to/training-project/outputs/wav2vec2-lora-kaa'
```

Replace the absolute path with the real server path. If the helper is not on that server yet, clone the published code into a **new** directory and run it from there, pointing `--run-dir` at the existing training output:

```bash
git clone --branch main https://github.com/vvadr/karakalpak-stt-model.git kaa-publication-tools
cd kaa-publication-tools
python3 -B scripts/release/prepare_huggingface.py model --run-dir '/absolute/path/to/training-project/outputs/wav2vec2-lora-kaa'
```

The model helper uses Python's standard library; it does not retrain or require the ASR runtime. It stops before creating an export if essential assets are missing. It expects:

```text
outputs/wav2vec2-lora-kaa/
  final_model/model.safetensors
  final_model/config.json
  final_model/vocab.json
  final_model/tokenizer_config.json
  final_model/processor_config.json OR preprocessor_config.json
  lm/kaa_4gram.arpa
  results/decoder_results.json
  results/decoded_stt_test.csv
  results/decoded_hf_test.csv
  history.csv
```

The prepared model upload folder is:

```text
publication/models/wav2vec2-xls-r-300m-karakalpak/
  model.safetensors + matching model/processor/tokenizer JSON files
  lm/kaa_4gram.arpa
  decoder_config.json
  evaluation/decoder_results.json + test predictions
  training/history.csv
  README.md
  export_manifest.json
  checksums.sha256
```

The helper rewrites exported evaluation audio paths into portable dataset-relative paths and records decoder settings without changing reference/prediction text. The initial published release also includes the two original resumable checkpoints under `training/checkpoints/`, separate from the root inference assets. The standalone client excludes those checkpoints from normal inference downloads.

Complete the model card, licensing/notices, exact training provenance and dataset revision links. Set the actual training commit in the manifest once established. Check a reload and evaluation of this actual model before presenting the reported scores as reproduced. The helper assembles files; it does not certify accuracy or implement a new inference loader.

## 6. Upload the model

Run this block on the machine where the prepared model folder exists. Log in there if needed and set your actual account name:

```bash
HF_NAMESPACE='your-hf-username'
python3 -B scripts/release/prepare_huggingface.py checksums publication/models/wav2vec2-xls-r-300m-karakalpak
python3 -B scripts/release/prepare_huggingface.py verify publication/models/wav2vec2-xls-r-300m-karakalpak
uvx hf auth login
uvx hf repos create "$HF_NAMESPACE/wav2vec2-xls-r-300m-karakalpak" --repo-type model --private
uvx hf upload "$HF_NAMESPACE/wav2vec2-xls-r-300m-karakalpak" publication/models/wav2vec2-xls-r-300m-karakalpak . --repo-type model
```

After downloading/reloading the uploaded revision and completing the model card, make the model repository public. Its URL will be `https://huggingface.co/iamabdulaziz/wav2vec2-xls-r-300m-karakalpak`.

Uploading only weights is incomplete. The approximately 18% reported WER requires the matching KenLM and decoder settings. A plain Transformers greedy call has different reported results. The new `scripts/release/transcribe_huggingface.py` command loads the published bundle and decoder directly. The archived `src/transcribe.py` retains its original local paths.

## 7. Link the release

Add all three public HF URLs and exact revisions to the GitHub README. Link the model card to both dataset revisions and the GitHub release. Add dataset/model cards' creators and citation details. A public Hugging Face collection can group the three repositories.

An interactive demo is a later, separate Hugging Face Space. No Space or production transcription service is created by uploading these files. For a stable academic citation, finalize repository names/content before requesting a [Hugging Face DOI](https://huggingface.co/docs/hub/doi).

## Local preparation checks

The GitHub candidate passes the artifact guard, including the three intentionally retained historical root assets. The STT draft has 41 shards, approximately 9.03 GB; the literary draft has 3 shards, approximately 0.35 GB. Every exported row's ID, transcript and duration was compared with its source split. Datasets recognized all five splits and their Audio features; an embedded audio sample from each split matched the original file bytes. All 58 checksummed export files passed SHA256 verification.

The completed server model export was subsequently verified against server hashes. Its saved prediction metrics were recomputed, and the standalone client successfully reloaded the full acoustic model/KenLM for a real CPU spot check. See the readiness report's 2026-10-23 follow-up and the model card for the precise evidence and limitations.
