# Publishing the code, datasets and models

Plan prepared on 2026-10-22. No external repositories were created and no uploads were performed.

For the implemented preparation helpers, current export paths, and copyable commands, follow
[upload_steps.md](upload_steps.md). This document explains the publication design and required evidence.

## Recommended destinations

| Material | Destination | Proposed name or scope |
| --- | --- | --- |
| Code, training recipe, editable documentation, small reports | Existing GitHub repository | `vvadr/karakalpak-stt-model` |
| STT news dataset | Hugging Face dataset repository | `YOUR_HF_NAME/karakalpak-stt-news` |
| Cleaned literary dataset derived from the existing HF source | Separate Hugging Face dataset repository | `YOUR_HF_NAME/karakalpak-literary-cleaned` |
| Merged acoustic model plus matching LM and decoder configuration | Hugging Face model repository | `YOUR_HF_NAME/wav2vec2-xls-r-300m-karakalpak` |
| Optional training adapter | Separate model repository or clearly versioned adapter area | Include LoRA, trained CTC head and exact base-model revision |
| Public interactive demo | Hugging Face Space | `YOUR_HF_NAME/karakalpak-asr-demo` |
| Stable academic citation | Hugging Face DOI for a finalized model/dataset; optionally Zenodo for GitHub release archiving | Mint only after the release names, authors and contents are final |

`YOUR_HF_NAME` is a placeholder; the GitHub username does not establish the Hugging Face account name. The proposed artifact repositories have not been checked for name availability.

Hugging Face provides dedicated [model repositories](https://huggingface.co/docs/hub/models), [dataset publishing](https://huggingface.co/docs/hub/datasets-adding), and [Spaces for demos](https://huggingface.co/docs/hub/spaces-overview). Model and dataset cards provide discoverability and the account of how the work was produced. Link every artifact back to the exact GitHub release and link the code back to all artifact pages.

## Dataset release contents

Publish the two corpora separately because their sources, domains and split policies differ. State that they jointly supplied 17,685 training clips / approximately 77.45 training hours. Their full train/validation/test collection is 22,108 recordings / approximately 96.61 hours.

The release candidates are `data/final/stt/` and `data/final/hf/`. The older combined/Mendeley datasets are separate experiments and were excluded from the latest documented server training run.

For each corpus preserve:

- Fixed train/validation/test membership and stable recording IDs.
- Audio and prepared transcript, duration and actual audio format.
- Original transcript and source identifiers/revisions.
- The normalization recipe and the original-to-prepared changes log.
- PCM hashes, quality measurements, review flags and reviewed decisions.
- Source attribution, redistribution terms, known collection/speaker limitations and a citation.

Keep provenance available to readers. If source-location fields contain machine-specific or unsuitable paths, publish stable source references and an explicit mapping instead.

### Prepare a Hub export rather than uploading the current folder verbatim

The current CSV schema uses `audio_path,text,duration`. Hugging Face AudioFolder metadata expects a `file_name` column to associate audio and transcripts. A raw folder upload needs an adapted metadata export and explicit split configuration, or users may see CSV rows without correctly linked playable audio. See the official [audio dataset format](https://huggingface.co/docs/hub/datasets-audio).

There is also a concrete directory-size issue: STT train has 15,595 WAVs in one `train/audio` folder. The Hub documents a maximum of 10,000 files in one folder. Use a separate publication export with embedded-audio Parquet shards, or subdivide audio directories and update only the exported metadata. Preserve the original training data and split identities. Current [Hub storage guidance](https://huggingface.co/docs/hub/storage-limits) recommends fewer/larger data files for large collections.

For this approximately 11.16 GB collection, a practical release is Parquet shards of roughly 250–500 MB, with explicit train/validation/test files and an audio feature supported by Datasets. This shard size is a project recommendation. Check account storage terms before uploading; free public storage is currently best effort, not a blanket unlimited-storage promise.

Example target layout:

```text
karakalpak-stt-news/
  README.md                   dataset card and split definitions
  LICENSE                     applicable documented terms
  data/
    train-00000-of-....parquet
    validation-00000-of-....parquet
    test-00000-of-....parquet
  provenance/
    changes.csv
    quality.csv
    review.csv
    review_decisions.csv      after listening review
    sources.json             actual upstream references and revisions
  checksums.sha256
```

The literary corpus has train/test only; do not invent a validation split and present it as the original split. Any later split change requires a new dataset version and clear evaluation consequences.

### Source rights and credits need completion

The literary source is [`kaa-ml/karakalpak-audio-dataset`](https://huggingface.co/datasets/kaa-ml/karakalpak-audio-dataset). Local provenance records snapshot `0f64c03be3a9b97fbdfc64934d2a81cddff174a8`. Its current public page labels the license MIT and gates file access behind accepted conditions. The gate's complete conditions were not available through the public page. Review the applicable source terms and preserve the required credits before publishing a derived copy.

The STT provenance currently identifies local Parquet shards, not a public upstream record or documented redistribution permission. Establish the actual collection source, transcript/audio ownership and release terms. Do not assign a blanket open license based solely on having the files locally. If redistribution is unavailable, publish the recipe and permitted metadata with instructions for obtaining the upstream data.

The base [`facebook/wav2vec2-xls-r-300m`](https://huggingface.co/facebook/wav2vec2-xls-r-300m) model currently declares Apache 2.0. Record the exact base revision and applicable notices. The project code, dataset recordings/transcripts, derived model and KenLM source text need their own documented provenance; the base model's license does not settle the dataset's terms.

### Dataset card

Write one [dataset card](https://huggingface.co/docs/hub/datasets-cards) per repository. Include Karakalpak (`kaa`), automatic speech recognition, creators and upstream credits, collection/transcription method, source revisions, audio format, split counts/hours, normalization, license, review status and limitations. State that speaker IDs are unavailable and that exact duplicate separation does not establish unseen-speaker evaluation.

There are currently 284 flagged STT clips and 7 flagged literary clips. Review flags are not proof of incorrect transcripts. Retain the audit trail and version any corrections rather than silently replacing the published test references.

## Model release contents

Retrieve the **server** run `outputs/wav2vec2-lora-kaa/` from the actual training machine. The matching assets are missing locally. The `-local` output and older 39-token checkpoints have different quality, vocabulary and training data.

Assemble a dedicated export with the following contents:

```text
wav2vec2-xls-r-300m-karakalpak/
  README.md                      model card, supported usage and decoder-specific results
  LICENSE                        applicable terms and notices
  model.safetensors              server final_model weights
  config.json                    matching acoustic model configuration
  vocab.json                     matching 35-token vocabulary
  tokenizer_config.json          matching tokenizer settings
  processor_config.json          matching processor settings in the current export
  ...                            any additional processor/tokenizer files in the server export
  lm/
    kaa_4gram.arpa                matching server LM, not root legacy kaa.arpa
  decoder_config.json            release settings, not machine-local paths
  evaluation/
    decoder_results.json
    decoded_stt_test.csv
    decoded_hf_test.csv
    benchmarks/                  permitted portable evidence for outside sets
  training/
    run_config.json
    environment.json
    provenance.json              source commit, base/data revisions and split hashes
    history.csv
  checksums.sha256
```

This is a proposed release schema. Some files above still need to be created from verified server evidence; copying the existing folder does not create them automatically. Include every processor/tokenizer file actually emitted by the server export, even when it is not listed above.

`decoder_config.json` should record the decoder type, beam width, alpha, beta, LM path relative to the release, vocabulary identity and normalization expectations. The reported server settings are beam width 100, alpha 0.7 and beta 2.0; verify them against the actual server result JSON.

Uploading only `model.safetensors` is incomplete. Uploading weights and tokenizer without KenLM supports the acoustic decoding mode but does not reproduce the approximately 18% WER claims. A model card should distinguish greedy, beam-only and KenLM scores, and its usage example must load the correct decoder for the stated result. Existing CLI code will still need a future change to read a portable bundle configuration.

The [model card](https://huggingface.co/docs/hub/model-cards) should explain: base model/revision, LoRA rank/targets, trained CTC head, merged export, dataset links/revisions, training command/configuration/hardware, best-epoch selection, normalization, decoder settings, results, limitations and intended uses. Document the existing padding-metric issue and the later verified scoring procedure. Publish a tested loading example against the released revision.

Full optimizer checkpoints are optional reproducibility archives, separate from the default inference download. An adapter-only publication must retain the trained `lm_head`; LoRA matrices alone are insufficient for the custom output vocabulary.

## Publication sequence

1. Complete the blockers in [release readiness](release_readiness.md), including rights documentation and retrieval of the actual server artifacts.
2. Prepare separate dataset/model export folders and cards; inspect their manifests and hashes before transfer.
3. Create the two dataset repositories and one model repository under the chosen HF account or organization. A private draft can be used while assembling the release.
4. Upload the prepared folders using the Hub's documented [folder upload tools](https://huggingface.co/docs/huggingface_hub/guides/upload). For model creation the UI starts at [New model](https://huggingface.co/new); the dataset publishing guide covers dataset repository creation.
5. Download each uploaded revision into a clean location. Confirm split counts, playable audio/text pairing, relative paths, artifact hashes, model reload, expected decoder behavior and reproduced metrics.
6. Finalize public visibility, cross-links and version tags, then publish the GitHub release with exact HF revisions and `CITATION.cff`.
7. Optionally request a [Hugging Face DOI](https://huggingface.co/docs/hub/doi) for the finalized model/dataset. DOI generation restricts later rename, deletion and visibility changes; finalize these choices first. [Zenodo's GitHub integration](https://help.zenodo.org/docs/github/) is an additional option for archiving the code release.
8. Add a demo after the verified inference release works. Check current [Spaces plans and compute requirements](https://huggingface.co/docs/hub/spaces-overview) when choosing hosting. Uploading artifacts does not itself provide a production transcription service.

Readers should be able to follow one continuous chain: GitHub release → dataset revisions and preparation method → training configuration and evidence → released model/LM → reproduced evaluation → demo. A Hugging Face collection can group the related public repositories once they exist.
