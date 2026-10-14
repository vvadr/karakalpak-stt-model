# Datasets

Train on `data/final/`. It holds two separate, train-ready datasets with matching WAVs per split.
They are kept apart because their domain, clip length and splits differ. Every transcript uses
only the current (2016) Karakalpak Latin alphabet.

| Dataset | Train | Validation | Test | Hours | Clips | Changed transcripts |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `final/stt/` | 15,595 (74.68 h) | 1,950 (9.28 h) | 1,950 (9.18 h) | 93.13 | 5.9–38.5 s, news | 2,115 |
| `final/hf/` | 2,090 (2.77 h) | – | 523 (0.71 h) | 3.48 | 2.0–11.6 s, literary | 1 |

Rebuild with `python scripts/build_final_datasets.py --overwrite` (≈30 s) and check with
`python scripts/validate_final_datasets.py` (≈1 min, 37 checks, all passing on 2026-10-14).
`data/final` holds 22,127 files: 22,108 WAVs plus 19 CSV, JSON and README files.

## Folder map

| Folder | Status | Use |
| --- | --- | --- |
| `datasets/STT_NORMALIZED_FINAL/` | original, final STT version | source of `final/stt` |
| `datasets/STT_SPLITS/` | original | STT train/validation/test assignment |
| `datasets/TOZALANGAN_TRAIN_FINAL/` | original, superseded | identical audio; not needed |
| `karakalpak-combined/` | original | HF part is the source of `final/hf`; Mendeley part unused |
| `karakalpak-audio-dataset-hf/`, `karakalpak-mendeley/` | original | per-source exports (see `data/README.md`) |
| `final/` | **derived, train on this** | `stt/`, `hf/` |

Nothing under `datasets/` or `karakalpak-combined/` is edited by the build.

## STT corpus (`data/datasets/`)

19,495 recordings, 93.13 h, news and official-speech domain. Audio is stored inside the
parquet `audio` column as complete WAV files: mono, 16 kHz, PCM16. Clips run 5.9–38.5 s
(mean 17.2 s); 198 are longer than 30 s. IDs range 0–19553 (59 were dropped upstream).

**`TOZALANGAN_TRAIN_FINAL`** ("cleaned"; 20 shards, ZSTD): `id`, `audio`, `cleaned_text`,
`cleaned_text_lowercase`.

**`STT_NORMALIZED_FINAL`** (20 shards, Snappy): the same four columns plus `text_normalized`.
The validator confirms its audio and `cleaned_text` match TOZALANGAN byte for byte for all
19,495 ids. The two folders differ only in compression and the extra column, so
STT_NORMALIZED_FINAL is the final version and TOZALANGAN can be ignored.

The STT text columns, from raw to final:

- `cleaned_text`: cased, with punctuation (`Isbilermenlik subektleri ushın …, … qısqartıldı.`).
- `cleaned_text_lowercase`: lowercase, no punctuation or digits; numbers are already words.
- `text_normalized`: lowercase text with foreign letters folded. It differs from
  `cleaned_text_lowercase` in 1,890 rows, mostly `jüz → júz` (2,776×), plus `é → e`,
  `ε → epsilon`, `æ → ae`, `ö → ó`, `š`/`ş → sh`, `ñ → ń`, and `ʼ` removed. This is the
  source transcript for `final/stt`.

**`STT_SPLITS`**: `id,text_normalized` for train 15,595 / validation 1,950 / test 1,950.
These ids cover the parquet set exactly with no overlap, and each `text_normalized` equals
its parquet row. No audio hash or transcript crosses splits. Five transcripts were recorded
twice by different takes; both takes are always in the same split.

## karakalpak-combined

4,635 recordings, 8.55 h: HF (2,613; 2.0–11.6 s, literary) and Mendeley (2,022; 4.1–19.9 s).
It has a train/test split only: seed 42, 80/20, grouped by PCM hash and transcript, with the
published HF test set kept in test. Its `README.md` and `provenance/metadata.csv` give the
details. HF shares no audio hash and no transcript with the STT corpus. `final/hf` keeps each
HF recording's train/test assignment.

**Mendeley is excluded.** Its transcripts write `ś`/`ć` for `sh`/`ch`, which is not part of the
Karakalpak alphabet. About half of its rows also follow Kazakh spelling: passages from
*Abai joly*, `qazaq tilinde …`, and 13 rows with Cyrillic `һ`. The original Mendeley data is
still in `karakalpak-mendeley/` and `karakalpak-combined/`.

## Alphabet

The current Karakalpak Latin alphabet (2016 version, per
[Wikipedia](https://en.wikipedia.org/wiki/Karakalpak_language)) is
`A Á B D E F G Ǵ H X Í I J K Q L M N Ń O Ó P R S T U Ú V W Y Z Sh C Ch`. In lowercase,
`Í` is `ı`, so transcripts may contain only:

```
a á b c d e f g ǵ h i ı j k l m n ń o ó p q r s t u ú v w x y z   (sh, ch as digraphs)
```

STT and HF already used this alphabet except for `í`, which appeared where a capital `Í` was
lowercased with Unicode rules (`Íssıxana → íssıxana`). The build changes `í → ı` in 17 STT rows
and 1 HF row; each dataset's `changes.csv` lists them with the original text. The build refuses
any character outside the alphabet, and the validator checks every row.

An earlier build converted `sh → ś` and `ch → ć` to match Mendeley. That has been reversed.

## Number-word repairs

The STT text has no digits: an upstream step wrote numbers out as words. That step left
leftovers, which the build repairs in 2,115 STT rows (1,689 train, 209 validation, 217 test):

| Repair | Rows | Example |
| --- | ---: | --- |
| Re-attach a suffix split off a number (the dative gets the right vowel and voicing), or `ǵı`/`gi` split off a word | 799 | `júz den` → `júzden`, `toǵız júz ǵa` → `toǵız júzge`, `úsh ge` → `úshke`, `haqqında ǵı` → `haqqındaǵı` |
| Split an abbreviation glued to a number | 28 | `covidon toǵız` → `covid on toǵız`, `topseksen ge` → `top seksenge` |
| One spelling for 20 and 50 (as in the hand-written HF text) | 1,473 | `jıyırma` → `jigirma`, `elli` → `eliw`, `mın` → `mıń` |

Both spellings of 40 (`qırıq`, `qırq`) are kept: the HF text uses both. Whole words only are
respelled, so `ellikqala` (a district) and `qırqıp` ("cutting") are untouched. Each change is in
`changes.csv` with the rule that made it.

`review.csv` lists 284 STT rows left unchanged because only listening can decide them:

- 55 rows where a number is said twice in a row (`on on eki jas`); some are real, such as the
  tennis score `altı bir altı bir`.
- 229 rows with a standalone suffix after a word or abbreviation (`ipo ǵa`, `aqsh tan`). These
  aren't re-attached because similar tokens are often names or other languages: `si czinpin`
  (Xi Jinping), `made in uzbekistan`, `van den bosh`.

The original text with digits is not in our data, so the conversion of each number to its value
(e.g. that `2026` became `eki mıń jigirma altı`) cannot be checked here.

## final/ layout

```
final/
  README.md  summary.json                       totals and rules for both datasets
  stt/  hf/
    <split>/audio/*.wav                         stt_000123.wav, hf_audio_000001.wav
    <split>.csv                                 audio_path, text, duration   (stt: train/validation/test;
                                                                              hf: train/test)
    metadata.csv                                + split, source_id, source_location, text_original,
                                                  text_rules, pcm_sha256
    changes.csv                                 edited transcripts with originals
    provenance/quality.csv                      peak/RMS dBFS, clipped fraction, chars per second
    split_summary.json  README.md
```

`audio_path` resolves relative to its dataset folder (`final/stt/` or `final/hf/`). Every WAV is
byte-identical to its source, and `source_location` points back to it (a parquet shard and id,
or a `karakalpak-combined/…` path). No audio or transcript crosses splits, even across the two
datasets, so their CSVs can be concatenated safely.

### Training with it

`src/` trains wav2vec2 XLS-R 300M with LoRA directly on `final/stt` and `final/hf` and reports
WER/CER for greedy, beam search and beam search + KenLM decoding. See
`docs/wav2vec2_training.md`. The vocabulary is the 32 letters plus `|`, `[UNK]` and `[PAD]`
(35 tokens). Validation (`stt/validation.csv`) picks the best epoch; the two test sets are scored
once at the end, separately.

## Caveats

- **Long clips.** 198 STT clips exceed 30 s (161 train). This is fine for wav2vec2 CTC
  (memory permitting), but Whisper needs them filtered or cut.
- **Imbalance.** STT is 96% of the hours, so report WER per dataset.
- **Quiet audio.** 177 STT clips have RMS below −45 dBFS, and 3 STT clips clip more than
  0.1% of samples. HF has neither problem. Speaking rate is plausible throughout:
  4.8–23.9 characters/s in STT and 5.7–26.0 in HF.
- **Joined hyphenated words.** Upstream normalization dropped hyphens without a space in STT
  (`sociallıqekonomikalıq`), while HF splits them (`az aq`). This costs some WER; it was left
  as-is.
- **No speaker IDs** in either source, so splits are not speaker-disjoint.
