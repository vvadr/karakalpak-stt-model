# Karakalpak ASR: remote training runbook

2026-10-15

## Situation

The training data is built on the Mac. Training runs on the server. Since 2026-10-14 the Mac's SSH
key is installed on the server, so Claude Code can also run these commands itself.

| Machine | Address | Repo path | Role |
| --- | --- | --- | --- |
| Mac (local) | – | `~/Documents/IT/repos/AI & ML` (branch `model-train-2`, older local work) | Claude Code runs here; builds `data/final`; pushes to GitHub |
| Server `spark-f886` | `intern@192.168.1.88` (SSH key) | `/home/intern/AI & ML` (`.venv`, Python 3.12.3, branch `lora-training-clean`) | GPU training; the up-to-date code |

The server's code is on GitHub in `main` (and `lora-training`). The server's own SSH key is not
on GitHub, so its commits reach GitHub through the Mac (Step 5).

How the data moved:

1. The STT parquet set (`data/datasets`, 19 GB) was copied from the server to the Mac.
2. On the Mac, `scripts/build_final_datasets.py` turned it plus the HF part of
   `data/karakalpak-combined` into `data/final`. That is 22,127 files: 22,108 WAVs plus 19 CSV,
   JSON and README files. `scripts/validate_final_datasets.py` passed all 37 checks.
3. `data/final` was copied to the server with `rsync -c` (Step 1) and checked there (Step 2).
4. The model was trained on the server (2 h 31 min), then decoded and benchmarked (Steps 3 and 4).
   The cleaned logs of that run are `train.log`, `decode.log` and `benchmark.log` in the repo root.

Run each block from the repo root on the machine its heading names. Every block is safe to run
again.

## Data

`data/final` holds two separate datasets. They are kept apart because their domain, clip
length and splits differ. All audio is mono 16 kHz PCM16 WAV, byte-identical to its source.

| Dataset | Train | Validation | Test | Hours | Clip length | Source |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `final/stt` | 15,595 | 1,950 | 1,950 | 93.13 | 5.9–38.5 s, news | `datasets/STT_NORMALIZED_FINAL` + `STT_SPLITS` |
| `final/hf` | 2,090 | – | 523 | 3.48 | 2.0–11.6 s, literary | HF part of `karakalpak-combined` |

Each dataset folder contains:

- `<split>/audio/*.wav`
- `<split>.csv` with `audio_path,text,duration`; paths are relative to the dataset folder
- `metadata.csv`, which adds source location, original text and PCM SHA256
- `changes.csv`, listing every edited transcript next to its original

**Alphabet.** Transcripts use only the current (2016) Karakalpak Latin alphabet:
`a á b c d e f g ǵ h i ı j k l m n ń o ó p q r s t u ú v w x y z`, with `sh` and `ch` as
digraphs. Edits: `í → ı` (`Í` is the capital of `ı`) and repairs of number words left by the
upstream digit-to-word step (`júz den` → `júzden`, `covidon` → `covid on`, `elli` → `eliw`); see
`docs/datasets.md`. Mendeley is excluded because it writes `ś`/`ć` and much of
it follows Kazakh spelling.

No audio or transcript appears in more than one split, even across the two datasets. The full
description is in `docs/datasets.md`.

## Step 1 (Mac): overwrite the server copy

One `rsync` command makes the server's `data/final` an exact copy of the Mac's. Run it in a Mac
terminal, not the SSH one.

- **`-c` (checksum):** compares file contents instead of sizes or dates. This is required
  because `ś` and `sh` take the same number of bytes, so the old and new CSVs are the same
  size. Only missing or different files are sent. Most WAVs are already on the server and
  unchanged, so they are skipped.
- **`--delete-after`:** removes files that no longer exist on the Mac, such as `mendeley/`
  and any `manifests/` from an earlier Step 3, once the transfer has finished.

Do a dry run first (`-n`). It lists what would be sent or deleted and changes nothing:

```bash
cd ~/Documents/IT/repos/"AI & ML"
rsync -avcn --delete-after "data/final/" "intern@192.168.1.88:/home/intern/AI\ \&\ ML/data/final/"
```

Expect `deleting mendeley/…` lines, the CSV/JSON/README files, and any WAVs the interrupted
`scp` missed. If that looks right, run it for real, which is the same command without `n`:

```bash
rsync -avc --delete-after "data/final/" "intern@192.168.1.88:/home/intern/AI\ \&\ ML/data/final/"
```

The Mac's built-in `rsync` passes the remote path through the server's shell, which is why
`AI\ \&\ ML` is escaped. If it fails with `rsync: command not found`, run
`sudo apt install -y rsync` in the SSH terminal and try again. Checksumming reads all 10 GB on
both machines, so expect a minute or two before files start moving.

Quick count check. Both commands should print **22127**:

```bash
find data/final -type f ! -name .DS_Store | wc -l
ssh intern@192.168.1.88 'find "/home/intern/AI & ML/data/final" -type f ! -name .DS_Store | wc -l'
```

## Step 2 (server): verify every file

Paste this whole block into the SSH terminal. It checks:

- every WAV exists, is mono 16 kHz PCM16 and complete, and matches `metadata.csv` in PCM SHA256
  and duration
- no file exists without a CSV row
- every transcript uses only alphabet letters, which proves the old `ś`/`ć` text was replaced
- `mendeley/` is gone

It uses only the Python standard library and takes about a minute.

```bash
cd "/home/intern/AI & ML"
python3 - <<'EOF'
import csv, hashlib, wave
from pathlib import Path

FINAL = Path("data/final")
ALPHABET = set("aábcdefgǵhiıjklmnńoópqrstuúvwxyz ")
EXPECTED = {"stt": {"train": 15595, "validation": 1950, "test": 1950},
            "hf": {"train": 2090, "test": 523}}
problems = 0
for name, splits in EXPECTED.items():
    folder = FINAL / name
    with open(folder / "metadata.csv", encoding="utf-8", newline="") as f:
        meta = {row["audio_path"]: row for row in csv.DictReader(f)}
    for split, count in splits.items():
        with open(folder / f"{split}.csv", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        listed = {row["audio_path"] for row in rows}
        on_disk = {f"{split}/audio/{p.name}" for p in (folder / split / "audio").glob("*.wav")}
        missing, extra, bad = sorted(listed - on_disk), sorted(on_disk - listed), []
        letters = sum(not set(row["text"]) <= ALPHABET for row in rows)
        for path in sorted(listed & on_disk):
            try:
                with wave.open(str(folder / path), "rb") as w:
                    fmt = (w.getframerate(), w.getnchannels(), w.getsampwidth())
                    frames = w.readframes(w.getnframes())
                    complete = len(frames) == w.getnframes() * 2
            except Exception as error:
                bad.append(f"{path} ({error})")
                continue
            row = meta[path]
            if not (fmt == (16000, 1, 2) and complete and frames
                    and hashlib.sha256(frames).hexdigest() == row["pcm_sha256"]
                    and abs(len(frames) / 32000 - float(row["duration"])) < 1e-6):
                bad.append(path)
        ok = len(rows) == count and not (missing or extra or bad or letters)
        problems += not ok
        print(f"{'OK ' if ok else 'BAD'} {name}/{split}: {len(rows)} rows (expected {count}), "
              f"{len(missing)} missing, {len(extra)} extra, {len(bad)} corrupt, "
              f"{letters} with non-alphabet letters")
        for path in (missing + bad)[:10]:
            print("      ", path)
if (FINAL / "mendeley").exists():
    problems += 1
    print("BAD data/final/mendeley still exists")
print("ALL GOOD" if problems == 0 else f"{problems} problem(s): re-run Step 1, then this block")
EOF
```

Expected result: five `OK` lines and `ALL GOOD`. A `BAD` line lists up to 10 missing or corrupt
files; re-running Step 1 fixes them.

## Step 3: train, decode and read WER/CER

Training, the KenLM language model and the decoders are in `src/`, explained file by file in
`docs/wav2vec2_training.md`. The new code reads `data/final/stt` and `data/final/hf` directly; no
manifests or edits to `src/` are needed. Follow that doc's "Run it on the server" section:

1. Mac: copy `src/`, the three scripts and `docs/` to the server with `rsync`.
2. Server: install `pyctcdecode` and `kenlm`, and make sure KenLM's `lmplz` exists.
3. Server: `python src/vocab.py`, then `python src/train.py` (prints validation WER/CER per epoch
   and test WER/CER at the end).
4. Server: `python src/lm.py`, then `python src/decode.py` (greedy vs beam search vs beam search
   + KenLM).

All results go to `outputs/wav2vec2-lora-kaa/` on the server.

## Step 4 (server): test on other data

```bash
cd "/home/intern/AI & ML" && source .venv/bin/activate
python scripts/prepare_benchmarks.py
python src/benchmark.py data/benchmarks/asr_corpus/test.csv
python src/benchmark.py data/benchmarks/asr_corpus/validation.csv
python src/benchmark.py data/benchmarks/whisper_test/test.csv
```

`prepare_benchmarks.py` builds the test CSVs from `~/asr_project` and `~/whisper_kaa` (see
`docs/datasets.md`); `benchmark.py` scores each with greedy, beam search and beam search + KenLM
and writes `outputs/wav2vec2-lora-kaa/results/benchmarks/<name>.json`.

## Logs

`train.log`, `decode.log` and `benchmark.log` in the repo root are cleaned summaries of the
2026-10-14 run: model, data, settings and results as tables, without progress bars or warnings.
The original raw logs are in `outputs/raw_logs/` on the server (ignored by Git). For a new run,
write the raw output to `outputs/raw_logs/<name>_<date>.log` as shown in
`docs/wav2vec2_training.md`, so the cleaned logs are not overwritten. That doc also lists the older
Mac logs and which model and data each one belongs to.

## Step 5 (Mac): put the server's commits on GitHub

Commit on the server as usual. The server cannot push (its key is not on GitHub), so the Mac
fetches the server's branch and pushes it:

```bash
cd ~/Documents/IT/repos/"AI & ML"
git fetch "intern@192.168.1.88:/home/intern/AI & ML" lora-training-clean
git log --oneline -3 FETCH_HEAD
git push origin FETCH_HEAD:main
```

The push is refused if GitHub's `main` has commits the server doesn't have (for example a PR
merged on GitHub). Don't force it: send those commits to the server, merge them there, then run
the block above again.

```bash
# Mac
git fetch origin
git push "intern@192.168.1.88:/home/intern/AI & ML" origin/main:refs/remotes/origin/main
# server
git merge origin/main
```

To push from the server directly, add the
server's public key (`cat ~/.ssh/id_ed25519.pub` on the server) to the GitHub repo under
Settings → Deploy keys, with "Allow write access" ticked; then `git push origin HEAD:main` works
on the server.

## Caveats

- **Long clips.** 198 STT clips exceed 30 s, the longest 38.5 s. If training runs out of GPU
  memory, set `BATCH_SIZE = 4` and `GRADIENT_ACCUMULATION = 8` in `src/config.py`.
- **Validation is STT only.** HF has no validation split, so the best checkpoint is chosen on
  news speech. Its test score shows whether that hurt it.
- **Mendeley stays out.** It is excluded from `data/final`. Its originals remain in
  `data/karakalpak-mendeley/` and `data/karakalpak-combined/`.
- **No speaker IDs** exist in either dataset, so a speaker can appear in both train and test.
