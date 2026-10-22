"""Prepare local Hugging Face exports. Never uploads or edits source data."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import wave


ROOT = Path(__file__).resolve().parents[2]
EXPORTS = ROOT / "publication"
DATASETS = {
    "stt": ("karakalpak-stt-news", ("train", "validation", "test"), "news and official speech"),
    "hf": ("karakalpak-literary-cleaned", ("train", "test"), "literary speech"),
}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export_provenance():
    return {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "exporter_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "exporter_worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "exporter_sha256": digest(Path(__file__)),
        "note": "Exporter revision is not evidence of the original training commit.",
    }


def fresh_stage(output):
    output = output.resolve()
    if not output.is_relative_to(EXPORTS.resolve()) or output == EXPORTS.resolve():
        raise SystemExit(f"Choose an export subfolder inside {EXPORTS}; source folders are never overwritten.")
    stage = output.with_name(output.name + ".partial")
    if output.exists() or stage.exists():
        raise SystemExit(f"Export or partial export already exists: {output}. Choose a new --output inside publication/.")
    stage.mkdir(parents=True)
    return output, stage


def write_checksums(folder):
    paths = sorted(path for path in folder.rglob("*") if path.is_file() and path.name != "checksums.sha256"
                   and not {".cache", ".git"}.intersection(path.relative_to(folder).parts))
    lines = [f"{digest(path)}  {path.relative_to(folder).as_posix()}\n" for path in paths]
    (folder / "checksums.sha256").write_text("".join(lines), encoding="utf-8")
    print(f"Checksums written for {len(lines)} files: {folder}", flush=True)


def verify_checksums(folder):
    failures = []
    lines = (folder / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    if not lines:
        raise SystemExit("Checksum manifest is empty.")
    for line in lines:
        expected, name = line.split("  ", 1)
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file() or digest(path) != expected:
            failures.append(name)
    if failures:
        raise SystemExit("Checksum failures: " + ", ".join(failures))
    print(f"PASS: all {len(lines)} published file checksums match.")


def dataset_card(name, splits, domain, summary):
    lines = ["---", "language:", "- kaa", "task_categories:", "- automatic-speech-recognition",
             "tags:", "- audio", "configs:", "- config_name: default", "  data_files:"]
    for split in splits:
        lines += [f"  - split: {split}", f'    path: "data/{split}-*.parquet"']
    lines += ["---", "", f"# {name}", "", "DRAFT: complete source credits and release terms before public publication.", "",
              f"Karakalpak {domain}, mono 16 kHz PCM16 WAV embedded in Parquet.", "",
              "| Split | Recordings | Hours |", "| --- | ---: | ---: |"]
    for split, part in summary.items():
        lines.append(f"| {split} | {part['recordings']} | {part['hours']:.3f} |")
    lines += ["", "## Preparation and provenance", "",
              "Recipe: https://github.com/vvadr/karakalpak-stt-model/tree/main/scripts", "",
              "Prepared with build_final_datasets.py; exported with scripts/release/prepare_huggingface.py.",
              "Split membership and transcripts are preserved. Audio bytes are embedded unchanged.",
              "Original text, source IDs/locations, text rules and PCM hashes accompany each row.",
              "provenance/ retains source manifests, changes, quality measurements and review flags.", "",
              "## Sources and release terms", "",
              "TODO: identify creators, upstream URLs/revisions, collection terms and applicable license.",
              "TODO: add LICENSE and the verified license identifier to the YAML metadata.", "",
              "## Limitations", "",
              "Speaker IDs are unavailable. Splits separate exact PCM/transcript duplicates, not speakers.",
              "Review flags are listening candidates, not confirmed errors; review.csv is preserved.",
              "TODO: document review decisions, intended use, citation and release version.", ""]
    return "\n".join(lines)


def export_dataset(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Audio, Features, Value

    name, splits, domain = DATASETS[args.dataset]
    source = ROOT / "data/final" / args.dataset
    metadata = read_csv(source / "metadata.csv")
    by_path = {row["audio_path"]: row for row in metadata}
    if len(by_path) != len(metadata):
        raise SystemExit("Duplicate paths in source metadata.")
    detail_fields = ["source_id", "source_location", "text_original", "text_rules", "pcm_sha256"]
    features = Features({"audio": Audio(sampling_rate=16000), "text": Value("string"),
                         "duration": Value("float64"), "id": Value("string"),
                         **{key: Value("string") for key in detail_fields}})
    output, stage = fresh_stage(args.output or EXPORTS / "datasets" / name)
    (stage / "data").mkdir()
    summary, manifest_hashes, exported_paths = {}, {}, set()
    for split in splits:
        manifest = source / f"{split}.csv"
        manifest_hashes[manifest.name] = digest(manifest)
        rows = read_csv(manifest)
        if not rows:
            raise ValueError(f"Empty source split: {manifest}")
        pending, pending_bytes, shard = [], 0, 0

        def flush():
            nonlocal pending, pending_bytes, shard
            target = stage / "data" / f"{split}-{shard:05d}.parquet"
            pq.write_table(pa.Table.from_pylist(pending, schema=features.arrow_schema), target, compression="zstd")
            print(f"{name}/{target.name}: {len(pending)} recordings", flush=True)
            pending, pending_bytes, shard = [], 0, shard + 1

        for row in rows:
            key = row["audio_path"]
            path = (source / key).resolve()
            detail = by_path[key]
            if (not path.is_relative_to(source.resolve()) or key in exported_paths or detail["split"] != split
                    or row["text"] != detail["text"] or float(row["duration"]) != float(detail["duration"])):
                raise ValueError(f"Source manifest mismatch: {key}")
            data = path.read_bytes()
            with wave.open(io.BytesIO(data), "rb") as audio:
                if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
                    raise ValueError(f"Expected mono 16 kHz PCM16: {key}")
                frames = audio.readframes(audio.getnframes())
                if not frames or len(frames) != audio.getnframes() * 2 or abs(len(frames) / 32000 - float(row["duration"])) > 1e-6:
                    raise ValueError(f"Incomplete audio or duration mismatch: {key}")
            if hashlib.sha256(frames).hexdigest() != detail["pcm_sha256"]:
                raise ValueError(f"PCM hash mismatch: {key}")
            pending.append({"audio": {"bytes": data, "path": path.name}, "text": row["text"],
                            "duration": float(row["duration"]), "id": path.stem,
                            **{field: detail[field] for field in detail_fields}})
            pending_bytes += len(data)
            exported_paths.add(key)
            if pending_bytes >= args.shard_mb * 1024 * 1024:
                flush()
        if pending:
            flush()
        summary[split] = {"recordings": len(rows), "hours": sum(float(r["duration"]) for r in rows) / 3600, "shards": shard}
    if exported_paths != set(by_path):
        raise ValueError("Split manifests do not cover source metadata exactly.")
    provenance = stage / "provenance"
    provenance.mkdir()
    for relative in ["metadata.csv", "changes.csv", "review.csv", "split_summary.json", "provenance/quality.csv"]:
        shutil.copy2(source / relative, provenance / Path(relative).name)
    (stage / "README.md").write_text(dataset_card(name, splits, domain, summary), encoding="utf-8")
    write_json(stage / "export_manifest.json", {**export_provenance(), "dataset": args.dataset,
               "source_manifest_sha256": manifest_hashes, "splits": summary, "public_release_status": "draft"})
    write_checksums(stage)
    stage.rename(output)
    print(f"Dataset draft ready: {output}\nComplete the card and source terms, then refresh checksums before uploading.")


def export_model(args):
    run = args.run_dir.resolve()
    required = ["final_model/model.safetensors", "final_model/config.json", "final_model/vocab.json",
                "final_model/tokenizer_config.json", "lm/kaa_4gram.arpa", "results/decoder_results.json",
                "results/decoded_stt_test.csv", "results/decoded_hf_test.csv", "history.csv"]
    missing = [name for name in required if not (run / name).is_file()]
    if not any((run / "final_model" / name).is_file() for name in ["processor_config.json", "preprocessor_config.json"]):
        missing.append("final_model/processor_config.json or preprocessor_config.json")
    if missing:
        raise SystemExit("Incomplete run; no export created. Missing:\n  " + "\n  ".join(missing))
    acoustic = json.loads((run / "final_model/config.json").read_text())
    vocab = json.loads((run / "final_model/vocab.json").read_text())
    decoder = json.loads((run / "results/decoder_results.json").read_text())
    if acoustic["vocab_size"] != len(vocab) or acoustic["pad_token_id"] != vocab["[PAD]"]:
        raise SystemExit("Model head and vocabulary do not match.")
    for name in ["alpha", "beta", "beam_width", "results"]:
        if name not in decoder:
            raise SystemExit(f"Decoder results missing {name}.")
    output, stage = fresh_stage(args.output or EXPORTS / "models/wav2vec2-xls-r-300m-karakalpak")
    model_files = []
    for path in sorted((run / "final_model").iterdir()):
        if path.is_file() and (path.suffix == ".json" or path.name == "model.safetensors"):
            shutil.copy2(path, stage / path.name)
            model_files.append(path.name)
    (stage / "lm").mkdir()
    shutil.copy2(run / "lm/kaa_4gram.arpa", stage / "lm/kaa_4gram.arpa")
    write_json(stage / "decoder_config.json", {"type": "kenlm", "beam_width": decoder["beam_width"],
               "alpha": decoder["alpha"], "beta": decoder["beta"], "lm_path": "lm/kaa_4gram.arpa",
               "sampling_rate": 16000, "vocab_sha256": digest(stage / "vocab.json")})
    evaluation = stage / "evaluation"
    evaluation.mkdir()
    write_json(evaluation / "decoder_results.json", {**decoder, "lm": "lm/kaa_4gram.arpa"})
    for name, dataset in [("decoded_stt_test.csv", "stt"), ("decoded_hf_test.csv", "hf")]:
        rows = read_csv(run / "results" / name)
        if not rows:
            raise ValueError(f"Empty evaluation file: {name}")
        for row in rows:
            row["audio_path"] = f"{dataset}/test/audio/{Path(row['audio_path']).name}"
        with (evaluation / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (stage / "training").mkdir()
    shutil.copy2(run / "history.csv", stage / "training/history.csv")
    write_json(stage / "export_manifest.json", {**export_provenance(), "source_run_name": run.name,
               "model_files": model_files, "training_git_commit": None, "public_release_status": "draft"})
    card = ["---", "language:", "- kaa", "pipeline_tag: automatic-speech-recognition", "library_name: transformers",
            "base_model: facebook/wav2vec2-xls-r-300m", "---", "", "# Karakalpak wav2vec2 XLS-R 300M + KenLM", "",
            "DRAFT: verify server provenance, complete credits/license and test loading before public release.", "",
            "Acoustic model: LoRA merged into Wav2Vec2ForCTC; matching tokenizer and processor are included.",
            "Plain Transformers/argmax inference is greedy. KenLM scores require lm/ and decoder_config.json.",
            "The existing project CLI does not yet consume this portable decoder_config.json directly.", "",
            "Code: https://github.com/vvadr/karakalpak-stt-model", "",
            "| Set | Decoder | WER | CER |", "| --- | --- | ---: | ---: |"]
    for name, decoders in decoder["results"].items():
        for mode, metrics in decoders.items():
            card.append(f"| {name} | {mode} | {metrics['wer']:.2%} | {metrics['cer']:.2%} |")
    card += ["", "Scores above are copied from the supplied run; exporting is not a new evaluation.", "",
             "TODO: exact training commit, base revision, dependency versions, configuration and dataset revisions.",
             "TODO: add dataset links, credits, LICENSE/notices, supported loading example and release citation.",
             "Limitations: speaker identities unavailable; news-only validation; normalization affects WER.", ""]
    (stage / "README.md").write_text("\n".join(card), encoding="utf-8")
    write_checksums(stage)
    stage.rename(output)
    print(f"Model draft ready: {output}\nComplete and verify the card, then refresh checksums before uploading.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dataset = commands.add_parser("dataset", help="export one final dataset as embedded-audio Parquet shards")
    dataset.add_argument("dataset", choices=DATASETS)
    dataset.add_argument("--output", type=Path)
    dataset.add_argument("--shard-mb", type=int, default=256)
    model = commands.add_parser("model", help="assemble a complete server model and KenLM draft")
    model.add_argument("--run-dir", type=Path, default=ROOT / "outputs/wav2vec2-lora-kaa")
    model.add_argument("--output", type=Path)
    for name in ["checksums", "verify"]:
        command = commands.add_parser(name, help="refresh or verify an export's SHA256 checksums")
        command.add_argument("folder", type=Path)
    args = parser.parse_args()
    if args.command == "dataset":
        if args.shard_mb <= 0:
            parser.error("--shard-mb must be positive")
        export_dataset(args)
    elif args.command == "model":
        export_model(args)
    elif args.command == "checksums":
        write_checksums(args.folder.resolve())
    else:
        verify_checksums(args.folder.resolve())


if __name__ == "__main__":
    main()
