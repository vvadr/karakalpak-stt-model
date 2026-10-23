"""Check GitHub-bound files without changing the index or uploading anything."""

import argparse
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIRS = {
    "data", "outputs", "checkpoints", "publication", ".venv", ".venv-inference", "venv", "env",
    ".cache", "huggingface", "wandb", "mlruns", "runs", ".aws", ".ssh", ".codex", ".agents",
}
LOCAL_FILES = {"Intern Week 1 — Data (ASR).pdf"}
RESEARCH_FILES = {"kaa.arpa", "lm_text.txt", "vocab.json"}
ARTIFACT_SUFFIXES = {
    ".wav", ".flac", ".mp3", ".m4a", ".mp4", ".ogg", ".opus", ".webm",
    ".parquet", ".arrow", ".safetensors", ".bin", ".pt", ".pth", ".ckpt",
    ".onnx", ".tflite", ".arpa", ".zip", ".7z", ".tar", ".tgz", ".pem", ".key",
}
CURATED_LOGS = {"train.log", "decode.log", "benchmark.log"}
MAX_BYTES = 10 * 1024 * 1024


def git_files(staged):
    command = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"] if staged else [
        "git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"
    ]
    output = subprocess.check_output(command, cwd=ROOT).decode("utf-8")
    return sorted(set(name for name in output.split("\0") if name))


def problem(name):
    path = Path(name)
    if path.parts[0] in LOCAL_DIRS or name in LOCAL_FILES:
        return "local data or generated artifact"
    if any(part in {"__pycache__", ".ipynb_checkpoints"} for part in path.parts):
        return "cache or notebook autosave"
    if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
        return "local environment file"
    if name not in RESEARCH_FILES and (path.suffix.lower() in ARTIFACT_SUFFIXES or name.endswith((".tar.gz", ".tar.zst"))):
        return "binary artifact, archive, or key"
    if path.suffix == ".log" and name not in CURATED_LOGS:
        return "raw log"
    local = ROOT / path
    if local.is_file() and local.stat().st_size > MAX_BYTES:
        return "larger than this project's 10 MiB GitHub file budget"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="check only staged additions and modifications")
    args = parser.parse_args()
    names = git_files(args.staged)
    failures = [(name, reason) for name in names if (reason := problem(name))]
    if failures:
        for name, reason in failures:
            print(f"EXCLUDE {name}: {reason}")
        raise SystemExit(1)
    total = sum((ROOT / name).stat().st_size for name in names if (ROOT / name).is_file())
    print(f"PASS: {len(names)} GitHub-bound paths, {total / 1024 / 1024:.2f} MiB; no excluded artifacts found.")
    print("Check only: this command does not stage or upload files. KenLM remains a pinned Git submodule.")


if __name__ == "__main__":
    main()
