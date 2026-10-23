"""Compare every release file with its committed Hugging Face content hash."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi


def file_hash(path, algorithm, git_blob=False):
    value = hashlib.new(algorithm)
    if git_blob:
        value.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def verify_release(repo_id, folder, repo_type, revision=None):
    info = HfApi().repo_info(repo_id, repo_type=repo_type, revision=revision, files_metadata=True)
    remote = {item.rfilename: item for item in info.siblings}
    local = {path.relative_to(folder).as_posix(): path for path in folder.rglob("*")
             if path.is_file() and not {".git", ".cache"}.intersection(path.relative_to(folder).parts)
             and path.name != ".DS_Store"}
    extra = set(remote) - set(local) - {".gitattributes"}
    if extra:
        raise ValueError("Unexpected remote files: " + ", ".join(sorted(extra)))
    total = 0
    for name, path in sorted(local.items()):
        item = remote.get(name)
        if item is None:
            raise ValueError(f"Missing remote file: {name}")
        if item.size != path.stat().st_size:
            raise ValueError(f"Remote size mismatch: {name}")
        if item.lfs is not None:
            actual, expected = file_hash(path, "sha256"), item.lfs.sha256
        else:
            actual, expected = file_hash(path, "sha1", git_blob=True), item.blob_id
        if actual != expected:
            raise ValueError(f"Remote content hash mismatch: {name}")
        total += path.stat().st_size
    return {"repo_id": repo_id, "repo_type": repo_type, "revision": info.sha,
            "files_verified": len(local), "bytes_verified": total,
            "private": info.private, "content_hashes_match": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--repo-type", choices=["model", "dataset"], required=True)
    parser.add_argument("--revision")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify_release(args.repo_id, args.folder.resolve(), args.repo_type, args.revision)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
