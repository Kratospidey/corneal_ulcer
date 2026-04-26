from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import subprocess


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_git_tracked(path: str | Path) -> bool:
    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return False
    return True


def describe_file(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    payload: dict[str, Any] = {
        "path": str(file_path),
        "exists": file_path.exists(),
        "git_tracked": is_git_tracked(file_path),
    }
    if not file_path.exists():
        return payload
    payload.update(
        {
            "sha256": sha256_file(file_path),
            "size_bytes": file_path.stat().st_size,
        }
    )
    return payload


def build_data_provenance(*, manifest_path: str | Path, split_file: str | Path) -> dict[str, Any]:
    return {
        "manifest": describe_file(manifest_path),
        "split_file": describe_file(split_file),
    }
