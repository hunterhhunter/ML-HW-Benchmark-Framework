"""Local checkpoint acquisition and immutable evidence for TimesFM 2.5."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path


TIMESFM25_REPOSITORY = "google/timesfm-2.5-200m-transformers"
_MANIFEST_NAME = "timesfm25-manifest.json"


def write_checkpoint_manifest(checkpoint: Path) -> Path:
    """Hash each top-level checkpoint file once without overwriting evidence."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        raise ValueError(f"TimesFM 2.5 checkpoint directory is missing: {checkpoint}")
    manifest = checkpoint / _MANIFEST_NAME
    if manifest.exists():
        raise FileExistsError(f"TimesFM 2.5 manifest already exists: {manifest}")
    files = {
        path.name: _sha256(path)
        for path in sorted(checkpoint.iterdir())
        if path.is_file() and path.name != _MANIFEST_NAME
    }
    if not files:
        raise ValueError("TimesFM 2.5 checkpoint has no regular files to manifest")
    payload = {
        "repository": TIMESFM25_REPOSITORY,
        "files": files,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "transformers": _transformers_version(),
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def download_checkpoint(output_dir: Path) -> Path:
    """Download the official checkpoint into a new directory and hash it."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose a new empty TimesFM 2.5 output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("TimesFM 2.5 download requires huggingface_hub") from exc
    snapshot_download(repo_id=TIMESFM25_REPOSITORY, local_dir=str(output_dir))
    write_checkpoint_manifest(output_dir)
    return output_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _transformers_version() -> str:
    try:
        import transformers
    except ImportError:
        return "not_installed"
    return str(transformers.__version__)
