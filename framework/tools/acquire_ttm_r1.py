#!/usr/bin/env python3
"""Acquire one immutable local copy of the TTM-R1 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


MODEL_ID = "ibm-granite/granite-timeseries-ttm-r1"


def snapshot_download(*args: Any, **kwargs: Any) -> str:
    """Import the Hub client only when an acquisition is actually requested."""
    from huggingface_hub import snapshot_download as download

    return download(*args, **kwargs)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_files(destination: Path) -> dict[str, dict[str, int | str]]:
    selected = [destination / "config.json"]
    selected.extend(sorted(destination.rglob("*.safetensors")))
    selected.extend(sorted(destination.rglob("*.bin")))
    if not selected[0].is_file():
        raise ValueError("TTM-R1 download did not contain config.json")
    files: dict[str, dict[str, int | str]] = {}
    for path in selected:
        if not path.is_file():
            continue
        files[str(path.relative_to(destination))] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    if len(files) < 2:
        raise ValueError("TTM-R1 download did not contain a model weight file")
    return files


def acquire(destination: str | Path, *, model_id: str = MODEL_ID) -> Path:
    """Download an exact checkpoint once and write a weight identity manifest."""
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"TTM-R1 destination must be absent or empty, not nonempty: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=str(destination))
    if not destination.is_dir():
        raise ValueError(f"TTM-R1 download did not create destination: {destination}")
    manifest = {
        "model_id": model_id,
        "path": str(destination),
        "files": _manifest_files(destination),
    }
    (destination / "ttm-r1-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the fixed IBM Granite TTM-R1 checkpoint")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    print(acquire(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
