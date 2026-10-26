"""Acquire and identify the fixed 512-96-r2 checkpoint."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

TTM_R2_REPOSITORY = "ibm-granite/granite-timeseries-ttm-r2"
TTM_R2_REVISION = "main"


def snapshot_download(*args: Any, **kwargs: Any) -> str:
    """Import Hub support only when checkpoint acquisition is requested."""
    from huggingface_hub import snapshot_download as download

    return download(*args, **kwargs)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_files(checkpoint: Path) -> dict[str, dict[str, int | str]]:
    required = checkpoint / "config.json"
    weights = sorted(checkpoint.rglob("*.safetensors")) + sorted(checkpoint.rglob("*.bin"))
    if not required.is_file() or not weights:
        raise ValueError("TTM-R2 checkpoint must include config.json and one weight file")
    selected = [required, *weights]
    return {
        str(path.relative_to(checkpoint)): {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
        for path in selected
    }


def download_checkpoint(output_dir: Path) -> Path:
    """Download R2 main into a new directory and write one immutable manifest."""
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"TTM-R2 destination must not be nonempty: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=TTM_R2_REPOSITORY,
        revision=TTM_R2_REVISION,
        local_dir=str(output_dir),
    )
    if not output_dir.is_dir():
        raise ValueError(f"TTM-R2 download did not create destination: {output_dir}")
    manifest = output_dir / "ttm-r2-manifest.json"
    if manifest.exists():
        raise FileExistsError(f"TTM-R2 manifest already exists: {manifest}")
    manifest.write_text(
        json.dumps(
            {
                "repository": TTM_R2_REPOSITORY,
                "revision": TTM_R2_REVISION,
                "path": str(output_dir),
                "files": _checkpoint_files(output_dir),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_dir
