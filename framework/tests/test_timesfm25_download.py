import hashlib
import json
from pathlib import Path

import pytest

from timesfm25.download import (
    TIMESFM25_REPOSITORY,
    write_checkpoint_manifest,
)


def test_write_manifest_hashes_every_regular_checkpoint_file(tmp_path: Path):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "nested").mkdir()
    (checkpoint / "nested" / "ignored.txt").write_text("nested", encoding="utf-8")

    manifest_path = write_checkpoint_manifest(checkpoint)

    assert manifest_path == checkpoint / "timesfm25-manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["repository"] == TIMESFM25_REPOSITORY
    assert payload["files"] == {
        "config.json": hashlib.sha256(b"{}").hexdigest(),
        "model.safetensors": hashlib.sha256(b"weights").hexdigest(),
    }


def test_manifest_refuses_to_overwrite_existing_evidence(tmp_path: Path):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    write_checkpoint_manifest(checkpoint)

    with pytest.raises(FileExistsError, match="manifest already exists"):
        write_checkpoint_manifest(checkpoint)
