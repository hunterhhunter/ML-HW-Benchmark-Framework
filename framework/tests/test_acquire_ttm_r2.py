import json

import pytest

from ttm_r2.download import (
    TTM_R2_REPOSITORY,
    TTM_R2_REVISION,
    download_checkpoint,
)


def test_r2_download_rejects_a_nonempty_destination(tmp_path):
    destination = tmp_path / "checkpoint"
    destination.mkdir()
    (destination / "stale").write_text("x")

    with pytest.raises(FileExistsError, match="nonempty"):
        download_checkpoint(destination)


def test_r2_download_records_main_revision_and_hashes(monkeypatch, tmp_path):
    destination = tmp_path / "checkpoint"

    def fake_snapshot_download(*, repo_id, revision, local_dir):
        assert repo_id == TTM_R2_REPOSITORY
        assert revision == TTM_R2_REVISION
        root = type(destination)(local_dir)
        root.mkdir(parents=True)
        (root / "config.json").write_text("{}")
        (root / "model.safetensors").write_bytes(b"weights")
        return str(root)

    monkeypatch.setattr("ttm_r2.download.snapshot_download", fake_snapshot_download)

    checkpoint = download_checkpoint(destination)

    manifest = json.loads((checkpoint / "ttm-r2-manifest.json").read_text())
    assert manifest["repository"] == TTM_R2_REPOSITORY
    assert manifest["revision"] == "main"
    assert manifest["files"]["model.safetensors"]["sha256"]
