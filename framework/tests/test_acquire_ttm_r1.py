import json

import pytest

from tools import acquire_ttm_r1


def test_acquire_rejects_nonempty_destination(tmp_path):
    """Catches a checkpoint download silently mixing with old weights."""
    destination = tmp_path / "model"
    destination.mkdir()
    (destination / "stale").write_text("x")

    with pytest.raises(FileExistsError, match="nonempty"):
        acquire_ttm_r1.acquire(destination)


def test_acquire_writes_a_manifest_for_the_downloaded_config_and_weights(
    monkeypatch, tmp_path
):
    """Catches runs without a checkpoint identity record."""
    destination = tmp_path / "model"

    def fake_snapshot_download(*, repo_id, local_dir):
        assert repo_id == "ibm-granite/granite-timeseries-ttm-r1"
        local = destination.__class__(local_dir)
        local.mkdir(parents=True)
        (local / "config.json").write_text("{}")
        (local / "model.safetensors").write_bytes(b"weights")
        return str(local)

    monkeypatch.setattr(acquire_ttm_r1, "snapshot_download", fake_snapshot_download)

    acquired = acquire_ttm_r1.acquire(destination)

    assert acquired == destination.resolve()
    manifest = json.loads((destination / "ttm-r1-manifest.json").read_text())
    assert manifest["model_id"] == "ibm-granite/granite-timeseries-ttm-r1"
    assert set(manifest["files"]) == {"config.json", "model.safetensors"}
