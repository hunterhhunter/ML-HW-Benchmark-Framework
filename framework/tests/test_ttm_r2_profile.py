import hashlib

import pytest

from core.model_profiles import SUPPORTED_PROFILES
from core.model_spec import Task
from ttm_r2 import profile


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_ttm_r2_profile_has_fixed_semantic_abi():
    profile_data = SUPPORTED_PROFILES["ttm-r2"]

    assert profile_data["task"] is Task.TIME_SERIES_FORECASTING
    assert profile_data["input_shapes"] == {"past_values": (1, 512, 1)}
    assert profile_data["input_dtype"] == {"past_values": "float32"}
    assert profile_data["output_shapes"] == {"forecast": (1, 96, 1)}
    assert profile_data["default_model_path"].endswith(
        "ibm-granite_granite-timeseries-ttm-r2"
    )


def test_validate_checkpoint_rejects_correct_names_with_wrong_bytes(tmp_path):
    checkpoint = tmp_path / "ibm-granite_granite-timeseries-ttm-r2"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_bytes(b"wrong")
    (checkpoint / "model.safetensors").write_bytes(b"wrong")

    with pytest.raises(ValueError, match="SHA-256"):
        profile.validate_checkpoint(checkpoint)


def test_validate_checkpoint_returns_verified_hashes(monkeypatch, tmp_path):
    checkpoint = tmp_path / "ibm-granite_granite-timeseries-ttm-r2"
    checkpoint.mkdir()
    config = b"config"
    weights = b"weights"
    (checkpoint / "config.json").write_bytes(config)
    (checkpoint / "model.safetensors").write_bytes(weights)
    monkeypatch.setattr(profile, "TTM_R2_CONFIG_SHA256", _sha256(config))
    monkeypatch.setattr(profile, "TTM_R2_MODEL_SHA256", _sha256(weights))

    assert profile.validate_checkpoint(checkpoint) == {
        "config_sha256": _sha256(config),
        "model_sha256": _sha256(weights),
    }


def test_validate_dataset_requires_a_regular_file_with_canonical_bytes(
    monkeypatch, tmp_path
):
    dataset = tmp_path / "ETTh1.csv"
    content = b"date,OT\n0,1.0\n"
    dataset.write_bytes(content)
    monkeypatch.setattr(profile, "TTM_R2_DATASET_SHA256", _sha256(content))

    assert profile.validate_dataset(dataset) == _sha256(content)

    dataset.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        profile.validate_dataset(dataset)


def test_artifact_evidence_reports_content_identity_and_rejects_empty_files(tmp_path):
    artifact = tmp_path / "model.mxq"
    artifact.write_bytes(b"artifact")

    assert profile.artifact_evidence(artifact) == {
        "sha256": _sha256(b"artifact"),
        "size_bytes": 8,
    }

    artifact.write_bytes(b"")
    with pytest.raises(ValueError, match="nonempty"):
        profile.artifact_evidence(artifact)
