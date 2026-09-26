"""Canonical resource identity and evidence helpers for TTM-R2."""

from __future__ import annotations

import hashlib
from pathlib import Path


TTM_R2_MODEL_NAME = "ttm-r2"
TTM_R2_CONTRACT_ID = "ttm-r2-etth1-ot-512-96-v1"
TTM_R2_EXPECTED_WINDOWS = 240
TTM_R2_CONFIG_SHA256 = (
    "5e2367547c103e92cb8ebc63cbd5ad4d7bf83facad5a2a2ec261fa770ed659d5"
)
TTM_R2_MODEL_SHA256 = (
    "a706726a7eb01bbcb42994b7dcb3c06ea9557898dbae8d480eb04fe8ccb89710"
)
TTM_R2_DATASET_SHA256 = (
    "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
)

_CHECKPOINT_DIRNAME = "ibm-granite_granite-timeseries-ttm-r2"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(path: Path) -> dict[str, str]:
    """Require the canonical local TTM-R2 checkpoint and its exact bytes."""
    checkpoint = Path(path)
    if not checkpoint.is_dir():
        raise ValueError(f"TTM-R2 checkpoint directory does not exist: {checkpoint}")
    if checkpoint.name != _CHECKPOINT_DIRNAME:
        raise ValueError(
            "TTM-R2 checkpoint directory must be named "
            f"'{_CHECKPOINT_DIRNAME}': {checkpoint}"
        )

    config_path = checkpoint / "config.json"
    model_path = checkpoint / "model.safetensors"
    config_sha256 = sha256_file(config_path)
    model_sha256 = sha256_file(model_path)
    if config_sha256 != TTM_R2_CONFIG_SHA256:
        raise ValueError(
            "TTM-R2 config.json SHA-256 mismatch: "
            f"expected {TTM_R2_CONFIG_SHA256}, got {config_sha256}"
        )
    if model_sha256 != TTM_R2_MODEL_SHA256:
        raise ValueError(
            "TTM-R2 model.safetensors SHA-256 mismatch: "
            f"expected {TTM_R2_MODEL_SHA256}, got {model_sha256}"
        )
    return {
        "config_sha256": config_sha256,
        "model_sha256": model_sha256,
    }


def validate_dataset(path: Path) -> str:
    """Require the canonical ETTh1 CSV bytes."""
    dataset_sha256 = sha256_file(Path(path))
    if dataset_sha256 != TTM_R2_DATASET_SHA256:
        raise ValueError(
            "TTM-R2 ETTh1 dataset SHA-256 mismatch: "
            f"expected {TTM_R2_DATASET_SHA256}, got {dataset_sha256}"
        )
    return dataset_sha256


def artifact_evidence(path: Path) -> dict[str, str | int]:
    """Return path-independent identity for one nonempty compiled artifact."""
    artifact = Path(path)
    if not artifact.is_file():
        raise ValueError(f"TTM-R2 artifact must be a regular file: {artifact}")
    size_bytes = artifact.stat().st_size
    if size_bytes <= 0:
        raise ValueError(f"TTM-R2 artifact must be nonempty: {artifact}")
    return {
        "sha256": sha256_file(artifact),
        "size_bytes": size_bytes,
    }
