"""Validation for metadata that restores unnamed RBLN output bindings."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
import stat


_SCHEMA_KEYS = frozenset(
    {"schema_version", "artifact_sha256", "output_names"}
)
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_OUTPUT_NAME_LENGTH = 256


def _validate_expected_names(
    expected_names: tuple[str, ...], descriptor_count: int
) -> None:
    if type(descriptor_count) is not int or descriptor_count < 2:
        raise ValueError(
            "RBLN output sidecar descriptor_count must be an integer >= 2."
        )
    if type(expected_names) is not tuple or len(expected_names) != descriptor_count:
        raise ValueError(
            "RBLN output sidecar expected names must match descriptor_count."
        )
    if any(
        type(name) is not str
        or not name
        or len(name) > _MAX_OUTPUT_NAME_LENGTH
        for name in expected_names
    ) or len(set(expected_names)) != len(expected_names):
        raise ValueError(
            "RBLN output sidecar expected names must be unique bounded strings."
        )


def _read_manifest(manifest_path: Path) -> object:
    try:
        manifest_stat = manifest_path.lstat()
    except OSError as exc:
        raise ValueError("RBLN output sidecar manifest is required.") from exc
    if not stat.S_ISREG(manifest_stat.st_mode):
        raise ValueError(
            "RBLN output sidecar manifest must be a regular file."
        )
    if manifest_stat.st_size > _MAX_MANIFEST_BYTES:
        raise ValueError(
            "RBLN output sidecar manifest must not exceed 64 KiB."
        )
    try:
        with manifest_path.open("rb") as manifest_file:
            raw_manifest = manifest_file.read(_MAX_MANIFEST_BYTES + 1)
    except OSError as exc:
        raise ValueError("RBLN output sidecar manifest could not be read.") from exc
    if len(raw_manifest) > _MAX_MANIFEST_BYTES:
        raise ValueError(
            "RBLN output sidecar manifest must not exceed 64 KiB."
        )
    try:
        return json.loads(raw_manifest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("RBLN output sidecar manifest is invalid JSON.") from exc


def _artifact_sha256(artifact_path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with artifact_path.open("rb") as artifact_file:
            for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError("RBLN artifact SHA256 could not be computed.") from exc
    return digest.hexdigest()


def load_output_names(
    artifact_path: Path,
    *,
    descriptor_count: int,
    expected_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Return trusted positional names for multiple unnamed RBLN outputs."""

    _validate_expected_names(expected_names, descriptor_count)
    artifact_path = Path(artifact_path)
    document = _read_manifest(Path(f"{artifact_path}.json"))
    if type(document) is not dict or set(document) != _SCHEMA_KEYS:
        raise ValueError(
            "RBLN output sidecar manifest must contain exactly the schema keys."
        )
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != 1
    ):
        raise ValueError(
            "RBLN output sidecar schema_version must be exactly 1."
        )

    artifact_digest = document["artifact_sha256"]
    if (
        type(artifact_digest) is not str
        or len(artifact_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in artifact_digest
        )
    ):
        raise ValueError(
            "RBLN output sidecar artifact_sha256 must be lowercase SHA256."
        )

    output_names = document["output_names"]
    if type(output_names) is not list or len(output_names) != descriptor_count:
        raise ValueError(
            "RBLN output sidecar output count does not match inspection."
        )
    if any(
        type(name) is not str
        or not name
        or len(name) > _MAX_OUTPUT_NAME_LENGTH
        for name in output_names
    ):
        raise ValueError(
            "RBLN output sidecar output names must be bounded strings."
        )
    if len(set(output_names)) != len(output_names):
        raise ValueError("RBLN output sidecar output names must be unique.")
    if set(output_names) != set(expected_names):
        raise ValueError(
            "RBLN output sidecar names do not match Model_Spec."
        )
    if not hmac.compare_digest(
        artifact_digest, _artifact_sha256(artifact_path)
    ):
        raise ValueError(
            "RBLN output sidecar SHA256 does not match the artifact."
        )
    return tuple(output_names)
