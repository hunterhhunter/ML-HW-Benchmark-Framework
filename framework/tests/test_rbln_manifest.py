import hashlib
import json
from pathlib import Path

import pytest

from runtimes.rbln_manifest import load_output_names


EXPECTED_NAMES = ("start_logits", "end_logits")


def _write_manifest(
    artifact: Path,
    *,
    output_names=EXPECTED_NAMES,
    artifact_sha256: str | None = None,
    schema_version=1,
    extra: dict | None = None,
) -> Path:
    manifest = {
        "schema_version": schema_version,
        "artifact_sha256": artifact_sha256
        or hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "output_names": list(output_names),
    }
    if extra:
        manifest.update(extra)
    manifest_path = Path(f"{artifact}.json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _load(artifact: Path) -> tuple[str, ...]:
    return load_output_names(
        artifact,
        descriptor_count=2,
        expected_names=EXPECTED_NAMES,
    )


def test_load_output_names_accepts_sha_bound_manifest(tmp_path):
    artifact = tmp_path / "model.rbln"
    artifact.write_bytes(b"compiled")
    _write_manifest(artifact)

    assert _load(artifact) == EXPECTED_NAMES


def test_load_output_names_preserves_declared_position_order(tmp_path):
    artifact = tmp_path / "model.rbln"
    artifact.write_bytes(b"compiled")
    _write_manifest(artifact, output_names=tuple(reversed(EXPECTED_NAMES)))

    assert _load(artifact) == tuple(reversed(EXPECTED_NAMES))


def test_load_output_names_requires_sidecar(tmp_path):
    artifact = tmp_path / "model.rbln"
    artifact.write_bytes(b"compiled")

    with pytest.raises(ValueError, match="sidecar manifest is required"):
        _load(artifact)


def test_load_output_names_rejects_stale_artifact_hash(tmp_path):
    artifact = tmp_path / "model.rbln"
    artifact.write_bytes(b"compiled")
    _write_manifest(artifact, artifact_sha256="0" * 64)

    with pytest.raises(ValueError, match="SHA256 does not match"):
        _load(artifact)


@pytest.mark.parametrize(
    ("manifest_kwargs", "message"),
    [
        ({"schema_version": True}, "schema_version"),
        ({"schema_version": 2}, "schema_version"),
        ({"artifact_sha256": "A" * 64}, "lowercase SHA256"),
        ({"artifact_sha256": "0" * 63}, "lowercase SHA256"),
        ({"output_names": ("start_logits",)}, "output count"),
        ({"output_names": ("start_logits", "start_logits")}, "unique"),
        ({"output_names": ("start_logits", "scores")}, "Model_Spec"),
        ({"output_names": ("start_logits", "")}, "bounded strings"),
        ({"output_names": ("start_logits", "x" * 257)}, "bounded strings"),
        ({"extra": {"comment": "untrusted"}}, "exactly the schema keys"),
    ],
)
def test_load_output_names_rejects_invalid_schema(
    tmp_path, manifest_kwargs, message
):
    artifact = tmp_path / "model.rbln"
    artifact.write_bytes(b"compiled")
    _write_manifest(artifact, **manifest_kwargs)

    with pytest.raises(ValueError, match=message):
        _load(artifact)


def test_load_output_names_rejects_non_object_json(tmp_path):
    artifact = tmp_path / "model.rbln"
    artifact.write_bytes(b"compiled")
    Path(f"{artifact}.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly the schema keys"):
        _load(artifact)


@pytest.mark.parametrize("content", [b"{", b"\xff"])
def test_load_output_names_rejects_invalid_json_or_utf8(tmp_path, content):
    artifact = tmp_path / "model.rbln"
    artifact.write_bytes(b"compiled")
    Path(f"{artifact}.json").write_bytes(content)

    with pytest.raises(ValueError, match="invalid JSON"):
        _load(artifact)


def test_load_output_names_rejects_oversized_manifest(tmp_path):
    artifact = tmp_path / "model.rbln"
    artifact.write_bytes(b"compiled")
    Path(f"{artifact}.json").write_bytes(b" " * (64 * 1024 + 1))

    with pytest.raises(ValueError, match="64 KiB"):
        _load(artifact)


def test_load_output_names_rejects_non_regular_manifest(tmp_path):
    artifact = tmp_path / "model.rbln"
    artifact.write_bytes(b"compiled")
    target = tmp_path / "manifest-target.json"
    target.write_text("{}", encoding="utf-8")
    Path(f"{artifact}.json").symlink_to(target)

    with pytest.raises(ValueError, match="regular file"):
        _load(artifact)


def test_load_output_names_rejects_non_builtin_descriptor_count(tmp_path):
    artifact = tmp_path / "model.rbln"
    artifact.write_bytes(b"compiled")
    _write_manifest(artifact)

    with pytest.raises(ValueError, match="descriptor_count"):
        load_output_names(
            artifact,
            descriptor_count=True,
            expected_names=EXPECTED_NAMES,
        )
