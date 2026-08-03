"""Import proven Mobilint BERT compiler artifacts into a result record."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from tools.mobilint_bert_compile.common import get_task_spec, sha256_file
from tools.mobilint_compile_recipes.attempt import STAGES


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"BERT {label} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"BERT {label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"BERT {label} must be a JSON object: {path}")
    return value


def _checked_record(
    task_root: Path,
    record: object,
    *,
    label: str,
    suffix: str,
) -> dict[str, object]:
    if not isinstance(record, Mapping):
        raise ValueError(f"BERT {label} artifact record is missing")
    relative = record.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"BERT {label} artifact path is missing")
    candidate = Path(relative)
    if candidate.is_absolute() or candidate.suffix != suffix:
        raise ValueError(f"BERT {label} artifact path is invalid: {relative!r}")
    path = (task_root / candidate).resolve()
    try:
        normalized = path.relative_to(task_root).as_posix()
    except ValueError as error:
        raise ValueError(f"BERT {label} artifact must stay inside task root") from error
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"BERT {label} artifact must be a non-empty file: {path}")
    size = record.get("size_bytes")
    if type(size) is not int or size != path.stat().st_size:
        raise ValueError(f"BERT {label} artifact size does not match stored evidence")
    stored_hash = record.get("sha256")
    actual_hash = sha256_file(path)
    if stored_hash != actual_hash:
        raise ValueError(f"BERT {label} artifact SHA256 does not match stored evidence")
    return {"path": normalized, "size_bytes": size, "sha256": actual_hash}


def _validate_identity(
    manifest: Mapping[str, object], report: Mapping[str, object]
) -> None:
    task = manifest.get("task")
    if not isinstance(task, str):
        raise ValueError("BERT calibration manifest task is missing")
    spec = get_task_spec(task)
    for field, expected in (
        ("task", spec.name),
        ("model_id", spec.model_id),
        ("target_device", spec.target_device),
    ):
        if manifest.get(field) != expected:
            raise ValueError(f"BERT calibration manifest {field} mismatch")
        if report.get(field) != expected:
            raise ValueError(f"BERT compile report {field} mismatch")


def _empty_stage() -> dict[str, object]:
    return {
        "status": "not_run",
        "started_at": None,
        "finished_at": None,
        "elapsed_seconds": None,
        "exit_code": None,
        "signal": None,
        "error": None,
    }


def _new_result() -> dict[str, Any]:
    return {
        "compile_status": "not_run",
        "runtime_status": "not_run",
        "contract_status": "not_run",
        "quality_status": "not_run",
        "failed_at": None,
        "stages": {stage: _empty_stage() for stage in STAGES},
        "artifacts": [],
    }


def _load_output(path: Path) -> tuple[dict[str, Any], bool]:
    if not path.exists():
        return _new_result(), False
    result = _read_object(path, "result")
    stages = result.get("stages")
    if not isinstance(stages, dict) or tuple(stages) != STAGES:
        raise ValueError("BERT result has invalid attempt stages")
    for field in ("runtime_status", "contract_status", "quality_status"):
        if field not in result:
            raise ValueError(f"BERT result is missing {field}")
    if not isinstance(result.get("artifacts"), list):
        raise ValueError("BERT result artifacts must be a list")
    return result, True


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(payload, temporary, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def import_bert_compile_result(
    task_root: str | Path, output: str | Path
) -> dict[str, Any]:
    """Verify existing BERT evidence and map compiler stages only."""
    root = Path(task_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"BERT task root not found: {root}")
    manifest = _read_object(root / "calibration_manifest.json", "calibration manifest")
    report = _read_object(root / "compile-report.json", "compile report")
    _validate_identity(manifest, report)

    weights = _checked_record(
        root,
        manifest.get("weights"),
        label="embedding weights",
        suffix=".pth",
    )
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("BERT compile report artifacts are missing")
    compiled = [
        _checked_record(root, artifacts.get("mblt"), label="MBLT", suffix=".mblt"),
        _checked_record(root, artifacts.get("mxq"), label="MXQ", suffix=".mxq"),
    ]

    output_path = Path(output).expanduser().resolve()
    result, existing = _load_output(output_path)
    result["compile_status"] = "pass"
    for stage in ("MBLT_COMPILE", "MXQ_COMPILE"):
        stage_result = result["stages"][stage]
        stage_result.update(
            {"status": "pass", "exit_code": 0, "signal": None, "error": None}
        )
    if not existing:
        result["artifacts"] = compiled
    result["bert_provenance"] = {
        "task_root": str(root),
        "weights": weights,
        "calibration_manifest": "calibration_manifest.json",
        "compile_report": "compile-report.json",
    }
    _write_json_atomic(output_path, result)
    return result


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _create_parser().parse_args(argv)
    result = import_bert_compile_result(args.task_root, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
