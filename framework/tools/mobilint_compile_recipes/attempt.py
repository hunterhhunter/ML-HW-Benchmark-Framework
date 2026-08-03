"""Immutable, evidence-oriented records for Mobilint compiler attempts."""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from tools.mobilint_compile_recipes.contracts import sha256_file


STAGES = (
    "SOURCE_PREPARE",
    "SOURCE_SMOKE",
    "CALIBRATION_PREPARE",
    "MBLT_COMPILE",
    "MXQ_COMPILE",
    "ARIES_LOAD",
    "CONTRACT_CHECK",
    "TASK_SMOKE",
)

_QUALITY_SAMPLE_COLUMNS = ("total_samples", "samples", "Total Samples")
_QUALITY_METRICS = (
    "accuracy",
    "f1",
    "MSE",
    "MAE",
    "RMSE",
    "Top-1",
    "Top-5",
    "Top-1 Accuracy",
    "Top-5 Accuracy",
    "mAP@0.5",
    "mAP@0.5:0.95",
    "precision",
    "recall",
    "EM",
    "exact_match",
)
_PACKAGE_NAMES = ("qbcompiler", "qbruntime", "torch", "torchvision", "transformers")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_segment(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError(f"{field} must be a path segment")
    if Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"{field} must be a path segment")
    return value


def _attempt_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.is_dir() or not (root / "result.json").is_file():
        raise ValueError(f"not an attempt root: {root}")
    return root


@contextmanager
def _attempt_lock(root: Path):
    """Hold the per-attempt OS lock for one complete state transaction."""
    with (root / ".attempt.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _relative_attempt_path(root: Path, path: str | Path, field: str) -> str:
    candidate = Path(path).expanduser().resolve()
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"{field} must be inside the attempt root") from error


def _environment_report() -> dict[str, object]:
    packages: dict[str, str | None] = {}
    for name in _PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "os": {"system": platform.system(), "release": platform.release()},
        "architecture": platform.machine(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "version_info": list(sys.version_info[:3]),
        },
        "packages": packages,
        "wheel": {
            "filename": os.environ.get("MOBILINT_QBCOMPILER_WHEEL_NAME"),
            "sha256": os.environ.get("MOBILINT_QBCOMPILER_WHEEL_SHA256"),
        },
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _load_result(root: Path) -> dict[str, Any]:
    with (root / "result.json").open(encoding="utf-8") as handle:
        result = json.load(handle)
    if tuple(result.get("stages", {})) != STAGES:
        raise ValueError(f"invalid attempt result stages: {root}")
    return result


def _save_result(root: Path, result: dict[str, Any]) -> None:
    _write_json_atomic(root / "result.json", result)


def _stage_status(result: Mapping[str, Any], stages: Sequence[str]) -> str:
    values = [result["stages"][stage]["status"] for stage in stages]
    if "fail" in values:
        return "fail"
    if values and all(value == "pass" for value in values):
        return "pass"
    return "not_run"


def _refresh_independent_statuses(result: dict[str, Any]) -> None:
    result["compile_status"] = _stage_status(result, ("MBLT_COMPILE", "MXQ_COMPILE"))
    result["runtime_status"] = _stage_status(result, ("ARIES_LOAD", "TASK_SMOKE"))
    result["contract_status"] = _stage_status(result, ("CONTRACT_CHECK",))


def create_attempt(
    output_root: str | Path,
    attempt_id: str,
    model: str,
    variant: str,
    metadata: Mapping[str, Any],
) -> Path:
    """Create a fresh attempt directory; existing roots are never reused."""
    attempt_id = _safe_segment(attempt_id, "attempt_id")
    model = _safe_segment(model, "model")
    variant = _safe_segment(variant, "variant")
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    try:
        saved_metadata = json.loads(json.dumps(dict(metadata)))
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must be JSON serializable") from error

    base = Path(output_root).expanduser().resolve()
    root = base / attempt_id / model / variant
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"attempt root already exists: {root}") from error

    result: dict[str, Any] = {
        "attempt_id": attempt_id,
        "model": model,
        "variant": variant,
        "created_at": _utc_now(),
        "metadata": saved_metadata,
        "failed_at": None,
        "compile_status": "not_run",
        "runtime_status": "not_run",
        "contract_status": "not_run",
        "quality_status": "not_run",
        "stages": {
            stage: {
                "status": "not_run",
                "started_at": None,
                "finished_at": None,
                "elapsed_seconds": None,
                "exit_code": None,
                "signal": None,
                "error": None,
            }
            for stage in STAGES
        },
        "artifacts": [],
    }
    _write_json_atomic(root / "environment.json", _environment_report())
    _write_json_atomic(root / "result.json", result)
    (root / "compile.log").touch(exist_ok=False)
    (root / ".attempt.lock").touch(exist_ok=False)
    return root


def execute_stage(
    attempt_root: str | Path, stage: str, command: Sequence[str]
) -> int:
    """Run one stage, streaming combined output and atomically recording it."""
    root = _attempt_root(attempt_root)
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("command must be a non-empty sequence of strings")

    with _attempt_lock(root):
        result = _load_result(root)
        stage_result = result["stages"][stage]
        if stage_result["status"] != "not_run":
            raise ValueError(f"stage already recorded: {stage}")
        if result["failed_at"] is not None:
            raise RuntimeError(f"attempt already failed at {result['failed_at']}")

        started_at = _utc_now()
        started = time.monotonic()
        with (root / "compile.log").open(
            "a", encoding="utf-8", buffering=1
        ) as log:
            process = subprocess.Popen(
                list(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            return_code = process.wait()

        elapsed = time.monotonic() - started
        signal = -return_code if return_code < 0 else None
        error = None
        if return_code < 0:
            error = f"terminated by signal {signal}"
        elif return_code:
            error = f"exit code {return_code}"
        stage_result.update(
            {
                "status": "pass" if return_code == 0 else "fail",
                "started_at": started_at,
                "finished_at": _utc_now(),
                "elapsed_seconds": elapsed,
                "exit_code": return_code,
                "signal": signal,
                "error": error,
                "command": list(command),
            }
        )
        if return_code != 0 and result["failed_at"] is None:
            result["failed_at"] = stage
        _refresh_independent_statuses(result)
        _save_result(root, result)
    return return_code


def record_artifact(attempt_root: str | Path, artifact: str | Path) -> None:
    """Append immutable size and digest evidence for an in-attempt artifact."""
    root = _attempt_root(attempt_root)
    relative = _relative_attempt_path(root, artifact, "artifact")
    path = root / relative
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"artifact must be a non-empty file: {path}")

    with _attempt_lock(root):
        result = _load_result(root)
        if any(entry["path"] == relative for entry in result["artifacts"]):
            raise ValueError(f"artifact already recorded: {relative}")
        result["artifacts"].append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        _save_result(root, result)


def _ensure_quality_not_recorded(result: Mapping[str, Any]) -> None:
    if result["quality_status"] != "not_run":
        raise ValueError("quality evidence is already recorded for this immutable attempt")


def record_quality_csv(attempt_root: str | Path, result_csv: str | Path) -> None:
    """Record allowed task-quality metrics from a framework result CSV."""
    root = _attempt_root(attempt_root)
    relative = _relative_attempt_path(root, result_csv, "result CSV")
    path = root / relative
    if not path.is_file():
        raise ValueError(f"result CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or rows[-1] is None:
        raise ValueError("result CSV must contain a data row")
    final_row = rows[-1]
    sample_column = next(
        (column for column in _QUALITY_SAMPLE_COLUMNS if final_row.get(column)), None
    )
    if sample_column is None:
        raise ValueError("result CSV must contain a non-empty sample-count field")

    with _attempt_lock(root):
        result = _load_result(root)
        _ensure_quality_not_recorded(result)
        result["quality_status"] = "pass"
        result["quality"] = {
            "result_csv": relative,
            "sha256": sha256_file(path),
            "sample_count": final_row[sample_column],
            "metrics": {
                column: final_row[column]
                for column in _QUALITY_METRICS
                if final_row.get(column) not in (None, "")
            },
        }
        _save_result(root, result)


def record_quality_failure(
    attempt_root: str | Path, exit_code: int, log_path: str | Path
) -> None:
    """Record failed framework-E2E evidence without changing stage outcomes."""
    root = _attempt_root(attempt_root)
    if isinstance(exit_code, bool) or not isinstance(exit_code, int) or exit_code == 0:
        raise ValueError("quality failure requires a nonzero integer exit code")
    relative = _relative_attempt_path(root, log_path, "quality failure log")
    path = root / relative
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("quality failure log must be a non-empty regular file")

    with _attempt_lock(root):
        result = _load_result(root)
        _ensure_quality_not_recorded(result)
        result["quality_status"] = "fail"
        result["quality_failure"] = {
            "exit_code": exit_code,
            "log": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        _save_result(root, result)


def _parse_metadata(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("metadata must be JSON") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("metadata must be a JSON object")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="subcommand", required=True)

    create = commands.add_parser("create")
    create.add_argument("--output-root", required=True)
    create.add_argument("--attempt-id", required=True)
    create.add_argument("--model", required=True)
    create.add_argument("--variant", required=True)
    create.add_argument("--metadata-json", type=_parse_metadata, default={})

    run = commands.add_parser("run")
    run.add_argument("--attempt-root", required=True)
    run.add_argument("--stage", required=True, choices=STAGES)
    run.add_argument("command", nargs=argparse.REMAINDER)

    artifact = commands.add_parser("artifact")
    artifact.add_argument("--attempt-root", required=True)
    artifact.add_argument("--artifact", required=True)

    quality = commands.add_parser("quality")
    quality.add_argument("--attempt-root", required=True)
    quality.add_argument("--result-csv", required=True)

    quality_failure = commands.add_parser("quality-failure")
    quality_failure.add_argument("--attempt-root", required=True)
    quality_failure.add_argument("--exit-code", required=True, type=int)
    quality_failure.add_argument("--log", required=True)

    show = commands.add_parser("show")
    show.add_argument("--attempt-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the attempt recorder CLI and return its process exit code."""
    args = _build_parser().parse_args(argv)
    if args.subcommand == "create":
        print(
            create_attempt(
                args.output_root,
                args.attempt_id,
                args.model,
                args.variant,
                args.metadata_json,
            )
        )
        return 0
    if args.subcommand == "run":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        code = execute_stage(args.attempt_root, args.stage, command)
        return code if code >= 0 else 128 + -code
    if args.subcommand == "artifact":
        record_artifact(args.attempt_root, args.artifact)
        return 0
    if args.subcommand == "quality":
        record_quality_csv(args.attempt_root, args.result_csv)
        return 0
    if args.subcommand == "quality-failure":
        record_quality_failure(args.attempt_root, args.exit_code, args.log)
        return 0
    if args.subcommand == "show":
        root = _attempt_root(args.attempt_root)
        print(json.dumps(_load_result(root), indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"unexpected subcommand: {args.subcommand}")


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI
    raise SystemExit(main())
