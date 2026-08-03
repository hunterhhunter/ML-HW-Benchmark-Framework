#!/usr/bin/env python3
"""Reproduce strict Furiosa RNGD compile failures in isolated processes."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.furiosa_compile_repro import (  # noqa: E402
    CaseConfig,
    CaseResult,
    StageResult,
    match_known_signature,
    run_case,
    safe_error_line,
    write_json,
)


_CASES = ("resnet50", "yolov5m", "patchtst")
_INPUT_CONTRACTS = {
    "resnet50": (
        {"name": "images", "shape": [1, 3, 224, 224], "dtype": "float32"},
    ),
    "yolov5m": (
        {"name": "images", "shape": [1, 3, 640, 640], "dtype": "float32"},
    ),
    "patchtst": (
        {
            "name": "past_values",
            "shape": [1, 512, 7],
            "dtype": "float32",
        },
        {
            "name": "past_observed_mask",
            "shape": [1, 512, 7],
            "dtype": "bool",
        },
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run CPU reference and strict Furiosa RNGD first-call compilation "
            "for previously failing models."
        )
    )
    parser.add_argument("--case", required=True, choices=(*_CASES, "all"))
    parser.add_argument("--device", default="furiosa:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/furiosa-compile-repro"),
    )
    parser.add_argument(
        "--yolov5-path",
        type=Path,
        default=Path("models/yolov5m/yolov5mu.pt"),
    )
    parser.add_argument(
        "--patchtst-path",
        type=Path,
        default=Path("models/ibm-research_patchtst-fm-r1"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--_child-result",
        type=Path,
        help=argparse.SUPPRESS,
    )
    return parser


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def collect_environment() -> dict[str, Any]:
    """Collect version evidence without importing vendor runtime packages."""
    evidence: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _package_version("torch"),
        "torchvision": _package_version("torchvision"),
        "furiosa_torch": _package_version("furiosa-torch"),
        "transformers": _package_version("transformers"),
        "ultralytics": _package_version("ultralytics"),
        "granite_tsfm": _package_version("granite-tsfm"),
    }
    smi = shutil.which("furiosa-smi")
    if smi is not None:
        try:
            completed = subprocess.run(
                [smi, "info"],
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
            )
            evidence["furiosa_smi_info"] = (
                completed.stdout.strip() or completed.stderr.strip()
            )
            evidence["furiosa_smi_exit_code"] = completed.returncode
        except (OSError, subprocess.SubprocessError) as exc:
            evidence["furiosa_smi_error"] = f"{type(exc).__name__}: {exc}"
    else:
        evidence["furiosa_smi_info"] = None
    return evidence


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else None


def build_invocation_evidence(config: CaseConfig) -> dict[str, Any]:
    """Describe the exact code, artifact, device, and input contract used."""
    model_path = None
    if config.model_path is not None:
        model_path = str(config.model_path.expanduser().resolve())
    return {
        "case": config.case,
        "model_path": model_path,
        "device": config.device,
        "seed": config.seed,
        "git_revision": _git_revision(),
        "inputs": list(_INPUT_CONTRACTS[config.case]),
    }


def _model_path_for_case(args: argparse.Namespace, case: str) -> Path | None:
    if case == "yolov5m":
        return args.yolov5_path
    if case == "patchtst":
        return args.patchtst_path
    return None


def run_child(args: argparse.Namespace) -> int:
    if args.case == "all":
        raise ValueError("Internal child mode accepts exactly one model case.")
    if args._child_result is None:
        raise ValueError("Internal child mode requires --_child-result.")
    config = CaseConfig(
        case=args.case,
        model_path=_model_path_for_case(args, args.case),
        device=args.device,
        seed=args.seed,
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    environment = collect_environment()
    invocation = build_invocation_evidence(config)
    write_json(
        args._child_result,
        {
            "generated_at": generated_at,
            "environment": environment,
            "invocation": invocation,
            "result": {"case": args.case, "status": "running"},
        },
    )
    try:
        result = run_case(config, traceback_sink=sys.stderr)
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        error_line = safe_error_line(exc)
        result = CaseResult(
            case=args.case,
            status="failed",
            stages=(StageResult("prerequisites", "failed", error_line),),
            error_type=type(exc).__name__,
            error_line=error_line,
            matched_known_signature=match_known_signature(str(exc)),
        )
    payload = {
        "generated_at": generated_at,
        "environment": environment,
        "invocation": invocation,
        "result": result,
    }
    write_json(args._child_result, payload)
    print(f"[{args.case}] result={result.status}", flush=True)
    print(f"[{args.case}] child_result={args._child_result}", flush=True)
    return 0 if result.status == "passed" else 1


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _child_command(
    args: argparse.Namespace,
    case: str,
    child_result_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--case",
        case,
        "--device",
        args.device,
        "--output-dir",
        str(args.output_dir),
        "--seed",
        str(args.seed),
        "--_child",
        "--_child-result",
        str(child_result_path),
    ]
    if case == "yolov5m":
        command.extend(("--yolov5-path", str(args.yolov5_path)))
    elif case == "patchtst":
        command.extend(("--patchtst-path", str(args.patchtst_path)))
    return command


def run_parent(
    args: argparse.Namespace,
    *,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    timestamp_factory: Callable[[], str] = _timestamp,
) -> int:
    cases = _CASES if args.case == "all" else (args.case,)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exit_codes = []

    for case in cases:
        timestamp = timestamp_factory()
        prefix = args.output_dir / f"{timestamp}-{case}"
        log_path = prefix.with_suffix(".log")
        child_result_path = args.output_dir / f"{timestamp}-{case}.child.json"
        report_path = prefix.with_suffix(".json")
        command = _child_command(args, case, child_result_path)
        print(f"[{case}] log={log_path}", flush=True)
        process = popen_factory(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with log_path.open("w") as log_file:
            if process.stdout is not None:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log_file.write(line)
        exit_code = int(process.wait())
        exit_codes.append(exit_code)
        log_text = log_path.read_text(errors="replace")
        child_payload = None
        if child_result_path.is_file():
            try:
                child_payload = json.loads(child_result_path.read_text())
            except (OSError, UnicodeError, json.JSONDecodeError):
                child_payload = None
        child_status = None
        child_signature = None
        if isinstance(child_payload, dict):
            child_result = child_payload.get("result", {})
            child_status = child_result.get("status")
            child_signature = child_result.get("matched_known_signature")
        log_signature = match_known_signature(log_text)
        terminal_status = (
            "passed"
            if exit_code == 0 and child_status == "passed"
            else "failed"
        )
        report = {
            "case": case,
            "status": terminal_status,
            "exit_code": exit_code,
            "log_path": str(log_path),
            "child_result_path": str(child_result_path),
            "matched_known_signature": log_signature or child_signature,
            "invocation": (
                child_payload.get("invocation")
                if isinstance(child_payload, dict)
                else None
            ),
            "child": child_payload,
        }
        write_json(report_path, report)
        print(f"[{case}] report={report_path}", flush=True)

    return 0 if all(code == 0 for code in exit_codes) else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_child(args) if args._child else run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
