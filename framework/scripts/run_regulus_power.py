#!/usr/bin/env python3
"""Run an existing REGULUS benchmark matrix with external Jetson power.

Only this script is added to the benchmark host. It captures idle before
model/data loading, then injects a monitor-shaped object at the framework's
existing warmup/measurement lifecycle boundary;
src/main.py and its runners remain untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from regulus_power_link import RemotePowerSession


def _option(args: list[str], name: str) -> str | None:
    if name not in args:
        return None
    index = args.index(name)
    if index + 1 >= len(args):
        raise ValueError(f"{name} requires a value")
    return args[index + 1]


def _validate_case(case: dict) -> None:
    if not isinstance(case, dict) or not isinstance(case.get("args"), list):
        raise ValueError("case must have an args list")
    args = case["args"]
    if not all(isinstance(item, str) for item in args):
        raise ValueError("case args must be strings")
    for key in ("name", "model", "mode"):
        if not isinstance(case.get(key), str) or not case[key]:
            raise ValueError(f"case {key} is required")
    if case["mode"] not in {"e2e", "async_queue"}:
        raise ValueError("case mode must be e2e or async_queue")
    expected = case.get("expected_samples")
    if type(expected) is not int or expected <= 0:
        raise ValueError("expected_samples must be positive")
    if _option(args, "--target") != "mobilint-regulus":
        raise ValueError("power suite requires --target mobilint-regulus")
    if _option(args, "--model") != case["model"]:
        raise ValueError("case model does not match benchmark arguments")
    if (_option(args, "--inference-mode") or "e2e") != case["mode"]:
        raise ValueError("case mode does not match benchmark arguments")
    if any(flag in args for flag in ("--power-trace", "--monitor", "--results-path")):
        raise ValueError("power suite owns monitoring and results path")
    count_flag = "--max-steps" if case["mode"] == "e2e" else "--max-samples"
    if _option(args, count_flag) != str(expected):
        raise ValueError(f"case must set {count_flag}={expected}")
    if case["mode"] == "async_queue" and _option(args, "--async-pass") != "runtime-only":
        raise ValueError("async power pass must be runtime-only")


def _result_row(results_path: Path) -> dict:
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1 or not rows[0].get("run_id"):
        raise ValueError("expected exactly one framework result with run_id")
    return rows[0]


def _completed_count(row: dict, mode: str) -> int:
    if mode == "e2e":
        return int(row["Total Samples"])
    if row.get("async_run_status") != "valid":
        raise ValueError("async framework run is not valid")
    completed = int(row["async_runtime_completed_samples"])
    if completed != int(row["async_completed_samples"]):
        raise ValueError("async runtime and lifecycle counts disagree")
    return completed


class _FrameworkRemoteMonitor:
    def __init__(self, session: RemotePowerSession, case: dict):
        self.session = session
        self.case = case

    def record_after_load_vram(self) -> None:
        pass

    def startup_timeout_hint_sec(self) -> float:
        return self.session.startup_timeout_hint_sec()

    def start(self) -> None:
        self.session.start(
            model=self.case["model"], mode=self.case["mode"],
            expected_samples=self.case["expected_samples"],
        )

    def stop(self) -> None:
        self.session.stop()

    def summary(self) -> dict:
        return {}


def execute_framework_case(
    *, case: dict, main_module, case_dir: Path,
    jetson_host: str, jetson_port: int, baseline_sec: float,
    framework_root: Path | None = None,
) -> dict:
    """Execute one original main() run with only a process-local monitor hook."""
    _validate_case(case)
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    results_path = case_dir / "results.csv"
    if results_path.exists() or (case_dir / "power-summary.json").exists():
        raise FileExistsError(f"case output already exists: {case_dir}")
    measurement_id = f"{case['name']}-{uuid.uuid4().hex[:12]}"
    session = RemotePowerSession(
        host=jetson_host, port=jetson_port, measurement_id=measurement_id,
        baseline_sec=baseline_sec, output_dir=case_dir,
    )
    monitor = _FrameworkRemoteMonitor(session, case)
    modern_hook = hasattr(main_module, "_create_requested_hw_monitor")
    legacy_monitors = None if modern_hook else importlib.import_module("monitors")
    original_factory = (
        main_module._create_requested_hw_monitor if modern_hook
        else legacy_monitors.create_hw_monitor
    )
    original_argv = sys.argv
    original_cwd = Path.cwd()
    framework_root = Path(framework_root) if framework_root is not None else original_cwd

    def create_monitor(args, target):
        if target.target_id != "mobilint-regulus":
            raise ValueError("remote power monitor attached to wrong target")
        return monitor

    if modern_hook:
        main_module._create_requested_hw_monitor = create_monitor
    else:
        legacy_monitors.create_hw_monitor = lambda **kwargs: monitor
    sys.argv = [str(framework_root / "src" / "main.py"), *case["args"],
                *(["--monitor"] if not modern_hook else []),
                "--results-path", str(results_path)]
    exit_code = None
    try:
        session.prepare(
            model=case["model"], mode=case["mode"],
            expected_samples=case["expected_samples"],
        )
        os.chdir(framework_root)
        exit_code = main_module.main()
    except BaseException:
        if session._started and not session._stopped:
            try:
                session.stop()
            except Exception:
                pass
        if session._stopped:
            try:
                session.finalize(completed_samples=0, run_status="failed")
            except Exception:
                pass
        elif session._prepared and not session._started:
            try:
                session.abort()
            except Exception:
                pass
        raise
    finally:
        os.chdir(original_cwd)
        sys.argv = original_argv
        if modern_hook:
            main_module._create_requested_hw_monitor = original_factory
        else:
            legacy_monitors.create_hw_monitor = original_factory

    if exit_code not in (None, 0):
        if session._stopped:
            session.finalize(completed_samples=0, run_status="failed")
        raise RuntimeError(f"framework returned {exit_code}")
    row = _result_row(results_path)
    try:
        completed = _completed_count(row, case["mode"])
    except (KeyError, ValueError):
        if session._stopped:
            session.finalize(completed_samples=0, run_status="failed")
        raise
    run_status = "valid" if completed == case["expected_samples"] else "invalid"
    artifact = session.finalize(completed_samples=completed, run_status=run_status)
    artifact["framework_run_id"] = row["run_id"]
    artifact["framework_results_sha256"] = hashlib.sha256(results_path.read_bytes()).hexdigest()
    artifact["benchmark_args"] = case["args"]
    (case_dir / "power-summary.json").write_text(json.dumps(artifact, indent=2) + "\n")
    if artifact["status"] != "complete":
        raise RuntimeError(f"power measurement {artifact['status']} for {case['name']}")
    return artifact


def _import_framework_main(framework_root: Path):
    source_root = framework_root / "src"
    if not (source_root / "main.py").is_file():
        raise FileNotFoundError(source_root / "main.py")
    sys.path.insert(0, str(source_root))
    spec = importlib.util.spec_from_file_location("regulus_original_main", source_root / "main.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config.get("cases"), list) or not config["cases"]:
        raise ValueError("config must contain cases")
    for case in config["cases"]:
        _validate_case(case)
    names = [case["name"] for case in config["cases"]]
    if len(set(names)) != len(names):
        raise ValueError("case names must be unique")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = _load_config(args.config)
    framework_root = Path(config["framework_root"]).expanduser().resolve()
    output_root = Path(config["output_root"]).expanduser().resolve()
    if args.dry_run:
        for case in config["cases"]:
            print(case["name"], sys.executable, "src/main.py", *case["args"])
        return 0
    output_root.mkdir(parents=True, exist_ok=True)
    if args.case_index is not None:
        case = config["cases"][args.case_index]
        module = _import_framework_main(framework_root)
        artifact = execute_framework_case(
            case=case, main_module=module, case_dir=output_root / case["name"],
            jetson_host=config["jetson_host"], jetson_port=int(config["jetson_port"]),
            baseline_sec=float(config.get("baseline_sec", 60)),
            framework_root=framework_root,
        )
        print(json.dumps({"case": case["name"], "run_id": artifact["framework_run_id"],
                          "power_status": artifact["status"],
                          "incremental_j_per_sample": artifact["analysis"]["incremental_j_per_sample"]}))
        return 0

    summaries = []
    for index, case in enumerate(config["cases"]):
        case_dir = output_root / case["name"]
        case_dir.mkdir(exist_ok=False)
        command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config.resolve()),
                   "--case-index", str(index)]
        print(f"[REGULUS power] running {case['name']} ({index + 1}/{len(config['cases'])})", flush=True)
        with (case_dir / "run.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=framework_root, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"case {case['name']} failed with exit code {process.returncode}")
        summaries.append(json.loads((case_dir / "power-summary.json").read_text()))
    (output_root / "suite-summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
