"""Convert a verified selected edge-power archive to the web result CSV schema."""

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


BASE_COLUMNS = [
    "run_id", "timestamp", "model_name", "task", "backend", "device", "target_id",
    "batch_size", "warmup_runs", "inference_mode", "scenario", "worker_count",
    "async_run_status", "quality_status", "comparison_eligible", "result_dataset",
    "mode", "samples_processed", "measurement_boundary", "latency_boundary",
    "Average Latency (ms)", "P99 Latency (ms)",
    "async_runtime_ready_latency_p50_ms", "async_runtime_ready_latency_p99_ms",
    "throughput_samples_per_sec",
    "idle_median_w", "inference_mean_w", "incremental_mean_w",
    "incremental_j_per_sample", "quality_metric", "power_scope",
    "power_evidence_status", "source_case", "selected_case", "selection_sha256",
]
POWER_KEYS = (
    "throughput_samples_per_sec", "idle_median_w", "inference_mean_w",
    "incremental_mean_w", "incremental_j_per_sample",
)
LATENCY_KEYS = ("Average Latency (ms)", "P99 Latency (ms)")
ASYNC_LATENCY_KEYS = (
    "async_runtime_ready_latency_p50_ms", "async_runtime_ready_latency_p99_ms",
)
MODES = ("system-e2e", "async-runtime-return-w4")


def _equal(actual: str, expected: object) -> bool:
    try:
        return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-10)
    except (TypeError, ValueError):
        return False


def _archive_case(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"selected_case is outside archive root: {relative}")
    for name in ("results.csv", "power-summary.json", "sensor.csv", "markers.csv"):
        if not (path / name).is_file():
            raise ValueError(f"archive evidence missing: {relative}/{name}")
    return path


def _convert_row(selected: dict[str, str], archive_root: Path, digest: str) -> dict[str, str]:
    run_id = selected["run_id"]
    case = _archive_case(archive_root, selected["selected_case"])
    with (case / "results.csv").open(newline="") as handle:
        matches = [row for row in csv.DictReader(handle) if row.get("run_id") == run_id]
    if len(matches) != 1:
        raise ValueError(f"framework results row mismatch: {run_id}")
    framework = matches[0]
    summary = json.loads((case / "power-summary.json").read_text())
    async_mode = selected["mode"].startswith("async-runtime-return-")
    if selected["mode"] not in MODES:
        raise ValueError(f"unsupported measurement mode: {selected['mode']}")
    expected_mode = "async_queue" if async_mode else "e2e"
    for summary_key, framework_key in (
        ("model", "model_name"), ("platform", "target_id"),
        ("batch_size", "batch_size"),
    ):
        if selected[summary_key] != framework.get(framework_key):
            raise ValueError(f"framework {framework_key} mismatch: {run_id}")
    if framework.get("inference_mode") != expected_mode:
        raise ValueError(f"framework inference_mode mismatch: {run_id}")
    if async_mode and framework.get("worker_count") != selected["worker_count"]:
        raise ValueError(f"framework worker_count mismatch: {run_id}")
    if async_mode and framework.get("async_measurement_boundary") != selected["measurement_boundary"]:
        raise ValueError(f"framework measurement_boundary mismatch: {run_id}")
    if summary.get("measurement_boundary") != selected["measurement_boundary"]:
        raise ValueError(f"power measurement_boundary mismatch: {run_id}")
    if not _equal(selected["samples"], summary.get("completed_samples")):
        raise ValueError(f"completed samples mismatch: {run_id}")

    power = {
        "throughput_samples_per_sec": summary["throughput_samples_per_sec"],
        "idle_median_w": summary["idle_baseline_W"] if async_mode else summary["idle_median_w"],
        "inference_mean_w": summary["inference"]["mean_power_W" if async_mode else "mean_w"],
        "incremental_mean_w": summary["delta_mean_W"] if async_mode else summary["incremental_mean_w"],
        "incremental_j_per_sample": summary["incremental_energy_J_per_sample"] if async_mode else summary["incremental_j_per_sample"],
    }
    for key in POWER_KEYS:
        if not _equal(selected[key], power[key]):
            raise ValueError(f"power {key} mismatch: {run_id}")
    quality_key = selected["quality_metric"]
    original_quality_key = f"quality_{quality_key}" if async_mode else quality_key
    if not _equal(selected["quality_value"], framework.get(original_quality_key)):
        raise ValueError(f"quality mismatch: {run_id}")
    if selected["power_scope"] != "jetson_dc_input":
        raise ValueError(f"unsupported power scope: {run_id}")

    latency = {}
    for key in ASYNC_LATENCY_KEYS if async_mode else LATENCY_KEYS:
        value = framework.get(key, "")
        try:
            valid = math.isfinite(float(value)) and float(value) >= 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f"framework {key} missing or invalid: {run_id}")
        latency[key] = value

    return {
        **{key: framework.get(key, "") for key in (
            "run_id", "timestamp", "model_name", "task", "backend", "device", "target_id",
            "batch_size", "warmup_runs", "inference_mode", "scenario", "worker_count",
            "async_run_status", "quality_status", "comparison_eligible",
        )},
        "result_dataset": "edge_power_selected",
        "mode": selected["mode"],
        "samples_processed": selected["samples"],
        "measurement_boundary": selected["measurement_boundary"],
        "latency_boundary": "request_issued_to_runtime_return" if async_mode else "runtime_call",
        **latency,
        **{key: selected[key] for key in POWER_KEYS},
        "quality_metric": quality_key,
        quality_key: selected["quality_value"],
        "power_scope": selected["power_scope"],
        "power_evidence_status": "archived",
        "source_case": selected["source_case"],
        "selected_case": selected["selected_case"],
        "selection_sha256": digest,
    }


def convert_selected_results(selected_csv: Path, archive_root: Path, output_csv: Path) -> int:
    """Validate chosen archive rows and publish a separate web result CSV."""
    selected_csv, archive_root, output_csv = map(Path, (selected_csv, archive_root, output_csv))
    digest = hashlib.sha256(selected_csv.read_bytes()).hexdigest()
    with selected_csv.open(newline="") as handle:
        selected = list(csv.DictReader(handle))
    if not selected:
        raise ValueError("no selected results")
    run_ids = [row["run_id"] for row in selected]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("selected results contain duplicate run IDs")
    converted = [_convert_row(row, archive_root, digest) for row in selected]
    quality_columns = sorted({row["quality_metric"] for row in converted})
    columns = BASE_COLUMNS + quality_columns
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", newline="", dir=output_csv.parent,
                                         prefix=f".{output_csv.name}.", suffix=".tmp",
                                         delete=False) as handle:
            temp_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(converted)
        os.replace(temp_name, output_csv)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
    return len(converted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selected_csv", type=Path)
    parser.add_argument("archive_root", type=Path,
                        help="Directory containing the selected_case paths")
    parser.add_argument("output_csv", type=Path,
                        help="Destination within WEB_RESULTS_ROOTS")
    args = parser.parse_args()
    print(f"Converted {convert_selected_results(args.selected_csv, args.archive_root, args.output_csv)} selected results")


if __name__ == "__main__":
    main()
