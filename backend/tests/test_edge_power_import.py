"""Selected power archive rows become a separate web-readable result source."""

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.edge_power_import import convert_selected_results


def fixture(tmp_path, *, quality="73.78", service_p99=11.525694090000002):
    archive = tmp_path / "archive"
    case = archive / "edge-power/hailo8/async-runtime-return-w4/resnet50"
    case.mkdir(parents=True)
    with (case / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "run_id", "timestamp", "model_name", "task", "backend", "device", "target_id",
            "batch_size", "warmup_runs", "inference_mode", "scenario", "worker_count",
            "async_run_status", "quality_status", "quality_Top-1 Accuracy", "details_path",
            "async_measurement_boundary", "async_runtime_ready_latency_p50_ms",
            "async_runtime_ready_latency_p95_ms", "async_runtime_ready_latency_p99_ms",
            "async_queue_wait_p99_ms", "async_service_time_p99_ms",
        ])
        writer.writeheader()
        writer.writerow({
            "run_id": "4e4f2b57", "timestamp": "2026-10-16 18:04:00", "model_name": "resnet50",
            "task": "IMAGE_CLASSIFICATION", "backend": "hailort", "device": "device0",
            "target_id": "hailo8", "batch_size": "1", "warmup_runs": "10",
            "inference_mode": "async_queue", "scenario": "offline", "worker_count": "4",
            "async_run_status": "valid", "quality_status": "passed",
            "quality_Top-1 Accuracy": quality,
            "details_path": "resnet50/details/4e4f2b57.json",
            "async_measurement_boundary": "framework_runtime_call_return",
            "async_runtime_ready_latency_p50_ms": "270.7282845",
            "async_runtime_ready_latency_p95_ms": "384.13346005",
            "async_runtime_ready_latency_p99_ms": "392.81324752",
            "async_queue_wait_p99_ms": "383.26921517",
            "async_service_time_p99_ms": "11.525694090000002",
        })
    (case / "details").mkdir()
    (case / "details/4e4f2b57.json").write_text(json.dumps({
        "run_id": "4e4f2b57",
        "async_measurement_boundary": "framework_runtime_call_return",
        "timing_ms": {"service_time": {"count": 5000, "mean": 9.1821112814, "p99": service_p99}},
    }))
    (case / "power-summary.json").write_text(json.dumps({
        "runtime_run_id": "4e4f2b57", "measurement_boundary": "framework_runtime_call_return",
        "completed_samples": 5000, "throughput_samples_per_sec": 391.75297519844855,
        "idle_baseline_W": 10.869248, "inference": {"mean_power_W": 13.963975000309661},
        "delta_mean_W": 3.0947270003096605,
        "incremental_energy_J_per_sample": 0.00789968984598516,
    }))
    (case / "sensor.csv").write_text("timestamp,power_w\n")
    (case / "markers.csv").write_text("timestamp,phase\n")
    selected = tmp_path / "selected-results.csv"
    with selected.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "platform", "mode", "model", "run_id", "samples", "batch_size", "worker_count",
            "measurement_boundary", "throughput_samples_per_sec", "idle_median_w",
            "inference_mean_w", "incremental_mean_w", "incremental_j_per_sample",
            "quality_metric", "quality_value", "power_scope", "source_case", "selected_case",
        ])
        writer.writeheader()
        writer.writerow({
            "platform": "hailo8", "mode": "async-runtime-return-w4", "model": "resnet50",
            "run_id": "4e4f2b57", "samples": "5000", "batch_size": "1", "worker_count": "4",
            "measurement_boundary": "framework_runtime_call_return",
            "throughput_samples_per_sec": "391.75297519844855", "idle_median_w": "10.869248",
            "inference_mean_w": "13.963975000309661", "incremental_mean_w": "3.0947270003096605",
            "incremental_j_per_sample": "0.00789968984598516", "quality_metric": "Top-1 Accuracy",
            "quality_value": "73.78", "power_scope": "jetson_dc_input",
            "source_case": "results/original/resnet50",
            "selected_case": "edge-power/hailo8/async-runtime-return-w4/resnet50",
        })
    return selected, archive


def append_e2e_case(selected: Path, archive: Path) -> None:
    case = archive / "edge-power/deepx/system-e2e/resnet50"
    case.mkdir(parents=True)
    with (case / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "run_id", "timestamp", "model_name", "task", "backend", "device", "target_id",
            "batch_size", "warmup_runs", "inference_mode", "Top-1 Accuracy",
            "Average Latency (ms)", "P99 Latency (ms)",
        ])
        writer.writeheader()
        writer.writerow({
            "run_id": "e2e-fixture", "timestamp": "2026-10-16 19:00:00", "model_name": "resnet50",
            "task": "IMAGE_CLASSIFICATION", "backend": "deepx", "device": "npu0",
            "target_id": "deepx", "batch_size": "1", "warmup_runs": "10",
            "inference_mode": "e2e", "Top-1 Accuracy": "74.43",
            "Average Latency (ms)": "3.373877753125271",
            "P99 Latency (ms)": "3.826147468789715",
        })
    (case / "power-summary.json").write_text(json.dumps({
        "measurement_boundary": "system_e2e_after_warmup_including_evaluator_compute",
        "completed_samples": 5000, "throughput_samples_per_sec": 296.3948527991821,
        "idle_median_w": 10.0, "inference": {"mean_w": 13.0},
        "incremental_mean_w": 3.0, "incremental_j_per_sample": 0.010121633,
    }))
    (case / "sensor.csv").write_text("timestamp,power_w\n")
    (case / "markers.csv").write_text("timestamp,phase\n")
    with selected.open("a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "deepx", "system-e2e", "resnet50", "e2e-fixture", "5000", "1", "",
            "system_e2e_after_warmup_including_evaluator_compute", "296.3948527991821",
            "10.0", "13.0", "3.0", "0.010121633", "Top-1 Accuracy", "74.43",
            "jetson_dc_input", "results/original/resnet50",
            "edge-power/deepx/system-e2e/resnet50",
        ])


def test_converts_verified_power_row_into_web_result_without_mixing_quality_and_throughput(tmp_path):
    selected, archive = fixture(tmp_path)
    output = tmp_path / "web-results.csv"

    assert convert_selected_results(selected, archive, output) == 1
    with output.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["run_id"] == "4e4f2b57"
    assert row["model_name"] == "resnet50"
    assert row["target_id"] == "hailo8"
    assert row["inference_mode"] == "async_queue"
    assert row["result_dataset"] == "edge_power_selected"
    assert row["measurement_boundary"] == "framework_runtime_call_return"
    assert row["throughput_samples_per_sec"] == "391.75297519844855"
    assert row["incremental_j_per_sample"] == "0.00789968984598516"
    assert row["Top-1 Accuracy"] == "73.78"
    assert "quality_value" not in row
    assert row["power_scope"] == "jetson_dc_input"
    assert row["selected_case"] == "edge-power/hailo8/async-runtime-return-w4/resnet50"
    assert row["async_runtime_ready_latency_p50_ms"] == "270.7282845"
    assert row["async_runtime_ready_latency_p95_ms"] == "384.13346005"
    assert row["async_runtime_ready_latency_p99_ms"] == "392.81324752"
    assert row["async_queue_wait_p99_ms"] == "383.26921517"
    assert row["async_service_time_p99_ms"] == "11.525694090000002"
    assert row["async_service_time_mean_ms"] == "9.1821112814"
    assert row["latency_boundary"] == "worker_executor_execute"


def test_rejects_quality_mismatch_without_publishing_csv(tmp_path):
    selected, archive = fixture(tmp_path, quality="70.00")
    output = tmp_path / "web-results.csv"

    with pytest.raises(ValueError, match="quality"):
        convert_selected_results(selected, archive, output)
    assert not output.exists()


def test_rejects_async_service_time_p99_mismatch_without_publishing_csv(tmp_path):
    selected, archive = fixture(tmp_path, service_p99=12.0)
    output = tmp_path / "web-results.csv"

    with pytest.raises(ValueError, match="service_time p99"):
        convert_selected_results(selected, archive, output)
    assert not output.exists()


def test_web_selection_keeps_both_modes_and_their_distinct_latencies(tmp_path):
    selected, archive = fixture(tmp_path)
    append_e2e_case(selected, archive)
    output = tmp_path / "web-results.csv"

    assert convert_selected_results(selected, archive, output) == 2
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    by_mode = {row["mode"]: row for row in rows}
    assert by_mode["async-runtime-return-w4"]["worker_count"] == "4"
    assert by_mode["async-runtime-return-w4"]["latency_boundary"] == "worker_executor_execute"
    assert by_mode["async-runtime-return-w4"]["async_runtime_ready_latency_p50_ms"] == "270.7282845"
    assert by_mode["async-runtime-return-w4"]["async_runtime_ready_latency_p99_ms"] == "392.81324752"
    assert by_mode["system-e2e"]["run_id"] == "e2e-fixture"
    assert by_mode["system-e2e"]["latency_boundary"] == "runtime_call"
    assert by_mode["system-e2e"]["Average Latency (ms)"] == "3.373877753125271"
    assert by_mode["system-e2e"]["P99 Latency (ms)"] == "3.826147468789715"
