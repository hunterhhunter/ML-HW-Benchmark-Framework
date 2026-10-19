"""Selected power archive rows become a separate web-readable result source."""

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.edge_power_import import convert_selected_results


def fixture(tmp_path, *, quality="73.78"):
    archive = tmp_path / "archive"
    case = archive / "edge-power/hailo8/async-runtime-return-w4/resnet50"
    case.mkdir(parents=True)
    with (case / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "run_id", "timestamp", "model_name", "task", "backend", "device", "target_id",
            "batch_size", "warmup_runs", "inference_mode", "scenario", "worker_count",
            "async_run_status", "quality_status", "quality_Top-1 Accuracy",
            "async_measurement_boundary",
        ])
        writer.writeheader()
        writer.writerow({
            "run_id": "4e4f2b57", "timestamp": "2026-10-16 18:04:00", "model_name": "resnet50",
            "task": "IMAGE_CLASSIFICATION", "backend": "hailort", "device": "device0",
            "target_id": "hailo8", "batch_size": "1", "warmup_runs": "10",
            "inference_mode": "async_queue", "scenario": "offline", "worker_count": "4",
            "async_run_status": "valid", "quality_status": "passed",
            "quality_Top-1 Accuracy": quality,
            "async_measurement_boundary": "framework_runtime_call_return",
        })
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


def test_rejects_quality_mismatch_without_publishing_csv(tmp_path):
    selected, archive = fixture(tmp_path, quality="70.00")
    output = tmp_path / "web-results.csv"

    with pytest.raises(ValueError, match="quality"):
        convert_selected_results(selected, archive, output)
    assert not output.exists()
