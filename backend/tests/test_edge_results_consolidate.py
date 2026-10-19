"""A single web CSV keeps selected power runs and older report runs distinct."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.edge_results_consolidate import consolidate_edge_results


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def check_consolidates_distinct_runs_without_attaching_report_quality_or_power(tmp_path):
    power = tmp_path / "selected-web.csv"
    report = tmp_path / "reported-values.csv"
    output = tmp_path / "benchmark_results.csv"
    write_csv(power, [
        "run_id", "model_name", "target_id", "inference_mode", "result_dataset",
        "quality_metric", "Top-1 Accuracy", "throughput_samples_per_sec",
        "incremental_mean_w", "selection_sha256",
    ], [{
        "run_id": "power-run", "model_name": "resnet50", "target_id": "hailo8",
        "inference_mode": "e2e", "result_dataset": "edge_power_selected",
        "quality_metric": "Top-1 Accuracy", "Top-1 Accuracy": "73.78",
        "throughput_samples_per_sec": "240", "incremental_mean_w": "3.1",
        "selection_sha256": "selected-digest",
    }])
    write_csv(report, [
        "model", "npu", "quality_metric", "quality_value",
        "quality_secondary_metric", "quality_secondary_value",
        "sync_runtime_mean_ms", "sync_runtime_p99_ms",
        "sync_runtime_inverse_samples_per_sec", "async_service_mean_ms",
        "async_service_p99_ms", "async_runtime_completed_samples_per_sec",
        "sync_run_id", "async_run_id", "verification_status",
    ], [{
        "model": "resnet50", "npu": "mobilint-regulus", "quality_metric": "Top-1 %",
        "quality_value": "77.42", "quality_secondary_metric": "Top-5 %",
        "quality_secondary_value": "93.82", "sync_runtime_mean_ms": "4.563",
        "sync_runtime_p99_ms": "4.678", "sync_runtime_inverse_samples_per_sec": "219.172",
        "async_service_mean_ms": "27.330", "async_service_p99_ms": "53.360",
        "async_runtime_completed_samples_per_sec": "77.787",
        "sync_run_id": "sync-run", "async_run_id": "async-run",
        "verification_status": "verified_separate_quality_pass",
    }])

    assert consolidate_edge_results(power, report, output) == 3
    with output.open(newline="") as handle:
        rows = {row["run_id"]: row for row in csv.DictReader(handle)}
    assert set(rows) == {"power-run", "sync-run", "async-run"}
    assert rows["power-run"]["Top-1 Accuracy"] == "73.78"
    assert rows["power-run"]["incremental_mean_w"] == "3.1"
    assert rows["sync-run"]["result_dataset"] == "edge_report_5_5"
    assert rows["sync-run"]["Average Latency (ms)"] == "4.563"
    assert rows["sync-run"]["sync_runtime_inverse_samples_per_sec"] == "219.172"
    assert rows["sync-run"]["throughput_samples_per_sec"] == ""
    assert rows["async-run"]["Average Latency (ms)"] == "27.330"
    assert rows["async-run"]["throughput_samples_per_sec"] == "77.787"
    assert rows["async-run"]["incremental_mean_w"] == ""
    assert rows["async-run"]["Top-1 Accuracy"] == ""
    assert rows["async-run"]["reported_Top-1 %"] == "77.42"
    assert rows["async-run"]["reported_Top-5 %"] == "93.82"
    assert "reported_quality_value" not in rows["async-run"]
    assert rows["async-run"]["verification_status"] == "verified_separate_quality_pass"


def check_rejects_colliding_run_ids_without_replacing_existing_output(tmp_path):
    power = tmp_path / "selected-web.csv"
    report = tmp_path / "reported-values.csv"
    output = tmp_path / "benchmark_results.csv"
    output.write_text("original\n")
    write_csv(power, ["run_id", "result_dataset"], [
        {"run_id": "duplicate", "result_dataset": "edge_power_selected"}])
    report_columns = [
        "model", "npu", "quality_metric", "quality_value",
        "quality_secondary_metric", "quality_secondary_value",
        "sync_runtime_mean_ms", "sync_runtime_p99_ms",
        "sync_runtime_inverse_samples_per_sec", "async_service_mean_ms",
        "async_service_p99_ms", "async_runtime_completed_samples_per_sec",
        "sync_run_id", "async_run_id", "verification_status",
    ]
    write_csv(report, report_columns, [{
        "sync_run_id": "duplicate", "async_run_id": "async-run",
        "model": "resnet50", "npu": "hailo8",
    }])

    try:
        consolidate_edge_results(power, report, output)
    except ValueError as exc:
        assert "duplicate run ID" in str(exc)
    else:
        raise AssertionError("duplicate run ID was accepted")
    assert output.read_text() == "original\n"


class EdgeResultsConsolidateTest(unittest.TestCase):
    def test_keeps_report_metrics_separate_from_power_run(self):
        with tempfile.TemporaryDirectory() as directory:
            check_consolidates_distinct_runs_without_attaching_report_quality_or_power(Path(directory))

    def test_rejects_run_id_collision_without_replacing_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            check_rejects_colliding_run_ids_without_replacing_existing_output(Path(directory))
