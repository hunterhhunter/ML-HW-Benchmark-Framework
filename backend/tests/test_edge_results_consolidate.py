"""A single web CSV keeps selected power runs and older report runs distinct."""

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.edge_results_consolidate import REPORT_COLUMNS, consolidate_edge_results


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
    }, {
        "run_id": "power-async", "model_name": "resnet50", "target_id": "hailo8",
        "inference_mode": "async_queue", "result_dataset": "edge_power_selected",
        "quality_metric": "Top-1 Accuracy", "Top-1 Accuracy": "73.78",
        "throughput_samples_per_sec": "390", "incremental_mean_w": "3.2",
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
        "model": "resnet50", "npu": "hailo8", "quality_metric": "Top-1 %",
        "quality_value": "73.78", "quality_secondary_metric": "Top-5 %",
        "quality_secondary_value": "90.68", "sync_runtime_mean_ms": "3.914",
        "sync_runtime_p99_ms": "4.174", "sync_runtime_inverse_samples_per_sec": "255.464",
        "async_service_mean_ms": "9.222", "async_service_p99_ms": "11.814",
        "async_runtime_completed_samples_per_sec": "389.605",
        "sync_run_id": "older-hailo-sync", "async_run_id": "older-hailo-async",
        "verification_status": "verified",
    }, {
        "model": "resnet50", "npu": "mobilint-regulus", "quality_metric": "Top-1 %",
        "quality_value": "77.42", "quality_secondary_metric": "Top-5 %",
        "quality_secondary_value": "93.82", "sync_runtime_mean_ms": "4.563",
        "sync_runtime_p99_ms": "4.678", "sync_runtime_inverse_samples_per_sec": "219.172",
        "async_service_mean_ms": "27.330", "async_service_p99_ms": "53.360",
        "async_runtime_completed_samples_per_sec": "77.787",
        "sync_run_id": "sync-run", "async_run_id": "async-run",
        "verification_status": "verified_separate_quality_pass",
    }])

    assert consolidate_edge_results(power, report, output) == 4
    with output.open(newline="") as handle:
        rows = {row["run_id"]: row for row in csv.DictReader(handle)}
    assert set(rows) == {"power-run", "power-async", "sync-run", "async-run"}
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
    write_csv(power, ["run_id", "model_name", "target_id", "inference_mode", "result_dataset"], [
        {"run_id": "duplicate", "model_name": "resnet50", "target_id": "hailo8",
         "inference_mode": "e2e", "result_dataset": "edge_power_selected"}])
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
        "model": "resnet50", "npu": "mobilint-regulus",
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

    def test_required_grid_rejects_missing_combination_without_replacing_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            power = root / "selected-web.csv"
            report = root / "reported-values.csv"
            output = root / "benchmark_results.csv"
            output.write_text("original\n")
            write_csv(power, ["run_id", "model_name", "target_id", "inference_mode", "result_dataset"], [{
                "run_id": "power", "model_name": "resnet50", "target_id": "hailo8",
                "inference_mode": "e2e", "result_dataset": "edge_power_selected",
            }])
            write_csv(report, [
                "model", "npu", "quality_metric", "quality_value",
                "quality_secondary_metric", "quality_secondary_value",
                "sync_runtime_mean_ms", "sync_runtime_p99_ms",
                "sync_runtime_inverse_samples_per_sec", "async_service_mean_ms",
                "async_service_p99_ms", "async_runtime_completed_samples_per_sec",
                "sync_run_id", "async_run_id", "verification_status",
            ], [{"model": "resnet50", "npu": "hailo8",
                 "sync_run_id": "old-sync", "async_run_id": "old-async"}])
            expected = {
                ("resnet50", "hailo8", "e2e"),
                ("resnet50", "hailo8", "async_queue"),
                ("resnet50", "mobilint-regulus", "e2e"),
            }
            with self.assertRaisesRegex(ValueError, "missing combinations"):
                consolidate_edge_results(power, report, output, required_combinations=expected)
            self.assertEqual(output.read_text(), "original\n")

    def test_regulus_original_evidence_replaces_report_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)
            report_dir = archive / "edge-report-5.5"
            report_dir.mkdir()
            power = archive / "selected-web.csv"
            report = report_dir / "reported-values.csv"
            output = archive / "consolidated.csv"
            sync_root = report_dir / "evidence" / "sync" / "mobilint-regulus"
            async_root = report_dir / "evidence" / "async"
            write_csv(power, ["run_id", "model_name", "target_id", "inference_mode", "result_dataset"], [{
                "run_id": "power", "model_name": "resnet50", "target_id": "hailo8",
                "inference_mode": "e2e", "result_dataset": "edge_power_selected",
            }])
            write_csv(report, list(REPORT_COLUMNS), [{"model": "resnet50", "npu": "mobilint-regulus",
                                                    "sync_run_id": "stale-sync", "async_run_id": "stale-async"}])
            for model, task, quality_key, quality in (
                ("resnet50", "IMAGE_CLASSIFICATION", "Top-1 Accuracy", "77.42"),
                ("yolov5m", "OBJECT_DETECTION", "COCO AP50", "0.538"),
                ("yolov8s-pose", "POSE_ESTIMATION", "OKS mAP", "0.552"),
            ):
                sync_dir = sync_root / model
                async_dir = async_root / model
                (sync_dir / "timings").mkdir(parents=True)
                (async_dir / "details").mkdir(parents=True)
                sync_id = f"mobilint-regulus-{model}-seed-abcdef123456"
                sync_csv = sync_dir / "run.csv"
                sync_row = {
                    "run_id": sync_id, "timestamp": "2026-09-27", "model_name": model,
                    "task": task, "backend": "vendor-native", "device": "mobilint-regulus",
                    "batch_size": "1", "warmup_runs": "10", "target_id": "mobilint-regulus",
                    "inference_mode": "vendor_native_e2e", "measurement_sample_count": "2",
                    "synchronous_timing_sample_count": "2", "Total Samples": "2",
                    "System E2E Measured Samples": "2", "System E2E Average Latency (ms)": "25.5",
                    "System E2E P99 Latency (ms)": "30.5",
                    "Average Latency (ms)": "4.5", "P99 Latency (ms)": "5.5",
                    "System E2E Throughput (samples/s)": "39.2156862745",
                    quality_key: quality,
                }
                if model == "resnet50":
                    sync_row["Top-5 Accuracy"] = "93.82"
                write_csv(sync_csv, list(sync_row), [sync_row])
                write_csv(sync_dir / "timings" / f"{sync_id}.samples.csv", ["sample"],
                          [{"sample": "1"}, {"sample": "2"}])
                async_id = f"{model}-async"
                async_row = {
                    "run_id": async_id, "timestamp": "2026-10-01", "model_name": model,
                    "target_id": "mobilint-regulus", "backend": "mobilint", "device": "0",
                    "batch_size": "1", "warmup_runs": "10", "inference_mode": "async_queue",
                    "scenario": "offline", "worker_count": "4", "async_run_status": "valid",
                    "async_measurement_boundary": "framework_runtime_call_return",
                    "async_submitted_requests": "2", "async_completed_requests": "2",
                    "async_completed_samples": "2", "async_runtime_completed_samples": "2",
                    "async_failed_requests": "0", "async_rejected_requests": "0",
                    "async_timed_out_requests": "0", "async_runtime_completed_samples_per_sec": "80.5",
                    "quality_status": "skipped",
                }
                write_csv(async_dir / "results.csv", list(async_row), [async_row])
                (async_dir / "details" / f"{async_id}.json").write_text(json.dumps({
                    "run_id": async_id, "status": "valid", "timing_ms": {
                        "service_time": {"count": 2, "mean": 12.5, "p99": 18.5},
                    },
                }))
            count = consolidate_edge_results(power, report, output,
                                             regulus_e2e_root=sync_root, regulus_async_root=async_root)
            self.assertEqual(count, 7)
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 7)
            self.assertNotIn("stale-sync", {row["run_id"] for row in rows})
            self.assertNotIn("stale-async", {row["run_id"] for row in rows})
            sync = next(row for row in rows if row["model_name"] == "resnet50" and
                        row["target_id"] == "mobilint-regulus" and row["inference_mode"] == "e2e")
            async_run = next(row for row in rows if row["model_name"] == "resnet50" and
                             row["target_id"] == "mobilint-regulus" and row["inference_mode"] == "async_queue")
            self.assertEqual(sync["samples_processed"], "2")
            self.assertEqual(sync["Average Latency (ms)"], "4.5")
            self.assertEqual(sync["P99 Latency (ms)"], "5.5")
            self.assertEqual(sync["latency_boundary"], "runtime_call")
            self.assertEqual(sync["throughput_samples_per_sec"], "39.2156862745")
            self.assertEqual(sync["source_csv_sha256"], hashlib.sha256(
                (sync_root / "resnet50" / "run.csv").read_bytes()).hexdigest())
            self.assertEqual(async_run["samples_processed"], "2")
            self.assertEqual(async_run["Average Latency (ms)"], "12.5")
            self.assertEqual(async_run["throughput_samples_per_sec"], "80.5")
            self.assertEqual(async_run["quality_status"], "skipped")
            self.assertEqual(async_run["quality_run_id"], sync["run_id"])
            self.assertFalse(any(column.startswith("reported_") for column in rows[0]))
            for model, metric, value in (
                ("resnet50", "Top-1 Accuracy", "77.42"),
                ("yolov5m", "mAP@0.5", "0.538"),
                ("yolov8s-pose", "OKS mAP", "0.552"),
            ):
                for mode in ("e2e", "async_queue"):
                    row = next(item for item in rows if item["model_name"] == model and
                               item["target_id"] == "mobilint-regulus" and
                               item["inference_mode"] == mode)
                    self.assertEqual(row["quality_metric"], metric)
                    self.assertEqual(row[metric], value)
                    self.assertEqual(row["quality_status"], "measured" if mode == "e2e" else "skipped")
            self.assertEqual(sync["Top-5 Accuracy"], "93.82")
            self.assertNotIn("source_quality_metric", sync)
            yolov5m = next(item for item in rows if item["model_name"] == "yolov5m" and
                            item["target_id"] == "mobilint-regulus" and item["inference_mode"] == "e2e")
            self.assertEqual(yolov5m["mAP@0.5"], "0.538")
