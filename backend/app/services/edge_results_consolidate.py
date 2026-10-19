"""Combine selected edge power runs and separate historical report runs for the web CSV."""

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

from .edge_power_import import convert_selected_results


REPORT_COLUMNS = (
    "model", "npu", "quality_metric", "quality_value",
    "quality_secondary_metric", "quality_secondary_value",
    "sync_runtime_mean_ms", "sync_runtime_p99_ms",
    "sync_runtime_inverse_samples_per_sec", "async_service_mean_ms",
    "async_service_p99_ms", "async_runtime_completed_samples_per_sec",
    "sync_run_id", "async_run_id", "verification_status",
)
EXTRA_COLUMNS = (
    "source_csv", "source_csv_sha256", "verification_status", "throughput_kind",
    "sync_runtime_inverse_samples_per_sec",
)
EXPECTED_COMBINATIONS = frozenset(
    (model, target, mode)
    for model in ("resnet50", "yolov5m", "yolov8s-pose")
    for target in ("hailo8", "deepx", "hailo10h", "mobilint-regulus")
    for mode in ("e2e", "async_queue")
)
REGULUS_MODELS = ("resnet50", "yolov5m", "yolov8s-pose")
REGULUS_QUALITY = {
    "resnet50": ("Top-1 Accuracy", "Top-1 Accuracy", "Top-5 Accuracy", "Top-5 Accuracy"),
    "yolov5m": ("mAP@0.5", "COCO AP50", "", ""),
    "yolov8s-pose": ("OKS mAP", "OKS mAP", "", ""),
}


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)
    if not rows or any(None in row for row in rows):
        raise ValueError(f"empty or malformed CSV: {path}")
    return columns, rows


def _report_row(source: dict[str, str], mode: str, digest: str, source_csv: str) -> dict[str, str]:
    async_mode = mode == "async"
    return {
        "run_id": source["async_run_id" if async_mode else "sync_run_id"],
        "model_name": source["model"],
        "task": {
            "resnet50": "IMAGE_CLASSIFICATION",
            "yolov5m": "OBJECT_DETECTION",
            "yolov8s-pose": "POSE_ESTIMATION",
        }.get(source["model"], ""),
        "backend": source["npu"],
        "target_id": source["npu"],
        "inference_mode": "async_queue" if async_mode else "e2e",
        "mode": "report-async-runtime-return" if async_mode else "report-sync-runtime-call",
        "result_dataset": "edge_report_5_5",
        "measurement_boundary": "framework_runtime_call_return" if async_mode else "runtime_call",
        "latency_boundary": "runtime_call",
        "Average Latency (ms)": source["async_service_mean_ms" if async_mode else "sync_runtime_mean_ms"],
        "P99 Latency (ms)": source["async_service_p99_ms" if async_mode else "sync_runtime_p99_ms"],
        "throughput_samples_per_sec": source["async_runtime_completed_samples_per_sec"] if async_mode else "",
        "throughput_kind": "completed_over_measurement_duration" if async_mode else "runtime_mean_inverse",
        "sync_runtime_inverse_samples_per_sec": "" if async_mode else source["sync_runtime_inverse_samples_per_sec"],
        "quality_metric": source["quality_metric"],
        f"reported_{source['quality_metric']}": source["quality_value"],
        "reported_quality_secondary_metric": source["quality_secondary_metric"],
        **({f"reported_{source['quality_secondary_metric']}": source["quality_secondary_value"]}
           if source["quality_secondary_metric"] else {}),
        "quality_status": "report_summary",
        "comparison_eligible": "False",
        "power_evidence_status": "not_measured_in_report",
        "verification_status": source["verification_status"],
        "source_csv": source_csv,
        "source_csv_sha256": digest,
    }


def _single_row(path: Path) -> dict[str, str]:
    _, rows = _read_rows(path)
    if len(rows) != 1:
        raise ValueError(f"expected one run in {path}, found {len(rows)}")
    return rows[0]


def _positive_count(value: str, label: str) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {value!r}") from exc
    if count <= 0:
        raise ValueError(f"invalid {label}: {value!r}")
    return count


def _source_fields(path: Path, archive_root: Path) -> dict[str, str]:
    return {
        "source_csv": path.relative_to(archive_root).as_posix(),
        "source_csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _regulus_rows(sync_root: Path, async_root: Path, archive_root: Path) -> list[dict[str, str]]:
    rows = []
    for model in REGULUS_MODELS:
        sync_paths = list((sync_root / model).glob("*.csv"))
        if len(sync_paths) != 1:
            raise ValueError(f"expected one REGULUS E2E CSV for {model}, found {len(sync_paths)}")
        sync_path = sync_paths[0]
        sync = _single_row(sync_path)
        if (sync.get("model_name"), sync.get("target_id"), sync.get("inference_mode")) != (
            model, "mobilint-regulus", "vendor_native_e2e"
        ):
            raise ValueError(f"REGULUS E2E identity mismatch: {sync_path}")
        sync_samples = _positive_count(sync.get("System E2E Measured Samples", ""), "E2E samples")
        if not all(_positive_count(sync.get(key, ""), key) == sync_samples for key in (
            "measurement_sample_count", "synchronous_timing_sample_count", "Total Samples"
        )):
            raise ValueError(f"REGULUS E2E sample counts disagree: {sync_path}")
        timing_path = sync_path.parent / "timings" / f"{sync['run_id']}.samples.csv"
        _, timing_rows = _read_rows(timing_path)
        if len(timing_rows) != sync_samples:
            raise ValueError(f"REGULUS E2E timing count disagrees: {timing_path}")

        metric, source_metric, secondary, source_secondary = REGULUS_QUALITY[model]
        quality = {metric: sync[source_metric]}
        if secondary:
            quality[secondary] = sync[source_secondary]
        common = {
            "model_name": model,
            "task": sync["task"],
            "target_id": "mobilint-regulus",
            "result_dataset": "edge_report_5_5",
            "quality_metric": metric,
            "quality_secondary_metric": secondary,
            "power_evidence_status": "not_measured_in_report",
            "comparison_eligible": "False",
            **quality,
        }
        rows.append({
            **common,
            "run_id": sync["run_id"],
            "timestamp": sync["timestamp"],
            "backend": sync["backend"],
            "device": sync["device"],
            "batch_size": sync["batch_size"],
            "warmup_runs": sync["warmup_runs"],
            "inference_mode": "e2e",
            "mode": "system-e2e",
            "measurement_boundary": "vendor_native_e2e_single_stream",
            "latency_boundary": "runtime_call",
            "Average Latency (ms)": sync["Average Latency (ms)"],
            "P99 Latency (ms)": sync["P99 Latency (ms)"],
            "throughput_samples_per_sec": sync["System E2E Throughput (samples/s)"],
            "throughput_kind": "system_e2e_inverse_mean_latency",
            "samples_processed": str(sync_samples),
            "quality_status": "measured",
            "verification_status": "verified_archived",
            **_source_fields(sync_path, archive_root),
        })

        async_path = async_root / model / "results.csv"
        async_run = _single_row(async_path)
        detail_path = async_path.parent / "details" / f"{async_run['run_id']}.json"
        detail = json.loads(detail_path.read_text())
        if (async_run.get("model_name"), async_run.get("target_id"),
            async_run.get("inference_mode"), async_run.get("async_run_status"),
            async_run.get("async_measurement_boundary"), detail.get("run_id"),
            detail.get("status")) != (
            model, "mobilint-regulus", "async_queue", "valid",
            "framework_runtime_call_return", async_run["run_id"], "valid"
        ):
            raise ValueError(f"REGULUS async identity/status mismatch: {async_path}")
        async_samples = _positive_count(async_run.get("async_runtime_completed_samples", ""),
                                        "async completed samples")
        if not all(_positive_count(async_run.get(key, ""), key) == async_samples for key in (
            "async_submitted_requests", "async_completed_requests", "async_completed_samples"
        )) or detail["timing_ms"]["service_time"]["count"] != async_samples:
            raise ValueError(f"REGULUS async sample counts disagree: {async_path}")
        if any(int(async_run.get(key, "0")) != 0 for key in (
            "async_failed_requests", "async_rejected_requests", "async_timed_out_requests"
        )):
            raise ValueError(f"REGULUS async requests failed: {async_path}")
        service = detail["timing_ms"]["service_time"]
        rows.append({
            **common,
            "run_id": async_run["run_id"],
            "timestamp": async_run["timestamp"],
            "backend": async_run["backend"],
            "device": async_run["device"],
            "batch_size": async_run["batch_size"],
            "warmup_runs": async_run["warmup_runs"],
            "inference_mode": "async_queue",
            "scenario": async_run["scenario"],
            "worker_count": async_run["worker_count"],
            "async_run_status": async_run["async_run_status"],
            "mode": "report-async-runtime-return",
            "measurement_boundary": async_run["async_measurement_boundary"],
            "latency_boundary": "runtime_call",
            "Average Latency (ms)": str(service["mean"]),
            "P99 Latency (ms)": str(service["p99"]),
            "throughput_samples_per_sec": async_run["async_runtime_completed_samples_per_sec"],
            "throughput_kind": "completed_over_measurement_duration",
            "samples_processed": str(async_samples),
            "quality_status": async_run["quality_status"],
            "verification_status": "verified_separate_quality_pass",
            "source_detail_json": detail_path.relative_to(archive_root).as_posix(),
            "source_detail_sha256": hashlib.sha256(detail_path.read_bytes()).hexdigest(),
            "quality_source_csv": sync_path.relative_to(archive_root).as_posix(),
            "quality_source_csv_sha256": hashlib.sha256(sync_path.read_bytes()).hexdigest(),
            "quality_run_id": sync["run_id"],
            **_source_fields(async_path, archive_root),
        })
    return rows


def _combination(row: dict[str, str]) -> tuple[str, str, str]:
    key = (row.get("model_name", ""), row.get("target_id", ""), row.get("inference_mode", ""))
    if not all(key):
        raise ValueError(f"missing model/NPU/mode combination: {key}")
    return key


def consolidate_edge_results(
    power_web_csv: Path, report_csv: Path, output_csv: Path,
    *, required_combinations: set[tuple[str, str, str]] | frozenset[tuple[str, str, str]] | None = None,
    regulus_e2e_root: Path | None = None, regulus_async_root: Path | None = None,
) -> int:
    """Choose one run per model/NPU/mode, using original REGULUS evidence when supplied."""
    power_web_csv, report_csv, output_csv = map(Path, (power_web_csv, report_csv, output_csv))
    if (regulus_e2e_root is None) != (regulus_async_root is None):
        raise ValueError("REGULUS E2E and async evidence roots must be supplied together")
    power_columns, power_rows = _read_rows(power_web_csv)
    report_columns, report_rows = _read_rows(report_csv)
    missing = set(REPORT_COLUMNS) - set(report_columns)
    if missing:
        raise ValueError(f"report columns missing: {', '.join(sorted(missing))}")
    report_digest = hashlib.sha256(report_csv.read_bytes()).hexdigest()
    report_source = f"{report_csv.parent.name}/{report_csv.name}"
    rows = []
    selected_combinations: set[tuple[str, str, str]] = set()
    for row in power_rows:
        if row.get("result_dataset") != "edge_power_selected":
            raise ValueError("power input is not a selected edge power result CSV")
        key = _combination(row)
        if key in selected_combinations:
            raise ValueError(f"duplicate power combination: {key}")
        selected_combinations.add(key)
        rows.append({
            **row,
            "source_csv": "edge-power/selected-results.csv",
            "source_csv_sha256": row.get("selection_sha256", ""),
            "verification_status": "verified_archived",
            "throughput_kind": "completed_over_measurement_duration",
        })
    report_combinations: set[tuple[str, str, str]] = set()
    for source in report_rows:
        if regulus_e2e_root is not None and source["npu"] == "mobilint-regulus":
            continue
        for mode in ("sync", "async"):
            row = _report_row(source, mode, report_digest, report_source)
            key = _combination(row)
            if key in report_combinations:
                raise ValueError(f"duplicate report combination: {key}")
            report_combinations.add(key)
            if key not in selected_combinations:
                selected_combinations.add(key)
                rows.append(row)
    if regulus_e2e_root is not None and regulus_async_root is not None:
        for row in _regulus_rows(Path(regulus_e2e_root), Path(regulus_async_root), report_csv.parent.parent):
            key = _combination(row)
            if key in selected_combinations:
                raise ValueError(f"duplicate REGULUS combination: {key}")
            selected_combinations.add(key)
            rows.append(row)
    if required_combinations is not None:
        missing = required_combinations - selected_combinations
        extra = selected_combinations - required_combinations
        if missing or extra:
            raise ValueError(f"missing combinations: {sorted(missing)}; extra combinations: {sorted(extra)}")
    run_ids = [row.get("run_id", "") for row in rows]
    if any(not run_id for run_id in run_ids) or len(set(run_ids)) != len(run_ids):
        raise ValueError("empty or duplicate run ID in consolidated edge results")
    columns = list(dict.fromkeys([*power_columns, *EXTRA_COLUMNS,
                                  *(key for row in rows for key in row)]))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", newline="", dir=output_csv.parent,
                                         prefix=f".{output_csv.name}.", suffix=".tmp",
                                         delete=False) as handle:
            temp_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, output_csv)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path, help="Directory containing edge-power and edge-report-5.5")
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        power_web_csv = Path(directory) / "selected-web.csv"
        convert_selected_results(args.archive_root / "edge-power/selected-results.csv",
                                 args.archive_root, power_web_csv)
        count = consolidate_edge_results(
            power_web_csv, args.archive_root / "edge-report-5.5/reported-values.csv",
            args.output_csv, required_combinations=EXPECTED_COMBINATIONS,
            regulus_e2e_root=args.archive_root / (
                "edge-report-5.5/evidence/results/results-20260927-regulus-5000-runtime-call/"
                "vendor-native-e2e/vendor-reference-single-stream-v1/mobilint-regulus"
            ),
            regulus_async_root=args.archive_root / (
                "edge-report-5.5/evidence/results/"
                "2026-10-01-regulus-vg-e2e-async-runtime-v2-preloaded-5000"
            ),
        )
    print(f"Consolidated {count} edge runs into {args.output_csv}")


if __name__ == "__main__":
    main()
