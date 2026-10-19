"""Combine selected edge power runs and separate historical report runs for the web CSV."""

import argparse
import csv
import hashlib
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
    "sync_runtime_inverse_samples_per_sec", "reported_quality_secondary_metric",
)


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


def consolidate_edge_results(power_web_csv: Path, report_csv: Path, output_csv: Path) -> int:
    """Write one benchmark-compatible CSV while preserving each original run ID."""
    power_web_csv, report_csv, output_csv = map(Path, (power_web_csv, report_csv, output_csv))
    power_columns, power_rows = _read_rows(power_web_csv)
    report_columns, report_rows = _read_rows(report_csv)
    missing = set(REPORT_COLUMNS) - set(report_columns)
    if missing:
        raise ValueError(f"report columns missing: {', '.join(sorted(missing))}")
    report_digest = hashlib.sha256(report_csv.read_bytes()).hexdigest()
    report_source = f"{report_csv.parent.name}/{report_csv.name}"
    rows = []
    for row in power_rows:
        if row.get("result_dataset") != "edge_power_selected":
            raise ValueError("power input is not a selected edge power result CSV")
        rows.append({
            **row,
            "source_csv": "edge-power/selected-results.csv",
            "source_csv_sha256": row.get("selection_sha256", ""),
            "verification_status": "verified_archived",
            "throughput_kind": "completed_over_measurement_duration",
        })
    for source in report_rows:
        rows.extend((_report_row(source, "sync", report_digest, report_source),
                     _report_row(source, "async", report_digest, report_source)))
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
            args.output_csv,
        )
    print(f"Consolidated {count} edge runs into {args.output_csv}")


if __name__ == "__main__":
    main()
