"""Offline analysis of a complete Jetson DC input power trace.

The result is a system-level, baseline-subtracted estimate. It does not
attribute power to the NPU core or the vendor runtime alone.
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from statistics import median


def _finite_nonnegative(raw: str, field: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {raw!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {field}: {raw!r}")
    return value


def _completed_samples(row: dict[str, str]) -> int:
    def read_count(field: str) -> int:
        raw = row.get(field, "")
        try:
            count = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"missing completed sample count: {field}") from exc
        if count <= 0:
            raise ValueError(f"invalid completed sample count: {field}={raw!r}")
        return count

    if row.get("inference_mode") != "async_queue":
        return read_count("Total Samples")
    if row.get("async_run_status") != "valid":
        raise ValueError("async run is not valid")
    runtime_count = read_count("async_runtime_completed_samples")
    if runtime_count != read_count("async_completed_samples"):
        raise ValueError("Runtime-call completion count does not match run lifecycle")
    return runtime_count


def analyze_run(results_path: str | Path, run_id: str) -> dict[str, float | int | str]:
    """Verify a result and trace, then integrate its inference power in joules."""
    results_path = Path(results_path)
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("run_id") == run_id]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one result row for run_id={run_id!r}")
    row = rows[0]
    if row.get("power_trace_status") != "complete":
        raise ValueError("power trace is not complete")
    if row.get("power_scope") != "jetson_dc_input":
        raise ValueError("power scope is not Jetson DC input")

    expected_relative = Path("power") / f"{run_id}.power.csv"
    if row.get("power_trace_path") != expected_relative.as_posix():
        raise ValueError("power trace path does not match run ID")
    trace_path = results_path.parent / expected_relative
    if hashlib.sha256(trace_path.read_bytes()).hexdigest() != row.get("power_trace_sha256"):
        raise ValueError("power trace SHA-256 mismatch")

    samples: dict[str, list[tuple[float, float]]] = {
        "baseline": [],
        "inference": [],
    }
    with trace_path.open(newline="", encoding="utf-8") as handle:
        for sample in csv.DictReader(handle):
            phase = sample.get("phase", "")
            if sample.get("run_id") != run_id or sample.get("power_scope") != "jetson_dc_input":
                raise ValueError("trace sample does not match the selected run or scope")
            if phase not in samples or sample.get("sample_status") != "ok":
                raise ValueError("trace contains an invalid or missing power sample")
            timestamp_s = _finite_nonnegative(
                sample.get("observed_elapsed_ms", ""), "observed_elapsed_ms"
            ) / 1000.0
            power_w = _finite_nonnegative(sample.get("power_w", ""), "power_w")
            samples[phase].append((timestamp_s, power_w))

    baseline = samples["baseline"]
    inference = samples["inference"]
    if len(baseline) < 2 or len(inference) < 2:
        raise ValueError("at least two valid samples are required in each phase")
    if baseline[-1][0] >= inference[0][0]:
        raise ValueError("baseline and inference phase timestamps overlap")
    if any(right[0] <= left[0] for left, right in zip(inference, inference[1:])):
        raise ValueError("inference sample timestamps must increase")

    idle_w = median(power for _, power in baseline)
    inference_median_w = median(power for _, power in inference)
    duration_s = inference[-1][0] - inference[0][0]
    system_energy_j = sum(
        (right_t - left_t) * (left_w + right_w) / 2.0
        for (left_t, left_w), (right_t, right_w) in zip(inference, inference[1:])
    )
    incremental_energy_j = system_energy_j - idle_w * duration_s
    count = _completed_samples(row)
    return {
        "run_id": run_id,
        "power_scope": "jetson_dc_input",
        "completed_samples": count,
        "idle_median_w": idle_w,
        "inference_median_w": inference_median_w,
        "inference_mean_w": system_energy_j / duration_s,
        "incremental_mean_w": incremental_energy_j / duration_s,
        "inference_duration_s": duration_s,
        "system_energy_j": system_energy_j,
        "incremental_energy_j": incremental_energy_j,
        "system_j_per_sample": system_energy_j / count,
        "incremental_j_per_sample": incremental_energy_j / count,
    }
