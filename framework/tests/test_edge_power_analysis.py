"""Offline energy estimates must use the linked run and completed sample count."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def _sample(phase, elapsed_ms, power_w, status="ok"):
    return {
        "run_id": "run-1",
        "phase": phase,
        "observed_elapsed_ms": str(elapsed_ms),
        "power_w": str(power_w),
        "sample_status": status,
        "power_scope": "jetson_dc_input",
    }


def _make_run(tmp_path, *, status="complete", sample_status="ok", mode="async_queue"):
    trace = tmp_path / "power" / "run-1.power.csv"
    trace.parent.mkdir()
    _write_csv(trace, [
        _sample("baseline", 0, 5),
        _sample("baseline", 500, 5),
        _sample("inference", 1000, 10),
        _sample("inference", 2000, 12, sample_status),
        _sample("inference", 3000, 10),
    ])
    results = tmp_path / "results.csv"
    _write_csv(results, [{
        "run_id": "run-1",
        "inference_mode": mode,
        "async_completed_samples": "4",
        "async_runtime_completed_samples": "4",
        "async_run_status": "valid",
        "Total Samples": "4",
        "power_trace_status": status,
        "power_trace_path": "power/run-1.power.csv",
        "power_trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "power_scope": "jetson_dc_input",
    }])
    return results


def test_edge_analysis_integrates_run_linked_trace_and_completed_samples(tmp_path):
    from core.edge_power_analysis import analyze_run

    report = analyze_run(_make_run(tmp_path), "run-1")

    assert report["idle_median_w"] == pytest.approx(5)
    assert report["inference_median_w"] == pytest.approx(10)
    assert report["inference_mean_w"] == pytest.approx(11)
    assert report["incremental_mean_w"] == pytest.approx(6)
    assert report["inference_duration_s"] == pytest.approx(2)
    assert report["system_energy_j"] == pytest.approx(22)
    assert report["incremental_energy_j"] == pytest.approx(12)
    assert report["incremental_j_per_sample"] == pytest.approx(3)
    assert report["completed_samples"] == 4


def test_edge_analysis_cli_reads_the_same_run(tmp_path):
    results = _make_run(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_edge_power.py"
    completed = subprocess.run(
        [sys.executable, str(script), str(results), "--run-id", "run-1"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout)["incremental_j_per_sample"] == pytest.approx(3)


def test_edge_analysis_uses_e2e_completed_sample_count(tmp_path):
    from core.edge_power_analysis import analyze_run

    report = analyze_run(_make_run(tmp_path, mode="e2e"), "run-1")
    assert report["completed_samples"] == 4


def test_edge_analysis_rejects_runtime_return_count_mismatch(tmp_path):
    from core.edge_power_analysis import analyze_run

    results = _make_run(tmp_path)
    with results.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["async_runtime_completed_samples"] = "2"
    _write_csv(results, rows)

    with pytest.raises(ValueError, match="Runtime-call completion count"):
        analyze_run(results, "run-1")


@pytest.mark.parametrize("status,sample_status", [("partial", "ok"), ("complete", "read_error")])
def test_edge_analysis_refuses_incomplete_power_evidence(tmp_path, status, sample_status):
    from core.edge_power_analysis import analyze_run

    results = _make_run(tmp_path, status=status, sample_status=sample_status)
    with pytest.raises(ValueError):
        analyze_run(results, "run-1")


def test_edge_analysis_refuses_modified_trace(tmp_path):
    from core.edge_power_analysis import analyze_run

    results = _make_run(tmp_path)
    trace = tmp_path / "power" / "run-1.power.csv"
    trace.write_bytes(trace.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        analyze_run(results, "run-1")


def test_edge_analysis_identifies_preload_baseline_and_rejects_mismatch(tmp_path):
    from core.edge_power_analysis import analyze_run

    results = _make_run(tmp_path)
    trace = tmp_path / "power" / "run-1.power.csv"
    with trace.open(newline="", encoding="utf-8") as handle:
        samples = list(csv.DictReader(handle))
    for sample in samples:
        sample["baseline_state"] = "preload_framework_idle"
        if sample["phase"] == "inference":
            sample["observed_elapsed_ms"] = str(
                float(sample["observed_elapsed_ms"]) + 60_000
            )
    _write_csv(trace, samples)
    with results.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["power_baseline_state"] = "preload_framework_idle"
    rows[0]["power_trace_sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
    _write_csv(results, rows)

    report = analyze_run(results, "run-1")
    assert report["baseline_state"] == "preload_framework_idle"
    assert report["inference_duration_s"] == pytest.approx(2)
    assert report["incremental_j_per_sample"] == pytest.approx(3)

    rows[0]["power_baseline_state"] = "post_warmup_idle"
    _write_csv(results, rows)
    with pytest.raises(ValueError, match="baseline state"):
        analyze_run(results, "run-1")
