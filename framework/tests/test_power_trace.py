"""Regression tests for run-bound raw power traces."""

import csv
import hashlib
import importlib

from core.result_store import reserve_run_artifacts


def test_power_trace_writer_links_raw_rows_to_reserved_run(tmp_path):
    """A missing writer or reservation path must fail this contract."""
    power_trace = importlib.import_module("core.power_trace")
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="power-run-1",
    )
    writer = power_trace.PowerTraceWriter(
        reservation,
        power_trace.PowerTraceConfig(
            run_id=reservation.run_id,
            target_id="mobilint-aries",
        ),
        power_trace.PowerTraceSource(
            collector="mobilint",
            monitor_source="mbltml",
            device_id="0",
            power_scope="device_total",
            sample_interval_sec=0.1,
        ),
    )
    writer.start()
    writer.write_attempt(
        power_trace.PowerTraceSample(
            phase="baseline",
            scheduled_elapsed_ms=0.0,
            observed_elapsed_ms=0.1,
            query_latency_ms=0.2,
            reading=power_trace.PowerReading(status="ok", power_w=7.913),
        )
    )
    artifact = writer.finish()
    path = tmp_path / artifact.path
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert artifact.status == "complete"
    assert artifact.path == "power/power-run-1.power.csv"
    assert artifact.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert artifact.sample_count == 1
    assert rows[0]["run_id"] == reservation.run_id
    assert rows[0]["power_w"] == "7.913"
