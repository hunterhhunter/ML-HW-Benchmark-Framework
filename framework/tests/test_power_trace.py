"""Raw power trace contract and writer tests."""

import csv
import hashlib
import math
from pathlib import Path

import pytest

import core.power_trace as power_trace_module
from core.power_trace import (
    PowerReading,
    PowerTraceArtifact,
    PowerTraceConfig,
    PowerTraceSample,
    PowerTraceSource,
    PowerTraceWriter,
)
from core.result_store import reserve_run_artifacts


EXPECTED_HEADER = [
    "schema_version",
    "run_id",
    "sample_index",
    "phase",
    "target_id",
    "collector",
    "monitor_source",
    "device_id",
    "power_scope",
    "scheduled_elapsed_ms",
    "observed_elapsed_ms",
    "query_latency_ms",
    "power_w",
    "sample_status",
    "error_code",
]


def make_writer(tmp_path: Path, *, run_id: str = "fixed123"):
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results" / "benchmark_results.csv",
        run_id=run_id,
    )
    writer = PowerTraceWriter(
        reservation,
        PowerTraceConfig(run_id=run_id, target_id="mobilint-aries"),
        PowerTraceSource(
            collector="mobilint",
            monitor_source="mbltml",
            device_id="0",
            power_scope="device_total",
            sample_interval_sec=0.2,
        ),
    )
    return writer, reservation


def make_sample(
    *,
    phase="baseline",
    scheduled_elapsed_ms=0.0,
    observed_elapsed_ms=0.1,
    query_latency_ms=0.2,
    reading=None,
):
    return PowerTraceSample(
        phase=phase,
        scheduled_elapsed_ms=scheduled_elapsed_ms,
        observed_elapsed_ms=observed_elapsed_ms,
        query_latency_ms=query_latency_ms,
        reading=reading or PowerReading(status="ok", power_w=7.913),
    )


@pytest.mark.parametrize("power_w", [None, -0.01, math.nan, math.inf, -math.inf])
def test_ok_power_reading_requires_finite_non_negative_watts(power_w):
    with pytest.raises(ValueError):
        PowerReading(status="ok", power_w=power_w)

    assert PowerReading(status="ok", power_w=0.0).power_w == 0.0


def test_failed_power_reading_requires_empty_power_and_bounded_error_code():
    with pytest.raises(ValueError):
        PowerReading(status="read_error", power_w=1.0, error_code="sdk_error")

    for error_code in ("", "contains space", "한글", "x" * 129):
        with pytest.raises(ValueError):
            PowerReading(
                status="read_error",
                power_w=None,
                error_code=error_code,
            )

    reading = PowerReading(
        status="read_error",
        power_w=None,
        error_code="sdk:error-1",
    )
    assert reading.error_code == "sdk:error-1"


def test_power_trace_artifact_exposes_only_linkage_metadata():
    artifact = PowerTraceArtifact(
        status="complete",
        path="power/run-1.power.csv",
        sha256="a" * 64,
        sample_count=17,
        monitor_source="mbltml",
        power_scope="device_total",
    )

    assert artifact.as_result_metadata() == {
        "power_trace_status": "complete",
        "power_trace_path": "power/run-1.power.csv",
        "power_trace_sha256": "a" * 64,
        "power_trace_sample_count": 17,
        "power_monitor_source": "mbltml",
        "power_scope": "device_total",
    }


def test_writer_publishes_exact_raw_schema_and_sha256(tmp_path):
    writer, reservation = make_writer(tmp_path)
    writer.start()
    writer.write_attempt(make_sample())
    writer.write_attempt(
        make_sample(
            phase="inference",
            scheduled_elapsed_ms=200.0,
            observed_elapsed_ms=201.0,
            query_latency_ms=0.3,
            reading=PowerReading(status="ok", power_w=8.25),
        )
    )

    artifact = writer.finish()

    assert artifact.status == "complete"
    assert artifact.path == "power/fixed123.power.csv"
    assert artifact.sample_count == 2
    published = reservation.power_trace_path
    assert artifact.sha256 == hashlib.sha256(published.read_bytes()).hexdigest()
    with published.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        assert handle.seek(0) == 0
        assert next(csv.reader(handle)) == EXPECTED_HEADER
    assert [row["sample_index"] for row in rows] == ["0", "1"]
    assert rows[0]["power_w"] == "7.913"
    assert rows[1]["phase"] == "inference"


def test_writer_records_failed_and_overrun_attempts_with_empty_power(tmp_path):
    writer, reservation = make_writer(tmp_path)
    writer.start()
    writer.write_attempt(
        make_sample(
            reading=PowerReading(
                status="read_error", error_code="sdk:read-error"
            )
        )
    )
    writer.write_attempt(
        make_sample(
            phase="inference",
            reading=PowerReading(status="overrun", error_code="deadline_missed"),
        )
    )

    artifact = writer.finish()

    assert artifact.status == "partial"
    with reservation.power_trace_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["sample_status"] for row in rows] == ["read_error", "overrun"]
    assert [row["power_w"] for row in rows] == ["", ""]
    assert [row["error_code"] for row in rows] == [
        "sdk:read-error",
        "deadline_missed",
    ]


def test_writer_never_overwrites_existing_final_trace(tmp_path):
    writer, reservation = make_writer(tmp_path)
    writer.start()
    writer.write_attempt(make_sample())
    reservation.power_trace_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        writer.finish()

    assert reservation.power_trace_path.read_text(encoding="utf-8") == "existing\n"
    assert writer.fail().status == "failed"


def test_writer_failure_leaves_no_final_artifact(tmp_path, monkeypatch):
    writer, reservation = make_writer(tmp_path)
    writer.start()
    writer.write_attempt(make_sample())

    def fail_publication(*args, **kwargs):
        raise OSError("injected publication failure")

    monkeypatch.setattr(power_trace_module, "link_no_overwrite", fail_publication)
    with pytest.raises(OSError, match="injected publication failure"):
        writer.finish()

    assert not reservation.power_trace_path.exists()
    artifact = writer.fail()
    assert artifact.status == "failed"
    assert artifact.path == ""
    assert artifact.sha256 == ""
    assert artifact.sample_count is None
