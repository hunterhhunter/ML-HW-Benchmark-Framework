"""Raw power trace contract tests."""

import math

import pytest

from core.power_trace import PowerReading, PowerTraceArtifact


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
