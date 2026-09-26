"""Shared contracts for recording raw accelerator power readings.

This module deliberately contains no energy integration or derived metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Literal


PowerSampleStatus = Literal["ok", "unavailable", "read_error", "overrun"]
PowerTraceStatus = Literal[
    "disabled", "complete", "partial", "unavailable", "failed"
]
PowerTracePhase = Literal["baseline", "inference"]

_POWER_SAMPLE_STATUSES = {"ok", "unavailable", "read_error", "overrun"}
_POWER_TRACE_STATUSES = {
    "disabled",
    "complete",
    "partial",
    "unavailable",
    "failed",
}
_POWER_TRACE_PHASES = {"baseline", "inference"}
_ERROR_CODE_PATTERN = re.compile(r"[A-Za-z0-9_:-]{0,128}\Z", re.ASCII)


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_finite_non_negative(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite non-negative number")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{field_name} must be a finite non-negative number")


@dataclass(frozen=True)
class PowerTraceSource:
    collector: str
    monitor_source: str
    device_id: str
    power_scope: str
    sample_interval_sec: float

    def __post_init__(self) -> None:
        for field_name in (
            "collector",
            "monitor_source",
            "device_id",
            "power_scope",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        _require_finite_non_negative(
            self.sample_interval_sec, "sample_interval_sec"
        )
        if self.sample_interval_sec == 0:
            raise ValueError("sample_interval_sec must be greater than zero")


@dataclass(frozen=True)
class PowerReading:
    status: PowerSampleStatus
    power_w: float | None = None
    error_code: str = ""

    def __post_init__(self) -> None:
        if self.status not in _POWER_SAMPLE_STATUSES:
            raise ValueError(f"invalid power reading status: {self.status!r}")
        if not isinstance(self.error_code, str) or not _ERROR_CODE_PATTERN.fullmatch(
            self.error_code
        ):
            raise ValueError("error_code must be at most 128 safe ASCII characters")

        if self.status == "ok":
            if self.power_w is None:
                raise ValueError("ok power reading requires power_w")
            _require_finite_non_negative(self.power_w, "power_w")
            if self.error_code:
                raise ValueError("ok power reading cannot contain error_code")
        else:
            if self.power_w is not None:
                raise ValueError("failed power reading cannot contain power_w")
            if not self.error_code:
                raise ValueError("failed power reading requires error_code")


@dataclass(frozen=True)
class PowerTraceConfig:
    run_id: str
    target_id: str
    baseline_duration_sec: float = 3.0

    def __post_init__(self) -> None:
        _require_non_empty(self.run_id, "run_id")
        _require_non_empty(self.target_id, "target_id")
        _require_finite_non_negative(
            self.baseline_duration_sec, "baseline_duration_sec"
        )


@dataclass(frozen=True)
class PowerTraceSample:
    phase: PowerTracePhase
    scheduled_elapsed_ms: float
    observed_elapsed_ms: float
    query_latency_ms: float
    reading: PowerReading

    def __post_init__(self) -> None:
        if self.phase not in _POWER_TRACE_PHASES:
            raise ValueError(f"invalid power trace phase: {self.phase!r}")
        for field_name in (
            "scheduled_elapsed_ms",
            "observed_elapsed_ms",
            "query_latency_ms",
        ):
            _require_finite_non_negative(getattr(self, field_name), field_name)
        if not isinstance(self.reading, PowerReading):
            raise ValueError("reading must be a PowerReading")


@dataclass(frozen=True)
class PowerTraceArtifact:
    status: PowerTraceStatus
    path: str = ""
    sha256: str = ""
    sample_count: int | None = None
    monitor_source: str = ""
    power_scope: str = ""

    def __post_init__(self) -> None:
        if self.status not in _POWER_TRACE_STATUSES:
            raise ValueError(f"invalid power trace status: {self.status!r}")
        if self.sample_count is not None:
            if type(self.sample_count) is not int or self.sample_count < 0:
                raise ValueError("sample_count must be a non-negative integer or None")

    def as_result_metadata(self) -> dict[str, object]:
        """Return linkage metadata only; never expose derived power metrics."""
        return {
            "power_trace_status": self.status,
            "power_trace_path": self.path,
            "power_trace_sha256": self.sha256,
            "power_trace_sample_count": self.sample_count,
            "power_monitor_source": self.monitor_source,
            "power_scope": self.power_scope,
        }
