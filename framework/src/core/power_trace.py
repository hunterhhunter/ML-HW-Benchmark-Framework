"""Shared contracts for recording raw accelerator power readings.

This module deliberately contains no energy integration or derived metrics.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import math
import os
import re
import stat
from typing import Literal
import uuid

from .artifact_reservation import (
    RunArtifactReservation,
    directory_binding_matches,
    link_no_overwrite,
    reservation_binding_matches,
    revalidate_reservation,
    verify_reservation,
)


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
_TRACE_DIRECTORY = "power"
_TRACE_FILE_MODE = 0o600
_TRACE_DIRECTORY_MODE = 0o755
_TRACE_HEADER = [
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


class PowerTraceWriter:
    """Stream raw readings to one run-bound CSV and publish it atomically."""

    def __init__(
        self,
        reservation: RunArtifactReservation,
        config: PowerTraceConfig,
        source: PowerTraceSource,
    ):
        if type(reservation) is not RunArtifactReservation:
            raise ValueError("a valid RunArtifactReservation is required")
        if config.run_id != reservation.run_id:
            raise ValueError("power trace run_id does not match reservation")
        self._reservation = reservation
        self._config = config
        self._source = source
        self._verification_context = None
        self._verified = None
        self._directory_fd: int | None = None
        self._handle = None
        self._temporary_name: str | None = None
        self._temporary_identity: tuple[int, int] | None = None
        self._sample_count = 0
        self._has_failed_attempt = False
        self._state = "new"
        self._artifact: PowerTraceArtifact | None = None

    def start(self) -> None:
        if self._state != "new":
            raise RuntimeError("power trace writer is already started")

        context = verify_reservation(
            self._reservation,
            self._config.run_id,
            results_path=self._reservation.results_path,
            require_active=True,
        )
        self._verification_context = context
        try:
            verified = context.__enter__()
            self._verified = verified
            root_fd = verified.root.file_descriptor
            try:
                os.mkdir(
                    _TRACE_DIRECTORY,
                    mode=_TRACE_DIRECTORY_MODE,
                    dir_fd=root_fd,
                )
            except FileExistsError:
                pass
            else:
                os.fsync(root_fd)

            directory_fd = os.open(
                _TRACE_DIRECTORY,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root_fd,
            )
            self._directory_fd = directory_fd
            if not self._directories_match():
                raise ValueError("power trace directory identity changed")

            final_name = self._reservation.power_trace_path.name
            temporary_name = f".{final_name}.{uuid.uuid4().hex}.tmp"
            file_fd = os.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                _TRACE_FILE_MODE,
                dir_fd=directory_fd,
            )
            self._temporary_name = temporary_name
            self._temporary_identity = self._regular_file_identity(file_fd)
            self._handle = os.fdopen(
                file_fd,
                "w",
                encoding="utf-8",
                newline="",
            )
            csv.DictWriter(self._handle, fieldnames=_TRACE_HEADER).writeheader()
            self._state = "started"
        except BaseException:
            self._state = "publication_failed"
            self._cleanup(close_verification=True, preserve_partial=False)
            raise

    def write_attempt(self, sample: PowerTraceSample) -> None:
        if self._state != "started" or self._handle is None:
            raise RuntimeError("power trace writer is not active")
        if type(sample) is not PowerTraceSample:
            raise ValueError("sample must be a PowerTraceSample")

        reading = sample.reading
        csv.DictWriter(self._handle, fieldnames=_TRACE_HEADER).writerow(
            {
                "schema_version": "1.0",
                "run_id": self._config.run_id,
                "sample_index": self._sample_count,
                "phase": sample.phase,
                "target_id": self._config.target_id,
                "collector": self._source.collector,
                "monitor_source": self._source.monitor_source,
                "device_id": self._source.device_id,
                "power_scope": self._source.power_scope,
                "scheduled_elapsed_ms": sample.scheduled_elapsed_ms,
                "observed_elapsed_ms": sample.observed_elapsed_ms,
                "query_latency_ms": sample.query_latency_ms,
                "power_w": reading.power_w if reading.status == "ok" else "",
                "sample_status": reading.status,
                "error_code": reading.error_code,
            }
        )
        self._sample_count += 1
        if reading.status != "ok":
            self._has_failed_attempt = True

    def flush(self) -> None:
        if self._state != "started" or self._handle is None:
            raise RuntimeError("power trace writer is not active")
        self._handle.flush()

    def finish(self) -> PowerTraceArtifact:
        if self._state == "finished" and self._artifact is not None:
            return self._artifact
        if self._state != "started" or self._handle is None:
            raise RuntimeError("power trace writer is not active")

        final_published = False
        final_name = self._reservation.power_trace_path.name
        try:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._temporary_identity = self._regular_file_identity(
                self._handle.fileno()
            )
            self._handle.close()
            self._handle = None
            if not self._directories_match():
                raise ValueError("power trace directory identity changed")
            revalidate_reservation(self._verified, require_active=True)
            link_no_overwrite(
                self._temporary_name,
                final_name,
                source_directory_fd=self._directory_fd,
                target_directory_fd=self._directory_fd,
            )
            final_published = True
            self._validate_final(final_name)
            os.unlink(self._temporary_name, dir_fd=self._directory_fd)
            self._temporary_name = None
            os.fsync(self._directory_fd)
            self._validate_final(final_name)
            revalidate_reservation(self._verified, require_active=True)

            sha256 = self._hash_final(final_name)
            status: PowerTraceStatus = (
                "partial" if self._has_failed_attempt else "complete"
            )
            artifact = PowerTraceArtifact(
                status=status,
                path=str(self._reservation.power_trace_path.relative_to(
                    self._reservation.results_root
                )),
                sha256=sha256,
                sample_count=self._sample_count,
                monitor_source=self._source.monitor_source,
                power_scope=self._source.power_scope,
            )
            self._artifact = artifact
            self._state = "finished"
            self._cleanup(close_verification=True, preserve_partial=False)
            return artifact
        except BaseException:
            self._state = "publication_failed"
            if final_published:
                self._remove_owned_final(final_name)
            raise

    def fail(self) -> PowerTraceArtifact:
        if self._state == "finished" and self._artifact is not None:
            return self._artifact
        if self._state == "failed" and self._artifact is not None:
            return self._artifact

        self._cleanup(close_verification=True, preserve_partial=True)
        artifact = PowerTraceArtifact(
            status="failed",
            monitor_source=self._source.monitor_source,
            power_scope=self._source.power_scope,
        )
        self._artifact = artifact
        self._state = "failed"
        return artifact

    @staticmethod
    def _regular_file_identity(file_descriptor: int) -> tuple[int, int]:
        opened = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("power trace temporary artifact is not regular")
        return opened.st_dev, opened.st_ino

    def _directories_match(self) -> bool:
        return (
            self._verified is not None
            and self._directory_fd is not None
            and reservation_binding_matches(self._verified)
            and directory_binding_matches(
                self._reservation.power_trace_path.parent,
                self._directory_fd,
            )
        )

    def _validate_final(self, final_name: str) -> None:
        if not self._directories_match():
            raise ValueError("power trace directory identity changed")
        opened = os.stat(
            final_name,
            dir_fd=self._directory_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != self._temporary_identity:
            raise ValueError("power trace final artifact identity changed")

    def _hash_final(self, final_name: str) -> str:
        file_fd = os.open(
            final_name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=self._directory_fd,
        )
        digest = hashlib.sha256()
        try:
            while True:
                block = os.read(file_fd, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
        finally:
            os.close(file_fd)
        return digest.hexdigest()

    def _remove_owned_final(self, final_name: str) -> None:
        try:
            opened = os.stat(
                final_name,
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
            if (opened.st_dev, opened.st_ino) != self._temporary_identity:
                return
            os.unlink(final_name, dir_fd=self._directory_fd)
            os.fsync(self._directory_fd)
        except BaseException:
            pass

    def _cleanup(self, *, close_verification: bool, preserve_partial: bool) -> None:
        if self._handle is not None:
            try:
                self._handle.flush()
                self._temporary_identity = self._regular_file_identity(
                    self._handle.fileno()
                )
            except BaseException:
                pass
            try:
                self._handle.close()
            except BaseException:
                pass
            self._handle = None

        directory_fd = self._directory_fd
        temporary_name = self._temporary_name
        if directory_fd is not None and temporary_name is not None:
            if preserve_partial and self._directories_match():
                try:
                    link_no_overwrite(
                        temporary_name,
                        f"{self._config.run_id}.power.partial.csv",
                        source_directory_fd=directory_fd,
                        target_directory_fd=directory_fd,
                    )
                    os.fsync(directory_fd)
                except BaseException:
                    pass
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
                os.fsync(directory_fd)
            except BaseException:
                pass
            self._temporary_name = None

        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except BaseException:
                pass
            self._directory_fd = None

        if close_verification and self._verification_context is not None:
            context = self._verification_context
            self._verification_context = None
            self._verified = None
            try:
                context.__exit__(None, None, None)
            except BaseException:
                pass
