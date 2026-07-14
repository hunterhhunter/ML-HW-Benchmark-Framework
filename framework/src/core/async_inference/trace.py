import json
import math
import os
import queue
import threading
import time
import uuid
from pathlib import Path

from ..artifact_reservation import (
    RunArtifactReservation,
    directory_binding_matches,
    link_no_overwrite,
    reservation_binding_matches,
    verify_reservation,
)
from .types import RequestTrace, TerminalStatus


_STOP = object()
_INTEGER_FIELDS = (
    "request_id",
    "sample_index",
    "scheduled_ns",
    "issued_ns",
    "enqueued_ns",
    "runtime_started_ns",
    "runtime_finished_ns",
    "completed_ns",
    "worker_id",
    "batch_size",
    "sample_count",
)


def _safe_error(phase, exc):
    error_type = "<unknown>"
    try:
        name = type.__getattribute__(type(exc), "__name__")
        if type(name) is str:
            error_type = name
    except BaseException:
        pass
    message = f"<{error_type}>"
    try:
        args = BaseException.args.__get__(exc, type(exc))
    except BaseException:
        args = ()
    if type(args) is tuple and len(args) == 1 and type(args[0]) is str:
        message = args[0]
    return {
        "phase": phase,
        "error_type": error_type,
        "error_message": message,
    }


def _trace_row(trace):
    if type(trace) is not RequestTrace:
        raise TypeError("RequestTraceWriter accepts only exact RequestTrace values")
    values = {}
    for field in _INTEGER_FIELDS:
        value = object.__getattribute__(trace, field)
        if type(value) is not int:
            raise TypeError(f"RequestTrace.{field} must be an exact int")
        values[field] = value
    status = object.__getattribute__(trace, "status")
    if type(status) is not TerminalStatus:
        raise TypeError("RequestTrace.status must be an exact TerminalStatus")
    timed_out = object.__getattribute__(trace, "timed_out")
    if type(timed_out) is not bool:
        raise TypeError("RequestTrace.timed_out must be an exact bool")
    error_type = object.__getattribute__(trace, "error_type")
    error_message = object.__getattribute__(trace, "error_message")
    if error_type is not None and type(error_type) is not str:
        raise TypeError("RequestTrace.error_type must be str or None")
    if error_message is not None and type(error_message) is not str:
        raise TypeError("RequestTrace.error_message must be str or None")
    return {
        "request_id": values["request_id"],
        "sample_index": values["sample_index"],
        "status": object.__getattribute__(status, "_value_"),
        "scheduled_ns": values["scheduled_ns"],
        "issued_ns": values["issued_ns"],
        "enqueued_ns": values["enqueued_ns"],
        "runtime_started_ns": values["runtime_started_ns"],
        "runtime_finished_ns": values["runtime_finished_ns"],
        "completed_ns": values["completed_ns"],
        "worker_id": values["worker_id"],
        "batch_size": values["batch_size"],
        "timed_out": timed_out,
        "sample_count": values["sample_count"],
        "error_type": error_type,
        "error_message": error_message,
    }


class RequestTraceWriter:
    """Bounded, non-blocking RequestTrace sink with atomic publication."""

    _CREATED = "created"
    _RUNNING = "running"
    _CLOSING = "closing"
    _CLOSED = "closed"
    _FAILED = "failed"
    _ABANDONED = "abandoned"

    def __init__(self, path, capacity=1024, reservation=None):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if type(reservation) is not RunArtifactReservation:
            raise ValueError("a valid run artifact reservation is required")
        requested_path = Path(os.path.abspath(os.fspath(path)))
        if requested_path != reservation.trace_path:
            raise ValueError("reservation trace path does not match")
        with verify_reservation(
            reservation,
            reservation.run_id,
            results_root=reservation.results_root,
        ):
            pass
        self.path = requested_path
        self._reservation = reservation
        self._queue = queue.Queue(maxsize=capacity)
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._state = self._CREATED
        self._dropped = 0
        self._error = None
        self._close_result = None
        self._worker_succeeded = False
        self._worker_ready = False
        self._abandoned = False
        self._parent_fd = None
        self._temporary_fd = None
        self._temporary_name = None
        self._thread = threading.Thread(
            target=self._run,
            name="async-trace-writer",
            daemon=True,
        )

    @property
    def dropped(self):
        with self._lock:
            return self._dropped

    @property
    def error(self):
        with self._lock:
            if self._error is None:
                return None
            result = dict(self._error)
            if "secondary_errors" in result:
                result["secondary_errors"] = [
                    dict(error) for error in result["secondary_errors"]
                ]
            return result

    def start(self):
        with self._lock:
            if self._state != self._CREATED:
                raise RuntimeError("RequestTraceWriter.start() requires created state")
            try:
                self._prepare_temporary_file()
                self._state = self._RUNNING
                self._thread.start()
            except BaseException as exc:
                self._state = self._FAILED
                self._error = _safe_error("start", exc)
                self._cleanup_resources_locked()
                raise

    def write(self, trace):
        row = _trace_row(trace)
        with self._lock:
            if self._state != self._RUNNING:
                raise RuntimeError("RequestTraceWriter.write() requires running state")
            try:
                self._queue.put_nowait(row)
            except queue.Full:
                self._dropped += 1

    def close(self, timeout):
        if (
            type(timeout) not in (int, float)
            or type(timeout) is bool
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout must be a finite value >= 0")
        deadline = time.monotonic() + timeout
        with self._lock:
            if self._close_result is not None:
                return self._close_result
            if self._state == self._CREATED:
                raise RuntimeError("RequestTraceWriter.close() requires start()")
            if self._state == self._CLOSING:
                raise RuntimeError("RequestTraceWriter.close() is already in progress")
            if self._state == self._RUNNING:
                self._state = self._CLOSING
                self._closing.set()
                try:
                    self._queue.put_nowait(_STOP)
                except queue.Full:
                    pass

        remaining = max(0.0, deadline - time.monotonic())
        self._thread.join(remaining)
        if self._thread.is_alive():
            with self._lock:
                self._abandoned = True
                self._state = self._ABANDONED
                if self._error is None:
                    self._error = {
                        "phase": "close",
                        "error_type": "TimeoutError",
                        "error_message": "trace writer close deadline expired",
                    }
                self._close_result = False
            return False

        if not self._worker_ready:
            self._cleanup_caller_resources()
            with self._lock:
                self._close_result = False
                self._state = self._FAILED
                return False

        if time.monotonic() >= deadline:
            with self._lock:
                self._abandoned = True
                self._state = self._ABANDONED
                if self._error is None:
                    self._error = {
                        "phase": "close",
                        "error_type": "TimeoutError",
                        "error_message": "trace writer close deadline expired",
                    }
            self._cleanup_caller_resources()
            with self._lock:
                self._close_result = False
            return False

        publication_succeeded = self._publish_from_close(deadline)
        with self._lock:
            self._worker_succeeded = publication_succeeded
            self._close_result = publication_succeeded
            self._state = self._CLOSED if publication_succeeded else self._FAILED
            return self._close_result

    def _prepare_temporary_file(self):
        with verify_reservation(
            self._reservation,
            self._reservation.run_id,
            results_root=self._reservation.results_root,
        ) as verified:
            try:
                os.mkdir(
                    "traces",
                    mode=0o755,
                    dir_fd=verified.root.file_descriptor,
                )
            except FileExistsError:
                pass
            else:
                os.fsync(verified.root.file_descriptor)
            directory_flags = (
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            self._parent_fd = os.open(
                "traces",
                directory_flags,
                dir_fd=verified.root.file_descriptor,
            )
            if not reservation_binding_matches(verified) or not (
                directory_binding_matches(self.path.parent, self._parent_fd)
            ):
                raise OSError("trace parent directory changed during start")
            try:
                os.stat(
                    self.path.name,
                    dir_fd=self._parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError(
                    17,
                    "trace target already exists",
                    str(self.path),
                )
            self._temporary_name = (
                f".{self.path.name}.{uuid.uuid4().hex}.tmp"
            )
            self._temporary_fd = os.open(
                self._temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=self._parent_fd,
            )

    def _cleanup_resources_locked(self):
        if self._temporary_fd is not None:
            try:
                os.close(self._temporary_fd)
            except BaseException:
                pass
            self._temporary_fd = None
        if self._temporary_name is not None and self._parent_fd is not None:
            try:
                os.unlink(self._temporary_name, dir_fd=self._parent_fd)
            except BaseException:
                pass
            self._temporary_name = None
        if self._parent_fd is not None:
            try:
                os.close(self._parent_fd)
            except BaseException:
                pass
            self._parent_fd = None

    def _is_abandoned(self):
        with self._lock:
            return self._abandoned

    def _record_failure(
        self,
        phase,
        exc,
        *,
        temporary_file_may_remain=False,
        temporary_path=None,
        publication_state_uncertain=False,
        final_file_may_remain=False,
        final_path=None,
    ):
        diagnostic = _safe_error(phase, exc)
        if temporary_file_may_remain:
            diagnostic["temporary_file_may_remain"] = True
        if temporary_path is not None:
            diagnostic["temporary_path"] = str(temporary_path)
        if publication_state_uncertain:
            diagnostic["publication_state_uncertain"] = True
        if final_file_may_remain:
            diagnostic["final_file_may_remain"] = True
        if final_path is not None:
            diagnostic["final_path"] = str(final_path)
        with self._lock:
            if self._error is None:
                self._error = diagnostic
            else:
                self._error.setdefault("secondary_errors", []).append(diagnostic)
            if self._state != self._ABANDONED:
                self._state = self._FAILED
            self._closing.set()

    def _cleanup_caller_resources(self):
        if self._temporary_fd is not None:
            try:
                os.close(self._temporary_fd)
            except BaseException as exc:
                self._record_failure("close_descriptor", exc)
            self._temporary_fd = None
        if self._temporary_name is not None and self._parent_fd is not None:
            temporary_name = self._temporary_name
            try:
                os.unlink(temporary_name, dir_fd=self._parent_fd)
            except FileNotFoundError:
                self._temporary_name = None
            except BaseException as exc:
                self._record_failure(
                    "cleanup_temp",
                    exc,
                    temporary_file_may_remain=True,
                    temporary_path=self.path.parent / temporary_name,
                )
            else:
                self._temporary_name = None
                try:
                    os.fsync(self._parent_fd)
                except BaseException as exc:
                    self._record_failure(
                        "cleanup_directory_fsync",
                        exc,
                        temporary_file_may_remain=True,
                        temporary_path=self.path.parent / temporary_name,
                        publication_state_uncertain=True,
                    )
        if self._parent_fd is not None:
            parent_fd = self._parent_fd
            self._parent_fd = None
            try:
                os.close(parent_fd)
            except BaseException as exc:
                self._record_failure("close_directory", exc)

    def _publish_from_close(self, deadline):
        phase = "verify_reservation"
        final_published = False
        committed = False
        try:
            with verify_reservation(
                self._reservation,
                self._reservation.run_id,
                results_root=self._reservation.results_root,
            ) as verified:
                phase = "validate_parent_before_publish"
                if not reservation_binding_matches(verified) or not (
                    directory_binding_matches(self.path.parent, self._parent_fd)
                ):
                    raise OSError(
                        "trace parent directory changed before publication"
                    )
                phase = "close"
                self._require_publication_deadline(deadline)
                phase = "publish"
                link_no_overwrite(
                    self._temporary_name,
                    self.path.name,
                    source_directory_fd=self._parent_fd,
                    target_directory_fd=self._parent_fd,
                )
                final_published = True
                phase = "close"
                self._require_publication_deadline(deadline)
                phase = "validate_parent_after_publish"
                if not reservation_binding_matches(verified) or not (
                    directory_binding_matches(self.path.parent, self._parent_fd)
                ):
                    raise OSError(
                        "trace parent directory changed during publication"
                    )
                phase = "close"
                self._require_publication_deadline(deadline)
                phase = "cleanup_temp"
                os.unlink(self._temporary_name, dir_fd=self._parent_fd)
                self._temporary_name = None
                phase = "close"
                self._require_publication_deadline(deadline)
                phase = "directory_fsync"
                os.fsync(self._parent_fd)
                phase = "close"
                self._require_publication_deadline(deadline)
                phase = "validate_parent_after_fsync"
                if not reservation_binding_matches(verified) or not (
                    directory_binding_matches(self.path.parent, self._parent_fd)
                ):
                    raise OSError(
                        "trace parent directory changed during directory fsync"
                    )
                phase = "close"
                self._require_publication_deadline(deadline)
                committed = True
        except BaseException as exc:
            self._record_failure(phase, exc)
        finally:
            if final_published and not committed and self._parent_fd is not None:
                try:
                    os.unlink(self.path.name, dir_fd=self._parent_fd)
                except BaseException as exc:
                    self._record_failure(
                        "rollback_final",
                        exc,
                        publication_state_uncertain=True,
                        final_file_may_remain=True,
                        final_path=self.path,
                    )
                else:
                    try:
                        os.fsync(self._parent_fd)
                    except BaseException as exc:
                        self._record_failure(
                            "rollback_directory_fsync",
                            exc,
                            publication_state_uncertain=True,
                            final_file_may_remain=True,
                            final_path=self.path,
                        )
            self._cleanup_caller_resources()
        with self._lock:
            return committed and self._error is None

    @staticmethod
    def _require_publication_deadline(deadline):
        if time.monotonic() >= deadline:
            raise TimeoutError("trace writer close deadline expired")

    def _run(self):
        file_descriptor = self._temporary_fd
        self._temporary_fd = None
        handle = None
        phase = "open"
        try:
            handle = os.fdopen(file_descriptor, "w", encoding="utf-8")
            file_descriptor = None
            while True:
                row = self._queue.get()
                try:
                    if row is _STOP:
                        break
                    if not self._is_abandoned():
                        phase = "serialize"
                        line = json.dumps(
                            row,
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                        if not self._is_abandoned():
                            phase = "write"
                            handle.write(line + "\n")
                finally:
                    self._queue.task_done()
                if self._closing.is_set() and self._queue.empty():
                    break

            if self._is_abandoned():
                return
            phase = "flush"
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            handle = None
            if self._is_abandoned():
                return
            with self._lock:
                self._worker_ready = True
        except BaseException as exc:
            self._record_failure(phase, exc)
        finally:
            if file_descriptor is not None:
                try:
                    os.close(file_descriptor)
                except BaseException as exc:
                    self._record_failure("close_descriptor", exc)
            if handle is not None:
                try:
                    handle.close()
                except BaseException as exc:
                    self._record_failure("close_file", exc)
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._queue.task_done()
            if self._is_abandoned():
                self._cleanup_caller_resources()
