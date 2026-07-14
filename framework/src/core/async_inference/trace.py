import json
import math
import os
import queue
import tempfile
import threading
import time
from pathlib import Path

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

    def __init__(self, path, capacity=1024):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.path = Path(path)
        self._queue = queue.Queue(maxsize=capacity)
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._state = self._CREATED
        self._dropped = 0
        self._error = None
        self._close_result = None
        self._worker_succeeded = False
        self._abandoned = False
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
            return None if self._error is None else dict(self._error)

    def start(self):
        with self._lock:
            if self._state != self._CREATED:
                raise RuntimeError("RequestTraceWriter.start() requires created state")
            self._state = self._RUNNING
            try:
                self._thread.start()
            except BaseException as exc:
                self._state = self._FAILED
                self._error = _safe_error("start", exc)
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

        with self._lock:
            self._close_result = self._worker_succeeded
            self._state = self._CLOSED if self._worker_succeeded else self._FAILED
            return self._close_result

    def _is_abandoned(self):
        with self._lock:
            return self._abandoned

    def _record_failure(self, phase, exc):
        with self._lock:
            if self._error is None:
                self._error = _safe_error(phase, exc)
            if self._state != self._ABANDONED:
                self._state = self._FAILED
            self._closing.set()

    def _run(self):
        temporary_path = None
        file_descriptor = None
        handle = None
        phase = "mkdir"
        published = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            phase = "open"
            file_descriptor, temporary_name = tempfile.mkstemp(
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
            )
            temporary_path = Path(temporary_name)
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
            phase = "replace"
            os.replace(temporary_path, self.path)
            published = True
            phase = "directory_fsync"
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            with self._lock:
                self._worker_succeeded = True
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
            if temporary_path is not None and not published:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
                except BaseException as exc:
                    self._record_failure("cleanup", exc)
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._queue.task_done()
