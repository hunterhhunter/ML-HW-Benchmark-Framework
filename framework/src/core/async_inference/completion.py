import logging
import queue
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable, Optional

from .types import (
    BatchCompletion,
    InferenceRequest,
    RequestTrace,
    TerminalStatus,
)


_STOP = object()
_TERMINAL_PENDING = 0
_TERMINAL_CLAIMED = 1
_TERMINAL_COMMITTED = 2
_COORDINATOR_RUNNING = "running"
_COORDINATOR_STOPPING = "stopping"
_COORDINATOR_FAILED = "failed"
_COORDINATOR_STOPPED = "stopped"
LOGGER = logging.getLogger(__name__)
_UNSPECIFIED_TOKEN = object()


def _exact_int(value) -> int:
    converted = int(value)
    return converted if type(converted) is int else int(str(converted))


@dataclass(frozen=True)
class _Reservation:
    attempt_token: int | None
    request: InferenceRequest


def _safe_error_message(message) -> str:
    return " ".join(str(message).split())[:512]


class FirstTokenTracker:
    """Lifecycle contract for future real streaming runtime integrations."""

    def __init__(self, metrics):
        self.metrics = metrics
        self.pending = {}
        self.events = {}
        self.lock = threading.Lock()

    def register(self, request: InferenceRequest) -> None:
        with self.lock:
            self.pending[request.request_id] = request

    def record(self, event) -> bool:
        invalid = False
        request = None
        with self.lock:
            request = self.pending.get(event.request_id)
            if (
                request is None
                or event.request_id in self.events
                or event.first_token_ns < request.issued_ns
                or event.token_count <= 0
            ):
                invalid = True
            else:
                self.events[event.request_id] = event
        if invalid:
            self.metrics.add_invalid_reason("timing_invariant_failed")
            return False
        self.metrics.record_first_token(request, event)
        return True

    def finalize(self, request_id: int, generated_tokens: int) -> bool:
        with self.lock:
            request = self.pending.pop(request_id, None)
            event = self.events.pop(request_id, None)
        valid = request is not None and (
            (event is None and generated_tokens == 0)
            or (event is not None and generated_tokens >= event.token_count)
        )
        if not valid:
            self.metrics.add_invalid_reason("timing_invariant_failed")
        return valid


class CompletionCoordinator:
    def __init__(
        self,
        pipeline,
        evaluator,
        decoder,
        metrics,
        queue_capacity: int,
        request_timeout_ms: float = 0.0,
        trace_callback: Optional[Callable[[RequestTrace], None]] = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ):
        self.pipeline = pipeline
        self.evaluator = evaluator
        self.decoder = decoder
        self.metrics = metrics
        self.request_timeout_ns = int(request_timeout_ms * 1_000_000)
        self.trace_callback = trace_callback
        self.clock_ns = clock_ns
        self.queue = queue.Queue(maxsize=queue_capacity)
        self.condition = threading.Condition()
        self.reservations = {}
        self.outstanding = {}
        self.terminal = bytearray()
        self.thread_error = None
        self.state = _COORDINATOR_RUNNING
        self._cleanup_started = False
        self.thread = threading.Thread(
            target=self._run,
            name="async-completion",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def _reserve_registration_locked(
        self,
        request: InferenceRequest,
        attempt_token: int | None = None,
    ) -> None:
        if self.state != _COORDINATOR_RUNNING:
            raise RuntimeError(f"completion coordinator is {self.state}")
        if request.request_id < 0:
            raise ValueError("request_id must be non-negative")
        if (
            request.request_id in self.reservations
            or request.request_id in self.outstanding
            or (
                request.request_id < len(self.terminal)
                and self.terminal[request.request_id]
            )
        ):
            raise ValueError(f"duplicate request_id: {request.request_id}")
        self.reservations[request.request_id] = _Reservation(
            attempt_token,
            request,
        )
        required = request.request_id + 1 - len(self.terminal)
        if required > 0:
            self.terminal.extend(b"\x00" * required)

    def reserve_registration(
        self,
        request: InferenceRequest,
        attempt_token: int | None = None,
    ) -> None:
        request_id = _exact_int(request.request_id)
        normalized_token = (
            None if attempt_token is None else _exact_int(attempt_token)
        )
        if type(request.request_id) is not int:
            request = replace(request, request_id=request_id)
        with self.condition:
            self._reserve_registration_locked(request, normalized_token)

    def _commit_registration_locked(
        self,
        request: InferenceRequest,
        expected_token=_UNSPECIFIED_TOKEN,
    ) -> None:
        self._validate_registration_locked(request.request_id, expected_token)
        self.outstanding[request.request_id] = request
        self.reservations.pop(request.request_id, None)
        self.condition.notify_all()

    def _validate_registration_locked(
        self,
        request_id: int,
        expected_token=_UNSPECIFIED_TOKEN,
    ) -> None:
        if self.state != _COORDINATOR_RUNNING:
            raise RuntimeError(f"completion coordinator is {self.state}")
        reserved = self.reservations.get(request_id)
        if reserved is None:
            raise RuntimeError(
                f"request {request_id} registration is not reserved"
            )
        if (
            expected_token is not _UNSPECIFIED_TOKEN
            and reserved.attempt_token != expected_token
        ):
            raise RuntimeError(
                f"request {request_id} reservation token does not match"
            )

    def commit_registration(
        self,
        request: InferenceRequest,
        expected_token=_UNSPECIFIED_TOKEN,
    ) -> None:
        request_id = _exact_int(request.request_id)
        normalized_token = (
            expected_token
            if expected_token is _UNSPECIFIED_TOKEN
            else _exact_int(expected_token)
        )
        if type(request.request_id) is not int:
            request = replace(request, request_id=request_id)
        with self.condition:
            self._commit_registration_locked(request, normalized_token)

    def abort_registration(
        self,
        request_id: int,
        expected_token=_UNSPECIFIED_TOKEN,
    ) -> bool:
        request_id = _exact_int(request_id)
        normalized_token = (
            expected_token
            if expected_token is _UNSPECIFIED_TOKEN
            else _exact_int(expected_token)
        )
        with self.condition:
            reservation = self.reservations.get(request_id)
            if reservation is None or (
                normalized_token is not _UNSPECIFIED_TOKEN
                and reservation.attempt_token != normalized_token
            ):
                return False
            removed = self.reservations.pop(request_id, None) is reservation
            self.condition.notify_all()
            return removed

    def _reservation_matches_locked(
        self,
        request_id: int,
        expected_token: int,
    ) -> bool:
        reservation = self.reservations.get(request_id)
        return bool(
            reservation is not None
            and reservation.attempt_token == expected_token
        )

    def register(self, request: InferenceRequest) -> None:
        request_id = _exact_int(request.request_id)
        if type(request.request_id) is not int:
            request = replace(request, request_id=request_id)
        with self.condition:
            self._reserve_registration_locked(request, None)
            if self.state != _COORDINATOR_RUNNING:
                raise RuntimeError(f"completion coordinator is {self.state}")
            self._commit_registration_locked(request)

    def unregister_rejected(self, request_id: int) -> None:
        with self.condition:
            self.reservations.pop(request_id, None)
            self.outstanding.pop(request_id, None)
            self.condition.notify_all()

    def submit(
        self,
        completion: BatchCompletion,
        timeout: float | None = None,
    ) -> None:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self.condition:
            while self.state == _COORDINATOR_RUNNING:
                try:
                    self.queue.put_nowait(completion)
                    self.condition.notify_all()
                    return
                except queue.Full:
                    if deadline is None:
                        self.condition.wait()
                        continue
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("completion submission timed out")
                    self.condition.wait(timeout=remaining)
            if self.state == _COORDINATOR_FAILED:
                raise RuntimeError(
                    f"completion coordinator failed: {self.thread_error}"
                )
            raise RuntimeError(f"completion coordinator is {self.state}")

    def wait_for_all(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        invalid_reason = None
        with self.condition:
            while self.outstanding and self.thread_error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    invalid_reason = "flush_timeout"
                    break
                self.condition.wait(timeout=remaining)
            if invalid_reason is None and self.thread_error is not None:
                invalid_reason = "completion_thread_failed"
        if invalid_reason is not None:
            self.metrics.add_invalid_reason(invalid_reason)
            return False
        return True

    def snapshot_outstanding(self):
        with self.condition:
            return tuple(self.outstanding)

    def wait_for_requests(self, request_ids, timeout: float) -> bool:
        pending = frozenset(request_ids)
        deadline = time.monotonic() + timeout
        invalid_reason = None
        with self.condition:
            while (
                any(request_id in self.outstanding for request_id in pending)
                and self.thread_error is None
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    invalid_reason = "flush_timeout"
                    break
                self.condition.wait(timeout=remaining)
            if invalid_reason is None and self.thread_error is not None:
                invalid_reason = "completion_thread_failed"
        if invalid_reason is not None:
            self.metrics.add_invalid_reason(invalid_reason)
            return False
        return True

    def stop(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self.condition:
            if self.state == _COORDINATOR_FAILED:
                thread_failed = True
            elif self.state == _COORDINATOR_STOPPED:
                return True
            elif self.state == _COORDINATOR_STOPPING:
                while self.state == _COORDINATOR_STOPPING:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self.condition.wait(timeout=remaining)
                return self.state == _COORDINATOR_STOPPED
            else:
                thread_failed = False
                self.state = _COORDINATOR_STOPPING
                self.condition.notify_all()
        if thread_failed:
            self.thread.join(timeout=max(0.0, deadline - time.monotonic()))
            self.metrics.add_invalid_reason("completion_thread_failed")
            return False

        sentinel_enqueued = False
        stop_failed = False
        with self.condition:
            while self.state == _COORDINATOR_STOPPING:
                try:
                    self.queue.put_nowait(_STOP)
                except queue.Full:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self.state = _COORDINATOR_FAILED
                        self.thread_error = (
                            "TimeoutError: completion stop queue was full"
                        )
                        stop_failed = True
                        self.condition.notify_all()
                        break
                    self.condition.wait(timeout=remaining)
                else:
                    sentinel_enqueued = True
                    self.condition.notify_all()
                    break

        if stop_failed:
            self.metrics.add_invalid_reason("completion_thread_failed")

        if not sentinel_enqueued:
            self.thread.join(timeout=max(0.0, deadline - time.monotonic()))
            self.metrics.add_invalid_reason("completion_thread_failed")
            return False

        self.thread.join(timeout=max(0.0, deadline - time.monotonic()))
        with self.condition:
            thread_failed = self.state == _COORDINATOR_FAILED
            stopped = self.state == _COORDINATOR_STOPPED
        if self.thread.is_alive() or thread_failed or not stopped:
            self.metrics.add_invalid_reason("completion_thread_failed")
            return False
        return True

    def _run(self) -> None:
        normal_stop = False
        try:
            while True:
                has_item, item = self._dequeue()
                if not has_item:
                    break
                try:
                    if item is _STOP:
                        normal_stop = True
                        break
                    self._handle(item)
                finally:
                    self.queue.task_done()
                    item = None
        except BaseException as exc:
            LOGGER.exception("async completion coordinator failed")
            with self.condition:
                self.thread_error = _safe_error_message(
                    f"{type(exc).__name__}: {exc}"
                )
                self.state = _COORDINATOR_FAILED
                self.condition.notify_all()
            self.metrics.add_invalid_reason("completion_thread_failed")
        finally:
            with self.condition:
                failed = self.state == _COORDINATOR_FAILED
                error_message = self.thread_error
            if failed:
                self._finalize(
                    final_state=_COORDINATOR_FAILED,
                    error_type="CompletionThreadError",
                    error_message=error_message
                    or "completion coordinator failed",
                )
            elif normal_stop:
                self._finalize(
                    final_state=_COORDINATOR_STOPPED,
                    error_type="CompletionStopped",
                    error_message=(
                        "completion coordinator stopped before request completion"
                    ),
                )

    def _dequeue(self):
        with self.condition:
            while self.state in (
                _COORDINATOR_RUNNING,
                _COORDINATOR_STOPPING,
            ):
                try:
                    item = self.queue.get_nowait()
                except queue.Empty:
                    self.condition.wait()
                else:
                    self.condition.notify_all()
                    return True, item
            return False, None

    def _finalize(
        self,
        final_state: str,
        error_type: str,
        error_message: str,
    ) -> None:
        with self.condition:
            if self._cleanup_started:
                return
            self._cleanup_started = True
            while True:
                try:
                    queued = self.queue.get_nowait()
                except queue.Empty:
                    break
                del queued
                self.queue.task_done()
            self.condition.notify_all()

        try:
            self._fail_outstanding(
                error_type=error_type,
                error_message=error_message,
            )
        finally:
            with self.condition:
                if self.state != _COORDINATOR_FAILED:
                    self.state = final_state
                self.condition.notify_all()

    def _fail_outstanding(self, error_type: str, error_message: str) -> None:
        with self.condition:
            requests = list(self.outstanding.values())

        for request in requests:
            already_terminal = False
            claimed_collision = False
            with self.condition:
                if request.request_id not in self.outstanding:
                    continue
                state = self.terminal[request.request_id]
                if state != _TERMINAL_PENDING:
                    claimed_collision = state == _TERMINAL_CLAIMED
                    self.terminal[request.request_id] = _TERMINAL_COMMITTED
                    self.outstanding.pop(request.request_id, None)
                    self.condition.notify_all()
                    already_terminal = True
                else:
                    self.terminal[request.request_id] = _TERMINAL_CLAIMED

            if claimed_collision:
                self.metrics.add_invalid_reason("counter_invariant_failed")
            if already_terminal:
                continue

            try:
                completed_ns = max(self.clock_ns(), request.enqueued_ns)
                trace = RequestTrace(
                    request_id=request.request_id,
                    sample_index=request.sample_index,
                    status=TerminalStatus.FAILED,
                    scheduled_ns=request.scheduled_ns,
                    issued_ns=request.issued_ns,
                    enqueued_ns=request.enqueued_ns,
                    runtime_started_ns=completed_ns,
                    runtime_finished_ns=completed_ns,
                    completed_ns=completed_ns,
                    worker_id=-1,
                    batch_size=request.sample_count,
                    timed_out=bool(
                        self.request_timeout_ns
                        and completed_ns - request.issued_ns
                        > self.request_timeout_ns
                    ),
                    sample_count=request.sample_count,
                    error_type=error_type,
                    error_message=error_message,
                )
                self.metrics.record_terminal(trace)
            except BaseException:
                LOGGER.exception("failed to record crash terminal state")
                self.metrics.add_invalid_reason("counter_invariant_failed")
            else:
                if self.trace_callback is not None:
                    try:
                        self.trace_callback(trace)
                    except BaseException:
                        self.metrics.add_warning("request_trace_write_failed")
            finally:
                with self.condition:
                    self.terminal[request.request_id] = _TERMINAL_COMMITTED
                    self.outstanding.pop(request.request_id, None)
                    self.condition.notify_all()

    def _handle(self, completion: BatchCompletion) -> None:
        known = []
        seen_ids = set()
        membership_error = False
        membership_invalid_reasons = set()
        with self.condition:
            for request in completion.requests:
                request_id = request.request_id
                if request_id in seen_ids:
                    membership_invalid_reasons.add("duplicate_completion")
                    membership_error = True
                    continue
                seen_ids.add(request_id)
                if (
                    0 <= request_id < len(self.terminal)
                    and self.terminal[request_id]
                ):
                    membership_invalid_reasons.add("duplicate_completion")
                    membership_error = True
                    continue
                if request_id not in self.outstanding:
                    membership_invalid_reasons.add("unknown_completion")
                    membership_error = True
                    continue
                known.append(self.outstanding[request_id])

        for reason in membership_invalid_reasons:
            self.metrics.add_invalid_reason(reason)

        if not known:
            return

        error_type = completion.error_type
        error_message = (
            _safe_error_message(completion.error_message)
            if completion.error_message is not None
            else None
        )
        if membership_error:
            error_type = "InvalidCompletionMembership"
            error_message = "batch contained duplicate or unknown request IDs"
        if error_type is None:
            try:
                outputs = completion.outputs
                if self.decoder is not None:
                    outputs = self.decoder.decode(outputs)
                labels = self.pipeline.prepare_eval_labels(completion.collated)
                self.evaluator.add_batch(outputs, labels, completion.timing_ms)
            except Exception as exc:
                LOGGER.exception("async decoder or evaluator failed")
                error_type = type(exc).__name__
                error_message = _safe_error_message(exc)

        if error_type is None:
            self.metrics.record_generation(
                completion.generated_tokens,
                completion.timing_ms,
            )

        completed_ns = self.clock_ns()
        for request in known:
            elapsed_ns = completed_ns - request.issued_ns
            timed_out = bool(
                self.request_timeout_ns
                and elapsed_ns > self.request_timeout_ns
            )
            status = (
                TerminalStatus.COMPLETED
                if error_type is None
                else TerminalStatus.FAILED
            )
            trace = RequestTrace(
                request_id=request.request_id,
                sample_index=request.sample_index,
                status=status,
                scheduled_ns=request.scheduled_ns,
                issued_ns=request.issued_ns,
                enqueued_ns=request.enqueued_ns,
                runtime_started_ns=completion.runtime_started_ns,
                runtime_finished_ns=completion.runtime_finished_ns,
                completed_ns=completed_ns,
                worker_id=completion.worker_id,
                batch_size=completion.batch_size,
                timed_out=timed_out,
                sample_count=request.sample_count,
                error_type=error_type,
                error_message=error_message,
            )
            with self.condition:
                self.terminal[request.request_id] = _TERMINAL_CLAIMED
            self.metrics.record_terminal(trace)
            with self.condition:
                self.terminal[request.request_id] = _TERMINAL_COMMITTED
            if self.trace_callback is not None:
                try:
                    self.trace_callback(trace)
                except Exception:
                    self.metrics.add_warning("request_trace_write_failed")
            with self.condition:
                self.outstanding.pop(request.request_id, None)
                self.condition.notify_all()
