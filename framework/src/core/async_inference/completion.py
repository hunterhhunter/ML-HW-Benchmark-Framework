import logging
import queue
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable, Optional

from .metrics import derive_generation_timing
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
_MAX_ERROR_TYPE_LENGTH = 256


def _safe_error_type_name(error) -> str:
    try:
        name = type.__getattribute__(type(error), "__name__")
        if type(name) is not str:
            return "BaseException"
        return name[:_MAX_ERROR_TYPE_LENGTH] or "BaseException"
    except BaseException:
        return "BaseException"


def _exact_int(value) -> int:
    converted = int(value)
    return converted if type(converted) is int else int(str(converted))


@dataclass(frozen=True)
class _Reservation:
    attempt_token: int | None
    request: InferenceRequest


@dataclass
class _TerminalRecord:
    attempt_token: int | None = None
    token_bound: bool = False
    state: int = _TERMINAL_PENDING


@dataclass
class _CompletionHandoff:
    completion: BatchCompletion
    queued: object
    state: str = "ENQUEUING"
    producer_active: bool = False


@dataclass(frozen=True)
class _QueuedCompletion:
    operation_key: object
    completion: BatchCompletion


class _TerminalRecordView:
    def __init__(self, records, field, default):
        self._records = records
        self._field = field
        self._default = default

    def __len__(self):
        return len(self._records)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        record = self._records.get(index)
        return (
            self._default
            if record is None
            else getattr(record, self._field)
        )


def _safe_error_message(message) -> str:
    try:
        text = str(message)
    except BaseException:
        type_name = _safe_error_type_name(message)
        try:
            args = BaseException.__getattribute__(message, "args")
        except BaseException:
            args = ()
        safe_args = []
        try:
            for arg in args[:4]:
                if type(arg) is str:
                    safe_args.append(arg)
                else:
                    try:
                        safe_args.append(repr(arg))
                    except BaseException:
                        safe_args.append("<unprintable>")
        except BaseException:
            safe_args = []
        detail = ", ".join(safe_args)
        text = f"{type_name}({detail})" if detail else type_name
    try:
        return " ".join(text.split())[:512]
    except BaseException:
        return "unprintable error"


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
        queue_capacity: int | None,
        request_timeout_ms: float = 0.0,
        trace_callback: Optional[Callable[[RequestTrace], None]] = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        raise_callback_errors: bool = False,
    ):
        self.pipeline = pipeline
        self.evaluator = evaluator
        self.decoder = decoder
        self.metrics = metrics
        self.request_timeout_ns = int(request_timeout_ms * 1_000_000)
        self.trace_callback = trace_callback
        self.clock_ns = clock_ns
        self.raise_callback_errors = raise_callback_errors
        self.queue = (
            None
            if queue_capacity is None
            else queue.Queue(maxsize=queue_capacity)
        )
        self.condition = threading.Condition()
        self.reservations = {}
        self.outstanding = {}
        self._completion_handoffs = {}
        self.handoff_ack_callback = None
        self._terminal_records = {}
        self.terminal = _TerminalRecordView(
            self._terminal_records,
            "state",
            _TERMINAL_PENDING,
        )
        self.terminal_tokens = _TerminalRecordView(
            self._terminal_records,
            "attempt_token",
            None,
        )
        self.thread_error = None
        self.state = _COORDINATOR_RUNNING
        self._cleanup_started = False
        self.thread = (
            None
            if queue_capacity is None
            else threading.Thread(
                target=self._run,
                name="async-completion",
                daemon=True,
            )
        )

    def start(self) -> None:
        if self.thread is None:
            return
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
        reservation = self.reservations.get(request.request_id)
        if reservation is not None:
            if reservation.attempt_token == attempt_token:
                return
            raise ValueError(f"duplicate request_id: {request.request_id}")
        if (
            request.request_id in self.outstanding
            or self._terminal_record_locked(request.request_id) is not None
        ):
            raise ValueError(f"duplicate request_id: {request.request_id}")
        self.reservations[request.request_id] = _Reservation(
            attempt_token,
            request,
        )

    def _terminal_record_locked(self, request_id: int):
        if request_id < 0:
            return None
        return self._terminal_records.get(request_id)

    def _terminal_matches_locked(
        self,
        request_id: int,
        expected_token: int | None,
    ) -> bool:
        record = self._terminal_record_locked(request_id)
        return bool(
            record is not None
            and record.token_bound
            and record.attempt_token == expected_token
        )

    def _terminal_state_locked(
        self,
        request_id: int,
        expected_token: int | None,
    ):
        record = self._terminal_record_locked(request_id)
        if (
            record is None
            or not record.token_bound
            or record.attempt_token != expected_token
        ):
            return None
        return record.state

    def _allocate_terminal_record_locked(self, request_id: int):
        record = self._terminal_records.get(request_id)
        if record is None:
            record = _TerminalRecord()
            self._terminal_records[request_id] = record
        return record

    def _bind_terminal_token_locked(
        self,
        request_id: int,
        attempt_token: int | None,
    ) -> None:
        record = self._terminal_record_locked(request_id)
        if record is None:
            raise RuntimeError("terminal record is not allocated")
        if not record.token_bound:
            record.attempt_token = attempt_token
            record.token_bound = True
        elif record.attempt_token != attempt_token:
            raise RuntimeError(
                f"request {request_id} terminal token does not match"
            )

    def _publish_outstanding_locked(
        self,
        request: InferenceRequest,
        attempt_token: int | None,
    ) -> None:
        existing = self.outstanding.get(request.request_id)
        if existing is None:
            self.outstanding[request.request_id] = request
            return
        if not self._outstanding_matches_locked(
            request.request_id,
            attempt_token,
        ):
            raise RuntimeError(
                f"request {request.request_id} outstanding token does not match"
            )

    def _remove_reservation_locked(
        self,
        request_id: int,
        attempt_token: int | None,
    ) -> None:
        reservation = self.reservations.get(request_id)
        if reservation is None:
            return
        if reservation.attempt_token != attempt_token:
            raise RuntimeError(
                f"request {request_id} reservation token does not match"
            )
        if self.reservations.get(request_id) is reservation:
            self.reservations.pop(request_id, None)

    def _set_terminal_state_locked(self, request_id: int, state: int) -> None:
        record = self._terminal_record_locked(request_id)
        if record is None or not record.token_bound:
            raise RuntimeError("terminal record token is not bound")
        record.state = state
        self.condition.notify_all()

    def reserve_registration(
        self,
        request: InferenceRequest,
        attempt_token: int | None = None,
    ) -> None:
        request_id = _exact_int(request.request_id)
        request_token = (
            None
            if request.submission_token is None
            else _exact_int(request.submission_token)
        )
        normalized_token = (
            request_token
            if attempt_token is None
            else _exact_int(attempt_token)
        )
        if (
            type(request.request_id) is not int
            or (
                request.submission_token is not None
                and type(request.submission_token) is not int
            )
        ):
            request = replace(
                request,
                request_id=request_id,
                submission_token=request_token,
            )
        with self.condition:
            self._reserve_registration_locked(request, normalized_token)

    def _commit_registration_locked(
        self,
        request: InferenceRequest,
        expected_token=_UNSPECIFIED_TOKEN,
    ) -> None:
        if self.state != _COORDINATOR_RUNNING:
            raise RuntimeError(f"completion coordinator is {self.state}")
        reservation = self.reservations.get(request.request_id)
        attempt_token = (
            reservation.attempt_token
            if expected_token is _UNSPECIFIED_TOKEN and reservation is not None
            else (
                request.submission_token
                if expected_token is _UNSPECIFIED_TOKEN
                else expected_token
            )
        )
        has_ownership = bool(
            (
                reservation is not None
                and reservation.attempt_token == attempt_token
            )
            or self._outstanding_matches_locked(
                request.request_id,
                attempt_token,
            )
            or self._terminal_matches_locked(
                request.request_id,
                attempt_token,
            )
        )
        if not has_ownership:
            raise RuntimeError(
                f"request {request.request_id} registration ownership missing"
            )
        record = self._allocate_terminal_record_locked(request.request_id)
        self._bind_terminal_token_locked(
            request.request_id,
            attempt_token,
        )
        if record.state == _TERMINAL_PENDING:
            self._publish_outstanding_locked(request, attempt_token)
        self._remove_reservation_locked(
            request.request_id,
            attempt_token,
        )
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
        request_token = (
            None
            if request.submission_token is None
            else _exact_int(request.submission_token)
        )
        normalized_token = (
            expected_token
            if expected_token is _UNSPECIFIED_TOKEN
            else _exact_int(expected_token)
        )
        if (
            type(request.request_id) is not int
            or (
                request.submission_token is not None
                and type(request.submission_token) is not int
            )
        ):
            request = replace(
                request,
                request_id=request_id,
                submission_token=request_token,
            )
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

    def _outstanding_matches_locked(
        self,
        request_id: int,
        expected_token: int | None,
    ) -> bool:
        request = self.outstanding.get(request_id)
        return bool(
            request is not None
            and (
                (
                    request.submission_token is None
                    and expected_token is None
                )
                or (
                    type(request.submission_token) is int
                    and request.submission_token == expected_token
                )
            )
        )

    def register(self, request: InferenceRequest) -> None:
        request_id = _exact_int(request.request_id)
        request_token = (
            None
            if request.submission_token is None
            else _exact_int(request.submission_token)
        )
        if (
            type(request.request_id) is not int
            or (
                request.submission_token is not None
                and type(request.submission_token) is not int
            )
        ):
            request = replace(
                request,
                request_id=request_id,
                submission_token=request_token,
            )
        with self.condition:
            self._reserve_registration_locked(request, request_token)
            if self.state != _COORDINATOR_RUNNING:
                raise RuntimeError(f"completion coordinator is {self.state}")
            self._commit_registration_locked(request)

    def unregister_rejected(
        self,
        request_id: int,
        expected_token: int,
    ) -> bool:
        request_id = _exact_int(request_id)
        expected_token = _exact_int(expected_token)
        with self.condition:
            removed = False
            if self._reservation_matches_locked(request_id, expected_token):
                self.reservations.pop(request_id, None)
                removed = True
            if self._outstanding_matches_locked(request_id, expected_token):
                self.outstanding.pop(request_id, None)
                removed = True
            record = self._terminal_record_locked(request_id)
            if (
                record is not None
                and record.token_bound
                and record.attempt_token == expected_token
                and record.state == _TERMINAL_PENDING
            ):
                self._terminal_records.pop(request_id, None)
                removed = True
            if removed:
                self.condition.notify_all()
            return removed

    def submit(
        self,
        completion: BatchCompletion,
        timeout: float | None = None,
        *,
        operation_key=None,
    ) -> None:
        if self.queue is None:
            if operation_key is not None:
                raise ValueError(
                    "operation_key is not supported by inline completion"
                )
            with self.condition:
                if self.state != _COORDINATOR_RUNNING:
                    self._raise_unavailable_locked()
            self._handle(completion)
            return

        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        if operation_key is None:
            self._submit_unjournaled(completion, deadline)
            return

        while True:
            with self.condition:
                handoff = self._completion_handoffs.get(operation_key)
                if handoff is None:
                    queued = _QueuedCompletion(operation_key, completion)
                    handoff = _CompletionHandoff(completion, queued)
                    self._completion_handoffs[operation_key] = handoff
                elif handoff.completion is not completion:
                    raise RuntimeError("completion handoff ownership changed")
                if handoff.state in ("ENQUEUED", "DEQUEUED", "ACKED"):
                    return
                if self._handoff_is_queued_locked(handoff):
                    handoff.state = "ENQUEUED"
                    self.condition.notify_all()
                    return
                if self.state != _COORDINATOR_RUNNING:
                    self._raise_unavailable_locked()
                if handoff.producer_active:
                    self._wait_for_submission_locked(deadline)
                    continue
                handoff.producer_active = True

            try:
                self.queue.put(handoff.queued, block=False)
            except queue.Full:
                with self.condition:
                    handoff.producer_active = False
                    self.condition.notify_all()
                    self._wait_for_submission_locked(deadline)
                continue
            except BaseException:
                with self.condition:
                    if (
                        handoff.state == "ENQUEUING"
                        and self._handoff_is_queued_locked(handoff)
                    ):
                        handoff.state = "ENQUEUED"
                    handoff.producer_active = False
                    committed = handoff.state in (
                        "ENQUEUED",
                        "DEQUEUED",
                        "ACKED",
                    )
                    self.condition.notify_all()
                if committed:
                    return
                raise
            else:
                with self.condition:
                    if handoff.state == "ENQUEUING":
                        handoff.state = "ENQUEUED"
                    handoff.producer_active = False
                    self.condition.notify_all()
                return

    def _submit_unjournaled(self, completion, deadline) -> None:
        while True:
            with self.condition:
                if self.state != _COORDINATOR_RUNNING:
                    self._raise_unavailable_locked()
            try:
                self.queue.put(completion, block=False)
            except queue.Full:
                with self.condition:
                    self._wait_for_submission_locked(deadline)
            else:
                with self.condition:
                    self.condition.notify_all()
                return

    def _wait_for_submission_locked(self, deadline) -> None:
        if self.state != _COORDINATOR_RUNNING:
            self._raise_unavailable_locked()
        if deadline is None:
            self.condition.wait()
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("completion submission timed out")
        self.condition.wait(timeout=remaining)

    def _raise_unavailable_locked(self) -> None:
        if self.state == _COORDINATOR_FAILED:
            raise RuntimeError(
                f"completion coordinator failed: {self.thread_error}"
            )
        raise RuntimeError(f"completion coordinator is {self.state}")

    def _handoff_is_queued_locked(self, handoff) -> bool:
        with self.queue.mutex:
            return any(item is handoff.queued for item in self.queue.queue)

    def completion_handoff_committed(self, operation_key) -> bool:
        with self.condition:
            handoff = self._completion_handoffs.get(operation_key)
            if handoff is None:
                return False
            if handoff.state in ("ENQUEUED", "DEQUEUED", "ACKED"):
                return True
            if self._handoff_is_queued_locked(handoff):
                handoff.state = "ENQUEUED"
                return True
            return False

    def completion_handoff_state(self, operation_key):
        with self.condition:
            handoff = self._completion_handoffs.get(operation_key)
            if (
                handoff is not None
                and handoff.state != "ACKED"
                and self._handoff_terminal_locked(handoff)
            ):
                handoff.state = "ACKED"
                self.condition.notify_all()
            return None if handoff is None else handoff.state

    def _handoff_terminal_locked(self, handoff) -> bool:
        if handoff.state in ("ENQUEUING", "ENQUEUED"):
            if self._handoff_is_queued_locked(handoff):
                return False
            if self.state == _COORDINATOR_RUNNING:
                return False
        for request in handoff.completion.requests:
            request_id = request.request_id
            attempt_token = request.submission_token
            record = self._terminal_record_locked(request_id)
            if (
                record is None
                or record.state != _TERMINAL_COMMITTED
                or not record.token_bound
                or record.attempt_token != attempt_token
                or self._outstanding_matches_locked(
                    request_id,
                    attempt_token,
                )
            ):
                return False
        return True

    def wait_for_completion_handoff(
        self,
        operation_key,
        timeout: float,
    ) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        with self.condition:
            while True:
                handoff = self._completion_handoffs.get(operation_key)
                if handoff is None:
                    return False
                if handoff.state == "ACKED":
                    return True
                if self._handoff_terminal_locked(handoff):
                    handoff.state = "ACKED"
                    self.condition.notify_all()
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.condition.wait(timeout=remaining)

    def _mark_completion_handoff_acked_locked(self, operation_key) -> bool:
        handoff = self._completion_handoffs.get(operation_key)
        if handoff is None:
            return False
        if handoff.state == "ACKED":
            return True
        if handoff.state != "DEQUEUED":
            raise RuntimeError("completion handoff was not dequeued")
        handoff.state = "ACKED"
        self.condition.notify_all()
        return True

    def _retire_completion_handoff_locked(self, operation_key) -> bool:
        handoff = self._completion_handoffs.get(operation_key)
        if handoff is None:
            return False
        if handoff.state != "ACKED":
            return False
        self._completion_handoffs.pop(operation_key, None)
        self.condition.notify_all()
        return True

    def acknowledge_completion_handoff(self, operation_key) -> bool:
        with self.condition:
            return self._retire_completion_handoff_locked(operation_key)

    @property
    def completion_handoff_count(self) -> int:
        with self.condition:
            return len(self._completion_handoffs)

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
        if self.thread is None:
            with self.condition:
                stopped = not self.reservations and not self.outstanding
                if stopped:
                    self.state = _COORDINATOR_STOPPED
                self.condition.notify_all()
            if not stopped:
                self.metrics.add_invalid_reason("counter_invariant_failed")
            return stopped

        deadline = time.monotonic() + timeout
        never_started = False
        stopped_without_thread = False
        reservations_remain = False
        with self.condition:
            if self.thread.ident is None:
                reservations_remain = bool(self.reservations)
                self.state = _COORDINATOR_STOPPED
                self.condition.notify_all()
                never_started = True
            else:
                if self.state == _COORDINATOR_FAILED:
                    thread_failed = True
                    reservations_remain = bool(self.reservations)
                elif self.state == _COORDINATOR_STOPPED:
                    stopped_without_thread = True
                    reservations_remain = bool(self.reservations)
                    stopped = True
                elif self.state == _COORDINATOR_STOPPING:
                    while self.state == _COORDINATOR_STOPPING:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            return False
                        self.condition.wait(timeout=remaining)
                    stopped_without_thread = True
                    reservations_remain = bool(self.reservations)
                    stopped = self.state == _COORDINATOR_STOPPED
                else:
                    thread_failed = False
                    self.state = _COORDINATOR_STOPPING
                    self.condition.notify_all()
        if never_started:
            if reservations_remain:
                self.metrics.add_invalid_reason("counter_invariant_failed")
            return not reservations_remain
        if stopped_without_thread:
            if reservations_remain:
                self.metrics.add_invalid_reason("counter_invariant_failed")
            return stopped and not reservations_remain
        if thread_failed:
            self.thread.join(timeout=max(0.0, deadline - time.monotonic()))
            self.metrics.add_invalid_reason("completion_thread_failed")
            if reservations_remain:
                self.metrics.add_invalid_reason("counter_invariant_failed")
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
            with self.condition:
                reservations_remain = bool(self.reservations)
            self.metrics.add_invalid_reason("completion_thread_failed")
            if reservations_remain:
                self.metrics.add_invalid_reason("counter_invariant_failed")
            return False

        self.thread.join(timeout=max(0.0, deadline - time.monotonic()))
        with self.condition:
            thread_failed = self.state == _COORDINATOR_FAILED
            stopped = self.state == _COORDINATOR_STOPPED
            reservations_remain = bool(self.reservations)
        if (
            self.thread.is_alive()
            or thread_failed
            or not stopped
        ):
            self.metrics.add_invalid_reason("completion_thread_failed")
            if reservations_remain:
                self.metrics.add_invalid_reason("counter_invariant_failed")
            return False
        if reservations_remain:
            self.metrics.add_invalid_reason("counter_invariant_failed")
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
                    queued_handoff = (
                        item if isinstance(item, _QueuedCompletion) else None
                    )
                    completion = (
                        queued_handoff.completion
                        if queued_handoff is not None
                        else item
                    )
                    self._handle(completion)
                    if queued_handoff is not None:
                        with self.condition:
                            try:
                                self._mark_completion_handoff_acked_locked(
                                    queued_handoff.operation_key
                                )
                            except BaseException:
                                handoff = self._completion_handoffs.get(
                                    queued_handoff.operation_key
                                )
                                if handoff is None or handoff.state != "ACKED":
                                    raise
                        callback = self.handoff_ack_callback
                        if callback is not None:
                            self._notify_handoff_terminal(callback)
                finally:
                    self.queue.task_done()
                    item = None
                    queued_handoff = None
                    completion = None
        except BaseException as exc:
            error_type = _safe_error_type_name(exc)
            error_message = _safe_error_message(exc)
            thread_error = _safe_error_message(
                f"{error_type}: {error_message}"
            )
            try:
                LOGGER.error(
                    "async completion coordinator failed: %s",
                    thread_error,
                )
            except BaseException:
                pass
            with self.condition:
                self.thread_error = thread_error
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

    def _notify_handoff_terminal(self, callback=None) -> None:
        callback = self.handoff_ack_callback if callback is None else callback
        if callback is None:
            return
        try:
            callback()
        except BaseException:
            LOGGER.exception("completion handoff terminal callback failed")

    def _dequeue(self):
        while True:
            with self.queue.not_empty:
                while not self.queue._qsize():
                    self.queue.not_empty.wait()
            with self.condition:
                if self.state not in (
                    _COORDINATOR_RUNNING,
                    _COORDINATOR_STOPPING,
                ):
                    return False, None
                with self.queue.not_empty:
                    if not self.queue._qsize():
                        continue
                    candidate = self.queue.queue[0]
                    if isinstance(candidate, _QueuedCompletion):
                        handoff = self._completion_handoffs.get(
                            candidate.operation_key
                        )
                        if handoff is None or handoff.queued is not candidate:
                            raise RuntimeError(
                                "completion handoff ownership missing"
                            )
                        if handoff.state in ("ENQUEUING", "ENQUEUED"):
                            handoff.state = "DEQUEUED"
                        elif handoff.state not in ("DEQUEUED", "ACKED"):
                            raise RuntimeError(
                                "completion handoff state changed"
                            )
                    try:
                        item = self.queue._get()
                    except BaseException:
                        if any(
                            queued is candidate
                            for queued in self.queue.queue
                        ):
                            if isinstance(candidate, _QueuedCompletion):
                                handoff = self._completion_handoffs.get(
                                    candidate.operation_key
                                )
                                if (
                                    handoff is not None
                                    and handoff.state == "DEQUEUED"
                                ):
                                    handoff.state = "ENQUEUED"
                            raise
                        item = candidate
                    self.queue.not_full.notify()
                self.condition.notify_all()
                return True, item

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
            self._notify_handoff_terminal()

    def _fail_outstanding(self, error_type: str, error_message: str) -> None:
        with self.condition:
            requests = list(self.outstanding.values())

        for request in requests:
            already_terminal = False
            claimed_collision = False
            missing_record = False
            with self.condition:
                if request.request_id not in self.outstanding:
                    continue
                record = self._terminal_record_locked(request.request_id)
                if (
                    record is None
                    or not record.token_bound
                    or not self._outstanding_matches_locked(
                        request.request_id,
                        record.attempt_token,
                    )
                ):
                    self.outstanding.pop(request.request_id, None)
                    self.condition.notify_all()
                    missing_record = True
                else:
                    state = record.state
                    if state != _TERMINAL_PENDING:
                        claimed_collision = state == _TERMINAL_CLAIMED
                        self._set_terminal_state_locked(
                            request.request_id,
                            _TERMINAL_COMMITTED,
                        )
                        self.outstanding.pop(request.request_id, None)
                        self.condition.notify_all()
                        already_terminal = True
                    else:
                        self._set_terminal_state_locked(
                            request.request_id,
                            _TERMINAL_CLAIMED,
                        )

            if missing_record:
                self.metrics.add_invalid_reason("counter_invariant_failed")
                continue
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
                    self._set_terminal_state_locked(
                        request.request_id,
                        _TERMINAL_COMMITTED,
                    )
                    self.outstanding.pop(request.request_id, None)
                    self.condition.notify_all()

    def _handle(self, completion: BatchCompletion) -> None:
        normalized_requests = []
        for request in completion.requests:
            try:
                request_id = _exact_int(request.request_id)
                attempt_token = (
                    None
                    if request.submission_token is None
                    else _exact_int(request.submission_token)
                )
            except (TypeError, ValueError, OverflowError):
                normalized_requests.append((request, None, None, False))
            else:
                normalized_requests.append(
                    (request, request_id, attempt_token, True)
                )
        known = []
        seen_ids = set()
        membership_error = False
        membership_invalid_reasons = set()
        with self.condition:
            for request, request_id, attempt_token, valid in normalized_requests:
                if not valid:
                    membership_invalid_reasons.add("unknown_completion")
                    membership_error = True
                    continue
                if request_id in seen_ids:
                    membership_invalid_reasons.add("duplicate_completion")
                    membership_error = True
                    continue
                seen_ids.add(request_id)
                record = self._terminal_record_locked(request_id)
                outstanding = self.outstanding.get(request_id)
                if record is not None and record.attempt_token != attempt_token:
                    membership_invalid_reasons.add("stale_completion")
                    membership_error = True
                    continue
                if outstanding is not None and not (
                    type(outstanding.submission_token) is type(attempt_token)
                    and outstanding.submission_token == attempt_token
                ):
                    membership_invalid_reasons.add("stale_completion")
                    membership_error = True
                    continue
                if record is not None and record.state != _TERMINAL_PENDING:
                    membership_invalid_reasons.add("duplicate_completion")
                    membership_error = True
                    continue
                if outstanding is None or record is None:
                    membership_invalid_reasons.add("unknown_completion")
                    membership_error = True
                    continue
                known.append(outstanding)

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
            error_message = (
                "batch contained duplicate, unknown, or stale request ownership"
            )
        callback_error = None
        if error_type is None:
            try:
                outputs = completion.outputs
                if self.decoder is not None:
                    outputs = self.decoder.decode(outputs)
                labels = self.pipeline.prepare_eval_labels(completion.collated)
                self.evaluator.add_batch(outputs, labels, completion.timing_ms)
            except BaseException as exc:
                callback_error = exc
                error_type = _safe_error_type_name(exc)
                error_message = _safe_error_message(exc)
                if isinstance(exc, Exception):
                    try:
                        LOGGER.error(
                            "async decoder or evaluator failed: %s: %s",
                            error_type,
                            error_message,
                        )
                    except BaseException:
                        pass

        trace_generation_sample = None
        trace_generation_observation = None
        if error_type is None:
            generation_observation = completion.generation_observation
            if generation_observation is not None:
                try:
                    generation_events = generation_observation.events
                    final_event_ns = (
                        None
                        if not generation_events
                        else _exact_int(generation_events[-1].observed_ns)
                    )
                    runtime_finished_ns = _exact_int(
                        completion.runtime_finished_ns
                    )
                except (AttributeError, TypeError, ValueError, OverflowError):
                    pass
                else:
                    if (
                        final_event_ns is not None
                        and final_event_ns > runtime_finished_ns
                    ):
                        self.metrics.add_invalid_reason(
                            "timing_invariant_failed"
                        )
                        generation_observation = None
            trace_generation_sample, _ = derive_generation_timing(
                completion.generated_tokens,
                generation_observation,
                tuple(known),
            )
            if trace_generation_sample is not None:
                trace_generation_observation = generation_observation
            self.metrics.record_generation(
                completion.generated_tokens,
                completion.timing_ms,
                observation=generation_observation,
                requests=tuple(known),
            )

        completed_ns = self.clock_ns()
        trace_callback_error = None
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
                generated_tokens=(
                    _exact_int(completion.generated_tokens)
                    if error_type is None and len(known) == 1
                    else 0
                ),
                backend_submitted_ns=(
                    None
                    if trace_generation_observation is None
                    else trace_generation_observation.backend_submitted_ns
                ),
                generation_events=(
                    ()
                    if trace_generation_observation is None
                    else trace_generation_observation.events
                ),
                generation_timing_source=(
                    None
                    if trace_generation_sample is None
                    else trace_generation_sample.source
                ),
            )
            with self.condition:
                self._set_terminal_state_locked(
                    request.request_id,
                    _TERMINAL_CLAIMED,
                )
            self.metrics.record_terminal(trace)
            with self.condition:
                self._set_terminal_state_locked(
                    request.request_id,
                    _TERMINAL_COMMITTED,
                )
            try:
                if self.trace_callback is not None:
                    try:
                        self.trace_callback(trace)
                    except Exception:
                        self.metrics.add_warning("request_trace_write_failed")
                    except BaseException as exc:
                        if trace_callback_error is None:
                            trace_callback_error = exc
            finally:
                with self.condition:
                    self.outstanding.pop(request.request_id, None)
                    self.condition.notify_all()

        if (
            callback_error is not None
            and (
                not isinstance(callback_error, Exception)
                or (self.queue is None and self.raise_callback_errors)
            )
        ):
            raise callback_error
        if trace_callback_error is not None:
            raise trace_callback_error


def _reconcile_registration_internal(
    coordinator: CompletionCoordinator,
    request: InferenceRequest,
    attempt_token: int,
) -> None:
    CompletionCoordinator._commit_registration_locked(
        coordinator,
        request,
        attempt_token,
    )
