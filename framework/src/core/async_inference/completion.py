import logging
import queue
import threading
import time
from typing import Callable, Optional

from .types import (
    BatchCompletion,
    InferenceRequest,
    RequestTrace,
    TerminalStatus,
)


_STOP = object()
LOGGER = logging.getLogger(__name__)


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
        with self.lock:
            request = self.pending.get(event.request_id)
            if (
                request is None
                or event.request_id in self.events
                or event.first_token_ns < request.issued_ns
                or event.token_count <= 0
            ):
                self.metrics.add_invalid_reason("timing_invariant_failed")
                return False
            self.events[event.request_id] = event
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
        self.outstanding = {}
        self.terminal = bytearray()
        self.thread_error = None
        self.thread = threading.Thread(
            target=self._run,
            name="async-completion",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def register(self, request: InferenceRequest) -> None:
        with self.condition:
            if request.request_id < 0:
                raise ValueError("request_id must be non-negative")
            if request.request_id in self.outstanding or (
                request.request_id < len(self.terminal)
                and self.terminal[request.request_id]
            ):
                raise ValueError(f"duplicate request_id: {request.request_id}")
            self.outstanding[request.request_id] = request
            required = request.request_id + 1 - len(self.terminal)
            if required > 0:
                self.terminal.extend(b"\x00" * required)

    def unregister_rejected(self, request_id: int) -> None:
        with self.condition:
            self.outstanding.pop(request_id, None)
            self.condition.notify_all()

    def submit(self, completion: BatchCompletion) -> None:
        self.queue.put(completion)

    def wait_for_all(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.outstanding and self.thread_error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.metrics.add_invalid_reason("flush_timeout")
                    return False
                self.condition.wait(timeout=remaining)
            if self.thread_error is not None:
                self.metrics.add_invalid_reason("completion_thread_failed")
                return False
            return True

    def stop(self, timeout: float) -> bool:
        try:
            self.queue.put(_STOP, timeout=timeout)
        except queue.Full:
            self.metrics.add_invalid_reason("completion_thread_failed")
            return False
        self.thread.join(timeout=timeout)
        if self.thread.is_alive() or self.thread_error is not None:
            self.metrics.add_invalid_reason("completion_thread_failed")
            return False
        return True

    def _run(self) -> None:
        try:
            while True:
                item = self.queue.get()
                try:
                    if item is _STOP:
                        return
                    self._handle(item)
                finally:
                    self.queue.task_done()
        except BaseException as exc:
            LOGGER.exception("async completion coordinator failed")
            with self.condition:
                self.thread_error = (
                    f"{type(exc).__name__}: {_safe_error_message(exc)}"
                )
                self.metrics.add_invalid_reason("completion_thread_failed")
                self.condition.notify_all()

    def _handle(self, completion: BatchCompletion) -> None:
        known = []
        seen_ids = set()
        membership_error = False
        with self.condition:
            for request in completion.requests:
                request_id = request.request_id
                if request_id in seen_ids:
                    self.metrics.add_invalid_reason("duplicate_completion")
                    membership_error = True
                    continue
                seen_ids.add(request_id)
                if (
                    0 <= request_id < len(self.terminal)
                    and self.terminal[request_id]
                ):
                    self.metrics.add_invalid_reason("duplicate_completion")
                    membership_error = True
                    continue
                if request_id not in self.outstanding:
                    self.metrics.add_invalid_reason("unknown_completion")
                    membership_error = True
                    continue
                known.append(self.outstanding[request_id])

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
            self.metrics.record_terminal(trace)
            if self.trace_callback is not None:
                try:
                    self.trace_callback(trace)
                except Exception:
                    self.metrics.add_warning("request_trace_write_failed")
            with self.condition:
                self.terminal[request.request_id] = 1
                self.outstanding.pop(request.request_id, None)
                self.condition.notify_all()
