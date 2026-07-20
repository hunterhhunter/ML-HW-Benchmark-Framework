import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RuntimeExecution:
    outputs: Optional[Dict[str, Any]]
    timing_ms: float | Dict[str, Any] | None
    generated_tokens: int = 0
    dispatch_token: Optional[int] = None
    vendor_job_id: Any = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class NativeAsyncOutcome:
    outputs: Optional[Dict[str, Any]] = None
    timing_ms: float | Dict[str, Any] | None = None
    generated_tokens: int = 0
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class NativeAsyncExecutorSnapshot:
    inflight: int
    duplicate_callbacks: int
    late_callbacks: int
    submit_failures: int
    timeouts: int


@dataclass
class _NativeDispatch:
    token: int
    inputs: Any
    event: threading.Event = field(default_factory=threading.Event)
    vendor_job_id: Any = None
    outcome: NativeAsyncOutcome | None = None
    terminal_kind: str | None = None
    acknowledged: bool = False


class RuntimeExecutor(ABC):
    @abstractmethod
    def execute(self, inputs, timeout=None) -> RuntimeExecution:
        raise NotImplementedError

    @abstractmethod
    def acknowledge(self, execution: RuntimeExecution) -> None:
        raise NotImplementedError

    @abstractmethod
    def shutdown(self, timeout: float) -> bool:
        raise NotImplementedError


class BlockingRuntimeExecutor(RuntimeExecutor):
    def __init__(
        self,
        runtime,
        *,
        is_llm: bool,
        max_new_tokens: int = 256,
        stop_token_ids=None,
    ):
        self.runtime = runtime
        self.is_llm = is_llm
        self.max_new_tokens = max_new_tokens
        self.stop_token_ids = stop_token_ids

    def execute(self, inputs, timeout=None) -> RuntimeExecution:
        if self.is_llm:
            result = self.runtime.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                stop_token_ids=self.stop_token_ids,
            )
            outputs = {"generated_ids": result.generated_ids}
            if result.generated_lengths is not None:
                outputs["generated_lengths"] = result.generated_lengths
            timing = {
                "total_ms": result.total_ms,
                "timing_mode": result.timing_mode,
                "uses_kv_cache": result.uses_kv_cache,
                "timing_source": result.timing_source,
            }
            if result.ttft_ms is not None:
                timing["ttft_ms"] = result.ttft_ms
            if result.tpot_ms is not None:
                timing["tpot_ms"] = result.tpot_ms
            return RuntimeExecution(
                outputs=outputs,
                timing_ms=timing,
                generated_tokens=result.num_tokens,
            )

        started = time.perf_counter()
        outputs = self.runtime.run(inputs)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return RuntimeExecution(outputs=outputs, timing_ms=elapsed_ms)

    def acknowledge(self, execution: RuntimeExecution) -> None:
        return None

    def shutdown(self, timeout: float) -> bool:
        return True


class NativeAsyncRuntimeExecutor(RuntimeExecutor):
    """Bridge a callback-based vendor API to the framework executor contract."""

    def __init__(
        self,
        backend,
        *,
        max_inflight: int,
        completion_timeout_sec: float,
    ):
        if max_inflight <= 0:
            raise ValueError("max_inflight must be positive")
        if completion_timeout_sec <= 0:
            raise ValueError("completion_timeout_sec must be positive")
        self.backend = backend
        self.max_inflight = int(max_inflight)
        self.completion_timeout_sec = float(completion_timeout_sec)
        self._permits = threading.BoundedSemaphore(self.max_inflight)
        self._condition = threading.Condition(threading.RLock())
        self._dispatches: dict[int, _NativeDispatch] = {}
        self._acknowledged_tokens: set[int] = set()
        self._next_dispatch_token = 1
        self._duplicate_callbacks = 0
        self._late_callbacks = 0
        self._submit_failures = 0
        self._timeouts = 0
        self._shutdown_requested = False

    def execute(self, inputs, timeout=None) -> RuntimeExecution:
        wait_timeout = self.completion_timeout_sec
        if timeout is not None:
            wait_timeout = min(wait_timeout, max(0.0, float(timeout)))
        if not self._permits.acquire(timeout=wait_timeout):
            return RuntimeExecution(
                outputs=None,
                timing_ms=None,
                error_type="NativeAsyncBackpressureTimeout",
                error_message="timed out waiting for a native async inflight slot",
            )

        with self._condition:
            if self._shutdown_requested:
                self._permits.release()
                return RuntimeExecution(
                    outputs=None,
                    timing_ms=None,
                    error_type="NativeAsyncShutdown",
                    error_message="native async executor is shutting down",
                )
            token = self._next_dispatch_token
            self._next_dispatch_token += 1
            dispatch = _NativeDispatch(token=token, inputs=inputs)
            self._dispatches[token] = dispatch

        def callback(outcome: NativeAsyncOutcome) -> None:
            if not isinstance(outcome, NativeAsyncOutcome):
                outcome = NativeAsyncOutcome(
                    error_type="NativeAsyncProtocolError",
                    error_message="native callback returned an invalid outcome",
                )
            with self._condition:
                if dispatch.terminal_kind is not None:
                    if dispatch.terminal_kind == "timeout":
                        self._late_callbacks += 1
                    else:
                        self._duplicate_callbacks += 1
                    return
                dispatch.outcome = outcome
                dispatch.terminal_kind = "callback"
                dispatch.event.set()
                self._condition.notify_all()

        try:
            vendor_job_id = self.backend.submit_async(inputs, callback)
        except BaseException as exc:
            with self._condition:
                dispatch.outcome = NativeAsyncOutcome(
                    error_type=type(exc).__name__,
                    error_message="native async submission failed",
                )
                dispatch.terminal_kind = "submit_failure"
                self._submit_failures += 1
                dispatch.event.set()
                self._condition.notify_all()
        else:
            with self._condition:
                dispatch.vendor_job_id = vendor_job_id

        if not dispatch.event.wait(timeout=wait_timeout):
            with self._condition:
                if dispatch.terminal_kind is None:
                    dispatch.outcome = NativeAsyncOutcome(
                        error_type="NativeAsyncTimeout",
                        error_message="native async completion timed out",
                    )
                    dispatch.terminal_kind = "timeout"
                    self._timeouts += 1
                    dispatch.event.set()
                    self._condition.notify_all()

        with self._condition:
            outcome = dispatch.outcome
            vendor_job_id = dispatch.vendor_job_id
        if outcome is None:
            outcome = NativeAsyncOutcome(
                error_type="NativeAsyncProtocolError",
                error_message="native async dispatch ended without an outcome",
            )
        return RuntimeExecution(
            outputs=outcome.outputs,
            timing_ms=outcome.timing_ms,
            generated_tokens=outcome.generated_tokens,
            dispatch_token=token,
            vendor_job_id=vendor_job_id,
            error_type=outcome.error_type,
            error_message=outcome.error_message,
        )

    def acknowledge(self, execution: RuntimeExecution) -> None:
        token = execution.dispatch_token
        if token is None:
            return
        with self._condition:
            if token in self._acknowledged_tokens:
                return
            dispatch = self._dispatches.pop(token, None)
            if dispatch is None:
                raise RuntimeError(f"unknown native async dispatch token: {token}")
            dispatch.acknowledged = True
            dispatch.inputs = None
            dispatch.outcome = None
            self._acknowledged_tokens.add(token)
            self._condition.notify_all()
        self._permits.release()

    def shutdown(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            self._shutdown_requested = True
            while self._dispatches:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    def snapshot(self) -> NativeAsyncExecutorSnapshot:
        with self._condition:
            return NativeAsyncExecutorSnapshot(
                inflight=len(self._dispatches),
                duplicate_callbacks=self._duplicate_callbacks,
                late_callbacks=self._late_callbacks,
                submit_failures=self._submit_failures,
                timeouts=self._timeouts,
            )
