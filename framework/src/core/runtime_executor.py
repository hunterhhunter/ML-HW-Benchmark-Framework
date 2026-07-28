import math
import time
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Dict, Optional


def _bounded_error_text(value, *, limit: int, fallback: str) -> str:
    try:
        text = str(value)
    except BaseException:
        text = fallback
    try:
        normalized = " ".join(text.split())
    except BaseException:
        normalized = fallback
    return normalized[:limit] or fallback


class RuntimeExecutionError(RuntimeError):
    """A runtime-returned failed execution at the synchronous e2e boundary."""

    def __init__(self, *, error_type, error_message, dispatch_token):
        self.error_type = _bounded_error_text(
            error_type,
            limit=256,
            fallback="RuntimeExecutionError",
        )
        self.error_message = _bounded_error_text(
            error_message,
            limit=512,
            fallback="runtime execution failed",
        )
        self.dispatch_token = dispatch_token
        token_text = _bounded_error_text(
            dispatch_token,
            limit=128,
            fallback="unknown",
        )
        super().__init__(
            f"{self.error_type}: {self.error_message} "
            f"(dispatch_token={token_text})"
        )


@dataclass(frozen=True)
class GenerationOutputEvent:
    observed_ns: int
    cumulative_tokens: int


@dataclass(frozen=True)
class GenerationObservation:
    backend_submitted_ns: int
    events: tuple[GenerationOutputEvent, ...]
    source: str


@dataclass(frozen=True)
class RuntimeExecution:
    outputs: Optional[Dict[str, Any]]
    timing_ms: float | Dict[str, Any] | None
    generated_tokens: int = 0
    dispatch_token: Optional[int] = None
    vendor_job_id: Any = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    generation_observation: GenerationObservation | None = None


@dataclass(frozen=True)
class NativeAsyncOutcome:
    outputs: Optional[Dict[str, Any]] = None
    timing_ms: float | Dict[str, Any] | None = None
    generated_tokens: int = 0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    generation_observation: GenerationObservation | None = None


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
    callback_started: bool = False
    physical_completion_proven: bool = False


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
                generation_observation=getattr(
                    result, "generation_observation", None
                ),
            )

        started = time.perf_counter()
        outputs = self.runtime.run(inputs)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return RuntimeExecution(outputs=outputs, timing_ms=elapsed_ms)

    def acknowledge(self, execution: RuntimeExecution) -> None:
        return None

    def shutdown(self, timeout: float) -> bool:
        return True


def _positive_integer(value, name: str) -> int:
    error_message = f"{name} must be a positive integer"
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(error_message)
    try:
        converted = int(value)
    except BaseException:
        raise ValueError(error_message) from None
    if converted <= 0:
        raise ValueError(error_message)
    return converted


def _finite_timeout(value, name: str, *, allow_zero: bool) -> float:
    qualifier = "non-negative" if allow_zero else "positive"
    error_message = f"{name} must be a finite {qualifier} real number"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(error_message)
    try:
        converted = float(value)
    except BaseException:
        raise ValueError(error_message) from None
    if (
        not math.isfinite(converted)
        or converted > threading.TIMEOUT_MAX
        or converted < 0
        or (not allow_zero and converted == 0)
    ):
        raise ValueError(error_message)
    return converted


_INVALID_PROTOCOL_VALUE = object()
_MAX_TIMING_ITEMS = 32
_MAX_TIMING_KEY_LENGTH = 128
_MAX_TIMING_TEXT_LENGTH = 512
_MAX_ERROR_TYPE_LENGTH = 256
_MAX_ERROR_MESSAGE_LENGTH = 512
_MAX_VENDOR_ID_TEXT_LENGTH = 512
_MAX_VENDOR_ID_INTEGER_BITS = 128
_MAX_GENERATION_EVENTS = 4_096
_MAX_GENERATION_SOURCE_LENGTH = 128
_UNKNOWN_DISPATCH_TOKEN_MESSAGE = "unknown native async dispatch token"


def _protocol_failure() -> NativeAsyncOutcome:
    return NativeAsyncOutcome(
        error_type="NativeAsyncProtocolError",
        error_message="native callback returned an invalid outcome",
    )


def _copy_protocol_text(value, *, max_length: int):
    if value is None:
        return None
    if type(value) is not str or len(value) > max_length:
        return _INVALID_PROTOCOL_VALUE
    return (" " + value)[1:]


def _copy_timing_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return _INVALID_PROTOCOL_VALUE
    if isinstance(value, Real):
        try:
            number = float(value)
        except (OverflowError, TypeError, ValueError):
            return _INVALID_PROTOCOL_VALUE
        if not math.isfinite(number) or number < 0:
            return _INVALID_PROTOCOL_VALUE
        return number
    if not isinstance(value, Mapping):
        return _INVALID_PROTOCOL_VALUE
    try:
        copied = {}
        for item_index, (key, item) in enumerate(value.items(), start=1):
            if item_index > _MAX_TIMING_ITEMS:
                return _INVALID_PROTOCOL_VALUE
            if (
                type(key) is not str
                or not key
                or len(key) > _MAX_TIMING_KEY_LENGTH
            ):
                return _INVALID_PROTOCOL_VALUE
            copied_key = (" " + key)[1:]
            if item is None or type(item) is bool:
                copied[copied_key] = item
            elif type(item) is str:
                if len(item) > _MAX_TIMING_TEXT_LENGTH:
                    return _INVALID_PROTOCOL_VALUE
                copied[copied_key] = (" " + item)[1:]
            elif isinstance(item, Real) and not isinstance(item, bool):
                try:
                    number = float(item)
                except (OverflowError, TypeError, ValueError):
                    return _INVALID_PROTOCOL_VALUE
                if not math.isfinite(number) or number < 0:
                    return _INVALID_PROTOCOL_VALUE
                copied[copied_key] = number
            else:
                return _INVALID_PROTOCOL_VALUE
        return copied
    except BaseException:
        return _INVALID_PROTOCOL_VALUE


def _copy_generation_observation(value):
    if value is None:
        return None
    try:
        if type(value) is not GenerationObservation:
            return _INVALID_PROTOCOL_VALUE
        backend_submitted_ns = object.__getattribute__(
            value,
            "backend_submitted_ns",
        )
        events = object.__getattribute__(value, "events")
        source = object.__getattribute__(value, "source")
        if (
            type(backend_submitted_ns) is not int
            or backend_submitted_ns < 0
            or type(events) is not tuple
            or len(events) > _MAX_GENERATION_EVENTS
            or type(source) is not str
            or not source
            or len(source) > _MAX_GENERATION_SOURCE_LENGTH
        ):
            return _INVALID_PROTOCOL_VALUE

        copied_events = []
        previous_observed_ns = backend_submitted_ns
        previous_cumulative_tokens = 0
        for event in events:
            if type(event) is not GenerationOutputEvent:
                return _INVALID_PROTOCOL_VALUE
            observed_ns = object.__getattribute__(event, "observed_ns")
            cumulative_tokens = object.__getattribute__(
                event,
                "cumulative_tokens",
            )
            if (
                type(observed_ns) is not int
                or type(cumulative_tokens) is not int
                or observed_ns < previous_observed_ns
                or cumulative_tokens < previous_cumulative_tokens
            ):
                return _INVALID_PROTOCOL_VALUE
            copied_events.append(
                GenerationOutputEvent(
                    observed_ns=observed_ns,
                    cumulative_tokens=cumulative_tokens,
                )
            )
            previous_observed_ns = observed_ns
            previous_cumulative_tokens = cumulative_tokens

        return GenerationObservation(
            backend_submitted_ns=backend_submitted_ns,
            events=tuple(copied_events),
            source=(" " + source)[1:],
        )
    except BaseException:
        return _INVALID_PROTOCOL_VALUE


def _protocol_outcome(outcome) -> NativeAsyncOutcome:
    try:
        if not isinstance(outcome, NativeAsyncOutcome):
            return _protocol_failure()
        if outcome.outputs is not None and not isinstance(outcome.outputs, dict):
            return _protocol_failure()
        if (
            isinstance(outcome.generated_tokens, bool)
            or not isinstance(outcome.generated_tokens, Integral)
            or outcome.generated_tokens < 0
        ):
            return _protocol_failure()
        timing_ms = _copy_timing_value(outcome.timing_ms)
        generation_observation = _copy_generation_observation(
            outcome.generation_observation
        )
        error_type = _copy_protocol_text(
            outcome.error_type,
            max_length=_MAX_ERROR_TYPE_LENGTH,
        )
        error_message = _copy_protocol_text(
            outcome.error_message,
            max_length=_MAX_ERROR_MESSAGE_LENGTH,
        )
        if (
            timing_ms is _INVALID_PROTOCOL_VALUE
            or generation_observation is _INVALID_PROTOCOL_VALUE
            or error_type is _INVALID_PROTOCOL_VALUE
            or error_message is _INVALID_PROTOCOL_VALUE
        ):
            return _protocol_failure()
        outputs = None if outcome.outputs is None else dict(outcome.outputs)
        return NativeAsyncOutcome(
            outputs=outputs,
            timing_ms=timing_ms,
            generated_tokens=int(outcome.generated_tokens),
            error_type=error_type,
            error_message=error_message,
            generation_observation=generation_observation,
        )
    except BaseException:
        return _protocol_failure()


def _diagnostic_vendor_job_id(value):
    try:
        value_type = type(value)
        if value is None or value_type is bool:
            return value
        if value_type is int:
            bit_count = abs(value).bit_length()
            if bit_count <= _MAX_VENDOR_ID_INTEGER_BITS:
                return value
            sign = "negative" if value < 0 else "positive"
            return f"<int sign={sign} bits={bit_count}>"
        if value_type is str:
            return str.__getitem__(
                value,
                slice(0, _MAX_VENDOR_ID_TEXT_LENGTH),
            )
        if value_type is float and math.isfinite(value):
            return value
        module = type.__getattribute__(value_type, "__module__")
        qualname = type.__getattribute__(value_type, "__qualname__")
        if type(module) is not str or type(qualname) is not str:
            return "<unavailable-vendor-job-id>"
        summary = f"<{module}.{qualname}>"
        return summary[:_MAX_VENDOR_ID_TEXT_LENGTH]
    except BaseException:
        return "<unavailable-vendor-job-id>"


def _bounded_exception_type_name(exception) -> str:
    try:
        name = type.__getattribute__(type(exception), "__name__")
        if type(name) is not str:
            return "NativeAsyncSubmitError"
        return name[:_MAX_ERROR_TYPE_LENGTH]
    except BaseException:
        return "NativeAsyncSubmitError"


class NativeAsyncRuntimeExecutor(RuntimeExecutor):
    """Bridge one callback-based SDK job to one blocking worker execution.

    ``backend.submit_async`` must publish work and return promptly; it is a
    nonblocking submission boundary. A logical timeout prevents a late result
    from becoming terminal, but this executor does not physically cancel the
    vendor job. A timed-out dispatch retires only after logical acknowledgement
    and callback completion (or future adapter-specific cancellation proof), so
    unresolved vendor work keeps shutdown and runtime unload unsafe. A submit
    exception proves that no job was accepted unless a callback was already
    delivered. Vendor-specific cancellation is an adapter follow-up.
    """

    def __init__(
        self,
        backend,
        *,
        max_inflight: int,
        completion_timeout_sec: float,
    ):
        submit_async = getattr(backend, "submit_async", None)
        if not callable(submit_async):
            raise ValueError("backend must provide callable submit_async")
        self.backend = backend
        self.max_inflight = _positive_integer(max_inflight, "max_inflight")
        self.completion_timeout_sec = _finite_timeout(
            completion_timeout_sec,
            "completion_timeout_sec",
            allow_zero=False,
        )
        self._permits = threading.BoundedSemaphore(self.max_inflight)
        self._condition = threading.Condition(threading.RLock())
        self._dispatches: dict[int, _NativeDispatch] = {}
        self._next_dispatch_token = 1
        self._duplicate_callbacks = 0
        self._late_callbacks = 0
        self._submit_failures = 0
        self._timeouts = 0
        self._closed = False

    @staticmethod
    def _failure(error_type: str, error_message: str) -> RuntimeExecution:
        return RuntimeExecution(
            outputs=None,
            timing_ms=None,
            error_type=error_type,
            error_message=error_message,
        )

    def _retire_dispatch_locked(self, dispatch: _NativeDispatch) -> bool:
        if (
            not dispatch.acknowledged
            or not dispatch.physical_completion_proven
            or self._dispatches.get(dispatch.token) is not dispatch
        ):
            return False
        del self._dispatches[dispatch.token]
        dispatch.inputs = None
        dispatch.outcome = None
        self._permits.release()
        self._condition.notify_all()
        return True

    def execute(self, inputs, timeout=None) -> RuntimeExecution:
        requested_timeout = self.completion_timeout_sec
        if timeout is not None:
            requested_timeout = min(
                requested_timeout,
                _finite_timeout(timeout, "timeout", allow_zero=True),
            )
        deadline = time.monotonic() + requested_timeout

        with self._condition:
            if self._closed:
                return self._failure(
                    "NativeAsyncShutdown",
                    "native async executor is shutting down",
                )

        permit_timeout = max(0.0, deadline - time.monotonic())
        if not self._permits.acquire(timeout=permit_timeout):
            return self._failure(
                "NativeAsyncBackpressureTimeout",
                "timed out waiting for a native async inflight slot",
            )

        with self._condition:
            if self._closed:
                self._permits.release()
                return self._failure(
                    "NativeAsyncShutdown",
                    "native async executor is shutting down",
                )
            token = self._next_dispatch_token
            self._next_dispatch_token += 1
            dispatch = _NativeDispatch(token=token, inputs=inputs)
            self._dispatches[token] = dispatch

        def commit_timeout_locked() -> bool:
            if dispatch.terminal_kind is not None:
                return False
            dispatch.outcome = NativeAsyncOutcome(
                error_type="NativeAsyncTimeout",
                error_message="native async completion timed out",
            )
            dispatch.terminal_kind = "timeout"
            self._timeouts += 1
            dispatch.event.set()
            self._condition.notify_all()
            return True

        def classify_existing_terminal_locked() -> bool:
            if dispatch.terminal_kind is None:
                return False
            if dispatch.terminal_kind == "timeout":
                self._late_callbacks += 1
            else:
                self._duplicate_callbacks += 1
            return True

        def callback(outcome) -> None:
            with self._condition:
                dispatch.callback_started = True
                if (
                    dispatch.physical_completion_proven
                    and classify_existing_terminal_locked()
                ):
                    return
            normalized = _protocol_outcome(outcome)
            with self._condition:
                if dispatch.physical_completion_proven:
                    classify_existing_terminal_locked()
                    return
                dispatch.physical_completion_proven = True
                if classify_existing_terminal_locked():
                    self._retire_dispatch_locked(dispatch)
                    return
                if time.monotonic() >= deadline:
                    commit_timeout_locked()
                    self._late_callbacks += 1
                    self._retire_dispatch_locked(dispatch)
                    return
                dispatch.outcome = normalized
                dispatch.terminal_kind = "callback"
                dispatch.event.set()
                self._retire_dispatch_locked(dispatch)
                self._condition.notify_all()

        with self._condition:
            submit_allowed = time.monotonic() < deadline
            if not submit_allowed:
                commit_timeout_locked()
                dispatch.physical_completion_proven = True

        if submit_allowed:
            try:
                vendor_job_id = self.backend.submit_async(inputs, callback)
            except BaseException as exc:
                observed_at = time.monotonic()
                submit_error_type = _bounded_exception_type_name(exc)
                with self._condition:
                    if not dispatch.callback_started:
                        dispatch.physical_completion_proven = True
                    if (
                        dispatch.terminal_kind is None
                        and not dispatch.callback_started
                    ):
                        if observed_at >= deadline:
                            commit_timeout_locked()
                        else:
                            dispatch.outcome = NativeAsyncOutcome(
                                error_type=submit_error_type,
                                error_message="native async submission failed",
                            )
                            dispatch.terminal_kind = "submit_failure"
                            self._submit_failures += 1
                            dispatch.event.set()
                            self._condition.notify_all()
            else:
                observed_at = time.monotonic()
                diagnostic_vendor_job_id = _diagnostic_vendor_job_id(
                    vendor_job_id
                )
                with self._condition:
                    dispatch.vendor_job_id = diagnostic_vendor_job_id
                    if (
                        dispatch.terminal_kind is None
                        and observed_at >= deadline
                    ):
                        commit_timeout_locked()

        remaining = max(0.0, deadline - time.monotonic())
        if not dispatch.event.wait(timeout=remaining):
            with self._condition:
                commit_timeout_locked()

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
            generated_tokens=int(outcome.generated_tokens),
            dispatch_token=token,
            vendor_job_id=vendor_job_id,
            error_type=outcome.error_type,
            error_message=outcome.error_message,
            generation_observation=outcome.generation_observation,
        )

    def acknowledge(self, execution: RuntimeExecution) -> None:
        token = execution.dispatch_token
        if token is None:
            return
        if type(token) is not int:
            raise RuntimeError(_UNKNOWN_DISPATCH_TOKEN_MESSAGE)
        with self._condition:
            dispatch = self._dispatches.get(token)
            if dispatch is None:
                if 1 <= token < self._next_dispatch_token:
                    return
                raise RuntimeError(_UNKNOWN_DISPATCH_TOKEN_MESSAGE)
            dispatch.acknowledged = True
            self._retire_dispatch_locked(dispatch)

    def shutdown(self, timeout: float) -> bool:
        wait_timeout = _finite_timeout(
            timeout,
            "timeout",
            allow_zero=True,
        )
        deadline = time.monotonic() + wait_timeout
        with self._condition:
            self._closed = True
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
