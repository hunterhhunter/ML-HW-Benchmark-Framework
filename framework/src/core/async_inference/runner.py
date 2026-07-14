import math
import queue
import time
from numbers import Integral
from threading import Event, Lock, Thread

import numpy as np

from core.inference_pipeline import InferencePipeline

from .completion import CompletionCoordinator
from .engine import AsyncInferenceEngine
from .metrics import AsyncMetricsCollector
from .producers import OfflineProducer, ServerLikeProducer
from .types import AsyncBenchmarkResult, AsyncScenario, RunStatus


def _safe_type_name(value):
    value_type = type(value)
    try:
        name = type.__getattribute__(value_type, "__name__")
    except BaseException:
        return "<unknown>"
    return name if type(name) is str else "<unknown>"


_SAFE_EXCEPTION_TYPES = (
    BaseException,
    Exception,
    ArithmeticError,
    AssertionError,
    AttributeError,
    EOFError,
    ImportError,
    LookupError,
    MemoryError,
    NameError,
    OSError,
    ReferenceError,
    RuntimeError,
    StopAsyncIteration,
    StopIteration,
    SyntaxError,
    SystemError,
    TypeError,
    ValueError,
    Warning,
    GeneratorExit,
    KeyboardInterrupt,
    SystemExit,
)


def _safe_message_argument(value):
    value_type = type(value)
    if value_type is str:
        return value
    if value is None:
        return "None"
    if value_type in (bool, int, float):
        try:
            return str(value)
        except BaseException:
            return None
    return None


def _safe_error_details(exc):
    error_type = _safe_type_name(exc)
    message = f"<unprintable {error_type}>"
    exc_type = type(exc)
    if any(exc_type is safe_type for safe_type in _SAFE_EXCEPTION_TYPES):
        try:
            args = BaseException.args.__get__(exc, exc_type)
        except BaseException:
            args = None
        if type(args) is tuple and tuple.__len__(args) <= 8:
            normalized = [_safe_message_argument(item) for item in args]
            if all(item is not None for item in normalized):
                if not normalized:
                    message = ""
                elif len(normalized) == 1:
                    message = normalized[0]
                else:
                    message = ", ".join(normalized)
    return {
        "error_type": error_type,
        "error_message": message,
    }


class _CallbackOutcome:
    def __init__(
        self,
        *,
        value=None,
        exception=None,
        diagnostic=None,
        timed_out=False,
    ):
        self.value = value
        self.exception = exception
        self.diagnostic = diagnostic
        self.timed_out = timed_out


class _TotalSerializer:
    PLACEHOLDER = "<serialization_error>"
    MAX_DEPTH = 32
    MAX_ITEMS = 10_000
    MAX_ARRAY_ITEMS = 4_096
    _NUMPY_INTEGER_TYPES = (
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    )
    _NUMPY_FLOAT_TYPES = (
        np.float16,
        np.float32,
        np.float64,
        np.longdouble,
    )

    def __init__(self):
        self.diagnostics = []
        self._active = set()
        self._item_count = 0

    def serialize(self, value, phase, path="$"):
        self._active.clear()
        self._item_count = 0
        try:
            return self._serialize(value, phase, path, depth=0)
        except BaseException as exc:
            return self._exception_failure(
                phase,
                path,
                "trusted_serialization",
                exc,
            )

    def _serialize(self, value, phase, path, depth):
        if depth > self.MAX_DEPTH:
            return self._budget_failure(phase, path, "depth_budget")
        if self._item_count >= self.MAX_ITEMS:
            return self._budget_failure(phase, path, "item_budget")
        self._item_count += 1

        value_type = type(value)
        if value is None or value_type in (bool, int, str):
            return value
        if value_type is float:
            return value if math.isfinite(value) else None
        if value_type is dict:
            return self._mapping(value, phase, path, depth)
        if value_type is list:
            return self._sequence(value, phase, path, depth, list)
        if value_type is tuple:
            return self._sequence(value, phase, path, depth, tuple)
        if value_type is np.ndarray:
            return self._numpy_array(value, phase, path, depth)
        if value_type is np.bool_:
            return bool(value)
        if any(
            value_type is numpy_type
            for numpy_type in self._NUMPY_INTEGER_TYPES
        ):
            return int(value)
        if any(
            value_type is numpy_type
            for numpy_type in self._NUMPY_FLOAT_TYPES
        ):
            converted = float(value)
            return converted if math.isfinite(converted) else None
        if value_type is np.str_:
            return str(value)
        return self._unsupported_failure(phase, path, value)

    def _mapping(self, value, phase, path, depth):
        size = dict.__len__(value)
        if size > self.MAX_ITEMS - self._item_count:
            return self._budget_failure(phase, path, "item_budget")
        if not self._enter(value, phase, path):
            return self.PLACEHOLDER
        result = {}
        try:
            for index, (key, item) in enumerate(dict.items(value)):
                key_type = type(key)
                if key_type is str:
                    normalized_key = key
                elif key is None:
                    normalized_key = "None"
                elif key_type in (bool, int):
                    normalized_key = str(key)
                elif key_type is float and math.isfinite(key):
                    normalized_key = str(key)
                else:
                    normalized_key = f"<serialization_error_key_{index}>"
                    self._unsupported_failure(
                        phase,
                        f"{path}[{index}].key",
                        key,
                        operation="mapping_key_type",
                    )
                result[normalized_key] = self._serialize(
                    item,
                    phase,
                    f"{path}[{index}]",
                    depth + 1,
                )
            return result
        finally:
            self._active.discard(id(value))

    def _sequence(self, value, phase, path, depth, container_type):
        size = container_type.__len__(value)
        if size > self.MAX_ITEMS - self._item_count:
            return self._budget_failure(phase, path, "item_budget")
        if not self._enter(value, phase, path):
            return [self.PLACEHOLDER]
        result = []
        try:
            for index in range(size):
                result.append(
                    self._serialize(
                        container_type.__getitem__(value, index),
                        phase,
                        f"{path}[{index}]",
                        depth + 1,
                    )
                )
            return result
        finally:
            self._active.discard(id(value))

    def _numpy_array(self, value, phase, path, depth):
        size = value.size
        if size > self.MAX_ARRAY_ITEMS:
            return self._budget_failure(
                phase,
                path,
                "array_size_budget",
            )
        if value.ndim + depth > self.MAX_DEPTH:
            return self._budget_failure(phase, path, "depth_budget")
        try:
            converted = np.ndarray.tolist(value)
        except BaseException as exc:
            return self._exception_failure(
                phase,
                path,
                "numpy_array_conversion",
                exc,
            )
        return self._serialize(converted, phase, path, depth)

    def _enter(self, value, phase, path):
        identity = id(value)
        if identity in self._active:
            self._diagnostic(
                phase,
                path,
                "cycle",
                "SerializationCycle",
                "cyclic result value",
            )
            return False
        self._active.add(identity)
        return True

    def _unsupported_failure(
        self,
        phase,
        path,
        value,
        *,
        operation="unsupported_type",
    ):
        value_type = _safe_type_name(value)
        return self._diagnostic(
            phase,
            path,
            operation,
            "SerializationUnsupportedType",
            f"unsupported result type: {value_type}",
            value_type=value_type,
        )

    def _budget_failure(self, phase, path, operation):
        return self._diagnostic(
            phase,
            path,
            operation,
            "SerializationBudgetExceeded",
            f"serialization {operation} exceeded",
        )

    def _exception_failure(self, phase, path, operation, exc):
        return self._diagnostic(
            phase,
            path,
            operation,
            **_safe_error_details(exc),
        )

    def _diagnostic(
        self,
        phase,
        path,
        operation,
        error_type,
        error_message,
        **extra,
    ):
        self.diagnostics.append(
            {
                "phase": phase,
                "path": path,
                "operation": operation,
                "error_type": error_type,
                "error_message": error_message,
                **extra,
            }
        )
        return self.PLACEHOLDER


class _BoundedCallbacks:
    LIMITATION = (
        "Python callback threads cannot be forcibly terminated; timed-out "
        "daemon callbacks may continue after the benchmark result is returned."
    )

    def __init__(self):
        self._next_id = 1
        self._invocations = []

    def invoke(self, phase, callback, deadline):
        callback_id = f"{phase}:{self._next_id}"
        self._next_id += 1
        done = Event()
        outcome = {}
        thread_name = f"async-callback-{phase}-{callback_id.rsplit(':', 1)[1]}"

        def call():
            try:
                outcome["value"] = callback()
            except BaseException as exc:
                outcome["exception"] = exc
            finally:
                done.set()

        thread = Thread(target=call, name=thread_name, daemon=True)
        invocation = {
            "callback_id": callback_id,
            "phase": phase,
            "thread": thread,
            "done": done,
            "timed_out": False,
        }
        self._invocations.append(invocation)
        try:
            thread.start()
        except BaseException as exc:
            diagnostic = {"phase": phase, **_safe_error_details(exc)}
            return _CallbackOutcome(
                exception=exc,
                diagnostic=diagnostic,
            )

        if not done.wait(timeout=max(0.0, deadline - time.monotonic())):
            invocation["timed_out"] = True
            diagnostic = {
                "phase": phase,
                "error_type": "TimeoutError",
                "error_message": (
                    f"{phase} callback exceeded configured deadline"
                ),
                "callback_id": callback_id,
                "callback_thread": thread.name,
                "callback_alive": thread.is_alive(),
            }
            return _CallbackOutcome(
                diagnostic=diagnostic,
                timed_out=True,
            )

        exception = outcome.get("exception")
        if exception is not None:
            diagnostic = {
                "phase": phase,
                **_safe_error_details(exception),
            }
            return _CallbackOutcome(
                exception=exception,
                diagnostic=diagnostic,
            )
        return _CallbackOutcome(value=outcome.get("value"))

    def outstanding(self):
        return [
            {
                "callback_id": invocation["callback_id"],
                "phase": invocation["phase"],
                "thread_name": invocation["thread"].name,
                "alive": invocation["thread"].is_alive(),
            }
            for invocation in self._invocations
            if invocation["timed_out"]
            and not invocation["done"].is_set()
        ]


class _CallbackJob:
    def __init__(self, callback_id, phase, callback):
        self.callback_id = callback_id
        self.phase = phase
        self.callback = callback
        self.done = Event()
        self.value = None
        self.exception = None
        self.state = "queued"
        self.timed_out = False


class _SerializedCallbackLane:
    LIMITATION = _BoundedCallbacks.LIMITATION
    _CLOSE = object()

    def __init__(self):
        self._next_id = 1
        self._jobs = []
        self._queue = queue.Queue()
        self._closed = False
        self._exited = Event()
        self._thread = Thread(
            target=self._run,
            name="async-callback-monitor-lane",
            daemon=True,
        )
        self._start_error = None
        try:
            self._thread.start()
        except BaseException as exc:
            self._start_error = exc
            self._exited.set()

    def submit(self, phase, callback):
        callback_id = f"{phase}:{self._next_id}"
        self._next_id += 1
        job = _CallbackJob(callback_id, phase, callback)
        self._jobs.append(job)
        if self._start_error is not None:
            job.state = "done"
            job.exception = self._start_error
            job.done.set()
        elif self._closed:
            job.state = "done"
            job.exception = RuntimeError("callback lane is closed")
            job.done.set()
        else:
            self._queue.put(job)
        return job

    def wait(self, job, deadline):
        if not job.done.wait(
            timeout=max(0.0, deadline - time.monotonic())
        ):
            job.timed_out = True
            diagnostic = {
                "phase": job.phase,
                "error_type": "TimeoutError",
                "error_message": (
                    f"{job.phase} callback exceeded configured deadline"
                ),
                "callback_id": job.callback_id,
                "callback_thread": self._thread.name,
                "callback_alive": self._thread.is_alive(),
                "callback_state": job.state,
            }
            return _CallbackOutcome(
                diagnostic=diagnostic,
                timed_out=True,
            )
        if job.exception is not None:
            return _CallbackOutcome(
                exception=job.exception,
                diagnostic={
                    "phase": job.phase,
                    **_safe_error_details(job.exception),
                },
            )
        return _CallbackOutcome(value=job.value)

    def invoke(self, phase, callback, deadline):
        return self.wait(self.submit(phase, callback), deadline)

    def close(self, deadline):
        if not self._closed:
            self._closed = True
            if self._start_error is None:
                self._queue.put(self._CLOSE)
        self._exited.wait(timeout=max(0.0, deadline - time.monotonic()))
        return self._exited.is_set()

    def outstanding(self):
        return [
            {
                "callback_id": job.callback_id,
                "phase": job.phase,
                "thread_name": self._thread.name,
                "alive": self._thread.is_alive(),
                "state": job.state,
            }
            for job in self._jobs
            if job.timed_out and not job.done.is_set()
        ]

    def _run(self):
        try:
            while True:
                job = self._queue.get()
                if job is self._CLOSE:
                    return
                job.state = "running"
                try:
                    job.value = job.callback()
                except BaseException as exc:
                    job.exception = exc
                finally:
                    job.callback = None
                    job.state = "done"
                    job.done.set()
        finally:
            self._exited.set()


class _MeasuredSubmitter:
    def __init__(
        self,
        engine,
        metrics,
        monitor,
        clock_ns,
        callbacks,
        callback_errors,
        callback_timeout_sec,
    ):
        self.engine = engine
        self.metrics = metrics
        self.monitor = monitor
        self.clock_ns = clock_ns
        self.callbacks = callbacks
        self.callback_errors = callback_errors
        self.callback_timeout_sec = callback_timeout_sec
        self.started = False
        self.monitor_start_attempted = False
        self.monitor_stop_job = None
        self.attempted = 0
        self.accepted = 0
        self.rejected = 0

    def ensure_measurement(self, *, start_monitor, started_ns=None):
        if self.started:
            return
        self.metrics.try_begin_measurement(
            self.clock_ns() if started_ns is None else started_ns
        )
        self.started = True
        if start_monitor and self.monitor is not None:
            self.monitor_start_attempted = True
            result = self.callbacks.invoke(
                "monitor_start",
                self.monitor.start,
                time.monotonic() + self.callback_timeout_sec,
            )
            if result.diagnostic is not None:
                self.callback_errors.append(result.diagnostic)
                self.metrics.add_warning("hardware_monitor_start_failed")
            if result.timed_out:
                self.metrics.add_invalid_reason("callback_timeout")
                self.monitor_stop_job = self.callbacks.submit(
                    "monitor_stop",
                    self.monitor.stop,
                )
            if result.exception is not None and not isinstance(
                result.exception,
                Exception,
            ):
                raise result.exception

    def submit(self, request, block):
        self.ensure_measurement(
            start_monitor=True,
            started_ns=request.issued_ns,
        )
        self.attempted += 1
        accepted = self.engine.submit(request, block=block)
        if accepted:
            self.accepted += 1
        else:
            self.rejected += 1
        return accepted


class AsyncBenchmarkRunner:
    def __init__(
        self,
        dataloader,
        runtime,
        evaluator,
        max_new_tokens=256,
        monitor=None,
        decoder=None,
        trace_callback=None,
    ):
        self.dataloader = dataloader
        self.runtime = runtime
        self.evaluator = evaluator
        self.max_new_tokens = max_new_tokens
        self.monitor = monitor
        self.decoder = decoder
        self.trace_callback = trace_callback
        self._run_claim_lock = Lock()
        self._run_claimed = False

    def run(self, config, warmup_runs=1):
        monitor_callback_owner = []
        try:
            return self._run(config, warmup_runs, monitor_callback_owner)
        finally:
            if monitor_callback_owner:
                monitor_callback_owner[0].close(
                    time.monotonic() + config.flush_timeout_sec
                )

    def _run(self, config, warmup_runs, monitor_callback_owner):
        config.validate()
        if (
            isinstance(warmup_runs, bool)
            or not isinstance(warmup_runs, Integral)
            or warmup_runs < 0
        ):
            raise ValueError("warmup_runs must be a non-negative integer")
        with self._run_claim_lock:
            if self._run_claimed:
                raise RuntimeError("AsyncBenchmarkRunner can only be run once")
            self._run_claimed = True
        pipeline = InferencePipeline(
            self.dataloader,
            self.runtime,
            max_new_tokens=self.max_new_tokens,
        )
        metrics = AsyncMetricsCollector(
            time.monotonic_ns(),
            config.worker_count,
            latency_slo_ms=config.latency_slo_ms,
        )
        coordinator = CompletionCoordinator(
            pipeline=pipeline,
            evaluator=self.evaluator,
            decoder=self.decoder,
            metrics=metrics,
            queue_capacity=config.worker_count,
            request_timeout_ms=config.request_timeout_ms,
            trace_callback=self.trace_callback,
        )
        engine = AsyncInferenceEngine(
            runtime=self.runtime,
            pipeline=pipeline,
            config=config,
            coordinator=coordinator,
            metrics=metrics,
        )

        if warmup_runs > 0:
            try:
                warmup_batch = self.dataloader.load_batch(
                    config.max_batch_size
                )
                if warmup_batch:
                    collated = pipeline.collate_batch(warmup_batch)
                    runtime_input = pipeline.prepare_runtime_input(
                        collated["input"]
                    )
                    self.runtime.warmup(
                        runtime_input,
                        num_runs=warmup_runs,
                    )
            finally:
                self._reset_dataloader_cursor()

        callbacks = _BoundedCallbacks()
        monitor_callbacks = (
            _SerializedCallbackLane() if self.monitor is not None else None
        )
        if monitor_callbacks is not None:
            monitor_callback_owner.append(monitor_callbacks)
        serializer = _TotalSerializer()
        callback_errors = []
        submitter = _MeasuredSubmitter(
            engine,
            metrics,
            self.monitor,
            time.monotonic_ns,
            monitor_callbacks,
            callback_errors,
            config.flush_timeout_sec,
        )
        producer_class = (
            OfflineProducer
            if config.scenario is AsyncScenario.OFFLINE
            else ServerLikeProducer
        )
        producer = producer_class(self.dataloader, submitter, config)

        lifecycle_errors = []
        producer_error = None
        producer_result = None
        fatal_error = None
        start_succeeded = False
        flushed = False
        shutdown = False
        flush_started_ns = time.monotonic_ns()
        flush_finished_ns = flush_started_ns
        try:
            try:
                engine.start()
                start_succeeded = True
            except KeyboardInterrupt as exc:
                producer_error = self._error_details(exc)
                metrics.add_invalid_reason("producer_error")
                lifecycle_errors.append(
                    {"phase": "start", **self._error_details(exc)}
                )
            except Exception as exc:
                metrics.add_invalid_reason("worker_shutdown_failed")
                lifecycle_errors.append(
                    {"phase": "start", **self._error_details(exc)}
                )
            except BaseException as exc:
                metrics.add_invalid_reason("worker_shutdown_failed")
                lifecycle_errors.append(
                    {"phase": "start", **self._error_details(exc)}
                )
                fatal_error = exc

            if start_succeeded:
                try:
                    producer_result = producer.run()
                except KeyboardInterrupt as exc:
                    producer_error = self._error_details(exc)
                    metrics.add_invalid_reason("producer_error")
                    try:
                        engine.cancel_queued("KeyboardInterrupt")
                    except Exception as cancel_exc:
                        lifecycle_errors.append(
                            {
                                "phase": "cancel_queued",
                                **self._error_details(cancel_exc),
                            }
                        )
                        metrics.add_invalid_reason("request_failed")
                    except BaseException as cancel_exc:
                        lifecycle_errors.append(
                            {
                                "phase": "cancel_queued",
                                **self._error_details(cancel_exc),
                            }
                        )
                        metrics.add_invalid_reason("request_failed")
                        if fatal_error is None:
                            fatal_error = cancel_exc
                except Exception as exc:
                    producer_error = self._error_details(exc)
                    metrics.add_invalid_reason("producer_error")
                except BaseException as exc:
                    fatal_error = exc
        finally:
            try:
                submitter.ensure_measurement(start_monitor=False)
            except BaseException as exc:
                lifecycle_errors.append(
                    {"phase": "measurement_start", **self._error_details(exc)}
                )
                metrics.add_invalid_reason("metrics_unavailable")
                if fatal_error is None and not isinstance(exc, Exception):
                    fatal_error = exc

            try:
                engine.close_submission()
            except Exception as exc:
                lifecycle_errors.append(
                    {"phase": "close_submission", **self._error_details(exc)}
                )
                metrics.add_invalid_reason("worker_shutdown_failed")
            except BaseException as exc:
                lifecycle_errors.append(
                    {"phase": "close_submission", **self._error_details(exc)}
                )
                metrics.add_invalid_reason("worker_shutdown_failed")
                if fatal_error is None:
                    fatal_error = exc
            finally:
                close_submission_internal = getattr(
                    engine,
                    "_close_submission_internal",
                    None,
                )
                if close_submission_internal is not None:
                    try:
                        close_submission_internal()
                    except Exception as exc:
                        lifecycle_errors.append(
                            {
                                "phase": "close_submission_internal",
                                **self._error_details(exc),
                            }
                        )
                        metrics.add_invalid_reason("worker_shutdown_failed")
                    except BaseException as exc:
                        lifecycle_errors.append(
                            {
                                "phase": "close_submission_internal",
                                **self._error_details(exc),
                            }
                        )
                        metrics.add_invalid_reason("worker_shutdown_failed")
                        if fatal_error is None:
                            fatal_error = exc

            flush_started_ns = time.monotonic_ns()
            try:
                flushed = bool(engine.flush())
            except Exception as exc:
                lifecycle_errors.append(
                    {"phase": "flush", **self._error_details(exc)}
                )
                metrics.add_invalid_reason("flush_timeout")
            except BaseException as exc:
                lifecycle_errors.append(
                    {"phase": "flush", **self._error_details(exc)}
                )
                metrics.add_invalid_reason("flush_timeout")
                if fatal_error is None:
                    fatal_error = exc
            finally:
                flush_finished_ns = time.monotonic_ns()

            if submitter.monitor_start_attempted:
                stop_deadline = time.monotonic() + config.flush_timeout_sec
                if submitter.monitor_stop_job is None:
                    result = monitor_callbacks.invoke(
                        "monitor_stop",
                        self.monitor.stop,
                        stop_deadline,
                    )
                else:
                    result = monitor_callbacks.wait(
                        submitter.monitor_stop_job,
                        stop_deadline,
                    )
                if result.diagnostic is not None:
                    lifecycle_errors.append(result.diagnostic)
                    callback_errors.append(result.diagnostic)
                    metrics.add_warning("hardware_monitor_stop_failed")
                if result.timed_out:
                    metrics.add_invalid_reason("callback_timeout")
                if result.exception is not None and not isinstance(
                    result.exception,
                    Exception,
                ):
                    if fatal_error is None:
                        fatal_error = result.exception

            try:
                shutdown = bool(engine.shutdown())
            except Exception as exc:
                lifecycle_errors.append(
                    {"phase": "shutdown", **self._error_details(exc)}
                )
                metrics.add_invalid_reason("worker_shutdown_failed")
            except BaseException as exc:
                lifecycle_errors.append(
                    {"phase": "shutdown", **self._error_details(exc)}
                )
                metrics.add_invalid_reason("worker_shutdown_failed")
                if fatal_error is None:
                    fatal_error = exc

        if fatal_error is not None:
            raise fatal_error

        collected = metrics.finalize(flush_finished_ns)
        details = collected["details"]
        details["config"] = self._config_details(config)
        producer_details = {
            "attempted": (
                submitter.attempted
                if producer_result is None
                else producer_result.attempted
            ),
            "accepted": (
                submitter.accepted
                if producer_result is None
                else producer_result.accepted
            ),
            "rejected": (
                submitter.rejected
                if producer_result is None
                else producer_result.rejected
            ),
            "producer_load_ms": (
                None
                if producer_result is None
                else producer_result.producer_load_ms
            ),
        }
        if producer_error is not None:
            producer_details["error"] = producer_error
        details["producer"] = producer_details
        details["lifecycle_errors"] = lifecycle_errors
        try:
            outstanding_request_ids = list(engine.outstanding_request_ids())
        except Exception as exc:
            outstanding_request_ids = []
            lifecycle_errors.append(
                {
                    "phase": "outstanding_snapshot",
                    **self._error_details(exc),
                }
            )
        details["outstanding_request_ids"] = outstanding_request_ids
        details["flush_duration_ms"] = (
            flush_finished_ns - flush_started_ns
        ) / 1_000_000.0

        completed_samples = collected["summary"]["async_completed_samples"]
        invalid_reasons = set(details["invalid_reasons"])
        if completed_samples == 0:
            invalid_reasons.add("no_samples")
        if completed_samples < config.min_samples:
            invalid_reasons.add("min_samples_not_met")
        if details["measurement_duration_sec"] < config.min_duration_sec:
            invalid_reasons.add("min_duration_not_met")
        if not flushed:
            invalid_reasons.add("flush_timeout")
        if not shutdown:
            invalid_reasons.add("worker_shutdown_failed")
        p99 = collected["summary"]["async_e2e_latency_p99_ms"]
        if (
            config.latency_slo_ms is not None
            and p99 is not None
            and p99 > config.latency_slo_ms
        ):
            invalid_reasons.add("latency_slo_not_met")

        warnings = set(details["warnings"])
        if completed_samples < 1000:
            warnings.add("tail_percentile_low_sample_count")

        quality_evaluation_skipped = None
        if shutdown:
            result = callbacks.invoke(
                "evaluator_compute",
                self.evaluator.compute,
                time.monotonic() + config.flush_timeout_sec,
            )
            if result.diagnostic is None:
                serialized_quality = serializer.serialize(
                    result.value,
                    "evaluator_compute_result",
                )
                if type(result.value) is not dict:
                    quality_metrics = {}
                    callback_errors.append(
                        self._result_shape_diagnostic(
                            "evaluator_compute_result",
                            "evaluator_compute",
                            result.value,
                        )
                    )
                    invalid_reasons.add("quality_result_invalid")
                else:
                    quality_metrics = (
                        serialized_quality
                        if type(serialized_quality) is dict
                        else {}
                    )
            else:
                quality_metrics = {}
                callback_errors.append(result.diagnostic)
                invalid_reasons.add("request_failed")
                if result.timed_out:
                    invalid_reasons.add("callback_timeout")
                if result.exception is not None and not isinstance(
                    result.exception,
                    Exception,
                ):
                    raise result.exception
        else:
            quality_metrics = {}
            quality_evaluation_skipped = "engine_shutdown_failed"

        final_metrics = {}
        for key, value in quality_metrics.items():
            if key.startswith("async_"):
                warnings.add("quality_metric_namespace_collision")
                continue
            final_metrics[key] = value
        final_metrics.update(collected["summary"])
        final_metrics["async_achieved_qps"] = final_metrics[
            "async_completed_samples_per_sec"
        ]
        if config.target_qps is not None:
            final_metrics["async_target_qps"] = config.target_qps
            final_metrics["async_target_qps_gap"] = (
                final_metrics["async_achieved_qps"] - config.target_qps
            )
        evaluator_samples = self._evaluator_sample_count(quality_metrics)
        if evaluator_samples is not None:
            final_metrics["async_evaluator_samples"] = evaluator_samples
            if evaluator_samples != completed_samples:
                invalid_reasons.add("counter_invariant_failed")

        hardware_metrics = {}
        if submitter.monitor_start_attempted:
            result = monitor_callbacks.invoke(
                "monitor_summary",
                self.monitor.summary,
                time.monotonic() + config.flush_timeout_sec,
            )
            if result.diagnostic is None:
                serialized_hardware = serializer.serialize(
                    result.value,
                    "monitor_summary_result",
                )
                if type(result.value) is not dict:
                    hardware_metrics = {}
                    callback_errors.append(
                        self._result_shape_diagnostic(
                            "monitor_summary_result",
                            "monitor_summary",
                            result.value,
                        )
                    )
                    warnings.add("hardware_monitor_summary_failed")
                    invalid_reasons.add("hardware_result_invalid")
                else:
                    hardware_metrics = (
                        serialized_hardware
                        if type(serialized_hardware) is dict
                        else {}
                    )
            else:
                callback_errors.append(result.diagnostic)
                warnings.add("hardware_monitor_summary_failed")
                if result.timed_out:
                    invalid_reasons.add("callback_timeout")
                if result.exception is not None and not isinstance(
                    result.exception,
                    Exception,
                ):
                    raise result.exception
        for key, value in hardware_metrics.items():
            if not key.startswith("hw_"):
                warnings.add("hardware_metric_namespace_violation")
                continue
            final_metrics[key] = value

        if serializer.diagnostics:
            invalid_reasons.add("result_serialization_failed")

        reasons = tuple(sorted(invalid_reasons))
        warning_values = tuple(sorted(warnings))
        status = RunStatus.INVALID if reasons else RunStatus.VALID
        final_metrics["async_run_status"] = status.value
        final_metrics["async_invalid_reasons"] = ",".join(reasons)
        outstanding_callbacks = callbacks.outstanding()
        if monitor_callbacks is not None:
            outstanding_callbacks.extend(monitor_callbacks.outstanding())
        details.update(
            {
                "invalid_reasons": list(reasons),
                "warnings": list(warning_values),
                "quality_metrics": quality_metrics,
                "hardware_metrics": hardware_metrics,
                "evaluator_samples": evaluator_samples,
                "callback_errors": callback_errors,
                "outstanding_callbacks": outstanding_callbacks,
                "callback_timeout_limitation": (
                    callbacks.LIMITATION
                    if outstanding_callbacks
                    else None
                ),
                "serialization_errors": serializer.diagnostics,
                "quality_evaluation_skipped": quality_evaluation_skipped,
                "status": status.value,
            }
        )
        serialized_metrics = serializer.serialize(
            final_metrics,
            "final_metrics",
        )
        serialized_details = serializer.serialize(
            details,
            "result_details",
        )
        if serializer.diagnostics:
            invalid_reasons.add("result_serialization_failed")
            reasons = tuple(sorted(invalid_reasons))
            status = RunStatus.INVALID
            serialized_metrics["async_run_status"] = status.value
            serialized_metrics["async_invalid_reasons"] = ",".join(reasons)
            serialized_details["invalid_reasons"] = list(reasons)
            serialized_details["serialization_errors"] = list(
                serializer.diagnostics
            )
            serialized_details["status"] = status.value
        return AsyncBenchmarkResult(
            metrics=serialized_metrics,
            details=serialized_details,
            status=status,
            invalid_reasons=reasons,
            warnings=warning_values,
        )

    def _reset_dataloader_cursor(self):
        if hasattr(self.dataloader, "current_idx"):
            self.dataloader.current_idx = 0
        elif hasattr(self.dataloader, "_current_idx"):
            self.dataloader._current_idx = 0

    @staticmethod
    def _config_details(config):
        return {
            "scenario": config.scenario.value,
            "queue_capacity": config.queue_capacity,
            "worker_count": config.worker_count,
            "max_batch_size": config.max_batch_size,
            "batch_timeout_ms": config.batch_timeout_ms,
            "submit_timeout_sec": config.submit_timeout_sec,
            "flush_timeout_sec": config.flush_timeout_sec,
            "request_timeout_ms": config.request_timeout_ms,
            "min_samples": config.min_samples,
            "min_duration_sec": config.min_duration_sec,
            "max_samples": config.max_samples,
            "target_qps": config.target_qps,
            "schedule_seed": config.schedule_seed,
            "latency_slo_ms": config.latency_slo_ms,
        }

    @staticmethod
    def _error_details(exc):
        return _safe_error_details(exc)

    @staticmethod
    def _result_shape_diagnostic(phase, callback_name, value):
        return {
            "phase": phase,
            "operation": "result_shape",
            "error_type": "ResultShapeError",
            "error_message": (
                f"{callback_name} result must be an exact dict"
            ),
            "expected_type": "dict",
            "actual_type": _safe_type_name(value),
        }

    @staticmethod
    def _evaluator_sample_count(quality_metrics):
        for key in ("Total Samples", "total_samples", "num_samples"):
            value = quality_metrics.get(key)
            if type(value) is int:
                return value
            if type(value) is float and math.isfinite(value):
                return value
        return None

    @classmethod
    def _serializable(cls, value):
        return _TotalSerializer().serialize(value, "serialization")
