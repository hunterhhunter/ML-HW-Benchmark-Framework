import math
import queue
import time
from collections.abc import Iterable, Mapping
from enum import Enum
from numbers import Integral, Number
from threading import Event, Lock, Thread

from core.inference_pipeline import InferencePipeline

from .completion import CompletionCoordinator
from .engine import AsyncInferenceEngine
from .metrics import AsyncMetricsCollector
from .producers import OfflineProducer, ServerLikeProducer
from .types import AsyncBenchmarkResult, AsyncScenario, RunStatus


def _safe_error_details(exc):
    error_type = type(exc).__name__
    try:
        message = str(exc)
    except BaseException:
        message = f"<unprintable {error_type}>"
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

    def __init__(self):
        self.diagnostics = []
        self._active = set()

    def serialize(self, value, phase, path="$"):
        try:
            return self._serialize(value, phase, path)
        except BaseException as exc:
            return self._failure(phase, path, "serialize", exc)

    def _serialize(self, value, phase, path):
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, Enum):
            try:
                enum_value = value.value
            except BaseException as exc:
                return self._failure(phase, path, "enum_value", exc)
            return self.serialize(enum_value, phase, f"{path}.value")
        if isinstance(value, str):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, Mapping):
            return self._mapping(value, phase, path)
        if isinstance(value, (list, tuple, set, frozenset)):
            return self._iterable(value, phase, path)
        if isinstance(value, Number):
            try:
                item_method = getattr(value, "item")
            except AttributeError:
                item_method = None
            except BaseException as exc:
                return self._failure(phase, path, "item_access", exc)
            if callable(item_method):
                try:
                    item = item_method()
                except BaseException as exc:
                    return self._failure(phase, path, "item", exc)
                if item is not value:
                    return self.serialize(item, phase, f"{path}.item")
        try:
            tolist = getattr(value, "tolist")
        except AttributeError:
            tolist = None
        except BaseException as exc:
            return self._failure(phase, path, "tolist_access", exc)
        if callable(tolist):
            try:
                converted = tolist()
            except BaseException as exc:
                return self._failure(phase, path, "tolist", exc)
            return self.serialize(converted, phase, f"{path}.tolist")
        if isinstance(value, Iterable):
            return self._iterable(value, phase, path)
        try:
            return str(value)
        except BaseException as str_exc:
            try:
                return repr(value)
            except BaseException:
                return self._failure(
                    phase,
                    path,
                    "fallback_string",
                    str_exc,
                )

    def _mapping(self, value, phase, path):
        if not self._enter(value, phase, path):
            return self.PLACEHOLDER
        result = {}
        try:
            try:
                items = value.items()
            except BaseException as exc:
                return self._failure(phase, path, "mapping_items", exc)
            try:
                iterator = iter(items)
            except BaseException as exc:
                return self._failure(
                    phase,
                    path,
                    "mapping_items_iter",
                    exc,
                )
            index = 0
            while True:
                try:
                    pair = next(iterator)
                except StopIteration:
                    break
                except BaseException as exc:
                    result[f"<serialization_error_item_{index}>"] = (
                        self._failure(
                            phase,
                            f"{path}[{index}]",
                            "mapping_items_next",
                            exc,
                        )
                    )
                    break
                try:
                    key, item = pair
                except BaseException as exc:
                    result[f"<serialization_error_item_{index}>"] = (
                        self._failure(
                            phase,
                            f"{path}[{index}]",
                            "mapping_item_unpack",
                            exc,
                        )
                    )
                    index += 1
                    continue
                try:
                    normalized_key = str(key)
                except BaseException as exc:
                    normalized_key = f"<serialization_error_key_{index}>"
                    self._failure(
                        phase,
                        f"{path}[{index}].key",
                        "mapping_key",
                        exc,
                    )
                result[normalized_key] = self.serialize(
                    item,
                    phase,
                    f"{path}[{index}]",
                )
                index += 1
            return result
        finally:
            self._active.discard(id(value))

    def _iterable(self, value, phase, path):
        if not self._enter(value, phase, path):
            return [self.PLACEHOLDER]
        result = []
        try:
            try:
                iterator = iter(value)
            except BaseException as exc:
                return [self._failure(phase, path, "iterable", exc)]
            index = 0
            while True:
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                except BaseException as exc:
                    result.append(
                        self._failure(
                            phase,
                            f"{path}[{index}]",
                            "iterable_next",
                            exc,
                        )
                    )
                    break
                result.append(
                    self.serialize(item, phase, f"{path}[{index}]")
                )
                index += 1
            return result
        finally:
            self._active.discard(id(value))

    def _enter(self, value, phase, path):
        identity = id(value)
        if identity in self._active:
            self._failure(
                phase,
                path,
                "cycle",
                RuntimeError("cyclic result value"),
            )
            return False
        self._active.add(identity)
        return True

    def _failure(self, phase, path, operation, exc):
        self.diagnostics.append(
            {
                "phase": phase,
                "path": path,
                "operation": operation,
                **_safe_error_details(exc),
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

        callbacks = _BoundedCallbacks()
        monitor_callbacks = (
            _SerializedCallbackLane() if self.monitor is not None else None
        )
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
            if monitor_callbacks is not None:
                monitor_callbacks.close(
                    time.monotonic() + config.flush_timeout_sec
                )
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
                quality_metrics = (
                    serialized_quality
                    if isinstance(serialized_quality, dict)
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
                hardware_metrics = (
                    serialized_hardware
                    if isinstance(serialized_hardware, dict)
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
        if monitor_callbacks is not None:
            monitor_callbacks.close(
                time.monotonic() + config.flush_timeout_sec
            )
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
    def _evaluator_sample_count(quality_metrics):
        for key in ("Total Samples", "total_samples", "num_samples"):
            value = quality_metrics.get(key)
            if (
                not isinstance(value, bool)
                and isinstance(value, Number)
                and math.isfinite(float(value))
            ):
                return value
        return None

    @classmethod
    def _serializable(cls, value):
        return _TotalSerializer().serialize(value, "serialization")
