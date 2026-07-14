import math
import time
from collections.abc import Mapping
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
        callback_errors = []
        submitter = _MeasuredSubmitter(
            engine,
            metrics,
            self.monitor,
            time.monotonic_ns,
            callbacks,
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
                result = callbacks.invoke(
                    "monitor_stop",
                    self.monitor.stop,
                    time.monotonic() + config.flush_timeout_sec,
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
                lambda: dict(self.evaluator.compute()),
                time.monotonic() + config.flush_timeout_sec,
            )
            if result.diagnostic is None:
                quality_metrics = self._serializable(result.value)
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
            result = callbacks.invoke(
                "monitor_summary",
                lambda: dict(self.monitor.summary()),
                time.monotonic() + config.flush_timeout_sec,
            )
            if result.diagnostic is None:
                hardware_metrics = self._serializable(result.value)
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

        reasons = tuple(sorted(invalid_reasons))
        warning_values = tuple(sorted(warnings))
        status = RunStatus.INVALID if reasons else RunStatus.VALID
        final_metrics["async_run_status"] = status.value
        final_metrics["async_invalid_reasons"] = ",".join(reasons)
        outstanding_callbacks = callbacks.outstanding()
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
                "quality_evaluation_skipped": quality_evaluation_skipped,
                "status": status.value,
            }
        )
        return AsyncBenchmarkResult(
            metrics=self._serializable(final_metrics),
            details=self._serializable(details),
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
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, Mapping):
            return {
                str(key): cls._serializable(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [cls._serializable(item) for item in value]
        if isinstance(value, Number):
            if hasattr(value, "item"):
                item = value.item()
                if item is not value:
                    return cls._serializable(item)
            return str(value)
        if hasattr(value, "tolist"):
            return cls._serializable(value.tolist())
        return str(value)
