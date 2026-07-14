import math
import time
from collections.abc import Mapping
from numbers import Integral, Number

from core.inference_pipeline import InferencePipeline

from .completion import CompletionCoordinator
from .engine import AsyncInferenceEngine
from .metrics import AsyncMetricsCollector
from .producers import OfflineProducer, ServerLikeProducer
from .types import AsyncBenchmarkResult, AsyncScenario, RunStatus


class _MeasuredSubmitter:
    def __init__(self, engine, metrics, monitor, clock_ns):
        self.engine = engine
        self.metrics = metrics
        self.monitor = monitor
        self.clock_ns = clock_ns
        self.started = False
        self.monitor_start_attempted = False
        self.attempted = 0
        self.accepted = 0
        self.rejected = 0

    def ensure_measurement(self, *, start_monitor, started_ns=None):
        if self.started:
            return
        self.metrics.begin_measurement(
            self.clock_ns() if started_ns is None else started_ns
        )
        self.started = True
        if start_monitor and self.monitor is not None:
            self.monitor_start_attempted = True
            try:
                self.monitor.start()
            except Exception:
                self.metrics.add_warning("hardware_monitor_start_failed")

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

    def run(self, config, warmup_runs=1):
        config.validate()
        if (
            isinstance(warmup_runs, bool)
            or not isinstance(warmup_runs, Integral)
            or warmup_runs < 0
        ):
            raise ValueError("warmup_runs must be a non-negative integer")
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

        submitter = _MeasuredSubmitter(
            engine,
            metrics,
            self.monitor,
            time.monotonic_ns,
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

        try:
            engine.start()
            start_succeeded = True
        except KeyboardInterrupt as exc:
            submitter.ensure_measurement(start_monitor=False)
            producer_error = self._error_details(exc)
            metrics.add_invalid_reason("producer_error")
            lifecycle_errors.append(
                {"phase": "start", **self._error_details(exc)}
            )
        except Exception as exc:
            submitter.ensure_measurement(start_monitor=False)
            metrics.add_invalid_reason("worker_shutdown_failed")
            lifecycle_errors.append(
                {"phase": "start", **self._error_details(exc)}
            )
        except BaseException as exc:
            submitter.ensure_measurement(start_monitor=False)
            metrics.add_invalid_reason("worker_shutdown_failed")
            lifecycle_errors.append(
                {"phase": "start", **self._error_details(exc)}
            )
            fatal_error = exc

        if start_succeeded:
            try:
                producer_result = producer.run()
            except KeyboardInterrupt as exc:
                submitter.ensure_measurement(start_monitor=False)
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
                submitter.ensure_measurement(start_monitor=False)
                producer_error = self._error_details(exc)
                metrics.add_invalid_reason("producer_error")
            except BaseException as exc:
                submitter.ensure_measurement(start_monitor=False)
                fatal_error = exc

        submitter.ensure_measurement(start_monitor=False)
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
                try:
                    self.monitor.stop()
                except Exception as exc:
                    lifecycle_errors.append(
                        {"phase": "monitor_stop", **self._error_details(exc)}
                    )
                    metrics.add_warning("hardware_monitor_stop_failed")
                except BaseException as exc:
                    lifecycle_errors.append(
                        {"phase": "monitor_stop", **self._error_details(exc)}
                    )
                    metrics.add_warning("hardware_monitor_stop_failed")
                    if fatal_error is None:
                        fatal_error = exc

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

        callback_errors = []
        quality_evaluation_skipped = None
        if shutdown:
            try:
                quality_metrics = self._serializable(
                    dict(self.evaluator.compute())
                )
            except Exception as exc:
                quality_metrics = {}
                callback_errors.append(
                    {
                        "phase": "evaluator_compute",
                        **self._error_details(exc),
                    }
                )
                invalid_reasons.add("request_failed")
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
            try:
                hardware_metrics = self._serializable(
                    dict(self.monitor.summary())
                )
            except Exception as exc:
                callback_errors.append(
                    {"phase": "monitor_summary", **self._error_details(exc)}
                )
                warnings.add("hardware_monitor_summary_failed")
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
        details.update(
            {
                "invalid_reasons": list(reasons),
                "warnings": list(warning_values),
                "quality_metrics": quality_metrics,
                "hardware_metrics": hardware_metrics,
                "evaluator_samples": evaluator_samples,
                "callback_errors": callback_errors,
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
        error_type = type(exc).__name__
        try:
            message = str(exc)
        except BaseException:
            message = f"<unprintable {error_type}>"
        return {
            "error_type": error_type,
            "error_message": message,
        }

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
