import json
from threading import Lock

import numpy as np
import pytest

import core.async_inference as async_inference
import core.async_inference.runner as runner_module

from core.async_inference.runner import AsyncBenchmarkRunner
from core.async_inference.types import (
    AsyncInferenceConfig,
    AsyncScenario,
    RunStatus,
)


class Loader:
    def __init__(self, events=None):
        self.samples = [
            {"input": np.array([1.0]), "label": 2.0},
            {"input": np.array([2.0]), "label": 4.0},
            {"input": np.array([3.0]), "label": 6.0},
        ]
        self.current_idx = 0
        self.events = events

    def get_metadata(self):
        return {
            "total_samples": len(self.samples),
            "is_static_batched": False,
        }

    def load_by_index(self, index):
        if self.events is not None:
            self.events.append(f"load:{index}")
        return self.samples[index]

    def load_batch(self, batch_size):
        if self.events is not None:
            self.events.append("warmup_load")
        start = self.current_idx
        end = min(start + batch_size, len(self.samples))
        self.current_idx = end
        return self.samples[start:end]


class Runtime:
    compiled_model = None

    def __init__(self, events=None):
        self.events = events

    def supports_generate(self):
        return False

    def max_concurrent_workers(self):
        return 1

    def supports_dynamic_batching(self):
        return True

    def max_dynamic_batch_size(self):
        return None

    def run(self, inputs):
        if self.events is not None:
            self.events.append("runtime")
        return {"output": inputs["input"] * 2}

    def warmup(self, inputs, num_runs=1):
        if self.events is not None:
            self.events.append(f"warmup:{num_runs}")


class Evaluator:
    def __init__(self, events=None):
        self.correct = 0
        self.total = 0
        self.events = events
        self._lock = Lock()

    def add_batch(self, outputs, labels, timing_ms):
        predicted = outputs["output"].reshape(-1)
        expected = np.asarray(labels)
        with self._lock:
            self.correct += int(np.sum(predicted == expected))
            self.total += len(expected)
            if self.events is not None:
                self.events.append("evaluate")

    def compute(self):
        return {
            "accuracy": self.correct / self.total if self.total else 0.0,
            "Total Samples": self.total,
        }


class Monitor:
    def __init__(self, events=None):
        self.events = events if events is not None else []

    def start(self):
        self.events.append("monitor_start")

    def stop(self):
        self.events.append("monitor_stop")

    def summary(self):
        return {"hw_test_samples": 1}


def test_runner_returns_quality_async_and_hardware_metrics():
    events = []
    monitor = Monitor(events)
    result = AsyncBenchmarkRunner(
        dataloader=Loader(events),
        runtime=Runtime(events),
        evaluator=Evaluator(events),
        monitor=monitor,
    ).run(
        AsyncInferenceConfig(
            queue_capacity=4,
            max_batch_size=2,
            batch_timeout_ms=0,
            min_samples=1,
        ),
        warmup_runs=1,
    )

    assert result.status is RunStatus.VALID
    assert result.metrics["accuracy"] == 1.0
    assert result.metrics["Total Samples"] == 3
    assert result.metrics["async_completed_requests"] == 3
    assert result.metrics["async_evaluator_samples"] == 3
    assert result.details["flush_duration_ms"] >= 0
    assert result.metrics["hw_test_samples"] == 1
    assert events.index("warmup:1") < events.index("load:0")
    assert events.index("load:0") < events.index("monitor_start")
    assert events.index("monitor_start") < events.index("runtime")
    assert events.index("evaluate") < events.index("monitor_stop")
    assert events.count("monitor_stop") == 1
    json.dumps(result.metrics)
    json.dumps(result.details)


def test_server_like_reports_target_achieved_and_gap():
    result = AsyncBenchmarkRunner(
        dataloader=Loader(),
        runtime=Runtime(),
        evaluator=Evaluator(),
    ).run(
        AsyncInferenceConfig(
            scenario=AsyncScenario.SERVER_LIKE,
            queue_capacity=4,
            batch_timeout_ms=0,
            target_qps=1_000_000_000_000_000_000,
            min_samples=3,
            max_samples=3,
        ),
        warmup_runs=0,
    )

    assert result.metrics["async_target_qps"] == 1_000_000_000_000_000_000
    assert result.metrics["async_achieved_qps"] > 0
    assert result.metrics["async_target_qps_gap"] == (
        result.metrics["async_achieved_qps"]
        - 1_000_000_000_000_000_000
    )


def test_measurement_starts_at_the_first_request_issue_boundary():
    traces = []
    result = AsyncBenchmarkRunner(
        Loader(),
        Runtime(),
        Evaluator(),
        trace_callback=traces.append,
    ).run(
        AsyncInferenceConfig(batch_timeout_ms=0, min_samples=1),
        warmup_runs=0,
    )

    first = min(traces, key=lambda trace: trace.request_id)
    assert result.details["measurement"]["started_monotonic_ns"] == (
        first.issued_ns
    )


class StaticLoader:
    def __init__(self):
        self.inputs = np.asarray([[1.0], [2.0], [3.0]])
        self.labels = np.asarray([2.0, 4.0, 6.0])
        self.current_idx = 0
        self.load_batch_calls = []
        self.indexes = []

    def get_metadata(self):
        return {"total_samples": 3, "is_static_batched": True}

    def load_batch(self, batch_size):
        self.load_batch_calls.append(batch_size)
        start = self.current_idx
        end = min(start + batch_size, 3)
        self.current_idx = end
        return {
            "input": self.inputs[start:end],
            "label": self.labels[start:end],
        }

    def load_by_index(self, index):
        self.indexes.append(index)
        return {"input": self.inputs[index], "label": self.labels[index]}


class WarmupShapeRuntime(Runtime):
    def __init__(self):
        super().__init__()
        self.warmup_shapes = []

    def warmup(self, inputs, num_runs=1):
        self.warmup_shapes.append((inputs["input"].shape, num_runs))


def test_static_loader_warmup_uses_real_batch_and_resets_cursor():
    loader = StaticLoader()
    runtime = WarmupShapeRuntime()

    result = AsyncBenchmarkRunner(
        loader,
        runtime,
        Evaluator(),
    ).run(
        AsyncInferenceConfig(
            queue_capacity=2,
            max_batch_size=2,
            batch_timeout_ms=0,
            min_samples=1,
        ),
        warmup_runs=2,
    )

    assert result.status is RunStatus.VALID
    assert loader.load_batch_calls == [2]
    assert runtime.warmup_shapes == [((2, 1), 2)]
    assert loader.current_idx == 0
    assert loader.indexes == [0, 1, 2]
    assert result.metrics["async_completed_samples"] == 3


def test_zero_warmup_does_not_touch_sequential_loader_cursor():
    loader = Loader()
    loader.current_idx = 2

    result = AsyncBenchmarkRunner(
        loader,
        Runtime(),
        Evaluator(),
    ).run(
        AsyncInferenceConfig(batch_timeout_ms=0, min_samples=1),
        warmup_runs=0,
    )

    assert result.status is RunStatus.VALID
    assert loader.current_idx == 2


class SideEffectProbeLoader(Loader):
    def __init__(self):
        super().__init__()
        self.metadata_calls = 0
        self.warmup_load_calls = 0

    def get_metadata(self):
        self.metadata_calls += 1
        return super().get_metadata()

    def load_batch(self, batch_size):
        self.warmup_load_calls += 1
        return super().load_batch(batch_size)


def test_config_and_warmup_validation_precede_loader_side_effects():
    loader = SideEffectProbeLoader()
    runner = AsyncBenchmarkRunner(loader, Runtime(), Evaluator())

    with pytest.raises(ValueError, match="queue_capacity"):
        runner.run(AsyncInferenceConfig(queue_capacity=0))
    assert loader.metadata_calls == 0
    assert loader.warmup_load_calls == 0

    with pytest.raises(ValueError, match="warmup_runs"):
        runner.run(AsyncInferenceConfig(), warmup_runs=-1)
    assert loader.metadata_calls == 0
    assert loader.warmup_load_calls == 0


class PartialFailureLoader(Loader):
    def load_by_index(self, index):
        if index == 1:
            raise RuntimeError("dataset read failed")
        return super().load_by_index(index)


class UnprintableError(RuntimeError):
    def __str__(self):
        raise RuntimeError("error string conversion failed")


class UnprintableFailureLoader(Loader):
    def load_by_index(self, index):
        if index == 1:
            raise UnprintableError()
        return super().load_by_index(index)


def test_partial_producer_failure_closes_and_drains_accepted_requests():
    monitor = Monitor()
    result = AsyncBenchmarkRunner(
        PartialFailureLoader(),
        Runtime(),
        Evaluator(),
        monitor=monitor,
    ).run(
        AsyncInferenceConfig(batch_timeout_ms=0, min_samples=1),
        warmup_runs=0,
    )

    assert result.status is RunStatus.INVALID
    assert "producer_error" in result.invalid_reasons
    assert result.metrics["async_accepted_requests"] == 1
    assert result.metrics["async_completed_requests"] == 1
    assert result.details["producer"] == {
        "attempted": 1,
        "accepted": 1,
        "rejected": 0,
        "producer_load_ms": None,
        "error": {
            "error_type": "RuntimeError",
            "error_message": "dataset read failed",
        },
    }
    assert result.details["outstanding_request_ids"] == []
    assert monitor.events == ["monitor_start", "monitor_stop"]


def test_unprintable_producer_error_cannot_mask_lifecycle_cleanup():
    monitor = Monitor()
    result = AsyncBenchmarkRunner(
        UnprintableFailureLoader(),
        Runtime(),
        Evaluator(),
        monitor=monitor,
    ).run(
        AsyncInferenceConfig(batch_timeout_ms=0, min_samples=1),
        warmup_runs=0,
    )

    assert result.details["producer"]["error"] == {
        "error_type": "UnprintableError",
        "error_message": "<unprintable UnprintableError>",
    }
    assert monitor.events == ["monitor_start", "monitor_stop"]
    assert result.details["outstanding_request_ids"] == []


class RejectingLifecycleEngine:
    instances = []

    def __init__(self, runtime, pipeline, config, coordinator, metrics):
        del runtime, pipeline, config, coordinator
        self.metrics = metrics
        self.events = []
        type(self).instances.append(self)

    def start(self):
        self.events.append("start")

    def submit(self, request, block):
        del block
        self.events.append(f"submit:{request.request_id}")
        self.metrics.record_submitted()
        self.metrics.record_rejected("forced_rejection")
        return False

    def close_submission(self):
        self.events.append("close")

    def flush(self):
        self.events.append("flush")
        raise RuntimeError("flush exploded")

    def shutdown(self):
        self.events.append("shutdown")
        return False

    def outstanding_request_ids(self):
        return (7,)


def test_flush_exception_still_stops_monitor_and_runs_shutdown(monkeypatch):
    RejectingLifecycleEngine.instances.clear()
    monkeypatch.setattr(
        runner_module,
        "AsyncInferenceEngine",
        RejectingLifecycleEngine,
    )
    monitor = Monitor()

    result = AsyncBenchmarkRunner(
        Loader(),
        Runtime(),
        Evaluator(),
        monitor=monitor,
    ).run(
        AsyncInferenceConfig(batch_timeout_ms=0, min_samples=1),
        warmup_runs=0,
    )

    engine = RejectingLifecycleEngine.instances[-1]
    assert engine.events == [
        "start",
        "submit:0",
        "submit:1",
        "submit:2",
        "close",
        "flush",
        "shutdown",
    ]
    assert monitor.events == ["monitor_start", "monitor_stop"]
    assert result.status is RunStatus.INVALID
    assert "flush_timeout" in result.invalid_reasons
    assert "worker_shutdown_failed" in result.invalid_reasons
    assert result.details["outstanding_request_ids"] == [7]
    assert result.details["lifecycle_errors"] == [
        {
            "phase": "flush",
            "error_type": "RuntimeError",
            "error_message": "flush exploded",
        }
    ]


class LowercaseCountEvaluator(Evaluator):
    def __init__(self, reported_total):
        super().__init__()
        self.reported_total = reported_total

    def compute(self):
        return {
            "accuracy": 1.0,
            "total_samples": self.reported_total,
        }


def test_lowercase_evaluator_count_mismatch_invalidates_run():
    result = AsyncBenchmarkRunner(
        Loader(),
        Runtime(),
        LowercaseCountEvaluator(reported_total=2),
    ).run(
        AsyncInferenceConfig(batch_timeout_ms=0, min_samples=1),
        warmup_runs=0,
    )

    assert result.metrics["async_evaluator_samples"] == 2
    assert result.status is RunStatus.INVALID
    assert "counter_invariant_failed" in result.invalid_reasons


def test_minimums_and_latency_slo_are_independent_validity_reasons():
    result = AsyncBenchmarkRunner(
        Loader(),
        Runtime(),
        Evaluator(),
    ).run(
        AsyncInferenceConfig(
            batch_timeout_ms=0,
            min_samples=4,
            min_duration_sec=100,
            latency_slo_ms=0.000_001,
        ),
        warmup_runs=0,
    )

    assert {
        "min_samples_not_met",
        "min_duration_not_met",
        "latency_slo_not_met",
    }.issubset(result.invalid_reasons)


class CollidingEvaluator(Evaluator):
    def compute(self):
        return {
            "accuracy": np.float32(1.0),
            "Total Samples": np.int64(self.total),
            "async_completed_requests": 999,
            "hw_power_w": "quality",
            "quality_vector": np.asarray([1, 2]),
            "quality_complex": 1 + 2j,
        }


class CollidingMonitor(Monitor):
    def summary(self):
        return {
            "hw_power_w": np.float32(12.5),
            "accuracy": -1,
            "async_completed_requests": -1,
        }


def test_metric_namespaces_have_safe_precedence_and_json_values():
    result = AsyncBenchmarkRunner(
        Loader(),
        Runtime(),
        CollidingEvaluator(),
        monitor=CollidingMonitor(),
    ).run(
        AsyncInferenceConfig(batch_timeout_ms=0, min_samples=1),
        warmup_runs=0,
    )

    assert result.metrics["accuracy"] == 1.0
    assert result.metrics["async_completed_requests"] == 3
    assert result.metrics["hw_power_w"] == 12.5
    assert result.metrics["quality_vector"] == [1, 2]
    assert result.metrics["quality_complex"] == "(1+2j)"
    assert "quality_metric_namespace_collision" in result.warnings
    assert "hardware_metric_namespace_violation" in result.warnings
    json.dumps(result.metrics, allow_nan=False)
    json.dumps(result.details, allow_nan=False)


class ComputeFailureEvaluator(Evaluator):
    def compute(self):
        raise RuntimeError("quality summary failed")


class SummaryFailureMonitor(Monitor):
    def summary(self):
        raise RuntimeError("hardware summary failed")


def test_evaluator_and_monitor_summary_failures_are_normalized():
    result = AsyncBenchmarkRunner(
        Loader(),
        Runtime(),
        ComputeFailureEvaluator(),
        monitor=SummaryFailureMonitor(),
    ).run(
        AsyncInferenceConfig(batch_timeout_ms=0, min_samples=1),
        warmup_runs=0,
    )

    assert result.status is RunStatus.INVALID
    assert "request_failed" in result.invalid_reasons
    assert "hardware_monitor_summary_failed" in result.warnings
    assert result.details["callback_errors"] == [
        {
            "phase": "evaluator_compute",
            "error_type": "RuntimeError",
            "error_message": "quality summary failed",
        },
        {
            "phase": "monitor_summary",
            "error_type": "RuntimeError",
            "error_message": "hardware summary failed",
        },
    ]


class StartFailureMonitor(Monitor):
    def start(self):
        self.events.append("monitor_start")
        raise RuntimeError("monitor unavailable")


def test_monitor_start_failure_is_warning_and_stop_is_still_exactly_once():
    monitor = StartFailureMonitor()
    result = AsyncBenchmarkRunner(
        Loader(),
        Runtime(),
        Evaluator(),
        monitor=monitor,
    ).run(
        AsyncInferenceConfig(batch_timeout_ms=0, min_samples=1),
        warmup_runs=0,
    )

    assert result.status is RunStatus.VALID
    assert "hardware_monitor_start_failed" in result.warnings
    assert monitor.events == ["monitor_start", "monitor_stop"]


def test_trace_callback_failure_is_a_warning_not_a_false_request_failure():
    def fail_trace(_trace):
        raise RuntimeError("trace sink unavailable")

    result = AsyncBenchmarkRunner(
        Loader(),
        Runtime(),
        Evaluator(),
        trace_callback=fail_trace,
    ).run(
        AsyncInferenceConfig(batch_timeout_ms=0, min_samples=1),
        warmup_runs=0,
    )

    assert result.status is RunStatus.VALID
    assert result.metrics["async_completed_requests"] == 3
    assert "request_trace_write_failed" in result.warnings


class SuccessfulRejectingEngine(RejectingLifecycleEngine):
    def flush(self):
        self.events.append("flush")
        return True

    def shutdown(self):
        self.events.append("shutdown")
        return True

    def outstanding_request_ids(self):
        return ()


class TimeoutLifecycleEngine(RejectingLifecycleEngine):
    def flush(self):
        self.events.append("flush")
        return False


class UnloadProbeRuntime(Runtime):
    def __init__(self):
        super().__init__()
        self.unload_calls = 0

    def unload(self):
        self.unload_calls += 1


class ComputeProbeEvaluator(Evaluator):
    def __init__(self):
        super().__init__()
        self.compute_calls = 0

    def compute(self):
        self.compute_calls += 1
        return super().compute()


def test_submit_rejections_are_preserved_as_invalid_results(monkeypatch):
    SuccessfulRejectingEngine.instances.clear()
    monkeypatch.setattr(
        runner_module,
        "AsyncInferenceEngine",
        SuccessfulRejectingEngine,
    )

    result = AsyncBenchmarkRunner(
        Loader(),
        Runtime(),
        Evaluator(),
    ).run(
        AsyncInferenceConfig(batch_timeout_ms=0, min_samples=1),
        warmup_runs=0,
    )

    assert result.metrics["async_rejected_requests"] == 3
    assert result.details["producer"]["rejected"] == 3
    assert "request_rejected" in result.invalid_reasons
    assert "no_samples" in result.invalid_reasons


def test_flush_and_shutdown_timeout_never_unload_a_live_runtime(monkeypatch):
    TimeoutLifecycleEngine.instances.clear()
    monkeypatch.setattr(
        runner_module,
        "AsyncInferenceEngine",
        TimeoutLifecycleEngine,
    )
    runtime = UnloadProbeRuntime()
    evaluator = ComputeProbeEvaluator()
    monitor = Monitor()

    result = AsyncBenchmarkRunner(
        Loader(),
        runtime,
        evaluator,
        monitor=monitor,
    ).run(
        AsyncInferenceConfig(batch_timeout_ms=0, min_samples=1),
        warmup_runs=0,
    )

    assert "flush_timeout" in result.invalid_reasons
    assert "worker_shutdown_failed" in result.invalid_reasons
    assert runtime.unload_calls == 0
    assert evaluator.compute_calls == 0
    assert result.details["quality_evaluation_skipped"] == (
        "engine_shutdown_failed"
    )
    assert monitor.events == ["monitor_start", "monitor_stop"]


class InterruptLifecycleEngine:
    instances = []

    def __init__(self, runtime, pipeline, config, coordinator, metrics):
        del runtime, pipeline, config, coordinator, metrics
        self.events = []
        type(self).instances.append(self)

    def start(self):
        self.events.append("start")

    def cancel_queued(self, reason):
        self.events.append(f"cancel:{reason}")
        return 0

    def close_submission(self):
        self.events.append("close")

    def flush(self):
        self.events.append("flush")
        return True

    def shutdown(self):
        self.events.append("shutdown")
        return True

    def outstanding_request_ids(self):
        return ()


class KeyboardInterruptProducer:
    def __init__(self, dataloader, submitter, config):
        del dataloader, submitter, config

    def run(self):
        raise KeyboardInterrupt("user interrupt")


def test_keyboard_interrupt_cancels_then_closes_and_cleans_up(monkeypatch):
    InterruptLifecycleEngine.instances.clear()
    monkeypatch.setattr(
        runner_module,
        "AsyncInferenceEngine",
        InterruptLifecycleEngine,
    )
    monkeypatch.setattr(
        runner_module,
        "OfflineProducer",
        KeyboardInterruptProducer,
    )

    result = AsyncBenchmarkRunner(
        Loader(),
        Runtime(),
        Evaluator(),
    ).run(
        AsyncInferenceConfig(batch_timeout_ms=0, min_samples=1),
        warmup_runs=0,
    )

    engine = InterruptLifecycleEngine.instances[-1]
    assert engine.events == [
        "start",
        "cancel:KeyboardInterrupt",
        "close",
        "flush",
        "shutdown",
    ]
    assert "producer_error" in result.invalid_reasons
    assert result.details["producer"]["error"]["error_type"] == (
        "KeyboardInterrupt"
    )


class SystemExitProducer(KeyboardInterruptProducer):
    def run(self):
        raise SystemExit("fatal producer exit")


class FatalStartEngine(InterruptLifecycleEngine):
    def start(self):
        self.events.append("start")
        raise SystemExit("fatal engine start")


class InterruptingCloseEngine(InterruptLifecycleEngine):
    def close_submission(self):
        self.events.append("close")
        raise KeyboardInterrupt("cleanup interrupt")


def test_fatal_producer_baseexception_is_reraised_after_cleanup(monkeypatch):
    InterruptLifecycleEngine.instances.clear()
    monkeypatch.setattr(
        runner_module,
        "AsyncInferenceEngine",
        InterruptLifecycleEngine,
    )
    monkeypatch.setattr(runner_module, "OfflineProducer", SystemExitProducer)

    with pytest.raises(SystemExit, match="fatal producer exit"):
        AsyncBenchmarkRunner(
            Loader(),
            Runtime(),
            Evaluator(),
        ).run(
            AsyncInferenceConfig(batch_timeout_ms=0, min_samples=1),
            warmup_runs=0,
        )

    assert InterruptLifecycleEngine.instances[-1].events == [
        "start",
        "close",
        "flush",
        "shutdown",
    ]


def test_fatal_engine_start_is_reraised_after_cleanup(monkeypatch):
    FatalStartEngine.instances.clear()
    monkeypatch.setattr(
        runner_module,
        "AsyncInferenceEngine",
        FatalStartEngine,
    )

    with pytest.raises(SystemExit, match="fatal engine start"):
        AsyncBenchmarkRunner(
            Loader(),
            Runtime(),
            Evaluator(),
        ).run(
            AsyncInferenceConfig(batch_timeout_ms=0, min_samples=1),
            warmup_runs=0,
        )

    assert FatalStartEngine.instances[-1].events == [
        "start",
        "close",
        "flush",
        "shutdown",
    ]


def test_cleanup_baseexception_does_not_mask_original_producer_exit(
    monkeypatch,
):
    InterruptingCloseEngine.instances.clear()
    monkeypatch.setattr(
        runner_module,
        "AsyncInferenceEngine",
        InterruptingCloseEngine,
    )
    monkeypatch.setattr(runner_module, "OfflineProducer", SystemExitProducer)

    with pytest.raises(SystemExit, match="fatal producer exit"):
        AsyncBenchmarkRunner(
            Loader(),
            Runtime(),
            Evaluator(),
        ).run(
            AsyncInferenceConfig(batch_timeout_ms=0, min_samples=1),
            warmup_runs=0,
        )

    assert InterruptingCloseEngine.instances[-1].events == [
        "start",
        "close",
        "flush",
        "shutdown",
    ]


class StopFailureMonitor(Monitor):
    def stop(self):
        self.events.append("monitor_stop")
        raise RuntimeError("monitor stop failed")


def test_monitor_stop_failure_is_warning_and_does_not_skip_shutdown():
    monitor = StopFailureMonitor()
    result = AsyncBenchmarkRunner(
        Loader(),
        Runtime(),
        Evaluator(),
        monitor=monitor,
    ).run(
        AsyncInferenceConfig(batch_timeout_ms=0, min_samples=1),
        warmup_runs=0,
    )

    assert result.status is RunStatus.VALID
    assert "hardware_monitor_stop_failed" in result.warnings
    assert monitor.events == ["monitor_start", "monitor_stop"]


def test_runner_is_exported_from_async_inference_package():
    assert async_inference.AsyncBenchmarkRunner is AsyncBenchmarkRunner
    assert "AsyncBenchmarkRunner" in async_inference.__all__


def test_request_timeout_is_reported_without_changing_terminal_category():
    result = AsyncBenchmarkRunner(
        Loader(),
        Runtime(),
        Evaluator(),
    ).run(
        AsyncInferenceConfig(
            batch_timeout_ms=0,
            request_timeout_ms=0.000_001,
            min_samples=1,
        ),
        warmup_runs=0,
    )

    assert result.metrics["async_completed_requests"] == 3
    assert result.metrics["async_timed_out_requests"] == 3
    assert "request_timeout" in result.invalid_reasons


class FailingRuntime(Runtime):
    def run(self, inputs):
        del inputs
        raise RuntimeError("runtime failed")


def test_runtime_failures_report_no_completed_samples_without_deadlock():
    result = AsyncBenchmarkRunner(
        Loader(),
        FailingRuntime(),
        Evaluator(),
    ).run(
        AsyncInferenceConfig(batch_timeout_ms=0, min_samples=1),
        warmup_runs=0,
    )

    assert result.metrics["async_completed_samples"] == 0
    assert result.metrics["async_failed_requests"] >= 1
    assert "request_failed" in result.invalid_reasons
    assert "no_samples" in result.invalid_reasons
    assert result.details["outstanding_request_ids"] == []
