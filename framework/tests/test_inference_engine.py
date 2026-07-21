import threading

import numpy as np
import pytest

from core.async_inference.types import AsyncInferenceConfig, RunStatus, TerminalStatus
from core.benchmarkrunner import BenchmarkRunner
from core.inference_engine import InferenceEngine
import core.runtime_executor as runtime_executor_module
from core.runtime_executor import RuntimeExecution, RuntimeExecutor


class FakeLoader:
    def __init__(self):
        self.current_idx = 0
        self.samples = [
            {"input": np.array([1.0, 2.0], dtype=np.float32), "label": 3},
            {"input": np.array([3.0, 4.0], dtype=np.float32), "label": 7},
        ]

    def get_metadata(self):
        return {
            "is_static_batched": False,
            "total_samples": len(self.samples),
        }

    def load_batch(self, batch_size):
        batch = self.samples[self.current_idx:self.current_idx + batch_size]
        self.current_idx += len(batch)
        return batch

    def load_by_index(self, index):
        return self.samples[index]


class FakeRuntime:
    compiled_model = None

    def __init__(self):
        self.warmup_calls = 0
        self.warmup_invocations = []

    def supports_generate(self):
        return False

    def max_concurrent_workers(self):
        return 1

    def supports_dynamic_batching(self):
        return False

    def max_dynamic_batch_size(self):
        return 1

    def supports_batch_generation(self):
        return False

    def run(self, inputs):
        values = inputs["input"]
        return {"output": values.sum(axis=1, keepdims=True)}

    def warmup(self, inputs, num_runs=1):
        self.warmup_invocations.append((inputs, num_runs))
        self.warmup_calls += num_runs


class FailingRuntime(FakeRuntime):
    def __init__(self, primary):
        super().__init__()
        self.primary = primary

    def run(self, inputs):
        del inputs
        raise self.primary


class RecordingExecutor(RuntimeExecutor):
    def __init__(self):
        self.executions = []
        self.acknowledged = []
        self.engine = None

    def execute(self, inputs, timeout=None):
        del timeout
        values = inputs["input"]
        execution = RuntimeExecution(
            outputs={"output": values.sum(axis=1, keepdims=True)},
            timing_ms=1.0,
        )
        self.executions.append(execution)
        return execution

    def acknowledge(self, execution):
        outstanding = self.engine.completion.snapshot_outstanding()
        self.acknowledged.append((execution, outstanding))

    def shutdown(self, timeout):
        del timeout
        return True


class FailingExecutor(RecordingExecutor):
    def __init__(self, primary):
        super().__init__()
        self.primary = primary

    def execute(self, inputs, timeout=None):
        del inputs, timeout
        raise self.primary


class FailingAcknowledgeExecutor(RecordingExecutor):
    def __init__(self, acknowledge_error):
        super().__init__()
        self.acknowledge_error = acknowledge_error

    def acknowledge(self, execution):
        super().acknowledge(execution)
        raise self.acknowledge_error


class AsyncRecordingExecutor(RecordingExecutor):
    def acknowledge(self, execution):
        self.acknowledged.append(execution)


class StaticLoader:
    def __init__(self):
        self._current_idx = 0
        self.sample = {
            "input": {
                "input": np.array(
                    [[1.0, 2.0], [3.0, 4.0]],
                    dtype=np.float32,
                )
            },
            "label": [3, 7],
        }

    def get_metadata(self):
        return {"is_static_batched": True, "total_samples": 2}

    def load_batch(self, batch_size):
        del batch_size
        if self._current_idx:
            return {}
        self._current_idx = 1
        return self.sample

    def load_by_index(self, index):
        assert index == 0
        return self.sample


class FakeEvaluator:
    def __init__(self):
        self.rows = []
        self.compute_calls = 0

    def add_batch(self, outputs, labels, timing_ms):
        del timing_ms
        predictions = outputs["output"].reshape(-1).tolist()
        self.rows.extend(zip(predictions, labels))

    def compute(self):
        self.compute_calls += 1
        return {
            "pairs": list(self.rows),
            "Total Samples": len(self.rows),
        }


class JsonSafeFakeEvaluator(FakeEvaluator):
    def compute(self):
        metrics = super().compute()
        metrics["pairs"] = [list(pair) for pair in metrics["pairs"]]
        return metrics


class FailingEvaluator(FakeEvaluator):
    def __init__(self, primary):
        super().__init__()
        self.primary = primary

    def add_batch(self, outputs, labels, timing_ms):
        del outputs, labels, timing_ms
        raise self.primary


class FailingDecoder:
    def __init__(self, primary):
        self.primary = primary

    def decode(self, outputs):
        del outputs
        raise self.primary


class HostileStringFatal(BaseException):
    def __str__(self):
        raise RuntimeError("hostile string conversion")


class TraceFatal(BaseException):
    pass


class HostileTypeNameMeta(type):
    def __getattribute__(cls, name):
        if name == "__name__":
            raise RuntimeError("hostile type-name access")
        return super().__getattribute__(name)


class TotallyHostileFatal(BaseException, metaclass=HostileTypeNameMeta):
    def __str__(self):
        raise RuntimeError("hostile string conversion")


class OwnedExecutionExecutor(RuntimeExecutor):
    """Native-like ownership probe for the e2e terminal transaction."""

    def __init__(
        self,
        *,
        returned=None,
        execute_error=None,
        acknowledge_error=None,
        shutdown_result=True,
        shutdown_error=None,
        events=None,
    ):
        self.returned = list(returned or [])
        self.execute_error = execute_error
        self.acknowledge_error = acknowledge_error
        self.shutdown_result = shutdown_result
        self.shutdown_error = shutdown_error
        self.events = [] if events is None else events
        self.executions = []
        self.acknowledged = []
        self.inflight = {}
        self.shutdown_calls = []
        self.engine = None
        self._next_token = 1

    def execute(self, inputs, timeout=None):
        del timeout
        self.events.append("execute")
        if self.execute_error is not None:
            raise self.execute_error
        if self.returned:
            template = self.returned.pop(0)
            token = template.dispatch_token
            if token is None:
                token = self._next_token
            execution = RuntimeExecution(
                outputs=template.outputs,
                timing_ms=template.timing_ms,
                generated_tokens=template.generated_tokens,
                dispatch_token=token,
                vendor_job_id=template.vendor_job_id,
                error_type=template.error_type,
                error_message=template.error_message,
            )
        else:
            token = self._next_token
            values = inputs["input"]
            execution = RuntimeExecution(
                outputs={"output": values.sum(axis=1, keepdims=True)},
                timing_ms=1.0,
                dispatch_token=token,
            )
        self._next_token = max(self._next_token + 1, token + 1)
        self.executions.append(execution)
        self.inflight[token] = execution
        return execution

    def acknowledge(self, execution):
        self.events.append("ack")
        self.acknowledged.append(execution)
        if self.acknowledge_error is not None:
            raise self.acknowledge_error
        assert self.inflight.get(execution.dispatch_token) is execution
        self.inflight.pop(execution.dispatch_token)

    def shutdown(self, timeout):
        self.events.append("shutdown")
        self.shutdown_calls.append(timeout)
        if self.shutdown_error is not None:
            raise self.shutdown_error
        return self.shutdown_result and not self.inflight


def _runtime_execution_error_type():
    assert hasattr(runtime_executor_module, "RuntimeExecutionError")
    return runtime_executor_module.RuntimeExecutionError


def _assert_exact_ack(executor):
    assert len(executor.acknowledged) == len(executor.executions)
    assert all(
        acknowledged is execution
        for acknowledged, execution in zip(
            executor.acknowledged,
            executor.executions,
        )
    )


def _assert_exact_primary(actual, expected):
    if actual is not expected:
        pytest.fail("exact primary object was not preserved", pytrace=False)


class RecordingMonitor:
    def __init__(self, events, summary):
        self.events = events
        self._summary = summary
        self.summary_calls = 0

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")

    def summary(self):
        self.summary_calls += 1
        return dict(self._summary)


def test_benchmark_runner_owns_monitor_and_delegates_inference():
    events = []
    monitor = RecordingMonitor(events, {"hw_samples": 1})
    runner = BenchmarkRunner(
        FakeLoader(),
        FakeRuntime(),
        FakeEvaluator(),
        monitor=monitor,
    )

    result = runner.run(warmup_runs=1, batch_size=2)

    assert isinstance(runner.engine, InferenceEngine)
    assert events == ["start", "stop"]
    assert result["Total Samples"] == 2
    assert result["hw_samples"] == 1
    assert monitor.summary_calls == 1


def test_benchmark_runner_stops_monitor_when_engine_raises():
    events = []
    monitor = RecordingMonitor(events, {})
    runner = BenchmarkRunner(
        FakeLoader(),
        FakeRuntime(),
        FailingEvaluator(RuntimeError("failed")),
        monitor=monitor,
    )

    with pytest.raises(RuntimeError, match="failed"):
        runner.run(warmup_runs=0, batch_size=2)

    assert events == ["start", "stop"]
    assert monitor.summary_calls == 0


def test_benchmark_runner_logs_exact_limiter_and_forwards_max_steps(capsys):
    runner = BenchmarkRunner(FakeLoader(), FakeRuntime(), FakeEvaluator())

    result = runner.run(warmup_runs=0, batch_size=1, max_steps=1)

    assert result["Total Samples"] == 1
    assert (
        "[BenchmarkRunner] 🛑 사용자가 요청한 리미터에 도달했습니다! "
        "(1 steps) - 즉각 탈출하여 결과를 채점합니다."
    ) in capsys.readouterr().out.splitlines()


def test_benchmark_runner_logs_every_tenth_batch_with_size_and_latency(capsys):
    loader = FakeLoader()
    loader.samples = [
        {
            "input": np.array([float(index), 1.0], dtype=np.float32),
            "label": index + 1,
        }
        for index in range(20)
    ]
    runner = BenchmarkRunner(loader, FakeRuntime(), FakeEvaluator())
    executor = RecordingExecutor()
    executor.engine = runner.engine
    runner.engine.runtime_executor = executor

    result = runner.run(warmup_runs=0, batch_size=2)

    assert result["Total Samples"] == 20
    assert (
        "  - Completed batch 10 (2 samples), Latency: 1.00 ms"
        in capsys.readouterr().out.splitlines()
    )


def test_benchmark_runner_logs_before_evaluator_compute(capsys):
    class PrintingEvaluator(FakeEvaluator):
        def compute(self):
            print("[Evaluator] compute called")
            return super().compute()

    runner = BenchmarkRunner(FakeLoader(), FakeRuntime(), PrintingEvaluator())

    runner.run(warmup_runs=0, batch_size=2)

    lines = capsys.readouterr().out.splitlines()
    assert lines.index(
        "[BenchmarkRunner] 🏆 Computing final metrics..."
    ) < lines.index("[Evaluator] compute called")


def test_benchmark_runner_recreates_engine_for_repeated_runs():
    evaluator = FakeEvaluator()
    runner = BenchmarkRunner(FakeLoader(), FakeRuntime(), evaluator)

    first = runner.run(warmup_runs=0, batch_size=2)
    first_engine = runner.engine
    second = runner.run(warmup_runs=0, batch_size=2)

    assert first["Total Samples"] == 2
    assert second["Total Samples"] == 4
    assert runner.engine is not first_engine
    assert runner._pipeline is runner.engine.pipeline
    assert runner.is_static_batched is runner.engine.pipeline.is_static_batched
    assert runner._stop_token_ids is runner.engine.pipeline.stop_token_ids


def test_e2e_engine_uses_inline_completion_and_no_async_threads():
    before = {thread.ident for thread in threading.enumerate()}
    engine = InferenceEngine(FakeLoader(), FakeRuntime(), FakeEvaluator())
    result = engine.run_e2e(batch_size=2)
    assert result == {"pairs": [(3.0, 3), (7.0, 7)], "Total Samples": 2}
    assert engine.completion.queue is None
    assert engine.completion.thread is None
    created = [t for t in threading.enumerate() if t.ident not in before]
    assert not [t for t in created if t.name.startswith("async-")]


def test_inference_engine_exposes_safe_async_diagnostics_before_run():
    engine = InferenceEngine(FakeLoader(), FakeRuntime(), FakeEvaluator())

    assert engine.failure_phase == "created"
    assert engine.runtime_unload_safe_after_failure is True


def test_inference_engine_exposes_controller_diagnostics_after_validation():
    engine = InferenceEngine(FakeLoader(), FakeRuntime(), FakeEvaluator())

    with pytest.raises(ValueError, match="warmup_runs"):
        engine.run_async(
            AsyncInferenceConfig(min_samples=1),
            warmup_runs=-1,
        )

    assert engine.failure_phase == "validation"
    assert engine.runtime_unload_safe_after_failure is True


def test_inference_engine_exposes_controller_diagnostics_after_success():
    engine = InferenceEngine(FakeLoader(), FakeRuntime(), FakeEvaluator())

    result = engine.run_async(
        AsyncInferenceConfig(batch_timeout_ms=0, min_samples=1),
        warmup_runs=0,
    )

    assert result.status is RunStatus.VALID
    assert engine.failure_phase == "complete"
    assert engine.runtime_unload_safe_after_failure is True


def test_same_inference_engine_type_owns_async_pipeline_and_executor():
    executor = AsyncRecordingExecutor()
    engine = InferenceEngine(
        FakeLoader(),
        FakeRuntime(),
        FakeEvaluator(),
        runtime_executor=executor,
    )
    result = engine.run_async(
        AsyncInferenceConfig(
            queue_capacity=4,
            worker_count=1,
            max_batch_size=1,
            min_samples=1,
        ),
        warmup_runs=0,
    )

    assert result.metrics["Total Samples"] == 2
    assert engine._async_controller.pipeline is engine.pipeline
    assert engine._async_controller.runtime_executor is executor


def test_async_engine_preserves_completion_coordinator_after_success():
    engine = InferenceEngine(FakeLoader(), FakeRuntime(), FakeEvaluator())

    result = engine.run_async(
        AsyncInferenceConfig(min_samples=1),
        warmup_runs=0,
    )

    assert result.metrics["Total Samples"] == 2
    assert engine.completion is not None
    assert engine.completion is engine._async_controller.completion
    assert engine.completion.snapshot_outstanding() == ()


def test_async_engine_preserves_completion_coordinator_after_failure():
    engine = InferenceEngine(
        FakeLoader(),
        FailingRuntime(RuntimeError("runtime failed")),
        FakeEvaluator(),
    )

    result = engine.run_async(
        AsyncInferenceConfig(min_samples=1),
        warmup_runs=0,
    )

    assert "request_failed" in result.invalid_reasons
    assert engine.completion is not None
    assert engine.completion is engine._async_controller.completion
    assert engine.completion.snapshot_outstanding() == ()


def test_same_engine_type_runs_e2e_and_async_with_same_quality():
    e2e = InferenceEngine(
        FakeLoader(),
        FakeRuntime(),
        JsonSafeFakeEvaluator(),
    ).run_e2e(batch_size=2)
    async_result = InferenceEngine(
        FakeLoader(),
        FakeRuntime(),
        JsonSafeFakeEvaluator(),
    ).run_async(
        AsyncInferenceConfig(
            queue_capacity=2,
            worker_count=1,
            max_batch_size=1,
            min_samples=1,
        ),
        warmup_runs=0,
    )

    assert e2e["Total Samples"] == async_result.metrics["Total Samples"] == 2
    assert async_result.metrics["pairs"] == e2e["pairs"]
    assert async_result.metrics["async_outstanding_requests"] == 0


def test_inference_engine_allows_only_one_execution_mode():
    engine = InferenceEngine(FakeLoader(), FakeRuntime(), FakeEvaluator())
    engine.run_e2e(batch_size=2)

    with pytest.raises(RuntimeError, match="may only be run once"):
        engine.run_async(
            AsyncInferenceConfig(min_samples=1),
            warmup_runs=0,
        )


def test_inference_engine_async_run_precludes_every_later_mode():
    engine = InferenceEngine(FakeLoader(), FakeRuntime(), FakeEvaluator())
    config = AsyncInferenceConfig(min_samples=1)
    engine.run_async(config, warmup_runs=0)

    with pytest.raises(RuntimeError, match="may only be run once"):
        engine.run_async(config, warmup_runs=0)
    with pytest.raises(RuntimeError, match="may only be run once"):
        engine.run_e2e(batch_size=2)


def test_e2e_engine_warmup_resets_loader_before_measurement():
    loader, runtime, evaluator = FakeLoader(), FakeRuntime(), FakeEvaluator()
    engine = InferenceEngine(loader, runtime, evaluator)
    engine.warmup(runs=1, batch_size=1)
    result = engine.run_e2e(batch_size=2)
    assert runtime.warmup_calls == 1
    assert result["Total Samples"] == 2


def test_e2e_runtime_exception_terminalizes_before_reraising_same_error():
    primary = RuntimeError("device execution failed")
    traces = []
    evaluator = FakeEvaluator()
    engine = InferenceEngine(
        FakeLoader(),
        FailingRuntime(primary),
        evaluator,
        trace_callback=traces.append,
    )

    with pytest.raises(RuntimeError) as raised:
        engine.run_e2e(batch_size=2)

    assert raised.value is primary
    assert engine.completion.snapshot_outstanding() == ()
    assert evaluator.rows == []
    assert len(traces) == 1
    assert traces[0].status is TerminalStatus.FAILED
    assert traces[0].sample_count == 2


def test_e2e_executor_exception_does_not_ack_missing_execution():
    primary = RuntimeError("executor failed")
    executor = FailingExecutor(primary)
    engine = InferenceEngine(
        FakeLoader(),
        FakeRuntime(),
        FakeEvaluator(),
        runtime_executor=executor,
    )
    executor.engine = engine

    with pytest.raises(RuntimeError) as raised:
        engine.run_e2e(batch_size=2)

    assert raised.value is primary
    assert executor.acknowledged == []
    assert engine.completion.snapshot_outstanding() == ()


@pytest.mark.parametrize("failure_site", ["decoder", "evaluator"])
@pytest.mark.parametrize("fatal_type", [KeyboardInterrupt, SystemExit])
def test_e2e_fatal_decoder_or_evaluator_terminalizes_acks_and_reraises_exact_primary(
    failure_site,
    fatal_type,
):
    primary = fatal_type(f"{failure_site} stopped")
    evaluator = (
        FailingEvaluator(primary)
        if failure_site == "evaluator"
        else FakeEvaluator()
    )
    decoder = FailingDecoder(primary) if failure_site == "decoder" else None
    traces = []
    executor = OwnedExecutionExecutor()
    engine = InferenceEngine(
        FakeLoader(),
        FakeRuntime(),
        evaluator,
        decoder=decoder,
        runtime_executor=executor,
        trace_callback=traces.append,
    )
    executor.engine = engine

    with pytest.raises(BaseException) as raised:
        engine.run_e2e(batch_size=2)

    assert raised.value is primary
    assert len(traces) == 1
    assert traces[0].status is TerminalStatus.FAILED
    assert engine.completion.terminal[0] == 2
    assert engine.completion.snapshot_outstanding() == ()
    _assert_exact_ack(executor)
    assert executor.inflight == {}
    assert executor.shutdown_calls == [0.0]
    assert evaluator.compute_calls == 0


def test_e2e_trace_baseexception_retires_execution_before_exact_reraise():
    primary = TraceFatal("trace sink stopped")
    traces = []
    events = []

    def write_trace(trace):
        traces.append(trace)
        events.append("trace")
        raise primary

    executor = OwnedExecutionExecutor(events=events)
    evaluator = FakeEvaluator()
    engine = InferenceEngine(
        FakeLoader(),
        FakeRuntime(),
        evaluator,
        runtime_executor=executor,
        trace_callback=write_trace,
    )
    executor.engine = engine

    with pytest.raises(TraceFatal) as raised:
        engine.run_e2e(batch_size=2)

    assert raised.value is primary
    assert len(traces) == 1
    assert engine.completion.terminal[0] == 2
    assert engine.completion.snapshot_outstanding() == ()
    _assert_exact_ack(executor)
    assert executor.inflight == {}
    assert events.index("trace") < events.index("ack") < events.index("shutdown")
    assert evaluator.compute_calls == 0


def test_e2e_ordinary_trace_exception_is_warning_only():
    traces = []

    def write_trace(trace):
        traces.append(trace)
        raise RuntimeError("trace storage unavailable")

    executor = OwnedExecutionExecutor()
    evaluator = FakeEvaluator()
    engine = InferenceEngine(
        FakeLoader(),
        FakeRuntime(),
        evaluator,
        runtime_executor=executor,
        trace_callback=write_trace,
    )
    executor.engine = engine

    result = engine.run_e2e(batch_size=2)

    assert result["Total Samples"] == 2
    assert len(traces) == 1
    assert engine.completion.snapshot_outstanding() == ()
    _assert_exact_ack(executor)
    assert executor.inflight == {}
    assert executor.shutdown_calls == [0.0]
    assert evaluator.compute_calls == 1


def test_e2e_runtime_hostile_string_baseexception_preserves_exact_primary():
    primary = HostileStringFatal("device exploded")
    traces = []
    executor = OwnedExecutionExecutor(execute_error=primary)
    evaluator = FakeEvaluator()
    engine = InferenceEngine(
        FakeLoader(),
        FakeRuntime(),
        evaluator,
        runtime_executor=executor,
        trace_callback=traces.append,
    )
    executor.engine = engine

    with pytest.raises(HostileStringFatal) as raised:
        engine.run_e2e(batch_size=2)

    assert raised.value is primary
    assert len(traces) == 1
    assert traces[0].status is TerminalStatus.FAILED
    assert traces[0].error_type == "HostileStringFatal"
    assert "device exploded" in traces[0].error_message
    assert len(traces[0].error_message) <= 512
    assert engine.completion.terminal[0] == 2
    assert engine.completion.snapshot_outstanding() == ()
    assert executor.acknowledged == []
    assert executor.shutdown_calls == [0.0]
    assert evaluator.compute_calls == 0


@pytest.mark.parametrize("failure_site", ["decoder", "evaluator"])
def test_e2e_hostile_type_name_callback_terminalizes_and_preserves_exact_primary(
    failure_site,
):
    primary = TotallyHostileFatal("callback payload")
    evaluator = (
        FailingEvaluator(primary)
        if failure_site == "evaluator"
        else FakeEvaluator()
    )
    decoder = FailingDecoder(primary) if failure_site == "decoder" else None
    traces = []
    executor = OwnedExecutionExecutor()
    engine = InferenceEngine(
        FakeLoader(),
        FakeRuntime(),
        evaluator,
        decoder=decoder,
        runtime_executor=executor,
        trace_callback=traces.append,
    )
    executor.engine = engine

    with pytest.raises(BaseException) as raised:
        engine.run_e2e(batch_size=2)

    _assert_exact_primary(raised.value, primary)
    assert len(traces) == 1
    assert traces[0].status is TerminalStatus.FAILED
    assert traces[0].error_type == "TotallyHostileFatal"
    assert "callback payload" in traces[0].error_message
    assert len(traces[0].error_message) <= 512
    assert engine.completion.terminal[0] == 2
    assert engine.completion.snapshot_outstanding() == ()
    _assert_exact_ack(executor)
    assert executor.inflight == {}
    assert executor.shutdown_calls == [0.0]
    assert evaluator.compute_calls == 0


def test_e2e_execute_hostile_type_name_terminalizes_without_ack():
    primary = TotallyHostileFatal("execute payload")
    traces = []
    executor = OwnedExecutionExecutor(execute_error=primary)
    evaluator = FakeEvaluator()
    engine = InferenceEngine(
        FakeLoader(),
        FakeRuntime(),
        evaluator,
        runtime_executor=executor,
        trace_callback=traces.append,
    )
    executor.engine = engine

    with pytest.raises(BaseException) as raised:
        engine.run_e2e(batch_size=2)

    _assert_exact_primary(raised.value, primary)
    assert len(traces) == 1
    assert traces[0].status is TerminalStatus.FAILED
    assert traces[0].error_type == "TotallyHostileFatal"
    assert "execute payload" in traces[0].error_message
    assert len(traces[0].error_message) <= 512
    assert engine.completion.terminal[0] == 2
    assert engine.completion.snapshot_outstanding() == ()
    assert executor.acknowledged == []
    assert executor.shutdown_calls == [0.0]
    assert evaluator.compute_calls == 0


def test_e2e_prepare_runtime_input_fatal_creates_no_known_request():
    primary = KeyboardInterrupt("input preparation stopped")
    executor = OwnedExecutionExecutor()
    evaluator = FakeEvaluator()
    engine = InferenceEngine(
        FakeLoader(),
        FakeRuntime(),
        evaluator,
        runtime_executor=executor,
    )
    executor.engine = engine

    def fail_preparation(_inputs):
        raise primary

    engine.pipeline.prepare_runtime_input = fail_preparation

    with pytest.raises(KeyboardInterrupt) as raised:
        engine.run_e2e(batch_size=2)

    assert raised.value is primary
    assert engine.completion.snapshot_outstanding() == ()
    assert len(engine.completion.terminal) == 0
    assert executor.executions == []
    assert executor.acknowledged == []
    assert executor.shutdown_calls == [0.0]
    assert evaluator.compute_calls == 0


def test_e2e_callback_primary_precedes_ack_and_shutdown_failures():
    primary = KeyboardInterrupt("evaluation stopped")
    acknowledge_error = RuntimeError("ack failed")
    shutdown_error = RuntimeError("shutdown failed")
    executor = OwnedExecutionExecutor(
        acknowledge_error=acknowledge_error,
        shutdown_error=shutdown_error,
    )
    evaluator = FailingEvaluator(primary)
    engine = InferenceEngine(
        FakeLoader(),
        FakeRuntime(),
        evaluator,
        runtime_executor=executor,
    )
    executor.engine = engine

    with pytest.raises(BaseException) as raised:
        engine.run_e2e(batch_size=2)

    assert raised.value is primary
    _assert_exact_ack(executor)
    assert executor.shutdown_calls == [0.0]
    assert len(executor.inflight) == 1
    assert engine.completion.snapshot_outstanding() == ()
    assert evaluator.compute_calls == 0


def test_e2e_ack_failure_is_primary_when_completion_succeeds():
    primary = RuntimeError("ack failed")
    executor = OwnedExecutionExecutor(acknowledge_error=primary)
    evaluator = FakeEvaluator()
    engine = InferenceEngine(
        FakeLoader(),
        FakeRuntime(),
        evaluator,
        runtime_executor=executor,
    )
    executor.engine = engine

    with pytest.raises(RuntimeError) as raised:
        engine.run_e2e(batch_size=2)

    assert raised.value is primary
    _assert_exact_ack(executor)
    assert executor.shutdown_calls == [0.0]
    assert engine.completion.terminal[0] == 2
    assert engine.completion.snapshot_outstanding() == ()
    assert evaluator.compute_calls == 0


@pytest.mark.parametrize("failure_kind", ["false", "raise"])
def test_e2e_shutdown_failure_prevents_compute(failure_kind):
    shutdown_error = RuntimeError("shutdown exploded")
    executor = OwnedExecutionExecutor(
        shutdown_result=failure_kind != "false",
        shutdown_error=shutdown_error if failure_kind == "raise" else None,
    )
    evaluator = FakeEvaluator()
    engine = InferenceEngine(
        FakeLoader(),
        FakeRuntime(),
        evaluator,
        runtime_executor=executor,
    )
    executor.engine = engine

    with pytest.raises(RuntimeError) as raised:
        engine.run_e2e(batch_size=2)

    if failure_kind == "raise":
        assert raised.value is shutdown_error
    else:
        assert str(raised.value) == "e2e runtime executor shutdown failed"
    assert executor.shutdown_calls == [0.0]
    assert evaluator.compute_calls == 0


def test_e2e_compute_occurs_after_executor_shutdown():
    events = []

    class EventEvaluator(FakeEvaluator):
        def compute(self):
            events.append("compute")
            return super().compute()

    executor = OwnedExecutionExecutor(events=events)
    evaluator = EventEvaluator()
    engine = InferenceEngine(
        FakeLoader(),
        FakeRuntime(),
        evaluator,
        runtime_executor=executor,
        trace_callback=lambda trace: events.append("terminal"),
    )
    executor.engine = engine

    engine.run_e2e(batch_size=2)

    assert events == ["execute", "terminal", "ack", "shutdown", "compute"]
    assert executor.shutdown_calls == [0.0]
    assert evaluator.compute_calls == 1


@pytest.mark.parametrize("failed_batch", [0, 1])
def test_e2e_failure_valued_execution_terminalizes_acks_then_raises(
    failed_batch,
):
    success = RuntimeExecution(
        outputs={"output": np.array([[3.0]], dtype=np.float32)},
        timing_ms=1.0,
        dispatch_token=11,
    )
    failure = RuntimeExecution(
        outputs=None,
        timing_ms=2.0,
        dispatch_token=12,
        vendor_job_id="vendor-diagnostic-only",
        error_type="DeviceError",
        error_message="failed",
    )
    returned = [failure] if failed_batch == 0 else [success, failure]
    executor = OwnedExecutionExecutor(returned=returned)
    traces = []
    evaluator = FakeEvaluator()
    engine = InferenceEngine(
        FakeLoader(),
        FakeRuntime(),
        evaluator,
        runtime_executor=executor,
        trace_callback=traces.append,
    )
    executor.engine = engine
    error_type = _runtime_execution_error_type()

    with pytest.raises(error_type) as raised:
        engine.run_e2e(batch_size=1)

    assert raised.value.error_type == "DeviceError"
    assert raised.value.error_message == "failed"
    assert raised.value.dispatch_token == 12
    assert len(executor.executions) == failed_batch + 1
    _assert_exact_ack(executor)
    assert executor.inflight == {}
    assert executor.shutdown_calls == [0.0]
    assert [engine.completion.terminal[index] for index in range(failed_batch + 1)] == [
        2
    ] * (failed_batch + 1)
    assert engine.completion.snapshot_outstanding() == ()
    assert len([trace for trace in traces if trace.status is TerminalStatus.FAILED]) == 1
    assert evaluator.compute_calls == 0


def test_failure_valued_execution_primary_precedes_ack_and_shutdown_failures():
    failure = RuntimeExecution(
        outputs=None,
        timing_ms=None,
        dispatch_token=21,
        error_type="DeviceError",
        error_message="failed",
    )
    executor = OwnedExecutionExecutor(
        returned=[failure],
        acknowledge_error=RuntimeError("ack failed"),
        shutdown_error=RuntimeError("shutdown failed"),
    )
    evaluator = FakeEvaluator()
    engine = InferenceEngine(
        FakeLoader(),
        FakeRuntime(),
        evaluator,
        runtime_executor=executor,
    )
    executor.engine = engine
    error_type = _runtime_execution_error_type()

    with pytest.raises(error_type) as raised:
        engine.run_e2e(batch_size=1)

    assert raised.value.error_type == "DeviceError"
    assert raised.value.dispatch_token == 21
    _assert_exact_ack(executor)
    assert executor.shutdown_calls == [0.0]
    assert len(executor.inflight) == 1
    assert engine.completion.snapshot_outstanding() == ()
    assert evaluator.compute_calls == 0


def test_benchmark_runner_propagates_runtime_execution_error_and_stops_monitor():
    events = []
    monitor = RecordingMonitor(events, {"hw_samples": 1})
    failure = RuntimeExecution(
        outputs=None,
        timing_ms=None,
        dispatch_token=31,
        error_type="DeviceError",
        error_message="failed",
    )
    executor = OwnedExecutionExecutor(returned=[failure])
    evaluator = FakeEvaluator()
    runner = BenchmarkRunner(
        FakeLoader(),
        FakeRuntime(),
        evaluator,
        monitor=monitor,
    )
    runner.engine.runtime_executor = executor
    executor.engine = runner.engine
    error_type = _runtime_execution_error_type()

    with pytest.raises(error_type) as raised:
        runner.run(warmup_runs=0, batch_size=1)

    assert raised.value.error_type == "DeviceError"
    assert events == ["start", "stop"]
    assert monitor.summary_calls == 0
    assert evaluator.compute_calls == 0


def test_injected_executor_is_shared_with_pipeline_invoke():
    executor = RecordingExecutor()
    engine = InferenceEngine(
        FakeLoader(),
        FakeRuntime(),
        FakeEvaluator(),
        runtime_executor=executor,
    )
    executor.engine = engine

    invocation = engine.pipeline.invoke(
        {"input": np.array([[1.0, 2.0]], dtype=np.float32)}
    )

    assert engine.runtime_executor is executor
    assert engine.pipeline._compat_executor is executor
    assert executor.executions == [invocation]


def test_evaluator_failure_is_acknowledged_after_terminal_commit():
    primary = ValueError("quality failed")
    executor = RecordingExecutor()
    engine = InferenceEngine(
        FakeLoader(),
        FakeRuntime(),
        FailingEvaluator(primary),
        runtime_executor=executor,
    )
    executor.engine = engine

    with pytest.raises(ValueError) as raised:
        engine.run_e2e(batch_size=2)

    assert raised.value is primary
    assert len(executor.acknowledged) == 1
    assert executor.acknowledged[0][1] == ()


@pytest.mark.parametrize("failure_site", ["evaluator", "decoder"])
def test_callback_error_precedes_acknowledge_error(failure_site):
    primary = ValueError(f"{failure_site} failed")
    acknowledge_error = RuntimeError("acknowledge failed")
    executor = FailingAcknowledgeExecutor(acknowledge_error)
    evaluator = (
        FailingEvaluator(primary)
        if failure_site == "evaluator"
        else FakeEvaluator()
    )
    decoder = FailingDecoder(primary) if failure_site == "decoder" else None
    engine = InferenceEngine(
        FakeLoader(),
        FakeRuntime(),
        evaluator,
        decoder=decoder,
        runtime_executor=executor,
    )
    executor.engine = engine

    with pytest.raises(Exception) as raised:
        engine.run_e2e(batch_size=2)

    assert raised.value is primary
    assert len(executor.acknowledged) == 1
    assert executor.acknowledged[0][1] == ()
    assert engine.completion.snapshot_outstanding() == ()


def test_static_batch_is_one_atomic_request_with_actual_sample_count():
    loader = StaticLoader()
    runtime = FakeRuntime()
    evaluator = FakeEvaluator()
    traces = []
    engine = InferenceEngine(
        loader,
        runtime,
        evaluator,
        trace_callback=traces.append,
    )
    engine.warmup(runs=1, batch_size=1)

    result = engine.run_e2e(batch_size=1)

    assert result == {"pairs": [(3.0, 3), (7.0, 7)], "Total Samples": 2}
    assert len(traces) == 1
    assert traces[0].request_id == 0
    assert traces[0].sample_count == 2
    assert traces[0].batch_size == 2


def test_e2e_engine_rejects_a_second_run():
    engine = InferenceEngine(FakeLoader(), FakeRuntime(), FakeEvaluator())
    engine.run_e2e(batch_size=2)

    with pytest.raises(
        RuntimeError,
        match=r"run_e2e\(\) may only be called once",
    ):
        engine.run_e2e(batch_size=2)


def test_e2e_engine_honors_max_steps():
    engine = InferenceEngine(FakeLoader(), FakeRuntime(), FakeEvaluator())

    result = engine.run_e2e(batch_size=1, max_steps=1)

    assert result["Total Samples"] == 1


def test_zero_warmup_does_not_consume_or_invoke_runtime():
    loader = FakeLoader()
    runtime = FakeRuntime()
    engine = InferenceEngine(loader, runtime, FakeEvaluator())

    engine.warmup(runs=0, batch_size=1)

    assert loader.current_idx == 0
    assert runtime.warmup_invocations == []
