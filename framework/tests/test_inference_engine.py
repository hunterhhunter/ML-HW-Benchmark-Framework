import threading

import numpy as np
import pytest

from core.async_inference.types import AsyncInferenceConfig, TerminalStatus
from core.benchmarkrunner import BenchmarkRunner
from core.inference_engine import InferenceEngine
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

    def add_batch(self, outputs, labels, timing_ms):
        del timing_ms
        predictions = outputs["output"].reshape(-1).tolist()
        self.rows.extend(zip(predictions, labels))

    def compute(self):
        return {
            "pairs": list(self.rows),
            "Total Samples": len(self.rows),
        }


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


def test_inference_engine_allows_only_one_execution_mode():
    engine = InferenceEngine(FakeLoader(), FakeRuntime(), FakeEvaluator())
    engine.run_e2e(batch_size=2)

    with pytest.raises(RuntimeError, match="may only be run once"):
        engine.run_async(
            AsyncInferenceConfig(min_samples=1),
            warmup_runs=0,
        )


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
