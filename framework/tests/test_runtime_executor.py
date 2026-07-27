import numpy as np

from core.generation_result import GenerationResult
from core.runtime_executor import (
    BlockingRuntimeExecutor,
    NativeAsyncOutcome,
    NativeAsyncRuntimeExecutor,
    create_async_runtime_executor,
)


class ArrayRuntime:
    def __init__(self):
        self.calls = []

    def run(self, inputs):
        self.calls.append(inputs)
        return {"output": inputs["input"] + 1}


class GenerationRuntime:
    def generate(self, inputs, max_new_tokens, stop_token_ids):
        return GenerationResult(
            generated_ids=np.array([[4, 5]], dtype=np.int64),
            generated_lengths=np.array([2], dtype=np.int64),
            total_ms=3.0,
            ttft_ms=1.0,
            tpot_ms=2.0,
            num_tokens=2,
            timing_mode="native",
            uses_kv_cache=True,
            timing_source="measured",
        )


class TotalOnlyGenerationRuntime:
    def generate(self, inputs, max_new_tokens, stop_token_ids):
        return GenerationResult(
            generated_ids=np.array([[7]], dtype=np.int64),
            generated_lengths=np.array([1], dtype=np.int64),
            total_ms=4.0,
            ttft_ms=None,
            tpot_ms=None,
            num_tokens=1,
            timing_mode="kv_cache",
            uses_kv_cache=True,
            timing_source="wall_clock_total_only",
        )


def test_blocking_executor_runs_array_runtime_and_reports_latency():
    runtime = ArrayRuntime()
    executor = BlockingRuntimeExecutor(runtime, is_llm=False)
    inputs = {"input": np.array([[1.0]], dtype=np.float32)}
    execution = executor.execute(inputs)
    np.testing.assert_array_equal(execution.outputs["output"], [[2.0]])
    assert execution.timing_ms >= 0.0
    assert execution.generated_tokens == 0
    assert execution.dispatch_token is None
    assert execution.error_type is None
    assert runtime.calls == [inputs]
    executor.acknowledge(execution)
    assert executor.shutdown(timeout=0.0) is True


def test_blocking_executor_preserves_generation_result_mapping():
    runtime = GenerationRuntime()
    executor = BlockingRuntimeExecutor(
        runtime,
        is_llm=True,
        max_new_tokens=17,
        stop_token_ids=[2, 3],
    )
    execution = executor.execute(
        {"input_ids": np.array([[1, 2]], dtype=np.int64)}
    )
    np.testing.assert_array_equal(execution.outputs["generated_ids"], [[4, 5]])
    np.testing.assert_array_equal(execution.outputs["generated_lengths"], [2])
    assert execution.generated_tokens == 2
    assert execution.timing_ms["total_ms"] == 3.0
    assert execution.timing_ms["timing_source"] == "measured"


def test_blocking_executor_omits_unavailable_generation_timings():
    executor = BlockingRuntimeExecutor(TotalOnlyGenerationRuntime(), is_llm=True)

    execution = executor.execute(
        {"input_ids": np.array([[1, 2]], dtype=np.int64)}
    )

    assert execution.timing_ms["total_ms"] == 4.0
    assert execution.timing_ms["timing_source"] == "wall_clock_total_only"
    assert "ttft_ms" not in execution.timing_ms
    assert "tpot_ms" not in execution.timing_ms


class OptInNativeRuntime:
    def supports_native_async(self):
        return True

    def native_async_max_inflight(self):
        return 8

    def native_async_completion_timeout_sec(self):
        return 2.5

    def submit_async(self, inputs, callback):
        callback(
            NativeAsyncOutcome(
                outputs={"output": np.array(inputs["input"], copy=True)},
                timing_ms=1.0,
            )
        )
        return 17


def test_async_executor_factory_selects_native_runtime_with_bounded_workers():
    """Catches silently routing an opted-in vendor runtime through blocking run()."""
    executor = create_async_runtime_executor(
        OptInNativeRuntime(),
        worker_count=3,
    )

    assert isinstance(executor, NativeAsyncRuntimeExecutor)
    assert executor.max_inflight == 3
    assert executor.completion_timeout_sec == 2.5
    execution = executor.execute({"input": np.asarray([[4]], dtype=np.float32)})
    np.testing.assert_array_equal(execution.outputs["output"], [[4]])
    assert execution.vendor_job_id == 17
    executor.acknowledge(execution)
    assert executor.shutdown(timeout=0.0) is True


def test_async_executor_factory_keeps_non_native_runtime_on_default_path():
    """Catches requiring new optional methods from every existing runtime."""
    assert create_async_runtime_executor(ArrayRuntime(), worker_count=1) is None
