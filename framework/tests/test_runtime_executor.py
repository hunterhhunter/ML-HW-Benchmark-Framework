import numpy as np

from core.generation_result import GenerationResult
from core.runtime_executor import BlockingRuntimeExecutor


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
