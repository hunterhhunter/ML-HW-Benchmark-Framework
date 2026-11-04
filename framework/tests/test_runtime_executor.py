import dataclasses
from types import SimpleNamespace

import numpy as np

from core.generation_result import GenerationResult
from core.runtime_executor import (
    BlockingRuntimeExecutor,
    GenerationObservation,
    GenerationOutputEvent,
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


class LegacyGenerationRuntime:
    def generate(self, inputs, max_new_tokens, stop_token_ids):
        return SimpleNamespace(
            generated_ids=np.array([[9]], dtype=np.int64),
            generated_lengths=np.array([1], dtype=np.int64),
            total_ms=2.0,
            ttft_ms=1.0,
            tpot_ms=1.0,
            num_tokens=1,
            timing_mode="reported",
            uses_kv_cache=False,
            timing_source="test",
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


def test_blocking_executor_accepts_legacy_generation_result_without_observation():
    execution = BlockingRuntimeExecutor(
        LegacyGenerationRuntime(), is_llm=True
    ).execute({})

    assert execution.generation_observation is None


class ObservedGenerationRuntime(GenerationRuntime):
    def __init__(self, observation):
        self.observation = observation

    def generate(self, inputs, max_new_tokens, stop_token_ids):
        result = super().generate(inputs, max_new_tokens, stop_token_ids)
        return dataclasses.replace(
            result,
            generation_observation=self.observation,
        )


def test_blocking_executor_forwards_generation_observation():
    observation = GenerationObservation(
        backend_submitted_ns=100,
        events=(GenerationOutputEvent(observed_ns=150, cumulative_tokens=1),),
        source="mobilint_transformers_streamer",
    )
    runtime = ObservedGenerationRuntime(observation)
    execution = BlockingRuntimeExecutor(runtime, is_llm=True).execute({})
    assert execution.generation_observation == observation


def test_generation_result_defaults_to_no_observation():
    result = GenerationResult(
        generated_ids=np.array([1], dtype=np.int64),
        ttft_ms=1.0,
        tpot_ms=None,
        total_ms=1.0,
        num_tokens=1,
    )
    assert result.generation_observation is None
