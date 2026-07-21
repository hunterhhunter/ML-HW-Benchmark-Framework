import asyncio
import sys
import threading
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest

from core.runtime_executor import (
    GenerationObservation,
    NativeAsyncRuntimeExecutor,
)
from runtimes.furiosa_llm_rt import FuriosaNativeBackend


class _SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Completion:
    def __init__(self, token_ids):
        self.token_ids = list(token_ids)


class _Output:
    def __init__(self, token_ids):
        self.outputs = [_Completion(token_ids)]


def _runtime():
    return SimpleNamespace(
        _llm=object(),
        _sampling_params_cls=_SamplingParams,
        _trim_prompt_tokens=lambda inputs: [
            [int(token) for token in np.asarray(inputs["input_ids"]).reshape(-1)]
        ],
        _normalize_outputs=lambda output: (
            np.asarray([output.outputs[0].token_ids], dtype=np.int64),
            np.asarray([len(output.outputs[0].token_ids)], dtype=np.int64),
        ),
    )


def _install_async_sdk(monkeypatch, generate, abort=None):
    state = {
        "from_llm_threads": [],
        "calls": [],
        "aborts": [],
        "abort_threads": [],
    }

    class AsyncLLMEngine:
        @classmethod
        def from_llm(cls, llm):
            state["from_llm_threads"].append(threading.current_thread().name)
            return cls()

        def generate(self, prompt, sampling_params, request_id):
            state["calls"].append((prompt, sampling_params, request_id))
            return generate(prompt, sampling_params, request_id)

        async def abort(self, request_id):
            state["aborts"].append(request_id)
            state["abort_threads"].append(threading.current_thread().name)
            if abort is not None:
                await abort(request_id)

    sdk = types.ModuleType("furiosa_llm")
    sdk.AsyncLLMEngine = AsyncLLMEngine
    monkeypatch.setitem(sys.modules, "furiosa_llm", sdk)
    return state


def test_furiosa_backend_uses_loop_thread_and_measures_stream_events(monkeypatch):
    async def generate(prompt, sampling_params, request_id):
        await asyncio.sleep(0.01)
        yield _Output([])
        await asyncio.sleep(0.02)
        yield _Output([31])
        await asyncio.sleep(0.02)
        yield _Output([31, 32, 33])

    state = _install_async_sdk(monkeypatch, generate)
    backend = FuriosaNativeBackend(
        _runtime(), max_new_tokens=4, stop_token_ids=[2, 3]
    )
    completed = []
    done = threading.Event()

    request_id = backend.submit_async(
        {"input_ids": np.array([[11, 12]], dtype=np.int64)},
        lambda outcome: (completed.append(outcome), done.set()),
    )

    assert done.wait(timeout=1.0)
    outcome = completed[0]
    assert state["from_llm_threads"] == ["furiosa-native-loop"]
    prompt, sampling_params, submitted_id = state["calls"][0]
    assert prompt == {"prompt_token_ids": [11, 12]}
    assert submitted_id == request_id
    assert sampling_params.kwargs == {
        "max_tokens": 4,
        "temperature": 0.0,
        "stop_token_ids": [2, 3],
    }
    np.testing.assert_array_equal(outcome.outputs["generated_ids"], [[31, 32, 33]])
    np.testing.assert_array_equal(outcome.outputs["generated_lengths"], [3])
    assert outcome.generated_tokens == 3
    assert outcome.timing_ms["ttft_ms"] >= 20.0
    assert outcome.timing_ms["tpot_ms"] >= 5.0
    assert outcome.timing_ms["total_ms"] >= outcome.timing_ms["ttft_ms"]
    assert outcome.timing_ms["timing_source"] == "furiosa_async_python_stream"
    assert type(outcome.generation_observation) is GenerationObservation
    assert outcome.generation_observation.source == "furiosa_async_python_stream"
    assert [
        event.cumulative_tokens
        for event in outcome.generation_observation.events
    ] == [1, 3]
    assert outcome.generation_observation.backend_submitted_ns <= (
        outcome.generation_observation.events[0].observed_ns
    )
    assert backend.shutdown(timeout=1.0) is True


def test_furiosa_backend_records_each_single_token_stream_increment(monkeypatch):
    async def generate(prompt, sampling_params, request_id):
        del prompt, sampling_params, request_id
        yield _Output([])
        yield _Output([31])
        yield _Output([31, 32])
        yield _Output([31, 32, 33])

    _install_async_sdk(monkeypatch, generate)
    backend = FuriosaNativeBackend(_runtime(), max_new_tokens=3)
    completed = []
    done = threading.Event()

    backend.submit_async(
        {"input_ids": np.array([[11]], dtype=np.int64)},
        lambda outcome: (completed.append(outcome), done.set()),
    )

    assert done.wait(timeout=1.0)
    observation = completed[0].generation_observation
    assert [event.cumulative_tokens for event in observation.events] == [1, 2, 3]
    assert [event.observed_ns for event in observation.events] == sorted(
        event.observed_ns for event in observation.events
    )
    assert backend.shutdown(timeout=1.0) is True


def test_furiosa_backend_ignores_repeated_cumulative_outputs(monkeypatch):
    async def generate(prompt, sampling_params, request_id):
        del prompt, sampling_params, request_id
        yield _Output([31])
        yield _Output([31])
        yield _Output([31, 32])

    _install_async_sdk(monkeypatch, generate)
    backend = FuriosaNativeBackend(_runtime(), max_new_tokens=2)
    completed = []
    done = threading.Event()

    backend.submit_async(
        {"input_ids": np.array([[11]], dtype=np.int64)},
        lambda outcome: (completed.append(outcome), done.set()),
    )

    assert done.wait(timeout=1.0)
    observation = completed[0].generation_observation
    assert [event.cumulative_tokens for event in observation.events] == [1, 2]
    assert backend.shutdown(timeout=1.0) is True


def test_furiosa_backend_rejects_decreasing_cumulative_output(monkeypatch):
    async def generate(prompt, sampling_params, request_id):
        del prompt, sampling_params, request_id
        yield _Output([31, 32])
        yield _Output([31])

    _install_async_sdk(monkeypatch, generate)
    backend = FuriosaNativeBackend(_runtime(), max_new_tokens=2)
    completed = []
    done = threading.Event()
    backend.submit_async(
        {"input_ids": np.array([[11]], dtype=np.int64)},
        lambda outcome: (completed.append(outcome), done.set()),
    )

    assert done.wait(timeout=1.0)
    assert completed[0].error_type == "RuntimeError"
    assert completed[0].outputs is None
    assert backend.shutdown(timeout=1.0) is True


@pytest.mark.parametrize(
    ("stream_token_ids", "expected_ttft", "expected_tpot"),
    [
        ([], None, None),
        ([9], "present", None),
        ([9, 10], "present", "present"),
    ],
)
def test_furiosa_backend_does_not_invent_undefined_tpot(
    monkeypatch,
    stream_token_ids,
    expected_ttft,
    expected_tpot,
):
    async def generate(prompt, sampling_params, request_id):
        del prompt, sampling_params, request_id
        yield _Output(stream_token_ids)

    _install_async_sdk(monkeypatch, generate)
    backend = FuriosaNativeBackend(
        _runtime(), max_new_tokens=max(1, len(stream_token_ids))
    )
    completed = []
    done = threading.Event()
    backend.submit_async(
        {"input_ids": np.array([[11]], dtype=np.int64)},
        lambda outcome: (completed.append(outcome), done.set()),
    )

    assert done.wait(timeout=1.0)
    timing = completed[0].timing_ms
    if expected_ttft == "present":
        assert timing["ttft_ms"] >= 0.0
    else:
        assert timing.get("ttft_ms") is None
    if expected_tpot == "present":
        assert timing["tpot_ms"] >= 0.0
    else:
        assert timing.get("tpot_ms") is None
    assert backend.shutdown(timeout=1.0) is True


def test_furiosa_backend_completes_requests_out_of_order_and_once(monkeypatch):
    async def generate(prompt, sampling_params, request_id):
        token = prompt["prompt_token_ids"][0]
        await asyncio.sleep(0.04 if token == 1 else 0.005)
        yield _Output([token * 10])
        yield _Output([token * 10, token * 10 + 1])

    _install_async_sdk(monkeypatch, generate)
    backend = FuriosaNativeBackend(_runtime(), max_new_tokens=2)
    completed = []
    done = threading.Event()

    def callback(name):
        def collect(outcome):
            completed.append((name, outcome))
            if len(completed) == 2:
                done.set()

        return collect

    first_id = backend.submit_async(
        {"input_ids": np.array([[1]], dtype=np.int64)}, callback("first")
    )
    second_id = backend.submit_async(
        {"input_ids": np.array([[2]], dtype=np.int64)}, callback("second")
    )

    assert first_id != second_id
    assert done.wait(timeout=1.0)
    assert [name for name, _ in completed] == ["second", "first"]
    assert len(completed) == 2
    assert backend.shutdown(timeout=1.0) is True


def test_furiosa_backend_normalizes_sdk_exception_without_sensitive_data(monkeypatch):
    async def generate(prompt, sampling_params, request_id):
        raise RuntimeError("secret prompt tokens=[11, 12]")
        yield  # pragma: no cover

    state = _install_async_sdk(monkeypatch, generate)
    backend = FuriosaNativeBackend(_runtime(), max_new_tokens=2)
    completed = []
    done = threading.Event()
    backend.submit_async(
        {"input_ids": np.array([[11, 12]], dtype=np.int64)},
        lambda outcome: (completed.append(outcome), done.set()),
    )

    assert done.wait(timeout=1.0)
    outcome = completed[0]
    assert outcome.error_type == "RuntimeError"
    assert "secret" not in outcome.error_message
    assert "11" not in outcome.error_message
    assert state["aborts"] == [state["calls"][0][2]]
    assert state["abort_threads"] == ["furiosa-native-loop"]
    assert backend.shutdown(timeout=1.0) is True


def test_shutdown_aborts_pending_request_before_stopping_loop(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    async def generate(prompt, sampling_params, request_id):
        started.set()
        while not release.is_set():
            await asyncio.sleep(0.005)
        if False:
            yield _Output([])

    async def abort(request_id):
        release.set()

    state = _install_async_sdk(monkeypatch, generate, abort=abort)
    backend = FuriosaNativeBackend(_runtime(), max_new_tokens=2)
    completed = []
    request_id = backend.submit_async(
        {"input_ids": np.array([[7]], dtype=np.int64)},
        completed.append,
    )

    assert started.wait(timeout=1.0)
    assert backend.shutdown(timeout=1.0) is True

    assert state["aborts"] == [request_id]
    assert state["abort_threads"] == ["furiosa-native-loop"]
    assert not backend._thread.is_alive()
    assert len(completed) == 1
    assert completed[0].error_type == "FuriosaAsyncShutdown"


def test_furiosa_backend_late_completion_after_executor_timeout_is_safe(monkeypatch):
    async def generate(prompt, sampling_params, request_id):
        await asyncio.sleep(0.05)
        yield _Output([9])

    _install_async_sdk(monkeypatch, generate)
    backend = FuriosaNativeBackend(_runtime(), max_new_tokens=1)
    executor = NativeAsyncRuntimeExecutor(
        backend, max_inflight=1, completion_timeout_sec=0.005
    )

    execution = executor.execute(
        {"input_ids": np.array([[1]], dtype=np.int64)}
    )
    assert execution.error_type == "NativeAsyncTimeout"
    executor.acknowledge(execution)
    deadline = time.monotonic() + 1.0
    while executor.snapshot().late_callbacks == 0 and time.monotonic() < deadline:
        time.sleep(0.005)

    assert executor.snapshot().late_callbacks == 1
    assert executor.snapshot().inflight == 0
    assert backend.shutdown(timeout=1.0) is True
