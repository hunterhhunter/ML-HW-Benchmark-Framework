import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

from core.compiled_model import CompiledModel
from core.async_inference.types import AsyncInferenceConfig, RunStatus
from core.inference_engine import InferenceEngine
from core.model_spec import Model_Spec, Task
from core.runtime_executor import NativeAsyncRuntimeExecutor
from runtimes.deepx_rt import DeepXRuntime


class FakeDXRTState:
    def __init__(self, input_names=("input",), output_names=("output",)):
        self.callback = None
        self.sync_calls = 0
        self.async_calls = []
        self.async_methods = []
        self.engines = []
        self.auto_complete = True
        self.inline_outputs = None
        self.submit_error = None
        self.input_names = list(input_names)
        self.output_names = list(output_names)
        self.events = []
        self.submit_entered = None
        self.submit_release = None
        self.unregister_hook = None

    def complete(self, token, outputs=None):
        if outputs is None:
            outputs = [np.asarray([[7.0, 8.0]], dtype=np.float32)]
        self.callback(outputs, token)


def _install_fake_dx_engine(
    monkeypatch,
    input_names=("input",),
    output_names=("output",),
):
    state = FakeDXRTState(input_names=input_names, output_names=output_names)

    class FakeBoundOption:
        NPU_ALL = "NPU_ALL"

    class FakeInferenceOption:
        BOUND_OPTION = FakeBoundOption

        def __init__(self):
            self.devices = None
            self.bound_option = None
            self.buffer_count = None

        def set_devices(self, devices):
            self.devices = list(devices)

        def set_bound_option(self, value):
            self.bound_option = value

        def set_buffer_count(self, value):
            self.buffer_count = value

    class FakeInferenceEngine:
        def __init__(self, model_path, option=None):
            self.model_path = model_path
            self.option = option
            self.disposed = False
            state.engines.append(self)

        def get_input_tensor_names(self):
            return list(state.input_names)

        def get_output_tensor_names(self):
            return list(state.output_names)

        def get_input_tensors_info(self):
            return [
                {
                    "name": name,
                    "shape": [1, 3, 4, 4],
                    "dtype": np.dtype("float32"),
                    "elem_size": 4,
                }
                for name in state.input_names
            ]

        def get_output_tensors_info(self):
            return [
                {
                    "name": name,
                    "shape": [1, 2],
                    "dtype": np.dtype("float32"),
                    "elem_size": 4,
                }
                for name in state.output_names
            ]

        def register_callback(self, callback):
            previous = state.callback
            if callback is None and state.unregister_hook is not None:
                state.unregister_hook(previous)
            state.callback = callback
            state.events.append("register" if callback is not None else "unregister")

        def run(self, input_data):
            state.sync_calls += 1
            outputs = [np.asarray([[3.0, 4.0]], dtype=np.float32)]
            if state.callback is not None:
                state.callback(outputs, None)
            return outputs

        def run_async(self, input_data, user_arg=None, output_buffer=None):
            if state.submit_error is not None:
                raise state.submit_error
            state.async_methods.append(("run_async", input_data, user_arg))
            state.async_calls.append((input_data, user_arg, output_buffer))
            job_id = len(state.async_calls)
            if state.submit_entered is not None:
                state.submit_entered.set()
            if state.submit_release is not None:
                state.submit_release.wait(timeout=2.0)
            if state.auto_complete:
                state.complete(user_arg, state.inline_outputs)
            return job_id

        def run_async_multi_input(
            self, input_data, user_arg=None, output_buffer=None
        ):
            if state.submit_error is not None:
                raise state.submit_error
            state.async_methods.append(
                ("run_async_multi_input", input_data, user_arg)
            )
            state.async_calls.append((input_data, user_arg, output_buffer))
            job_id = len(state.async_calls)
            if state.auto_complete:
                state.complete(user_arg, state.inline_outputs)
            return job_id

        def dispose(self):
            self.disposed = True
            state.events.append("dispose")

    fake_module = types.ModuleType("dx_engine")
    fake_module.__version__ = "3.3.2-test"
    fake_module.InferenceOption = FakeInferenceOption
    fake_module.InferenceEngine = FakeInferenceEngine
    monkeypatch.setitem(sys.modules, "dx_engine", fake_module)
    return state


def _compiled_model(
    tmp_path: Path,
    input_names=("input",),
    output_names=("output",),
) -> CompiledModel:
    artifact = tmp_path / "model.dxnn"
    artifact.write_bytes(b"DXNN-test")
    spec = Model_Spec(
        name="deepx-test",
        task=Task.IMAGE_CLASSIFICATION,
        input_shapes={name: (1, 3, 4, 4) for name in input_names},
        input_dtype={name: "float32" for name in input_names},
        output_shapes={name: (1, 2) for name in output_names},
        model_paths={},
    )
    return CompiledModel(
        spec=spec,
        backend_name="deepx",
        artifact_path=artifact,
    )


def test_native_async_warmup_never_calls_sync_run(monkeypatch, tmp_path):
    """Catches sync warmup contaminating an already registered DX-RT callback."""
    state = _install_fake_dx_engine(monkeypatch)
    runtime = DeepXRuntime(buffer_count=6)
    runtime.load(_compiled_model(tmp_path))
    backend = runtime.create_native_backend()
    inputs = {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)}

    runtime.warmup(inputs, num_runs=2)
    outcomes = []
    backend.submit_async(inputs, outcomes.append)

    assert state.sync_calls == 0
    assert len(state.async_calls) == 3
    assert len(outcomes) == 1
    assert outcomes[0].error_type is None
    np.testing.assert_array_equal(
        outcomes[0].outputs["output"],
        np.asarray([[7.0, 8.0]], dtype=np.float32),
    )
    runtime.unload()


def test_callback_owned_nested_arrays_are_copied_before_sdk_reuse(
    monkeypatch, tmp_path
):
    state = _install_fake_dx_engine(monkeypatch)
    state.auto_complete = False
    runtime = DeepXRuntime(buffer_count=6)
    runtime.load(_compiled_model(tmp_path))
    backend = runtime.create_native_backend()
    inputs = {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)}
    outcomes = []

    token = backend.submit_async(inputs, outcomes.append)
    nested = np.asarray([[5.0, 6.0]], dtype=np.float32)
    callback_output = np.empty((1,), dtype=object)
    callback_output[0] = nested
    state.complete(token, {"output": callback_output})
    nested.fill(-1)

    np.testing.assert_array_equal(
        outcomes[0].outputs["output"][0],
        np.asarray([[5.0, 6.0]], dtype=np.float32),
    )
    runtime.unload()


def test_callbacks_match_out_of_order_tokens(monkeypatch, tmp_path):
    state = _install_fake_dx_engine(monkeypatch)
    state.auto_complete = False
    runtime = DeepXRuntime(buffer_count=6)
    runtime.load(_compiled_model(tmp_path))
    backend = runtime.create_native_backend()
    completion_order = []

    first = backend.submit_async(
        {"input": np.full((1, 3, 4, 4), 1, dtype=np.float32)},
        lambda outcome: completion_order.append((1, outcome)),
    )
    second = backend.submit_async(
        {"input": np.full((1, 3, 4, 4), 2, dtype=np.float32)},
        lambda outcome: completion_order.append((2, outcome)),
    )
    state.complete(second, [np.asarray([[2.0, 2.0]], dtype=np.float32)])
    state.complete(first, [np.asarray([[1.0, 1.0]], dtype=np.float32)])

    assert [item[0] for item in completion_order] == [2, 1]
    assert all(item[1].error_type is None for item in completion_order)
    runtime.unload()


def test_inline_callback_before_job_id_return_completes_once(
    monkeypatch, tmp_path
):
    state = _install_fake_dx_engine(monkeypatch)
    state.inline_outputs = [np.asarray([[9.0, 10.0]], dtype=np.float32)]
    runtime = DeepXRuntime(buffer_count=6)
    runtime.load(_compiled_model(tmp_path))
    backend = runtime.create_native_backend()
    inputs = {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)}
    outcomes = []

    assert backend.submit_async(inputs, outcomes.append) == 1
    assert len(outcomes) == 1
    np.testing.assert_array_equal(outcomes[0].outputs["output"], [[9.0, 10.0]])
    runtime.unload()


def test_named_multi_input_uses_named_async_api(monkeypatch, tmp_path):
    input_names = ("left", "right")
    state = _install_fake_dx_engine(monkeypatch, input_names=input_names)
    runtime = DeepXRuntime(buffer_count=6)
    runtime.load(_compiled_model(tmp_path, input_names=input_names))
    backend = runtime.create_native_backend()
    inputs = {
        "right": np.full((1, 3, 4, 4), 2, dtype=np.float32),
        "left": np.full((1, 3, 4, 4), 1, dtype=np.float32),
    }
    outcomes = []

    backend.submit_async(inputs, outcomes.append)

    method, payload, _ = state.async_methods[0]
    assert method == "run_async_multi_input"
    assert list(payload) == ["left", "right"]
    assert outcomes[0].error_type is None
    runtime.unload()


def test_unknown_callback_token_fails_pending_job_safely(
    monkeypatch, tmp_path
):
    state = _install_fake_dx_engine(monkeypatch)
    state.auto_complete = False
    runtime = DeepXRuntime(buffer_count=6)
    runtime.load(_compiled_model(tmp_path))
    backend = runtime.create_native_backend()
    inputs = {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)}
    outcomes = []
    backend.submit_async(inputs, outcomes.append)

    state.complete(999, [np.asarray([[1.0, 2.0]], dtype=np.float32)])

    assert len(outcomes) == 1
    assert outcomes[0].error_type == "DeepXAsyncProtocolError"
    runtime.unload()


@pytest.mark.parametrize(
    "outputs",
    [
        None,
        [],
        {},
        [[np.asarray([[1.0, 2.0]], dtype=np.float32)]],
        [
            np.asarray([[1.0, 2.0]], dtype=np.float32),
            np.asarray([[3.0, 4.0]], dtype=np.float32),
        ],
        {"unexpected": np.asarray([[1.0, 2.0]], dtype=np.float32)},
    ],
    ids=["none", "empty-list", "empty-dict", "batched", "count", "name"],
)
def test_invalid_callback_outputs_publish_completion_error(
    monkeypatch, tmp_path, outputs
):
    state = _install_fake_dx_engine(monkeypatch)
    state.auto_complete = False
    runtime = DeepXRuntime(buffer_count=6)
    runtime.load(_compiled_model(tmp_path))
    backend = runtime.create_native_backend()
    inputs = {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)}
    outcomes = []
    token = backend.submit_async(inputs, outcomes.append)

    state.callback(outputs, token)

    assert len(outcomes) == 1
    assert outcomes[0].error_type == "DeepXAsyncCompletionError"
    runtime.unload()


def test_submission_failure_retires_job_without_callback(
    monkeypatch, tmp_path
):
    state = _install_fake_dx_engine(monkeypatch)
    state.submit_error = RuntimeError("DX-RT rejected input")
    runtime = DeepXRuntime(buffer_count=6)
    runtime.load(_compiled_model(tmp_path))
    backend = runtime.create_native_backend()
    inputs = {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)}
    outcomes = []

    try:
        backend.submit_async(inputs, outcomes.append)
    except RuntimeError as exc:
        assert str(exc) == "DX-RT rejected input"
    else:
        raise AssertionError("submission should fail")

    assert outcomes == []
    assert backend.shutdown(timeout=0.01) is True
    runtime.unload()


def test_warmup_timeout_keeps_physical_job_tracked(monkeypatch, tmp_path):
    state = _install_fake_dx_engine(monkeypatch)
    state.auto_complete = False
    runtime = DeepXRuntime(buffer_count=6)
    runtime.load(_compiled_model(tmp_path))
    backend = runtime.create_native_backend()
    inputs = {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)}

    with pytest.raises(TimeoutError, match="warmup timed out"):
        backend.run_warmup_blocking(inputs, timeout=0.001)

    assert backend.shutdown(timeout=0.001) is False
    state.complete(1)
    assert backend.shutdown(timeout=1.0) is True
    runtime.unload()


def test_unload_waits_for_consumer_callback_to_return(monkeypatch, tmp_path):
    state = _install_fake_dx_engine(monkeypatch)
    state.auto_complete = False
    runtime = DeepXRuntime(
        buffer_count=6,
        async_completion_timeout_sec=0.001,
    )
    runtime.load(_compiled_model(tmp_path))
    backend = runtime.create_native_backend()
    inputs = {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)}
    callback_entered = threading.Event()
    callback_release = threading.Event()

    def blocking_callback(outcome):
        callback_entered.set()
        callback_release.wait(timeout=2.0)

    token = backend.submit_async(inputs, blocking_callback)
    callback_thread = threading.Thread(target=state.complete, args=(token,))
    callback_thread.start()
    assert callback_entered.wait(timeout=1.0)

    with pytest.raises(RuntimeError, match="did not drain"):
        runtime.unload()
    assert state.engines[0].disposed is False

    callback_release.set()
    callback_thread.join(timeout=1.0)
    assert not callback_thread.is_alive()
    runtime.unload()
    assert state.events[-2:] == ["unregister", "dispose"]


def test_unload_from_native_callback_is_rejected(monkeypatch, tmp_path):
    state = _install_fake_dx_engine(monkeypatch)
    runtime = DeepXRuntime(buffer_count=6)
    runtime.load(_compiled_model(tmp_path))
    backend = runtime.create_native_backend()
    inputs = {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)}
    unload_errors = []

    def unload_inside_callback(outcome):
        try:
            runtime.unload()
        except RuntimeError as exc:
            unload_errors.append(str(exc))

    backend.submit_async(inputs, unload_inside_callback)

    assert unload_errors == [
        "Cannot shut down DeepX native backend from its callback."
    ]
    assert state.engines[0].disposed is False
    runtime.unload()


def test_shutdown_tracks_run_async_call_until_it_returns(monkeypatch, tmp_path):
    state = _install_fake_dx_engine(monkeypatch)
    state.submit_entered = threading.Event()
    state.submit_release = threading.Event()
    runtime = DeepXRuntime(buffer_count=6)
    runtime.load(_compiled_model(tmp_path))
    backend = runtime.create_native_backend()
    inputs = {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)}
    submitter = threading.Thread(
        target=backend.submit_async,
        args=(inputs, lambda outcome: None),
    )
    submitter.start()
    assert state.submit_entered.wait(timeout=1.0)

    assert backend.shutdown(timeout=0.001) is False
    state.submit_release.set()
    submitter.join(timeout=1.0)
    assert not submitter.is_alive()
    assert backend.shutdown(timeout=1.0) is True
    runtime.unload()


def test_unload_drains_callback_dispatched_during_unregister(
    monkeypatch, tmp_path
):
    state = _install_fake_dx_engine(monkeypatch)
    runtime = DeepXRuntime(buffer_count=6)
    runtime.load(_compiled_model(tmp_path))
    backend = runtime.create_native_backend()
    callback_entered = threading.Event()
    callback_release = threading.Event()
    callback_threads = []
    original_retired_check = backend._is_retired_token

    def blocking_retired_check(token):
        callback_entered.set()
        callback_release.wait(timeout=2.0)
        return original_retired_check(token)

    backend._is_retired_token = blocking_retired_check

    def dispatch_during_unregister(previous_callback):
        thread = threading.Thread(
            target=previous_callback,
            args=([np.asarray([[1.0, 2.0]], dtype=np.float32)], 999),
        )
        callback_threads.append(thread)
        thread.start()
        assert callback_entered.wait(timeout=1.0)

    state.unregister_hook = dispatch_during_unregister
    unload_errors = []
    unload_thread = threading.Thread(
        target=lambda: _capture_exception(runtime.unload, unload_errors)
    )
    unload_thread.start()
    assert callback_entered.wait(timeout=1.0)
    assert state.engines[0].disposed is False

    callback_release.set()
    unload_thread.join(timeout=1.0)
    callback_threads[0].join(timeout=1.0)

    assert not unload_thread.is_alive()
    assert not callback_threads[0].is_alive()
    assert unload_errors == []
    assert state.engines[0].disposed is True


def _capture_exception(action, errors):
    try:
        action()
    except BaseException as exc:
        errors.append(exc)


def test_sdk_free_async_lifecycle_warms_up_and_completes_eight_samples(
    monkeypatch, tmp_path
):
    state = _install_fake_dx_engine(monkeypatch)
    runtime = DeepXRuntime(buffer_count=6)
    runtime.load(_compiled_model(tmp_path))
    backend = runtime.create_native_backend()
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=4,
        completion_timeout_sec=1.0,
    )

    class Loader:
        def __init__(self):
            self.samples = [
                {
                    "input": np.full(
                        (1, 3, 4, 4), index, dtype=np.float32
                    ),
                    "label": index,
                }
                for index in range(8)
            ]

        def get_metadata(self):
            return {
                "total_samples": len(self.samples),
                "is_static_batched": False,
            }

        def load_by_index(self, index):
            return self.samples[index]

        def load_batch(self, batch_size):
            return self.samples[:batch_size]

    class Evaluator:
        def __init__(self):
            self.samples = 0
            self.lock = threading.Lock()

        def add_batch(self, outputs, labels, timing_ms):
            with self.lock:
                self.samples += len(labels)

        def compute(self):
            return {"Total Samples": self.samples}

    result = InferenceEngine(
        Loader(),
        runtime,
        Evaluator(),
        runtime_executor=executor,
    ).run_async(
        AsyncInferenceConfig(
            queue_capacity=16,
            worker_count=4,
            max_batch_size=1,
            batch_timeout_ms=0,
            submit_timeout_sec=1.0,
            flush_timeout_sec=1.0,
            min_samples=8,
            max_samples=8,
        ),
        warmup_runs=2,
    )

    assert result.status is RunStatus.VALID
    assert result.metrics["async_submitted_requests"] == 8
    assert result.metrics["async_accepted_requests"] == 8
    assert result.metrics["async_completed_requests"] == 8
    assert result.metrics["async_failed_requests"] == 0
    assert result.metrics["async_outstanding_requests"] == 0
    assert state.sync_calls == 0
    assert len(state.async_calls) == 10
    runtime.unload()


@pytest.mark.parametrize(
    "buffer_count",
    [True, 1.5, "1.5", float("nan"), 0, 101],
)
def test_invalid_buffer_count_is_rejected(monkeypatch, tmp_path, buffer_count):
    _install_fake_dx_engine(monkeypatch)

    with pytest.raises(ValueError, match="DeepX buffer_count"):
        runtime = DeepXRuntime(buffer_count=buffer_count)
        runtime.load(_compiled_model(tmp_path))


@pytest.mark.parametrize(
    "timeout",
    [True, 0, -1, float("nan"), float("inf"), "invalid"],
)
def test_invalid_async_completion_timeout_is_rejected(
    monkeypatch, tmp_path, timeout
):
    _install_fake_dx_engine(monkeypatch)

    with pytest.raises(
        ValueError,
        match="DeepX async_completion_timeout_sec",
    ):
        runtime = DeepXRuntime(async_completion_timeout_sec=timeout)
        runtime.load(_compiled_model(tmp_path))
