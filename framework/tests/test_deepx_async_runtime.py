import sys
import threading
import types
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from pathlib import Path

import numpy as np
import pytest

from core.compiled_model import CompiledModel
from core.model_spec import Model_Spec, Task
from core.runtime_executor import (
    NativeAsyncOutcome,
    NativeAsyncRuntimeExecutor,
    create_async_runtime_executor,
)
from runtimes.deepx_rt import DeepXRuntime


class FakeDXRTState:
    def __init__(
        self,
        *,
        input_names=("input",),
        inline_outputs=None,
        submit_return_gate=None,
        register_error=None,
        unregister_errors=0,
        callback_during_unregister=False,
        submit_error=None,
    ):
        self.input_names = list(input_names)
        self.output_names = ["output"]
        self.inline_outputs = inline_outputs
        self.submit_return_gate = submit_return_gate
        self.inline_callback_finished = threading.Event()
        self.register_error = register_error
        self.unregister_errors = unregister_errors
        self.callback_during_unregister = callback_during_unregister
        self.unregister_callback_entered = threading.Event()
        self.unregister_callback_threads = []
        self.submit_error = submit_error
        self.condition = threading.Condition()
        self.jobs = {}
        self.next_job_id = 1
        self.registered_callbacks = []
        self.option = None
        self.engine = None
        self.disposed = False

    def wait_for_jobs(self, count, timeout=1.0):
        with self.condition:
            assert self.condition.wait_for(
                lambda: len(self.jobs) >= count,
                timeout=timeout,
            )
            return list(self.jobs)

    def complete(self, job_id, outputs):
        with self.condition:
            job = self.jobs[job_id]
        job["callback"](outputs, job["user_arg"])


def _install_fake_dx_engine(monkeypatch, state: FakeDXRTState):
    class FakeBoundOption:
        NPU_ALL = "NPU_ALL"

    class FakeInferenceOption:
        BOUND_OPTION = FakeBoundOption

        def __init__(self):
            state.option = self

        def set_devices(self, devices):
            self.devices = list(devices)

        def set_bound_option(self, value):
            self.bound_option = value

        def set_use_ort(self, value):
            self.use_ort = value

        def set_buffer_count(self, value):
            self.buffer_count = value

    class FakeInferenceEngine:
        def __init__(self, model_path, option=None):
            self.model_path = model_path
            self.option = option
            self.callback = None
            state.engine = self

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
                    "name": "output",
                    "shape": [1, 2],
                    "dtype": np.dtype("float32"),
                    "elem_size": 4,
                }
            ]

        def register_callback(self, callback):
            if callback is not None and state.register_error is not None:
                raise state.register_error
            if callback is None and state.unregister_errors > 0:
                state.unregister_errors -= 1
                raise RuntimeError("unregister failed")
            if callback is None and state.callback_during_unregister:
                callback_thread = threading.Thread(
                    target=self.callback,
                    args=([], 999),
                    daemon=True,
                )
                state.unregister_callback_threads.append(callback_thread)
                callback_thread.start()
                assert state.unregister_callback_entered.wait(timeout=1.0)
            self.callback = callback
            state.registered_callbacks.append(callback)

        def run(self, input_data):
            return [np.asarray([[1.0, 2.0]], dtype=np.float32)]

        def run_async(self, input_data, user_arg=None, output_buffer=None):
            return self._submit("single", input_data, user_arg, output_buffer)

        def run_async_multi_input(
            self,
            input_tensors,
            user_arg=None,
            output_buffer=None,
        ):
            return self._submit(
                "multi",
                input_tensors,
                user_arg,
                output_buffer,
            )

        def _submit(self, kind, payload, user_arg, output_buffer):
            if state.submit_error is not None:
                raise state.submit_error
            with state.condition:
                job_id = state.next_job_id
                state.next_job_id += 1
                state.jobs[job_id] = {
                    "kind": kind,
                    "payload": payload,
                    "user_arg": user_arg,
                    "output_buffer": output_buffer,
                    "callback": self.callback,
                }
                state.condition.notify_all()
            if state.inline_outputs is not None:
                self.callback(state.inline_outputs, user_arg)
                state.inline_callback_finished.set()
                if state.submit_return_gate is not None:
                    assert state.submit_return_gate.wait(timeout=1.0)
            return job_id

        def dispose(self):
            state.disposed = True

    module = types.SimpleNamespace(
        InferenceEngine=FakeInferenceEngine,
        InferenceOption=FakeInferenceOption,
        __version__="3.3.0-test",
    )
    monkeypatch.setitem(sys.modules, "dx_engine", module)


def _compiled_model(tmp_path: Path, *, input_names=("input",)):
    artifact = tmp_path / "model.dxnn"
    artifact.write_bytes(b"DXNN-test")
    spec = Model_Spec(
        name="deepx-test",
        task=Task.IMAGE_CLASSIFICATION,
        input_shapes={name: (1, 3, 4, 4) for name in input_names},
        input_dtype={name: "float32" for name in input_names},
        output_shapes={"output": (1, 2)},
        model_paths={},
    )
    return CompiledModel(
        spec=spec,
        backend_name="deepx",
        artifact_path=artifact,
    )


def test_deepx_v33_load_registers_callback_and_exposes_native_capacity(
    monkeypatch,
    tmp_path,
):
    """Catches loading the callback-capable SDK while silently using blocking run()."""
    state = FakeDXRTState()
    _install_fake_dx_engine(monkeypatch, state)
    runtime = DeepXRuntime(
        buffer_count=3,
        async_completion_timeout_sec=2.5,
    )

    runtime.load(_compiled_model(tmp_path))

    assert callable(state.engine.callback)
    assert state.option.buffer_count == 3
    assert runtime.supports_native_async() is True
    assert runtime.max_concurrent_workers() == 3
    assert runtime.native_async_max_inflight() == 3
    assert runtime.native_async_completion_timeout_sec() == pytest.approx(2.5)

    runtime.unload()
    assert state.registered_callbacks[-1] is None
    assert state.disposed is True


def test_deepx_submit_async_copies_callback_owned_output(monkeypatch, tmp_path):
    """Catches retaining an SDK-owned output view after the callback returns."""
    state = FakeDXRTState()
    _install_fake_dx_engine(monkeypatch, state)
    runtime = DeepXRuntime(buffer_count=2)
    runtime.load(_compiled_model(tmp_path))
    outcomes = []
    input_array = np.zeros((1, 3, 4, 4), dtype=np.float32)

    vendor_job_id = runtime.submit_async(
        {"input": input_array},
        outcomes.append,
    )
    sdk_output = np.asarray([[7.0, 8.0]], dtype=np.float32)
    state.complete(vendor_job_id, [sdk_output])
    sdk_output.fill(-1.0)

    assert len(outcomes) == 1
    assert isinstance(outcomes[0], NativeAsyncOutcome)
    assert outcomes[0].error_type is None
    assert outcomes[0].timing_ms >= 0.0
    np.testing.assert_array_equal(
        outcomes[0].outputs["output"],
        np.asarray([[7.0, 8.0]], dtype=np.float32),
    )
    job = state.jobs[vendor_job_id]
    assert job["kind"] == "single"
    assert isinstance(job["payload"], list)
    np.testing.assert_array_equal(job["payload"][0], input_array)
    runtime.unload()


def test_deepx_submit_async_uses_named_multi_input_api(monkeypatch, tmp_path):
    """Catches sending named multi-input tensors through the ambiguous list API."""
    state = FakeDXRTState(input_names=("image", "scale"))
    _install_fake_dx_engine(monkeypatch, state)
    runtime = DeepXRuntime(buffer_count=2)
    runtime.load(
        _compiled_model(tmp_path, input_names=("image", "scale"))
    )
    outcomes = []

    vendor_job_id = runtime.submit_async(
        {
            "image": np.zeros((1, 3, 4, 4), dtype=np.float32),
            "scale": np.ones((1, 3, 4, 4), dtype=np.float32),
        },
        outcomes.append,
    )

    job = state.jobs[vendor_job_id]
    assert job["kind"] == "multi"
    assert list(job["payload"]) == ["image", "scale"]
    state.complete(
        vendor_job_id,
        [np.asarray([[3.0, 4.0]], dtype=np.float32)],
    )
    np.testing.assert_array_equal(outcomes[0].outputs["output"], [[3.0, 4.0]])
    runtime.unload()


def test_deepx_submit_async_matches_out_of_order_callbacks(monkeypatch, tmp_path):
    """Catches associating completion with submission order instead of user_arg token."""
    state = FakeDXRTState()
    _install_fake_dx_engine(monkeypatch, state)
    runtime = DeepXRuntime(buffer_count=2)
    runtime.load(_compiled_model(tmp_path))
    observed = []

    first = runtime.submit_async(
        {"input": np.full((1, 3, 4, 4), 1, dtype=np.float32)},
        lambda outcome: observed.append(("first", outcome.outputs["output"])),
    )
    second = runtime.submit_async(
        {"input": np.full((1, 3, 4, 4), 2, dtype=np.float32)},
        lambda outcome: observed.append(("second", outcome.outputs["output"])),
    )

    state.complete(second, [np.asarray([[20.0, 21.0]], dtype=np.float32)])
    state.complete(first, [np.asarray([[10.0, 11.0]], dtype=np.float32)])

    assert [label for label, _ in observed] == ["second", "first"]
    np.testing.assert_array_equal(observed[0][1], [[20.0, 21.0]])
    np.testing.assert_array_equal(observed[1][1], [[10.0, 11.0]])
    runtime.unload()


def test_deepx_submit_async_accepts_inline_callback_before_job_id_return(
    monkeypatch,
    tmp_path,
):
    """Catches publishing job state only after DX-RT can invoke an inline callback."""
    state = FakeDXRTState(
        inline_outputs=[np.asarray([[5.0, 6.0]], dtype=np.float32)]
    )
    _install_fake_dx_engine(monkeypatch, state)
    runtime = DeepXRuntime(buffer_count=1)
    runtime.load(_compiled_model(tmp_path))
    outcomes = []

    vendor_job_id = runtime.submit_async(
        {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)},
        outcomes.append,
    )

    assert vendor_job_id == 1
    assert len(outcomes) == 1
    np.testing.assert_array_equal(outcomes[0].outputs["output"], [[5.0, 6.0]])
    runtime.unload()


def test_deepx_inline_callback_keeps_submission_alive_until_run_async_returns(
    monkeypatch,
    tmp_path,
):
    """Catches unload racing the tail of a still-active DX-RT submission call."""
    release_submit_return = threading.Event()
    state = FakeDXRTState(
        inline_outputs=[np.asarray([[5.0, 6.0]], dtype=np.float32)],
        submit_return_gate=release_submit_return,
    )
    _install_fake_dx_engine(monkeypatch, state)
    runtime = DeepXRuntime(buffer_count=1)
    runtime.load(_compiled_model(tmp_path))
    outcomes = []

    with ThreadPoolExecutor(max_workers=1) as pool:
        submission = pool.submit(
            runtime.submit_async,
            {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)},
            outcomes.append,
        )
        assert state.inline_callback_finished.wait(timeout=1.0)
        assert len(outcomes) == 1
        with pytest.raises(RuntimeError, match="in flight"):
            runtime.unload()
        assert state.disposed is False
        release_submit_return.set()
        assert submission.result(timeout=1.0) == 1

    runtime.unload()
    assert state.disposed is True


def test_deepx_submit_async_rejects_batch_input(monkeypatch, tmp_path):
    """Catches passing unsupported asynchronous batch input into DX-RT v3.3."""
    state = FakeDXRTState()
    _install_fake_dx_engine(monkeypatch, state)
    runtime = DeepXRuntime(buffer_count=2)
    runtime.load(_compiled_model(tmp_path))

    with pytest.raises(ValueError, match="single-sample"):
        runtime.submit_async(
            {"input": np.zeros((2, 3, 4, 4), dtype=np.float32)},
            lambda _outcome: None,
        )

    assert state.jobs == {}
    runtime.unload()


def test_deepx_unknown_future_callback_fails_pending_jobs(monkeypatch, tmp_path):
    """Catches a corrupt DX-RT user_arg leaving requests pending forever."""
    state = FakeDXRTState()
    _install_fake_dx_engine(monkeypatch, state)
    runtime = DeepXRuntime(buffer_count=2)
    runtime.load(_compiled_model(tmp_path))
    outcomes = []
    runtime.submit_async(
        {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)},
        outcomes.append,
    )

    assert runtime._handle_async_completion([], 999) == 0

    assert len(outcomes) == 1
    assert outcomes[0].error_type == "DeepXAsyncProtocolError"
    assert "user_arg" in outcomes[0].error_message
    runtime.unload()


def test_deepx_unmatched_callback_waits_for_safe_multi_job_elimination(
    monkeypatch,
    tmp_path,
):
    """Catches treating one corrupt callback as proof all SDK jobs completed."""
    state = FakeDXRTState()
    _install_fake_dx_engine(monkeypatch, state)
    runtime = DeepXRuntime(buffer_count=2)
    runtime.load(_compiled_model(tmp_path))
    first_outcomes = []
    second_outcomes = []
    first = runtime.submit_async(
        {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)},
        first_outcomes.append,
    )
    runtime.submit_async(
        {"input": np.ones((1, 3, 4, 4), dtype=np.float32)},
        second_outcomes.append,
    )

    runtime._handle_async_completion([], 999)
    assert first_outcomes == []
    assert second_outcomes == []
    with pytest.raises(RuntimeError, match="in flight"):
        runtime.unload()

    state.complete(
        first,
        [np.asarray([[1.0, 2.0]], dtype=np.float32)],
    )

    assert first_outcomes[0].error_type is None
    assert second_outcomes[0].error_type == "DeepXAsyncProtocolError"
    runtime.unload()


@pytest.mark.parametrize(
    "sdk_outputs",
    [
        None,
        [],
        [[np.asarray([[1.0, 2.0]], dtype=np.float32)]],
        [
            np.asarray([[1.0, 2.0]], dtype=np.float32),
            np.asarray([[3.0, 4.0]], dtype=np.float32),
        ],
    ],
    ids=["none", "empty", "nested-batch", "wrong-count"],
)
def test_deepx_malformed_completion_becomes_bounded_error(
    monkeypatch,
    tmp_path,
    sdk_outputs,
):
    """Catches publishing missing DX-RT outputs as a successful inference."""
    state = FakeDXRTState()
    _install_fake_dx_engine(monkeypatch, state)
    runtime = DeepXRuntime(buffer_count=1)
    runtime.load(_compiled_model(tmp_path))
    outcomes = []
    vendor_job_id = runtime.submit_async(
        {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)},
        outcomes.append,
    )

    state.complete(vendor_job_id, sdk_outputs)

    assert len(outcomes) == 1
    assert outcomes[0].outputs is None
    assert outcomes[0].error_type == "DeepXAsyncCompletionError"
    assert len(outcomes[0].error_message) <= 512
    assert "output" in outcomes[0].error_message.lower()
    runtime.unload()


@pytest.mark.parametrize("buffer_count", [1.5, 2.9])
def test_deepx_rejects_fractional_buffer_count(
    monkeypatch,
    tmp_path,
    buffer_count,
):
    """Catches silently truncating a non-integral DX-RT buffer count."""
    state = FakeDXRTState()
    _install_fake_dx_engine(monkeypatch, state)
    runtime = DeepXRuntime(buffer_count=buffer_count)

    with pytest.raises(ValueError, match="DeepX buffer_count"):
        runtime.load(_compiled_model(tmp_path))

    assert state.engine is None


def test_deepx_runtime_refuses_unload_until_native_callback(
    monkeypatch,
    tmp_path,
):
    """Catches disposing DX-RT while its callback still owns job buffers."""
    state = FakeDXRTState()
    _install_fake_dx_engine(monkeypatch, state)
    runtime = DeepXRuntime(buffer_count=1)
    runtime.load(_compiled_model(tmp_path))
    vendor_job_id = runtime.submit_async(
        {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)},
        lambda _outcome: None,
    )

    with pytest.raises(RuntimeError, match="in flight"):
        runtime.unload()

    assert state.disposed is False
    state.complete(
        vendor_job_id,
        [np.asarray([[1.0, 2.0]], dtype=np.float32)],
    )
    runtime.unload()
    assert state.disposed is True


def test_deepx_runtime_refuses_unload_while_callback_is_copying_outputs(
    monkeypatch,
    tmp_path,
):
    """Catches disposing SDK-owned output memory during callback processing."""
    state = FakeDXRTState()
    _install_fake_dx_engine(monkeypatch, state)
    runtime = DeepXRuntime(buffer_count=1)
    runtime.load(_compiled_model(tmp_path))
    copy_started = threading.Event()
    release_copy = threading.Event()
    original_copy = runtime._copy_async_output

    def blocking_copy(value):
        copy_started.set()
        assert release_copy.wait(timeout=1.0)
        return original_copy(value)

    monkeypatch.setattr(runtime, "_copy_async_output", blocking_copy)

    vendor_job_id = runtime.submit_async(
        {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)},
        lambda _outcome: None,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        completion = pool.submit(
            state.complete,
            vendor_job_id,
            [np.asarray([[1.0, 2.0]], dtype=np.float32)],
        )
        assert copy_started.wait(timeout=1.0)
        with pytest.raises(RuntimeError, match="in flight"):
            runtime.unload()
        assert state.disposed is False
        release_copy.set()
        completion.result(timeout=1.0)

    runtime.unload()
    assert state.disposed is True


def test_deepx_unload_waits_for_callback_publication_to_return(
    monkeypatch,
    tmp_path,
):
    """Catches teardown racing the callback after completion publication."""
    state = FakeDXRTState()
    _install_fake_dx_engine(monkeypatch, state)
    runtime = DeepXRuntime(buffer_count=1, async_completion_timeout_sec=1.0)
    runtime.load(_compiled_model(tmp_path))
    publication_started = threading.Event()
    release_publication = threading.Event()

    def blocking_publication(_outcome):
        publication_started.set()
        assert release_publication.wait(timeout=1.0)

    vendor_job_id = runtime.submit_async(
        {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)},
        blocking_publication,
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        completion = pool.submit(
            state.complete,
            vendor_job_id,
            [np.asarray([[1.0, 2.0]], dtype=np.float32)],
        )
        assert publication_started.wait(timeout=1.0)
        unloading = pool.submit(runtime.unload)
        with pytest.raises(FutureTimeoutError):
            unloading.result(timeout=0.05)
        assert state.disposed is False
        release_publication.set()
        completion.result(timeout=1.0)
        unloading.result(timeout=1.0)

    assert state.disposed is True


def test_deepx_submit_racing_with_unload_never_calls_disposed_engine(
    monkeypatch,
    tmp_path,
):
    """Catches publishing a job after unload passed its in-flight check."""
    state = FakeDXRTState()
    _install_fake_dx_engine(monkeypatch, state)
    runtime = DeepXRuntime(buffer_count=1)
    runtime.load(_compiled_model(tmp_path))
    prepare_started = threading.Event()
    release_prepare = threading.Event()
    original_prepare = runtime._prepare_ordered_inputs

    def blocking_prepare(inputs):
        prepared = original_prepare(inputs)
        prepare_started.set()
        assert release_prepare.wait(timeout=1.0)
        return prepared

    monkeypatch.setattr(runtime, "_prepare_ordered_inputs", blocking_prepare)
    with ThreadPoolExecutor(max_workers=1) as pool:
        submission = pool.submit(
            runtime.submit_async,
            {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)},
            lambda _outcome: None,
        )
        assert prepare_started.wait(timeout=1.0)
        runtime.unload()
        release_prepare.set()
        with pytest.raises(RuntimeError, match="unloading or not loaded"):
            submission.result(timeout=1.0)

    assert state.jobs == {}
    assert state.disposed is True


def test_deepx_callback_registration_failure_preserves_blocking_e2e(
    monkeypatch,
    tmp_path,
):
    """Catches breaking synchronous inference on a partial/older DX-RT build."""
    state = FakeDXRTState(register_error=RuntimeError("callback unavailable"))
    _install_fake_dx_engine(monkeypatch, state)
    runtime = DeepXRuntime()
    runtime.load(_compiled_model(tmp_path))

    assert runtime.supports_native_async() is False
    outputs = runtime.run(
        {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)}
    )
    np.testing.assert_array_equal(outputs["output"], [[1.0, 2.0]])
    runtime.unload()
    assert state.disposed is True


def test_deepx_submit_failure_releases_retained_job_state(monkeypatch, tmp_path):
    """Catches a failed SDK submission permanently blocking runtime unload."""
    state = FakeDXRTState(submit_error=RuntimeError("queue rejected"))
    _install_fake_dx_engine(monkeypatch, state)
    runtime = DeepXRuntime(buffer_count=1)
    runtime.load(_compiled_model(tmp_path))

    with pytest.raises(RuntimeError, match="queue rejected"):
        runtime.submit_async(
            {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)},
            lambda _outcome: None,
        )

    runtime.unload()
    assert state.disposed is True


def test_deepx_unregister_failure_is_retry_safe(monkeypatch, tmp_path):
    """Catches disposing an engine whose DX-RT callback is still registered."""
    state = FakeDXRTState(unregister_errors=1)
    _install_fake_dx_engine(monkeypatch, state)
    runtime = DeepXRuntime()
    runtime.load(_compiled_model(tmp_path))

    with pytest.raises(RuntimeError, match="unregister failed"):
        runtime.unload()

    assert state.disposed is False
    assert state.engine.callback is not None
    runtime.unload()
    assert state.registered_callbacks[-1] is None
    assert state.disposed is True


def test_deepx_unload_waits_for_callback_dispatched_during_unregister(
    monkeypatch,
    tmp_path,
):
    """Catches dispose racing a callback already dispatched by DX-RT."""
    state = FakeDXRTState(callback_during_unregister=True)
    _install_fake_dx_engine(monkeypatch, state)
    runtime = DeepXRuntime(async_completion_timeout_sec=1.0)
    runtime.load(_compiled_model(tmp_path))
    release_callback = threading.Event()
    publication_started = threading.Event()

    def blocking_publication(_outcome):
        publication_started.set()
        assert release_callback.wait(timeout=1.0)

    synthetic_record = {
        "callback": blocking_publication,
        "started_ns": 0,
        "completion_started": True,
        "completion_finished": False,
        "submission_finished": True,
    }

    def blocking_claim():
        state.unregister_callback_entered.set()
        return [(123, synthetic_record)]

    monkeypatch.setattr(
        runtime,
        "_claim_unmatched_protocol_jobs_locked",
        blocking_claim,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        unloading = pool.submit(runtime.unload)
        assert state.unregister_callback_entered.wait(timeout=1.0)
        assert publication_started.wait(timeout=1.0)
        with pytest.raises(FutureTimeoutError):
            unloading.result(timeout=0.05)
        assert state.disposed is False
        release_callback.set()
        unloading.result(timeout=1.0)

    for callback_thread in state.unregister_callback_threads:
        callback_thread.join(timeout=1.0)
    assert state.disposed is True


def test_deepx_async_deep_copies_ragged_object_outputs(monkeypatch, tmp_path):
    """Catches shallow-copying child arrays in an SDK-owned object output."""
    state = FakeDXRTState()
    _install_fake_dx_engine(monkeypatch, state)
    runtime = DeepXRuntime(buffer_count=1)
    runtime.load(_compiled_model(tmp_path))
    outcomes = []
    vendor_job_id = runtime.submit_async(
        {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)},
        outcomes.append,
    )
    child = np.asarray([3.0, 4.0], dtype=np.float32)
    sdk_output = np.empty((1,), dtype=object)
    sdk_output[0] = child

    state.complete(vendor_job_id, [sdk_output])
    child.fill(-1.0)

    np.testing.assert_array_equal(outcomes[0].outputs["output"][0], [3.0, 4.0])
    runtime.unload()


def test_deepx_native_async_executor_runs_callback_job_end_to_end(
    monkeypatch,
    tmp_path,
):
    """Catches advertising native async without satisfying the executor contract."""
    state = FakeDXRTState()
    _install_fake_dx_engine(monkeypatch, state)
    runtime = DeepXRuntime(buffer_count=2)
    runtime.load(_compiled_model(tmp_path))
    executor = create_async_runtime_executor(runtime, worker_count=4)

    assert isinstance(executor, NativeAsyncRuntimeExecutor)
    assert executor.max_inflight == 2
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            executor.execute,
            {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)},
        )
        vendor_job_id = state.wait_for_jobs(1)[0]
        state.complete(
            vendor_job_id,
            [np.asarray([[9.0, 10.0]], dtype=np.float32)],
        )
        execution = future.result(timeout=1.0)

    assert execution.error_type is None
    assert execution.vendor_job_id == vendor_job_id
    np.testing.assert_array_equal(execution.outputs["output"], [[9.0, 10.0]])
    executor.acknowledge(execution)
    assert executor.shutdown(timeout=0.1) is True
    runtime.unload()
