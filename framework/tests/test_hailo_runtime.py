from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from core.compiled_model import CompiledModel
from core.model_spec import Model_Spec, Task
from core.runtime_executor import NativeAsyncOutcome, NativeAsyncRuntimeExecutor
from decoders.object_detection import HailoYoloNMSDecoder
from runtimes.hailo_rt import HailoRuntime


class _FakeVStreamInfo:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _FakeActivation:
    def __init__(self, state):
        self._state = state

    def __enter__(self):
        self._state["activation_entered"] = True
        return self

    def __exit__(self, *_args):
        self._state["activation_exited"] = True


def _make_compiled_model(tmp_path: Path) -> CompiledModel:
    hef_path = tmp_path / "model.hef"
    hef_path.write_bytes(b"fake hef")
    spec = Model_Spec(
        name="fake_hailo_model",
        task=Task.IMAGE_CLASSIFICATION,
        input_shapes={"input": (1, 3, 4, 4)},
        input_dtype={"input": "float32"},
        output_shapes={"output": (1, 10)},
        model_paths={"hef": str(hef_path)},
    )
    return CompiledModel(spec=spec, backend_name="hailort", artifact_path=hef_path)


def _fake_hailo_platform(
    state,
    *,
    activate_returns_none=False,
    supports_infer_model_api=True,
    async_queue_size=4,
    output_shape=(10,),
):
    state.setdefault("async_job_submitted", threading.Event())
    state.setdefault("async_jobs_condition", threading.Condition())

    class FakeHEF:
        def __init__(self, path):
            state["hef_path"] = path

        def get_network_group_names(self):
            return ["net"]

        def get_input_vstream_infos(self):
            return [_FakeVStreamInfo("input", (4, 4, 3))]

        def get_output_vstream_infos(self):
            return [_FakeVStreamInfo("output", output_shape)]

    class FakeInferTensor:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape
            self.format = SimpleNamespace(type="UINT8", order="NORMAL")

        def set_format_type(self, format_type):
            state.setdefault("format_types", {})[self.name] = format_type

    class FakeBindingEndpoint:
        def __init__(self, buffers, name):
            self._buffers = buffers
            self._name = name

        def set_buffer(self, buffer):
            self._buffers[self._name] = buffer

        def get_buffer(self):
            return self._buffers[self._name]

    class FakeBindings:
        def __init__(self, output_buffers):
            self._input_buffers = {}
            self._output_buffers = output_buffers
            self._output_names = list(output_buffers)

        def input(self, name="input"):
            return FakeBindingEndpoint(self._input_buffers, name)

        def output(self, name="output"):
            return FakeBindingEndpoint(self._output_buffers, name)

    class FakeConfiguredInferModel:
        def create_bindings(self, output_buffers=None):
            binding_error = state.pop("async_binding_error", None)
            if binding_error is not None:
                raise binding_error
            binding = FakeBindings(
                output_buffers
                or {"output": np.zeros(output_shape, dtype=np.uint8)}
            )
            state.setdefault("bindings", []).append(binding)
            return binding

        def run(self, bindings, timeout_ms=1000):
            state["configured_infer_model_run"] = True
            state["run_timeout_ms"] = timeout_ms
            state["run_bindings_count"] = len(bindings)
            state["last_input_buffers"] = [binding._input_buffers for binding in bindings]

        def get_async_queue_size(self):
            return async_queue_size

        def wait_for_async_ready(self, timeout_ms=1000, frames_count=1):
            state.setdefault("async_ready_calls", []).append(
                (timeout_ms, frames_count)
            )
            ready_entered = state.get("async_ready_entered")
            if ready_entered is not None:
                ready_entered.set()
            ready_release = state.get("async_ready_release")
            if ready_release is not None:
                ready_release.wait(timeout=1.0)
            ready_sequence = state.get("async_ready_sequence")
            if ready_sequence is not None:
                ready_index = len(state["async_ready_calls"]) - 1
                ready_entered, ready_release = ready_sequence[ready_index]
                ready_entered.set()
                ready_release.wait(timeout=1.0)
            ready_error = state.pop("async_ready_error", None)
            if ready_error is not None:
                raise ready_error

        def run_async(self, bindings, callback=None):
            run_error = state.pop("async_run_error", None)
            if run_error is not None:
                raise run_error
            with state["async_jobs_condition"]:
                job_id = len(state.setdefault("async_jobs", {})) + 1
                state["async_jobs"][job_id] = SimpleNamespace(
                    bindings=bindings,
                    callback=callback,
                )
                state["async_jobs_condition"].notify_all()
            state["async_job_submitted"].set()
            if state.get("async_inline_completion"):
                callback(completion_info=SimpleNamespace(exception=None))
            return SimpleNamespace(job_id=job_id)

    class FakeConfiguredInferModelContext:
        def __enter__(self):
            state["configured_infer_model_entered"] = True
            return FakeConfiguredInferModel()

        def __exit__(self, *_args):
            state["configured_infer_model_exited"] = True

    class FakeInferModel:
        def __init__(self, path):
            state["infer_model_path"] = path
            self.inputs = [FakeInferTensor("input", (4, 4, 3))]
            self.outputs = [FakeInferTensor("output", output_shape)]

        def set_batch_size(self, batch_size):
            state["infer_model_batch_size"] = batch_size

        def input(self, name="input"):
            return self.inputs[0]

        def output(self, name="output"):
            return self.outputs[0]

        def configure(self):
            state["infer_model_configure_called"] = True
            return FakeConfiguredInferModelContext()

    class FakeConfigureParams:
        @staticmethod
        def create_from_hef(hef, interface):
            state["configure_interface"] = interface
            params = {"net": SimpleNamespace(batch_size=None)}
            state["configure_params"] = params
            return params

    class FakeNetworkGroup:
        def create_params(self):
            state["network_group_params_created"] = True
            return SimpleNamespace()

        def activate(self, _params):
            state["activation_called"] = True
            if activate_returns_none:
                return None
            return _FakeActivation(state)

    class FakeVDevice:
        @staticmethod
        def create_params():
            params = SimpleNamespace(group_id=None, multi_process_service=None)
            state["vdevice_create_params"] = params
            return params

        def __init__(self, params=None, *, device_ids=None):
            state["vdevice_params"] = params
            state["vdevice_device_ids"] = device_ids

        def __enter__(self):
            state["vdevice_entered"] = True
            return self

        def __exit__(self, *_args):
            state["vdevice_exited"] = True

        def configure(self, hef, configure_params):
            state["configured_hef"] = hef
            state["configured_params"] = configure_params
            return [FakeNetworkGroup()]

    if supports_infer_model_api:
        def create_infer_model(self, path):
            return FakeInferModel(path)

        FakeVDevice.create_infer_model = create_infer_model

    class FakeInputVStreamParams:
        @staticmethod
        def make(network_group, **kwargs):
            state["input_vstream_kwargs"] = kwargs
            return {"input": "input_params"}

    class FakeOutputVStreamParams:
        @staticmethod
        def make(network_group, **kwargs):
            state["output_vstream_kwargs"] = kwargs
            return {"output": "output_params"}

    class FakeInferVStreams:
        def __init__(self, network_group, input_params, output_params, tf_nms_format=False):
            state["infer_input_params"] = input_params
            state["infer_output_params"] = output_params
            state["infer_tf_nms_format"] = tf_nms_format

        def __enter__(self):
            state["infer_entered"] = True
            return self

        def __exit__(self, *_args):
            state["infer_exited"] = True

        def infer(self, input_data):
            state["last_input"] = input_data
            batch_size = next(iter(input_data.values())).shape[0]
            return {"output": np.zeros((batch_size, 10), dtype=np.uint8)}

        def set_nms_score_threshold(self, threshold):
            state["nms_score_threshold"] = threshold

        def set_nms_iou_threshold(self, threshold):
            state["nms_iou_threshold"] = threshold

        def set_nms_max_proposals_per_class(self, max_proposals_per_class):
            state["nms_max_proposals_per_class"] = max_proposals_per_class

        def set_nms_max_accumulated_mask_size(self, max_accumulated_mask_size):
            state["nms_max_accumulated_mask_size"] = max_accumulated_mask_size

    return SimpleNamespace(
        HEF=FakeHEF,
        VDevice=FakeVDevice,
        ConfigureParams=FakeConfigureParams,
        HailoStreamInterface=SimpleNamespace(PCIe="pcie", ETH="eth", INTEGRATED="integrated"),
        FormatType=SimpleNamespace(FLOAT32="float32", UINT8="uint8", UINT16="uint16", AUTO="auto"),
        InputVStreamParams=FakeInputVStreamParams,
        OutputVStreamParams=FakeOutputVStreamParams,
        InferVStreams=FakeInferVStreams,
        HailoSchedulingAlgorithm=SimpleNamespace(ROUND_ROBIN="round_robin"),
    )


def _submitted_async_job(state, vendor_job_id):
    assert state["async_job_submitted"].wait(timeout=1.0)
    return state["async_jobs"][vendor_job_id]


def _latest_submitted_async_job(state):
    assert state["async_job_submitted"].wait(timeout=1.0)
    return state["async_jobs"][max(state["async_jobs"])]


def _wait_for_async_job_count(state, expected):
    with state["async_jobs_condition"]:
        assert state["async_jobs_condition"].wait_for(
            lambda: len(state.get("async_jobs", {})) >= expected,
            timeout=1.0,
        )
        return dict(state["async_jobs"])


class _AsyncOutcomeCollector:
    def __init__(self):
        self.values = []
        self.event = threading.Event()

    def __call__(self, outcome):
        self.values.append(outcome)
        self.event.set()

    def wait_one(self):
        assert self.event.wait(timeout=1.0)
        assert len(self.values) == 1
        return self.values[0]


def test_hailo_runtime_uses_create_infer_model_api(tmp_path, monkeypatch):
    state = {}
    runtime = HailoRuntime(batch_size=2, input_layout="NCHW")
    monkeypatch.setattr(runtime, "_import_hailo_platform", lambda: _fake_hailo_platform(state))

    runtime.load(_make_compiled_model(tmp_path))
    outputs = runtime.run({"image": np.zeros((2, 3, 4, 4), dtype=np.float32)})
    runtime.unload()

    assert state["infer_model_path"].endswith("model.hef")
    assert state["infer_model_batch_size"] == 2
    assert state["configured_infer_model_run"] is True
    assert state["configured_infer_model_exited"] is True
    assert state["vdevice_exited"] is True
    assert "configured_params" not in state
    assert state["last_input_buffers"][0]["input"].shape == (4, 4, 3)
    assert state["run_bindings_count"] == 2
    assert outputs["output"].shape == (2, 10)


def test_hailo_runtime_legacy_vstreams_handles_scheduler_activate_none(tmp_path, monkeypatch):
    state = {}
    runtime = HailoRuntime(batch_size=2, input_layout="NCHW")
    monkeypatch.setattr(
        runtime,
        "_import_hailo_platform",
        lambda: _fake_hailo_platform(
            state,
            activate_returns_none=True,
            supports_infer_model_api=False,
        ),
    )

    runtime.load(_make_compiled_model(tmp_path))
    outputs = runtime.run({"image": np.zeros((2, 3, 4, 4), dtype=np.float32)})
    runtime.unload()

    assert state["activation_called"] is True
    assert state.get("activation_entered") is None
    assert state["infer_entered"] is True
    assert state["infer_exited"] is True
    assert state["vdevice_exited"] is True
    assert state["configure_params"]["net"].batch_size == 2
    assert state["last_input"]["input"].shape == (2, 4, 4, 3)
    assert outputs["output"].shape == (2, 10)


def test_hailo_runtime_passes_explicit_device_id_to_vdevice(tmp_path, monkeypatch):
    state = {}
    runtime = HailoRuntime(device="0000:01:00.0")
    monkeypatch.setattr(runtime, "_import_hailo_platform", lambda: _fake_hailo_platform(state))

    runtime.load(_make_compiled_model(tmp_path))
    runtime.unload()

    assert state["vdevice_device_ids"] == ["0000:01:00.0"]
    assert state["vdevice_params"] is None


def test_hailo_runtime_reports_configured_accelerator_name():
    runtime = HailoRuntime(accelerator_name="Hailo-10H")

    assert runtime.get_device_spec()["accelerator_name"] == "Hailo-10H"


def test_hailo_runtime_rejects_device_ids_with_vdevice_params(tmp_path, monkeypatch):
    state = {}
    runtime = HailoRuntime(device_ids="0000:01:00.0", group_id="group-a")
    monkeypatch.setattr(runtime, "_import_hailo_platform", lambda: _fake_hailo_platform(state))

    with pytest.raises(ValueError, match="device_ids cannot be combined"):
        runtime.load(_make_compiled_model(tmp_path))


def test_hailo_runtime_casts_raw_float_image_to_uint8(tmp_path, monkeypatch):
    state = {}
    runtime = HailoRuntime(input_format_type="uint8")
    monkeypatch.setattr(runtime, "_import_hailo_platform", lambda: _fake_hailo_platform(state))

    runtime.load(_make_compiled_model(tmp_path))
    runtime.run({"input": np.full((1, 4, 4, 3), 260.0, dtype=np.float32)})
    runtime.unload()

    prepared = state["last_input_buffers"][0]["input"]
    assert prepared.dtype == np.uint8
    assert int(prepared.max()) == 255


def test_hailo_runtime_auto_output_format_uses_hef_metadata(tmp_path, monkeypatch):
    state = {}
    runtime = HailoRuntime(output_format_type="auto")
    monkeypatch.setattr(runtime, "_import_hailo_platform", lambda: _fake_hailo_platform(state))

    runtime.load(_make_compiled_model(tmp_path))
    outputs = runtime.run({"input": np.zeros((1, 4, 4, 3), dtype=np.float32)})
    runtime.unload()

    assert state["format_types"]["input"] == "float32"
    assert "output" not in state["format_types"]
    assert outputs["output"].dtype == np.uint8


def test_hailo_runtime_passes_tf_nms_format_and_nms_options(tmp_path, monkeypatch):
    state = {}
    runtime = HailoRuntime(
        tf_nms_format=True,
        hailo_nms_conf_threshold=0.2,
        hailo_nms_iou_threshold=0.5,
        hailo_nms_max_proposals_per_class=12,
        hailo_nms_max_accumulated_mask_size=4096,
    )
    monkeypatch.setattr(
        runtime,
        "_import_hailo_platform",
        lambda: _fake_hailo_platform(state, supports_infer_model_api=False),
    )

    runtime.load(_make_compiled_model(tmp_path))
    runtime.unload()

    assert state["infer_tf_nms_format"] is True
    assert state["nms_score_threshold"] == 0.2
    assert state["nms_iou_threshold"] == 0.5
    assert state["nms_max_proposals_per_class"] == 12
    assert state["nms_max_accumulated_mask_size"] == 4096


def test_hailo_runtime_native_async_resnet_preserves_batch_and_ready_contract(
    tmp_path, monkeypatch
):
    """Catches using one global binding or omitting frames_count on native submit."""
    state = {}
    runtime = HailoRuntime(
        batch_size=2,
        input_layout="NCHW",
        async_ready_timeout_ms=4321,
        async_completion_timeout_ms=7654,
    )
    monkeypatch.setattr(
        runtime,
        "_import_hailo_platform",
        lambda: _fake_hailo_platform(state, async_queue_size=3),
    )
    runtime.load(_make_compiled_model(tmp_path))
    collector = _AsyncOutcomeCollector()

    vendor_job_id = runtime.submit_async(
        {
            "image": np.stack(
                [
                    np.full((3, 4, 4), 11, dtype=np.float32),
                    np.full((3, 4, 4), 22, dtype=np.float32),
                ]
            )
        },
        collector,
    )

    assert vendor_job_id == 1
    assert runtime.supports_native_async() is True
    assert runtime.max_concurrent_workers() == 3
    assert runtime.native_async_max_inflight() == 3
    assert runtime.native_async_completion_timeout_sec() == pytest.approx(20.617)
    job = _submitted_async_job(state, vendor_job_id)
    assert state["async_ready_calls"] == [(4321, 2)]
    assert len(job.bindings) == 2
    assert int(job.bindings[0]._input_buffers["input"][0, 0, 0]) == 11
    assert int(job.bindings[1]._input_buffers["input"][0, 0, 0]) == 22
    job.bindings[0]._output_buffers["output"].fill(101)
    job.bindings[1]._output_buffers["output"].fill(202)

    job.callback(completion_info=SimpleNamespace(exception=None))

    outcome = collector.wait_one()
    assert isinstance(outcome, NativeAsyncOutcome)
    assert outcome.error_type is None
    assert outcome.timing_ms >= 0.0
    np.testing.assert_array_equal(
        outcome.outputs["output"][:, 0],
        np.asarray([101, 202]),
    )
    job.bindings[0]._output_buffers["output"].fill(9)
    assert int(outcome.outputs["output"][0, 0]) == 101
    runtime.unload()


def test_hailo_runtime_native_async_submission_does_not_block_on_sdk_readiness(
    tmp_path, monkeypatch
):
    """Keeps the native executor completion phase separate from SDK readiness."""
    state = {
        "async_ready_entered": threading.Event(),
        "async_ready_release": threading.Event(),
    }
    runtime = HailoRuntime()
    monkeypatch.setattr(
        runtime,
        "_import_hailo_platform",
        lambda: _fake_hailo_platform(state),
    )
    runtime.load(_make_compiled_model(tmp_path))
    collector = _AsyncOutcomeCollector()

    with ThreadPoolExecutor(max_workers=1) as pool:
        submission = pool.submit(
            runtime.submit_async,
            {"input": np.zeros((1, 4, 4, 3), dtype=np.float32)},
            collector,
        )
        try:
            assert state["async_ready_entered"].wait(timeout=1.0)
            vendor_job_id = submission.result(timeout=0.2)
        finally:
            state["async_ready_release"].set()

    _submitted_async_job(state, vendor_job_id).callback(
        completion_info=SimpleNamespace(exception=None)
    )
    assert collector.wait_one().error_type is None
    runtime.unload()


def test_hailo_runtime_last_queued_job_keeps_ready_and_completion_budget(
    tmp_path, monkeypatch
):
    """Accounts for every serialized ready phase before the final queued job."""
    ready_sequence = [
        (threading.Event(), threading.Event()),
        (threading.Event(), threading.Event()),
    ]
    state = {"async_ready_sequence": ready_sequence}
    runtime = HailoRuntime(
        async_ready_timeout_ms=200,
        async_completion_timeout_ms=100,
    )
    monkeypatch.setattr(
        runtime,
        "_import_hailo_platform",
        lambda: _fake_hailo_platform(state, async_queue_size=2),
    )
    runtime.load(_make_compiled_model(tmp_path))
    executor = NativeAsyncRuntimeExecutor(
        runtime,
        max_inflight=2,
        completion_timeout_sec=runtime.native_async_completion_timeout_sec(),
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                executor.execute,
                {"input": np.full((1, 4, 4, 3), value, dtype=np.float32)},
            )
            for value in (1, 2)
        ]
        assert ready_sequence[0][0].wait(timeout=1.0)
        threading.Event().wait(0.16)
        ready_sequence[0][1].set()
        assert ready_sequence[1][0].wait(timeout=1.0)
        threading.Event().wait(0.16)
        ready_sequence[1][1].set()
        jobs = _wait_for_async_job_count(state, 2)
        for job_id, job in jobs.items():
            job.bindings[0]._output_buffers["output"].fill(job_id)
            job.callback(completion_info=SimpleNamespace(exception=None))
        executions = [future.result(timeout=1.0) for future in futures]

    assert all(execution.error_type is None for execution in executions)
    for execution in executions:
        executor.acknowledge(execution)
    assert executor.shutdown(timeout=0.0) is True
    runtime.unload()


def test_hailo_runtime_native_async_yolov5m_preserves_nms_output(
    tmp_path, monkeypatch
):
    """Catches flattening or reordering YOLOv5m's Hailo NMS tensor in async completion."""
    state = {}
    runtime = HailoRuntime(output_format_type="float32")
    monkeypatch.setattr(
        runtime,
        "_import_hailo_platform",
        lambda: _fake_hailo_platform(
            state,
            output_shape=(80, 5, 3),
        ),
    )
    runtime.load(_make_compiled_model(tmp_path))
    collector = _AsyncOutcomeCollector()
    vendor_job_id = runtime.submit_async(
        {"input": np.zeros((1, 4, 4, 3), dtype=np.float32)},
        collector,
    )
    job = _submitted_async_job(state, vendor_job_id)
    nms_output = job.bindings[0]._output_buffers[
        "output"
    ]
    nms_output.fill(0.0)
    nms_output[4, :, 0] = np.asarray(
        [0.1, 0.2, 0.5, 0.6, 0.9],
        dtype=np.float32,
    )

    job.callback(
        completion_info=SimpleNamespace(exception=None)
    )

    outcome = collector.wait_one()
    assert outcome.outputs["output"].shape == (1, 80, 5, 3)
    decoded = HailoYoloNMSDecoder(conf_threshold=0.25).decode(
        outcome.outputs
    )["detections"]
    assert decoded.shape == (1, 7)
    np.testing.assert_allclose(
        decoded[0],
        np.asarray([0, 4, 0.9, 128, 64, 384, 320], dtype=np.float32),
    )
    runtime.unload()


def test_hailo_runtime_native_async_copies_ragged_yolo_nms_buffers(
    tmp_path, monkeypatch
):
    """Catches retaining Hailo-owned ragged child arrays after callback return."""
    state = {}
    runtime = HailoRuntime(output_format_type="float32")
    monkeypatch.setattr(
        runtime,
        "_import_hailo_platform",
        lambda: _fake_hailo_platform(state),
    )
    runtime.load(_make_compiled_model(tmp_path))
    collector = _AsyncOutcomeCollector()
    vendor_job_id = runtime.submit_async(
        {"input": np.zeros((1, 4, 4, 3), dtype=np.float32)},
        collector,
    )
    ragged = np.empty((2,), dtype=object)
    ragged[0] = np.asarray([[0.1, 0.2, 0.5, 0.6, 0.9]], dtype=np.float32)
    ragged[1] = np.empty((0, 5), dtype=np.float32)
    job = _submitted_async_job(state, vendor_job_id)
    job.bindings[0]._output_buffers[
        "output"
    ] = ragged

    job.callback(
        completion_info=SimpleNamespace(exception=None)
    )
    copied_child = collector.wait_one().outputs["output"].reshape(-1)[0]
    ragged[0][0, 4] = 0.0

    assert float(copied_child[0, 4]) == pytest.approx(0.9)
    runtime.unload()


def test_hailo_runtime_native_async_refuses_unload_until_callback(
    tmp_path, monkeypatch
):
    """Catches releasing the configured model while Hailo still owns buffers."""
    state = {}
    runtime = HailoRuntime()
    monkeypatch.setattr(
        runtime,
        "_import_hailo_platform",
        lambda: _fake_hailo_platform(state),
    )
    runtime.load(_make_compiled_model(tmp_path))
    collector = _AsyncOutcomeCollector()
    vendor_job_id = runtime.submit_async(
        {"input": np.zeros((1, 4, 4, 3), dtype=np.float32)},
        collector,
    )

    with pytest.raises(RuntimeError, match="in flight"):
        runtime.unload()

    _submitted_async_job(state, vendor_job_id).callback(
        completion_info=SimpleNamespace(exception=None)
    )
    assert collector.wait_one().error_type is None
    runtime.unload()
    assert state["configured_infer_model_exited"] is True


def test_hailo_runtime_native_async_keeps_sdk_callback_quick_while_copying(
    tmp_path, monkeypatch
):
    """Moves dense/ragged output ownership work off the Hailo callback thread."""
    state = {}
    runtime = HailoRuntime()
    monkeypatch.setattr(
        runtime,
        "_import_hailo_platform",
        lambda: _fake_hailo_platform(state),
    )
    runtime.load(_make_compiled_model(tmp_path))
    copy_entered = threading.Event()
    copy_release = threading.Event()
    collector = _AsyncOutcomeCollector()
    original_copy = runtime._copy_async_output

    def blocking_copy(value):
        copy_entered.set()
        assert copy_release.wait(timeout=1.0)
        return original_copy(value)

    monkeypatch.setattr(runtime, "_copy_async_output", blocking_copy)
    vendor_job_id = runtime.submit_async(
        {"input": np.zeros((1, 4, 4, 3), dtype=np.float32)},
        collector,
    )
    job = _submitted_async_job(state, vendor_job_id)

    with ThreadPoolExecutor(max_workers=1) as pool:
        sdk_callback = pool.submit(
            job.callback,
            completion_info=SimpleNamespace(exception=None),
        )
        try:
            assert copy_entered.wait(timeout=1.0)
            sdk_callback.result(timeout=0.2)
        finally:
            copy_release.set()

    assert collector.wait_one().error_type is None
    runtime.unload()


def test_hailo_runtime_unload_waits_for_adapter_completion_callback(
    tmp_path, monkeypatch
):
    """Prevents executor shutdown from racing adapter callback finalization."""
    state = {}
    runtime = HailoRuntime()
    monkeypatch.setattr(
        runtime,
        "_import_hailo_platform",
        lambda: _fake_hailo_platform(state),
    )
    runtime.load(_make_compiled_model(tmp_path))
    callback_entered = threading.Event()
    callback_release = threading.Event()

    def blocking_framework_callback(_outcome):
        callback_entered.set()
        assert callback_release.wait(timeout=1.0)

    vendor_job_id = runtime.submit_async(
        {"input": np.zeros((1, 4, 4, 3), dtype=np.float32)},
        blocking_framework_callback,
    )
    job = _submitted_async_job(state, vendor_job_id)
    with ThreadPoolExecutor(max_workers=2) as pool:
        sdk_callback = pool.submit(
            job.callback,
            completion_info=SimpleNamespace(exception=None),
        )
        try:
            assert callback_entered.wait(timeout=1.0)
            sdk_callback.result(timeout=0.2)
            unloading = pool.submit(runtime.unload)
            assert unloading.done() is False
        finally:
            callback_release.set()
        sdk_callback.result(timeout=1.0)
        unloading.result(timeout=1.0)

    assert state["configured_infer_model_exited"] is True


def test_hailo_runtime_native_async_maps_device_failure_and_closes_pipeline(
    tmp_path, monkeypatch
):
    """Catches treating a failed Hailo completion as a successful empty output."""
    state = {}
    runtime = HailoRuntime()
    monkeypatch.setattr(
        runtime,
        "_import_hailo_platform",
        lambda: _fake_hailo_platform(state),
    )
    runtime.load(_make_compiled_model(tmp_path))
    collector = _AsyncOutcomeCollector()
    vendor_job_id = runtime.submit_async(
        {"input": np.zeros((1, 4, 4, 3), dtype=np.float32)},
        collector,
    )

    _submitted_async_job(state, vendor_job_id).callback(
        completion_info=SimpleNamespace(exception=RuntimeError("device failed"))
    )

    outcome = collector.wait_one()
    assert outcome.outputs is None
    assert outcome.error_type == "HailoRTAsyncError"
    assert "RuntimeError" in outcome.error_message
    with pytest.raises(RuntimeError, match="pipeline is unavailable"):
        runtime.submit_async(
            {"input": np.zeros((1, 4, 4, 3), dtype=np.float32)},
            collector,
        )
    runtime.unload()


@pytest.mark.parametrize(
    ("completion_info", "copy_fails", "expected_error_type"),
    [
        (None, False, "HailoRTAsyncProtocolError"),
        (SimpleNamespace(exception=None), True, "HailoRTAsyncCompletionError"),
    ],
)
def test_hailo_runtime_native_async_protocol_and_copy_failures_close_pipeline(
    tmp_path,
    monkeypatch,
    completion_info,
    copy_fails,
    expected_error_type,
):
    """Fails closed when callback integrity or framework-owned output is unknown."""
    state = {}
    runtime = HailoRuntime()
    monkeypatch.setattr(
        runtime,
        "_import_hailo_platform",
        lambda: _fake_hailo_platform(state),
    )
    runtime.load(_make_compiled_model(tmp_path))
    if copy_fails:
        monkeypatch.setattr(
            runtime,
            "_copy_async_output",
            lambda _value: (_ for _ in ()).throw(RuntimeError("copy failed")),
        )
    collector = _AsyncOutcomeCollector()
    vendor_job_id = runtime.submit_async(
        {"input": np.zeros((1, 4, 4, 3), dtype=np.float32)},
        collector,
    )
    _submitted_async_job(state, vendor_job_id).callback(
        completion_info=completion_info
    )

    outcome = collector.wait_one()
    assert outcome.error_type == expected_error_type
    with pytest.raises(RuntimeError, match="pipeline is unavailable"):
        runtime.submit_async(
            {"input": np.zeros((1, 4, 4, 3), dtype=np.float32)},
            collector,
        )
    runtime.unload()


def test_hailo_runtime_native_async_ready_failure_is_recoverable(
    tmp_path, monkeypatch
):
    """A readiness timeout occurs before run_async and must not poison the pipeline."""
    state = {"async_ready_error": TimeoutError("queue busy")}
    runtime = HailoRuntime()
    monkeypatch.setattr(
        runtime,
        "_import_hailo_platform",
        lambda: _fake_hailo_platform(state),
    )
    runtime.load(_make_compiled_model(tmp_path))
    first = _AsyncOutcomeCollector()

    runtime.submit_async(
        {"input": np.zeros((1, 4, 4, 3), dtype=np.float32)},
        first,
    )
    assert first.wait_one().error_type == "HailoRTAsyncReadyError"

    second = _AsyncOutcomeCollector()
    state["async_job_submitted"].clear()
    runtime.submit_async(
        {"input": np.zeros((1, 4, 4, 3), dtype=np.float32)},
        second,
    )
    _latest_submitted_async_job(state).callback(
        completion_info=SimpleNamespace(exception=None)
    )
    assert second.wait_one().error_type is None
    runtime.unload()


def test_hailo_runtime_native_async_binding_failure_closes_pipeline(
    tmp_path, monkeypatch
):
    """Separates persistent binding/API faults from recoverable ready timeout."""
    state = {"async_binding_error": RuntimeError("binding failed")}
    runtime = HailoRuntime()
    monkeypatch.setattr(
        runtime,
        "_import_hailo_platform",
        lambda: _fake_hailo_platform(state),
    )
    runtime.load(_make_compiled_model(tmp_path))
    collector = _AsyncOutcomeCollector()

    runtime.submit_async(
        {"input": np.zeros((1, 4, 4, 3), dtype=np.float32)},
        collector,
    )

    assert collector.wait_one().error_type == "HailoRTAsyncBindingError"
    with pytest.raises(RuntimeError, match="pipeline is unavailable"):
        runtime.submit_async(
            {"input": np.zeros((1, 4, 4, 3), dtype=np.float32)},
            collector,
        )
    runtime.unload()


def test_hailo_runtime_native_async_run_failure_closes_pipeline(
    tmp_path, monkeypatch
):
    """A synchronous run_async failure follows Hailo's pipeline-shutdown rule."""
    state = {"async_run_error": RuntimeError("submit failed")}
    runtime = HailoRuntime()
    monkeypatch.setattr(
        runtime,
        "_import_hailo_platform",
        lambda: _fake_hailo_platform(state),
    )
    runtime.load(_make_compiled_model(tmp_path))
    collector = _AsyncOutcomeCollector()

    runtime.submit_async(
        {"input": np.zeros((1, 4, 4, 3), dtype=np.float32)},
        collector,
    )
    assert collector.wait_one().error_type == "HailoRTAsyncSubmitError"
    with pytest.raises(RuntimeError, match="pipeline is unavailable"):
        runtime.submit_async(
            {"input": np.zeros((1, 4, 4, 3), dtype=np.float32)},
            collector,
        )
    runtime.unload()


def test_hailo_runtime_native_async_completes_multiple_jobs_out_of_order(
    tmp_path, monkeypatch
):
    """Keeps each callback associated with its own bindings under reordering."""
    state = {}
    runtime = HailoRuntime(async_queue_size=2)
    monkeypatch.setattr(
        runtime,
        "_import_hailo_platform",
        lambda: _fake_hailo_platform(state, async_queue_size=2),
    )
    runtime.load(_make_compiled_model(tmp_path))
    first = _AsyncOutcomeCollector()
    second = _AsyncOutcomeCollector()

    runtime.submit_async(
        {"input": np.zeros((1, 4, 4, 3), dtype=np.float32)},
        first,
    )
    runtime.submit_async(
        {"input": np.ones((1, 4, 4, 3), dtype=np.float32)},
        second,
    )
    jobs = _wait_for_async_job_count(state, 2)
    jobs[1].bindings[0]._output_buffers["output"].fill(11)
    jobs[2].bindings[0]._output_buffers["output"].fill(22)

    jobs[2].callback(completion_info=SimpleNamespace(exception=None))
    jobs[1].callback(completion_info=SimpleNamespace(exception=None))

    np.testing.assert_array_equal(
        first.wait_one().outputs["output"],
        np.full((1, 10), 11),
    )
    np.testing.assert_array_equal(
        second.wait_one().outputs["output"],
        np.full((1, 10), 22),
    )
    runtime.unload()


def test_hailo_runtime_native_async_bounds_adapter_submission_queue(
    tmp_path, monkeypatch
):
    """Prevents direct adapter callers from growing executor queues without bound."""
    state = {
        "async_ready_entered": threading.Event(),
        "async_ready_release": threading.Event(),
    }
    runtime = HailoRuntime()
    monkeypatch.setattr(
        runtime,
        "_import_hailo_platform",
        lambda: _fake_hailo_platform(state, async_queue_size=2),
    )
    runtime.load(_make_compiled_model(tmp_path))
    collectors = [_AsyncOutcomeCollector(), _AsyncOutcomeCollector()]

    runtime.submit_async(
        {"input": np.zeros((1, 4, 4, 3), dtype=np.float32)},
        collectors[0],
    )
    assert state["async_ready_entered"].wait(timeout=1.0)
    runtime.submit_async(
        {"input": np.ones((1, 4, 4, 3), dtype=np.float32)},
        collectors[1],
    )
    with pytest.raises(RuntimeError, match="queue capacity"):
        runtime.submit_async(
            {"input": np.full((1, 4, 4, 3), 2, dtype=np.float32)},
            _AsyncOutcomeCollector(),
        )

    state["async_ready_release"].set()
    for job in _wait_for_async_job_count(state, 2).values():
        job.callback(completion_info=SimpleNamespace(exception=None))
    assert collectors[0].wait_one().error_type is None
    assert collectors[1].wait_one().error_type is None
    runtime.unload()


def test_hailo_runtime_native_async_accepts_inline_sdk_completion(
    tmp_path, monkeypatch
):
    """Handles a callback that fires before run_async returns its job handle."""
    state = {"async_inline_completion": True}
    runtime = HailoRuntime()
    monkeypatch.setattr(
        runtime,
        "_import_hailo_platform",
        lambda: _fake_hailo_platform(state),
    )
    runtime.load(_make_compiled_model(tmp_path))
    collector = _AsyncOutcomeCollector()

    vendor_job_id = runtime.submit_async(
        {"input": np.zeros((1, 4, 4, 3), dtype=np.float32)},
        collector,
    )

    assert vendor_job_id == 1
    assert collector.wait_one().error_type is None
    runtime.unload()


def test_hailo_runtime_legacy_vstreams_do_not_claim_native_async(
    tmp_path, monkeypatch
):
    """Catches routing legacy InferVStreams through a callback API it lacks."""
    state = {}
    runtime = HailoRuntime()
    monkeypatch.setattr(
        runtime,
        "_import_hailo_platform",
        lambda: _fake_hailo_platform(state, supports_infer_model_api=False),
    )
    runtime.load(_make_compiled_model(tmp_path))

    assert runtime.supports_native_async() is False
    assert runtime.max_concurrent_workers() == 1
    with pytest.raises(NotImplementedError, match="InferModel"):
        runtime.submit_async({}, lambda _outcome: None)
    runtime.unload()


def test_hailo_runtime_runs_through_native_async_executor(tmp_path, monkeypatch):
    """Catches an adapter callback shape that the framework executor cannot consume."""
    state = {}
    runtime = HailoRuntime(async_completion_timeout_ms=1000)
    monkeypatch.setattr(
        runtime,
        "_import_hailo_platform",
        lambda: _fake_hailo_platform(state, async_queue_size=2),
    )
    runtime.load(_make_compiled_model(tmp_path))
    executor = NativeAsyncRuntimeExecutor(
        runtime,
        max_inflight=2,
        completion_timeout_sec=runtime.native_async_completion_timeout_sec(),
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            executor.execute,
            {"input": np.zeros((1, 4, 4, 3), dtype=np.float32)},
        )
        job = _submitted_async_job(state, 1)
        job.bindings[0]._output_buffers["output"].fill(77)
        job.callback(completion_info=SimpleNamespace(exception=None))
        execution = future.result(timeout=1.0)

    np.testing.assert_array_equal(execution.outputs["output"], np.full((1, 10), 77))
    assert execution.vendor_job_id == 1
    executor.acknowledge(execution)
    assert executor.shutdown(timeout=0.0) is True
    runtime.unload()
