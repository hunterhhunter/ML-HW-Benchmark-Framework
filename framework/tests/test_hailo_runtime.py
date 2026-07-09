from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.compiled_model import CompiledModel
from core.model_spec import Model_Spec, Task
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


def _fake_hailo_platform(state, *, activate_returns_none=False, supports_infer_model_api=True):
    class FakeHEF:
        def __init__(self, path):
            state["hef_path"] = path

        def get_network_group_names(self):
            return ["net"]

        def get_input_vstream_infos(self):
            return [_FakeVStreamInfo("input", (4, 4, 3))]

        def get_output_vstream_infos(self):
            return [_FakeVStreamInfo("output", (10,))]

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
            binding = FakeBindings(output_buffers or {"output": np.zeros((10,), dtype=np.uint8)})
            state.setdefault("bindings", []).append(binding)
            return binding

        def run(self, bindings, timeout_ms=1000):
            state["configured_infer_model_run"] = True
            state["run_timeout_ms"] = timeout_ms
            state["run_bindings_count"] = len(bindings)
            state["last_input_buffers"] = [binding._input_buffers for binding in bindings]

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
            self.outputs = [FakeInferTensor("output", (10,))]

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
