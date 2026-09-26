from types import SimpleNamespace

import numpy as np
import pytest

from core.compiled_model import CompiledModel
from core.model_spec import Model_Spec, Task
from runtimes.mobilint_rt import MobilintRuntime
from runtimes.mobilint_ttm_r2 import (
    MOBILINT_TTM_R2_ADAPTER_ID,
    MobilintTTMR2Adapter,
)


def _scale(**overrides):
    values = {
        "scale": 0.0,
        "is_uniform": False,
        "scale_list": [1.0] * 64,
        "zero_point": 0,
        "is_asymmetric": False,
        "zero_points": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeModel:
    def __init__(
        self,
        *,
        input_shape=(1, 8, 64),
        input_dtype=np.int8,
        output_shape=(1, 1, 96),
        scale=None,
        output=None,
    ):
        self.input_shape = input_shape
        self.input_dtype = input_dtype
        self.output_shape = output_shape
        self.scale = scale or _scale()
        self.output = (
            np.arange(96, dtype=np.float32).reshape(1, 1, 96)
            if output is None
            else output
        )
        self.infer_calls = 0
        self.infer_to_float_calls = 0
        self.disposed = 0

    def get_model_input_shape(self):
        return [self.input_shape]

    def get_model_input_data_type(self):
        return self.input_dtype

    def get_model_output_shape(self):
        return [self.output_shape]

    def get_input_scale(self):
        return [self.scale]

    def infer(self, inputs):
        self.infer_calls += 1
        return [self.output]

    def infer_to_float(self, inputs):
        self.infer_to_float_calls += 1
        self.last_inputs = inputs
        return [self.output]

    def launch(self, accelerator):
        self.accelerator = accelerator

    def dispose(self):
        self.disposed += 1


@pytest.fixture
def fake_model():
    return _FakeModel()


def test_adapter_uses_infer_to_float_and_restores_forecast(fake_model):
    adapter = MobilintTTMR2Adapter()
    adapter.bind(fake_model)

    result = adapter.run(
        fake_model,
        {"past_values": np.zeros((1, 512, 1), dtype=np.float32)},
    )

    assert fake_model.infer_to_float_calls == 1
    assert fake_model.infer_calls == 0
    assert fake_model.last_inputs[0].shape == (1, 8, 64)
    assert fake_model.last_inputs[0].dtype == np.int8
    assert result["forecast"].shape == (1, 96, 1)
    assert result["forecast"].dtype == np.float32
    assert result["forecast"].flags.c_contiguous
    assert result["forecast"][0, 95, 0] == 95.0


def test_adapter_accepts_float32_logical_dtype_but_quantizes_raw_input():
    model = _FakeModel(input_dtype=np.float32)
    adapter = MobilintTTMR2Adapter()
    adapter.bind(model)

    adapter.run(
        model,
        {"past_values": np.zeros((1, 512, 1), dtype=np.float32)},
    )

    assert model.last_inputs[0].dtype == np.int8


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (_FakeModel(scale=_scale(is_asymmetric=True)), "symmetric"),
        (_FakeModel(scale=_scale(zero_point=3)), "zero point 0"),
        (_FakeModel(input_shape=(1, 512, 1)), "input shape"),
        (_FakeModel(output_shape=(1, 96, 1)), "output shape"),
    ],
)
def test_adapter_rejects_unvalidated_artifact_abi(model, message):
    with pytest.raises(ValueError, match=message):
        MobilintTTMR2Adapter().bind(model)


def test_adapter_rejects_wrong_semantic_input_shape(fake_model):
    adapter = MobilintTTMR2Adapter()
    adapter.bind(fake_model)

    with pytest.raises(ValueError, match="semantic input"):
        adapter.run(
            fake_model,
            {"past_values": np.zeros((1, 511, 1), dtype=np.float32)},
        )


def test_adapter_rejects_nonfinite_device_output():
    model = _FakeModel(
        output=np.full((1, 1, 96), np.nan, dtype=np.float32)
    )
    adapter = MobilintTTMR2Adapter()
    adapter.bind(model)

    with pytest.raises(ValueError, match="non-finite"):
        adapter.run(
            model,
            {"past_values": np.zeros((1, 512, 1), dtype=np.float32)},
        )


def _compiled_model(tmp_path):
    artifact = tmp_path / "ttm-r2-core.mxq"
    artifact.write_bytes(b"mxq")
    spec = Model_Spec(
        name="ttm-r2",
        task=Task.TIME_SERIES_FORECASTING,
        input_shapes={"past_values": (1, 512, 1)},
        input_dtype={"past_values": "float32"},
        output_shapes={"forecast": (1, 96, 1)},
        model_paths={"mxq": str(artifact)},
    )
    return CompiledModel(spec, "mobilint", artifact)


def _load_runtime(monkeypatch, tmp_path, model):
    class DeviceSession:
        def __init__(self, device_id, expected_family):
            self.device_id = device_id
            self.expected_family = expected_family

        def acquire(self):
            return SimpleNamespace(
                device_id=self.device_id,
                device_type=1,
                family=self.expected_family,
                validation_source="test",
            )

        def release(self):
            pass

    class ModelConfig:
        pass

    class QBRuntime:
        pass

    QBRuntime.__version__ = "test"
    QBRuntime.ModelConfig = ModelConfig
    QBRuntime.Accelerator = lambda device_id: SimpleNamespace(
        device_id=device_id
    )
    QBRuntime.Model = lambda path, config: model

    monkeypatch.setattr(
        "runtimes.mobilint_rt.MobilintDeviceSession", DeviceSession
    )
    monkeypatch.setattr(
        MobilintRuntime, "_load_qbruntime", staticmethod(lambda: QBRuntime)
    )
    runtime = MobilintRuntime(
        expected_family="aries",
        tensor_boundary_adapter_id=MOBILINT_TTM_R2_ADAPTER_ID,
    )
    runtime.load(_compiled_model(tmp_path))
    return runtime


def test_runtime_warmup_does_not_contaminate_measured_saturation(
    monkeypatch, tmp_path
):
    model = _FakeModel(scale=_scale(scale_list=[100.0] * 64))
    runtime = _load_runtime(monkeypatch, tmp_path, model)
    saturating_input = {
        "past_values": np.full((1, 512, 1), 2.0, dtype=np.float32)
    }

    runtime.warmup(saturating_input, num_runs=2)
    assert runtime.get_device_spec()["mobilint_saturation_elements"] == 0
    assert runtime.get_device_spec()["mobilint_saturation_total"] == 0

    runtime.run(saturating_input)

    spec = runtime.get_device_spec()
    assert spec["tensor_boundary_adapter_id"] == MOBILINT_TTM_R2_ADAPTER_ID
    assert spec["mobilint_quantization_status"] == "saturated"
    assert spec["mobilint_saturation_elements"] == 512
    assert spec["mobilint_saturation_total"] == 512
    assert model.infer_calls == 0
    assert model.infer_to_float_calls == 3


def test_runtime_rejects_unknown_tensor_boundary_adapter():
    with pytest.raises(ValueError, match="tensor_boundary_adapter_id"):
        MobilintRuntime(
            expected_family="aries",
            tensor_boundary_adapter_id="unknown",
        )
