import sys
import types
from pathlib import Path

import numpy as np

from core.compiled_model import CompiledModel
from core.model_spec import Model_Spec, Task
from runtimes.deepx_rt import DeepXRuntime


class FakeDXRTState:
    def __init__(self):
        self.callback = None
        self.sync_calls = 0
        self.async_calls = []
        self.engines = []


def _install_fake_dx_engine(monkeypatch):
    state = FakeDXRTState()

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
            return ["input"]

        def get_output_tensor_names(self):
            return ["output"]

        def get_input_tensors_info(self):
            return [
                {
                    "name": "input",
                    "shape": [1, 3, 4, 4],
                    "dtype": np.dtype("float32"),
                    "elem_size": 4,
                }
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
            state.callback = callback

        def run(self, input_data):
            state.sync_calls += 1
            outputs = [np.asarray([[3.0, 4.0]], dtype=np.float32)]
            if state.callback is not None:
                state.callback(outputs, None)
            return outputs

        def run_async(self, input_data, user_arg=None, output_buffer=None):
            state.async_calls.append((input_data, user_arg, output_buffer))
            state.callback(
                [np.asarray([[7.0, 8.0]], dtype=np.float32)],
                user_arg,
            )
            return len(state.async_calls)

        def dispose(self):
            self.disposed = True

    fake_module = types.ModuleType("dx_engine")
    fake_module.__version__ = "3.3.2-test"
    fake_module.InferenceOption = FakeInferenceOption
    fake_module.InferenceEngine = FakeInferenceEngine
    monkeypatch.setitem(sys.modules, "dx_engine", fake_module)
    return state


def _compiled_model(tmp_path: Path) -> CompiledModel:
    artifact = tmp_path / "model.dxnn"
    artifact.write_bytes(b"DXNN-test")
    spec = Model_Spec(
        name="deepx-test",
        task=Task.IMAGE_CLASSIFICATION,
        input_shapes={"input": (1, 3, 4, 4)},
        input_dtype={"input": "float32"},
        output_shapes={"output": (1, 2)},
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
