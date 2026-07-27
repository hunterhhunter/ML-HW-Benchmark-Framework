import contextlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from core.compiled_model import CompiledModel
from core.model_spec import Model_Spec, Task
import runtimes.furiosa_torch_rt as runtime_module
from runtimes.furiosa_torch_rt import FuriosaTorchRuntime


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeModel:
    def __init__(self):
        self.eval_calls = 0
        self.devices = []
        self.calls = []

    def eval(self):
        self.eval_calls += 1
        return self

    def to(self, device):
        self.devices.append(device)
        return self

    def __call__(self, *inputs):
        self.calls.append(inputs)
        return (
            _FakeTensor([[0.25, 0.75]]),
            _FakeTensor([[1.0, 2.0, 3.0, 4.0]]),
        )


def _compiled_model(tmp_path: Path, *, name="two-input", backend="furiosa_torch"):
    artifact = tmp_path / "model"
    artifact.mkdir()
    spec = Model_Spec(
        name=name,
        task=Task.NLP_CLASSIFICATION,
        input_shapes={"input_ids": (1, 4), "attention_mask": (1, 4)},
        input_dtype={"input_ids": "int64", "attention_mask": "int64"},
        output_shapes={"logits": (1, 2), "hidden": (1, 4)},
        model_paths={"pytorch_model": str(artifact)},
    )
    return CompiledModel(spec, backend, artifact)


def _install_fake_sdk(monkeypatch):
    state = {
        "compiler_configs": [],
        "backend_calls": [],
        "compile_calls": [],
        "as_tensor_calls": [],
    }

    class TacticHintConfig:
        Default = object()

    class CompilerConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            state["compiler_configs"].append(self)

    class Backend:
        def with_config(self, config, **kwargs):
            state["backend_calls"].append((config, kwargs))
            return "strict-furiosa-backend"

    torch_module = ModuleType("torch")
    torch_module.device = lambda value: f"device:{value}"

    def as_tensor(value):
        tensor = _FakeTensor(value)
        state["as_tensor_calls"].append(tensor)
        return tensor

    torch_module.as_tensor = as_tensor
    torch_module.inference_mode = contextlib.nullcontext

    def compile_model(model, **kwargs):
        state["compile_calls"].append((model, kwargs))
        return model

    torch_module.compile = compile_model

    furiosa_module = ModuleType("furiosa")
    furiosa_torch_module = ModuleType("furiosa.torch")
    furiosa_torch_module.backend = Backend()
    config_module = ModuleType("furiosa.torch.config")
    config_module.CompilerConfig = CompilerConfig
    config_module.TacticHintConfig = TacticHintConfig
    furiosa_module.torch = furiosa_torch_module

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "furiosa", furiosa_module)
    monkeypatch.setitem(sys.modules, "furiosa.torch", furiosa_torch_module)
    monkeypatch.setitem(sys.modules, "furiosa.torch.config", config_module)
    return state, TacticHintConfig


def test_load_builds_strict_static_fullgraph_backend(monkeypatch, tmp_path):
    state, tactic_hints = _install_fake_sdk(monkeypatch)
    model = _FakeModel()
    adapter = SimpleNamespace(
        input_names=("input_ids", "attention_mask"),
        output_names=("logits", "hidden"),
        tactic_hint="Default",
        loader=lambda path: model,
    )
    monkeypatch.setattr(runtime_module, "get_torch_model_adapter", lambda name: adapter)
    compiled_model = _compiled_model(tmp_path)
    runtime = FuriosaTorchRuntime(device="npu:0")

    runtime.load(compiled_model)

    assert model.eval_calls == 1
    assert model.devices == ["device:furiosa:0"]
    assert state["compiler_configs"][0].kwargs == {
        "tactic_hint": tactic_hints.Default
    }
    assert state["backend_calls"] == [
        (state["compiler_configs"][0], {"eager_fallback": False})
    ]
    assert state["compile_calls"] == [
        (
            model,
            {
                "backend": "strict-furiosa-backend",
                "fullgraph": True,
                "dynamic": False,
            },
        )
    ]
    assert runtime.compiled_model is compiled_model


def test_run_orders_inputs_and_returns_numpy_outputs(monkeypatch, tmp_path):
    state, _ = _install_fake_sdk(monkeypatch)
    model = _FakeModel()
    adapter = SimpleNamespace(
        input_names=("input_ids", "attention_mask"),
        output_names=("logits", "hidden"),
        tactic_hint="Default",
        loader=lambda path: model,
    )
    monkeypatch.setattr(runtime_module, "get_torch_model_adapter", lambda name: adapter)
    runtime = FuriosaTorchRuntime(device="npu:0")
    runtime.load(_compiled_model(tmp_path))

    outputs = runtime.run(
        {
            "attention_mask": np.ones((1, 4), dtype=np.int64),
            "input_ids": np.arange(4, dtype=np.int64).reshape(1, 4),
        }
    )

    assert list(outputs) == ["logits", "hidden"]
    np.testing.assert_array_equal(outputs["logits"], [[0.25, 0.75]])
    np.testing.assert_array_equal(outputs["hidden"], [[1.0, 2.0, 3.0, 4.0]])
    assert [tensor.value.tolist() for tensor in model.calls[0]] == [
        [[0, 1, 2, 3]],
        [[1, 1, 1, 1]],
    ]
    assert all(tensor.devices == ["device:furiosa:0"] for tensor in state["as_tensor_calls"])


def test_run_fails_closed_on_input_contract_mismatch(monkeypatch, tmp_path):
    _install_fake_sdk(monkeypatch)
    adapter = SimpleNamespace(
        input_names=("input_ids", "attention_mask"),
        output_names=("logits", "hidden"),
        tactic_hint="Default",
        loader=lambda path: _FakeModel(),
    )
    monkeypatch.setattr(runtime_module, "get_torch_model_adapter", lambda name: adapter)
    runtime = FuriosaTorchRuntime()
    runtime.load(_compiled_model(tmp_path))

    with pytest.raises(ValueError, match="input contract mismatch"):
        runtime.run({"input_ids": np.ones((1, 4), dtype=np.int64)})


def test_run_fails_closed_on_static_shape_mismatch(monkeypatch, tmp_path):
    _install_fake_sdk(monkeypatch)
    adapter = SimpleNamespace(
        input_names=("input_ids", "attention_mask"),
        output_names=("logits", "hidden"),
        tactic_hint="Default",
        loader=lambda path: _FakeModel(),
    )
    monkeypatch.setattr(runtime_module, "get_torch_model_adapter", lambda name: adapter)
    runtime = FuriosaTorchRuntime()
    runtime.load(_compiled_model(tmp_path))

    with pytest.raises(ValueError, match="static shape mismatch"):
        runtime.run(
            {
                "input_ids": np.ones((2, 4), dtype=np.int64),
                "attention_mask": np.ones((2, 4), dtype=np.int64),
            }
        )


def test_run_fails_closed_on_static_output_shape_mismatch(monkeypatch, tmp_path):
    _install_fake_sdk(monkeypatch)
    adapter = SimpleNamespace(
        input_names=("input_ids", "attention_mask"),
        output_names=("logits", "hidden"),
        tactic_hint="Default",
        loader=lambda path: _FakeModel(),
    )
    monkeypatch.setattr(runtime_module, "get_torch_model_adapter", lambda name: adapter)
    runtime = FuriosaTorchRuntime()
    runtime.load(_compiled_model(tmp_path))
    runtime._compiled = lambda *inputs: (
        _FakeTensor([[0.25, 0.75, 1.0]]),
        _FakeTensor([[1.0, 2.0, 3.0, 4.0]]),
    )

    with pytest.raises(ValueError, match="static output shape mismatch"):
        runtime.run(
            {
                "input_ids": np.ones((1, 4), dtype=np.int64),
                "attention_mask": np.ones((1, 4), dtype=np.int64),
            }
        )


def test_runtime_requires_load_and_exposes_single_worker_contract():
    runtime = FuriosaTorchRuntime(device="npu:0")

    with pytest.raises(RuntimeError, match="not loaded"):
        runtime.run({})

    assert runtime.max_concurrent_workers() == 1
    assert runtime.supports_dynamic_batching() is False
    assert runtime.max_dynamic_batch_size() == 1
    assert runtime.get_device_spec()["device"] == "furiosa:0"


def test_is_compatible_rejects_wrong_backend_or_unknown_adapter(monkeypatch, tmp_path):
    compiled_model = _compiled_model(tmp_path, backend="onnxruntime")
    runtime = FuriosaTorchRuntime()

    assert runtime.is_compatible(compiled_model) is False

    compatible_backend = CompiledModel(
        compiled_model.spec,
        "furiosa_torch",
        compiled_model.artifact_path,
    )
    monkeypatch.setattr(
        runtime_module,
        "get_torch_model_adapter",
        lambda name: (_ for _ in ()).throw(ValueError("unknown")),
    )
    assert runtime.is_compatible(compatible_backend) is False


def test_unload_clears_model_state(monkeypatch, tmp_path):
    _install_fake_sdk(monkeypatch)
    adapter = SimpleNamespace(
        input_names=("input_ids", "attention_mask"),
        output_names=("logits", "hidden"),
        tactic_hint="Default",
        loader=lambda path: _FakeModel(),
    )
    monkeypatch.setattr(runtime_module, "get_torch_model_adapter", lambda name: adapter)
    runtime = FuriosaTorchRuntime()
    runtime.load(_compiled_model(tmp_path))

    runtime.unload()

    assert runtime.compiled_model is None
    with pytest.raises(RuntimeError, match="not loaded"):
        runtime.run({})
