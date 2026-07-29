import contextlib
from dataclasses import replace
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from core.compiled_model import CompiledModel
from core.model_spec import Model_Spec, Task
import runtimes.furiosa_torch_rt as runtime_module
from runtimes.furiosa_torch_rt import FuriosaTorchRuntime
from runtimes.furiosa_torch_models import get_torch_model_adapter


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
        return _FakeTensor([[0.25, 0.75]])


def _compiled_model(tmp_path: Path, *, backend="furiosa_torch"):
    artifact = tmp_path / "textattack_bert-base-uncased-SST-2"
    artifact.mkdir()
    (artifact / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bert",
                "architectures": ["BertForSequenceClassification"],
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
                "vocab_size": 30522,
                "max_position_embeddings": 512,
                "id2label": {"0": "LABEL_0", "1": "LABEL_1"},
            }
        )
    )
    (artifact / "model.safetensors").touch()
    spec = Model_Spec(
        name="bert-base-uncased",
        task=Task.NLP_CLASSIFICATION,
        input_shapes={"input_ids": (1, 128), "attention_mask": (1, 128)},
        input_dtype={"input_ids": "int64", "attention_mask": "int64"},
        output_shapes={"logits": (1, 2)},
        model_paths={"pytorch_model": str(artifact)},
    )
    return CompiledModel(spec, backend, artifact)


def _install_fake_sdk(monkeypatch):
    state = {
        "compiler_configs": [],
        "backend_calls": [],
        "compile_calls": [],
        "as_tensor_calls": [],
        "as_tensor_inputs": [],
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
        state["as_tensor_inputs"].append(value)
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
    adapter = replace(
        get_torch_model_adapter("bert-base-uncased"),
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


def test_run_orders_inputs_and_returns_numpy_outputs(monkeypatch, tmp_path):
    state, _ = _install_fake_sdk(monkeypatch)
    model = _FakeModel()
    adapter = replace(
        get_torch_model_adapter("bert-base-uncased"),
        loader=lambda path: model,
    )
    monkeypatch.setattr(runtime_module, "get_torch_model_adapter", lambda name: adapter)
    runtime = FuriosaTorchRuntime(device="npu:0")
    runtime.load(_compiled_model(tmp_path))

    outputs = runtime.run(
        {
            "attention_mask": np.ones((1, 128), dtype=np.int64),
            "input_ids": np.arange(128, dtype=np.int64).reshape(1, 128),
        }
    )

    assert list(outputs) == ["logits"]
    np.testing.assert_array_equal(outputs["logits"], [[0.25, 0.75]])
    assert [tensor.value.tolist() for tensor in model.calls[0]] == [
        [list(range(128))],
        [[1] * 128],
    ]
    assert all(tensor.devices == ["device:furiosa:0"] for tensor in state["as_tensor_calls"])


def test_run_fails_closed_on_input_shape_and_dtype_mismatch(monkeypatch, tmp_path):
    _install_fake_sdk(monkeypatch)
    adapter = replace(
        get_torch_model_adapter("bert-base-uncased"),
        loader=lambda path: _FakeModel(),
    )
    monkeypatch.setattr(runtime_module, "get_torch_model_adapter", lambda name: adapter)
    runtime = FuriosaTorchRuntime()
    runtime.load(_compiled_model(tmp_path))

    with pytest.raises(ValueError, match="static shape mismatch"):
        runtime.run(
            {
                "input_ids": np.ones((2, 128), dtype=np.int64),
                "attention_mask": np.ones((2, 128), dtype=np.int64),
            }
        )


def test_run_copies_only_read_only_numpy_inputs(monkeypatch, tmp_path):
    state, _ = _install_fake_sdk(monkeypatch)
    adapter = replace(
        get_torch_model_adapter("bert-base-uncased"),
        loader=lambda path: _FakeModel(),
    )
    monkeypatch.setattr(runtime_module, "get_torch_model_adapter", lambda name: adapter)
    runtime = FuriosaTorchRuntime()
    runtime.load(_compiled_model(tmp_path))
    input_ids = np.arange(128, dtype=np.int64).reshape(1, 128)
    attention_mask = np.ones((1, 128), dtype=np.int64)
    input_ids.setflags(write=False)
    attention_mask.setflags(write=False)

    runtime.run(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    )

    assert all(array.flags.writeable for array in state["as_tensor_inputs"])
    with pytest.raises(ValueError, match="dtype mismatch"):
        runtime.run(
            {
                "input_ids": np.ones((1, 128), dtype=np.float32),
                "attention_mask": np.ones((1, 128), dtype=np.int64),
            }
        )


def test_runtime_exposes_single_worker_static_contract():
    runtime = FuriosaTorchRuntime(device="npu:0")

    with pytest.raises(RuntimeError, match="not loaded"):
        runtime.run({})

    assert runtime.max_concurrent_workers() == 1
    assert runtime.supports_dynamic_batching() is False
    assert runtime.max_dynamic_batch_size() == 1
    assert runtime.get_device_spec()["device"] == "furiosa:0"


def test_is_compatible_rejects_wrong_backend(monkeypatch, tmp_path):
    runtime = FuriosaTorchRuntime()

    assert runtime.is_compatible(_compiled_model(tmp_path, backend="onnxruntime")) is False


def test_is_compatible_rejects_non_verified_model_spec(tmp_path):
    compiled_model = _compiled_model(tmp_path)
    invalid_spec = Model_Spec(
        name=compiled_model.spec.name,
        task=compiled_model.spec.task,
        input_shapes={"input_ids": (1, 64), "attention_mask": (1, 64)},
        input_dtype=compiled_model.spec.input_dtype,
        output_shapes=compiled_model.spec.output_shapes,
        model_paths=compiled_model.spec.model_paths,
    )
    invalid = CompiledModel(
        invalid_spec,
        compiled_model.backend_name,
        compiled_model.artifact_path,
    )

    assert FuriosaTorchRuntime().is_compatible(invalid) is False
