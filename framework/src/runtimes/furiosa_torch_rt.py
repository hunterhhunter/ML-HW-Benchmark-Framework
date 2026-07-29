"""Strict PyTorch-to-RNGD runtime for server-verified BERT models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from core.compiled_model import CompiledModel
from .base import Runtime
from .furiosa_torch_models import get_torch_model_adapter


_BACKEND_NAMES = frozenset(
    {"furiosa_torch", "furiosa-rngd-torch", "rngd_torch"}
)


def _normalize_device(value: Any) -> str:
    device = str(value or "npu:0").strip().lower()
    if device in {"0", "npu0", "npu:0", "furiosa", "furiosa:0"}:
        return "furiosa:0"
    if device.startswith("npu:"):
        return f"furiosa:{device.split(':', 1)[1]}"
    if device.startswith("furiosa:"):
        return device
    raise ValueError(
        "FuriosaTorchRuntime device must be npu:<index> or furiosa:<index>."
    )


class FuriosaTorchRuntime(Runtime):
    """Compile a supported local BERT model with no eager/CPU fallback."""

    def __init__(self, **runtime_options):
        self.runtime_options = dict(runtime_options)
        self.device = _normalize_device(runtime_options.get("device", "npu:0"))
        self.compiled_model: CompiledModel | None = None
        self._adapter = None
        self._model = None
        self._compiled = None
        self._torch_device = None

    def load(self, compiled_model: CompiledModel) -> None:
        adapter, model_path = self._validate_compiled_model(compiled_model)

        import torch
        import furiosa.torch
        from furiosa.torch.config import CompilerConfig, TacticHintConfig

        torch_device = torch.device(self.device)
        model = adapter.loader(model_path).eval().to(torch_device)

        try:
            tactic_hint = getattr(TacticHintConfig, adapter.tactic_hint)
        except AttributeError:
            raise ValueError(
                f"Unknown Furiosa tactic hint: {adapter.tactic_hint}"
            ) from None

        compiler_config = CompilerConfig(tactic_hint=tactic_hint)
        backend = furiosa.torch.backend.with_config(
            compiler_config,
            eager_fallback=False,
        )
        compiled = torch.compile(
            model,
            backend=backend,
            fullgraph=True,
            dynamic=False,
        )

        self.compiled_model = compiled_model
        self._adapter = adapter
        self._model = model
        self._compiled = compiled
        self._torch_device = torch_device

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self._compiled is None or self.compiled_model is None:
            raise RuntimeError("FuriosaTorchRuntime is not loaded. Call load() first.")

        expected_names = tuple(self._adapter.input_names)
        actual_names = tuple(inputs.keys())
        if set(actual_names) != set(expected_names):
            missing = sorted(set(expected_names) - set(actual_names))
            extra = sorted(set(actual_names) - set(expected_names))
            raise ValueError(
                "Furiosa Torch input contract mismatch: "
                f"missing={missing}, extra={extra}, expected={list(expected_names)}"
            )

        import torch

        ordered = []
        for name in expected_names:
            array = np.asarray(inputs[name])
            expected_shape = tuple(self.compiled_model.spec.input_shapes[name])
            if tuple(array.shape) != expected_shape:
                raise ValueError(
                    f"Furiosa Torch static shape mismatch for '{name}': "
                    f"expected {expected_shape}, got {tuple(array.shape)}"
                )
            expected_dtype = np.dtype(self.compiled_model.spec.input_dtype[name])
            if array.dtype != expected_dtype:
                raise ValueError(
                    f"Furiosa Torch dtype mismatch for '{name}': "
                    f"expected {expected_dtype}, got {array.dtype}"
                )
            if not array.flags.writeable:
                array = array.copy()
            ordered.append(torch.as_tensor(array).to(self._torch_device))

        with torch.inference_mode():
            raw_outputs = self._compiled(*ordered)
        return self._normalize_outputs(raw_outputs)

    def _normalize_outputs(self, value: Any) -> Dict[str, np.ndarray]:
        names = tuple(self._adapter.output_names)
        if isinstance(value, Mapping):
            if set(value.keys()) != set(names):
                raise ValueError(
                    "Furiosa Torch output contract mismatch: "
                    f"expected={list(names)}, actual={list(value.keys())}"
                )
            values = [value[name] for name in names]
        elif isinstance(value, (tuple, list)):
            values = list(value)
        else:
            values = [value]

        if len(values) != len(names):
            raise ValueError(
                "Furiosa Torch output contract mismatch: "
                f"expected {len(names)} outputs, got {len(values)}"
            )
        arrays = [self._to_numpy(output) for output in values]
        expected_shapes = tuple(self.compiled_model.spec.output_shapes.values())
        if len(expected_shapes) != len(arrays):
            raise ValueError(
                "Furiosa Torch output spec mismatch: "
                f"spec declares {len(expected_shapes)} outputs, adapter returns "
                f"{len(arrays)}"
            )
        for name, array, expected_shape in zip(names, arrays, expected_shapes):
            if tuple(array.shape) != tuple(expected_shape):
                raise ValueError(
                    f"Furiosa Torch static output shape mismatch for '{name}': "
                    f"expected {tuple(expected_shape)}, got {tuple(array.shape)}"
                )
        return dict(zip(names, arrays))

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def warmup(self, inputs: Dict[str, np.ndarray], num_runs: int = 1) -> None:
        if num_runs < 0:
            raise ValueError("num_runs must be non-negative")
        for _ in range(num_runs):
            self.run(inputs)

    def unload(self) -> None:
        self._compiled = None
        self._model = None
        self._adapter = None
        self._torch_device = None
        self.compiled_model = None

    def get_device_spec(self) -> Dict[str, Any]:
        return {
            "backend": "furiosa_torch",
            "device": self.device,
            "accelerator_vendor": "FuriosaAI",
            "accelerator_name": "RNGD",
            "strict_fullgraph": True,
            "eager_fallback": False,
        }

    def is_compatible(self, compiled_model: CompiledModel) -> bool:
        try:
            self._validate_compiled_model(compiled_model)
        except (AttributeError, OSError, TypeError, ValueError):
            return False
        return True

    @staticmethod
    def _validate_compiled_model(compiled_model: CompiledModel):
        backend_name = str(compiled_model.backend_name).lower()
        if backend_name not in _BACKEND_NAMES:
            raise ValueError(
                f"Unsupported Furiosa Torch backend: {compiled_model.backend_name!r}"
            )
        if compiled_model.artifact_path is None:
            raise ValueError("Furiosa Torch requires a local model directory.")

        adapter = get_torch_model_adapter(compiled_model.spec.name)
        adapter.validate_spec(compiled_model.spec)
        model_path = Path(compiled_model.artifact_path)
        adapter.validate_source(model_path)
        return adapter, model_path

    def max_concurrent_workers(self) -> int:
        return 1

    def supports_dynamic_batching(self) -> bool:
        return False

    def max_dynamic_batch_size(self):
        return 1


__all__ = ["FuriosaTorchRuntime"]
