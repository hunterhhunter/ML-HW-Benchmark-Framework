from typing import Any, Dict

import numpy as np

from core.compiled_model import CompiledModel
from .base import Runtime


class MockNpuRuntime(Runtime):
    """
    SDK 없이 NPU runtime plugin 계약을 검증하기 위한 runtime.

    실제 벤더 runtime adapter는 load/run/warmup/unload와 numpy dict 출력 계약을
    이 클래스와 같은 형태로 맞추면 BenchmarkRunner를 그대로 사용할 수 있다.
    """

    def __init__(self, **runtime_options):
        self.device = runtime_options.get("device", "npu0")
        self.runtime_options = runtime_options
        self.compiled_model: CompiledModel | None = None
        self._loaded = False

    def load(self, compiled_model: CompiledModel) -> None:
        if not self.is_compatible(compiled_model):
            raise ValueError(f"Incompatible mock NPU artifact: {compiled_model.artifact_path}")
        self.compiled_model = compiled_model
        self._loaded = True

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self._loaded or self.compiled_model is None:
            raise RuntimeError("MockNpuRuntime is not loaded. Call load() first.")

        batch_size = self._infer_batch_size(inputs)
        outputs: Dict[str, np.ndarray] = {}
        for name, shape in self.compiled_model.spec.output_shapes.items():
            out_shape = tuple(shape)
            if out_shape:
                out_shape = (batch_size, *out_shape[1:])
            outputs[name] = np.zeros(out_shape, dtype=np.float32)
        return outputs

    def warmup(self, inputs: Dict[str, np.ndarray], num_runs: int = 1) -> None:
        for _ in range(num_runs):
            self.run(inputs)

    def unload(self) -> None:
        self.compiled_model = None
        self._loaded = False

    def get_device_spec(self) -> Dict[str, Any]:
        return {
            "backend": "mock_npu",
            "device": self.device,
            "accelerator_vendor": "MockNPU",
            "accelerator_name": "Mock NPU PCIe Adapter",
            "runtime_options": self.runtime_options,
        }

    def is_compatible(self, compiled_model: CompiledModel) -> bool:
        backend_match = "mock_npu" in compiled_model.backend_name.lower()
        suffix_match = str(compiled_model.artifact_path).endswith(".mockbin")
        return backend_match or suffix_match

    def _infer_batch_size(self, inputs: Dict[str, np.ndarray]) -> int:
        if not inputs:
            return 1
        first = next(iter(inputs.values()))
        if hasattr(first, "shape") and first.shape:
            return int(first.shape[0])
        return 1
