"""
Benchmark target registry.

TargetSpec는 한 측정 대상의 runtime, compiler, monitor, artifact format을 묶는
상위 계약이다. 벤더별 SDK adapter는 각 registry(Runtime/Compiler/Collector)에
등록하고, 여기에서 target_id로 조합한다.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    label: str
    runtime_name: str
    device: str
    compiler_name: Optional[str] = None
    monitor_names: tuple[str, ...] = ("system",)
    artifact_format: str = "onnx"
    accelerator_vendor: str = ""
    accelerator_name: str = ""
    device_selector: str = ""
    capabilities: tuple[str, ...] = ()
    runtime_options: Dict[str, Any] = field(default_factory=dict)
    compiler_options: Dict[str, Any] = field(default_factory=dict)
    monitor_options: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    description: str = ""

    @property
    def uses_compiler(self) -> bool:
        return bool(self.compiler_name)

    def to_response(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "label": self.label,
            "runtime_name": self.runtime_name,
            "device": self.device,
            "compiler_name": self.compiler_name,
            "monitor_names": list(self.monitor_names),
            "artifact_format": self.artifact_format,
            "accelerator_vendor": self.accelerator_vendor,
            "accelerator_name": self.accelerator_name,
            "device_selector": self.device_selector,
            "capabilities": list(self.capabilities),
            "description": self.description,
        }


_TARGET_REGISTRY: Dict[str, TargetSpec] = {}


def register_target(target: TargetSpec) -> None:
    _TARGET_REGISTRY[target.target_id] = target


def get_target(target_id: str) -> TargetSpec:
    try:
        return _TARGET_REGISTRY[target_id]
    except KeyError:
        supported = sorted(_TARGET_REGISTRY.keys())
        raise ValueError(f"Unknown target '{target_id}'. Available targets: {supported}")


def list_targets() -> list[TargetSpec]:
    return list(_TARGET_REGISTRY.values())


def get_target_for_backend_device(backend: str, device: str) -> TargetSpec:
    backend_key = backend.lower()
    device_key = device.lower()

    if backend_key in ("onnx", "onnxruntime"):
        if device_key.startswith("cuda"):
            return get_target("cuda")
        if device_key == "cpu":
            return get_target("cpu")
    if backend_key == "vllm":
        if device_key == "cpu":
            return get_target("vllm-cpu")
        return get_target("vllm-cuda")
    if backend_key in ("hailort", "hailo", "hailo8"):
        return get_target("hailo8")

    # 하위 호환: registry에 직접 backend 이름으로 등록된 target이 있으면 사용.
    if backend_key in _TARGET_REGISTRY:
        return get_target(backend_key)

    return TargetSpec(
        target_id=f"{backend_key}:{device}",
        label=f"{backend}/{device}",
        runtime_name=backend,
        device=device,
        artifact_format="unknown",
        accelerator_vendor="",
        accelerator_name=device,
        capabilities=("legacy",),
    )


def resolve_target(target_id: Optional[str], backend: str, device: str) -> TargetSpec:
    if target_id:
        return get_target(target_id)
    return get_target_for_backend_device(backend, device)


def target_metadata(target: TargetSpec, compile_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    metadata = {
        "target_id": target.target_id,
        "accelerator_vendor": target.accelerator_vendor,
        "accelerator_name": target.accelerator_name,
        "runtime_name": target.runtime_name,
        "compiler_name": target.compiler_name or "",
        "artifact_format": target.artifact_format,
    }
    if compile_metadata:
        metadata.update({
            "compiler_name": compile_metadata.get("compiler_name", metadata["compiler_name"]),
            "artifact_format": compile_metadata.get("artifact_format", metadata["artifact_format"]),
        })
    return metadata


register_target(TargetSpec(
    target_id="cpu",
    label="CPU / ONNX Runtime",
    runtime_name="onnxruntime",
    device="cpu",
    monitor_names=("system",),
    artifact_format="onnx",
    accelerator_vendor="Generic",
    accelerator_name="CPU",
    capabilities=("onnx", "local", "baseline"),
    description="CPU execution through ONNX Runtime",
))

register_target(TargetSpec(
    target_id="cuda",
    label="NVIDIA CUDA / ONNX Runtime",
    runtime_name="onnxruntime",
    device="cuda",
    monitor_names=("nvidia", "system"),
    artifact_format="onnx",
    accelerator_vendor="NVIDIA",
    accelerator_name="CUDA GPU",
    capabilities=("onnx", "local", "gpu"),
    description="CUDA execution through ONNX Runtime",
))

register_target(TargetSpec(
    target_id="vllm-cuda",
    label="NVIDIA CUDA / vLLM",
    runtime_name="vllm",
    device="cuda",
    monitor_names=("nvidia", "system"),
    artifact_format="hf_model",
    accelerator_vendor="NVIDIA",
    accelerator_name="CUDA GPU",
    capabilities=("generation", "local", "gpu"),
    description="vLLM generation on CUDA",
))

register_target(TargetSpec(
    target_id="vllm-cpu",
    label="CPU / vLLM",
    runtime_name="vllm",
    device="cpu",
    monitor_names=("system",),
    artifact_format="hf_model",
    accelerator_vendor="Generic",
    accelerator_name="CPU",
    capabilities=("generation", "local", "requires_vllm_cpu_backend"),
    description="vLLM generation on CPU. Requires a vLLM CPU build/backend.",
))

register_target(TargetSpec(
    target_id="vendor_mock_npu",
    label="Mock Vendor NPU",
    runtime_name="mock_npu",
    device="npu0",
    compiler_name="mock_npu",
    monitor_names=("mock_npu", "system"),
    artifact_format="mockbin",
    accelerator_vendor="MockNPU",
    accelerator_name="Mock NPU PCIe Adapter",
    device_selector="npu0",
    capabilities=("onnx", "compile", "monitor", "npu", "local"),
    compiler_options={"vendor": "MockNPU", "artifact_format": "mockbin"},
    monitor_options={"mock_npu": {"device_id": "npu0"}},
    description="SDK-free target used to validate NPU plugin wiring",
))

register_target(TargetSpec(
    target_id="hailo8",
    label="Hailo-8 M.2 / HailoRT",
    runtime_name="hailort",
    device="device0",
    monitor_names=("hailo", "system"),
    artifact_format="hef",
    accelerator_vendor="Hailo",
    accelerator_name="Hailo-8 M.2",
    device_selector="device0",
    capabilities=("hef", "sync", "latency", "throughput", "monitor", "npu", "local"),
    runtime_options={
        "interface": "pcie",
        "input_format_type": "float32",
        "output_format_type": "float32",
        "input_layout": "auto",
    },
    monitor_options={"hailo": {"device_id": "device0", "enable_power": True}},
    description="Runs precompiled HEF files on a Hailo-8/8L device through HailoRT sync inference",
))
