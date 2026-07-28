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
    if not target.target_id.strip():
        raise ValueError("target_id must not be empty")
    existing = _TARGET_REGISTRY.get(target.target_id)
    if existing is not None and existing != target:
        raise ValueError(
            f"target registry key '{target.target_id}' already belongs to "
            f"'{existing.label}', cannot register '{target.label}'"
        )
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
    if backend_key in ("deepx", "dxrt", "deepx_npu"):
        return get_target("deepx")
    if backend_key in ("furiosa_llm", "furiosa", "rngd"):
        return get_target("furiosa-rngd")
    if backend_key in ("rbln_vllm", "rbln-vllm"):
        return get_target("rbln-vllm")
    if backend_key == "rbln" and device_key == "0":
        return get_target("rbln-static")

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


def _graph_issue(target_id: str, field: str, value: Any, message: str) -> Dict[str, str]:
    return {
        "target_id": target_id,
        "field": field,
        "value": "" if value is None else str(value),
        "message": message,
    }


def _validate_capabilities(target: TargetSpec) -> list[Dict[str, str]]:
    errors: list[Dict[str, str]] = []
    seen: set[str] = set()
    for capability in target.capabilities:
        if not isinstance(capability, str) or not capability.strip():
            errors.append(_graph_issue(
                target.target_id,
                "capabilities",
                capability,
                "Target capability must be a non-empty string",
            ))
            continue
        normalized = capability.strip().lower()
        if capability != capability.strip() or capability != normalized:
            errors.append(_graph_issue(
                target.target_id,
                "capabilities",
                capability,
                "Target capability must be lowercase and trimmed for API/UI consistency",
            ))
        if normalized in seen:
            errors.append(_graph_issue(
                target.target_id,
                "capabilities",
                capability,
                "Target capability is duplicated",
            ))
        seen.add(normalized)
    return errors


def validate_registry_graph(
    targets: Optional[list[TargetSpec]] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Validate target-to-registry wiring without importing vendor SDK modules.

    `strict=True` keeps errors and warnings separate but makes warnings fail the
    top-level `ok` flag. This is useful for contributor diagnostics while keeping
    intentionally exposed placeholders visible in non-strict reports.
    """
    from compilers import get_compiler_entry
    from monitors import get_collector_entry
    from runtimes import get_runtime_entry

    selected_targets = list(targets) if targets is not None else list_targets()
    all_errors: list[Dict[str, str]] = []
    all_warnings: list[Dict[str, str]] = []
    target_reports: list[Dict[str, Any]] = []

    for target in selected_targets:
        target_errors: list[Dict[str, str]] = []
        target_warnings: list[Dict[str, str]] = []

        try:
            runtime_entry = get_runtime_entry(target.runtime_name)
        except ValueError as exc:
            target_errors.append(_graph_issue(
                target.target_id,
                "runtime_name",
                target.runtime_name,
                f"Target references an unregistered runtime: {exc}",
            ))
        else:
            if runtime_entry.unsupported_reason:
                target_warnings.append(_graph_issue(
                    target.target_id,
                    "runtime_name",
                    target.runtime_name,
                    f"Runtime is registered but marked unsupported: {runtime_entry.unsupported_reason}",
                ))

        if target.compiler_name:
            try:
                get_compiler_entry(target.compiler_name)
            except ValueError as exc:
                target_errors.append(_graph_issue(
                    target.target_id,
                    "compiler_name",
                    target.compiler_name,
                    f"Target references an unregistered compiler: {exc}",
                ))

        for monitor_name in target.monitor_names:
            try:
                get_collector_entry(monitor_name)
            except ValueError as exc:
                target_errors.append(_graph_issue(
                    target.target_id,
                    "monitor_names",
                    monitor_name,
                    f"Target references an unregistered collector: {exc}",
                ))

        if not target.monitor_names:
            target_warnings.append(_graph_issue(
                target.target_id,
                "monitor_names",
                "",
                "Target has no monitor collectors; hardware telemetry will be absent",
            ))

        if not isinstance(target.artifact_format, str) or not target.artifact_format.strip():
            target_errors.append(_graph_issue(
                target.target_id,
                "artifact_format",
                target.artifact_format,
                "Target artifact_format must be a non-empty string",
            ))

        compiler_artifact_format = target.compiler_options.get("artifact_format")
        if compiler_artifact_format and compiler_artifact_format != target.artifact_format:
            target_errors.append(_graph_issue(
                target.target_id,
                "artifact_format",
                target.artifact_format,
                "Target artifact_format does not match compiler_options['artifact_format']",
            ))

        target_errors.extend(_validate_capabilities(target))
        all_errors.extend(target_errors)
        all_warnings.extend(target_warnings)
        target_reports.append({
            "target_id": target.target_id,
            "ok": not target_errors and (not strict or not target_warnings),
            "errors": target_errors,
            "warnings": target_warnings,
        })

    return {
        "ok": not all_errors and (not strict or not all_warnings),
        "errors": all_errors,
        "warnings": all_warnings,
        "targets": target_reports,
    }


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
    target_id="furiosa-rngd",
    label="FuriosaAI RNGD / Furiosa-LLM",
    runtime_name="furiosa_llm",
    device="npu:0",
    monitor_names=("system",),
    artifact_format="fxb",
    accelerator_vendor="FuriosaAI",
    accelerator_name="RNGD",
    device_selector="npu:0",
    capabilities=("generation", "native_async", "streaming", "npu", "local"),
    description="Runs local Hugging Face weights with a precompiled FXB on RNGD",
))

register_target(TargetSpec(
    target_id="rbln-static",
    label="Rebellions ATOM / RBLN Runtime",
    runtime_name="rbln",
    device="0",
    monitor_names=("rbln", "system"),
    artifact_format="rbln",
    accelerator_vendor="Rebellions",
    accelerator_name="RBLN NPU",
    device_selector="0",
    capabilities=(
        "rbln",
        "sync",
        "native_async",
        "latency",
        "throughput",
        "monitor",
        "npu",
        "local",
        "static_shape",
    ),
    runtime_options={
        "device_id": 0,
        "async_parallel": 1,
        "runtime_timeout_sec": 60,
        "shutdown_timeout_sec": 300.0,
    },
    monitor_options={
        "rbln": {
            "device_id": 0,
            "sample_interval_sec": 1.0,
            "command_timeout_sec": 2.0,
        },
    },
    description="Runs precompiled static RBLN artifacts on Rebellions NPU device 0",
))

register_target(TargetSpec(
    target_id="rbln-vllm",
    label="Rebellions ATOM / vLLM RBLN",
    runtime_name="rbln_vllm",
    device="0",
    monitor_names=("rbln", "system"),
    artifact_format="rbln_llm_dir",
    accelerator_vendor="Rebellions",
    accelerator_name="RBLN NPU",
    device_selector="0",
    capabilities=(
        "rbln_llm_dir",
        "generation",
        "native_async",
        "streaming",
        "token_events",
        "continuous_batching",
        "latency",
        "throughput",
        "monitor",
        "npu",
        "local",
    ),
    runtime_options={
        "num_devices": 1,
        "max_num_seqs": 1,
        "tensor_parallel_size": 1,
        "startup_timeout_sec": 600.0,
        "shutdown_timeout_sec": 300.0,
    },
    monitor_options={
        "rbln": {
            "device_id": 0,
            "sample_interval_sec": 1.0,
            "command_timeout_sec": 2.0,
        },
    },
    description=(
        "Runs prepared Optimum RBLN language models through in-process "
        "vLLM RBLN"
    ),
))

register_target(TargetSpec(
    target_id="mobilint-aries",
    label="Mobilint ARIES / qb Runtime",
    runtime_name="mobilint",
    device="0",
    monitor_names=("mobilint", "system"),
    artifact_format="mxq",
    accelerator_vendor="Mobilint",
    accelerator_name="ARIES",
    device_selector="0",
    capabilities=(
        "mxq",
        "sync",
        "native_async",
        "latency",
        "throughput",
        "monitor",
        "npu",
        "local",
    ),
    runtime_options={
        "device_id": 0,
        "expected_family": "aries",
        "async_pipeline_enabled": False,
        "activation_slots": 1,
    },
    monitor_options={
        "mobilint": {
            "device_id": 0,
            "expected_family": "aries",
            "accelerator_name": "ARIES",
        },
    },
    description="Runs precompiled MXQ models on an explicitly validated ARIES device",
))

register_target(TargetSpec(
    target_id="mobilint-regulus",
    label="Mobilint REGULUS / qb Runtime",
    runtime_name="mobilint",
    device="0",
    monitor_names=("mobilint", "system"),
    artifact_format="mxq",
    accelerator_vendor="Mobilint",
    accelerator_name="REGULUS",
    device_selector="0",
    capabilities=(
        "mxq",
        "sync",
        "native_async",
        "latency",
        "throughput",
        "monitor",
        "npu",
        "local",
    ),
    runtime_options={
        "device_id": 0,
        "expected_family": "regulus",
        "async_pipeline_enabled": False,
        "activation_slots": 1,
    },
    monitor_options={
        "mobilint": {
            "device_id": 0,
            "expected_family": "regulus",
            "accelerator_name": "REGULUS",
        },
    },
    description="Runs precompiled MXQ models on an explicitly validated REGULUS PCIe or USB device",
))

register_target(TargetSpec(
    target_id="mobilint-aries-llm",
    label="Mobilint ARIES / Model Zoo LLM",
    runtime_name="mobilint_llm",
    device="0",
    monitor_names=("mobilint", "system"),
    artifact_format="hf_model",
    accelerator_vendor="Mobilint",
    accelerator_name="ARIES",
    device_selector="0",
    capabilities=(
        "hf_model",
        "generation",
        "token_events",
        "latency",
        "monitor",
        "npu",
        "local",
    ),
    runtime_options={"device_id": 0, "expected_family": "aries"},
    monitor_options={
        "mobilint": {
            "device_id": 0,
            "expected_family": "aries",
            "accelerator_name": "ARIES",
        },
    },
    description="Runs a local prepared Model Zoo LLM on Mobilint ARIES device 0",
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
    capabilities=(
        "hef",
        "sync",
        "async",
        "native_async",
        "latency",
        "throughput",
        "monitor",
        "npu",
        "local",
    ),
    runtime_options={
        "interface": "pcie",
        "input_format_type": "uint8",
        "output_format_type": "auto",
        "input_layout": "auto",
        "accelerator_name": "Hailo-8 M.2",
    },
    monitor_options={
        "hailo": {
            "device_id": "device0",
            "accelerator_name": "Hailo-8 M.2",
            "enable_power": True,
            "power_mode": "auto",
            "power_buffer_index": "MEASUREMENT_BUFFER_INDEX_0",
            "power_should_clear": True,
            "suppress_power_errors": True,
        }
    },
    description="Runs precompiled HEF files on Hailo-8/8L through HailoRT sync or native async InferModel inference",
))

register_target(TargetSpec(
    target_id="hailo10h",
    label="Hailo-10H / HailoRT",
    runtime_name="hailort",
    device="device0",
    monitor_names=("hailo", "system"),
    artifact_format="hef",
    accelerator_vendor="Hailo",
    accelerator_name="Hailo-10H",
    device_selector="device0",
    capabilities=(
        "hef",
        "sync",
        "async",
        "native_async",
        "latency",
        "throughput",
        "monitor",
        "npu",
        "local",
    ),
    runtime_options={
        "interface": "pcie",
        "input_format_type": "uint8",
        "output_format_type": "auto",
        "input_layout": "auto",
        "accelerator_name": "Hailo-10H",
    },
    monitor_options={
        "hailo": {
            "device_id": "device0",
            "accelerator_name": "Hailo-10H",
            "enable_power": True,
            "power_mode": "auto",
            "power_buffer_index": "MEASUREMENT_BUFFER_INDEX_0",
            "power_should_clear": True,
            "suppress_power_errors": True,
        }
    },
    description="Runs Hailo-10H HEF files through HailoRT v5.x sync or native async InferModel inference",
))

register_target(TargetSpec(
    target_id="deepx",
    label="DEEPX NPU / DX-RT",
    runtime_name="deepx",
    device="npu0",
    compiler_name="deepx",
    monitor_names=("deepx", "system"),
    artifact_format="dxnn",
    accelerator_vendor="DEEPX",
    accelerator_name="DEEPX NPU",
    device_selector="npu0",
    capabilities=("onnx", "compile", "dxnn", "sync", "native_async", "latency", "throughput", "monitor", "npu", "local"),
    runtime_options={
        "sdk_module": "dx_engine",
        "bound_option": "NPU_ALL",
        "compatible_suffixes": (".dxnn",),
        "input_layout": "auto",
        "batch_mode": "sdk_batch",
        "buffer_count": 6,
        "async_completion_timeout_sec": 30.0,
    },
    monitor_options={
        "deepx": {
            "device_id": "all",
        }
    },
    description="Compiles ONNX with DX-COM and runs DXNN artifacts through the DEEPX runtime SDK",
))
