"""
Runtime Package Initialization & Registry

런타임 구현체를 registry로 등록하고 lazy import로 생성한다. 특정 벤더 SDK가
설치되어 있지 않은 환경에서도 registry 조회와 나머지 런타임 사용은 유지된다.
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Dict, Optional

from .base import Runtime
from core.generation_result import GenerationResult
from core.registry import normalize_registry_key, register_entry_keys


@dataclass(frozen=True)
class RuntimeEntry:
    name: str
    module: str
    class_name: str
    aliases: tuple[str, ...] = ()
    description: str = ""
    unsupported_reason: Optional[str] = None

    def load(self) -> type[Runtime]:
        if self.unsupported_reason:
            raise NotImplementedError(self.unsupported_reason)
        module = import_module(self.module)
        return getattr(module, self.class_name)


_RUNTIME_REGISTRY: Dict[str, RuntimeEntry] = {}


def register_runtime(entry: RuntimeEntry) -> None:
    register_entry_keys(_RUNTIME_REGISTRY, entry, (entry.name, *entry.aliases), "runtime")


def get_runtime_entry(name: str) -> RuntimeEntry:
    key = normalize_registry_key(name, "runtime")
    entry = _RUNTIME_REGISTRY.get(key)
    if entry is None:
        supported = sorted(_RUNTIME_REGISTRY.keys())
        raise ValueError(f"지원하지 않는 백엔드입니다: {name}. 지원 목록: {supported}")
    return entry


def list_runtimes() -> list[dict]:
    seen: set[str] = set()
    result = []
    for entry in _RUNTIME_REGISTRY.values():
        if entry.name in seen:
            continue
        seen.add(entry.name)
        result.append({
            "name": entry.name,
            "aliases": list(entry.aliases),
            "description": entry.description,
            "available": entry.unsupported_reason is None,
            "unsupported_reason": entry.unsupported_reason,
        })
    return result


def create_runtime(backend_name: str, device: str = "cpu", **kwargs) -> Runtime:
    """
    Registry-backed Runtime factory.

    Args:
        backend_name: 등록된 런타임 이름 또는 alias
        device: 실행 디바이스 문자열. 벤더 런타임은 target registry에서 전달한 값을 사용한다.
        **kwargs: 런타임별 옵션
    """
    entry = get_runtime_entry(backend_name)

    try:
        runtime_cls = entry.load()
    except NotImplementedError:
        raise
    except Exception as exc:
        raise RuntimeError(f"런타임 플러그인 '{backend_name}' 로드 실패: {exc}") from exc
    return runtime_cls(device=device, **kwargs)


def __getattr__(name: str):
    """하위 호환용 lazy class export."""
    exports = {
        "OnnxRuntime": ("runtimes.onnx_rt", "OnnxRuntime"),
        "VllmRuntime": ("runtimes.vllm_rt", "VllmRuntime"),
        "IREERuntime": ("runtimes.iree_rt", "IREERuntime"),
        "MockNpuRuntime": ("runtimes.mock_npu_rt", "MockNpuRuntime"),
        "HailoRuntime": ("runtimes.hailo_rt", "HailoRuntime"),
        "DeepXRuntime": ("runtimes.deepx_rt", "DeepXRuntime"),
        "RblnRuntime": ("runtimes.rbln_rt", "RblnRuntime"),
        "RblnVllmRuntime": (
            "runtimes.rbln_vllm_rt",
            "RblnVllmRuntime",
        ),
        "MobilintRuntime": ("runtimes.mobilint_rt", "MobilintRuntime"),
        "MobilintLlmRuntime": (
            "runtimes.mobilint_llm_rt",
            "MobilintLlmRuntime",
        ),
        "FuriosaLlmRuntime": ("runtimes.furiosa_llm_rt", "FuriosaLlmRuntime"),
    }
    if name not in exports:
        raise AttributeError(name)
    module_name, class_name = exports[name]
    module = import_module(module_name)
    return getattr(module, class_name)


register_runtime(RuntimeEntry(
    name="onnxruntime",
    module="runtimes.onnx_rt",
    class_name="OnnxRuntime",
    aliases=("onnx",),
    description="ONNX Runtime backend",
))

register_runtime(RuntimeEntry(
    name="vllm",
    module="runtimes.vllm_rt",
    class_name="VllmRuntime",
    description="vLLM generation backend",
))

register_runtime(RuntimeEntry(
    name="iree",
    module="runtimes.iree_rt",
    class_name="IREERuntime",
    aliases=("mlir",),
    description="IREE backend",
    unsupported_reason="IREE 런타임은 현재 공통 인터페이스 맞춤 리팩토링 중입니다.",
))

register_runtime(RuntimeEntry(
    name="mock_npu",
    module="runtimes.mock_npu_rt",
    class_name="MockNpuRuntime",
    aliases=("vendor_mock_npu",),
    description="SDK-free runtime used to validate NPU plugin wiring",
))

register_runtime(RuntimeEntry(
    name="hailort",
    module="runtimes.hailo_rt",
    class_name="HailoRuntime",
    aliases=("hailo", "hailo8", "hailo10h"),
    description="HailoRT runtime for precompiled HEF artifacts on Hailo devices",
))

register_runtime(RuntimeEntry(
    name="deepx",
    module="runtimes.deepx_rt",
    class_name="DeepXRuntime",
    aliases=("dxrt", "deepx_npu"),
    description="DEEPX NPU runtime for precompiled DXNN artifacts",
))

register_runtime(RuntimeEntry(
    name="rbln",
    module="runtimes.rbln_rt",
    class_name="RblnRuntime",
    aliases=("rebel", "rbln-static"),
    description="Rebellions runtime for precompiled static RBLN artifacts",
))

register_runtime(RuntimeEntry(
    name="rbln_vllm",
    module="runtimes.rbln_vllm_rt",
    class_name="RblnVllmRuntime",
    aliases=("rbln-vllm",),
    description="In-process vLLM RBLN generation backend",
))

register_runtime(RuntimeEntry(
    name="mobilint",
    module="runtimes.mobilint_rt",
    class_name="MobilintRuntime",
    aliases=("qbruntime", "mxq"),
    description="Mobilint qb Runtime backend for precompiled ARIES/REGULUS MXQ artifacts",
))

register_runtime(RuntimeEntry(
    name="mobilint_llm",
    module="runtimes.mobilint_llm_rt",
    class_name="MobilintLlmRuntime",
    description="Mobilint Model Zoo generation runtime for ARIES",
))

register_runtime(RuntimeEntry(
    name="furiosa_llm",
    module="runtimes.furiosa_llm_rt",
    class_name="FuriosaLlmRuntime",
    aliases=("furiosa", "rngd"),
    description="Furiosa-LLM generation backend for RNGD FXB artifacts",
))


__all__ = [
    "Runtime",
    "GenerationResult",
    "RuntimeEntry",
    "register_runtime",
    "get_runtime_entry",
    "list_runtimes",
    "create_runtime",
    "OnnxRuntime",
    "IREERuntime",
    "VllmRuntime",
    "MockNpuRuntime",
    "HailoRuntime",
    "DeepXRuntime",
    "RblnRuntime",
    "RblnVllmRuntime",
    "MobilintRuntime",
    "MobilintLlmRuntime",
    "FuriosaLlmRuntime",
]
