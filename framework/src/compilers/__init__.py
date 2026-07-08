"""
Compilers Package Initialization & Registry

벤더별 컴파일러 SDK를 코어 코드 수정 없이 등록할 수 있는 registry 기반
팩토리를 제공합니다. 등록 엔트리는 lazy import를 사용하므로 특정 SDK가
설치되어 있지 않아도 프레임워크 import 자체는 실패하지 않습니다.
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Dict

from .base import Compiler, CompileResult, normalize_compile_result


@dataclass(frozen=True)
class CompilerEntry:
    name: str
    module: str
    class_name: str
    aliases: tuple[str, ...] = ()
    description: str = ""

    def load(self) -> type[Compiler]:
        module = import_module(self.module)
        return getattr(module, self.class_name)


_COMPILER_REGISTRY: Dict[str, CompilerEntry] = {}


def register_compiler(entry: CompilerEntry) -> None:
    keys = (entry.name, *entry.aliases)
    for key in keys:
        _COMPILER_REGISTRY[key.strip().lower()] = entry


def list_compilers() -> list[dict]:
    """중복 alias를 제거한 컴파일러 registry 요약을 반환한다."""
    seen: set[str] = set()
    result = []
    for entry in _COMPILER_REGISTRY.values():
        if entry.name in seen:
            continue
        seen.add(entry.name)
        result.append({
            "name": entry.name,
            "aliases": list(entry.aliases),
            "description": entry.description,
        })
    return result

def get_compiler(compiler_name: str, **compile_options) -> Compiler:
    """
    Factory Method for Compiler
    
    이름(compiler_name)을 입력받아 해당 백엔드 구체 컴파일러(Concrete Compiler) 인스턴스를 반환합니다.
    
    Args:
        compiler_name (str): 사용할 AI 컴파일러 이름 (예: "iree", "tvm")
        **compile_options: target_backend (예: llvm-cpu, cuda), 최적화 레벨 등의 컴파일 인자
        
    Returns:
        Compiler: 추상 베이스 클래스를 상속받은 구체 컴파일러 인스턴스
        
    Raises:
        ValueError: 지원하지 않는 컴파일러 이름이 들어명 예외 발생
    """
    key = compiler_name.strip().lower()
    entry = _COMPILER_REGISTRY.get(key)
    if entry is None:
        supported = sorted(_COMPILER_REGISTRY.keys())
        raise ValueError(f"현재 '{compiler_name}' 컴파일러 백엔드는 지원되지 않습니다. 지원 목록: {supported}")

    try:
        compiler_cls = entry.load()
    except Exception as exc:
        raise RuntimeError(f"컴파일러 플러그인 '{compiler_name}' 로드 실패: {exc}") from exc
    return compiler_cls(**compile_options)


register_compiler(CompilerEntry(
    name="iree",
    module="compilers.iree_compiler",
    class_name="IREECompiler",
    aliases=("mlir",),
    description="IREE compiler backend",
))

register_compiler(CompilerEntry(
    name="mock_npu",
    module="compilers.mock_npu_compiler",
    class_name="MockNpuCompiler",
    aliases=("vendor_mock_npu",),
    description="SDK-free compiler used to validate NPU plugin wiring",
))

register_compiler(CompilerEntry(
    name="deepx",
    module="compilers.deepx_compiler",
    class_name="DeepXCompiler",
    aliases=("dxcom", "dx_com"),
    description="DEEPX DX-COM compiler backend for ONNX to DXNN artifacts",
))

__all__ = [
    "Compiler",
    "CompileResult",
    "normalize_compile_result",
    "CompilerEntry",
    "register_compiler",
    "list_compilers",
    "get_compiler",
]
