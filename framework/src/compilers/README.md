# Benchmark Framework - Compilers

`compilers` 패키지는 원본 모델 artifact(ONNX, HuggingFace model 등)를 target 하드웨어가 실행할 수 있는 native artifact로 변환하는 adapter 계층입니다. 이번 MVP에서는 특정 벤더 SDK 완성 구현보다 **벤더 compiler adapter를 core 수정 없이 꽂을 수 있는 registry 구조**를 우선합니다.

## Public API

외부 컴포넌트는 개별 compiler 구현을 직접 import하지 않고 registry facade를 사용합니다.

```python
from compilers import get_compiler, normalize_compile_result

compiler = get_compiler(
    "mock_npu",
    vendor="MockNPU",
    artifact_format="mockbin",
)

result = normalize_compile_result(
    compiler.compile(model_spec, output_dir="artifacts/vendor_mock_npu")
)

print(result.artifact_path)
print(result.metadata)
```

`Compiler.compile()`은 하위 호환을 위해 `str` 경로를 반환해도 되지만, 새 구현은 `CompileResult` 반환을 권장합니다.

```python
from compilers.base import CompileResult

return CompileResult(
    artifact_path="/path/to/model.mockbin",
    metadata={
        "compiler_name": "vendor_npu",
        "artifact_format": "vendorbin",
        "cache_hit": False,
    },
)
```

## Built-in Compiler Registry

| name | aliases | 설명 |
|---|---|---|
| `iree` | `mlir` | IREE compiler backend |
| `mock_npu` | `vendor_mock_npu` | SDK-free NPU compiler wiring 검증용 |

## Compile-aware 실행 흐름

`src/core/targets.py`의 `TargetSpec.compiler_name`이 설정되어 있고 CLI/API에서 compile이 활성화되어 있으면 `main.py`가 다음 순서로 실행합니다.

1. `get_compiler(target.compiler_name, **compile_options)`로 compiler adapter를 생성합니다.
2. `framework/artifacts/<target_id>/` 아래에서 artifact cache를 확인합니다.
3. cache miss면 compiler가 target artifact를 생성합니다.
4. `CompileResult.artifact_path`를 runtime에 전달합니다.
5. `CompileResult.metadata`는 결과 저장 시 `compiler_name`, `artifact_format` 등으로 반영됩니다.

`--no-compile`을 지정하면 compiler를 건너뛰고 원본 ONNX/HF artifact를 runtime에 전달합니다.

## 새 Compiler 추가

새 벤더 NPU compiler를 추가할 때는 adapter와 registry entry만 추가하고 core 실행 흐름은 수정하지 않습니다.

### Step 1. Compiler 구현

```python
# src/compilers/vendor_npu_compiler.py
from pathlib import Path

from compilers.base import Compiler, CompileResult
from core.model_spec import Model_Spec


class VendorNpuCompiler(Compiler):
    def __init__(self, **compile_options):
        super().__init__(**compile_options)
        self.artifact_format = compile_options.get("artifact_format", "vendorbin")

    def get_artifact_name(self, model_spec: Model_Spec) -> str:
        return f"{model_spec.name}.{self.artifact_format}"

    def compile(self, model_spec: Model_Spec, output_dir: str) -> CompileResult:
        output_path = Path(output_dir) / self.get_artifact_name(model_spec)
        cache_hit = output_path.exists()
        if not cache_hit:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # 벤더 SDK compile 호출은 이 지점에 둡니다.
            output_path.write_bytes(b"compiled artifact")

        return CompileResult(
            artifact_path=str(output_path),
            metadata={
                "compiler_name": "vendor_npu",
                "artifact_format": self.artifact_format,
                "cache_hit": cache_hit,
                "compile_options": self.compile_options,
            },
        )
```

### Step 2. Registry 등록

```python
# src/compilers/__init__.py
register_compiler(CompilerEntry(
    name="vendor_npu",
    module="compilers.vendor_npu_compiler",
    class_name="VendorNpuCompiler",
    aliases=("vendor-x",),
    description="Vendor NPU compiler adapter",
))
```

### Step 3. TargetSpec 연결

`src/core/targets.py`에 runtime, compiler, monitor 조합을 등록합니다.

```python
# src/core/targets.py
register_target(TargetSpec(
    target_id="vendor_npu",
    label="Vendor NPU",
    runtime_name="vendor_npu",
    device="npu0",
    compiler_name="vendor_npu",
    monitor_names=("vendor_npu", "system"),
    artifact_format="vendorbin",
    accelerator_vendor="Vendor",
    accelerator_name="Vendor NPU",
    capabilities=("onnx", "compile", "monitor", "npu", "local"),
))
```

벤더 SDK import는 compiler module import 시점에 실패하지 않도록 가능한 한 compile 호출 내부로 늦춥니다. SDK가 없을 때는 adapter 단에서 원인을 포함한 명확한 `RuntimeError`를 발생시키는 것이 좋습니다.
