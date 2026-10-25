# Benchmark Framework - Compilers

`compilers` 패키지는 원본 모델 artifact를 target 하드웨어가 실행할 수 있는
native artifact로 변환하는 adapter 계층입니다. 전체 조립·확장 절차는
[개발자 설명서](../../../docs/developer-guide/02-component-contracts.md#compiler와-compiledmodel)에서
다룹니다.

## Public API

외부 컴포넌트는 개별 compiler 구현을 직접 import하지 않고 registry facade를 사용합니다.

```python
from compilers import get_compiler, get_compiler_entry, normalize_compile_result

entry = get_compiler_entry("mock_npu")
print(entry.description)

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
| `deepx` | `dxcom`, `dx_com` | DEEPX DX-COM compiler backend |

## Compile-aware 실행 흐름

`src/core/targets.py`의 `TargetSpec.compiler_name`이 설정되어 있고 CLI/API에서 compile이 활성화되어 있으면 `main.py`가 다음 순서로 실행합니다.

1. `get_compiler(target.compiler_name, **compile_options)`로 compiler adapter를 생성합니다.
2. `framework/artifacts/<target_id>/`를 출력 디렉터리로 전달합니다. 캐시 확인과
   재사용 여부는 compiler 구현에 달려 있습니다. 기본 `is_cached()`는 예상 파일
   존재 여부만 검사하므로 원본 모델·옵션·calibration의 일치까지 보증하지 않습니다.
3. compiler가 target artifact를 생성하거나 검증한 캐시를 사용합니다.
4. `CompileResult.artifact_path`를 runtime에 전달합니다.
5. `CompileResult.metadata`는 결과 저장 시 `compiler_name`, `artifact_format` 등으로 반영됩니다.

`--no-compile`은 compiler 호출을 건너뛰는 옵션입니다. ONNX처럼 원본을 직접
실행하는 target은 원본 경로를 사용할 수 있지만, native artifact가 필요한
target에는 그 형식의 사전 컴파일 파일을 명시해야 합니다.

## 새 Compiler 추가

새 벤더 NPU compiler는 adapter와 registry entry를 등록하고 `TargetSpec`에
연결합니다. 모델별 입력·출력 계약이 달라지면 Loader·Decoder 조립도 확인합니다.

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

### Step 4. Registry graph 검증

compiler entry를 등록한 뒤에는 target graph 검증을 통과해야 합니다.

```python
from core.targets import validate_registry_graph

report = validate_registry_graph()
assert report["ok"], report
```

`TargetSpec.compiler_options["artifact_format"]`이 있는 경우 target의
`artifact_format`과 일치해야 합니다. name 또는 alias가 이미 다른 compiler에
등록되어 있으면 등록 시점에 실패합니다.
