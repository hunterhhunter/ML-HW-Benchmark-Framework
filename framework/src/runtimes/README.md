# Runtimes Package

`runtimes` 패키지는 ONNX Runtime, vLLM, IREE, 벤더 NPU SDK처럼 기술 스택과 가속 방식이 서로 다른 **하드웨어 추론 엔진**을 `BenchmarkRunner` 입장에서 투명하게 제어하도록 캡슐화합니다.

## Architecture

- **`base.py` (`Runtime`)**: 모든 외부 런타임 wrapper가 준수해야 하는 추상 인터페이스입니다.
  - `load()`: 원본 또는 컴파일된 artifact를 target 메모리에 로드
  - `warmup()`: 추론 초기 성능 왜곡을 줄이기 위한 예열
  - `run()`: 배치 단위 추론을 실행하고 Numpy dictionary 또는 generation result를 반환
  - `unload()`: 런타임 리소스 해제

- **`__init__.py` (Registry Facade)**
  - `RuntimeEntry`를 registry에 등록하고 `create_runtime(name, device, **kwargs)`로 생성합니다.
  - 등록 entry는 lazy import를 사용합니다. 특정 벤더 SDK가 설치되지 않아도 프레임워크 import와 다른 target 실행은 깨지지 않습니다.
  - 기존 alias도 유지합니다. 예를 들어 `onnx`는 `onnxruntime`으로 매핑됩니다.

## Built-in Runtime Registry

| name | aliases | 설명 |
|---|---|---|
| `onnxruntime` | `onnx` | ONNX Runtime backend |
| `vllm` | - | vLLM generation backend |
| `iree` | `mlir` | IREE backend placeholder |
| `mock_npu` | `vendor_mock_npu` | SDK-free NPU plugin 검증 runtime |

## 새 Runtime 추가

새 벤더 NPU runtime을 추가할 때 core 실행 코드를 수정하지 않습니다. adapter와 registry entry만 추가합니다.

1. `Runtime`을 상속한 adapter 파일을 `src/runtimes/`에 추가합니다.
2. 벤더 SDK import는 가능한 한 adapter 내부의 `load()` 또는 초기화 시점으로 미룹니다.
3. `RuntimeEntry`를 등록합니다.
4. `src/core/targets.py`에서 해당 runtime을 사용하는 `TargetSpec`을 추가합니다.

```python
# src/runtimes/__init__.py
register_runtime(RuntimeEntry(
    name="vendor_npu",
    module="runtimes.vendor_npu_rt",
    class_name="VendorNpuRuntime",
    aliases=("vendor-x",),
    description="Vendor NPU runtime adapter",
))
```

실제 벤더 SDK adapter는 `Runtime.load()`에서 compiler artifact 또는 원본 artifact를 target device에 올리고, `Runtime.run()`에서 프레임워크 evaluator가 이해할 수 있는 출력 형태를 반환해야 합니다.
