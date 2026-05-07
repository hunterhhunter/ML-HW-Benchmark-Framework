# NPU 등록 방법

이 문서는 새로운 NPU 벤더를 ML HW Benchmark Framework에 연결할 때
어떤 항목을 구현하고 어디에 등록해야 하는지 정리한다.

핵심 원칙은 `BenchmarkRunner`를 수정하지 않고, 벤더별 SDK 차이를
adapter와 registry entry로 흡수하는 것이다.

## 등록해야 하는 항목

새 NPU target을 추가할 때는 보통 네 가지를 준비한다.

| 구분 | 위치 | 역할 |
|---|---|---|
| Runtime adapter | `framework/src/runtimes/` | 벤더 SDK의 모델 로드, warmup, 추론 실행 API를 공통 `Runtime` 인터페이스로 연결 |
| Compiler adapter | `framework/src/compilers/` | ONNX 또는 IR 모델을 NPU 실행 artifact로 변환 |
| Monitor collector | `framework/src/monitors/` | 벤더 profiler 또는 device API에서 사용률, 메모리, 전력, 온도 지표 수집 |
| TargetSpec | `framework/src/core/targets.py` | runtime, compiler, monitor, artifact format, device 정보를 하나의 `target_id`로 묶음 |

컴파일 단계가 필요 없는 NPU라면 `compiler_name`은 비워둘 수 있다.
모니터링을 아직 지원하지 않더라도 최소한 `system` collector는 함께 둘 수 있다.

## 1. Runtime adapter 추가

`framework/src/runtimes/base.py`의 `Runtime`을 상속한다.

필수 메서드:

- `load(compiled_model)`: 컴파일된 artifact 또는 원본 모델을 벤더 runtime에 로드
- `run(inputs)`: batch 입력을 받아 추론 결과를 `dict[str, np.ndarray]` 형태로 반환
- `warmup(inputs, num_runs)`: 측정 전 warmup 실행
- `unload()`: runtime/device 자원 해제
- `get_device_spec()`: 벤더명, 장치명 등 실행 장치 정보 반환
- `is_compatible(compiled_model)`: artifact format, device 조건 검사

예시:

```python
# framework/src/runtimes/vendor_npu_rt.py
from runtimes.base import Runtime


class VendorNpuRuntime(Runtime):
    def __init__(self, device: str = "npu0", **runtime_options):
        self.device = device
        self.runtime_options = runtime_options
        self.session = None

    def load(self, compiled_model):
        # vendor_sdk.load(compiled_model.artifact_path, device=self.device)
        ...

    def run(self, inputs):
        # outputs = self.session.run(inputs)
        # return {"output": outputs}
        ...

    def warmup(self, inputs, num_runs: int = 1):
        for _ in range(num_runs):
            self.run(inputs)

    def unload(self):
        self.session = None

    def get_device_spec(self):
        return {
            "accelerator_vendor": "VendorName",
            "accelerator_name": "Vendor NPU",
            "device": self.device,
        }

    def is_compatible(self, compiled_model):
        return compiled_model.artifact_path.endswith(".vendorbin")
```

그리고 `framework/src/runtimes/__init__.py`에 등록한다.

```python
register_runtime(RuntimeEntry(
    name="vendor_npu",
    module="runtimes.vendor_npu_rt",
    class_name="VendorNpuRuntime",
    aliases=("vendor_npu_v1",),
    description="Vendor NPU runtime adapter",
))
```

Registry는 lazy import 방식이므로, 해당 SDK가 설치되지 않은 환경에서도
다른 target 실행은 깨지지 않아야 한다. 벤더 SDK import는 adapter 파일 내부에서
필요한 시점에 수행하는 편이 안전하다.

## 2. Compiler adapter 추가

벤더 NPU가 전용 artifact를 요구하면 `framework/src/compilers/base.py`의
`Compiler`를 상속한다.

필수 메서드:

- `compile(model_spec, output_dir)`: 모델을 벤더 artifact로 변환하고 경로 반환
- `get_artifact_name(model_spec)`: 캐시 확인에 사용할 artifact 파일명 반환

`compile()`은 문자열 경로만 반환해도 되지만, 재현성을 위해 `CompileResult`로
metadata를 함께 남기는 것을 권장한다.

```python
# framework/src/compilers/vendor_npu_compiler.py
from compilers.base import Compiler, CompileResult


class VendorNpuCompiler(Compiler):
    def compile(self, model_spec, output_dir):
        artifact_path = f"{output_dir}/{self.get_artifact_name(model_spec)}"
        # vendor_compiler.compile(model_spec.model_path, artifact_path, **self.compile_options)
        return CompileResult(
            artifact_path=artifact_path,
            metadata={
                "compiler_name": "vendor_npu",
                "artifact_format": "vendorbin",
            },
        )

    def get_artifact_name(self, model_spec):
        return f"{model_spec.model_name}.vendorbin"
```

그리고 `framework/src/compilers/__init__.py`에 등록한다.

```python
register_compiler(CompilerEntry(
    name="vendor_npu",
    module="compilers.vendor_npu_compiler",
    class_name="VendorNpuCompiler",
    aliases=("vendor_npu_v1",),
    description="Vendor NPU compiler adapter",
))
```

컴파일 결과는 기본적으로 `framework/artifacts/<target_id>/` 아래에 캐시된다.
이 디렉터리는 실행 산출물이므로 Git에 포함하지 않는다.

## 3. Monitor collector 추가

`framework/src/monitors/base.py`의 `Collector`를 상속한다.

필수 메서드:

- `start()`: profiler/device API 초기화
- `collect()`: 현재 샘플 지표 반환
- `stop()`: profiler/device API 정리
- `is_available()`: 현재 환경에서 수집 가능 여부

NPU 지표는 벤더별 raw key 대신 공통 prefix인 `hw_accel_*`를 사용한다.

권장 key:

| Key | 의미 |
|---|---|
| `hw_accel_util` | NPU 사용률 |
| `hw_accel_mem_used_mb` | NPU 메모리 사용량 |
| `hw_accel_mem_proc_mb` | 현재 benchmark 프로세스 기준 NPU 메모리 사용량 |
| `hw_accel_power_w` | 전력 |
| `hw_accel_temp_c` | 온도 |

정적 정보가 있으면 `get_static_info()`를 제공할 수 있다.

```python
# framework/src/monitors/vendor_npu_collector.py
from monitors.base import Collector


class VendorNpuCollector(Collector):
    def __init__(self, device_id: str = "npu0", **options):
        self.device_id = device_id
        self.options = options

    def is_available(self):
        # return vendor_profiler.is_available()
        return True

    def start(self):
        # vendor_profiler.start(self.device_id)
        ...

    def collect(self):
        return {
            "hw_accel_util": 0.0,
            "hw_accel_mem_used_mb": 0.0,
            "hw_accel_power_w": 0.0,
            "hw_accel_temp_c": 0.0,
        }

    def stop(self):
        # vendor_profiler.stop()
        ...

    def get_static_info(self):
        return {
            "hw_accel_vendor": "VendorName",
            "hw_accel_name": "Vendor NPU",
            "hw_accel_device_id": self.device_id,
        }
```

그리고 `framework/src/monitors/__init__.py`에 등록한다.

```python
register_collector(CollectorEntry(
    name="vendor_npu",
    module="monitors.vendor_npu_collector",
    class_name="VendorNpuCollector",
    aliases=("vendor_npu_v1",),
    description="Vendor NPU profiler metrics",
))
```

`HWMonitor.summary()`는 `hw_accel_*` 시계열에서 avg/max/peak 요약 지표를 만든다.
측정 구간은 warmup 이후 inference loop이다.

## 4. TargetSpec 등록

마지막으로 `framework/src/core/targets.py`에 target 조합을 등록한다.

```python
register_target(TargetSpec(
    target_id="vendor_npu",
    label="Vendor NPU",
    runtime_name="vendor_npu",
    device="npu0",
    compiler_name="vendor_npu",
    monitor_names=("vendor_npu", "system"),
    artifact_format="vendorbin",
    accelerator_vendor="VendorName",
    accelerator_name="Vendor NPU",
    device_selector="npu0",
    capabilities=("onnx", "compile", "monitor", "npu", "local"),
    runtime_options={"option_key": "option_value"},
    compiler_options={"optimization": "default"},
    monitor_options={"vendor_npu": {"device_id": "npu0"}},
    description="Vendor NPU execution target",
))
```

`target_id`는 CLI와 API에서 사용하는 외부 식별자다.
한 벤더에 여러 장치나 실행 모드가 있으면 `vendor_npu`, `vendor_npu_fast`,
`vendor_npu_int8`처럼 target을 나눠 등록한다.

## 실행 확인

Target 목록:

```bash
cd framework
python src/main.py --help
```

백엔드 API를 실행 중이라면:

```bash
curl http://localhost:8000/api/benchmark/targets
```

벤치마크 실행:

```bash
cd framework
python src/main.py --model resnet50 --target vendor_npu --monitor
```

컴파일을 건너뛰어야 할 때:

```bash
python src/main.py --model resnet50 --target vendor_npu --no-compile
```

간단한 구조 검증은 기존 mock target으로 먼저 확인할 수 있다.

```bash
python src/main.py --model resnet50 --target vendor_mock_npu --max-steps 1 --warmup 0 --monitor
```

## 결과 저장 확인

실행 결과 CSV에는 target metadata가 함께 저장되어야 한다.

필수 metadata 컬럼:

- `target_id`
- `accelerator_vendor`
- `accelerator_name`
- `runtime_name`
- `compiler_name`
- `artifact_format`

NPU monitor가 연결되면 다음과 같은 metric 컬럼이 추가될 수 있다.

- `hw_accel_util_avg`
- `hw_accel_util_max`
- `hw_accel_mem_peak_mb`
- `hw_accel_power_w_avg`
- `hw_accel_power_w_max`
- `hw_accel_temp_c_avg`
- `hw_accel_temp_c_max`

새 metric 컬럼은 `save_result()`가 CSV header에 자동으로 추가한다.

## 등록 체크리스트

커밋 전 다음 항목을 확인한다.

- Runtime adapter가 `Runtime` 인터페이스를 구현한다.
- Runtime adapter가 `runtimes/__init__.py`에 `RuntimeEntry`로 등록되어 있다.
- 컴파일이 필요한 경우 Compiler adapter가 `Compiler` 인터페이스를 구현한다.
- Compiler adapter가 `compilers/__init__.py`에 `CompilerEntry`로 등록되어 있다.
- NPU metric 수집이 필요한 경우 Collector가 `Collector` 인터페이스를 구현한다.
- Collector가 `monitors/__init__.py`에 `CollectorEntry`로 등록되어 있다.
- `targets.py`에 `TargetSpec`이 등록되어 있다.
- `monitor_names`에 벤더 collector와 `system` collector가 함께 포함되어 있다.
- NPU metric key는 `hw_accel_*` prefix를 사용한다.
- 결과 CSV에 target metadata가 저장된다.
- 벤더 SDK가 없는 환경에서도 다른 target import와 실행이 깨지지 않는다.
