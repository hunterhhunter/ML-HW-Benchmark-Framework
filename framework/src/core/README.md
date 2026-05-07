# Core Package

`core` 패키지는 벤치마크 실행 루프, 모델 프로필, target 해석, 결과 저장을 담당합니다.

## 주요 파일

| 파일 | 역할 |
|------|------|
| `benchmarkrunner.py` | dataloader, runtime, evaluator를 연결해 benchmark loop 실행 |
| `model_profiles.py` | 모델 이름별 zero-config profile 정의 |
| `targets.py` | `target_id` 기반 Runtime/Compiler/Monitor 조합 registry |
| `compiled_model.py` | runtime에 전달되는 artifact path와 backend 정보를 담는 DTO |
| `result_store.py` | CSV 결과 저장/조회/삭제 |

## TargetSpec

`TargetSpec`은 하나의 실행 target을 표현합니다.

```python
TargetSpec(
    target_id="vendor_mock_npu",
    label="Mock Vendor NPU",
    runtime_name="mock_npu",
    device="npu0",
    compiler_name="mock_npu",
    monitor_names=("mock_npu", "system"),
    artifact_format="mockbin",
    accelerator_vendor="MockNPU",
    accelerator_name="Mock NPU PCIe Adapter",
    capabilities=("onnx", "compile", "monitor", "npu", "local"),
)
```

CLI/API에서 `target_id`가 들어오면 `resolve_target()`이 이를 해석합니다. `--target`이 지정된 경우 target의 `runtime_name`과 `device`가 기존 `--backend/--device`보다 우선합니다.

## 하위 호환

기존 요청처럼 `--backend onnxruntime --device cpu`를 사용하면 `cpu` target으로 매핑됩니다. `--backend vllm --device cuda`는 `vllm-cuda` target으로 매핑됩니다. registry에 직접 등록되지 않은 backend/device 조합은 legacy target으로 감싸서 가능한 한 기존 동작을 유지합니다.
