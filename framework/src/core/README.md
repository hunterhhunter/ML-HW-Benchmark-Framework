# Core Package

`core` 패키지는 벤치마크 실행 루프, 모델 프로필, target 해석, 결과 저장을 담당합니다.

## 주요 파일

| 파일 | 역할 |
|------|------|
| `inference_engine.py` | e2e와 async_queue의 추론 lifecycle 및 completion 소유 |
| `runtime_executor.py` | blocking runtime과 callback 기반 native async SDK 실행 경계 |
| `inference_pipeline.py` | 두 모드가 공유하는 collate, runtime input, output 정규화 |
| `benchmarkrunner.py` | 기존 e2e 호출자를 위한 `InferenceEngine` 호환 façade |
| `async_inference/runner.py` | private async run controller와 helper |
| `model_profiles.py` | 모델 이름별 zero-config profile 정의 |
| `targets.py` | `target_id` 기반 Runtime/Compiler/Monitor 조합 registry |
| `compiled_model.py` | runtime에 전달되는 artifact path와 backend 정보를 담는 DTO |
| `result_store.py` | CSV 결과 저장/조회/삭제 |

## 통합 추론 구조

```text
main.py
  ├─ e2e ──────────> BenchmarkRunner (호환 façade) ─┐
  └─ async_queue ──> InferenceEngine.run_async()    │
                                                    ▼
                                             InferenceEngine
       ┌──────────┴──────────┐
   e2e inline           async_queue
 queue/worker 없음      bounded queue + worker
       └──────────┬──────────┘
                  ▼
           RuntimeExecutor
       ├─ BlockingRuntimeExecutor
       └─ NativeAsyncRuntimeExecutor -> vendor SDK queue

DataLoader -> InferencePipeline -> CompletionCoordinator
           -> Decoder/Postprocessor -> Evaluator -> Result

e2e artifact   : CSV + RUN_ID
async artifact : RUN_ID_RESERVED + CSV + JSON details
                 + optional JSONL trace + RUN_ID
```

`main.py`가 모델/runtime과 컴포넌트를 준비하고 결과를 저장합니다. `BenchmarkRunner`만
e2e warmup, monitor와 engine 호출을 보존하는 호환 façade로 남습니다. async 경로는
`main.py`가 `InferenceEngine.run_async()`를 직접 호출합니다.

`InferenceEngine`이 request ID와 submission token, 실행 순서, completion, evaluator와
flush/shutdown을 관리합니다. completion membership과 exact-once terminal은 request ID와 exact
submission token의 쌍으로 판정합니다. `e2e`는 동일 completion 코드를 inline으로 실행하므로
framework queue나 worker를 만들지 않습니다. `async_queue`는 framework request의 소유권과 backpressure를
위해 bounded queue를 항상 유지합니다. NPU SDK가 자체 queue를 제공해도 그 queue는 장치
실행 계층일 뿐 framework queue를 대체하지 않습니다.

e2e에서 executor가 failure-valued `RuntimeExecution`을 반환하면 공개
`RuntimeExecutionError`가 발생하며 부분 품질 metric이나 성공 artifact를 만들지 않습니다.
반환된 execution은 terminal 처리 뒤 ACK하고, inline coordinator stop과
`RuntimeExecutor.shutdown(timeout=0.0)`을 거친 뒤에만 evaluator metric을 계산합니다. fatal
decoder/evaluator/trace 예외는 정리를 시도한 뒤 같은 primary 객체로 전파됩니다.
Fallible runtime input 준비는 request 등록 전에 수행하므로 이 단계의 실패는 PENDING request를
남기지 않습니다. 예외 type-name/message 진단은 hostile `__name__`/`__str__`에도 안전하고
각각 최대 256/512자로 제한됩니다.

기본 executor는 기존 `runtime.run()`/`generate()`를 호출하는
`BlockingRuntimeExecutor`입니다. callback 기반 vendor SDK adapter는
`NativeAsyncRuntimeExecutor`를 명시적으로 주입합니다. 이때 framework의 dispatch token은
native registry 조회와 ACK의 기준이고 vendor job ID는 진단용입니다. SDK callback이 와도 공통
completion이 결과를 인수해 ACK하기 전까지 native buffer와 in-flight slot을 해제하지 않습니다.
timeout은 vendor 작업을 취소하지 않으므로 logical ACK와 물리 callback 완료(또는 adapter별
cancellation 증명)가 모두 있어야 retire합니다. 미해결 timeout 작업은 shutdown과 runtime
unload를 계속 unsafe로 유지하며, submit 예외는 callback이 이미 전달된 경우를 제외하면 작업이
accept되지 않았음을 뜻합니다.

`_AsyncRunController`와 native dispatch registry는 private 구현입니다. 외부 호출자나 vendor
adapter에서 직접 접근하지 말고 `InferenceEngine`과 `RuntimeExecutor` 계약을 사용합니다.

## CLI 실행과 디버깅

기본 동기 실행은 기존처럼 `e2e`입니다.

```bash
python src/main.py ... \
  --inference-mode e2e \
  --results-path /tmp/e2e-results/results.csv
```

비동기 queue 실행과 request trace 저장은 다음처럼 활성화합니다.

```bash
python src/main.py ... \
  --inference-mode async_queue \
  --scenario offline \
  --queue-capacity 4 \
  --worker-count 1 \
  --batch-timeout-ms 20 \
  --save-request-trace \
  --debug \
  --results-path /tmp/async-results/results.csv
```

async 실행의 `RUN_ID_RESERVED=<id>`와 최종 `RUN_ID=<id>`는 stdout lifecycle marker입니다.
`--debug`를 켜면 별도의 `[AsyncDebug] phase=... event=...` 레코드가 stderr에 출력되며,
run ID가 확보된 이후의 레코드에는 `run_id=<id>`도 포함됩니다. 이 레코드는 reservation,
measurement, unload, sidecar/CSV 저장 같은 coarse lifecycle만 다루고 sample별 input,
prediction이나 label을 출력하지 않습니다. `--save-request-trace`는 request/sample ID,
worker, batch/sample count, status, timeout, scheduling부터 completion까지의 timestamp, error
type과 공백 정규화·최대 512자 제한 error message를 JSONL로 저장합니다. input, label, output
tensor나 prompt field를 직접 저장하지는 않지만 error message 내용의 redaction은 보장하지
않습니다. 정상 async 종료 시 CSV의 `details_path`와 `request_trace_path`가 같은 run ID의
artifact를 가리키고 details의 `counts.outstanding`은 0이어야 합니다. e2e는 CSV와 단일
`RUN_ID`만 저장합니다.

CI의 자동 async CLI smoke는 외부 다운로드 없이 생성한 ONNX 모델을 실제
`python src/main.py` subprocess로 실행해 `CPUExecutionProvider`, count/outstanding와 async
artifact 연결을 검증합니다. 별도 e2e CLI smoke는 exit 0, 선택한 CSV와 단일 `RUN_ID`를
검증하며 두 smoke 사이의 품질 parity를 직접 비교하지 않습니다. cross-mode output, evaluator
품질과 sample count parity는 in-process ONNX Runtime CPU 테스트가 검증합니다. 같은 asset을
사용한 수동 두 CLI 프로세스 인수에서도 4 sample과 분류 품질 parity를 확인했습니다.

이번 unified `InferenceEngine`/`async_queue` 실행 경로는 MLPerf LoadGen 코드를 사용하거나
통합하지 않았고 SUT/QSL API 및 로그 호환도 제공하지 않습니다. 저장소의 이전 버전에서 추가된
비활성 legacy `adapters/loadgen_adapter.py` 스켈레톤은 이번 경로가 호출하거나 수정하지
않습니다. exact-once 완료, outstanding/flush, monotonic timing 같은 신뢰성·측정 원칙만
behavioral reference로 삼았습니다. 이 결과는 MLPerf submission 또는 compliance 결과가
아닙니다.

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

## Registry graph validation

`validate_registry_graph()`는 모든 `TargetSpec`이 실제 Runtime/Compiler/Monitor
registry entry를 가리키는지 SDK import 없이 확인합니다.

```python
from core.targets import validate_registry_graph

report = validate_registry_graph()
assert report["ok"], report
```

검증은 다음 계약을 확인합니다.

- `runtime_name`은 `runtimes.get_runtime_entry()`로 해석된다.
- `compiler_name`이 있으면 `compilers.get_compiler_entry()`로 해석된다.
- `monitor_names`의 모든 항목은 `monitors.get_collector_entry()`로 해석된다.
- `artifact_format`은 비어 있지 않다.
- `capabilities`는 API/UI에 그대로 노출 가능한 lowercase/trimmed 문자열이다.

새 hardware target을 추가할 때는 `framework/tests/test_plugin_registry.py`에
graph validation case를 함께 추가합니다. `BenchmarkRunner`는 target 등록 위치가
아니며, runtime/dataloader/evaluator를 주입받아 실행만 담당합니다.

## 하위 호환

기존 요청처럼 `--backend onnxruntime --device cpu`를 사용하면 `cpu` target으로 매핑됩니다. `--backend vllm --device cuda`는 `vllm-cuda` target으로 매핑됩니다. registry에 직접 등록되지 않은 backend/device 조합은 legacy target으로 감싸서 가능한 한 기존 동작을 유지합니다.
