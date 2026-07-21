# 통합 InferenceEngine 설계 명세

**상태:** 구현 및 ONNX Runtime CPU 인수 검증 완료

**승인일:** 2026-07-20

**대상 범위:** `framework` 추론 오케스트레이션, 동기 `e2e`, `async_queue`

## 1. 결정

동기 `e2e`와 `async_queue`를 별도 추론 시스템으로 유지하지 않는다. 하나의
`InferenceEngine`이 sample 획득부터 결과 확정까지 전체 추론 흐름을 조율한다.

`DataLoader`, `Preprocessor`, `Postprocessor` 또는 `Decoder`, `Evaluator`는 두 모드가
동일한 인터페이스와 호출 순서로 사용한다. 두 모드의 차이는 요청 발행 방식, 선택적
Framework Queue와 `RuntimeExecutor` 전략으로 제한한다.

`InferenceEngine`은 각 컴포넌트의 데이터 변환이나 평가 알고리즘을 흡수하지 않는다.
의존 컴포넌트를 소유하고 request identity, 호출 순서, ownership, 측정 경계와 lifecycle을
관리하는 오케스트레이터다.

## 2. 구현된 상위 구조

```text
CLI / main.py
  ├─ runtime/model 준비와 결과 artifact 저장
  ├─ e2e ──────────> BenchmarkRunner (호환 façade) ─┐
  └─ async_queue ──> InferenceEngine.run_async()    │
                                                    ▼
                                             InferenceEngine
            ┌────────────┴─────────────┐
        e2e inline                 async_queue
  queue/worker를 만들지 않음    bounded Framework Queue
            └────────────┬─────────────┘
                         ▼
                  RuntimeExecutor
            ├─ BlockingRuntimeExecutor
            └─ NativeAsyncRuntimeExecutor
                         │
                  vendor SDK queue

두 모드의 공통 결과 경로
DataLoader -> InferencePipeline -> RuntimeExecution
-> CompletionCoordinator -> Decoder/Postprocessor -> Evaluator -> Result

CLI artifact 경로
e2e         -> CSV + RUN_ID
async_queue -> RUN_ID_RESERVED + CSV + JSON details
               + optional JSONL trace + RUN_ID
```

`BenchmarkRunner`만 기존 e2e 호출자를 위한 얇은 호환 façade로 남는다. async CLI는
`InferenceEngine.run_async()`를 직접 호출한다. 실제 run 안쪽의 추론 lifecycle은
`InferenceEngine`이 소유하고, CLI 조립과 결과 저장은 `main.py`와 `ResultStore`가 담당한다.
`_AsyncRunController`와 native dispatch registry는 구현 세부사항인 private 객체이며 공개
오케스트레이터가 아니다.

## 3. 공통 파이프라인

두 모드는 다음 논리 경로를 공유한다.

```text
DataLoader
  -> Preprocessor
  -> PreparedBatch
  -> RuntimeExecutor
  -> BatchCompletion
  -> Postprocessor / Decoder
  -> Evaluator
  -> Metrics / Result
```

### 3.1 동기 e2e

- `BlockingRuntimeExecutor`가 현재 `runtime.run()` 또는 `runtime.generate()`를 호출한다.
- 호출자는 장치 실행과 공통 completion 처리가 끝날 때까지 반환을 기다린다.
- Framework Queue, worker thread와 native in-flight registry를 만들지 않는다.
- `CompletionCoordinator`는 별도 completion worker 없이 inline으로 같은 terminal 경로를 수행한다.
- 기존 output, evaluator metric, runtime-only latency와 CLI 기본 동작을 보존한다.
- 반환된 모든 `RuntimeExecution`은 terminal commit과 outstanding 제거 뒤 정확히 한 번 ACK한다.
- collate 이후의 fallible runtime input 준비는 request 등록 전에 끝낸다. 이 단계가 실패하면
  아직 framework가 인수한 known request가 없으므로 PENDING/FAILED terminal을 만들지 않고,
  원래 예외를 보존한 채 executor shutdown만 수행한다. 다음 정상 request의 request/sample ID
  증가 규칙은 바뀌지 않는다.
- executor가 `error_type`을 담은 failure-valued execution을 반환하면 공개
  `RuntimeExecutionError`로 즉시 실패한다. 이전 성공 batch가 있어도 `Evaluator.compute()`와
  성공 CSV/`RUN_ID` 저장을 수행하지 않는다.
- decoder/evaluator 또는 trace의 fatal `BaseException`은 terminal/ACK 정리 뒤 같은 객체를
  다시 던진다. 일반 trace `Exception`은 warning-only다.
- exception type-name과 message는 hostile `__name__`/`__str__` 구현도 실행 경로를 깨지
  못하도록 각각 독립된 total formatter로 최대 256/512자 안에서 기록한다.
- inline coordinator stop 뒤 `RuntimeExecutor.shutdown(timeout=0.0)`을 정확히 한 번 호출하고,
  성공한 shutdown 뒤에만 `Evaluator.compute()`를 호출한다.

e2e 실패 우선순위는 최초 실행·callback·failure-valued execution, ACK, coordinator stop,
executor shutdown 순이다. 뒤따르는 정리 실패가 앞선 primary 객체를 대체하지 않는다.

### 3.2 async_queue

- load profile이 요청 발행 시각과 속도를 결정한다.
- `InferenceEngine`이 bounded Framework Queue, admission, batching과 backpressure를 관리한다.
- `BlockingRuntimeExecutor`를 worker에서 사용할 수도 있고, vendor SDK가 지원하면
  `NativeAsyncRuntimeExecutor`를 선택할 수 있다.
- completion은 공통 `CompletionCoordinator`로 전달되어 동기 경로와 동일한
  postprocess, decode, evaluate, metrics와 terminal commit을 실행한다.

Framework Queue는 framework 측 요청 소유권과 측정의 source of truth다. Vendor SDK
queue는 `NativeAsyncRuntimeExecutor` 내부의 장치 실행 세부사항이며 이를 대체하지 않는다.
현재 ONNX Runtime CPU CLI는 `BlockingRuntimeExecutor`를 async worker에서 사용한다. 실제
vendor SDK adapter는 후속 범위이며, 지원할 때 `NativeAsyncRuntimeExecutor`를 명시적으로
주입한다.

## 4. 컴포넌트 책임

| 컴포넌트 | 책임 | 하지 않는 일 |
|---|---|---|
| `main.py` | 모델/runtime과 컴포넌트 준비, 모드 선택, 결과 artifact 저장 | run 내부 request lifecycle 관리 |
| `BenchmarkRunner` | e2e warmup, monitor, engine 호출의 호환 façade | 모델/runtime 준비, artifact 저장, 요청별 queue 관리 |
| `InferenceEngine` | 전체 추론 순서, request ownership, batching, 선택적 queue, executor 선택, flush/shutdown | backend별 SDK 호출 구현, 전처리·평가 알고리즘 구현 |
| `DataLoader` | sample 조회, index/cursor/reset 계약 | runtime 실행과 terminal 관리 |
| `Preprocessor` | sample을 모델 입력 형식으로 변환 | queue와 장치 실행 관리 |
| `RuntimeExecutor` | `PreparedBatch`를 장치에서 실행하고 `BatchCompletion` 전달 | postprocess, evaluator, terminal 확정 |
| `Postprocessor` / `Decoder` | raw output을 task 결과로 변환 | request scheduling과 SDK lifecycle 관리 |
| `Evaluator` | task 품질 지표 누적과 finalize | concurrency, queue, retry 정책 결정 |
| `CompletionCoordinator` | completion membership 검증, 공통 후처리 호출, exact-once terminal commit | vendor job을 source of truth로 사용 |
| `ResultStore` | CSV, details, trace와 failure artifact 저장 | 측정 중 request lifecycle 변경 |

`OfflineProducer`와 `ServerLikeProducer`는 자체 부하 profile 구현이다. 시나리오 이름이나
동작이 MLPerf 공식 시나리오와 동등하다는 뜻이 아니다.

## 5. 공개 경계

### 5.1 InferenceEngine

`InferenceEngine`은 동기와 비동기 실행을 같은 request/completion 계약으로 노출한다.
구체적인 공개 메서드 이름은 구현 계획에서 현재 호출자와의 호환성을 확인한 뒤 정하지만,
다음 의미는 고정한다.

- run 시작 시 dependency와 capability snapshot을 검증한다.
- request를 전처리 완료 상태와 framework identity로 연결한다.
- batch를 만들고 선택된 executor에 한 번 제출한다.
- completion을 공통 후처리와 evaluator 경로로 전달한다.
- accepted 요청의 exact-once terminal과 outstanding 0을 보장한다.
- flush와 shutdown의 단일 deadline 및 실패 진실을 보존한다.

### 5.2 RuntimeExecutor

`RuntimeExecutor`는 장치 실행만 추상화한다.

- `BlockingRuntimeExecutor`는 기존 동기 runtime 호출을 보존한다.
- `NativeAsyncRuntimeExecutor`는 dispatch token, vendor job ID, native in-flight permit,
  callback과 buffer lifetime을 내부에서 관리한다.
- 두 구현은 동일한 `BatchCompletion` 성공/실패 계약을 사용한다.
- executor는 decoder, evaluator, ResultStore를 직접 호출하지 않는다.

Native 경로의 ID와 소유권은 다음처럼 나뉜다.

| ID | 소유자 | 용도 |
|---|---|---|
| request ID | Framework queue/completion | logical request와 sample을 식별하는 key |
| submission token | Framework queue/completion | 같은 request ID의 attempt를 구분하는 membership token |
| dispatch token | `NativeAsyncRuntimeExecutor` | framework가 생성하는 native 제출별 canonical key |
| vendor job ID | vendor SDK | 로그와 진단용 보조 값이며 terminal 판정의 기준이 아님 |

completion membership과 exact-once terminal 판정은 request ID와 exact submission token 쌍을
사용한다. dispatch token은 이 계층을 대체하지 않고 native registry 조회와 ACK에만 사용한다.

native input/output, in-flight permit과 registry entry는 callback 수신만으로 해제하지 않는다.
공통 completion 경로가 terminal 결과를 인수한 뒤 `acknowledge()`해야 해제된다. timeout은
논리적 deadline이며 첫 vendor adapter 전까지 물리 취소를 보장하지 않는다.

### 5.3 CompletionCoordinator

`CompletionCoordinator`는 `InferenceEngine`이 소유하는 공통 completion service다.

- 동기 경로는 completion을 inline으로 처리할 수 있다.
- 비동기 경로는 bounded Completion Queue로 전달한다.
- 두 경로는 동일한 membership 검증, decoder/postprocessor, evaluator, metrics와
  terminal commit 로직을 사용한다.

## 6. 유지되는 계약

기존 비동기 큐 명세의 다음 계약은 변경하지 않는다.

- `submitted = accepted + rejected`
- `accepted = completed + failed + outstanding`
- accepted request의 exact-once terminal
- bounded Framework Queue와 명시적 backpressure
- request ID, submission token과 dispatch token의 분리
- monotonic clock 기반 queue/service/completion/e2e timing
- timeout, duplicate, unknown, stale completion의 invalid 진단
- terminal ACK 전 native buffer와 slot ownership 유지
- CSV, JSON sidecar, 선택적 JSONL trace와 failure recovery record
- MLPerf LoadGen 비의존 및 비호환 범위

## 7. 마이그레이션

1. 현재 `AsyncInferenceEngine`에서 queue, batching, lifecycle과 runtime 호출 경계를 식별한다.
2. 현재 blocking 호출을 `BlockingRuntimeExecutor`로 추출하고 결과 parity를 증명한다.
3. 기존 `BenchmarkRunner`는 e2e 호환 façade로 공통 `InferenceEngine`을 사용하고, async CLI는
   `InferenceEngine.run_async()`를 직접 호출하도록 orchestration을 이동한다.
4. DataLoader, Preprocessor, Postprocessor/Decoder와 Evaluator의 기존 구현과 metric 이름을
   변경하지 않고 공통 dependency로 연결한다.
5. 동기 completion은 inline, 비동기 completion은 queue handoff를 사용하되 terminal 로직을
   하나로 통합한다.
6. `FakeNativeAsyncRuntime`으로 callback race와 shutdown을 검증한 뒤
   `NativeAsyncRuntimeExecutor`를 추가한다.
7. 첫 vendor adapter는 명시적 opt-in으로 연결하고 blocking rollback을 유지한다.

## 8. 구현 및 TDD 검증 결과

구현은 테스트가 책임 경계와 실패 조건을 먼저 고정한 뒤 production 코드를 작성하는 TDD
순서로 진행했다. 특히 다음 계약을 독립 테스트로 검증했다.

- e2e `BenchmarkRunner`와 async CLI가 하나의 `InferenceEngine` 경계를 사용하고 두 모드가
  동일한 pipeline/completion을 사용함
- e2e가 queue나 worker를 만들지 않고 기존 품질 metric을 보존함
- async의 bounded admission, exact-once terminal, outstanding 0과 shutdown 수렴
- native callback의 inline/late/out-of-order/duplicate, submit 실패, timeout과 ACK 소유권
- shutdown 중 late handoff와 cancellation retirement race의 결정적 재현과 회귀 방지
- 실제 ONNX Runtime CPU에서 e2e/async output, 품질 metric과 sample count parity

검증 증거는 다음 세 층으로 구분한다.

- 자동 async CLI smoke는 외부 다운로드 없이 작은 ONNX 모델과 4개 이미지를 만들고 실제
  `python src/main.py` subprocess를 실행해 exit code 0, `CPUExecutionProvider`, exact count,
  `outstanding=0`, async CSV/details/trace와 run ID 연결을 확인한다. 별도 자동 e2e CLI smoke는
  exit code 0, 선택한 CSV와 `RUN_ID`를 확인한다. 두 smoke 사이의 품질 parity를 직접 assertion하지
  않는다.
- in-process ONNX Runtime CPU 테스트가 같은 input에서 e2e/async output, evaluator 품질과 sample
  count parity 및 실제 provider를 assertion한다.
- 수동 인수에서 같은 asset으로 실제 e2e와 async CLI 프로세스를 각각 실행해 4 sample과 분류
  품질 parity를 대조했다.

성능 수치의 우열은 어느 검증에서도 합격 조건으로 사용하지 않는다.

`--debug`는 async run의 reservation, warmup, measurement, runtime unload, sidecar/CSV 저장
phase와 run ID를 출력한다. 요청별 사후 분석이 필요할 때만 `--save-request-trace`를 함께
사용한다. trace는 request/sample ID, worker, batch/sample count, status, timeout, scheduling부터
completion까지의 timestamp, error type과 공백 정규화·최대 512자 제한 error message를 담는다.
input, label, output tensor나 prompt 같은 request payload field를 직접 직렬화하지 않지만,
error message 내용의 비밀정보 redaction까지 보장하는 형식은 아니다.

## 9. 테스트와 승인 기준

- 기존 전체 테스트와 ONNX Runtime CPU 실제 CLI 검증이 통과한다.
- 동기 경로의 output, evaluator metric, CSV와 `RUN_ID` 계약이 리팩터링 전과 일치한다.
- async 경로는 `RUN_ID_RESERVED`, CSV, JSON details, 선택적 JSONL trace와 최종 `RUN_ID`를
  같은 run으로 연결한다.
- 같은 입력에서 동기와 async의 전처리·후처리·평가 결과가 일치한다.
- 동기 실행은 불필요한 Framework Queue나 worker thread를 생성하지 않는다.
- async 실행은 기존 queue capacity, counter, timing과 terminal 불변식을 유지한다.
- Fake native async fault matrix에서 out-of-order, duplicate, inline/late callback, submit failure,
  timeout과 shutdown이 exact-once로 수렴한다.
- 실제 vendor SDK가 없어도 CI에서 `BlockingRuntimeExecutor`와 fake native async 경로를 검증한다.

## 10. 비범위

- DataLoader, Preprocessor, Postprocessor/Decoder, Evaluator 공개 계약의 전면 재설계
- unified `InferenceEngine`/`async_queue` 실행 경로에서 MLPerf LoadGen 코드 사용·통합,
  SUT/QSL API, 측정 로그 호환
- MLPerf 공식 submission package와 compliance audit 대응
- 첫 단계에서 실제 NPU vendor adapter 구현
- 실행 중 자동 executor 전환
- 분산 queue 또는 multi-process inference engine

## 11. MLPerf 레퍼런스 경계와 최종 결정

`InferenceEngine`을 추론 전체의 유일한 오케스트레이터로 채택한다. 동기 `e2e`와
`async_queue`는 DataLoader, Preprocessor, Postprocessor/Decoder, Evaluator,
CompletionCoordinator와 결과 계약을 공유한다. 실행 방식과 필요한 queue 동작만 모드와
`RuntimeExecutor`에 따라 달라진다.

이 설계는 기존 비동기 큐 신뢰성 계약을 폐기하지 않는다. 현재 구현된 큐와 completion
불변식을 통합 `InferenceEngine` 아래로 이동시켜, backend 실행 방식이 늘어나도 데이터 처리와
품질 평가 경로가 갈라지지 않게 한다.

이번 unified `InferenceEngine`/`async_queue` 실행 경로에서 MLPerf LoadGen은 코드를 가져오거나
API·로그를 맞추는 통합 대상이 아니다. 요청 발행과 완료 분리, immutable ID, exact-once
completion, outstanding/flush, monotonic latency, tail percentile과 fault testing 같은
**신뢰성·측정 원칙의 behavioral reference**로만 삼았다. 저장소에는 이전 버전에서 추가된
비활성 legacy `framework/src/adapters/loadgen_adapter.py` 스켈레톤이 남아 있지만, 이번 경로는
이를 호출하거나 수정하지 않는다. 따라서 이 프레임워크의 `valid` 결과를 MLPerf 결과, 공식
submission 또는 compliance 통과로 해석하면 안 된다.
