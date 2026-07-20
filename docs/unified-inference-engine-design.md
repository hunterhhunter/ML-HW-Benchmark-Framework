# 통합 InferenceEngine 설계 명세

**상태:** 승인됨

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

## 2. 상위 구조

```text
BenchmarkRunner
    ├─ runtime/model 준비
    ├─ warmup과 hardware monitor
    ├─ InferenceEngine 실행
    └─ 결과 artifact 저장
              │
              ▼
InferenceEngine
    ├─ DataLoader
    ├─ Preprocessor
    ├─ request identity와 batching
    ├─ 선택적 Framework Queue와 backpressure
    ├─ RuntimeExecutor
    │    ├─ BlockingRuntimeExecutor
    │    └─ NativeAsyncRuntimeExecutor
    ├─ CompletionCoordinator
    ├─ Postprocessor / Decoder
    ├─ Evaluator
    └─ Metrics와 terminal lifecycle
```

`BenchmarkRunner`는 run 바깥쪽의 준비와 저장을 담당한다. `InferenceEngine`은 run 안쪽의
추론 lifecycle을 담당한다.

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
- 기존 output, evaluator metric, runtime-only latency와 CLI 기본 동작을 보존한다.

### 3.2 async_queue

- load profile이 요청 발행 시각과 속도를 결정한다.
- `InferenceEngine`이 bounded Framework Queue, admission, batching과 backpressure를 관리한다.
- `BlockingRuntimeExecutor`를 worker에서 사용할 수도 있고, vendor SDK가 지원하면
  `NativeAsyncRuntimeExecutor`를 선택할 수 있다.
- completion은 공통 `CompletionCoordinator`로 전달되어 동기 경로와 동일한
  postprocess, decode, evaluate, metrics와 terminal commit을 실행한다.

Framework Queue는 framework 측 요청 소유권과 측정의 source of truth다. Vendor SDK
queue는 `NativeAsyncRuntimeExecutor` 내부의 장치 실행 세부사항이며 이를 대체하지 않는다.

## 4. 컴포넌트 책임

| 컴포넌트 | 책임 | 하지 않는 일 |
|---|---|---|
| `BenchmarkRunner` | 모델/runtime 준비, warmup, monitor, engine 호출, artifact 저장 | 요청별 queue, batching, completion 관리 |
| `InferenceEngine` | 전체 추론 순서, request ownership, batching, 선택적 queue, executor 선택, flush/shutdown | backend별 SDK 호출 구현, 전처리·평가 알고리즘 구현 |
| `DataLoader` | sample 조회, index/cursor/reset 계약 | runtime 실행과 terminal 관리 |
| `Preprocessor` | sample을 모델 입력 형식으로 변환 | queue와 장치 실행 관리 |
| `RuntimeExecutor` | `PreparedBatch`를 장치에서 실행하고 `BatchCompletion` 전달 | postprocess, evaluator, terminal 확정 |
| `Postprocessor` / `Decoder` | raw output을 task 결과로 변환 | request scheduling과 SDK lifecycle 관리 |
| `Evaluator` | task 품질 지표 누적과 finalize | concurrency, queue, retry 정책 결정 |
| `CompletionCoordinator` | completion membership 검증, 공통 후처리 호출, exact-once terminal commit | vendor job을 source of truth로 사용 |
| `ResultStore` | CSV, details, trace와 failure artifact 저장 | 측정 중 request lifecycle 변경 |

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
3. 기존 `BenchmarkRunner`와 `AsyncBenchmarkRunner`가 공통 `InferenceEngine`을 사용하도록
   orchestration을 이동한다.
4. DataLoader, Preprocessor, Postprocessor/Decoder와 Evaluator의 기존 구현과 metric 이름을
   변경하지 않고 공통 dependency로 연결한다.
5. 동기 completion은 inline, 비동기 completion은 queue handoff를 사용하되 terminal 로직을
   하나로 통합한다.
6. `FakeNativeAsyncRuntime`으로 callback race와 shutdown을 검증한 뒤
   `NativeAsyncRuntimeExecutor`를 추가한다.
7. 첫 vendor adapter는 명시적 opt-in으로 연결하고 blocking rollback을 유지한다.

## 8. 테스트와 승인 기준

- 기존 전체 테스트와 ONNX Runtime CPU 실제 CLI 검증이 통과한다.
- 동기 경로의 output, evaluator metric, CSV와 details artifact가 리팩터링 전과 일치한다.
- 같은 입력에서 동기와 async의 전처리·후처리·평가 결과가 일치한다.
- 동기 실행은 불필요한 Framework Queue나 worker thread를 생성하지 않는다.
- async 실행은 기존 queue capacity, counter, timing과 terminal 불변식을 유지한다.
- Fake native async fault matrix에서 out-of-order, duplicate, inline/late callback, submit failure,
  timeout과 shutdown이 exact-once로 수렴한다.
- 실제 vendor SDK가 없어도 CI에서 `BlockingRuntimeExecutor`와 fake native async 경로를 검증한다.

## 9. 비범위

- DataLoader, Preprocessor, Postprocessor/Decoder, Evaluator 공개 계약의 전면 재설계
- MLPerf LoadGen API, 로그, 제출 또는 compliance 호환
- 첫 단계에서 실제 NPU vendor adapter 구현
- 실행 중 자동 executor 전환
- 분산 queue 또는 multi-process inference engine

## 10. 최종 결정

`InferenceEngine`을 추론 전체의 유일한 오케스트레이터로 채택한다. 동기 `e2e`와
`async_queue`는 DataLoader, Preprocessor, Postprocessor/Decoder, Evaluator,
CompletionCoordinator와 결과 계약을 공유한다. 실행 방식과 필요한 queue 동작만 모드와
`RuntimeExecutor`에 따라 달라진다.

이 설계는 기존 비동기 큐 신뢰성 계약을 폐기하지 않는다. 현재 구현된 큐와 completion
불변식을 통합 `InferenceEngine` 아래로 이동시켜, backend 실행 방식이 늘어나도 데이터 처리와
품질 평가 경로가 갈라지지 않게 한다.
