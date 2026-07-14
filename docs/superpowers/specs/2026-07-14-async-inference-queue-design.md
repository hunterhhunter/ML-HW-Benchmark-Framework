# 비동기 추론 큐 설계 명세

**상태:** 사용자 검토용 명세

**작성일:** 2026-07-14

**대상 범위:** `framework` 코어, CLI, 결과 저장

## 1. 목적

ML-HW-Benchmark-Framework의 추론 실행 방식을 다음 두 가지로 확장한다.

1. `e2e`: 현재 `BenchmarkRunner`가 수행하는 순차 추론 방식
2. `async_queue`: 요청 제출과 추론 완료를 분리하고, bounded queue와 상주 worker로 실행하는 비동기 추론 방식

비동기 모드의 목적은 단순히 더 높은 성능 수치를 만드는 것이 아니다. 순차 실행에서 드러나지 않는 큐 대기시간, 부하 포화, tail latency, backpressure, 동적 배칭 효과를 관찰하고, 요청 유실이나 종료 교착 없이 반복 가능한 측정을 제공하는 것이 목적이다.

첫 CI 검증 대상은 Mock Runtime과 ONNX Runtime CPU다. CUDA, Hailo, DEEPX, IREE, vLLM은 동일한 코어 계약을 따르되 실제 장치 연결 후 순차적으로 검증한다.

## 2. MLPerf LoadGen을 참고하는 범위

MLPerf LoadGen은 의존성이나 통합 대상이 아니라 신뢰성 설계를 위한 레퍼런스다. `~/inference/loadgen`의 다음 원칙을 분석해 우리 모듈에 필요한 부분만 적용한다.

- 요청 발행과 완료 보고를 분리한다.
- 각 요청은 불변의 query ID를 갖는다.
- 완료는 요청마다 정확히 한 번만 인정한다.
- outstanding 요청을 추적하고 `flush()`가 모든 terminal 상태를 기다린다.
- latency는 monotonic clock으로 요청별로 측정한다.
- 평균뿐 아니라 P50/P90/P95/P97/P99/P99.9를 보고한다.
- 처리율과 latency를 함께 기록한다.
- 최소 샘플 수와 최소 실행시간을 결과 신뢰성 조건으로 사용한다.
- queue, callback, timeout, out-of-order completion, worker failure를 테스트한다.

다음 항목은 구현하지 않는다.

- `mlperf_loadgen` 패키지 의존성
- MLPerf SUT/QSL API 또는 `LoadGenAdapter`
- MLPerf 로그 포맷 호환
- MLPerf의 공식 시나리오, early stopping, `VALID/INVALID` 판정 복제
- submission 디렉터리, audit, compliance 자동화
- LoadGen 전체 기능의 재구현

이 모듈의 결과는 MLPerf 결과가 아니며 문서와 저장 데이터에서 MLPerf 호환 또는 compliance를 주장하지 않는다. 기존 `framework/src/adapters/loadgen_adapter.py` 스켈레톤은 이번 구현 범위에서 제외하고 수정하지 않는다.

### 2.1 확인한 LoadGen 지표와 적용 여부

`~/inference/loadgen/results.h`, `results.cc`, `test_settings.h`, token metric demo를 기준으로 LoadGen이 보고하는 지표를 다음과 같이 분류했다.

| LoadGen 지표군 | 확인한 세부 지표 | 우리 모듈 적용 |
|---|---|---|
| 결과 신뢰성 | result validity, performance constraint 충족, min duration, min query count, early stopping, invalid reason | 자체 `valid/invalid`, 최소 조건, 불변식, invalid reason만 적용. MLPerf 판정식과 early stopping은 복제하지 않음 |
| 요청 수 | query count, Server over-latency query count | submitted/accepted/completed/failed/rejected/outstanding, 선택적 SLO 초과 수로 적용 |
| 공통 latency | min/max/mean, P50/P90/P95/P97/P99/P99.9 sample latency | e2e, queue wait, service, completion overhead 각각에 동일 통계 적용 |
| SingleStream | LoadGen overhead 포함/제외 QPS | 기존 e2e와 async 비교 지표로 대체. 동일 명칭은 사용하지 않음 |
| MultiStream | query min/max/mean 및 percentile | 1차 범위에서 제외. dynamic batch 통계로 필요한 정보를 기록 |
| Server | scheduled samples/s, completed samples/s, latency constraint, over-latency count | Server-like의 target/issued/completed QPS, scheduler delay, 선택적 e2e P99 SLO로 적용 |
| Offline | samples/s | async completed samples/s로 적용 |
| First token | TTFT min/max/mean 및 P50/P90/P95/P97/P99/P99.9 | 실제 streaming event가 있는 runtime에만 적용하는 계약 정의 |
| Token decode | TPOT min/max/mean 및 P50/P90/P95/P97/P99/P99.9 | 실제 또는 runtime-reported 값의 출처를 구분해 적용 |
| Token throughput | LoadGen overhead 포함/제외 throughput, completed tokens/s, Offline tokens/s | generated/completed tokens/s를 적용하되 LoadGen overhead 용어는 사용하지 않음 |
| Inferred token 지표 | inferred completed tokens/s, inferred Offline tokens/s | 1차 범위에서 제외. 실제 token count가 없는 값을 추론해 만들지 않음 |
| 비동기 제한 설정 | max async queries, issue query thread count, request coalescing | queue capacity, worker count, dynamic batch size/timeout으로 필요한 범위만 적용 |

LoadGen은 SUT 내부의 queue residence, runtime service, completion overhead, queue depth, worker utilization을 직접 알 수 없다. 이 값들은 우리 모듈이 추가로 계측하는 진단 지표다.

## 3. 범위와 비범위

### 3.1 포함 범위

- 프레임워크 내부 비동기 추론 엔진
- bounded queue와 명시적 backpressure
- worker lifecycle과 동적 배칭
- 단일 completion 처리 경로
- Offline형 및 Server-like 부하 producer
- 요청별 상태와 시간 측정
- latency, throughput, queue, worker 지표
- run 유효성 검사와 무효 사유
- 기존 evaluator, decoder, hardware monitor 연동
- CLI 옵션과 입력 검증
- CSV 요약, JSON sidecar, 선택적 JSONL request trace
- Mock Runtime 단위·통합 테스트
- ONNX Runtime CPU 통합 테스트
- 기존 `e2e` 경로 회귀 테스트

### 3.2 제외 범위

- Backend FastAPI 변경
- Frontend 변경
- 공식 MLPerf 실행 또는 결과 생성
- 가속기별 worker 동시성 최적화
- 자동 worker 수 탐색
- 자동 peak QPS 탐색
- 분산 큐, 멀티프로세스 큐, 네트워크 큐
- 실행 중 worker process 강제 종료
- vLLM의 실제 streaming generation 구현

## 4. 상위 아키텍처

```text
CLI
 ├─ --inference-mode e2e
 │    └─ BenchmarkRunner (기존 경로)
 │
 └─ --inference-mode async_queue
      └─ AsyncBenchmarkRunner
          ├─ WorkloadProducer
          │    ├─ OfflineProducer
          │    └─ ServerLikeProducer
          ├─ AsyncInferenceEngine
          │    ├─ bounded request queue
          │    ├─ dynamic batch assembler
          │    └─ runtime worker(s)
          ├─ CompletionCoordinator
          │    ├─ decoder
          │    ├─ evaluator
          │    └─ terminal state tracker
          ├─ AsyncMetricsCollector
          └─ AsyncResultArtifacts
               ├─ CSV summary
               ├─ JSON sidecar
               └─ optional JSONL trace
```

`AsyncInferenceEngine`은 runtime 실행과 worker lifecycle만 책임진다. 데이터셋 순회와 부하 생성은 `WorkloadProducer`, decoder/evaluator 직렬화는 `CompletionCoordinator`, 통계 계산은 `AsyncMetricsCollector`, 저장은 기존 `result_store`의 확장 기능이 담당한다.

## 5. 파일 경계

구현 시 다음 책임 경계를 사용한다.

- `framework/src/core/inference_pipeline.py`
  - 기존 e2e와 async가 공유하는 collate, runtime input 준비, run/generate 호출 결과 정규화
  - 현재 `BenchmarkRunner`의 private helper를 동작 변경 없이 이동
- `framework/src/core/async_inference/types.py`
  - 요청, batch, 완료, 오류, 설정, run 상태 데이터 타입
- `framework/src/core/async_inference/engine.py`
  - bounded queue, worker, batching, start/submit/flush/shutdown 상태 머신
- `framework/src/core/async_inference/completion.py`
  - decoder/evaluator 직렬 실행과 exact-once terminal 처리
- `framework/src/core/async_inference/metrics.py`
  - 요청·큐·worker·batch 통계와 percentile 집계
- `framework/src/core/async_inference/producers.py`
  - Offline 및 Server-like 요청 공급
- `framework/src/core/async_inference/runner.py`
  - warmup, monitor, producer, engine, completion, validity, 최종 metric 조립
- `framework/src/core/async_inference/trace.py`
  - 선택적 JSONL trace의 스트리밍 기록과 atomic finalize
- `framework/src/core/result_store.py`
  - 사전 할당 run ID, async metadata, JSON sidecar 저장
- `framework/src/core/benchmarkrunner.py`
  - 공유 pipeline helper를 사용하도록 내부 중복만 제거하고 기존 공개 동작 유지
- `framework/src/main.py`
  - 모드 선택, async 옵션 검증, runner 생성, 결과 저장

## 6. 핵심 데이터 계약

### 6.1 요청

`InferenceRequest`는 생성 후 변경하지 않는 dataclass다.

| 필드 | 타입 | 의미 |
|---|---|---|
| `request_id` | `int` | run 내부에서 0부터 증가하는 고유 ID |
| `sample_index` | `int` | DataLoader의 원본 sample index |
| `sample` | `Dict[str, Any]` | `input`, `label`, 선택적 metadata를 포함한 전처리 완료 샘플 |
| `scheduled_ns` | `int` | Server-like producer가 의도한 발행 시각. Offline은 `issued_ns`와 동일 |
| `issued_ns` | `int` | producer가 `submit()`을 호출한 시각 |
| `enqueued_ns` | `int` | bounded queue 진입이 완료된 시각 |
| `sample_count` | `int` | atomic request가 대표하는 실제 sample 수 |
| `task` | `Optional[str]` | 동적 배칭 호환성을 위한 선택적 task 식별자 |
| `generation_options` | `Optional[Dict[str, Any]]` | 동적 배칭 호환성을 위한 선택적 generation option |
| `batch_axis` | `Optional[int]` | 입력에 batch 축이 이미 존재할 때 제외할 명시적 축. 일반 단일 sample은 `None` |

`request_id`는 외부 데이터셋 index와 분리한다. 동일 sample이 반복 발행돼도 서로 다른 request ID를 갖는다.

### 6.2 실행 batch

`InferenceBatch`는 다음 정보를 보존한다.

- 순서가 유지된 `requests`
- collated runtime input
- request ID 목록
- sample index 목록
- batch seal 시각
- runtime 시작·종료 시각
- 실제 batch size

batch 결과가 out-of-order로 도착해도 ID 목록으로 원래 요청과 label을 연결한다.

### 6.3 완료

`BatchCompletion`은 성공 또는 실패 중 하나다.

- 성공: runtime outputs, runtime timing, generation timing metadata
- 실패: `error_type`, `error_message`, `worker_id`
- 공통: requests, runtime 시작·종료 시각, 실제 batch size

Exception 객체와 traceback 전체는 결과 JSON에 직렬화하지 않는다. 로그에는 traceback을 남기고 sidecar에는 오류 타입, 정규화된 메시지, 발생 횟수, 최대 5개의 request ID 예시만 저장한다.

## 7. Lifecycle과 상태 머신

엔진 상태는 다음과 같다.

```text
CREATED -> RUNNING -> DRAINING -> STOPPED
                  \-> FAILED -> DRAINING -> STOPPED
```

- `start()`는 `CREATED`에서 한 번만 허용한다.
- `submit()`은 `RUNNING`에서만 허용한다.
- `flush(timeout)`은 호출 시점까지 accepted된 요청이 모두 terminal 상태가 될 때까지 기다린다.
- `close_submission()`은 새 요청을 막고 `DRAINING`으로 전환한다.
- `shutdown()`은 queue drain 후 worker sentinel을 전달하고 join한다.
- `shutdown()`의 단일 absolute deadline은 함수 진입 시점에 생성하며, concurrent cancellation 대기 시간을 별도 timeout으로 더하지 않는다.
- worker가 sentinel을 dequeue하면 그 자리에서 queue task accounting을 정산한다. shutdown cleanup이 terminal epoch를 표시한 뒤 늦게 재개한 worker는 sentinel을 다시 게시하지 않으며, shutdown 반환 시점과 worker 종료 뒤 모두 request queue가 비고 unfinished task가 0이어야 한다.
- terminal shutdown은 request queue를 closed 상태로 바꾸고 모든 `take()` waiter를 broadcast로 깨운다. 물리 sentinel을 먼저 소유한 worker가 늦게 재개해도 closed predicate가 나머지 idle worker를 독립적으로 종료하므로 sentinel 재게시에는 의존하지 않는다.
- worker나 completion coordinator의 치명적 오류는 엔진을 `FAILED`로 전환하지만 이미 accepted된 요청은 terminal 상태로 정리한다.

worker가 blocking runtime call 안에서 영구 정지하면 Python thread를 안전하게 강제 종료할 수 없다. worker thread는 daemon으로 실행하며 `flush_timeout_sec`이 지나면 run을 invalid로 종결한다. 이 경우 runtime이 사용 중일 수 있으므로 `runtime.unload()`를 호출하지 않고 오류와 outstanding request ID를 저장한 뒤 CLI를 non-zero로 종료한다.

같은 제한은 decoder/evaluator처럼 framework가 호출한 임의의 외부 Python callback이 반환하지 않는 경우에도 적용된다. thread-only 구현은 callback stack이 잡고 있는 인자 참조를 안전하게 회수하거나 실제 callback 결과 없이 terminal 상태를 꾸며낼 수 없다. 이때 `shutdown()`은 제한 시간 안에 `False`를 반환하고 engine은 `FAILED`로 남으며, 아직 terminal이 아닌 outstanding request ID를 진단 데이터에 보존하고 CLI는 non-zero로 종료한다. callback gate가 나중에 풀리면 기존 `CompletionCoordinator`가 해당 요청을 정확히 한 번 terminal 처리하고 framework-owned queue/registry 참조를 정리한다. process 격리나 callback의 cooperative cancellation은 현재 core 범위 밖이다.

## 8. 부하 Producer

### 8.1 OfflineProducer

- `DataLoader.load_by_index()`로 sample을 읽어 준비되는 즉시 제출한다.
- queue가 가득 차면 `submit_timeout_sec`까지 block한다.
- timeout 안에 공간이 생기면 정상 제출한다.
- timeout이 지나면 해당 요청을 rejected로 종결하고 run을 invalid로 표시한다.
- 전체 dataset 또는 `--max-samples`에 도달하면 submission을 닫는다.

Offline은 최대 지속 처리량과 동적 배칭 효과를 확인하는 용도다. MLPerf Offline 시나리오를 구현하거나 그 결과와 동등함을 주장하지 않는다.

### 8.2 ServerLikeProducer

- `random.Random(schedule_seed).expovariate(target_qps)`로 재현 가능한 inter-arrival 간격을 만든다.
- 첫 요청 시각을 기준으로 절대 monotonic deadline을 누적해 drift를 줄인다.
- 각 deadline 전까지 `sleep()`하고, 늦은 경우 busy wait 없이 즉시 제출한다.
- queue가 가득 차면 block하지 않고 rejected로 종결한다. run은 invalid지만 producer는 설정된 종료 조건까지 계속해 포화 상태를 기록한다.
- `min_duration_sec`과 `min_samples`를 모두 만족할 때 정상적으로 submission을 닫는다.
- `max_samples`가 설정돼 먼저 도달하면 min 조건 미충족 여부를 validity에 반영한다.

Server-like는 서비스형 부하를 관찰하기 위한 제한된 자체 시나리오다. 자동 peak QPS 탐색, MLPerf의 request coalescing, early stopping은 포함하지 않는다.

### 8.3 측정 경계

- `DataLoader.load_by_index()`와 전처리는 request `issued_ns` 이전에 수행하며 request latency에서 제외한다.
- producer의 전체 wall duration에는 sample 준비 시간이 포함될 수 있으므로 `producer_load_ms`를 별도로 기록한다.
- warmup, runtime load, result save는 측정 구간에서 제외한다.
- hardware monitor는 첫 request 제출 직전 시작하고 flush 완료 직후 정지한다.

## 9. Queue, worker, 동적 배칭

### 9.1 Queue

- `queue.Queue(maxsize=queue_capacity)` 기반 bounded queue를 사용한다.
- worker 결과를 전달하는 completion queue도 `maxsize=worker_count`로 제한한다. coordinator가 느려지면 worker가 completion enqueue에서 block해 추가 output tensor 누적을 막는다.
- submission은 slot과 coordinator reservation을 먼저 확보한 뒤 lifecycle lock 밖에서 counter를 변경하지 않는 metrics availability preflight를 수행한다. preflight 중 close/shutdown된 transaction은 reservation과 slot을 취소하고, callback이 늦게 반환해도 stale reject한다.
- preflight가 끝난 transaction만 request queue에 visible publish한다. 이 실제 publication과 같은 queue mutex 구간에서 `enqueued_ns`, depth, 증가 sequence를 캡처하고, 외부 callback이나 logging이 없는 metrics 모듈의 sealed module-private acceptance primitive를 worker notification 전에 실행한다. engine은 public `claim_acceptance`, `commit_acceptance`, `finish_acceptance`를 호출하지 않으며 subclass override는 이 critical section에 진입할 수 없다. coordinator outstanding commit과 queue visibility도 같은 짧은 lifecycle commit 구간에 묶어 accepted-before-terminal 및 close-vs-submit 원자성을 보존한다.
- request queue dequeue와 drain도 실제 변경과 같은 queue mutex 구간에서 depth, monotonic timestamp, 증가 sequence를 캡처해 concurrent 전이가 중간의 낮은 depth를 숨기지 않게 한다. candidate dequeue timestamp/sequence는 pending ownership lock을 얻기 전 실제 제거 직후 캡처한다.
- dequeue에서는 첫 요청의 worker ownership 또는 candidate의 pending ownership을 먼저 확립하고, drain에서는 제거된 요청의 task accounting을 먼저 끝낸다. slot 반환과 failure-prone metrics callback은 non-reentrant queue mutex 밖에서 실행한다.
- engine state lock, completion coordinator condition, request queue mutex 안에서는 public/subclass-dispatch metrics 메서드를 호출하지 않는다. timeout, crash, membership 진단도 coordinator 상태를 캡처한 뒤 condition 밖에서 기록한다.
- queue-depth collector는 캡처한 전이를 sequence별 독립 event로 저장하고 finalize에서 정렬한다. sequence를 할당하는 순간 failure-prone delivery보다 먼저 sealed module-private primitive로 expected high-water를 기록하므로 마지막 또는 유일 callback이 block/실패해도 trailing gap을 검출한다. 저장량은 실제 관측 event 수에 비례하며 앞선 sequence를 기다리는 별도 pending backlog를 만들지 않는다. missing sequence, 같은 duplicate, conflicting duplicate를 구분해 진단하고, missing/conflict 또는 callback failure가 있으면 `metrics_unavailable`로 invalid 처리하며 queue depth mean/min/max를 정상값처럼 출력하지 않는다. finalize가 한 번 관측한 missing range는 run-level invalid 증거로 latch하며 나중 event가 도착해도 반복 finalize에서 제거하거나 depth 통계를 복원하지 않는다.
- `queue_capacity >= batch_size`를 검증한다.
- 메모리가 요청 수에 따라 무제한 증가하는 구조를 허용하지 않는다.
- 요청 payload는 terminal 처리 후 모든 참조를 제거한다.

### 9.2 Worker

- 기본 `worker_count=1`이다.
- Runtime base class는 기본적으로 concurrent run을 지원하지 않는다고 선언한다.
- `worker_count > 1`은 runtime이 concurrency capability를 명시한 경우에만 허용한다.
- 1차 구현에서는 Mock Runtime만 multi-worker capability 테스트를 제공한다.
- ONNX Runtime CPU CI는 단일 worker만 필수 검증한다.
- 향후 장치별 검증 없이 worker 수를 자동 증가시키지 않는다.

### 9.3 동적 배칭

- 기존 CLI의 `--batch-size`를 async mode의 `max_batch_size`로 해석한다.
- 기본값은 1이므로 초기 동작에는 batch wait가 없다.
- 첫 요청을 꺼낸 뒤 `batch_timeout_ms` 동안 compatible 요청을 최대 `max_batch_size`까지 모은다.
- input name, dtype, 명시된 batch 축을 제외한 shape, task, generation option이 같은 요청만 묶는다. 단일 sample처럼 입력에 batch 축이 아직 없으면 `batch_axis=None`으로 전체 sample shape를 비교한다.
- `batch_axis`가 명시된 compatible 입력은 해당 축으로 concatenate하며, `sample_count` 합이 `max_batch_size`를 넘기 전에 batch를 seal한다.
- acceptance 전에 declared batch-axis 길이와 `sample_count`가 같은지 검증하고, 단일 request의 실제 sample 수가 설정 또는 runtime cap을 넘으면 `invalid_request`로 reject한다.
- incompatible 요청은 다음 batch로 되돌릴 수 있도록 worker별 pending slot 하나에 보관한다.
- `is_static_batched=True`인 loader 결과는 하나의 atomic request로 취급하며 추가 동적 배칭을 적용하지 않는다.
- Runtime이 허용하는 batch 크기보다 큰 설정은 실행 전에 거부한다.
- NLP generation runtime이 batch generation capability를 선언하지 않으면 `batch_size=1`만 허용한다.

## 10. Completion 처리

모든 worker는 runtime 호출 결과를 completion queue에 넣고 다음 요청으로 이동한다. 단일 `CompletionCoordinator` thread가 다음 순서로 처리한다.

1. completion의 request ID가 accepted 상태인지 검증한다.
2. duplicate 또는 unknown completion을 기록하고 run을 invalid로 표시한다.
3. 성공 결과에 decoder를 한 번 적용한다.
4. 원래 request 순서의 label 및 preprocess context를 구성한다.
5. evaluator의 `add_batch()`를 한 번 호출한다.
6. request별 timing을 terminal state로 기록한다.
7. 선택된 경우 JSONL trace 한 줄을 기록한다.
8. payload 참조를 제거한다.

decoder 또는 evaluator가 실패하면 해당 batch의 요청을 failed로 종결한다. worker thread에서 evaluator를 호출하지 않으므로 기존 evaluator에 thread-safety를 요구하지 않는다.

coordinator가 stop 또는 자체 crash로 outstanding 요청의 실패 trace를 합성할 때 request별 `batch_size`는 상수 1이 아니라 그 요청의 실제 `sample_count`를 사용한다.

여기서 payload 참조 제거는 callback이 반환하거나 예외를 던져 terminal 처리가 가능한 경로의 framework-owned queue, registry, worker/coordinator local을 뜻한다. 외부 callback이 영구 block한 동안 그 callback stack이 보유한 인자 참조는 안전하게 제거할 수 없으며, 이 경우에는 7절의 invalid/outstanding 진단 계약을 따른다.

## 11. 시간과 지표 정의

모든 내부 시각은 `time.monotonic_ns()`로 기록하고 외부 출력에서 millisecond로 변환한다.

### 11.1 요청별 timing

| 지표 | 계산 | 의미 |
|---|---|---|
| `scheduler_delay_ms` | `issued - scheduled` | Server-like 의도 발행 대비 지연 |
| `submit_wait_ms` | `enqueued - issued` | queue 공간을 기다린 시간 |
| `queue_wait_ms` | `runtime_started - enqueued` | queue 및 batch coalescing 대기 |
| `service_time_ms` | `runtime_finished - runtime_started` | runtime 호출 시간 |
| `completion_overhead_ms` | `completed - runtime_finished` | decode, evaluate, terminal 처리 시간 |
| `e2e_latency_ms` | `completed - issued` | submit 호출부터 완료까지 |

정상 요청에 대해 다음 오차 허용 불변식을 검사한다.

```text
abs(e2e_latency_ms -
    (submit_wait_ms + queue_wait_ms + service_time_ms + completion_overhead_ms))
<= 0.05 ms
```

### 11.2 분포 지표

다음 timing 각각에 count, min, max, mean, P50, P90, P95, P97, P99, P99.9를 계산한다.

- scheduler delay
- submit wait
- queue wait
- service time
- completion overhead
- e2e latency

정확한 percentile 계산을 위해 Python `array('d')`에 millisecond 값을 저장하고 run 종료 시 NumPy percentile을 사용한다. 요청별 payload나 output은 누적하지 않는다.

### 11.3 처리량과 count

- `async_submitted_requests`
- `async_accepted_requests`
- `async_completed_requests`
- `async_failed_requests`
- `async_rejected_requests`
- `async_timed_out_requests`
- `async_outstanding_requests`
- `async_completed_samples_per_sec`
- `async_issued_requests_per_sec`
- `async_target_qps`와 `async_achieved_qps`의 차이
- 실제 batch count 및 batch size min/mean/max
- output sample 수와 evaluator sample 수

### 11.4 Queue와 worker 지표

- queue depth min/mean/max
- inflight min/mean/max
- queue full 발생 횟수
- submit block 총시간
- worker별 처리 batch 수와 sample 수
- worker별 busy time
- 전체 worker utilization
- flush duration

queue depth와 inflight 평균은 상태 변화 사이의 `value * duration` 면적을 측정 구간으로 나눈 time-weighted average다.

### 11.5 기존 품질과 하드웨어 지표

- 기존 evaluator의 정확도, F1, mAP, EM 등은 그대로 유지한다.
- hardware monitor의 `hw_*` 지표도 기존 방식으로 최종 결과에 병합한다.
- async timing 이름에는 `async_` prefix를 사용해 기존 `Average Latency (ms)`와 의미를 섞지 않는다.

## 12. LLM timing 계약

1차 production 검증은 ONNX Runtime CPU의 non-streaming task를 대상으로 한다. 다만 향후 LLM streaming을 수용할 수 있도록 다음 capability를 정의한다.

- `supports_streaming_generate() -> bool`: 기본값 `False`
- first-token event는 request ID와 `first_token_ns`를 전달한다.
- final completion은 generation 종료 시각, token count, 선택적 runtime TPOT metadata를 전달한다.
- 실제 first-token event가 없는 runtime은 TTFT event metric을 생성하지 않는다.
- 현재 `VllmRuntime.generate()`가 반환하는 추정 TTFT를 실제 first-token event로 변환하지 않는다.
- non-streaming generation은 기존 `GenerationResult`의 reported timing을 `reported_ttft_ms`, `reported_tpot_ms`, `timing_source`로 별도 저장한다.

Mock streaming runtime으로 first-token exact-once, final-before-first-token 오류, duplicate first-token, token count 불변식을 테스트한다. 실제 vLLM streaming engine 연동은 별도 후속 명세로 다룬다.

## 13. Run 유효성

우리 모듈의 상태는 lowercase `valid` 또는 `invalid`를 사용하며 MLPerf의 공식 판정과 구분한다.

### 13.1 Counter 불변식

```text
submitted = accepted + rejected
accepted = completed + failed + outstanding
flush 성공 후 outstanding = 0
terminal request ID는 정확히 한 번만 기록
```

accepted 계측은 extensible availability preflight와 sealed publication commit을 구분한다. lifecycle critical section의 commit은 public/subclass-dispatch 메서드, callback, logging을 포함하지 않는 module-private primitive만 사용한다. 따라서 public metrics override의 block, 예외, queue 재진입은 publication counter를 부분 변경하거나 queue/coordinator commit을 방해할 수 없다.

metrics availability preflight는 accepted counter를 변경하지 않으며 request queue, engine state, coordinator condition lock 밖에서만 실행한다. submission transaction은 `pending`에서 `accepted` 또는 `rejected`로 정확히 한 번만 전이한다. preflight가 shutdown deadline까지 반환하지 않으면 shutdown이 transaction을 `rejected`로 바꾸고 sealed rejection primitive로 counter를 즉시 commit한 뒤 reservation/slot을 회수한다. 따라서 shutdown 반환 시 `submitted = accepted + rejected`가 성립하며 callback이 영구 block돼도 counter invariant가 깨지지 않는다. 늦게 반환한 callback은 terminal transaction을 확인해 publish하거나 reject를 중복 기록하지 않는다. 실제 accepted counter와 queue-depth publication event는 worker visibility 직전 sealed commit에서만 함께 변경한다.

`timed_out`은 별도의 terminal category가 아니라 deadline을 넘긴 요청을 표시하는 진단 subset이다. 늦게라도 정상 완료된 요청은 `completed`와 `timed_out`에 함께 집계하고, runtime 오류로 끝난 요청은 `failed`와 `timed_out`에 함께 집계할 수 있다. 따라서 `timed_out`은 counter 등식에 더하지 않는다. timeout이 한 건이라도 있으면 run은 invalid다.

### 13.2 Invalid reason code

다음 중 하나라도 발생하면 run은 invalid다.

- `no_samples`
- `min_samples_not_met`
- `min_duration_not_met`
- `producer_error`
- `queue_submit_timeout`
- `request_rejected`
- `request_failed`
- `request_timeout`
- `flush_timeout`
- `duplicate_completion`
- `unknown_completion`
- `counter_invariant_failed`
- `timing_invariant_failed`
- `completion_thread_failed`
- `worker_shutdown_failed`
- `latency_slo_not_met`

trace 저장 실패나 sidecar의 선택 필드 누락처럼 추론 측정 자체를 훼손하지 않은 문제는 `warnings`에 기록한다.

### 13.3 신뢰성 기본값

- Offline: `min_samples=100`, `min_duration_sec=0`
- Server-like: `min_samples=100`, `min_duration_sec=10`
- CI smoke test는 명시적으로 더 작은 값을 전달한다.
- sample 수가 1000보다 작으면 P99.9는 계산하되 `tail_percentile_low_sample_count` warning을 남긴다.
- 선택적 `latency_slo_ms`가 설정되면 e2e P99가 SLO를 넘을 때 `latency_slo_not_met`로 invalid 처리한다.

## 14. 오류, timeout, backpressure

- producer 오류는 submission을 닫고 accepted 요청을 drain한 뒤 invalid로 종료한다.
- runtime 오류는 batch 전체를 failed로 종결하고 다음 batch 처리를 계속한다.
- 같은 worker에서 연속 3개 batch가 runtime 오류로 실패하면 engine을 `FAILED`로 전환한다.
- request timeout은 실행 중인 Python call을 강제 취소하지 않는다. 완료 후 timeout 여부를 기록하거나 flush timeout으로 run을 종결한다.
- Offline queue full은 설정 시간 동안 block하고 timeout 시 invalid다.
- Server-like queue full은 즉시 reject해 발행 스케줄을 보존하며 invalid다.
- `KeyboardInterrupt`는 새 submission을 중단하고 queued 요청을 cancelled failure로 terminal 처리한 뒤 실행 중 요청에 제한된 flush를 적용한다.
- 저장 과정은 measurement timing 밖에서 수행한다.

## 15. CLI 계약

기존 명령은 옵션을 추가하지 않아도 동일하게 동작한다.

### 15.1 공통 선택

```text
--inference-mode {e2e,async_queue}  # default: e2e
```

### 15.2 Async 전용 옵션

```text
--scenario {offline,server_like}    # default: offline
--target-qps FLOAT                  # server_like에서 필수, > 0
--queue-capacity INT                # default: 256, >= batch_size
--worker-count INT                  # default: 1
--batch-timeout-ms FLOAT            # default: 1.0, >= 0
--submit-timeout-sec FLOAT           # default: 30.0, > 0
--flush-timeout-sec FLOAT            # default: 300.0, > 0
--request-timeout-ms FLOAT           # default: 0, 0이면 비활성
--min-samples INT                    # scenario별 기본값 적용, >= 1
--min-duration-sec FLOAT             # scenario별 기본값 적용, >= 0
--max-samples INT                    # 선택, >= 1
--schedule-seed INT                  # default: 0
--latency-slo-ms FLOAT               # 선택, > 0
--save-request-trace                 # default: false
```

기존 `--batch-size`는 e2e에서 기존 batch size, async에서 dynamic max batch size다. `--max-steps`는 e2e 전용이며 async에서 사용하면 `--max-samples` 안내와 함께 입력 오류를 반환한다.

async 전용 옵션을 `e2e`와 함께 사용하거나 `server_like`에서 `target_qps`를 생략하면 runtime 초기화 전에 종료 코드 2로 실패한다.

## 16. 결과 저장

### 16.1 Run ID

run ID를 측정 전에 한 번 생성한다. 기존 `save_result()`에 선택적 `run_id` 인자를 추가하되 미지정 시 현재처럼 내부 생성해 하위 호환을 유지한다.

### 16.2 CSV

기존 `framework/results/benchmark_results.csv`에 다음 metadata를 추가한다.

- `inference_mode`
- `scenario`
- `queue_capacity`
- `worker_count`
- `batch_timeout_ms`
- `target_qps`
- `schedule_seed`
- `async_run_status`
- `async_invalid_reasons`
- JSON sidecar 상대 경로
- request trace 상대 경로

CSV에는 주요 count, throughput, P50/P95/P99, queue max, worker utilization만 저장한다. 모든 percentile과 세부 오류는 sidecar에 저장해 CSV column 증가를 제한한다.

### 16.3 JSON sidecar

경로는 `framework/results/details/{run_id}.json`이다. `schema_version="1.0"`을 포함하고 다음 section으로 구성한다.

- run metadata와 async config
- measurement 시작·종료·duration
- validity status, invalid reasons, warnings
- counter snapshot
- timing distribution 전체
- queue/inflight 통계
- worker별 통계
- batch size 통계
- failure summary
- quality metric summary
- hardware metric summary
- 선택적 LLM timing metadata

임시 파일에 쓴 뒤 `os.replace()`로 atomic finalize한다.

### 16.4 JSONL request trace

`--save-request-trace`일 때만 `framework/results/traces/{run_id}.jsonl`을 생성한다. 한 줄은 한 request의 ID, sample index, terminal status, timing, worker ID, batch size, 오류 요약만 포함한다. input, label, output tensor와 원문 prompt는 저장하지 않는다.

trace는 run 중 스트리밍 기록하고 주기적으로 flush하되 measurement thread를 막지 않도록 completion coordinator가 bounded trace queue에 전달하고 단일 writer thread가 기록한다. trace queue 포화 시 run 측정은 계속하고 warning과 누락 row count를 sidecar에 남긴다.

## 17. 기존 e2e 호환성

- `--inference-mode` 기본값은 `e2e`다.
- 기존 CLI 명령과 출력의 `RUN_ID=` 계약을 유지한다.
- 기존 evaluator metric 이름을 변경하지 않는다.
- `BenchmarkRunner.run()` 공개 signature를 변경하지 않는다.
- 공유 `inference_pipeline.py` 추출 전후의 기존 e2e 테스트 결과가 동일해야 한다.
- async metric은 `async_` prefix로 기존 runtime-only latency와 구분한다.
- 기존 CSV 행은 새 metadata column이 비어 있는 상태로 정상 조회돼야 한다.

## 18. 테스트 전략

### 18.1 단위 테스트

- request ID 생성과 불변성
- engine 상태 전이와 잘못된 API 호출 거부
- queue capacity와 Offline block timeout
- Server-like immediate reject
- batch size 충족 seal과 timeout seal
- incompatible input 분리
- exact-once completion
- duplicate/unknown completion invalid 처리
- worker exception 후 terminal 처리
- decoder/evaluator exception 후 terminal 처리
- flush 성공, flush timeout, shutdown join
- dequeue/drain metrics callback의 예외, re-entry, block과 late sentinel 재개
- queue-depth missing/duplicate/conflict sequence와 acceptance preflight의 queue re-entry, close/shutdown block
- public acceptance hook override의 block/re-entry 격리, trailing/only sequence high-water gap의 반복 finalize latch, 영구 block preflight의 shutdown-time exact-once rejection
- `worker_count >= 2`에서 terminal queue broadcast와 late sentinel owner 종료
- percentile과 time-weighted queue depth 계산
- counter 및 timing 불변식
- Server-like seed 재현성
- JSON sidecar schema와 atomic replace
- JSONL trace에 payload가 기록되지 않음
- CLI 조건부 옵션 검증

### 18.2 Mock Runtime 통합 테스트

- deterministic latency runtime으로 queue wait과 service time 검증
- batch-aware runtime으로 실제 batch size 검증
- concurrency-capable runtime으로 out-of-order completion 검증
- 매 N번째 요청을 실패시키는 runtime으로 no-deadlock 검증
- 영구 block을 모사한 runtime으로 flush timeout 검증
- streaming mock으로 first-token 계약 검증

### 18.3 ONNX Runtime CPU 통합 테스트

- 테스트가 임시 디렉터리에 작은 ONNX identity 또는 linear model을 생성한다.
- CPUExecutionProvider로 model을 load한다.
- 동일 입력에 대해 e2e와 async output 및 evaluator 결과가 일치해야 한다.
- `worker_count=1`, `batch_size=1`에서 모든 요청이 완료돼야 한다.
- `batch_size>1`에서 dynamic batch가 실제 ONNX call에 전달돼야 한다.
- 두 번 반복한 run이 동일 sample count와 품질 metric을 반환해야 한다.
- 성능 향상을 assertion으로 사용하지 않고 측정값의 유한성, 양수 여부, 불변식만 검사한다.

### 18.4 회귀 테스트

- 기존 framework test suite
- `python src/main.py`의 기본 e2e 경로
- result CSV read/write와 backend가 사용하는 `RUN_ID=` 출력
- `mlperf_loadgen`이 설치되지 않은 환경에서도 모든 async 테스트가 실행됨

## 19. 완료 기준

다음 조건을 모두 만족해야 1차 구현이 완료된다.

- 기본 CLI는 기존 e2e 동작을 유지한다.
- async Offline과 Server-like가 Mock Runtime에서 정상 종료한다.
- ONNX Runtime CPU에서 output과 품질 metric이 e2e 결과와 일치한다.
- accepted 요청이 유실되지 않고 exact-once terminal 상태를 갖는다.
- 정상 flush 후 outstanding이 0이다.
- worker 및 예외를 반환한 decoder/evaluator 오류가 영구 대기를 만들지 않는다. 반환하지 않는 외부 callback은 제한 시간에 invalid/FAILED로 반환하고 outstanding ID를 보존하는 7절의 제한 계약을 따른다.
- bounded queue가 capacity를 넘지 않는다.
- 필수 timing과 percentile이 유한한 값으로 저장된다.
- counter 및 timing 불변식이 자동 검증된다.
- CSV, JSON sidecar, 선택적 JSONL trace가 run ID로 연결된다.
- 기존 전체 테스트가 통과한다.
- MLPerf 의존성이나 호환성 주장이 추가되지 않는다.

성능 향상은 완료 조건이 아니다. e2e 대비 처리량, e2e latency, queue wait, service time의 차이를 결과로 제공하고 사용자가 target별 최적 설정을 판단할 수 있게 한다.

## 20. 기대 효과와 위험

### 20.1 기대 효과

- 순차 e2e가 숨기던 queue wait과 tail latency 관찰
- target QPS 대비 포화 시점 확인
- 동적 배칭을 통한 accelerator utilization 개선 가능성
- 입력 공급, runtime, completion 처리의 overlap
- bounded memory와 명시적 backpressure
- 장치별 worker 수와 batch wait 튜닝 기반 제공
- 품질 metric과 비동기 성능 metric의 동일 run 비교

### 20.2 위험과 완화

| 위험 | 영향 | 완화 |
|---|---|---|
| queueing으로 e2e latency 증가 | 기존보다 느려 보일 수 있음 | service와 queue latency를 분리하고 동일 조건 비교 |
| runtime thread-safety 불명 | crash 또는 잘못된 output | 기본 worker 1, capability opt-in |
| 과도한 batch wait | tail latency 증가 | 기본 batch size 1, timeout 별도 기록 |
| queue 무제한 증가 | OOM | bounded queue와 payload terminal release |
| evaluator data race | 품질 오염 | 단일 completion coordinator |
| worker 예외 | flush deadlock | 모든 경로 terminal 처리와 fault test |
| blocking runtime hang | 프로세스 종료 지연 | daemon worker, flush timeout, unload skip, non-zero exit |
| blocking decoder/evaluator callback | callback stack payload 유지, terminal 미확정 | shutdown timeout, FAILED/outstanding ID 보존, 거짓 terminal 금지, non-zero exit |
| percentile 표본 부족 | tail 수치 오해 | min sample 조건과 low-sample warning |
| 기존 latency와 혼동 | 잘못된 비교 | `async_` prefix와 latency scope 문서화 |
| trace I/O 간섭 | 성능 왜곡 | 기본 비활성, 별도 bounded writer queue |
| 향후 장치별 특성 차이 | 옵션이 역효과 | ONNX CPU 기준부터 시작해 장치별 명시적 검증 |

## 21. 구현 순서

세부 구현 계획은 이 명세 승인 후 별도 문서로 작성한다. 구현은 다음 독립 검토 단위로 나눈다.

1. 공유 inference pipeline 추출과 e2e 회귀 보존
2. 요청 타입, engine 상태 머신, exact-once completion
3. metrics와 validity
4. dynamic batching과 runtime capability
5. Offline/Server-like producer
6. AsyncBenchmarkRunner와 monitor/evaluator 통합
7. CLI와 result artifact 저장
8. ONNX Runtime CPU 통합 검증
9. 측정 방법 및 운영 문서 갱신
