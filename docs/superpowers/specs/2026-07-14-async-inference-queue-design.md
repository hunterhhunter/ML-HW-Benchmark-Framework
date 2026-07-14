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
- accepted/rejected counter, inflight 누적 면적, queue transition/high-water 증거는 collector 인스턴스 필드가 아니라 metrics 모듈이 collector identity로 보관하는 sealed state에 둔다. state마다 module-owned private lock을 사용하며 public `metrics.lock`, `metrics.counters`, `metrics.inflight`를 내부 transaction에서 읽거나 호출하지 않는다. accepted publication은 이 state의 primitive 값과 module-owned collection만 직접 변경하므로 public lock을 영구 점유하거나 public inflight gauge를 block/예외 객체로 바꿔도 deadline, queue 재진입, counter/inflight 불변식에 영향을 주지 않는다. public base metrics API와 finalize도 같은 sealed state를 사용하고 공개 counter 조회는 격리된 snapshot을 반환한다.
- submission-attempt token별 sealed outcome membership이 accepted/rejected의 authoritative commit point이며 request ID는 outcome payload로만 저장한다. 따라서 같은 request ID의 reject/reject와 reject/accept 재시도는 각각 독립 집계되고, 이미 accepted/outstanding인 ID의 중복 submission은 새 attempt rejection으로 남는다. membership 기록 뒤 파생 counter·inflight·queue evidence 단계에서 `BaseException`이 발생하면 engine은 같은 attempt token의 sealed outcome을 먼저 조회하고 같은 방향의 queue/coordinator/slot ownership만 완결한 뒤 idempotent rebuild한다. committed acceptance를 rollback/reject하거나 rejection accounting 전에 transaction terminal ownership을 해제하지 않는다. registry에 넣는 token, ID, reason, timestamp, depth, sequence, error evidence는 lock 획득 전에 exact built-in `int`/`float`/`str` 및 plain `dict`/`list`/`set`으로 정규화해 extension object나 collector 역참조를 보존하지 않는다.
- submission transaction은 queue-visible payload, coordinator reservation commit/abort, terminal flag, registry removal, slot release를 독립적인 재진입 가능 stage로 추적한다. 각 stage 전후의 `BaseException`은 authoritative attempt outcome과 coordinator membership을 확인한 뒤 아직 남은 matching stage만 재실행한다. reservation 소유권은 같은 ID의 기존 reservation membership만으로 추정하지 않고 reserve 전 availability와 이번 attempt의 결과를 함께 사용하므로 duplicate attempt가 원래 owner의 reservation을 abort하지 않는다.
- worker, batch, timing, terminal error aggregate도 module-owned plain primitive container에 저장한다. finalize는 sealed lock 안에서 immutable built-in snapshot과 primitive queue/inflight 계산만 수행하고 percentile·schema formatting은 lock 밖에서 실행한다. collector의 replaceable aggregate/summary object나 subclass method는 sealed lock 안에서 호출하지 않는다.
- shutdown stop token enqueue 실패는 `_control_lock` 안에서 boolean evidence만 캡처하고 `worker_shutdown_failed` 진단은 lock을 놓은 뒤 기록한다. first-token duplicate/invalid 판정도 tracker lock 안에서 상태만 캡처하고 public metrics 진단은 lock 밖에서 호출한다.
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

accepted 계측은 extensible availability preflight와 sealed publication commit을 구분한다. lifecycle critical section의 commit은 collector object에 노출되지 않은 module-owned state와 private lock에서 counter, inflight time-weighted gauge, queue transition/high-water를 직접 갱신하며 public/subclass-dispatch 메서드, callback, logging, 공개 필드를 포함하지 않는다. finalize와 정상 public base API도 동일한 state를 사용한다. 따라서 public metrics lock의 영구 점유, public inflight/counter object의 교체, override의 block·예외·queue 재진입은 publication counter를 부분 변경하거나 queue/coordinator commit을 방해할 수 없다.

accepted/rejected commit은 unique submission-attempt token별 outcome membership을 먼저 기록하고 request ID, normalized rejection reason, `request_rejected` evidence를 payload에 둔다. 파생 aggregate와 rejection diagnostics는 이 membership으로 함께 재구성 가능하므로 어느 단계에서 비동기 `BaseException`이 발생해도 query 후 retry가 중복 count나 reason/evidence 유실을 만들지 않는다. engine ownership은 outcome과 같은 방향의 queue visibility, coordinator commit/abort, terminal/pop, slot stage만 정리하며, shutdown 성공은 복구 뒤 counter invariant와 outstanding 0을 계속 요구한다. 내부 invariant 계산용 zero default는 기존 `details.counts` 출력 shape에 새 key로 노출하지 않는다.

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
- public acceptance hook 및 public lock/inflight replacement의 block·failure·re-entry 격리, trailing/only sequence high-water gap의 반복 finalize latch, 영구 block preflight의 shutdown-time exact-once rejection
- 같은 request ID의 reject/reject, reject/accept와 accepted/reserved/outstanding duplicate attempt의 독립 accounting
- accepted/rejected outcome rebuild 및 rejection reason/evidence diagnostics 전후의 `BaseException` 복구
- coordinator commit, queue publication, terminal flag/pop, reservation abort, slot release 각 stage 전후의 fault와 idempotent ownership 복구
- extension primitive/self-reference 입력 정규화와 weakref/GC registry cleanup
- worker/terminal/finalize aggregate replacement의 re-entry/gate 격리와 lock 밖 formatting
- stop enqueue 실패와 duplicate first-token 진단의 lifecycle lock 밖 re-entry/gate
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

## 22. R9 submission 복구와 소유권 불변식

### 22.1 Queue publication

queue publication의 exception-safe 구간은 `_put` 첫 mutation부터 transaction
payload 증거, unfinished-task 증가, transition sequence 할당, sealed accepted
outcome commit, worker visibility까지다. accepted outcome이 아직 없으면 queue
mutex 아래에서 동일 객체 membership을 제거하고 unfinished-task 값을 복원하며,
이미 할당된 sequence를 failed evidence로 남긴다. accepted outcome이 있으면 item과
task ownership을 보존하고 notification만 best effort로 재시도한다. 따라서 wrapper가
return 직후 중단되더라도 sealed outcome이 rollback/보존 방향을 유일하게 결정한다.

### 22.2 Coordinator registration

reservation은 request ID뿐 아니라 metrics outcome namespace에서 할당한 exact built-in
attempt token을 함께 저장한다. validate, commit, abort는 같은 token을 비교하며 stale
cleanup은 같은 request ID로 다시 생성된 reservation을 제거하거나 commit할 수 없다.
accepted 복구에서 matching reservation과 outstanding membership 외에 non-zero terminal
bitmap도 authoritative registration-complete evidence다. worker가 outstanding을 이미
pop한 뒤 submitter가 복구를 시작해도 transaction registry가 남지 않는다.

### 22.3 Slot capacity

capacity의 유일한 근거는 slot lease pool의 held-token membership 크기다. acquire는
attempt token 하나를 membership에 추가하고 release는 그 token을 한 번만 제거하는
idempotent transition이다. 별도 semaphore count나 transaction release flag를 진실
근거로 사용하지 않는다. release 내부 mutation 뒤 예외와 concurrent cleanup은 membership
재확인으로 해결하며 capacity를 늘리거나 lease를 누수하지 않는다. 기존 테스트와
호출자를 위한 `acquire`/`release` facade는 같은 pool을 사용한다.

### 22.4 Token, normalization, exception propagation

attempt token allocator는 collector별 sealed accounting state에 있어 같은 collector를
공유하는 여러 engine도 outcome key가 충돌하지 않는다. request ID와 attempt token은
extension `__int__`가 sealed metrics lock, engine lifecycle condition, coordinator condition
안에서 실행되지 않도록 lock 획득 전에 exact built-in `int`로 정규화한다. interruption
복구는 abort/release/terminal/pop ambiguity를 authoritative membership으로 정리한 뒤
최초 `KeyboardInterrupt`, `SystemExit` 등 원래 `BaseException`을 그대로 다시 발생시키며,
cleanup fault로 이를 가리거나 정상 `False` 반환으로 변환하지 않는다.

### 22.5 회귀 증거

fault test는 queue의 실제 mutation 경계, terminal-only accepted membership, slot removal
직후 fault와 concurrent release, reservation ABA, shared collector의 두 engine, numeric
subclass lock guard, cleanup 중 2차 `BaseException`을 포함한다. 임의 sleep 없이 event,
barrier, future와 membership assertion으로 동기화하며, 기존 Task 4/5 focused 계약과 전체
framework suite를 함께 반복 검증한다.

## 23. R10 불확실한 outcome과 보수적 복구

### 23.1 Four-state outcome query

sealed accounting outcome 조회 결과는 `accepted`, `rejected`, 명시적 absent, `UNKNOWN`
네 상태다. 조회 자체가 `BaseException`으로 실패한 경우 이를 absent로 변환하지 않는다.
queue mutex 안의 UNKNOWN은 item, unfinished task, transaction payload, reservation, slot
lease를 보존하고 최초 publication 예외를 그대로 다시 발생시킨다. engine 복구는 모든
lifecycle lock과 queue mutex를 해제한 뒤 outcome을 재조회한다.

재조회가 absent이면 보존한 queue item을 identity로 제거하고 task와 failed-sequence
증거를 정리한 뒤에만 rejection을 commit한다. accepted이면 coordinator registration을
먼저 복구하고 queue visibility를 다시 알린다. UNKNOWN 또는 후속 visibility/cleanup
fault가 반복되면 transaction을 `recovery_unresolved` diagnostic으로 남기고 engine을
`FAILED`로 전환한다. 이 diagnostic은 outstanding request ID 조회에 포함되며 shutdown은
이를 `STOPPED` 성공으로 덮어쓰지 않는다. secondary query/cleanup 예외는 최초
`KeyboardInterrupt`, `SystemExit` 또는 다른 `BaseException`을 가리지 않는다.

### 23.2 Lease acquisition ambiguity

slot acquire의 반환값을 local 변수에 대입하기 전에 held-token 추가 뒤 예외가 발생할 수
있다. submit exception cleanup은 local `acquired` 값이 아니라
`slot_pool.contains(attempt_token)`을 조회해 실제 membership이 있을 때만 release한다.
transaction 생성 전 fault와 생성 뒤 rejection cleanup 모두 같은 authoritative pool을
사용한다.

### 23.3 Coordinator token membership

accepted recovery의 outstanding 증거는 request ID membership만으로 충분하지 않다.
저장된 request의 exact built-in `submission_token`이 transaction attempt token과 같을
때만 matching outstanding으로 인정한다. 동일 ID의 replacement outstanding은 old
transaction을 완료할 수 없다. legacy `unregister_rejected`도 expected token을 필수로
받아 lock 획득 전에 ID/token을 정규화하고, matching reservation 또는 outstanding만
compare-and-remove한다.

### 23.4 Registry callback re-entry

collector weakref는 registry lookup 중 마지막 strong reference가 사라지면 cleanup
callback을 동기 실행할 수 있다. registry lock은 같은 thread의 cleanup 재진입을 허용해야
하며, callback은 여전히 exact weakref identity를 확인한 뒤 entry를 제거한다. 이 조건은
nonblocking re-entry 회귀 테스트로 검증한다.

## 24. R11 prepared visibility와 증거 결합

### 24.1 Lease membership UNKNOWN

slot acquire가 held-token 추가 전후에 예외를 발생시키면 engine은 facade나 교체 가능한
public method를 호출하지 않고 pool condition 아래의 internal held set을 직접 조회한다.
이 authoritative membership 조회도 `BaseException`으로 실패하면 결과는 별도 `UNKNOWN`
상태다. UNKNOWN에서는 lease를 release하지 않고 attempt-token transaction diagnostic을
보존하며 engine을 `FAILED`로 바꾼다. 해당 request ID는 outstanding 진단에 포함되고
shutdown은 반드시 `False`를 반환한다. membership 조회의 2차 예외는 submit의 최초
예외를 대체하지 않는다.

### 24.2 PREPARED, ACCEPTED_PREPARED, VISIBLE

bounded request queue의 request item은 identity별 visibility 상태를 갖는다. acceptance
outcome을 아직 증명할 수 없는 `PREPARED`와 acceptance는 commit됐지만 exact-token
coordinator registration이 아직 commit되지 않은 `ACCEPTED_PREPARED`는 consumer가
claim할 수 없다. registration의 reservation, matching outstanding, 또는 matching
terminal-token 증거가 확인된 뒤 동일 item만 `VISIBLE`로 전환하고 queue waiter를
notify한다. outcome absent 복구는 동일 item을 identity로 제거하고 unfinished task를
복원한다. outcome UNKNOWN은 item을 PREPARED로 유지한다.

`_take`는 raw `qsize()`가 아니라 queue head의 visibility를 wait predicate로 사용한다.
따라서 timeout이나 spurious notify는 PREPARED head를 dequeue하지 않으며, FIFO상 그 뒤의
VISIBLE item도 head가 해결될 때까지 claim되지 않는다. queue close는 unresolved prepared
payload를 drain/cancel하지 않고 worker를 종료시켜 shutdown failure 진단에 남긴다.

physical queue capacity에는 `PREPARED`, `ACCEPTED_PREPARED`, `VISIBLE` request가 모두
포함된다. queue-depth metric은 logical accepted occupancy로 정의하며 `VISIBLE`과
`ACCEPTED_PREPARED`를 포함하고 outcome-UNKNOWN `PREPARED`와 stop control token은
제외한다. acceptance transition에는 지금 acceptance를 commit하는 item을 포함한다.
dequeue/drain transition은 mutation 뒤 남은 logical accepted occupancy를 기록한다.

### 24.3 Failed sequence와 terminal token

absent rollback은 item 제거와 task balance 뒤에도 allocated queue sequence의 failed
evidence가 sealed metrics에 commit될 때까지 transaction payload와 sequence 목록을
지우지 않는다. evidence 기록의 2차 fault는 cleanup을 retryable하게 유지하며 두 번의
복구로도 확정할 수 없으면 `recovery_unresolved`, engine `FAILED`, shutdown `False`로
끝난다. 이 경우에도 caller는 최초 publication 예외를 받는다.

completion coordinator는 request별 terminal bitmap과 함께 registration의 exact
submission attempt token을 저장한다. accepted transaction의 terminal-only 복구는 bitmap이
non-zero인 것만으로 충분하지 않고 저장된 token이 transaction token과 정확히 같아야
한다. 같은 request ID의 replacement terminal은 이전 attempt의 coordinator ownership을
증명하지 않는다.

### 24.4 Weakref registry identity

sealed accounting collector registry cleanup은 실제 weakref callback과 GC로 검증한다.
callback은 자신의 exact weakref가 현재 identity entry와 동일할 때만 제거하며, 같은 key가
다른 reference로 교체된 경우 replacement entry를 보존한다.

## 25. R12 head wakeup과 token-bound terminal record

### 25.1 Visible head wakeup

queue head를 제거하는 모든 경로는 mutation 직후 새 head visibility를 다시 평가한다.
absent rollback 또는 worker dequeue로 새 head가 `VISIBLE`이나 stop control token이 되면
같은 queue mutex 아래에서 `not_empty.notify_all()`을 실행한다. 따라서 PREPARED A 뒤의
VISIBLE B를 A rollback이 드러내거나, B를 먼저 visible로 만든 뒤 A를 visible/dequeue하는
reverse 순서에서도 sleeping worker가 남지 않는다. 여러 waiter는 FIFO head를 하나씩
claim하며 각 dequeue transition의 logical depth는 마지막 item에서 0이 된다.

### 25.2 Single terminal authority와 registration 재조정

completion terminal truth는 request ID별 `_TerminalRecord` 하나다. record는 token binding
여부, exact submission attempt token, terminal state를 함께 소유한다. 기존 indexed
`terminal`과 `terminal_tokens` 조회는 record에서 파생되는 read-only compatibility view며
internal mutation 근거로 사용하지 않는다.

registration reservation은 같은 token의 재호출에 idempotent하다. commit은 다음 네
단계를 순서대로 수행하고 각 단계는 이미 완료된 상태에서 재호출해도 같은 결과를 낸다.

1. terminal record slot allocation
2. exact token binding
3. matching outstanding request publication
4. matching reservation removal

각 실제 mutation 전후의 `BaseException`은 accepted accounting을 rollback하지 않는다.
engine recovery는 matching reservation, outstanding 또는 bound terminal record 중 하나를
ownership evidence로 삼아 네 단계를 처음부터 재조정한다. completion이 이미 terminal
record를 commit하고 outstanding을 pop한 경우에는 outstanding을 되살리지 않고 stale
reservation만 제거한다. registration 전체 wrapper가 계속 fault를 발생시켜도 recovery는
sealed internal reconciliation을 사용하고, 개별 stage fault는 그대로 retry된다.

coordinator stop과 engine shutdown은 reservation mapping이 비어 있을 때만 성공한다.
outstanding이 없어 flush가 끝났더라도 reservation이 남으면 counter invariant diagnostic을
기록하고 `False`/`FAILED`로 종료한다.

### 25.3 Completion attempt membership

completion coordinator는 evaluator/decoder 호출 전에 incoming request ID와
`submission_token`을 exact built-in 값으로 lifecycle lock 밖에서 정규화한다. condition
안에서는 incoming token을 outstanding request token과 bound terminal record token 모두와
비교한다. 같은 ID의 stale attempt completion은 `stale_completion` invalid diagnostic만
남기고 evaluator, terminal state, replacement outstanding을 변경하지 않는다. 이후 exact
token completion은 정상적으로 evaluator를 한 번 호출하고 terminal record를 commit한다.
같은 batch의 duplicate/unknown member가 known member를 failure terminal로 만드는 기존
membership 계약은 유지한다.

## 26. R13 terminal-before-visibility와 outcome-first stage recovery

### 26.1 Exact terminal state가 visibility보다 우선한다

accepted submission 복구는 queue visibility를 열기 전에 coordinator condition 아래에서
request ID와 attempt token이 모두 일치하는 terminal record의 state를 읽는다. record가
`PENDING`이면 registration reconciliation을 마친 뒤 같은 condition을 유지한 채 queue
item을 `VISIBLE`로 바꾼다. `CLAIMED` 또는 `COMMITTED`이면 coordinator가 이미 terminal
ownership을 가졌으므로 registration을 되살리거나 worker 실행을 허용하지 않는다.

non-pending 복구는 prepared item을 object identity로 제거하고 unfinished task를 한 번
감소시킨다. 같은 queue mutex 아래에서 남은 accepted logical depth의 새 sequence를
할당하고 slot waiter 및 새 visible head waiter를 깨운다. mutex 밖에서는 exact attempt
slot lease를 idempotent하게 release하고 dequeue와 같은 depth metric을 commit한다. 마지막으로
submission transaction을 `accepted`로 terminalize하고 registry에서 제거한다. coordinator
crash가 outstanding을 먼저 terminalize한 경우에도 runtime과 evaluator 호출 수는 0이다.

### 26.2 RuntimeError도 authoritative outcome을 먼저 따른다

registration/publication stage의 `RuntimeError`는 예외 타입만으로 rejection 처리하지 않는다.
먼저 sealed attempt outcome을 조회한다. `accepted` 또는 조회 `UNKNOWN`이면 공통 복구 경로로
넘겨 accepted ownership만 reconciliation하고, cleanup이 끝난 뒤 최초 stage exception 객체를
그대로 다시 발생시킨다. `rejected`이면 matching rejected ownership만 정리한다. 명시적
outcome absent인 coordinator-unavailable 경로는 matching rejection을 commit하고 기존
`False` submit 계약을 유지한다. accepted outcome에 rejected accounting을 시도하거나 그
충돌 예외로 최초 RuntimeError를 가리는 경로는 없다.

### 26.3 FAILED stop의 residual reservation

completion coordinator가 이미 `FAILED`인 stop 진입, stop sentinel enqueue 실패, join 뒤
failure 판정은 모두 condition 안에서 residual reservation 여부를 snapshot한다. condition을
해제한 뒤 `completion_thread_failed`를 기록하고 reservation이 남았으면 추가로
`counter_invariant_failed`를 기록한다. metrics callback은 lifecycle condition 아래에서
실행되지 않으며 stop은 항상 `False`다.

## 27. R14 lifecycle-gated visibility와 staged terminal cleanup

### 27.1 FAILED/PENDING은 worker-visible이 아니다

accepted-prepared item의 visibility를 여는 데에는 exact request ID와 attempt token이 일치하는
terminal record의 `PENDING`뿐 아니라 coordinator lifecycle `RUNNING`도 동시에 필요하다.
두 조건은 coordinator condition 아래에서 검사하며, 그 condition을 유지한 채 queue의
prepared marker를 제거한다. coordinator가 `FAILED`, `STOPPING`, 또는 `STOPPED`인데 exact
record가 아직 `PENDING`이면 item은 prepared 상태로 남고 worker는 runtime을 실행할 수 없다.

non-running/PENDING recovery는 coordinator condition에서 terminal-state 변경을 기다린다.
transaction은 첫 wait에서 engine flush deadline을 한 번만 계산하고 모든 retry가 같은
deadline을 사용한다. deadline 전에 exact record가 `CLAIMED`나 `COMMITTED`가 되면 terminal
prepared cleanup으로 진행한다. deadline까지 `PENDING`이면 payload, slot lease, exact
transaction ownership을 보존한 `recovery_unresolved` 진단으로 engine을 `FAILED`에 두며
shutdown은 `False`다. `_fail_outstanding()`의 모든 terminal-state 변경은 condition waiter를
notify한다.

### 27.2 Terminal prepared cleanup의 commit stages

coordinator가 exact terminal ownership을 확보한 accepted-prepared item은 다음 순서를
authoritative commit stage로 사용한다.

1. queued request object identity 제거와 prepared-state 제거
2. queue unfinished-task balance
3. post-removal logical-depth transition sequence 할당과 allocation evidence
4. 동일 transition의 depth metric delivery
5. exact attempt-token slot lease release
6. submission transaction의 `accepted` terminal mark
7. exact transaction registry pop

각 stage는 다음 stage 전에 완료되며, retry는 완료된 stage를 건너뛴다. physical removal
evidence는 item을 제거한 같은 queue mutex 구간에서 transaction에 저장한다. transition은
sequence 할당 직후 transaction에 저장하고 allocation evidence가 완료됐는지도 별도 보존한다.
따라서 `_capture_transition()`이 sequence 할당 뒤 예외를 발생시켜도 retry는 새 sequence를
만들거나 이미 제거된 item을 다시 찾지 않는다. unfinished-task, depth evidence, terminal
mark와 registry pop도 각 commit flag로 idempotent하며, slot release는 pool의 held-token
membership을 authoritative하게 재조회해 interrupted return과 이미 완료된 release를 구분한다.

depth evidence는 slot release보다 먼저 commit한다. terminal cleanup의 어느 실제 mutation
직전이나 직후에 `BaseException`이 발생해도 caller에게는 최초 submission-stage 예외가
그대로 전달되고, 두 번의 common recovery는 동일 transaction evidence로 남은 stage만
완료한다. 성공한 retry 뒤 queue ownership, unfinished task, slot lease, transaction registry에
잔여물이 없고 accepted queue sequence는 연속적이다.

## 28. R15 exact tombstone, operation allocation, shutdown classification

### 28.1 Exact terminal-removal tombstone

terminal prepared cleanup은 deque identity를 삭제하기 전에 prepared-state authority에
request ID와 attempt token을 함께 담은 tombstone을 commit한다. tombstone state는 일반
prepared state와 마찬가지로 consumer-visible이 아니므로, physical identity가 잠시 queue
head에 남아 있어도 `take()`는 이를 claim하거나 runtime에 전달할 수 없다. 다른 attempt의
같은 request ID tombstone은 현재 transaction의 제거 권한이 아니다.

cleanup stage는 다음처럼 분리한다.

1. exact terminal tombstone commit
2. physical deque identity removal
3. tombstone state-map cleanup
4. unfinished-task balance
5. transition/depth/slot/transaction cleanup

각 stage는 앞 stage의 authoritative state를 확인한다. physical delete 뒤 transaction flag를
쓰기 전에 fault가 나면 exact tombstone과 deque absence가 제거 완료 증거다. state-map pop
뒤 fault가 나면 physical-removal stage와 state absence가 cleanup 완료 증거다. 따라서
`_clear_entry_state()`의 실제 mutation 뒤 `BaseException`도 missing-ownership으로 바뀌지
않고 retry가 task balance부터 계속된다.

### 28.2 Operation-key transition allocation

queue transition sequence의 authority는 선증가 integer counter가 아니라 queue별
`operation key -> immutable transition` membership이다. publish, dequeue, drain, terminal
cleanup은 각 logical operation에 안정적인 opaque key를 사용한다. depth와 timestamp exact
정규화, monotonic clock read, transition construction 같은 fallible 작업을 모두 끝낸 뒤
mapping assignment 하나로 allocation membership을 commit한다.

새 sequence는 committed mapping entry 수에서 파생한다. membership 전 fault는 record가
없으므로 같은 sequence로 다시 시도한다. membership 뒤 fault는 같은 key lookup으로 기존
depth, timestamp와 sequence를 그대로 재사용한다. 다음 operation만 그 다음 sequence를
받는다. publication rollback의 failed-sequence evidence도 counter 범위가 아니라 해당
operation record에서 얻는다. 이 계약은 accepted-terminal path뿐 아니라 direct/general
publish, worker dequeue와 drain transition에도 동일하다.

### 28.3 Shutdown의 sealed-outcome 분류

shutdown deadline까지 active submitter가 남으면 transaction을 일괄 reject하지 않는다.
각 exact attempt를 sealed accounting outcome으로 분류한다. outcome이 명시적으로 absent이고
transaction이 아직 queued payload, publication uncertainty, visible ownership, coordinator
commit 또는 terminal tombstone을 갖지 않는 preflight 상태일 때만
`submission_closed` rejection을 commit한다.

accepted outcome, outcome query `UNKNOWN`, `recovery_unresolved`, 또는 publication-recovery
evidence가 있는 transaction은 payload, lease, reservation과 registry identity를 보존한다.
이 소유권이 deadline에 남으면 shutdown은 `False`와 engine `FAILED`로 끝나며 accepted
attempt에 rejected accounting을 호출하지 않는다. coordinator가 `FAILED/PENDING`이고
submit recovery가 동시에 condition에서 기다리는 경우에도 shutdown은 bounded하며,
prepared item은 worker-visible이 되지 않는다.

## 29. R16 persistent dequeue와 drain operation

### 29.1 Worker dequeue는 mutation 전 exact operation을 소유한다

worker가 visible request를 dequeue할 때 queue mutex 아래에서 먼저 opaque operation key,
request object identity, exact request ID, exact attempt token, worker ID를 정규화한다. 같은
구간에서 dequeue 뒤 logical depth를 계산하고 monotonic timestamp와 immutable transition을
완전히 생성한 다음 `_DequeueOperation`을 queue operation map에 publish한다. clock read,
integer normalization, transition constructor가 실패하면 deque는 아직 변경되지 않는다.

operation publish 뒤에만 physical `_get()`을 수행한다. operation은 다음 stage를 각각
독립적으로 보존한다.

1. exact deque identity physical removal
2. prepared-state map cleanup
3. worker pending/owned handoff 또는 recovery handoff
4. exact attempt-token slot release
5. 동일 transition의 depth delivery 또는 failed-sequence evidence
6. unfinished-task balance와 operation retirement

`_take()`가 physical removal 뒤 request를 반환하기 전에 중단되어도 worker exception cleanup은
worker ID로 미완료 operation을 찾는다. record의 request가 deque에 없으면 removal이 이미
commit된 것이며, 남아 있으면 exact object identity만 제거한다. pending map과 local owned
목록에 같은 object가 함께 있어도 identity로 한 번만 terminalize한다. slot membership과
stage flags를 재확인해 lease와 unfinished task를 각각 한 번만 release/balance하고, 최초
worker exception type으로 exact attempt failure completion을 제출한다.

정상 completion은 coordinator submit 뒤 task-balance stage를 commit하고 나서 record를
retire한다. callback과 metrics delivery는 queue mutex 밖에서 실행하므로 queue metric의
reentrant `qsize()`와 기존 shutdown deadline 계약은 유지된다.

### 29.2 Drain/cancel은 visible payload snapshot을 영속화한다

drain caller는 stable operation key를 제공한다. queue는 첫 mutation 전에 현재 visible
request들의 exact object tuple, 정규화된 request ID tuple, attempt-token lease tuple과
post-drain immutable transition을 `_DrainOperation`으로 publish한다. stop token과 prepared
또는 tombstoned entry는 snapshot에 포함하지 않는다.

physical drain은 snapshot에 포함된 exact identity만 제거한다. queue head가 target일 때의
`_get()`이 removal 직후 중단되면 같은 operation key retry가 record를 다시 읽고 deque
absence를 commit evidence로 사용한다. retained sentinel이나 non-visible entry를 임시로
꺼냈다가 되넣지 않으므로 interruption이 control token 또는 prepared payload를 잃게 하지
않는다. 모든 target removal 뒤 aggregate unfinished-task balance를 한 번 수행한다.

queue mutex 밖에서는 snapshot index별 slot release, transition depth delivery 또는 failure
evidence, failure completion submit, cancellation completion을 별도 stage로 commit한다. public
cancel의 clock/constructor/after-remove one-shot fault는 같은 key로 즉시 resume한다. resume도
실패하면 engine은 `FAILED`가 되고 exact operation record는 진단과 shutdown recovery를 위해
남는다. 성공한 cancellation completion 뒤에만 drain record를 retire한다.

cancel이 physical removal 뒤 멈춘 동안 shutdown이 시작되어도 shutdown은 빈 deque를 별도
payload ownership으로 해석하거나 stop token으로 대체하지 않는다. 먼저 시작한 cancel이
영속 record로 exact failure completion을 끝내면 outstanding이 0이 되고, shutdown worker는
stop token만 소비한다. 따라서 cancel count, terminal count, slot lease, task balance가 모두
한 번이며 queue sequence도 연속적이다.

## 30. R17 atomic reservation, task-token ledger, terminal handoff

### 30.1 Queue entry reservation과 task-token authority

dequeue와 drain은 physical deque mutation 전에 같은 queue mutex 구간에서 exact object와
operation을 묶은 reservation state를 entry에 publish한다. reservation state가 있는 entry는
일반 visible head, worker claim, cancel/drain snapshot에서 제외된다. 최초 owner의 retry만
operation key와 exact operation identity로 같은 record를 재개한다. 따라서 worker가 `_get()`
진입 직전 또는 직후 중단되어도 concurrent cancel이나 다른 worker가 같은 payload에 별도
operation을 만들 수 없다.

queue task accounting은 raw integer decrement가 아니라 각 `_put()`에서 만든 opaque task-token
membership을 authority로 사용한다. deque와 평행한 token deque가 physical identity를 보존하고,
active token set membership을 한 번 제거하는 것이 task balance의 유일한 mutation이다.
`unfinished_tasks`와 join notification은 매 balance마다 active membership 크기에서 재구성한다.
membership 제거 직후 flag 기록 전에 fault가 나면 retry는 absent membership을 확인하고 같은 0
상태를 재구성하므로 underflow하지 않는다. request뿐 아니라 shutdown stop/control token도 같은
ledger에 포함되고 dequeue 또는 terminal discard에서 exact token으로 balance된다.

### 30.2 Exact completion handoff 뒤 operation retirement

dequeue operation은 physical removal과 state cleanup 뒤에도 다음 authority가 모두 commit될
때까지 남는다.

1. exact attempt slot lease release
2. queue-depth delivery 또는 failed-sequence evidence
3. task-token balance
4. worker pending map/local owned handoff cleanup
5. exact completion operation의 coordinator handoff

completion coordinator는 operation key별 canonical `BatchCompletion` journal을 유지한다. submit
retry는 이미 enqueue된 exact object identity 또는 journal의 committed flag를 조회해 같은
completion을 두 번 enqueue하지 않는다. queue put 뒤 caller return 전에 fault가 나도 handoff
evidence로 남은 cleanup을 계속한다. dequeue record가 모든 stage를 만족해 retire되면 journal도
acknowledge하여 제거한다. completion submit이 끝내 commit되지 않으면 task balance는 ledger로
안전하게 끝낼 수 있지만 dequeue/journal authority는 남고 engine은 `FAILED`, shutdown은
`False`다.

cancel/drain은 engine의 active drain key를 사용한다. public cancel 재호출과 shutdown은 queue가
이미 비어 있어도 같은 `_DrainOperation`, cancellation request tuple, completion key와 최초 error
metadata를 재사용한다. slot/depth/task stage, pending dequeue stage, cancellation completion을
차례로 재개하고 exact terminal handoff가 확인된 뒤에만 drain/dequeue record와 active key를
함께 retire한다. post-submit ambiguity도 coordinator journal로 판정하며, concurrent cancel은
active drain lock 아래 같은 operation을 직렬 재개한다.

### 30.3 Bounded transition operation authority

queue transition authority는 monotonic next-sequence high-water와 현재 미완료 operation mapping만
보존한다. allocation evidence는 별도 set이 아니라 mapping value의 flag로 통합한다. direct
publication은 visibility commit 또는 rollback evidence 뒤, terminal cleanup은 depth delivery 뒤,
dequeue와 drain은 모든 cleanup/terminal handoff 뒤 해당 allocation record를 제거한다. 저장된
terminal transaction transition은 allocation retirement 뒤 retry에서도 그대로 재사용하므로 새
sequence를 소비하지 않는다. 정상 완료된 요청 수가 증가해도 queue-side operation/evidence map은
O(requests)로 누적되지 않는다.

## 31. R18 단일 queue entry, 연속 handoff CAS, bounded drain deadline

### 31.1 Payload, task, state의 단일 물리 권위

request queue의 물리 원소는 `_QueueEntry(payload, task_token, state)` 하나다. raw payload
deque, 평행 token deque, active-token set, 별도 prepared-state map을 함께 갱신하지 않는다.
put은 entry 하나를 append하고 dequeue는 같은 entry 하나를 popleft한다. append 또는 popleft가
실제 deque mutation 뒤 예외를 발생시켜도 payload, task identity와 visibility state는 분리되지
않는다.

deque에서 제거된 entry는 task balance가 끝날 때까지 dequeue, drain, stop 또는 terminal
operation record가 직접 소유한다. compatibility `get()/task_done()`도 별도 operation record로
같은 entry를 보존한다. `unfinished_tasks`와 join notification은 물리 queue와 모든 live operation이
소유한 아직 balance되지 않은 entry 집합에서 재구성한다. shutdown 성공은 unresolved operation뿐
아니라 live task entry 수가 0임도 확인한다.

### 31.2 Completion handoff의 연속 상태 전이

operation-key journal은 queue put 전에 `ENQUEUING`으로 생성된다. 물리 completion queue에는
key와 canonical completion을 함께 가진 wrapper가 들어간다. coordinator는 wrapper를 get하기
직전에 같은 journal을 `DEQUEUED`로 전이하고, producer는 put return 뒤 상태가 아직
`ENQUEUING`일 때만 `ENQUEUED`로 CAS한다. coordinator의 exact terminal 처리 뒤 상태는
`ACKED`가 된다. 따라서 put mutation 뒤 producer가 멈추거나 dequeue가 producer의 CAS보다 먼저
끝나도 뒤의 상태를 과거 상태로 덮지 않으며, retry는 journal 또는 exact queued-wrapper identity를
조회해 duplicate completion을 만들지 않는다.

dequeue operation의 task balance와 retirement는 ACK 또는 coordinator failure 뒤의 exact
request/token terminal evidence를 확인한 다음 수행한다. operation retirement 뒤 journal을
제거한다. ACK와 journal pop의 실제 mutation 뒤 fault는 상태/absence 재조회로 완료를 판정한다.
ACK는 request queue waiter를 깨워 유휴 worker도 즉시 operation을 정리하게 하며, flush도 ACK된
operation을 수습한다. worker는 미완료 handoff에서 payload tuple을 보존하지 않고 key만 보존해
terminal flush 뒤 payload lifetime을 연장하지 않는다.

### 31.3 Active drain과 transition의 원자적 권위

cancel, resume, final drain은 모두 shutdown 또는 public-cancel의 absolute deadline을 전달한다.
active-drain lock은 nonblocking 또는 남은 시간만큼만 획득한다. 최초 resume가 concurrent owner를
발견하면 final bounded drain audit까지 판단을 유예할 수 있지만, deadline까지 lock이 busy이면
shutdown은 `False`로 끝나며 무기한 lock wait를 하지 않는다.

transition authority는 `_TransitionState(next_sequence, allocations)` 한 값이다. allocation은
immutable map copy에 operation key와 transition을 추가하면서 next high-water도 함께 증가시킨 뒤
한 번의 state swap으로 commit한다. swap 전 fault는 둘 다 남기지 않고, swap 후 fault는 mapping과
증가한 high-water를 함께 노출한다. evidence flag 갱신과 retirement도 map copy/state swap을 쓰며,
retirement는 mapping만 제거하고 next high-water는 보존한다.

## 32. R19 선행 dequeue authority, canonical cancel, stable stop publication

### 32.1 Dequeue operation은 transition allocation보다 먼저 존재한다

worker dequeue는 queue mutex 아래에서 opaque operation key, exact queue entry, worker/request/token
identity와 post-dequeue logical depth를 먼저 `_DequeueOperation`에 저장한다. 같은 operation 객체를
가리키는 reservation을 exact entry에 기록하고 operation map membership을 commit한 뒤에만
transition state swap을 시도한다. 따라서 이 시점의 operation transition은 `None`일 수 있다.

swap 전 fault로 allocation membership이 없으면 reservation과 operation을 함께 rollback하여 entry를
다시 visible하게 한다. swap 후 fault면 stable operation key로 allocation을 조회해 immutable
transition을 operation에 연결하고 operation/reservation을 그대로 보존한다. 같은 worker의 retry와
exception recovery는 그 key로 allocation evidence까지 완료한 뒤 physical removal을 재개한다.
operation transition과 allocation transition identity가 다르거나 operation만 있는데 allocation이
없으면 새 sequence를 만들지 않고 authority 오류로 처리한다. 정상 handoff/task/terminal cleanup이
끝나면 operation과 allocation을 함께 retire한다.

shutdown final audit은 operation map과 live task entry뿐 아니라 남아 있는 transition allocation
membership 자체도 검사한다. 따라서 post-swap fault 뒤 operation이 복구 가능하고, owner 없는
allocation만 남은 상태는 성공으로 보고되지 않는다. accepted-prepared item이 registration 중 이미
terminal이 된 경로는 exact terminal state cleanup에서 원래 publication allocation도 retire하여 이
audit을 만족한다.

### 32.2 Cancel/drain은 하나의 canonical completion을 보존한다

cancel은 첫 completion submit 전에 exact request tuple, completion operation key, 최초 error metadata와
canonical `BatchCompletion`을 `_CancellationOperation`에 저장한다. visible drain이 있으면 같은 객체를
그 `_DrainOperation`에도 저장한다. coordinator journal까지 세 authority는 동일한 completion 객체와
operation key를 가리킨다.

completion queue가 가득 차 submit deadline이 끝나면 journal은 `ENQUEUING`으로 남고 cancel/drain
record는 retire하지 않는다. capacity가 열린 뒤 public cancel 또는 shutdown resume는 새 completion을
만들지 않고 record의 동일 객체를 submit한다. coordinator의 exact ACK 또는 terminal evidence 뒤에만
dequeue/drain cleanup, journal acknowledge와 active cancellation retirement를 수행한다. 마지막 record가
사라지면 canonical completion과 그 payload tuple도 release된다.

### 32.3 Compatibility task_done은 balance와 retirement를 분리한다

compatibility `_get()`은 exact entry를 가진 `_CompatibilityOperation`을 global operation map과 호출
thread의 retry stack에 함께 기록한다. `task_done()`은 stack top을 미리 pop하지 않는다. 먼저 entry의
task balance를 commit하고, 다음으로 exact compatibility map membership을 retire한 뒤 두 stage가 모두
끝났을 때만 stack handle을 제거한다.

balance mutation 뒤 fault면 entry의 `task_balanced` evidence로 operation stage를 commit하고 retry는
strict decrement를 반복하지 않는다. map pop 뒤 fault면 exact membership absence로 retirement를
인식하며, retry는 handle만 제거한다. 이 순서 때문에 두 fault 모두 unfinished task underflow 없이
같은 호출 권한으로 수습된다.

### 32.4 Shutdown stop publication은 exact entry evidence를 사용한다

shutdown은 `_STOP` append 전에 opaque key와 prebuilt `_QueueEntry`를 가진
`_StopPublicationOperation`을 queue authority map에 등록한다. 기존 queue `put()` hook을 유지하면서
그 exact entry를 append하고, append의 `finally` 구간에서 queue mutex를 잡은 상태로 exact identity
membership을 확인해 `publication_committed`를 기록한다.

append가 실제 mutation 뒤 `BaseException`을 던지면 engine은 operation entry의 commit evidence를
조회한다. committed이면 worker가 이미 소비했더라도 같은 entry evidence로 성공한 publication을
인정하고 bounded shutdown을 계속한다. absent이면 uncommitted operation을 abort하고 entry를 balanced
상태로 닫은 뒤 shutdown을 `False`로 끝낸다. committed entry는 physical stop dequeue 또는 final
discard가 task를 balance한 뒤 publication operation을 retire한다. 최종 audit에는 stop publication,
stop dequeue와 live task authority가 모두 포함되므로 stop entry나 task count가 누락된 성공은 없다.

## 33. R20 cancellation generation과 compatibility get recovery

### 33.1 Active cancellation은 immutable generation이다

`_CancellationOperation`은 exact request tuple, canonical completion, completion operation key와 함께
그 tuple을 물리적으로 소유한 exact drain operation key도 저장한다. active generation이 존재하는
동안 cancel retry는 이 record의 request와 저장된 drain key만 읽는다. 이 단계에서는 visible queue를
drain하거나 새 worker pending request를 claim하지 않는다.

따라서 A의 canonical completion이 full completion queue에서 `ENQUEUING` timeout인 사이 B가 accepted
되면 B는 기존 queue entry/reservation/task/slot 또는 worker pending/dequeue authority에 그대로 남는다.
A retry는 A 객체와 key만 coordinator에 다시 submit하고, A의 ACK 또는 exact terminal evidence 뒤
dequeue/drain/journal과 active generation을 retire한다. 같은 cancel 호출의 absolute deadline이 남아
있으면 그때 B를 별도 drain하여 새 operation key, 새 `_CancellationOperation`, 새 canonical
`BatchCompletion`을 만든다. deadline이 끝났으면 B를 건드리지 않고 다음 cancel/shutdown recovery가
claim할 수 있게 둔다.

generation별 request membership은 생성 뒤 변경하지 않는다. 새 drain operation을 old generation의
completion으로 표시하거나 old tuple로 새 drain 결과를 덮어쓰는 동작은 금지한다. generation chaining은
항상 최초 caller deadline을 공유하므로 계속 유입되는 submission 때문에 cancel이 무기한 연장되지 않는다.
각 accepted request는 정확히 한 canonical cancellation completion/operation 또는 그대로 복구 가능한
queue/pending authority를 가진다.

### 33.2 Compatibility get은 popleft 전에 retry authority를 publish한다

compatibility `get()`은 visible head의 exact `_QueueEntry`에 대해 `_CompatibilityOperation`과 opaque
operation key를 먼저 만든다. 같은 queue mutex 구간에서 entry에 exact operation reservation을 기록하고,
global compatibility map membership과 호출 thread의 get-retry handle을 commit한 뒤에만 popleft를
시도한다. reservation head는 worker와 다른 compatibility consumer에게 visible하지 않다.

operation은 physical removal, reservation cleanup, caller return, task balance, map retirement를 별도
stage로 기록한다. popleft가 entry를 실제 제거한 뒤 `BaseException`을 던지면 exact identity가 deque에
없는 것이 removal commit evidence다. operation은 removed entry와 아직 balance되지 않은 task를 계속
소유하고 capacity waiter를 깨우며, get-retry handle은 유지된다. mutation 전 fault면 exact reserved
entry가 deque에 있으므로 같은 handle이 popleft를 다시 수행한다.

동일 thread의 다음 nonblocking 또는 bounded `get()`은 queue empty/timeout 판정보다 먼저 get-retry
handle을 확인한다. 이미 제거된 exact entry의 reservation을 정리하고 그 payload를 한 번만 반환한 뒤
handle을 기존 returned-operation stack으로 이동한다. 이후 `task_done()`은 R19의 balance/retirement
stage를 사용해 task를 한 번만 balance하고 global operation을 retire한다. 두 번째 get은 payload를
중복 반환하지 않으며 마지막 상태에는 compatibility operation, live task entry와 unfinished task가 없다.

## 34. R21 compatibility retry의 visible-successor wake

### 34.1 Physical removal commit이 consumer visibility를 깨운다

pre-mutation `popleft()` fault 뒤 reserved head X는 deque에 남고, X의 get-retry handle을 가진 owner만
물리 제거를 재개할 수 있다. 다른 compatibility consumer는 X 뒤의 Y가 이미 queue에 있어도 visible-head
predicate가 false이므로 `not_empty`에서 기다린다. Owner retry가 X를 성공적으로 제거하는 순간 Y가
visible head가 되므로, 같은 queue condition mutex 구간에서 `physical_removed=True`를 commit한 직후
`_head_is_visible()`을 다시 평가하고 참이면 `not_empty.notify_all()`을 호출한다.

notification은 caller return이 아니라 physical-removal stage 전이에 묶인다. 정상 최초 제거와
pre-mutation retry는 그 stage를 commit한 invocation에서 한 번만 wake한다. post-mutation fault는 기존
exception recovery가 exact X absence를 확인하며 stage와 wake를 함께 commit하고, 이후 retry는 이미
`physical_removed=True`이므로 같은 wake block을 다시 실행하지 않는다. Waiter는 같은 mutex를 다시
획득한 뒤에만 진행하므로 owner의 reservation cleanup/return stage와 병렬로 queue state를 관찰하지 않는다.

### 34.2 Bounded/unbounded waiter와 authority retirement

reserved X, queued Y, parked second consumer 순서를 event로 고정한 회귀는 unrelated put이나 sleep 없이
owner retry만으로 unbounded `get()`과 bounded `get(timeout=...)`이 모두 깨어나 exact Y를 한 번 반환함을
검증한다. X와 Y는 각각 originating thread의 compatibility stack으로 이동하고 `task_done()`의 balance와
map-retirement stage를 정확히 한 번 수행한다. 마지막 nonblocking get은 empty이며 compatibility operation,
live task entry, unfinished task와 task token은 모두 0이다. 물리 queue도 empty/not-full이어서 두 bounded
queue-capacity slot이 모두 복구된다. Compatibility queue API 자체는 engine `_SlotLeasePool` authority를
생성하지 않는다.

## 35. Task 6 review: producer boundary와 completion handoff retirement

### 35.1 Flush와 worker는 exact handoff retirement evidence를 공유한다

worker는 dequeue operation에 completion operation key를 bind한 직후, coordinator submit 전에 그
exact key를 worker-local handoff authority로 등록한다. Flush와 worker의 retirement는 동일한
retirement lock으로 직렬화한다. Flush가 `ACKED` journal을 먼저 finalize/acknowledge할 때는 그 key가
worker-local authority에 실제로 등록되어 있는 경우에만 exact retired evidence를 남긴다. 이후 worker는
자신의 pending list에서 같은 key를 만났을 때 그 evidence를 한 번 소비하고 정상 retirement로 인정한다.

임의 key의 coordinator state가 처음부터 `None`인 것은 성공 증거가 아니다. 성공은 coordinator의 exact
acknowledge 반환값, 또는 호출 직전 `ACKED`였던 동일 key가 호출 중 `None`으로 전이된 증거로만 판정한다.
worker-local authority가 아닌 exception recovery handoff에는 flush-retired evidence를 만들지 않는다.
shutdown 성공 전에는 worker-local key, flush-retired evidence, dequeue operation, completion journal이
모두 비어 있어야 한다. 이 규칙은 static/non-static producer가 실제 engine/coordinator를 통과하는
경합에서도 request failure 없이 한 terminal outcome만 남기도록 한다.

### 35.2 Offline producer는 유한 dataset 경계를 넘지 않는다

Offline producer의 발행 수는 `max_samples`가 없으면 `total_samples`, 있으면
`min(total_samples, max_samples)`다. sample index는 `0`부터 그 상한 직전까지 한 번씩 사용하며 dataset을
modulo로 반복하지 않는다. Server-like producer만 duration/QPS 실행을 위해 dataset index를 순환한다.

`min_samples`와 non-`None` `max_samples`는 Python 또는 NumPy의 양의 integral 값이어야 한다. `bool`,
0/음수, fraction, NaN/Infinity, 문자열과 `None`인 `min_samples`는 producer 실행 전에 거부한다.
ETTm loader metadata의 `total_samples`는 실제 usable window 수인 `window_count`와 같다.

static indexed sample은 input과 label의 ndarray 및 dict-of-array leaf storage를 producer-owned copy로
만든 뒤 leading one-item batch dimension을 붙인다. 따라서 loader buffer의 후속 mutation은 accepted
request를 바꾸지 않는다. `producer_load_ms`는 `load_by_index()`뿐 아니라 이 copy/normalization 준비까지
포함하고, Offline `scheduled_ns`/`issued_ns`는 준비가 끝난 뒤 같은 시점에 기록한다. Submit 대기 시간은
계속 producer load 시간에서 제외한다.

## 36. Task 7 orchestration review 보강 계약

### 36.1 Run 소유권과 입력 검증

`AsyncBenchmarkRunner` 인스턴스는 evaluator와 monitor처럼 run별 mutable 상태를
소유하므로 한 번만 실행할 수 있다. `run()`은 config와 warmup 횟수를 먼저 검증한
뒤 lock 아래에서 one-shot claim을 획득한다. 따라서 잘못된 입력은 부작용 없이 다시
시도할 수 있지만, claim 이후 성공·실패한 run은 같은 runner에서 재실행하지 않는다.
`InferencePipeline` 생성과 loader metadata 조회도 이 검증과 claim 뒤에만 일어난다.

`AsyncInferenceConfig`는 문자열 enum을 암묵적으로 받지 않는다. scenario는
`AsyncScenario`여야 하고 count와 seed는 bool을 제외한 Python/NumPy integral이어야
한다. duration, rate, timeout, SLO는 bool을 제외한 finite real이어야 하며 각 필드의
0 허용 여부를 적용한다. NaN, 양·음의 무한대, fractional count, numeric string은
공개 API에서도 거부한다.

### 36.2 Partial start와 cleanup

engine start 이후 모든 단계는 runner의 outer cleanup guard 안에 있다. start가
coordinator만 시작하고 실패했거나 startup metric event를 이미 남겼더라도 measurement
fallback은 기존 event를 보존하며 primary error를 바꾸지 않는다. runner는 close,
flush, monitor stop(시작을 시도한 경우), shutdown을 각각 독립적으로 시도한다.

engine은 실제로 시작된 coordinator, completion monitor, worker를 추적해 시작되지 않은
thread를 join하지 않는다. runner가 먼저 `close_submission()`해 DRAINING이 된 경우
`shutdown()`은 close를 다시 호출하지 않으며, standalone RUNNING shutdown은 기존처럼
한 번 close한다.

### 36.3 외부 callback 제한시간

`monitor.start`, `monitor.stop`, `monitor.summary`, `evaluator.compute`는 호출 직전에
`monotonic + flush_timeout_sec`로 계산한 절대 deadline 안에서만 기다린다. callback은
daemon thread에서 실행되며 Python thread를 강제 종료할 수 없으므로 timeout 결과는
callback ID, phase, thread name, timeout type/message와 반환 시점의 live outstanding
목록을 기록한다. 늦은 return/exception은 invocation-private 저장소에만 남아 이미
반환한 result를 변경할 수 없다. monitor stop timeout도 engine shutdown을 건너뛰지
않으며 callback timeout은 `callback_timeout` invalid reason이 된다.

## 37. Task 7 orchestration review round 2 보강 계약

### 37.1 Monitor lifecycle은 하나의 직렬 daemon lane을 사용한다

hardware monitor의 `start`, 보상 `stop`, `summary`는 run별 단일 FIFO daemon lane에서
실행한다. `start`가 deadline을 넘으면 runner는 즉시 같은 lane의 뒤에 보상 `stop`을
예약한다. 따라서 늦게 끝난 start와 stop이 겹치지 않고, summary는 반드시 그 stop 뒤에
실행된다. 실제 `HWMonitor`도 collector start가 늦게 반환하면 polling thread를 만든 뒤
같은 lane의 stop이 collector와 polling thread를 닫은 다음에만 summary를 호출한다.

각 caller wait와 lane close wait는 독립적인 absolute deadline으로 제한한다. 영원히
끝나지 않는 start는 running start와 그 뒤에 queued stop/summary를 callback ID, phase,
lane thread, 상태와 함께 outstanding 진단에 남긴다. 정상 lane은 close sentinel까지
소비하고 runner 반환 전에 종료한다. callback job의 결과와 예외는 invocation-private
저장소만 갱신하므로 timeout 뒤의 늦은 완료가 이미 조립된 benchmark result를 바꾸지
않는다.

### 37.2 Callback 결과 변환은 hostile object에도 total이다

evaluator와 monitor 결과는 최종 metric merge 전에 total serializer를 통과한다.
mapping의 `items()`/iterator/item unpack/key 문자열화, enum `value`, numeric `item()`,
`tolist()`, 중첩 iterable의 생성과 `next()`, fallback `str()`/`repr()`를 각각 보호한다.
어느 단계가 `BaseException`을 던져도 run을 중단하지 않고 결정적인
`<serialization_error>` 계열 placeholder를 넣는다. cycle도 같은 방식으로 닫는다.

각 실패는 phase, object path, operation, 안정적으로 변환한 error type/message를
`details.serialization_errors`에 남기고 `result_serialization_failed` invalid reason을
추가한다. 최종 `metrics`와 `details`도 같은 변환을 거치므로 hostile key/value와 중첩
실패가 함께 있어도 `json.dumps(..., allow_nan=False)`가 가능하다.

### 37.3 Never-started coordinator와 close 관찰은 lock/exact-once 계약을 지킨다

아직 thread를 시작하지 않은 `CompletionCoordinator.stop()`은 condition 아래에서 state,
reservation 존재 여부와 notification만 commit한다. `add_invalid_reason()` 같은 public
metrics hook은 condition을 놓은 뒤 호출하므로 reentrant metrics가 coordinator 상태를
조회해도 deadlock하지 않는다. stop은 남은 reservation을 임의로 지우지 않으며 exact
request/attempt token 소유자가 `abort_registration()`으로 정리할 때까지 보존한다.

runner는 public `engine.close_submission()`을 정확히 한 번 시도한다. override가 state
transition 전에 실패해도 runner cleanup은 engine-owned internal transition으로
RUNNING을 DRAINING으로 바꾸고 notify한 뒤 flush/shutdown을 계속한다. 그래서 뒤의
shutdown이 public close를 관찰 가능하게 재호출하지 않는다. 반면 RUNNING 상태에서 직접
호출한 standalone `shutdown()`은 기존처럼 public close를 한 번 수행한다.

## 38. Task 7 orchestration review round 3 보강 계약

### 38.1 Monitor lane은 warmup 뒤 획득하고 call scope가 무조건 해제한다

config/warmup 검증, one-shot claim, lazy pipeline/engine 구성과 성공한 warmup까지는 monitor
callback lane을 만들지 않는다. 따라서 warmup load/collate/prepare/runtime 예외에는 닫을
thread 자체가 없다. 그 뒤 monitor가 있을 때만 lane을 만들고 call-local owner에 즉시
등록한다. public `run()`의 outer `finally`가 정상 result, engine fatal, evaluator
`BaseException`, monitor summary와 최종 result assembly의 어느 경로에서도 bounded close를
시도한다. 두 번째 concurrent `run()`은 첫 run의 owner를 공유하거나 지울 수 없다.

monitor start를 시도한 뒤에는 기존 직렬 보상 계약을 유지한다. start timeout이면 같은
FIFO lane에 stop을 즉시 예약하고, 그 밖의 start/producer/cleanup fatal 경로도 lifecycle
cleanup에서 stop을 시도한 뒤 outer owner가 close sentinel을 예약한다. 정상 lane은 반환 전
종료하고 영원히 막힌 lane만 R2의 명시적 outstanding 진단과 daemon 제한을 유지한다.

### 38.2 Result serializer는 closed-world와 유한 budget을 적용한다

serializer는 exact `None`/`bool`/`int`/`float`/`str`, exact `dict`/`list`/`tuple`, exact
NumPy `ndarray`와 명시적으로 허용한 exact NumPy bool/integer/float/string scalar만 읽는다.
custom subclass, `Mapping`, `Iterable`, enum, set, complex와 그 밖의 객체에는 `items`, iterator,
`next`, `item`, `tolist`, numeric conversion, `str` 또는 `repr`을 호출하지 않는다. 지원하지
않는 값과 key는 type/path/operation을 가진 `SerializationUnsupportedType` 진단과 결정적인
`<serialization_error>` placeholder로 즉시 바꾼다. NumPy array의 `tolist`만 exact ndarray에
대해 size/depth를 확인한 뒤 trusted operation으로 호출한다.

각 root conversion은 최대 depth 32, item 10,000, ndarray element 4,096 budget을 새로
적용한다. exact builtin container identity cycle도 active-set으로 차단한다. budget 초과,
cycle, trusted NumPy 변환 실패는 모두 JSON-safe structured diagnostic을 남긴다. callback
exception formatting도 custom `str`/`repr`을 호출하지 않고, exact builtin exception의 최대
8개 exact primitive argument만 읽는다. 따라서 hostile int subclass, blocking iterator와
blocking exception formatting이 runner thread를 점유하지 않는다.

### 38.3 Quality/hardware 결과 shape와 count는 명시적으로 검증한다

`evaluator.compute()` raw result의 exact type이 `dict`가 아니면 list/scalar/`None`을 포함해
quality metric으로 사용하지 않는다. callback errors에는 phase, `result_shape`, expected/actual
type을 기록하고 `quality_result_invalid`로 run을 INVALID 처리한다. completed async counters는
그대로 보존하고 quality metrics는 `{}`, evaluator sample count는 `None`으로 기록한다.
evaluator sample count는 strict serializer 뒤의 exact finite Python int/float만 읽으므로
numeric subclass conversion을 호출하지 않는다.

monitor summary에도 같은 exact-dict shape 검증을 적용한다. 위반 시
`hardware_result_invalid`, `hardware_monitor_summary_failed`와 structured shape diagnostic을
남기고 hardware metrics를 비운다. 두 callback 결과 모두 strict serializer를 거치므로
shape 실패와 unsupported nested value가 함께 있어도 최종 metrics/details는
`json.dumps(..., allow_nan=False)` 가능하다.

## 39. Task 7 orchestration review round 4 보강 계약

### 39.1 Raw reference 비전달과 cooperative direct disposal

일반 callback worker와 monitor 직렬 lane worker는 callback을 호출한 같은 thread에서 raw
return을 closed-world serializer로 변환하고 raw exception을 안전한 builtin diagnostic으로
snapshot한다. main thread로 전달하는 값은 exact JSON-safe builtin tree, exact-dict 여부, 안전한
type name, serialization diagnostic, fatal category뿐이다. raw return, raw exception, traceback은
공유 outcome/job에 저장하지 않는다. 따라서 shape 검증과 fatal 재구성도 raw 객체를 다시 읽지
않는다.

이 절의 보장은 direct raw-reference transfer와 cooperative direct disposal에 한정되며, cyclic
finalization과 bounded-timeout 이후 external GC 범위는 §40을 따른다.

worker는 safe snapshot이 준비되면 `ready`를 먼저 signal하고 상태를 `disposing`으로 바꾼 뒤,
같은 callback/lane thread에서 serializer temporary, raw return 또는 raw exception의 마지막
참조를 해제한다. 이 해제가 끝난 뒤에만 `finished`를 signal한다. caller는 같은 absolute deadline
안에서 `finished`까지 기다리고, 일반 callback thread는 남은 시간 안에 join한다. hostile
`__del__`이 영원히 막히면 main은 raw 객체를 인수하거나 소멸하지 않고 callback phase/ID/thread,
alive 상태, `ready`, `finished`, `disposing` 상태를 가진 `callback_timeout` INVALID 결과를
반환한다. 정상 callback은 join되고 정상 monitor lane은 close sentinel 뒤 join되어 daemon을
남기지 않는다.

subprocess 회귀는 evaluator return, evaluator exception, monitor summary return과 exception의
소멸자가 현재 thread 이름을 출력한 뒤 무기한 기다리게 한다. 모든 경로는 parent deadline 안에서
JSON-safe INVALID 결과를 출력하며 소멸 thread는 evaluator callback worker 또는 monitor lane이고
`MainThread`가 아니다.

### 39.2 정상 결과 조립은 monitor lane을 먼저 정리한 뒤 진단을 snapshot한다

monitor summary 처리 뒤 정상 result 경로는 fresh bounded deadline으로 lane을 명시적으로
close/join한다. 그 다음 일반 callback과 monitor job의 outstanding 상태를 새로 읽어
`details.outstanding_callbacks`와 timeout limitation을 조립한다. close 중 늦게 끝난 job은 완료된
상태로 반영되어 stale outstanding entry를 남기지 않는다. public `run()`의 outer `finally`는
fatal/assembly 예외를 위한 최종 안전망으로 유지되며, 이미 닫힌 lane의 두 번째 close는 sentinel을
추가하거나 다시 기다리지 않는 idempotent no-op이다.

## 40. Task 7 orchestration review round 5/6 보강 계약

### 40.1 Cooperative quarantine과 bounded-timeout ownership

safe builtin snapshot과 diagnostic을 게시한 `ready`는 raw callback 객체의 lifetime 종료를 뜻하지
않는다. callback worker 또는 monitor lane은 그 뒤 framework-owned GC quarantine lock을 획득할
때까지 raw return과 raw exception을 local strong reference로 계속 보유한다. Quarantine이
cooperative하게 진행되어 caller deadline 전에 끝날 때는 framework path가 direct/cyclic raw 객체를
callback worker/lane에서 정리한 뒤 `finished`를 게시한다.

lock을 획득한 daemon thread는 raw exception의 `__traceback__`, `__context__`, `__cause__`를
`BaseException`의 builtin attribute setter로 `None` 처리하고 callback, serializer, raw return과 raw
exception local을 모두 해제한다. cooperative direct refcount finalizer는 이 해제 지점의 daemon
thread에서 실행한다. self-cycle처럼 refcount만으로 끝나지 않으면 상태를 `collecting`으로 바꾸고 lock
안에서 명시적인 full `gc.collect()`를 호출한다. collection이 정상 반환한 뒤에만 `done`과
`finished`를 signal한다.

direct destructor, link clear, lock wait, collection 또는 cyclic finalizer가 caller deadline을 넘기면
runner는 safe snapshot만 사용해 INVALID 결과를 반환하고 그 outcome을 settled로 취급하지 않는다.
이때 `disposing`, `waiting_for_gc_quarantine`, `collecting` 상태의 callback ID/phase/thread/alive를
outstanding으로 남긴다. 이 bounded return 이후에는 thread-only core가 그 callback lifetime에 대한
배타적 finalizer-thread ownership을 보장하지 않는다.

### 40.2 Process-global GC 범위와 검증 한계는 명시적이다

Python cyclic GC generation은 process-global이므로 callback 하나의 unreachable cycle만 선택해
수집할 수 없다. quarantine lock은 framework가 시작한 release-to-collect critical section끼리
직렬화해 한 worker가 막 해제한 cycle을 owner collection 전에 다른 framework worker가 수집하지
못하게 한다. application이 직접 호출하는 `gc.collect()`와 자동 GC 자체를 intercept하거나
monkeypatch하지 않으며, explicit
collection은 application이 GC를 disabled한 상태에서도 동작하고 기존 enabled/disabled 설정을
바꾸지 않는다. Full collection은 같은 시점의 다른 unreachable cycle도 daemon thread에서 finalize할
수 있고 그 비용과 hostile finalizer 대기는 callback deadline에 포함된다.

runner가 deadline에 반환한 뒤 callback cycle이 unreachable해지면 외부 manual/automatic
process-global GC가 그 collection을 trigger한 thread에서 finalizer를 실행할 수 있으며, 그 thread에는
`MainThread`도 포함된다. 이를 엄격히 막으려면 process isolation이 필요하고 Task 7 이후 follow-up
범위다. Task 7 core는 process isolation을 추가하지 않는다.

GC quarantine state가 outstanding이면 결과는 JSON-safe
`details.callback_gc_external_finalization_possible` object를 추가한다. `callbacks`에는 각
phase/ID/thread/alive/state를 넣고, `external_gc_effect`에는 external/manual/automatic GC가 triggering
thread에서 finalize할 수 있음을, `strict_ownership_follow_up`에는 strict ownership이 process
isolation을 요구함을 기록한다. Quarantine callback이 없으면 이 field는 `None`이다.

자동 GC를 disabled한 cooperative subprocess 회귀는 evaluator return/exception과 monitor summary
return/exception 각각에 blocking `__del__`을 가진 self-cycle을 만든다. 현재 실행에서는 daemon full
collection이 먼저 finalizer를 시작하고 main의 후속 explicit collection이 완료되는 것을 확인하지만,
이는 timeout 이후 모든 외부 GC interleaving에 대한 unconditional ownership 보장이 아니다. 정상
callback은 collection 반환과 thread/lane join까지 끝나므로 daemon을 남기지 않는다.

## 41. Task 8 artifact durability와 trace lifecycle 보강 계약

### 41.1 Run ID와 CSV 호환성

자동 run ID는 8자리 소문자 16진수이고, 사전 할당 ID는 ASCII 영문자 또는 숫자로 시작하는
영문자·숫자·밑줄·하이픈만 허용한다. 유효한 사전 할당 ID는 변형하지 않고 그대로 CSV와
sidecar에 사용하며 빈 값, 절대 경로, 구분자, `..`, 제어 문자와 비 ASCII ID는 artifact를
만들기 전에 거부한다. 사전 할당 ID 검증은 results directory와 lock file을 만들기 전에
끝낸다. CSV lock을 잡은 상태에서 이미 존재하는 사전 할당 ID는 거부하고 자동 생성 ID가
기존 ID와 충돌하면 새 ID를 다시 생성한다. 기존 파일에 이미 중복 ID가 있더라도 migration은
그 행들을 그대로 보존하지만 새 중복 행은 추가하지 않는다.

`save_result()`의 기존 인자와 `results_path` 위치는 그대로 보존하고 async 인자는 그 뒤에
추가한다. Async metadata 이름과 같은 metric key는 metadata를 덮어쓸 수 없다. 기존 CSV를
확장할 때 기존 header와 metric column 순서, 모든 행의 순서를 먼저 보존한 뒤 누락 metadata와
새 metric을 붙인다. 이 전에 strict positional CSV read로 header가 non-empty·unique인지와
모든 data row가 header와 같은 cell 수인지 검증한다. 따라서 malformed quoting, 중복/빈
header, 짧거나 긴 행은 원본을 바꾸지 않고 실패하며 quoted comma, multiline, empty cell의
위치와 행 순서를 잃지 않는다. 확장과 삭제 rewrite는 기존 interprocess lock 안에서 같은 디렉터리의
owner-unique temporary file에 전체 내용을 쓰고 file `fsync`, `os.replace`, directory `fsync`로
완결한다. 새 CSV mode는 `0644`이고 rewrite는 기존 파일 mode를 보존한다. 실패 시 temporary
file을 정리하며 replace 전에 실패하면 기존 CSV는 그대로 남는다. Cleanup이나 descriptor close도
실패하면 최초 persistence 예외를 유지하고 secondary failure를 구조화된 diagnostic과 exception
note로 첨부한다.

### 41.2 JSON sidecar schema와 직렬화 경계

`save_async_details()`는 exact builtin primitive/container, 승인된 exact NumPy scalar/array,
`pathlib` concrete path와 enum value만 closed-world 방식으로 정규화한다. 임의 객체의 `item`,
`tolist`, iterator, `str` 또는 `repr` callback은 호출하지 않으며 unsupported type, cycle,
non-finite number와 문자열이 아닌 object key는 저장 전에 실패한다. Set 계열과 object key를
포함한 출력은 결정적으로 정렬하고 strict JSON(`allow_nan=False`)으로 인코딩한다. 정규화는
depth 32, 전체 item 10,000개, NumPy array 4,096개 element, 문자열 1,000,000자 상한을 가지며
enum identity도 active-cycle set에 포함한다. 상한 초과와 enum cycle은 publication 전의 typed
`ValueError`로 끝나고 `RecursionError`나 무제한 traversal로 진행하지 않는다.

호출자가 전달한 `schema_version`과 `run_id`는 신뢰하지 않고 최종 payload에서 각각 `1.0`과
검증된 ID로 고정한다. Results root와 `details`는 `O_DIRECTORY | O_NOFOLLOW` directory fd로 열고
entry와 열린 fd의 device/inode를 publication 전후에 다시 비교한다. 모든 temporary/final 연산은
고정된 `details` fd에 상대적으로 수행하므로 parent symlink나 deterministic directory swap이
requested root 밖에 파일을 만들 수 없다. 완성된 JSON bytes만 owner-unique same-directory
temporary file에 쓰고 file `fsync`한 뒤 hard-link no-overwrite publication과 directory `fsync`를
수행한다. 같은 run ID의 thread/process writer 중 정확히 하나만 성공하고 기존 regular file이나
symlink를 덮어쓰지 않는다. Serialize/write/fsync/publish 실패는 호출자에게 전파하고 temporary
file을 정리하므로 부분 JSON은 final path에 노출되지 않는다. Cleanup까지 실패하면 최초 예외를
유지한 채 secondary diagnostic에 temp leakage 가능성과 경로를 기록한다.

### 41.3 Request trace writer 상태와 실패 관찰

`RequestTraceWriter` capacity는 양의 exact integer다. 상태 전이는 `created -> running ->
closing -> closed`이고 double start, start 뒤가 아닌 close, close 중 재진입, running 이외 상태의
write는 명시적으로 실패한다. 완료된 close는 같은 boolean 결과를 idempotent하게 반환한다.
`write()`는 exact `RequestTrace`만 받으며 정의된 ID/status/timing/worker/batch/timeout/sample count와
오류 문자열 필드만 plain row로 복사한다. 따라서 request sample, input, label, output, prompt와
그 참조를 받을 schema 경로 자체가 없다.

Running write는 bounded queue에 `put_nowait()`만 수행한다. 포화 row는 thread-safe `dropped`
count에 포함하고 측정 경로를 기다리게 하지 않는다. Close는 호출 시작 시 하나의 absolute
deadline을 만들고 남은 시간으로 writer thread를 한 번만 join한다. Deadline을 넘기면 이후 write를
차단하고 publication을 abandon하며 `False`와 timeout diagnostic을 반환한다. Queue item과 stop
token은 성공·실패·abandon 모두 exact `task_done()` accounting을 지킨다.

Writer는 start에서 기존 target을 먼저 거부하고 JSONL을 owner-unique temporary file에 기록한다.
정상 close에서 file `fsync`, hard-link no-overwrite publication, directory `fsync` 후에만 final
path를 공개하므로 같은 target의 thread/process writer 중 하나만 성공하고 기존 file/symlink를
덮어쓰지 않는다. Serialization, open/write/flush/fsync, publish와 cleanup 실패는 `phase`, safe
error type/message로 관찰할 수 있고 close는 `False`를
반환한다. Final publication에 진입하기 전의 실패나 timeout은 temporary file을 정리하고 final path를
만들지 않는다. 최초 writer failure 뒤 descriptor close나 temporary cleanup도 실패하면 public
error snapshot의 `secondary_errors`에 모두 누적하고 temp leakage 가능성을 표시한다. Python
thread를 강제 종료하지 않으므로 이미 진입한 blocking OS call 자체는 중단하지 못한다. 특히
hard-link publication syscall이 deadline 전에 시작된 뒤 stall하면 close는 `False`를 반환하더라도
syscall은 나중에 완성될 수 있지만, 이 경우에도 final path에는 부분 JSONL이 아니라 fsync를 마친
전체 파일만 한 번에 나타난다.
