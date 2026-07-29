# 비동기 추론 큐 트러블슈팅 기록

**기록일:** 2026-07-24

**대상:** async metrics, completion retirement, Mobilint native async backend

**최종 상태:** CPU 회귀 테스트 통과, Mobilint ResNet50 3,000건 실제 NPU 인수 성공
사용자 보고(원본 artifact 미첨부)

## 문서 목적

비동기 추론 큐의 대규모 요청 검증 과정에서는 서로 다른 세 문제가 연속으로 드러났다.

1. 요청 수가 커질수록 metrics 집계 비용이 O(N³)에 가까워지는 CPU 병목
2. framework terminal 완료 뒤 native dispatch 자원을 정확히 회수하지 못하는 completion retirement 문제
3. Mobilint SDK 실행 슬롯과 framework callback 수명을 결합해 발생한 드문 제출 race

겉으로는 모두 "비동기 큐가 느리거나 요청을 잃는다"는 비슷한 증상으로 보였지만,
원인과 수정 계층은 각각 metrics, common completion lifecycle, vendor backend로 달랐다.
이 문서는 같은 증상이 재발했을 때 Queue나 NPU를 먼저 바꾸기 전에 어느 계층의 증거를
확인해야 하는지 남기기 위한 기록이다.

여기서 사용하는 증거 수준은 다음과 같다.

| 구분 | 의미 |
|---|---|
| 관찰 | 실제 실행 로그, traceback 또는 결과 artifact에서 확인한 현상 |
| CPU 재현 | NPU 없이 동일한 코드 경로와 동시성 순서를 재현한 결과 |
| 회귀 테스트 | 수정 전 실패하고 수정 후 통과하도록 고정한 자동 테스트 |
| 실제 NPU 인수 | 실제 Mobilint 장치에서 전체 요청 lifecycle이 정상 수렴한 결과 |

## 실행 구조와 용어

비동기 실행의 공통 흐름은 다음과 같다.

```text
Producer
  → bounded Framework Request Queue
  → Worker / dynamic batching
  → RuntimeExecutor
  → vendor runtime 또는 SDK
  → bounded Completion Queue
  → CompletionCoordinator
  → Decoder / Postprocessor
  → Evaluator / metrics
  → terminal commit
  → ACK / resource retirement
```

Framework Queue와 vendor SDK queue는 같은 큐가 아니다.

- **Framework Request Queue**는 요청 admission, backpressure와 논리적 요청 소유권의 기준이다.
- **Vendor SDK queue/Future**는 장치 실행을 구현하는 backend 내부 세부사항이다.
- **Completion Queue**는 장치 실행 결과를 단일 `CompletionCoordinator`에 전달한다.

### 혼동하기 쉬운 네 가지 경계

| 경계 | 소유 계층 | 종료 조건 |
|---|---|---|
| SDK physical completion | Vendor backend | Future 또는 callback이 실제 장치 작업 종료를 증명 |
| Framework terminal | `CompletionCoordinator` | membership 검증, 후처리, evaluator와 terminal metric 처리 완료 |
| Executor permit | `NativeAsyncRuntimeExecutor` | physical completion과 framework logical ACK가 모두 확인됨 |
| Mobilint SDK slot | `MobilintNativeBackend` | Future가 terminal이고 출력 정규화가 끝남 |

따라서 SDK callback이 도착한 시점과 framework request가 최종 완료된 시점은 같지 않다.

```text
SDK callback 도착 ≠ framework terminal 완료 ≠ 모든 자원 회수 완료
```

Completion membership은 request ID와 submission token을 함께 사용한다. 이 검증은 output의
정확도를 검사하는 것이 아니라, 도착한 completion이 현재 등록된 정확한 요청 시도에 속하는지
확인한다. 중복, 알 수 없는 요청, 이전 시도의 늦은 결과는 각각
`duplicate_completion`, `unknown_completion`, `stale_completion`으로 진단한다.

## 장애 연쇄 요약

```text
100건 async 실행 성공
  ↓
3,000건에서 measurement가 급격히 느려지고 NPU가 idle로 관찰
  ↓
CPU-only 재현으로 metrics O(N³) accounting 병목 확인
  ↓
정상 hot path를 증분 집계로 변경
  ↓
common completion의 terminal ACK와 native dispatch retirement 경계 보강
  ↓
처리량이 높아지자 Mobilint 1,000건 중 한 건의 slot race 노출
  ↓
SDK slot과 callback job 수명을 분리
  ↓
Mobilint ResNet50 3,000건 실제 NPU 인수 성공 사용자 보고
```

성능 수정이 Mobilint race를 새로 만든 것은 아니다. Metrics 병목이 줄어들면서 다음 요청이
더 빠르게 제출되어, 이미 존재하던 매우 짧은 slot/callback 경쟁 구간이 관찰 가능해졌다.

## 사건 1: Async metrics O(N³) 병목

### 증상

RBLN 장치에서 ResNet50 async offline 실행을 검증할 때 100건은 다음 조건으로 정상
완료됐다.

```text
accepted/completed/evaluator: 100
failed/rejected/timeout/outstanding: 0
async_run_status: valid
```

동일 구조로 3,000건을 실행하면 `measurement` 단계가 매우 느려지고 NPU가 idle로
관찰됐다. 최초에는 NPU runtime 정지, SDK crash 또는 context leak을 의심할 수 있는
모양이었다.

그러나 실행 중 사용자가 기다리다가 `Ctrl+C`를 입력해 발생한 traceback은
`KeyboardInterrupt`였고, framework는 이후 trace 종료, runtime unload, sidecar와 CSV 저장을
완료했다. 최종 장치 상태에서도 context와 사용 메모리가 모두 정리됐다. 따라서 확인된
문제는 NPU crash나 context leak이 아니었다.

### 관찰 증거

첫 interrupt가 멈춘 위치는 다음 함수 내부였다.

```text
framework/src/core/async_inference/metrics.py
└─ _rebuild_outcome_accounting_locked()
```

호출 경로는 producer가 새 요청을 Queue에 publish하며 acceptance accounting을 기록하는
hot path였다.

```text
OfflineProducer.run()
→ submitter.submit()
→ AsyncInferenceEngine.submit()
→ RequestQueue.publish_accepted()
→ _commit_acceptance_internal()
→ _rebuild_outcome_accounting_locked()
```

즉 장치가 idle로 보인 이유는 장치 문제가 아니라 producer가 metrics lock 아래의 CPU
집계를 기다려 새 요청을 공급하지 못했기 때문이다.

### CPU-only 재현

NPU 없이 `_commit_acceptance_internal()`만 반복해도 비선형 증가가 재현됐다.

| Requests | Elapsed |
|---:|---:|
| 50 | 0.001577 s |
| 100 | 0.007715 s |
| 200 | 0.056333 s |
| 400 | 0.381879 s |
| 800 | 3.141343 s |

요청 수가 두 배가 될 때 총 시간이 약 여덟 배씩 증가했다. Queue나 NPU를 사용하지 않는
재현이므로 병목을 metrics accounting으로 분리할 수 있었다.

### 근본 원인

기존 `_rebuild_outcome_accounting_locked()`는 호출될 때마다 다음 전체 이력을 다시 만들었다.

- accepted outcome 전체 순회
- rejected outcome 전체 순회
- accepted/rejected counter와 reason counter 재구축
- Queue transition 전체 재구축
- accepted sequence 집합 재구축
- acceptance와 terminal inflight event 재생성 및 정렬

특히 기존 transition comprehension 안에서 accepted sequence 집합을 반복 생성했다. 이 때문에
history 크기가 `k`일 때 rebuild 한 번이 O(k²)에 가까웠고, acceptance와 terminal마다 이를
반복해 누적 비용이 다음과 같이 O(N³)에 가까워졌다.

```text
Σ O(k²), k=1..N  →  O(N³)
```

일반적인 Request Queue의 enqueue/dequeue는 `append`/`popleft` 중심이며 요청당 평균 O(1)이다.
따라서 이 사건에서 bounded Queue 자료구조 자체는 주된 원인이 아니었다.

### 보존해야 했던 계약

전체 rebuild를 단순히 삭제할 수는 없었다. 기존 journal은 accounting mutation 중 interrupt가
발생해도 다음 호출이나 finalize에서 복구하기 위한 authoritative evidence였다.

반드시 보존해야 했던 조건은 다음과 같다.

- attempt token 하나당 accepted 또는 rejected outcome이 정확히 하나 존재
- 같은 outcome의 재기록은 idempotent하고 accepted/rejected 충돌은 거부
- Queue transition sequence와 missing sequence 증거 유지
- `submitted = accepted + rejected`
- `accepted = completed + failed + outstanding`
- acceptance부터 terminal까지 time-weighted inflight 계산 유지
- mutation 중 interrupt 뒤에도 canonical state 복구 가능

### 피해야 할 임시방편

- Queue capacity를 줄이거나 늘리는 방식은 metrics의 전체 이력 순회를 제거하지 않는다.
- Worker 수를 줄이면 장치 공급률만 낮아지고 점근 복잡도는 그대로다.
- Fault-recovery journal을 삭제하면 빠를 수는 있지만 interrupt와 partial mutation에서 counter
  invariant를 잃는다.
- Wall-time threshold만 CI에 추가하면 느린 runner 환경에서 flaky test가 된다.

### 실제 해결

수정은 authoritative evidence와 derived projection을 분리했다.

```text
Authoritative evidence
  ├─ outcomes: attempt token별 accepted/rejected journal
  ├─ terminal_times: request ID별 terminal timestamp
  └─ Queue transition/failure evidence

Derived projection
  ├─ accepted/rejected counters
  ├─ rejection reason counters
  ├─ Queue transition projection
  └─ time-weighted inflight gauge
```

정상 경로에서는 lock 아래에서 다음 값만 증분 갱신한다.

- acceptance: journal 저장, accepted +1, Queue transition 추가, inflight +1
- rejection: journal 저장, rejected/reason +1
- terminal: terminal timestamp 저장, inflight -1

Mutation 전 `outcome_accounting_dirty=True`를 기록하고 정상 완료 후 `False`로 되돌린다.
중간에 interrupt가 발생하면 dirty 상태가 남고, 다음 recovery 또는 finalize가 authoritative
evidence에서 canonical projection을 한 번 재구축한다.

수정 후 목표 복잡도는 다음과 같다.

| 경로 | 복잡도 |
|---|---:|
| 정상 acceptance/rejection/terminal | 요청당 O(1) |
| 정상 실행 전체 hot path | O(N) |
| Finalize 또는 interruption recovery | O(N log N) 이하 |

관련 구현 커밋은 `6b81b4d` (`fix: make async metrics accounting incremental`)이다.

### 검증과 교훈

회귀 테스트는 wall time만 비교하지 않고 정상 3,000건 accounting에서 full rebuild가
호출되지 않으며 finalize에서 canonical rebuild가 한 번만 호출되는지 구조적으로 확인한다.
Idempotency, accepted/rejected conflict, Queue sequence, inflight, mutation boundary fault injection도
함께 검증한다.

이 사건의 핵심 교훈은 다음과 같다.

> 장치가 idle이라고 해서 장치가 병목인 것은 아니다. 요청을 장치에 공급하는 producer가
> metrics나 logging 같은 관찰 계층의 lock에 막힐 수도 있다.

## 사건 2: Completion terminal과 native dispatch retirement

### 증상과 구조적 조건

Metrics 병목과 별개로 common async lifecycle에는 completion handoff와 native 자원 회수 사이의
경계 문제가 있었다. `AsyncInferenceEngine` worker는 `BatchCompletion`을 Completion Queue에
전달한 뒤 다음 iteration으로 이동한다. Native executor의 dispatch는 framework가 결과를
소비했다는 logical ACK 전까지 permit과 관련 자원을 유지한다.

`max_inflight=1`, Worker 한 개 조건에서 다음 순서가 가능했다.

```text
첫 번째 Worker 실행 완료
→ BatchCompletion 비동기 handoff
→ Worker가 다음 iteration으로 이동
→ 첫 dispatch는 terminal ACK 전이라 유일한 native permit 보유
→ 다음 execute()가 permit을 기다림
→ worker-local handoff가 terminal 뒤 retire되지 않으면 flush timeout
```

### 근본 원인

Completion thread는 terminal 결과를 commit하지만, 정상 worker path가 보관하던 dequeue
operation과 `RuntimeExecution` 정리를 같은 사건으로 정확히 한 번 연결하지 못했다. 기존
deferred-handoff retry만으로는 worker-local handoff를 항상 제때 retire할 수 없었다.

여기서 native permit을 runtime callback 즉시 반환하는 것도 안전하지 않다. Framework가
decoder/evaluator에서 output을 소비하기 전에 input/output buffer나 dispatch 자원을 재사용할
수 있기 때문이다.

### 피해야 할 임시방편

Worker가 completion ACK까지 매번 동기 대기하게 만들면 문제는 가려지지만 다음 overlap을
잃는다.

```text
runtime execution ─┐
next request prep  ├─ 기존 비동기 pipeline overlap
decode/evaluation ─┘
```

또한 `CompletionCoordinator`가 특정 `RuntimeExecutor`나 Mobilint 타입을 직접 알게 만들면
공통 completion 계층에 vendor lifecycle이 침투한다.

### 실제 해결: One-shot retirement lease

Worker는 normal completion cleanup capability를 generic one-shot retirement lease로 만들어
completion handoff에 함께 전달한다. Coordinator는 lease가 어떤 자원을 해제하는지 알지 않고
`retire()`만 호출한다.

Queued completion의 정상 순서는 다음과 같다.

```text
1. request ID / submission token membership 검증
2. Decoder / Postprocessor 실행
3. Evaluator 및 terminal metrics/trace 처리
4. request terminal commit
5. completion handoff ACK
6. retirement lease 실행
7. dequeue ownership 정리
8. RuntimeExecutor.acknowledge(execution)
```

Lease는 `PENDING → RETIRING → RETIRED` 상태를 사용해 completion, shutdown 또는 recovery가
경쟁해도 cleanup callback을 한 번만 실행한다. Cleanup이 실패하면 `FAILED`로 남고 성공한
shutdown 증거를 만들지 않는다.

이 설계는 다음 책임 경계를 유지한다.

- `AsyncInferenceEngine`: admission, Request Queue, batching, worker, flush와 shutdown
- `CompletionCoordinator`: membership, decoder/evaluator, terminal metric과 exact-once commit
- `RuntimeExecutor`: 장치 dispatch, logical ACK와 native permit
- Retirement lease: coordinator가 vendor를 몰라도 실행할 수 있는 일회성 cleanup capability

Inline e2e completion은 operation key나 retirement lease를 사용하지 않으며,
`BlockingRuntimeExecutor.acknowledge()`는 계속 no-op이다.

관련 구현 커밋은 다음과 같다.

- `0f069b5`: completion terminal 뒤 retirement lease 실행
- `224d981`: native dispatch를 completion terminal에서 retire
- `dab87ee`: retirement 실패 계약 회귀 테스트

### 검증과 교훈

회귀 테스트는 Worker 한 개, native inflight 한 개, 요청 두 개를 사용한다. 첫 요청의 terminal
처리가 끝나면 lease가 dispatch를 ACK해 두 번째 요청이 flush timeout 없이 실행되는지 확인한다.
중복 `retire()`와 retirement callback 실패에서도 cleanup을 다시 실행하지 않는지 검증한다.

이 사건의 핵심 교훈은 다음과 같다.

> "장치 실행 완료"와 "framework가 결과를 모두 소비함"은 다른 수명이다. Native 자원은
> 물리 완료와 logical ACK 계약을 모두 만족하는 계층에서 회수해야 한다.

## 사건 3: Mobilint SDK slot race

### 증상

Metrics 성능 문제와 common completion retirement를 보강한 뒤 실제 Mobilint native async
1,000건 실행에서 다음 결과가 한 번 관찰됐다.

```text
submitted: 1000
accepted: 1000
completed: 999
failed: 1
timed_out: 0
outstanding: 0
```

실패 요청은 timeout이나 NPU inference error가 아니라 backend 제출 경계에서 다음처럼
기록됐다.

```text
error_type: RuntimeError
error_message: native async submission failed
```

Common executor가 vendor 예외를 일반화했기 때문에 결과 artifact만으로는 capacity exhaustion과
SDK 내부 오류를 구분하기 어려웠다. Backend code와 결정적 CPU 재현을 함께 확인해야 했다.

### CPU 결정적 재현

Mobilint 하드웨어 없이 다음 조건을 만들면 기존 race를 반복 가능하게 재현할 수 있었다.

1. Mobilint backend activation slot을 1로 설정한다.
2. 첫 Fake Future의 `get()`은 즉시 terminal 결과를 반환한다.
3. 첫 framework callback은 `Event`에서 반환하지 않고 기다린다.
4. Callback이 active인 동안 두 번째 `submit_async()`를 호출한다.

기존 구현에서는 첫 Future가 이미 끝났는데도 두 번째 요청이 다음 오류로 거부됐다.

```text
Mobilint native async waiter capacity is exhausted.
```

### 근본 원인

`MobilintNativeBackend`는 `infer_async()` 전에 `_slots` semaphore를 획득한다. 이 슬롯은 pending
SDK Future 수를 제한하는 장치 실행 권리다. 하지만 기존 코드는 `Future.get()`이 끝난 시점이
아니라 framework callback이 완전히 반환된 뒤 슬롯을 해제했다.

```text
기존 순서
Future.get() terminal
→ output normalization
→ framework callback 진입
→ callback 반환
→ SDK slot 반환
```

한편 native executor callback은 결과를 publish하고 worker를 먼저 깨운다. Completion thread가
결과를 소비하고 logical ACK를 하면 executor permit은 반환될 수 있다. 이때 이전 Mobilint
waiter는 아직 callback에서 복귀하지 않아 backend `_slots`를 보유할 수 있다.

```text
Mobilint waiter                         Framework worker/completion
─────────────────────────────────────────────────────────────────
Future.get() terminal
output normalization
callback(outcome) 진입
  dispatch event 설정
                                       worker가 결과 handoff
                                       terminal commit
                                       executor ACK / permit 반환
                                       다음 execute()
                                       backend.submit_async()
                                       _slots.acquire(False) 실패
callback(outcome) 반환
SDK slot 반환
```

경쟁 구간이 callback 반환 직전의 매우 짧은 시간이어서 1,000건 중 한 건처럼 드물게 나타났다.

### 두 종류의 소유권

문제는 서로 다른 두 수명을 하나의 `finally`에 묶은 것이었다.

| 소유권 | 시작 | 종료 | 종료를 늦추는 조건 |
|---|---|---|---|
| SDK 실행 슬롯 | `infer_async()` 제출 직전 | `Future.get()` terminal과 output normalization 완료 | 실제 SDK 작업과 output 정규화 |
| Framework job/callback | `_jobs[job_id]` 등록 | callback 완전 반환 | 결과 전달, callback 실행, shutdown 안전성 |

SDK slot은 callback 속도와 무관하게 장치가 다음 작업을 받을 수 있는지를 나타낸다. 반면
`_jobs`는 callback이 실행 중일 때 shutdown이나 model unload가 안전하다고 잘못 판단하지 않도록
끝까지 유지해야 한다.

### 피해야 할 임시방편

| 임시방편 | 사용하지 않은 이유 |
|---|---|
| 실패 시 무조건 재시도 | SDK가 요청을 수락한 뒤 예외를 던졌다면 중복 추론 가능 |
| Queue capacity 증가 | Framework 대기열 크기는 backend-local slot race와 무관 |
| Worker 수 감소 | 경쟁 확률을 낮출 뿐 원인을 남기고 처리량을 제한 |
| `_slots.acquire()`를 무기한 blocking으로 변경 | 짧은 race를 긴 대기 또는 deadlock으로 바꿀 수 있음 |
| Executor permit 제거 | Framework result consumption과 buffer lifetime 계약을 약화 |
| Mobilint `_slots` 제거 | Backend를 직접 사용하는 호출자의 SDK capacity 보호 경계를 제거 |

### 실제 해결

수정 후에는 SDK slot과 framework job을 서로 다른 시점에 retire한다.

```text
infer_async() accepted
→ Future.get() terminal 또는 failure
→ output normalization 또는 failure outcome 생성
→ SDK slot을 정확히 한 번 반환
→ framework callback 호출
→ callback 완전 반환
→ _jobs entry 제거 및 input reference 정리
```

`_MobilintAsyncJob.slot_released`와 `_release_job_slot(job)`이 semaphore 획득 한 번당 release를
정확히 한 번만 수행한다. `claim_lock/claimed`는 waiter thread 시작 실패 후 inline fallback과
start-then-raise 경로에서도 `Future.get()`을 한 번만 호출하도록 유지한다.

다음 조건도 함께 보존한다.

- Future가 pending인 동안에는 slot을 반환하지 않는다.
- Output normalization이 끝나기 전에는 slot을 반환하지 않는다.
- Callback 예외가 발생해도 slot을 누수하지 않는다.
- `infer_async()`가 Future 반환 전에 예외를 던지면 pre-job slot을 직접 반환한다.
- `_jobs`와 input reference는 callback 종료까지 유지한다.
- `shutdown()`은 active callback과 waiter thread 종료를 계속 기다린다.
- Native executor permit은 framework terminal ACK까지 유지한다.

관련 변경은 이 문서와 함께 반영한 callback/slot race 회귀 테스트, Mobilint SDK
slot lifetime 분리, runtime ownership 설명이다.

### 검증과 교훈

CPU 회귀 테스트는 다음 경로를 포함한다.

- Callback이 반환되기 전 다음 제출 허용
- Pending Future가 있을 때는 실제 over-capacity 제출 거부
- Future failure와 callback exception 뒤 slot 누수 없음
- SDK slot 재사용을 모사해 callback에 전달한 output의 framework 소유권과 값 보존
- Waiter thread construction/start/start-then-raise fallback에서 exact-once `get()`과 release
- Callback이 active인 동안 shutdown 비수렴, callback 해제 후 정상 수렴

실제 Mobilint 장치에서는 사용자가 `verify/mobilint-aries` 환경에서 ResNet50 3,000건 인수
테스트 성공을 확인했다. 합격 기준은 다음 counter 수렴이다.

```text
submitted == accepted == completed
failed == 0
timed_out == 0
outstanding == 0
async_run_status == valid
```

이 인수 성공은 사용자 확인 사실이며, 이 문서에는 해당 실행의 QPS, P95, P99 원본 로그가
포함돼 있지 않으므로 성능 수치를 추정해 기록하지 않는다.

이 사건의 핵심 교훈은 다음과 같다.

> Semaphore의 이름이 같아 보여도 무엇의 수를 제한하는지에 따라 반환 시점이 다르다.
> SDK 실행 슬롯과 framework callback job은 별도 소유권으로 모델링해야 한다.

## 검증 결과

### 자동 회귀 기준선

문서 작성 시점에 다음 CPU 회귀 테스트를 실제 실행했다.

```bash
python -m pytest \
  tests/test_async_metrics.py \
  tests/test_async_completion.py \
  tests/test_native_async_runtime_executor.py \
  tests/test_mobilint_runtime.py \
  tests/test_mobilint_native_backend.py -q
```

결과:

```text
305 passed
```

이 결과는 관련 CPU 회귀 범위의 통과를 의미하며 framework 전체 테스트 수나 실제 NPU 성능
검증을 대신하지 않는다.

### 증거별 확인 범위

| 증거 | 확인한 것 | 확인하지 않는 것 |
|---|---|---|
| CPU complexity 재현 | 병목이 NPU 없이 metrics hot path에서 재현됨 | 실제 장치 throughput |
| Metrics fault injection | dirty mutation이 canonical accounting으로 복구됨 | Vendor callback lifecycle |
| Completion retirement 회귀 | terminal 뒤 ACK/retire가 exact-once로 수렴 | Mobilint backend-local slot |
| Fake Mobilint Future | Callback/slot race와 재사용 output buffer를 결정적으로 재현하고 수정 | 실제 SDK output lifetime 전부 |
| 사용자 보고 Mobilint 3,000건 인수 | 전체 request가 실패·timeout·outstanding 없이 수렴 | 저장소에 첨부되지 않은 원본 실행 artifact와 latency 개선 폭 |

### 결과를 유효하게 보는 최소 조건

성능 수치를 비교하기 전에 다음 조건을 먼저 확인한다.

```text
submitted = accepted + rejected
accepted = completed + failed + outstanding
flush 성공 후 outstanding = 0
evaluator sample 수 = 기대 sample 수
async_run_status = valid
```

Timeout은 terminal category에 더하는 별도 상태가 아니라 진단 subset이므로 counter 등식에
추가하지 않는다.

## 재발 시 진단 순서

| 증상 | 첫 확인 지점 | 구분 기준 | 조치 방향 |
|---|---|---|---|
| NPU가 idle이고 제출이 진행되지 않음 | Python traceback, producer와 metrics lock | `_rebuild_outcome_accounting_locked()` 또는 전체 이력 순회 내부인지 확인 | Hot path rebuild 호출과 lock-held 전체 순회 제거 여부 확인 |
| 표본 증가 시 두 배마다 약 여덟 배씩 느려짐 | CPU-only synthetic scaling | Queue 없이도 동일한 scaling인지 확인 | Accounting 중첩 comprehension, 반복 sort와 rebuild 횟수 검사 |
| Queue full 또는 submit wait 증가 | Queue depth, rejected reason, target QPS | 공급률이 처리율을 지속적으로 초과하는지 확인 | Capacity를 맹목적으로 늘리기 전에 load profile과 service time 비교 |
| 드문 `native async submission failed` | Backend 원본 오류, Mobilint `_slots`, active callback | Future는 terminal인데 callback이 아직 active인지 확인 | SDK slot과 callback job 수명 분리 여부 확인 |
| Outstanding 또는 flush timeout | Completion handoff journal, executor snapshot | Terminal ACK 뒤 dispatch가 retire됐는지 확인 | Retirement lease 실패·누락과 ACK ordering 검사 |
| Duplicate/unknown/stale completion | Request ID와 submission token | 현재 outstanding membership과 일치하는지 확인 | 해당 run을 성능 결과로 사용하지 말고 ownership 경로 조사 |
| Callback timeout 뒤 unload 불가 | Backend `_jobs`, waiter thread, physical Future 상태 | 논리 timeout과 실제 장치 작업 종료를 구분 | Physical completion 또는 cancellation 증명 전 강제 unload 금지 |
| NPU 실행은 빠르지만 전체 P99가 높음 | queue wait와 completion overhead | 실행 전 대기인지 후처리 병목인지 구분 | Queue/batch timeout 또는 decoder/evaluator 비용을 별도로 조사 |

### 진단 순서의 원칙

1. 먼저 counter invariant와 outstanding을 확인해 결과가 유효한지 판단한다.
2. 장치 상태만 보지 말고 Python traceback과 thread가 기다리는 lock/event를 확인한다.
3. Queue wait, service time, completion overhead를 나눠 병목 계층을 찾는다.
4. Framework request ownership과 vendor Future ownership을 분리한다.
5. Race는 sleep이나 retry로 완화하지 말고 Event로 경쟁 순서를 고정해 CPU에서 재현한다.
6. CPU 회귀 통과 뒤 실제 장치에서 동일한 counter 수렴을 확인한다.

## 남은 구조적 한계

이번 수정들은 확인된 P0 병목과 소유권 race를 해결했지만 native async 전체 구조를
event-driven 방식으로 바꾸지는 않았다.

### Worker와 waiter thread의 이중 대기

현재 `NativeAsyncRuntimeExecutor.execute()`는 SDK에 작업을 제출한 뒤 callback event를 기다려
`RuntimeExecution`을 반환한다. Mobilint backend도 요청마다 `mobilint-future-N` waiter thread를
만들고 `Future.get()`을 기다린다.

```text
Framework Worker
  → NativeAsyncRuntimeExecutor.execute()
  → callback event 대기

Mobilint waiter thread
  → Future.get() 대기
  → callback 호출
```

따라서 native 요청 하나에 framework worker와 Mobilint waiter thread가 각각 기다린다.
`queue_capacity=16`은 SDK Future 16개가 동시에 제출된다는 의미가 아니라 Framework Queue에
최대 16개 요청을 보관한다는 의미다. Worker 한 개라면 현재 blocking bridge에서는 보통 SDK에
한 번에 한 작업만 제출된다.

완전한 event-driven 구조를 만들려면 `execute() -> RuntimeExecution` 계약을
`submit(inputs, completion_callback) -> dispatch` 형태로 재설계하고, Future collection을 요청별
thread가 아닌 중앙 polling/owner loop에서 관리해야 한다. 이는 이번 수정의 범위를 넘어선다.

### Activation slot 자동 탐지

현재 `MobilintRuntime.max_concurrent_workers()`는 async pipeline에서 명시적
`activation_slots`가 없으면 1을 반환한다. 설치된 SDK ModelConfig의 실제 기본 slot 수를 자동으로
읽지 않는다. 장치가 두 slot을 제공해도 CLI 설정이 없으면 framework가 한 Worker로 제한될 수
있다. 자동 탐지는 별도 기능 변경과 장치별 검증이 필요하다.

### Completion Queue backlog 독립 계측 부재

현재 timestamp는 다음 여섯 경계를 기록한다.

```text
scheduled → issued → enqueued → runtime_started → runtime_finished → completed
```

`runtime_finished → completed`인 `completion_overhead`에는 Completion Queue handoff와 대기,
decoder, postprocessor, evaluator 비용이 함께 포함된다. Completion Queue 진입과 배출 timestamp가
따로 없기 때문에 Queue backlog 시간만 독립적으로 분리할 수 없다.

### 메모리와 최종 집계

Hot path accounting은 증분화됐지만 outcome journal, terminal timestamp, timing distribution과
선택적 request trace는 요청 수에 비례해 유지된다. 따라서 메모리는 여전히 O(N)이며 percentile
정렬과 canonical finalize 비용도 존재한다. 대규모 장시간 실행에서는 trace 설정과 result
artifact 크기도 함께 확인해야 한다.

## 관련 문서와 커밋

### 문서

- [비동기 추론 큐 측정 가이드](async-inference-queue.md)
- [통합 InferenceEngine 설계](unified-inference-engine-design.md)
- Metrics incremental accounting 구현: `6b81b4d`
- [Async completion retirement lease 설계](superpowers/specs/2026-07-23-async-completion-retirement-lease-design.md)
- [Mobilint native slot lifecycle 분석](#사건-3-mobilint-sdk-slot-race)

### 주요 커밋

| 커밋 | 역할 |
|---|---|
| `6b81b4d` | Async metrics accounting을 정상 hot path 증분 방식으로 변경 |
| `0f069b5` | Completion terminal 뒤 one-shot retirement lease 실행 |
| `224d981` | Native dispatch를 completion terminal ACK와 연결해 retire |
| `dab87ee` | Retirement failure와 exact-once cleanup 계약 보강 |
| 이번 PR | Mobilint callback/slot race 회귀 테스트, SDK slot 수명 분리와 ownership 문서화 |

## 최종 정리

세 장애의 공통점은 Queue 자체가 아니라 **요청 수명과 관찰·자원 소유권의 경계**에 있었다.

- Metrics는 hot path에서 전체 이력을 반복 집계하면 장치 공급을 멈출 수 있다.
- SDK physical completion과 framework terminal completion은 분리해야 한다.
- Generic executor permit과 vendor SDK slot은 같은 semaphore가 아니다.
- Callback이 끝났다는 사실과 다음 장치 작업을 제출할 수 있다는 사실도 다르다.
- 성능 수정 뒤 드러난 race는 성능 수정을 되돌릴 이유가 아니라 latent ownership bug를 고칠
  근거다.

새 backend를 연결할 때는 "누가 이 요청을 소유하는가", "어떤 사건이 완료를 증명하는가",
"어느 시점에 buffer와 slot을 재사용할 수 있는가"를 각각 독립적으로 답할 수 있어야 한다.
