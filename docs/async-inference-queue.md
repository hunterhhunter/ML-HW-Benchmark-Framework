# 비동기 추론 큐 측정 가이드

`async_queue`는 요청 제출과 완료 처리를 분리해 bounded queue, 상주 worker,
동적 배칭을 측정하는 프레임워크 자체 실행 모드다. 기존 순차 실행인 `e2e`는
그대로 유지된다.

현재 CLI의 async 결과는 `async_metric_schema_version=2`다. 공식 처리량은
프레임워크의 Runtime-call이 결과를 반환한 시점에 성공 sample을 세며,
`async_runtime_completed_samples_per_sec`로 저장한다. 이는 NPU 내부 커널만의
처리량이 아니다. 기존 `e2e` 모드와 schema v1 async 처리량의 측정 경계도
다르므로 직접 비교하지 않는다.

## 설계 목표와 두 패스 계약

목표는 async 실행에서 **결과를 반환한 Runtime-call의 완료 처리율**을 측정하는
것이다. 입력 공급·큐 대기·worker의 runtime 입력 준비가 공급률을 제한하는
현실은 유지하되, 모델별 decoder·후처리·evaluator의 실행 시간과 flush 뒤 정리
시간은 공식 처리량에서 제외한다. 동일 실행에서 정확도 계산까지 완료해야만
성능 결과를 공개하려 하면 완료 경계에 후처리 비용이 다시 섞인다. 기본 설정의
`async_queue` 명령은 아래 두 패스를 순서대로 실행하고 하나의 run ID에 묶는다.

```text
같은 artifact·데이터·전처리·sample 상한·worker 설정
  ├─ 1. QUALITY: offline → Runtime-call → decoder/후처리 → evaluator → 품질·표본 검증
  └─ 2. RUNTIME_ONLY: 요청 공급 → Runtime-call 반환 시각 기록 → 완료·오류 정리
                       └─ decoder/후처리/evaluator 호출 없음
```

정확도 패스는 `offline`으로 실행하며 QPS·SLO·최소 측정시간 판정을 적용하지 않는다.
evaluator 오류, lifecycle 실패 또는 완료 sample과 evaluator sample의 불일치가
있으면 성능 패스를 시작하지 않고 `accuracy_failed`로 저장한다. `--max-samples`는
데이터셋 크기 이내의 **상한**이며, 두 패스의 실제 처리 sample 수를 다시 대조한다.
`--async-pass runtime-only`를 지정하면 정확도 패스와 evaluator·decoder 구성을
건너뛰고 성능 패스만 실행한다. 이때 `quality_status=skipped`,
`quality_phase.reason=requested_runtime_only`, `comparison_eligible=false`로
저장한다. 입력 loader와 warmup, Runtime-call 성공·오류 검사는 그대로 실행하며
정확도 수치는 만들지 않는다. `LatencyOnlyEvaluator`처럼 정확도 evaluator가
없는 모델을 기본 설정으로 실행하면 성능 패스만 실행하되
`quality_status=unavailable`, `comparison_eligible=false`로 저장한다.
두 패스 모드에서는 각각 warmup을 수행한다. 정확도 패스 뒤 loader cursor를 초기화하고
새 `InferenceEngine`과 runtime executor를 구성하지만 runtime 모델은 안전한
종료가 확인된 뒤 한 번만 unload한다. 선택적 hardware monitor와 request trace는
성능 패스에 연결된다.

| 구현 위치 | 역할 |
|---|---|
| `framework/src/main.py`의 `execute_benchmark()` | run ID 예약, 정확도 gate, loader cursor 초기화, 성능 패스와 결과 저장 |
| `framework/src/core/async_inference/types.py`의 `AsyncPassKind` | `QUALITY`/`RUNTIME_ONLY` 구분. `offline`/`server_like` 시나리오와 독립 |
| `framework/src/core/inference_engine.py`의 `run_async()` | 패스 종류를 async controller와 completion coordinator로 전달 |
| `framework/src/core/async_inference/engine.py`의 `_worker()` | `executor.execute()` 직후 성공 결과의 sample·token 수와 반환 시각 기록 |
| `framework/src/core/async_inference/completion.py`의 `_handle()` | 정확도 패스에서만 decode·label 준비·`add_batch()` 실행. 두 패스 모두 소유권·timeout·terminal 정리 수행 |
| `framework/src/core/async_inference/metrics.py`와 `runner.py` | Runtime-call 완료 집계, producer 종료 시각, 분모·SLO·유효성 판정 |

벤더 SDK나 모델별 decoder/evaluator 자체는 이 패스 구분 때문에 바뀌지 않는다.
벤더 callback 대기, runtime이 반환 전에 수행하는 출력 복사·정규화 등은
`executor.execute()`의 일부이므로 Runtime-call 경계 **안**에 남는다.

## MLPerf LoadGen과의 관계

이 모듈은 MLPerf LoadGen을 가져오거나 다시 구현한 것이 아니다. LoadGen에서
검증된 다음 원칙을 신뢰성 설계의 레퍼런스로 사용했다.

- 요청 발행과 완료 보고의 분리
- 요청 ID별 exact-once 완료와 outstanding 추적
- monotonic clock 기반 시간 측정
- 평균뿐 아니라 tail percentile을 포함한 결과 기록
- 최소 표본·시간 조건과 불변식에 따른 유효성 판정

`async_queue` 모듈과 실행 경로는 `mlperf_loadgen`을 import하거나 사용하지 않으며
LoadGen의 SUT/QSL API, 시나리오, 공식 validity 규칙, 로그 형식을 제공하지 않는다.
기존 `framework/src/adapters/loadgen_adapter.py`는 이 경로와 분리된 비활성 legacy
skeleton이고 이번 구현의 통합 대상이 아니다. MLPerf submission 패키지,
compliance test, audit 대응도 현재 범위가 아니다. 따라서 이 모드의 결과는
MLPerf 결과가 아니고, MLPerf 제출이나 공식 결과와 직접 비교할 수 없다.

목표는 LoadGen과 API·로그를 호환하는 것이 아니라, 그 정도로 신뢰할 수 있는
자체 비동기 측정 모듈을 만드는 것이다.

## 실행 모드와 부하

| 구분 | 동작 | 주 용도 |
|---|---|---|
| `e2e` | 데이터 배치를 읽고 runtime 호출과 평가를 순차 반복 | 기존 기준선과 단순 end-to-end 실행 |
| `async_queue` + `offline` | 데이터셋을 가능한 한 빠르게 bounded queue에 공급 | 최대 처리량, 동적 배칭, backpressure 관찰 |
| `async_queue` + `server_like` | seed 기반 지수분포 간격으로 target QPS에 맞춰 요청 발행 | 서비스형 부하, 포화와 tail latency 관찰 |

`offline`과 `server_like`는 프레임워크 자체 시나리오다. 같은 이름의 MLPerf
시나리오를 구현하거나 동일한 결과를 만든다는 뜻이 아니다. schema v2의
`server_like`는 두 패스의 표본 수를 같게 만들기 위해 고정 요청 수를 먼저 정한다.
`--max-samples`가 있으면 그 수와 고유 데이터셋 크기 중 작은 값을 쓰고, 없으면
seed 기반 발행 간격으로 최소 표본·시간에 필요한 수를 계산하되 고유 데이터셋
크기를 넘기지 않는다. 정확도 패스는 이 고유 샘플을 offline으로 한 번씩 실행한다.
성능 패스는 첫 요청 발행 시점부터 원래 target QPS와 SLO를 적용한다. 고유 샘플이
부족해 최소 표본·Runtime-call 완료 시간에 못 미치거나 요청이 실패하면
`invalid`다. 따라서 데이터셋보다 긴 반복 부하를 이 결과로 주장하지 않는다.

### 주요 CLI 기본값

| 옵션 | 기본값 | 제약 또는 의미 |
|---|---:|---|
| `--inference-mode` | `e2e` | `e2e`, `async_queue` |
| `--scenario` | `offline` | async 전용 |
| `--async-pass` | `quality-and-runtime` | `runtime-only`를 선택하면 정확도 패스를 생략 |
| `--batch-size` | `1` | async에서는 동적 최대 batch size |
| `--queue-capacity` | `256` | batch size 이상 |
| `--worker-count` | `1` | runtime capability 이하여야 함 |
| `--batch-timeout-ms` | `1.0` | 동적 batch가 채워지기를 기다리는 최대 시간 |
| `--submit-timeout-sec` | `30.0` | Offline의 queue 공간 대기 제한 |
| `--flush-timeout-sec` | `300.0` | drain·callback·종료에 사용하는 제한시간 |
| `--request-timeout-ms` | `0` | `0`이면 요청 timeout 비활성 |
| `--min-samples` | `100` | 완료 sample 기준 유효성 최소값 |
| `--min-duration-sec` | Offline `0`, Server-like `10` | 측정시간 유효성 최소값 |
| `--schedule-seed` | `0` | Server-like 발행 간격 재현 seed |

Server-like에서는 `--target-qps`가 필수다. `--max-samples`를 지정해 최소 표본이나
최소 시간보다 먼저 끝나면 실행은 저장되지만 `invalid`가 된다. `--max-steps`는
`e2e` 전용이며 async에서는 `--max-samples`를 사용한다. async 전용 옵션을
`e2e`와 함께 전달하면 runtime 초기화 전에 입력 오류로 종료한다.

ONNX Runtime 인스턴스의 동시 worker capability는 현재 1이므로 CPU 기준 실행은
`--worker-count 1`을 사용한다. 동적 batch 축을 가진 ONNX 모델은 1보다 큰
`--batch-size`를 사용할 수 있다. 단, 독립 요청 coalescing에는 모델과 runtime의
dynamic batch 지원뿐 아니라 dataloader/pipeline metadata의
`is_static_batched=False`도 필요하다. `is_static_batched=True`인 loader는 이미
batch된 단일 request 경로를 사용하므로 `max_batch_size > 1`이어도 독립 요청을
합치지 않고 관측 batch가 1일 수 있다. 다른 장치와 runtime은 해당 capability를
실제 장치에서 검증한 뒤 worker 수와 batch 크기를 늘려야 한다.

Hailo `hailo8`/`hailo10h` target은 `native_async` capability와
`HailoRuntime.create_native_backend()`를 통해 `NativeAsyncRuntimeExecutor`를 선택한다.
실제 in-flight 수는 worker, framework queue, SDK async queue 크기 중 최솟값이며,
callback까지 bindings와 buffer를 유지한다. 버전 조합,
timeout 옵션, ResNet50·YOLOv5m 명령은 [Hailo native async runtime 가이드](hailo-async-runtime.md)를
참고한다. InferModel API가 없는 legacy VStreams 환경은 동기 `e2e`만 지원하고
`async_queue` 요청은 명시적으로 거부한다.

### 운영 디버깅 실행

저장 위치를 분리하고 lifecycle 로그와 요청 trace를 함께 남기는 CPU 실행 예시는
다음과 같다.

```bash
cd framework
.venv/bin/python src/main.py \
  --model resnet50 \
  --onnx models/Kalray_resnet50/resnet50-v1-7s.onnx \
  --dataset datasets/imagenet_1k \
  --target cpu \
  --inference-mode async_queue \
  --scenario offline \
  --max-samples 100 \
  --min-samples 100 \
  --batch-size 2 \
  --queue-capacity 256 \
  --worker-count 1 \
  --batch-timeout-ms 1 \
  --results-path /tmp/mlhw-results/benchmark_results.csv \
  --debug \
  --save-request-trace
```

`RUN_ID_RESERVED=<id>`는 artifact가 예약되어 실행이 시작됐음을 뜻한다.
`RUN_ID=<id>`는 같은 ID의 terminal CSV record가 영속화됐음을 뜻한다. 따라서
프로세스 감시자는 전자만 있고 후자가 없으면 stderr와 예약 상태를 먼저 확인해야
한다. `--debug`의 lifecycle 로그는 reservation, warmup, measurement, artifact
저장과 unload 같은 coarse phase를 보여 주며 요청별 이벤트 로그가 아니다. 이
로그의 출력과 저장은 요청별 측정 구간 밖에 있다. async 모드에서는 `--debug`가
evaluator나 decoder의 prediction, label, score, tensor 출력을 켜지 않는다. 개별
요청의 timestamp, worker, batch, terminal status를 사후 분석하려면 반드시
`--save-request-trace`로 생성한 JSONL trace를 사용해야 한다. CSV의
`details_path`, `failure_details_path`, `request_trace_path`는 `--results-path`의
상위 디렉터리를 기준으로 연결된다.

## 측정 경계

데이터의 `load_by_index()`와 loader 단계 전처리는 `issued_ns` 전에 수행되므로 요청별
`async_e2e_latency`에서 제외된다. warmup, runtime load, 결과 저장도 측정 구간에
포함되지 않는다. 데이터 준비 시간의 합은 sidecar의
`producer.producer_load_ms`에서 별도로 확인한다.

schema v2의 공식 처리량은 `async_runtime_completed_samples_per_sec`다. 성능
패스에서 첫 요청의 `issued_ns`를 측정 시작점으로 삼고, worker의
`executor.execute()`가 성공한 출력을 반환한 직후 `runtime_finished_ns`에
request·sample·생성 token을 기록한다. 출력이 없거나 오류인 호출은 성공 수에
넣지 않는다. `None`, 빈 출력 사전, 값이 모두 `None`인 출력 사전은 결과 없음으로
판정하지만, 검출 결과 0개를 나타내는 빈 배열은 유효한 출력이다. 완료 coordinator는
이후에도 결과 소유권과 terminal 상태를 검증하며, runtime 성공 수와 terminal 완료
수가 다르면 run을 invalid 처리한다.

```text
async_runtime_measurement_duration_sec
  = (max(producer_finished_ns, last_runtime_finished_ns) - first_issued_ns) / 1e9
async_runtime_completed_samples_per_sec
  = runtime 반환에 성공한 sample 수 / async_runtime_measurement_duration_sec
async_runtime_completed_tokens_per_sec
  = 최종 응답에서 센 생성 token 수 / 같은 측정시간
```

첫 sample의 로딩·전처리는 첫 `issued_ns`보다 앞에 있어 분모에 들어가지 않는다.
반면 측정 시작 뒤 이어지는 sample의 입력 로딩, queue 대기와 worker 입력 준비는
마지막 반환 또는 producer 종료를 늦추면 처리량을 제한한다. decoder·후처리·
evaluator와 flush 뒤 정리는 공식 분모에 넣지 않는다. 예를 들어 성공 sample
500개의 마지막 반환이 시작 뒤 4.8초, producer 종료가 5.0초, flush 종료가
5.4초라면 공식 처리량은 `500 / 5.0 = 100 samples/s`다.

이는 NPU 코어만의 처리량이 아니라 프레임워크 Runtime-call 결과 반환 기준
처리량이다. LLM은 스트림 token event마다 완료 수를 늘리지 않고 최종 응답
한 건을 request 한 건으로, 생성 token 수를 그 최종 완료에서 한 번만 센다.
스트림 event는 TTFT/TPOT 진단에 사용한다. LLM 벤더 스트림에서 최종 응답이
아예 없으면 실패로 기록하며, 실제 최종 응답이 생성 token 0개를 반환한 경우와
구분한다. 최소 측정시간과 `server_like` SLO도
각각 이 새 측정시간과 `issued → runtime_finished` 지연으로 판정한다.

과거 schema v1의 처리량 측정 구간은 첫 요청의 issue 시점부터 flush가 끝난
시점까지였다. 하드웨어 monitor를 사용하면 engine start 뒤 producer가 sample
load와 request issue를 시작하기 전에 bounded callback으로 먼저 시작하고 flush
직후 정지한다. Monitor startup 시간은 첫 request의
`issued_ns`, `submit_wait`, `e2e_latency`, `measurement_duration_sec`에
포함되지 않는다. 따라서 hardware monitor의 실제 활성 구간은 요청
latency 측정 구간보다 먼저 시작해 첫 sample 준비 시간을 포함할 수 있다.

```text
scheduled ── issued ── enqueued ── runtime_started ── runtime_finished ── completed
     │           │          │              │                   │              │
     └ scheduler ┘          │              │                   │              │
         delay              └ submit_wait ┘                   │              │
                                  └──── queue_wait ───────────┘              │
                                                    └ service_time ┘         │
                                                            └ completion ───┘
                                                               overhead

async_e2e_latency
  = submit_wait + queue_wait + service_time + completion_overhead
```

| sidecar `timing_ms` 키 | 경계 | 해석 |
|---|---|---|
| `scheduler_delay` | `issued - scheduled` | Server-like 목표 발행시각 대비 지연. 요청 e2e 합에는 포함되지 않음 |
| `submit_wait` | `enqueued - issued` | admission과 bounded queue 공간을 기다린 뒤 실제 publication까지 |
| `queue_wait` | `runtime_started - enqueued` | queue 체류, batch coalescing, collate와 runtime input 준비 |
| `service_time` | `runtime_finished - runtime_started` | worker의 `executor.execute()` 구간. 벤더 backend의 callback 대기·출력 반환 작업이 포함될 수 있으며 순수 NPU 커널 시간은 아님 |
| `runtime_ready_latency` | `runtime_finished - issued` | 제출 대기·queue 대기·Runtime-call을 합친 요청별 지연. 성능 패스 SLO 경계 |
| `completion_overhead` | `completed - runtime_finished` | 정확도 패스는 decoder/evaluator 포함. schema v2 성능 패스는 이를 호출하지 않는 완료 bookkeeping 구간 |
| `e2e_latency` | `completed - issued` | 요청 제출 시작부터 completion timestamp까지 |

정확도 패스의 `completed` timestamp는 decoder와 evaluator 처리가 끝난 뒤
기록된다. 성능 패스에서는 그 둘을 호출하지 않지만 `completed`는 여전히
후속 완료 처리 시각이다. 공식 처리량과 runtime-ready latency는 별도의
`runtime_finished` 시각을 사용한다. 결과 파일 저장은 두 패스의 요청 latency에
포함되지 않는다. 성능 패스의 `async_e2e_latency_*`는 이 구간까지 포함한 진단
지표이지 공식 Runtime-call latency가 아니다.

각 timing 분포에는 `count`, `min`, `max`, `mean`, `sum`, `p50`, `p90`, `p95`,
`p97`, `p99`, `p99_9`가 millisecond 단위로 저장된다. 정상 요청은 네 구간의 합과
e2e가 0.05 ms 이내에서 일치해야 한다. 표본이 1,000개보다 적어도 P99.9는
계산하지만 `tail_percentile_low_sample_count` 경고를 남긴다.

기존 evaluator가 내보내는 `Average Latency (ms)`나 `P99 Latency (ms)`는 runtime
호출 중심의 batch 기록에서 계산한다. 요청 단위의 queue 대기와 completion을
포함하는 `async_e2e_latency_*`와 같은 범위로 해석하면 안 된다.

## 결과 지표 인벤토리

### CSV와 최종 metric

`framework/results/benchmark_results.csv`의 한 행이 한 run이다. async 행에는
`inference_mode`, `scenario`, queue·worker·batch 설정, target QPS, seed,
`async_run_status`, `async_invalid_reasons`, 정상·failure sidecar와 trace 상대 경로가 추가된다.
정확도 패스의 `quality_*` metric, `hw_*` hardware metric, 다음 async summary도 같은 행의
metric column으로 저장된다.

| 지표군 | 현재 생성되는 키 | 의미 |
|---|---|---|
| 요청 수 | `async_submitted_requests`, `async_accepted_requests`, `async_completed_requests`, `async_failed_requests`, `async_rejected_requests`, `async_timed_out_requests`, `async_outstanding_requests` | 요청 lifecycle count. timeout은 terminal category와 별개인 진단 subset |
| sample·token | `async_completed_samples`, `quality_evaluator_samples`, `async_runtime_completed_tokens_per_sec` | 성능 패스 terminal sample, 정확도 패스 평가 sample, 최종 응답 token의 Runtime-call 완료 처리율 |
| 처리율 | `async_issued_requests_per_sec`, `async_runtime_completed_samples_per_sec`, `async_runtime_achieved_qps`, `async_runtime_measurement_duration_sec` | 성능 패스 issue request/s와 Runtime-call 결과 반환 sample/s. 공식 처리량은 후자이며 새 duration이 그 분모다 |
| Server-like | `async_target_qps`, `async_runtime_target_qps_gap` | target과 새 처리량의 차이. Server-like에서만 생성 |
| latency | `async_runtime_ready_latency_p50_ms`, `async_runtime_ready_latency_p95_ms`, `async_runtime_ready_latency_p99_ms`, `async_queue_wait_p99_ms`, `async_service_time_p99_ms` | 공식 경계의 요청별 지연과 queue/service 진단. 완료 후 정리 포함 `async_e2e_latency_*`도 별도 진단으로 남음 |
| queue·worker | `async_queue_depth_max`, `async_worker_utilization` | queue 최대 깊이와 전체 worker busy 비율 |
| SLO | `async_over_latency_slo_requests` | `--latency-slo-ms`를 넘은 요청 수. 옵션이 없어도 0으로 생성 |
| 상태 | `async_run_status`, `async_invalid_reasons` | 자체 판정인 `valid`/`invalid`와 쉼표 구분 reason |

`async_runtime_target_qps_gap`이 음수라고 단독으로 실패를 뜻하지는 않는다. 음수 폭이
커지는 동시에 queue wait, queue depth, e2e P99가 증가하는지를 함께 봐야 포화
여부를 판단할 수 있다. 처리량 역시 모델, 장치, batch, queue, worker, 부하 설정에
따라 달라지므로 `async_queue`가 항상 `e2e`보다 빠르다고 해석할 수 없다.

`--latency-slo-ms`를 설정하면 `async_over_latency_slo_requests`는
`issued → runtime_finished`가 SLO를 넘은 요청 수를 센다.
`async_runtime_ready_latency_p99_ms`가 SLO보다 크면
`latency_slo_not_met`으로 run을 invalid 처리한다.

과거 CSV 행의 `async_completed_samples_per_sec`,
`async_completed_tokens_per_sec`, `async_achieved_qps`는 schema v1의
decoder·evaluator·flush 포함 의미 그대로 보존한다. schema v2 새 행에는 이
처리량 키를 채우지 않으며, 두 버전을 직접 비교하면 안 된다. 정확도 수치는
`quality_*` 컬럼과 `quality_phase` sidecar에 저장된다.

### 결과 판독 순서

1. `async_metric_schema_version=2`와
   `async_measurement_boundary=framework_runtime_call_return`을 확인한다.
   v1·`e2e` 결과와 같은 처리량 열에서 직접 순위를 매기지 않는다.
2. `async_run_status=valid`, `comparison_eligible=true`,
   `quality_status=passed`와 두 패스의 sample 수 일치를 확인한다.
   `quality_status=passed`는 evaluator가 오류 없이 sample을 집계했다는
   **검증 상태**이지 AP·Top-1 등의 임계값 통과를 뜻하지 않는다. 실제 품질 수치,
   dataset 이미지·label 경로, artifact 동일성을 별도로 확인한다. 정확도
   evaluator가 없어 `quality_status=unavailable`인 결과와 의도적으로 정확도를
   건너뛰어 `quality_status=skipped`인 결과는 성능 진단에는 쓸 수 있지만 품질까지
   검증된 공식 비교 결과는 아니다.
3. 공식 처리량은 `async_runtime_completed_samples_per_sec`, 공식 요청 지연은
   `async_runtime_ready_latency_*`로 읽는다. `async_service_time_*`는
   Runtime-call 구간 진단이며 queue 대기를 빼므로 요청 지연과 다르다.
   `async_issued_requests_per_sec`도 완료 처리량이 아니며, 진단용 전체
   `measurement_duration_sec`를 분모로 쓴다.
4. `producer.producer_load_ms`, `runtime_measurement`, queue wait·depth,
   worker utilization, 실패·timeout·native callback 수를 함께 본다.
   `producer_load_ms`는 개별 load 호출의 시간 합이며 전체 producer wall time과
   동일하지 않다. producer 종료가 공식 분모를 결정했다면 낮은 samples/s를
   NPU 자체의 처리 한계로 단정하지 않는다.

비교 표에는 지표 이름과 함께 scenario, 모델·artifact·dataset/label, sample 수,
batch·worker·queue 설정, 정확도 수치와 상태를 적는다. 보드/host와 입력 공급
경로가 다르면 이 값은 각 배포 구성의 Runtime-call 완료 처리량이며, 동일 조건의
NPU 코어 성능 순위는 아니다. native async executor의 in-flight가 남거나
중복·지연 callback, 제출 실패, timeout 카운터가 0이 아니면 완료 수가 맞더라도
비교 가능한 valid 결과로 인정하지 않는다.

### JSON sidecar

`framework/results/details/{run_id}.json`은 `schema_version="1.0"`과 동일 run ID를
가지며 상세 진단을 보존한다. 이 `schema_version`은 sidecar 파일 구조의 버전이고,
CSV와 sidecar의 `async_metric_schema_version=2`는 처리량 계측 계약의 버전이다.
sidecar에는 `async_measurement_boundary=framework_runtime_call_return`도 기록한다.
이 두 필드는 정상 결과와 정확도 검증 실패 결과에 모두 남고, 파일 구조의
`schema_version`과는 다른 필드다.

| section | 내용 |
|---|---|
| `measurement`, `runtime_measurement` | 전자는 성능 패스의 completion·flush까지 포함한 진단 구간, 후자는 공식 처리량 시작·종료와 producer/runtime 완료 시각. `runtime_measurement_duration_sec`이 공식 분모 |
| `config` | 실제 적용한 scenario, queue, worker, batch, timeout, 최소 조건, QPS, seed, SLO |
| `producer` | attempted/accepted/rejected와 `producer_load_ms`, 선택적 producer error |
| `counts` | event-driven raw count와 terminal `outstanding` snapshot. terminal/sample/token count와 `rejected:<reason>` 등이 발생한 경우 포함 |
| `counter_invariants` | 두 counter 등식의 개별 결과와 종합 `valid` |
| `timing_ms` | scheduler, submit, queue, service, completion, e2e 전체 분포와 LLM timing 분포 |
| `queue` | depth min/max/time-weighted mean, transition sequence 진단, full event, submit block 합, inflight min/max/mean |
| `workers` | 전체 utilization과 worker별 busy ns, batch 수, sample 수 |
| `batch_size` | worker가 구성해 runtime 실행을 시도한 batch size의 전체 분포. collate, input 준비, runtime 실패도 시도 크기를 기록할 수 있음 |
| `failure_types`, `failure_request_examples` | 오류 타입별 횟수와 타입당 최대 5개 request ID |
| `generation` | 완료 token 수, timing source, 실제 event TTFT와 runtime-reported TTFT/TPOT 분포 |
| `quality_phase`, `performance_phase` | CLI의 정확도·성능 두 패스 상태, sample 수와 품질 진단 |
| `hardware_metrics` | `hw_` prefix의 monitor 결과 |
| `status`, `invalid_reasons`, `warnings` | 자체 run 판정과 진단 |
| `flush_duration_ms`, `outstanding_request_ids` | drain 시간과 종료 시 남은 요청 |
| `lifecycle_errors`, `callback_errors`, `serialization_errors` | 실패 단계별 제한된 진단 정보 |
| `outstanding_callbacks` | deadline 뒤에도 살아 있는 callback의 ID, phase, thread와 상태 |
| `callback_timeout_limitation` | outstanding callback이 있을 때 기록하는 Python thread 강제 종료 한계 |
| `callback_gc_external_finalization_possible` | GC quarantine 중 callback이 반환될 때 외부 process-global GC가 다른 thread에서 finalizer를 실행할 수 있다는 조건부 진단 |
| `quality_evaluation_skipped` | 이 sidecar의 성능 패스 값은 `runtime_only`; 정확도 평가 결과와 상태는 `quality_phase`에서 확인 |
| `persistence_errors` | trace 또는 sidecar 저장 실패 뒤 CLI가 추가하는 선택적 artifact 진단. sidecar 자체 저장 실패 시에는 그 sidecar에 기록되지 않을 수 있음 |
| `run` | 모델, task, backend, device, batch, warmup, target metadata |

이미 정상 sidecar 또는 CSV가 commit된 뒤 runtime unload나 artifact publication에서
fatal exception이 발생하면 기존 artifact는 수정하지 않는다. 이 경우
`framework/results/details/{run_id}.failure.json`에 immutable recovery record를
별도로 저장한다. CSV가 아직 writable이면 `failure_details_path`로 이 record를
연결하고 invalid 행을 commit한다. CSV가 이미 commit됐으면 행을 보존하고 stderr에
run ID와 deterministic recovery path를 출력한다. 정상 sidecar, CSV, recovery
record는 모두 no-overwrite이며 같은 run ID의 기존 bytes를 교체하지 않는다.

Queue depth와 inflight의 mean은 단순 event 평균이 아니라 각 상태가 지속된 시간을
반영한 time-weighted 평균이다. Worker utilization은 모든 worker의 service busy
시간 합을 `worker_count × measurement_duration`으로 나눈 진단값이다. 여기서
`measurement_duration`은 flush까지 포함하므로 공식 처리량 분모와 다르다.
`batch_size`는 요청 설정값이 아니라 worker가 구성해 runtime 실행을 시도한 sample 수다.
Collate, runtime input 준비 또는 runtime 호출이 실패해도 해당 시도 크기는 기록될
수 있으므로 성공한 runtime 호출만의 분포로 해석하면 안 된다.

### 선택적 request trace

`--save-request-trace`를 사용하면
`framework/results/traces/{run_id}.jsonl`을 만든다. 각 줄에는 request/sample ID,
terminal status, 여섯 timestamp, worker ID, worker가 구성한 시도 batch size,
timeout 여부, sample count와 제한된 오류 요약만 기록한다. input, label, output
tensor, prompt는 기록하지 않는다.

Trace writer는 bounded queue를 사용한다. 포화로 누락된 row는 측정을 중단시키지
않고 `request_trace_dropped:<n>` 경고로 남긴다. Trace는 기본 비활성이며, trace
I/O가 측정에 미칠 수 있는 영향도 비교 조건에 포함해야 한다.

## 유효성, 경고와 종료 코드

`valid`/`invalid`는 프레임워크 자체 신뢰성 판정이며 MLPerf validity가 아니다.
다음 counter 불변식을 검사한다.

```text
submitted = accepted + rejected
accepted = completed + failed + outstanding
flush 성공 후 outstanding = 0
```

`timed_out`은 완료·실패에 더해질 수 있는 subset이므로 위 등식에 별도로 더하지
않는다. 요청 하나라도 timeout이면 run은 invalid다. CLI는 invalid 결과도 가능한
artifact까지 저장하고 `RUN_ID=<id>`를 출력한 뒤 종료 코드 1을 반환한다.

runner 호출이 warmup, pipeline 또는 engine 구성처럼 worker 시작 시도 전에
실패하면 runner가 runtime unload 안전 상태를 유지하므로 CLI가 이미 load된
runtime을 해제한다. 시작 시도 직전에는 이 상태를 철회하며, engine shutdown이
성공하고 outstanding request가 없다는 사실을 runner가 확인한 경우에만 다시
허용한다. 안전성이 증명되지 않은 예외에서는 CLI가 runtime을 임의로 unload하지
않는다. Cleanup 실패는 최초 runner 예외를 대체하지 않고 secondary diagnostic으로
남긴다. 정상 종료에서도 outstanding이 0이면 runtime unload를 정상 details와 CSV
publication 전에 수행한다. 따라서 unload fatal을 이미 valid로 공개된 새 run으로
숨기지 않는다. outstanding이 남아 있으면 unload를 건너뛰고 invalid artifact를
가능한 범위에서 저장한다.

예약 뒤 발생한 fatal exception은 가능한 경우 같은 run ID로 invalid CSV row와
제한된 failure sidecar를 남기고 `benchmark_exception`을 기록한다. Warmup처럼
측정 시작 전 실패는 0으로 확인된 counter snapshot을 저장한다. 측정 시작 뒤
실패에서 신뢰할 수 있는 terminal counter snapshot을 얻지 못한 경우에는
`counts: null`, `counts_available: false`로 명시하며 추정값을 만들지 않는다. 원본
traceback은 stderr에서 확인하고 sidecar의 `failure`에는 phase, 제한된 error type,
generic error message만 사용한다.

이미 commit된 정상 artifact 또는 failure persistence 오류 때문에 별도 recovery
record가 필요하면 `details/{run_id}.failure.json`에 같은 제한을 적용한다. cleanup과
persistence secondary error는 allowlist된 phase/type과 generic message만 보존하며
원본 exception message, traceback, sample payload를 복사하지 않는다. CSV append의
commit 여부가 불명확하면 reservation transaction state를 확인해 exact pending row만
retry한다. consumed row를 다시 쓰거나 기존 정상 sidecar를 failure sidecar로
덮어쓰지 않는다.

현재 core와 CLI가 생성할 수 있는 invalid reason은 다음과 같다.

- 표본·부하: `no_samples`, `min_samples_not_met`, `min_duration_not_met`,
  `latency_slo_not_met`, `producer_error`
- 요청·queue: `request_rejected`, `queue_submit_timeout`, `request_failed`,
  `request_timeout`
- lifecycle: `flush_timeout`, `worker_shutdown_failed`,
  `completion_thread_failed`, `callback_timeout`
- 계측·소유권: `counter_invariant_failed`, `timing_invariant_failed`,
  `runtime_terminal_mismatch`, `metrics_unavailable`, `duplicate_completion`,
  `unknown_completion`, `stale_completion`
- 두 패스·native async: `accuracy_failed`, `native_async_failed`,
  `quality_performance_sample_mismatch`, `native_async_inflight_nonzero`,
  `native_async_executor_snapshot_invalid`, `native_async_duplicate_callbacks`,
  `native_async_late_callbacks`, `native_async_submit_failures`,
  `native_async_timeouts`
- 결과 shape·직렬화: `quality_result_invalid`, `hardware_result_invalid`,
  `result_serialization_failed`
- artifact: `request_trace_persistence_failed`,
  `async_details_persistence_failed`
- CLI fatal exception: `benchmark_exception`

`request_rejected`의 세부 원인은 sidecar `counts`의 `rejected:<reason>`에서 본다.
현재 reason에는 queue full, invalid request, submission close/interruption,
completion 또는 metrics unavailable 등이 있다. CSV 저장 자체가 실패하면 그
CSV에 실패 reason을 기록할 수 없으므로 stderr와 종료 코드 1이 최종 신호다.
Offline은 queue 공간을 submit timeout까지 기다린 뒤 실패하면
`queue_submit_timeout`을 추가한다. Server-like는 발행 스케줄을 지키기 위해 queue
full 요청을 즉시 reject하고 `request_rejected`로 run을 invalid 처리한다.

현재 warning은 다음과 같다. Warning만 있고 invalid reason이 없으면 측정 결과는
valid일 수 있다.

- `tail_percentile_low_sample_count`
- `request_trace_write_failed`, `request_trace_dropped:<n>`
- `hardware_monitor_start_failed`, `hardware_monitor_stop_failed`,
  `hardware_monitor_summary_failed`
- `runtime_device_spec_unavailable`
- `quality_metric_namespace_collision`,
  `hardware_metric_namespace_violation`

## e2e 대비 기대 효과

| 기대 지점 | 관찰 방법 |
|---|---|
| 입력 공급, runtime, completion 처리의 overlap | `async_runtime_completed_samples_per_sec`와 producer·worker 진단을 함께 확인 |
| 여러 요청을 queue에 유지하며 동적 batch 구성 | 시도 `batch_size` 분포와 worker batch/sample 수 확인 |
| Offline 최대 공급 또는 seed 기반 Server-like 부하 재현 | scenario, target QPS, seed와 issued rate 확인 |
| 순차 실행에서 숨겨진 queueing과 tail 노출 | queue wait, e2e P95/P99/P99.9, queue depth 확인 |
| 장치 utilization 개선 가능성 | 동일 장치·모델에서 batch/timeout별 throughput과 hardware metric 비교 |
| 메모리와 backpressure 경계 명시 | bounded queue, full event, rejected/timeout count 확인 |

이는 성능 우위를 보장하지 않는다. 개선 여부는 workload와 장치, runtime,
batch/queue/worker 설정에 따라 달라지며 같은 조건의 실제 측정으로 판단해야 한다.

## 위험과 해석 주의점

| 위험 | 결과에 나타나는 신호 | 대응 |
|---|---|---|
| 공급률이 처리 능력을 넘는 포화 | queue depth/wait와 e2e tail 증가, target gap 악화 | target QPS, batch, queue를 단계적으로 바꿔 포화점 확인 |
| backpressure, drop, timeout | queue full, rejected, timed out, invalid reason | capacity와 timeout을 명시하고 invalid run을 성공 결과로 비교하지 않음 |
| batch 대기 때문에 tail 증가 | batch size는 커지지만 queue wait/P99 증가 | `--batch-timeout-ms`를 throughput과 tail 양쪽으로 튜닝 |
| runtime의 concurrency/thread safety 부족 | capability 입력 오류, request/worker failure | 기본 worker 1, 장치별 실제 검증 후 opt-in |
| out-of-order 또는 중복 완료 | duplicate/unknown/stale completion, counter invalid | request ID와 exact-once 상태를 보존하고 해당 run 폐기 |
| 정확도 패스의 decoder/evaluator 실패 | `accuracy_failed`, `quality_phase` 진단 | 성능 패스를 시작하지 않고 입력·label·decoder·evaluator를 먼저 확인 |
| queue·thread·trace 계측 자체의 overhead | 작은 workload에서 상대적으로 큰 차이 | 충분한 표본, trace on/off와 동일한 비교 조건 사용 |
| 품질 또는 sample 수 불일치 | evaluator sample과 completed sample 불일치, counter invalid | 성능보다 먼저 품질·sample count 일치 확인 |
| 서로 다른 latency 범위를 직접 비교 | `service_time`, `runtime_ready_latency`, `async_e2e_latency`가 서로 다름 | 공식 요청 지연은 `issued → runtime_finished`, service는 실행 구간 진단으로 구분 |

공정한 비교를 위해 모델, dataset과 sample 수, 전처리, runtime/device, warmup,
품질 metric, monitor와 trace 설정을 고정한다. `--batch-size`는 e2e에서는 고정
입력 batch, async에서는 동적 최대 batch라는 차이도 결과와 함께 기록해야 한다.

## CI 기준

첫 실제 runtime 기준은 ONNX Runtime의 `CPUExecutionProvider`다. 통합 테스트는
네트워크나 외부 모델 없이 dynamic batch 축을 가진 작은 ONNX 모델을 생성하고,
같은 네 sample을 e2e와 async로 실행한다. 품질과 sample 수가 일치하고 실제 최대
batch size가 2이며 outstanding이 0인 valid 결과인지 검증한다. schema v2의
별도 테스트는 decoder/evaluator 콜백이 성능 패스에서 호출되지 않는지, producer
종료·마지막 Runtime-call 반환·flush 시각으로 계산한 분모, 결과 없는 응답과
runtime/terminal 수 불일치, 정확도 실패 시 성능 미실행, 단일 run ID 저장을
확인한다. 성능 향상이나 특정 latency 수치는 assertion으로 사용하지 않는다.
