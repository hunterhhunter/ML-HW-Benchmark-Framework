# Paper-Grade RNGD Latency Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Furiosa RNGD의 E2E·비동기 추론 결과에서 요청 단위 TTFT/평균 TPOT을 재현 가능하게 보고하고, 실제 토큰 간 p50/p85/p90/p95/p99는 관측 가능성이 증명된 스트림 또는 동일한 `furiosa-llm serve` 인스턴스의 공식 지표에서만 산출한다.

**Architecture:** 현재 `RuntimeExecutor → BatchCompletion → AsyncMetricsCollector` 경계에 불변 `GenerationObservation`을 추가한다. direct `AsyncLLMEngine` 경로는 각 non-empty stream output의 monotonic timestamp와 누적 토큰 수를 전달하고, completion 계층이 request-issued 시각과 결합해 요청 단위 지표를 만든다. 토큰 간 지표는 모든 생성 event가 정확히 한 토큰씩 증가한 요청에서만 `stream-event ITL`로 인정한다. 별도 server benchmark는 `furiosa-llm serve`에 대해 Furiosa가 사용하는 `vllm bench serve` 결과와 `/metrics` 전후 histogram delta를 독립 sidecar로 저장한다.

**Tech Stack:** Python 3.12, `dataclasses`, `time.monotonic_ns`, NumPy, pytest, httpx 0.28.1, prometheus-client 0.24.1, Furiosa-LLM 2026.3.0, vLLM benchmark CLI 0.16.0.

## Global Constraints

- 현재 `feat/rngd-runtime` PR에는 RNGD runtime 기능과 기존 요청 단위 평균 TPOT만 남긴다. 아래 측정 기능은 별도 PR로 분리한다.
- 모든 production change는 그 동작을 요구하는 실패 테스트와 예상 RED를 먼저 확인한 후 작성한다.
- 각 task는 `RED → GREEN → focused regression → commit → review` 순서로 끝낸다.
- 논문 본문에서 backend, scenario, input-token bucket, output-token bucket, concurrency 또는 target QPS가 다른 결과를 하나의 percentile로 합치지 않는다.
- `request TTFT`는 `request.issued_ns → first non-empty output observed_ns`, `backend TTFT`는 `backend_submitted_ns → first non-empty output observed_ns`로 구분한다.
- `request mean TPOT`은 `(last output observed_ns - first output observed_ns) / (generated_tokens - 1)`이다. 생성 토큰이 0 또는 1이면 `None`이며 0 ms로 만들지 않는다.
- direct stream의 한 event가 두 토큰 이상 증가하면 그 요청의 per-token ITL은 관측 불가능하다. event 간 시간을 토큰 수로 나누거나 복제하지 않는다.
- direct stream ITL은 `output_event_count == generated_tokens`이고 모든 positive delta가 1인 요청에서만 계산한다. 전체 완료 요청 coverage가 100%가 아니면 논문용 top-level ITL로 승격하지 않는다.
- server benchmark의 client-observed ITL과 `furiosa_llm_inter_token_latency_seconds`는 서로 다른 측정치다. 전자는 네트워크·SSE 전달을 포함하고 후자는 vendor server 내부 histogram이다.
- Prometheus histogram은 측정 전후 delta만 사용한다. 다른 트래픽이 섞였거나 request/token counter가 맞지 않으면 run을 invalid 처리한다.
- 모든 percentile은 `numpy.percentile(values, PERCENTILES, method="linear")`로 계산하고 method, 표본 수, 원시 bucket 또는 raw event를 함께 저장한다.
- p99 결과를 논문에 사용하려면 workload cell당 성공 요청 1,000개 이상을 기본 하한으로 하고, server ITL gap은 10,000개 이상을 기본 하한으로 한다. 이는 코드의 계산 가능 조건이 아니라 실험 protocol의 보고 조건이다.
- warmup은 percentile sample과 Prometheus delta에 포함하지 않는다.
- `furiosa_llm`, vLLM benchmark CLI는 기본 `framework/requirements.txt`에 넣지 않는다. RNGD 전용 환경을 사용한다.
- SDK가 없는 CI에서는 fake SDK, fake SSE result, fake `/metrics` response만으로 모든 경계와 통계 계약을 검증한다.
- 토큰·prompt·model content를 error message, diagnostics, trace metadata에 기록하지 않는다.

---

## PR/Branch Order

1. 현재 `feat/rngd-runtime`을 main에 먼저 merge한다.
2. 최신 main에서 `feat/generation-latency-metrics`를 만들고 Tasks 1–6을 한 PR로 보낸다.
3. 두 PR이 merge된 main에서 `feat/furiosa-server-itl-metrics`를 만들고 Tasks 7–10을 별도 PR로 보낸다.

```bash
git switch main
git pull --ff-only
git switch -c feat/generation-latency-metrics
```

두 번째 PR 시작 시:

```bash
git switch main
git pull --ff-only
git switch -c feat/furiosa-server-itl-metrics
```

---

### Task 1: Percentile 계약에 p85와 estimator provenance 추가

**Files:**
- Modify: `framework/src/core/async_inference/metrics.py`
- Modify: `framework/tests/test_async_metrics.py`

**Public result contract:**

```python
PERCENTILES = (50.0, 85.0, 90.0, 95.0, 97.0, 99.0, 99.9)
PERCENTILE_METHOD = "linear"
```

- [ ] **Step 1: p85와 estimator metadata를 요구하는 RED test 작성**

`test_timing_distribution_reports_every_percentile_count_and_sum()`의 기대값에 아래를 추가한다.

```python
assert e2e["p85"] == pytest.approx(3.55)
assert result["details"]["statistics"] == {
    "percentile_method": "numpy.percentile(method=linear)",
}
```

- [ ] **Step 2: focused RED 실행**

```bash
cd framework
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_async_metrics.py::test_timing_distribution_reports_every_percentile_count_and_sum -q
```

Expected: `KeyError: 'p85'` 또는 statistics metadata assertion failure.

- [ ] **Step 3: `TimingDistribution.summary()` 구현**

빈 분포와 non-empty 분포 양쪽에 `p85`를 넣고 다음처럼 method를 명시한다.

```python
percentiles = np.percentile(
    values,
    PERCENTILES,
    method=PERCENTILE_METHOD,
)
```

`finalize()` details 최상위에 estimator provenance를 추가한다.

- [ ] **Step 4: 모든 기존 percentile test 갱신 후 GREEN 확인**

```bash
cd framework
PYTHONPATH=src .venv/bin/python -m pytest tests/test_async_metrics.py -q
```

- [ ] **Step 5: commit**

```bash
git add framework/src/core/async_inference/metrics.py framework/tests/test_async_metrics.py
git commit -m "feat(metrics): add p85 percentile provenance"
```

---

### Task 2: 생성 stream observation을 executor protocol에 추가

**Files:**
- Modify: `framework/src/core/runtime_executor.py`
- Modify: `framework/src/core/async_inference/types.py`
- Modify: `framework/src/core/async_inference/engine.py`
- Modify: `framework/src/core/inference_engine.py`
- Modify: `framework/tests/test_native_async_runtime_executor.py`
- Modify: `framework/tests/test_async_engine.py`
- Modify: `framework/tests/test_inference_engine.py`

**New immutable types:**

```python
@dataclass(frozen=True)
class GenerationOutputEvent:
    observed_ns: int
    cumulative_tokens: int


@dataclass(frozen=True)
class GenerationObservation:
    backend_submitted_ns: int
    events: tuple[GenerationOutputEvent, ...]
    source: str
```

`RuntimeExecution`, `NativeAsyncOutcome`, `BatchCompletion`에는 다음 optional field를 끝에 추가한다.

```python
generation_observation: GenerationObservation | None = None
```

- [ ] **Step 1: exact pass-through RED test 작성**

`test_native_async_runtime_executor.py`의 fake callback이 다음 observation을 반환하게 하고 `RuntimeExecution`까지 identity가 아니라 정규화된 값으로 전달되는지 검증한다.

```python
observation = GenerationObservation(
    backend_submitted_ns=100,
    events=(
        GenerationOutputEvent(observed_ns=130, cumulative_tokens=1),
        GenerationOutputEvent(observed_ns=150, cumulative_tokens=2),
    ),
    source="fake_stream",
)
```

검증 항목:

- callback이 반환한 mutable/hostile container를 보존하지 않는다.
- `observed_ns`가 감소하거나 cumulative token count가 감소하면 `NativeAsyncProtocolError`가 된다.
- event 수가 4,096개를 넘거나 source가 128자를 넘으면 protocol error가 된다.
- timeout 후 late callback이 observation을 전달해도 permit/ACK 규칙이 바뀌지 않는다.

- [ ] **Step 2: RED 실행**

```bash
cd framework
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_native_async_runtime_executor.py \
  tests/test_async_engine.py -q
```

Expected: dataclass import 또는 constructor keyword failure.

- [ ] **Step 3: protocol validator 구현**

`_copy_generation_observation()`은 exact non-negative int, monotonic timestamps, non-decreasing cumulative counts, bounded event count/source만 허용한다. prompt, output tensor, text는 이 객체에 넣지 않는다.

`_protocol_outcome()`과 `NativeAsyncRuntimeExecutor.execute()`가 정규화된 observation을 복사하도록 한다.

- [ ] **Step 4: engine handoff 구현**

`engine.py`와 동기 `inference_engine.py`의 모든 `BatchCompletion(...)` 생성 지점에서 `execution.generation_observation`을 전달한다. exception completion은 `None`을 사용한다.

- [ ] **Step 5: GREEN과 regression 확인**

```bash
cd framework
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_native_async_runtime_executor.py \
  tests/test_async_engine.py \
  tests/test_inference_engine.py -q
```

- [ ] **Step 6: commit**

```bash
git add framework/src/core/runtime_executor.py \
  framework/src/core/async_inference/types.py \
  framework/src/core/async_inference/engine.py \
  framework/src/core/inference_engine.py \
  framework/tests/test_native_async_runtime_executor.py \
  framework/tests/test_async_engine.py \
  framework/tests/test_inference_engine.py
git commit -m "feat(runtime): carry generation stream observations"
```

---

### Task 3: Furiosa direct stream event를 손실 없이 관측

**Files:**
- Modify: `framework/src/runtimes/furiosa_llm_rt.py`
- Modify: `framework/tests/test_furiosa_native_backend.py`

**Observation rules:**

- clock은 `time.monotonic_ns()` 하나만 사용한다.
- SDK 기본 `RequestOutputKind.CUMULATIVE` 계약을 유지한다.
- empty output과 동일 cumulative count 반복은 event에 넣지 않는다.
- cumulative token count가 증가한 순간마다 `GenerationOutputEvent` 하나를 기록한다.
- timing source는 `furiosa_async_python_stream`으로 고정한다.
- 기존 `timing_ms`의 `total_ms`, `ttft_ms`, `tpot_ms`는 호환을 위해 유지하되, 1-token generation의 `tpot_ms`는 `None`으로 고친다.

- [ ] **Step 1: one-token, multi-token chunk, repeated-output RED tests 작성**

다음 fake stream 세 가지를 추가한다.

```python
# exact one-token increments
[], [31], [31, 32], [31, 32, 33]

# one event contains two new tokens
[], [31], [31, 32, 33]

# repeated cumulative output is not a token event
[31], [31], [31, 32]
```

첫 번째는 cumulative counts `(1, 2, 3)`, 두 번째는 `(1, 3)`, 세 번째는 `(1, 2)`를 기대한다. 모든 observed timestamp는 이전 timestamp보다 작지 않아야 한다.

- [ ] **Step 2: RED 실행**

```bash
cd framework
PYTHONPATH=src .venv/bin/python -m pytest tests/test_furiosa_native_backend.py -q
```

Expected: `NativeAsyncOutcome`에 observation이 없어서 assertion failure.

- [ ] **Step 3: `_consume_request()` 구현**

`backend_submitted_ns`는 `submit_async()`에서 vendor coroutine을 등록하기 직전에 기록한다. `_consume_request()`는 previous cumulative count와 event list를 관리한다. final output normalization과 exact-once callback 규칙은 변경하지 않는다.

```python
generation_observation = GenerationObservation(
    backend_submitted_ns=started_ns,
    events=tuple(events),
    source="furiosa_async_python_stream",
)
```

- [ ] **Step 4: timing semantics test 추가**

- 0-token: TTFT/TPOT 없음.
- 1-token: TTFT 있음, TPOT 없음.
- 2+ tokens: request mean TPOT만 있음.
- multi-token event가 있어도 event 시간을 토큰 수로 나누지 않음.

- [ ] **Step 5: GREEN 확인 및 commit**

```bash
cd framework
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_furiosa_native_backend.py \
  tests/test_furiosa_llm_runtime.py -q
git add framework/src/runtimes/furiosa_llm_rt.py \
  framework/tests/test_furiosa_native_backend.py
git commit -m "feat(furiosa): expose native stream observations"
```

---

### Task 4: 요청 단위 TTFT/TPOT와 gated stream-event ITL 집계

**Files:**
- Modify: `framework/src/core/async_inference/metrics.py`
- Modify: `framework/src/core/async_inference/completion.py`
- Modify: `framework/tests/test_async_metrics.py`
- Modify: `framework/tests/test_async_completion.py`

**New sealed distributions:**

```text
request_ttft
backend_ttft
request_mean_tpot
generated_tokens_per_request
stream_event_itl
```

**Top-level summary keys:**

```text
async_generation_observed_requests
async_generation_request_ttft_p50_ms
async_generation_request_ttft_p85_ms
async_generation_request_ttft_p90_ms
async_generation_request_ttft_p95_ms
async_generation_request_ttft_p99_ms
async_generation_request_mean_tpot_p50_ms
async_generation_request_mean_tpot_p85_ms
async_generation_request_mean_tpot_p90_ms
async_generation_request_mean_tpot_p95_ms
async_generation_request_mean_tpot_p99_ms
```

`async_generation_stream_itl_p{50,85,90,95,99}_ms`는 `stream_event_itl_coverage == 1.0`일 때만 top-level에 넣는다. coverage가 낮으면 details에 diagnostic distribution만 남기고 warning `generation_stream_itl_incomplete`를 추가한다.

- [ ] **Step 1: deterministic metric RED tests 작성**

한 요청의 `issued_ns=100`, `backend_submitted_ns=120`, events `(150,1), (170,2), (200,3)`일 때 다음을 기대한다.

```python
request_ttft_ms == 0.00005
backend_ttft_ms == 0.00003
request_mean_tpot_ms == 0.000025
stream_event_itl_ms == [0.00002, 0.00003]
```

실제 test에서는 nanosecond fixture를 millisecond 단위가 명확하도록 1,000,000배 스케일한다.

multi-token event `(150,1), (200,3)`인 요청은 request TTFT/mean TPOT에는 포함하지만 stream-event ITL에는 포함하지 않고 coverage를 낮춘다.

- [ ] **Step 2: completion membership RED tests 작성**

- native async의 single request completion은 observation을 기록한다.
- batch에 request가 둘 이상이면 request-level generation timing을 임의 복제하지 않고 `generation_timing_batch_ambiguous` warning을 남긴다.
- failure completion은 generation distribution에 들어가지 않는다.
- first event가 issued time보다 빠르거나 final event가 runtime finished 뒤의 비정상 값이면 `timing_invariant_failed`가 된다.

- [ ] **Step 3: RED 실행**

```bash
cd framework
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_async_metrics.py \
  tests/test_async_completion.py -q
```

- [ ] **Step 4: sealed accounting 구현**

`_SealedAccountingState.timings`와 public compatibility mirror를 동시에 갱신한다. 새 counter는 다음을 포함한다.

```text
generation_observed_requests
generation_stream_exact_requests
generation_stream_unobservable_requests
generation_stream_itl_samples
```

`record_generation()`의 exact signature는 다음으로 확장한다.

```python
def record_generation(
    self,
    generated_tokens: int,
    timing_ms,
    *,
    observation: GenerationObservation | None = None,
    requests: tuple[InferenceRequest, ...] = (),
) -> None:
```

모든 event delta가 1인 요청에서만 인접 event 차이를 `stream_event_itl`에 넣는다. 일부 event만 골라 넣지 않는다.

- [ ] **Step 5: summary/details schema 구현**

`details["generation"]`은 다음 구조를 가진다.

```python
{
    "definitions": {
        "request_ttft_ms": "issued_to_first_nonempty_stream_output",
        "backend_ttft_ms": "backend_submit_to_first_nonempty_stream_output",
        "request_mean_tpot_ms": "first_to_last_output_divided_by_generated_tokens_minus_one",
        "stream_event_itl_ms": "adjacent_single_token_python_stream_events",
    },
    "request_ttft_ms": TimingDistribution.summary(),
    "backend_ttft_ms": TimingDistribution.summary(),
    "request_mean_tpot_ms": TimingDistribution.summary(),
    "generated_tokens_per_request": TimingDistribution.summary(),
    "stream_event_itl_ms": TimingDistribution.summary(),
    "stream_event_itl_coverage": 1.0,
    "timing_sources": {...},
}
```

- [ ] **Step 6: hostile subclass/sealed-state regression 실행**

```bash
cd framework
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_async_metrics.py \
  tests/test_async_completion.py \
  tests/test_native_async_runtime_executor.py -q
```

- [ ] **Step 7: commit**

```bash
git add framework/src/core/async_inference/metrics.py \
  framework/src/core/async_inference/completion.py \
  framework/tests/test_async_metrics.py \
  framework/tests/test_async_completion.py
git commit -m "feat(metrics): report generation latency distributions"
```

---

### Task 5: raw request trace와 논문 재현 metadata 저장

**Files:**
- Modify: `framework/src/core/async_inference/types.py`
- Modify: `framework/src/core/async_inference/completion.py`
- Modify: `framework/src/core/async_inference/trace.py`
- Modify: `framework/src/main.py`
- Create: `framework/tests/test_async_trace.py`
- Modify: `framework/tests/test_async_cli.py`
- Modify: `framework/tests/test_async_result_artifacts.py`
- Create: `docs/rngd-paper-benchmark.md`

**RequestTrace additions:**

```python
generated_tokens: int = 0
backend_submitted_ns: int | None = None
generation_events: tuple[GenerationOutputEvent, ...] = ()
generation_timing_source: str | None = None
```

- [ ] **Step 1: trace JSON RED test 작성**

성공한 생성 request row에 raw ns와 아래 derived field가 모두 저장되는지 검증한다.

```json
{
  "generated_tokens": 3,
  "backend_submitted_ns": 120000000,
  "generation_events": [
    {"observed_ns": 150000000, "cumulative_tokens": 1},
    {"observed_ns": 170000000, "cumulative_tokens": 2},
    {"observed_ns": 200000000, "cumulative_tokens": 3}
  ],
  "generation_timing_source": "furiosa_async_python_stream",
  "request_ttft_ms": 50.0,
  "backend_ttft_ms": 30.0,
  "request_mean_tpot_ms": 25.0
}
```

failure/non-generation row는 generation field를 `null` 또는 빈 list로 기록한다. trace serializer는 exact type와 4,096 event bound를 재검증한다.

- [ ] **Step 2: RED 실행**

```bash
cd framework
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_async_trace.py \
  tests/test_async_result_artifacts.py -q
```

- [ ] **Step 3: completion에서 trace field 구성**

metric 집계와 trace 생성이 같은 normalized observation을 사용하도록 공통 pure helper를 `metrics.py`에 둔다. 서로 별도로 수식을 재구현하지 않는다.

- [ ] **Step 4: run metadata 확장**

`_async_run_metadata()`와 details `run`에 다음을 저장한다.

```text
furiosa_llm_version
python_version
framework_git_commit
percentile_method
model_artifact_path
input/output token policy
sampling policy (temperature, ignore_eos, max_new_tokens)
scenario, target_qps, worker_count, queue_capacity, seed
```

SDK version 조회 실패는 runtime load를 깨지 않고 warning과 `null`로 남긴다. git commit은 read-only `git rev-parse HEAD` 결과만 사용하며 dirty 여부도 boolean으로 저장한다.

- [ ] **Step 5: paper protocol 문서 작성**

`docs/rngd-paper-benchmark.md`에 다음 matrix를 고정한다.

- 모델별, input length bucket별, output length bucket별 결과 분리.
- offline과 server-like 분리.
- server-like target QPS/concurrency별 분리.
- warmup 후 workload cell당 1,000 successful requests 이상.
- 동일 설정 5회 이상 독립 반복, median run과 run-to-run spread 보고.
- `--save-request-trace` 필수, trace dropped가 0이 아니면 논문 run 폐기.
- clock source, percentile method, SDK/driver/FXB hash, host/NPU topology 기록.
- direct stream ITL coverage가 100%가 아니면 해당 ITL percentile을 표에 쓰지 않음.

- [ ] **Step 6: GREEN과 commit**

```bash
cd framework
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_async_trace.py \
  tests/test_async_cli.py \
  tests/test_async_result_artifacts.py -q
git add framework/src/core/async_inference/types.py \
  framework/src/core/async_inference/completion.py \
  framework/src/core/async_inference/trace.py \
  framework/src/main.py \
  framework/tests/test_async_trace.py \
  framework/tests/test_async_cli.py \
  framework/tests/test_async_result_artifacts.py \
  docs/rngd-paper-benchmark.md
git commit -m "feat(results): persist raw generation timing evidence"
```

---

### Task 6: direct-runtime PR acceptance

**Files:**
- Review: all files changed in Tasks 1–5

- [ ] **Step 1: focused regression**

```bash
cd framework
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_runtime_executor.py \
  tests/test_native_async_runtime_executor.py \
  tests/test_async_metrics.py \
  tests/test_async_completion.py \
  tests/test_async_trace.py \
  tests/test_async_engine.py \
  tests/test_inference_engine.py \
  tests/test_furiosa_native_backend.py \
  tests/test_furiosa_llm_runtime.py \
  tests/test_async_cli.py \
  tests/test_async_result_artifacts.py -q
```

- [ ] **Step 2: full offline regression**

```bash
cd framework
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src .venv/bin/python -m pytest tests -q
```

- [ ] **Step 3: schema self-check**

```bash
rg -n "p85|request_mean_tpot|stream_event_itl|GenerationObservation" \
  src tests docs/rngd-paper-benchmark.md
git diff --check
git status --short
```

Expected: no whitespace error; only intended files changed.

- [ ] **Step 4: request review and open PR**

PR title:

```text
feat(metrics): add paper-grade generation latency distributions
```

PR body는 request TTFT, request mean TPOT, gated stream-event ITL의 정의와 “multi-token event는 ITL에 포함하지 않는다”는 제약을 앞부분에 명시한다.

---

### Task 7: Furiosa server benchmark result normalizer

**Files:**
- Create: `framework/src/adapters/vllm_bench_result.py`
- Create: `framework/tests/test_vllm_bench_result.py`
- Create: `framework/tests/fixtures/vllm_bench_serve_0_16_0.json`

**Input contract:** vLLM 0.16.0 `vllm bench serve --save-result --save-detailed` JSON.

**Output namespaces:**

```text
server_client_ttft_p{50,85,90,95,99}_ms
server_client_tpot_p{50,85,90,95,99}_ms
server_client_itl_p{50,85,90,95,99}_ms
server_client_e2el_p{50,85,90,95,99}_ms
server_successful_requests
server_output_tokens
server_output_tokens_per_sec
```

- [ ] **Step 1: capture and freeze one sanitized fixture**

Fixture에는 prompt/response text를 제거하고 schema/version, aggregate metrics, per-request timing arrays와 token counts만 남긴다. fixture가 vLLM 0.16.0에서 나온 provenance를 adjacent README comment가 아니라 JSON metadata에 넣는다.

- [ ] **Step 2: normalizer RED tests 작성**

- requested percentile `(50, 85, 90, 95, 99)`가 모두 있어야 한다.
- non-finite, negative, string-number, missing success count는 reject한다.
- detailed request 수와 aggregate successful request 수가 다르면 invalid diagnostic을 반환한다.
- raw client ITL sample count와 percentile method를 함께 반환한다.
- text fields는 normalized output에 복사하지 않는다.

- [ ] **Step 3: RED 실행**

```bash
cd framework
PYTHONPATH=src .venv/bin/python -m pytest tests/test_vllm_bench_result.py -q
```

Expected: module import failure.

- [ ] **Step 4: pure normalizer 구현**

파일 I/O나 subprocess를 normalizer에 넣지 않는다. JSON mapping을 받아 `{metrics, details, invalid_reasons}`를 반환한다. detailed raw arrays에서 framework가 percentile을 다시 계산하고 vLLM aggregate 값과 tolerance 내 일치하는지 교차 검증한다.

- [ ] **Step 5: GREEN과 commit**

```bash
cd framework
PYTHONPATH=src .venv/bin/python -m pytest tests/test_vllm_bench_result.py -q
git add framework/src/adapters/vllm_bench_result.py \
  framework/tests/test_vllm_bench_result.py \
  framework/tests/fixtures/vllm_bench_serve_0_16_0.json
git commit -m "feat(furiosa): normalize serving benchmark results"
```

---

### Task 8: official Prometheus histogram delta와 quantile 구현

**Files:**
- Modify: `framework/requirements.txt`
- Create: `framework/src/monitors/prometheus_histogram.py`
- Create: `framework/src/monitors/furiosa_server_metrics.py`
- Create: `framework/tests/test_prometheus_histogram.py`
- Create: `framework/tests/test_furiosa_server_metrics.py`

Add dependency:

```text
prometheus-client==0.24.1
```

**Observed metrics:**

```text
furiosa_llm_request_success_total
furiosa_llm_request_generation_tokens
furiosa_llm_time_to_first_token_seconds
furiosa_llm_inter_token_latency_seconds
furiosa_llm_e2e_request_latency_seconds
```

- [ ] **Step 1: histogram fixture와 RED tests 작성**

before/after fixture는 `model_name`, `engine` labels와 cumulative `_bucket`, `_sum`, `_count`를 포함한다. 다음을 검증한다.

- after-before delta 계산.
- counter reset/negative delta reject.
- bucket boundary 또는 label set 변경 reject.
- `+Inf` delta와 `_count` 불일치 reject.
- empty delta는 percentile `None`.
- seconds를 milliseconds로 한 번만 변환.
- p50/p85/p90/p95/p99는 Prometheus histogram interpolation으로 계산.
- raw delta buckets, sum, count를 details에 보존.

- [ ] **Step 2: RED 실행**

```bash
cd framework
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_prometheus_histogram.py \
  tests/test_furiosa_server_metrics.py -q
```

- [ ] **Step 3: parser와 collector 구현**

`prometheus_client.parser.text_string_to_metric_families`로 text exposition을 parse한다. `httpx.Client.get()`은 connect/read/total timeout을 명시하고 redirect를 따르지 않는다. collector는 URL과 exact label filter를 받아 before/after immutable snapshot을 반환한다.

- [ ] **Step 4: provenance 구현**

normalized details에 다음을 넣는다.

```python
{
    "source": "furiosa_server_prometheus_histogram_delta",
    "endpoint": "/metrics",
    "labels": {"model_name": "llama-3.1-8b", "engine": "npu:0"},
    "unit_in": "seconds",
    "unit_out": "milliseconds",
    "quantile_algorithm": "prometheus_histogram_linear_interpolation",
    "raw_delta_buckets": [
        {"le": 0.005, "count": 120},
        {"le": 0.010, "count": 190},
        {"le": "+Inf", "count": 200},
    ],
}
```

- [ ] **Step 5: GREEN과 commit**

```bash
cd framework
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_prometheus_histogram.py \
  tests/test_furiosa_server_metrics.py -q
git add framework/requirements.txt \
  framework/src/monitors/prometheus_histogram.py \
  framework/src/monitors/furiosa_server_metrics.py \
  framework/tests/test_prometheus_histogram.py \
  framework/tests/test_furiosa_server_metrics.py
git commit -m "feat(furiosa): collect server histogram deltas"
```

---

### Task 9: 고정 argv 기반 RNGD server benchmark orchestrator

**Files:**
- Create: `framework/tools/rngd_server_benchmark.py`
- Create: `framework/tests/test_rngd_server_benchmark.py`
- Modify: `framework/src/core/result_store.py`
- Modify: `framework/tests/test_result_store.py`

**CLI:**

```text
--base-url http://127.0.0.1:8000
--model MODEL_ID
--input-tokens N
--output-tokens N
--num-prompts N
--max-concurrency N
--request-rate inf|FLOAT
--seed N
--result-dir PATH
--results-path PATH
--metrics-label key=value
```

- [ ] **Step 1: argv construction RED test 작성**

subprocess에는 shell string이 아니라 다음 fixed argv가 전달되어야 한다.

```text
vllm bench serve
--backend vllm
--base-url BASE_URL/v1
--endpoint /completions
--model MODEL_ID
--dataset-name random
--random-input-len INPUT
--random-output-len OUTPUT
--max-concurrency CONCURRENCY
--num-prompts NUM
--seed SEED
--temperature 0
--ignore-eos
--percentile-metrics ttft,tpot,itl,e2el
--metric-percentiles 50,85,90,95,99
--save-result
--save-detailed
```

`request-rate`가 finite이면 해당 flag를 추가한다. user 입력을 command fragment로 해석하지 않는다.

- [ ] **Step 2: lifecycle RED tests 작성**

fake HTTP/subprocess로 다음 순서를 검증한다.

```text
GET /version
GET /v1/models
GET /metrics (before)
run vllm bench serve
GET /metrics (after)
normalize client result
validate counter deltas
reserve/write details and CSV
```

검증 실패 시에도 raw vLLM result와 before/after metrics text의 hash 및 failure details를 보존하고 run status를 invalid로 저장한다.

- [ ] **Step 3: contamination gates 구현**

다음 조건 중 하나면 `vendor_metrics_scope_mismatch`로 invalid 처리한다.

- `request_success_total` delta != client successful requests.
- generation token histogram sum delta != client output token count.
- model/engine label set이 둘 이상이고 exact filter가 없음.
- benchmark 실행 중 server version/model identity가 바뀜.

ITL count는 일반적으로 `sum(max(output_tokens - 1, 0))`와 비교한다. vendor 정의 차이로 exact match가 불가능한 경우 tolerance를 쓰지 말고 observed 값과 expected 값을 모두 저장한 뒤 mismatch reason을 명시한다.

- [ ] **Step 4: result persistence 구현**

기존 `reserve_run_artifacts()`, `save_async_details()`, `save_result()`를 재사용한다. `inference_mode="external_server"`, `backend="furiosa_llm_server"`로 저장하고 client metric은 `server_client_*`, vendor metric은 `server_vendor_*` namespace를 사용한다.

- [ ] **Step 5: GREEN과 commit**

```bash
cd framework
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_rngd_server_benchmark.py \
  tests/test_result_store.py -q
git add framework/tools/rngd_server_benchmark.py \
  framework/tests/test_rngd_server_benchmark.py \
  framework/src/core/result_store.py \
  framework/tests/test_result_store.py
git commit -m "feat(furiosa): orchestrate reproducible server benchmarks"
```

---

### Task 10: server benchmark acceptance와 장비 protocol

**Files:**
- Modify: `docs/furiosa-rngd-setup.md`
- Modify: `docs/rngd-paper-benchmark.md`

- [ ] **Step 1: RNGD 전용 benchmark 환경 문서화**

Furiosa Apps가 사용하는 버전과 맞춰 별도 venv에 vLLM 0.16.0 benchmark CLI를 설치한다. server는 별도 terminal/process에서 다음 형태로 시작하며 실제 artifact 옵션은 `furiosa-llm serve --help`와 FXB 배포 방식에 맞춰 기록한다.

```bash
furiosa-llm serve ARTIFACT_PATH --host 127.0.0.1 --port 8000
```

server 시작 후 `/version`, `/v1/models`, `/metrics`가 모두 응답해야 한다.

- [ ] **Step 2: paper run matrix 문서화**

각 model/FXB에 대해 최소 다음 cell을 독립 실행한다.

```text
input tokens: 128, 512, 2048
output tokens: 32, 128, 256
max concurrency: 1, 4, 16, 32
request rate: inf(offline-like), 고정 server-like QPS 단계
repetitions: 5
percentiles: 50, 85, 90, 95, 99
```

모델 max length나 장비 수 때문에 불가능한 cell은 조용히 제외하지 않고 exclusion table에 이유를 기록한다.

- [ ] **Step 3: fake full regression**

```bash
cd framework
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_vllm_bench_result.py \
  tests/test_prometheus_histogram.py \
  tests/test_furiosa_server_metrics.py \
  tests/test_rngd_server_benchmark.py \
  tests/test_result_store.py -q
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src .venv/bin/python -m pytest tests -q
git diff --check
```

- [ ] **Step 4: 실제 장비 smoke run**

```bash
furiosa-smi info
fxb show MODEL.fxb
```

한 개의 작은 workload cell로 먼저 실행하고 다음을 확인한다.

- client successful request count와 server counter delta 일치.
- client output token count와 server generation-token delta 일치.
- raw vLLM detailed result, raw histogram delta, normalized details, CSV row 모두 동일 run ID로 연결.
- client ITL과 vendor ITL의 차이를 오류로 단정하지 않고 측정 경계 차이로 설명.

- [ ] **Step 5: request review and open second PR**

PR title:

```text
feat(furiosa): add server-side ITL benchmark evidence
```

PR body에 다음을 명시한다.

- 이 경로는 embedded `AsyncLLMEngine` runtime과 다른 `furiosa-llm serve` 실험이다.
- client ITL은 vLLM benchmark의 streaming 관측치다.
- vendor ITL은 Furiosa `/metrics` histogram delta다.
- 두 값을 같은 열이나 같은 이름으로 합치지 않는다.

---

## Final Acceptance Criteria

- direct async 결과에서 request TTFT와 request mean TPOT의 p50/p85/p90/p95/p99가 sample count와 함께 저장된다.
- raw request trace만으로 해당 direct percentile을 재계산할 수 있다.
- multi-token stream event가 하나라도 있는 요청은 direct per-token ITL sample을 만들지 않는다.
- direct stream ITL coverage가 100%가 아니면 top-level paper metric이 생성되지 않는다.
- server benchmark는 vLLM detailed JSON과 Furiosa Prometheus raw delta를 모두 보존한다.
- client/server request·token counter가 맞지 않는 server run은 valid가 될 수 없다.
- SDK 없는 fake test와 전체 framework regression이 통과한다.
- 실제 RNGD 성공은 각 PR의 CI gate는 아니지만 논문 결과 생성 전 필수 gate다.
