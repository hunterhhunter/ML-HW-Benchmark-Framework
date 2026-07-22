# Rebellions RBLN static runtime·native async·monitor 설계

**상태:** 사용자 승인 및 구현 계획 작성 완료, 구현 전

**승인일:** 2026-07-22

**대상 브랜치:** `feat/rbln-runtime-monitor`

## 1. 결정 요약

이번 브랜치는 Rebellions NPU 연동을 한 번에 모두 구현하지 않는다. 첫 번째 독립
증분으로 다음 세 기능만 추가한다.

1. 사전 컴파일된 `.rbln` 파일을 실행하는 `rbln-static` target
2. 기존 framework bounded queue에 `rebel.AsyncRuntime.async_run()`을 연결하는
   native async backend
3. `rbln-smi -j`를 이용해 선택한 NPU 한 장을 측정하는 `RblnCollector`

비전, BERT 계열 언어 이해, 시계열 모델은 이 static target을 사용한다. Llama 3.2
3B와 Llama 3.1 8B 생성은 static runtime에 억지로 포함하지 않고, 후속
`rbln-vllm` 브랜치에서 인프로세스 Python engine으로 구현한다. 외부 OpenAI 호환
서버 adapter와 framework compiler plugin은 그 뒤의 별도 작업이다.

이 분리는 현재 구조와 일치한다. `TargetSpec`이 runtime, monitor, artifact format을
조합하고, runtime/collector registry가 벤더 SDK를 지연 import하며,
`InferenceEngine`과 `NativeAsyncRuntimeExecutor`가 공통 큐·completion·metric
lifecycle을 계속 소유한다. RBLN adapter는 새 스케줄러나 별도 benchmark runner를
만들지 않는다.

## 2. 기준 코드와 브랜치 의존성

작업 브랜치는 다음 상태에서 시작한다.

- worktree: `/tmp/ml-hw-benchmark-rbln-runtime-monitor`
- branch: `feat/rbln-runtime-monitor`
- current head: `b159aab`
- latest `origin/main`: `ac921f0`, PR #24 generation latency metric 병합본
- base stack: `origin/agent/mobilint-runtime-monitor`의 `4f3454e` 위에 main 병합

PR #25/#26은 GitHub에서 merged 상태지만 2026-07-22 현재 main이 아닌 stacked
branch에 병합되어 있다. 이 브랜치는 PR #25가 도입한 다음 일반 계약을 사용한다.

```python
runtime.native_async_max_batch_size()
runtime.create_native_backend()
NativeAsyncRuntimeExecutor(...)
```

따라서 PR을 열 때까지 PR #25 계열이 main으로 승격되지 않으면 stacked PR로
표시한다. main 승격 뒤에는 최신 main에 rebase하고, Mobilint 전용 구현을 RBLN
의존성으로 취급하지 않는다. PR #23의 외부 Furiosa server 구조는 이번 설계의
선행 조건도, 재사용 대상도 아니다.

코드 변경 전 전체 기준 테스트는 다음 결과다.

- 1,359 passed
- 13 skipped
- 12 failed

실패 12건은 RBLN 코드가 없는 기준 commit에서 재현되었다. 11건은 기존 Furiosa
native async callback timeout, 1건은 sandbox의 Hugging Face DNS/다운로드 실패다.
구현 중 이 항목을 RBLN 회귀로 계산하지 않으며, focused suite와 전체 suite의 결과를
모두 기록한다.

## 3. 공식 API 근거와 환경 전제

설계는 다음 RBLN 공식 문서를 기준으로 한다.

- [RBLN Compiler API 개요](https://docs.rbln.ai/latest/software/api/): `.rbln`을
  `Runtime` 또는 `AsyncRuntime`으로 실행하고 NumPy/PyTorch tensor를 입출력한다.
- [동시실행](https://docs.rbln.ai/latest/ko/software/api/python/tutorial/advanced/concurrent_processing.html):
  `AsyncRuntime.async_run()`은 native `asyncio` coroutine이며 `parallel=1/2`로 입력
  준비 thread 수를 정한다.
- [Python API](https://docs.rbln.ai/v0.9.1/software/api/python/python_api.html):
  `RBLNCompiledModel.inspect()`는 모델을 host memory에 올리지 않고 input/output,
  compiler version, target NPU, tensor parallel size, UUID를 제공한다. 같은 compiled
  model instance에서는 sync/async runtime 생성이 상호 배타적이다.
- [장치 모니터링](https://docs.rbln.ai/latest/ko/software/system_management/management_tools/rbln_smi.html):
  `rbln-smi -j`는 JSON을, `-d`는 장치 선택을 제공하며 temperature, card power,
  utilization, memory, context와 P-state를 보고한다. `rbln-stat`은 사용하지 않는다.
- [vLLM RBLN](https://docs.rbln.ai/latest/software/model_serving/vllm_support/index.html):
  LLM은 `vllm-rbln` plugin과 in-process engine을 사용하는 후속 단계로 분리한다.

사용자가 제공한 첫 검증 서버는 다음과 같다.

| 항목 | 값 |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.10.12 |
| package | `rebel-compiler==0.11.0` |
| KMD / firmware | 3.2.2 / 3.2.2 |
| NPU | device 0, `RBLN-CA22`, `/dev/rbln0` |
| memory | 16,877,879,296 bytes, 약 15.7 GiB |
| PCI | `0000:ab:00.0`, NUMA node 1, 32.0 GT/s x8 |

개발 host에는 RBLN NPU와 SDK가 없어도 된다. 모든 production module은 `rebel`을
module import 시점에 요구하지 않으며 fake SDK와 fake `rbln-smi` 응답으로 테스트할
수 있어야 한다. 실제 NPU smoke/E2E만 사용자가 접근하는 서버에서 실행한다.

공식 최신 vLLM 문서의 설치 예시는 `rebel-compiler==0.11.0.post1`이지만, 이번 static
adapter가 임의로 package를 upgrade하거나 strict version equality를 강제하지 않는다.
실행 package version, artifact compiler version, KMD와 firmware를 결과에 기록하고,
실제 호환 실패가 있을 때 공식 support matrix에 맞춰 서버 환경을 조정한다.

## 4. 범위 분해

### 4.1 이번 브랜치의 포함 범위

- target id `rbln-static`
- runtime registry name `rbln`, aliases `rebel`, `rbln-static`
- `.rbln` artifact의 read-only inspect와 실행
- sync E2E `rebel.Runtime`
- async queue `rebel.AsyncRuntime.async_run()`
- ResNet50, YOLOv5m, BERT classification/QA, PatchTST의 기존 loader/evaluator/decoder
  재사용
- NPU 0에 대한 runtime/monitor selector 일치 보장
- `rbln-smi` JSON 기반 utilization, memory, temperature, power, energy, device metadata
- CLI, registry, docs, result metadata, SDK-free unit/integration test
- 단일 NPU, fixed input shape, request batch size 1의 안전한 초기 계약

### 4.2 후속 브랜치로 분리하는 범위

| 후속 작업 | 이유 |
|---|---|
| `rbln-vllm` in-process Llama engine | static tensor inference와 token streaming/continuous batching lifecycle이 다름 |
| vLLM OpenAI-compatible server adapter | 내부 Python engine 안정화 뒤 transport를 추가해야 metric 의미가 섞이지 않음 |
| framework RBLN compiler plugin | 사용자가 먼저 model zoo/manual `.rbln` 배포를 사용하며 compile latency를 benchmark와 분리함 |
| bucketing/dynamic shape | artifact inspect, batch 조립과 profile 선택 계약을 별도 설계해야 함 |
| multi-NPU/tensor parallel static runtime | 현재 장치 한 장과 `tensor_parallel_size == 1`만 검증함 |
| profiler/timer report 통합 | `RBLN_RUNTIME_TIMER=1`의 측정 perturbation과 device/host time 정의를 먼저 검증해야 함 |
| RSMD/gRPC 및 Prometheus exporter | 로컬 benchmark에 필요한 최소 monitor는 `rbln-smi` JSON으로 충족함 |

### 4.3 명시적 비범위

- raw Hugging Face/ONNX 모델을 framework 실행 중 자동 컴파일
- model zoo 다운로드 구현, credential 관리, artifact 배포
- Llama를 static logits 반복 호출로 생성하는 임시 경로
- framework queue를 vendor queue로 교체
- 기존 PR #24 generation observation schema 변경
- 기존 Furiosa baseline failure 수정
- 분산 실행, 여러 process가 한 NPU를 공유하는 admission controller

## 5. 목표 구조

```text
CLI --target rbln-static --artifact model.rbln
             │
             ▼
      TargetSpec(rbln-static)
       ├─ Runtime registry ──> RblnRuntime
       ├─ Monitor registry ──> RblnCollector + SystemCollector
       └─ artifact_format ───> rbln

E2E mode
InferenceEngine -> BlockingRuntimeExecutor -> RblnRuntime.run()
                                             └─ lazy rebel.Runtime

async_queue mode
producer -> framework bounded request queue
         -> AsyncInferenceEngine workers
         -> NativeAsyncRuntimeExecutor
         -> RblnNativeBackend owner event-loop thread
         -> rebel.AsyncRuntime.async_run()
         -> exactly-once NativeAsyncOutcome callback
         -> framework CompletionCoordinator
         -> evaluator/decoder/metrics/result store

measurement interval only
HWMonitor polling thread
  ├─ RblnCollector -> rbln-smi -b -j -d 0 (internally throttled)
  └─ SystemCollector
```

Framework Queue가 request identity, admission, bounded backlog, backpressure, timeout,
exact-once terminal과 metric의 source of truth다. `AsyncRuntime` 내부 queue는 하드웨어
실행을 위한 vendor detail일 뿐이다. adapter는 framework queue와 vendor queue 사이에
세 번째 Python scheduling queue를 만들지 않는다.

## 6. target과 CLI 계약

### 6.1 TargetSpec

`framework/src/core/targets.py`에 다음 의미의 target을 등록한다.

```python
TargetSpec(
    target_id="rbln-static",
    label="Rebellions ATOM / RBLN Runtime",
    runtime_name="rbln",
    device="0",
    monitor_names=("rbln", "system"),
    artifact_format="rbln",
    accelerator_vendor="Rebellions",
    accelerator_name="RBLN NPU",
    device_selector="0",
    capabilities=(
        "rbln", "sync", "native_async", "latency", "throughput",
        "monitor", "npu", "local", "static_shape",
    ),
    runtime_options={
        "device_id": 0,
        "async_parallel": 1,
        "runtime_timeout_sec": 60,
        "shutdown_timeout_sec": 300.0,
    },
    monitor_options={
        "rbln": {
            "device_id": 0,
            "sample_interval_sec": 1.0,
            "command_timeout_sec": 2.0,
        },
    },
)
```

`device_id`는 runtime과 monitor가 반드시 같아야 하는 locked selector다. 기존
Mobilint-only merge helper를 작은 일반화된 locked-target helper로 바꾸고 다음 mapping을
사용한다.

```text
mobilint-aries   -> collector mobilint, keys device_id/expected_family
mobilint-regulus -> collector mobilint, keys device_id/expected_family
rbln-static      -> collector rbln,    key  device_id
```

이 refactor는 기존 Mobilint case-insensitive family 비교와 error behavior를 그대로
보존한다. `--runtime-option device_id=1`처럼 monitor는 0인데 runtime만 바꾸는 입력은
명확한 CLI 오류로 거부한다. 나중에 device 1을 지원할 때는 target을 추가하거나 target
device selector override를 runtime/monitor 양쪽에 원자적으로 적용하는 별도 설계를 한다.

### 6.2 artifact 선택

`rbln-static`에는 compiler가 없다. 따라서 다음이 필수다.

```text
--target rbln-static --artifact /absolute/or/relative/model.rbln
```

경로가 없거나 파일이 아니거나 suffix가 `.rbln`이 아니면 runtime 생성 전에 실패한다.
`run_auto_prepare()`는 이 artifact를 Hugging Face나 ONNX 경로로 대체하지 않고 model
prepare script도 실행하지 않는다. dataset prepare는 기존 정책을 유지할 수 있다.

`--backend rbln`도 parser choice와 legacy resolution에 추가하지만, 문서의 기본 사용법은
runtime/monitor/device를 한 번에 고정하는 `--target rbln-static`이다.

### 6.3 모델 task 제한

target capability에 `generation`이 없으므로 `Task.NLP_GENERATION`은 CLI 조립 단계에서
실패시킨다. 오류는 Llama에 `rbln-vllm` 후속 target이 필요하다고 안내한다. static
adapter의 `supports_generate()`는 `False`를 유지한다.

이번 브랜치의 `--batch-size`는 1만 보장한다. async mode는
`native_async_max_batch_size() == 1`로 기존 generic validation이 즉시 거부한다. E2E
mode도 artifact fixed shape와 입력 shape 검증으로 N=1이 아닌 입력을 명시적으로
거부한다.

## 7. RblnRuntime 설계

### 7.1 파일과 공개 책임

새 `framework/src/runtimes/rbln_rt.py`는 다음 두 class만 공개한다.

- `RblnRuntime`: framework `Runtime` 구현, artifact·mode·SDK resource lifecycle 소유
- `RblnNativeBackend`: RBLN asyncio API를 framework callback API로 변환

module import 시 `rebel`을 import하지 않는다. `_load_rebel()`에서 지연 import하고,
package가 없을 때 설치 출처와 필요한 package 이름을 포함한 `ImportError`를 낸다.

### 7.2 상태 머신

```text
UNLOADED
   │ load(.rbln): inspect + validate only
   ▼
LOADED_NO_ENGINE
   ├─ first run/warmup ──> SYNC_ACTIVE
   │                       └─ rebel.Runtime exactly once
   └─ create_native_backend ─> ASYNC_ACTIVE
                               └─ rebel.AsyncRuntime exactly once

SYNC_ACTIVE  -- unload --> UNLOADED
ASYNC_ACTIVE -- drain + unload --> UNLOADED
```

한 번 `SYNC_ACTIVE`가 되면 native backend 생성은 실패한다. 한 번 `ASYNC_ACTIVE`가
되면 `run()`은 실패한다. mode를 바꾸려면 `unload()` 후 새 benchmark 조립을 해야 한다.
이로써 공식 API의 sync/async 상호 배타성을 지키고, 같은 `.rbln`에 sync와 async device
allocation이 동시에 생기는 것을 방지한다.

`load()`가 runtime object를 즉시 만들지 않는 이유는 현재 main flow가
`runtime.load()` 뒤에 `_build_async_runtime_executor()`를 호출하기 때문이다. executor
factory가 호출되기 전까지 mode 선택을 미루면 main에 RBLN 전용 async flag를 추가하지
않아도 된다.

### 7.3 load와 artifact 검증

`load(compiled_model)`은 다음 순서로 동작한다.

1. cleanup pending 또는 이미 loaded 상태인지 검사한다.
2. `CompiledModel.backend_name`이 `rbln/rebel/rbln-static` 중 하나이며 suffix가 `.rbln`인지
   검사한다.
3. `rebel.npu_is_available(device_id)`로 선택 장치가 보이는지 검사한다.
4. `rebel.get_npu_name(device_id)`로 실제 NPU 이름을 읽는다.
5. `rebel.RBLNCompiledModel.inspect(path)`로 metadata를 읽는다.
6. metadata의 target `npu`가 실제 NPU 이름과 다르면 runtime allocation 전에 실패한다.
7. `tensor_parallel_size`가 1이 아니면 single-device target에서 실패한다.
8. input count/name/shape/dtype와 Model_Spec 계약을 검증한다.
9. 결과에 필요한 작은 metadata만 복사하고 raw `subgraph` 전체는 보존하지 않는다.

보존하는 artifact metadata는 `compiler_version`, `npu`, `tensor_parallel_size`, `uuid`,
`alloc_per_node`, input/output descriptor다. metadata key가 누락되면 그 key의 사전 검증은
생략하되 runtime constructor의 실패를 숨기지 않는다. metadata 타입이 명백히 잘못된
경우에는 `RblnArtifactMetadataError` 성격의 `ValueError`로 실패한다.

단일 input 모델은 artifact와 profile의 이름이 달라도 positional input 하나로 실행할
수 있다. 다중 input 모델은 artifact input name을 우선해 dict를 정렬하며, 모든 이름이
profile/loader input에 존재해야 한다. 이름이 일치하지 않으면 순서를 추측하지 않고
expected/provided name을 포함한 오류를 낸다.

shape와 dtype은 암묵적으로 cast/pad하지 않는다. NumPy array를 contiguous하게 만드는
복사는 허용하지만 dtype과 rank/shape는 artifact descriptor와 정확히 일치해야 한다.
이 정책은 잘못된 BERT mask 순서나 PatchTST bool mask 변환이 성능 결과로 조용히
유입되는 것을 막는다.

### 7.4 sync 실행

첫 `run()` 또는 sync warmup에서 다음 runtime을 한 번 만든다.

```python
rebel.Runtime(
    str(artifact_path),
    device=device_id,
    tensor_type="np",
    timeout=runtime_timeout_sec,
)
```

입력은 inspect metadata 순서의 positional NumPy arguments로 전달한다. framework가
이미 host wall time을 측정하므로 `RblnRuntime.run()`은 별도 timer report를 활성화하지
않는다. output은 9절 규칙에 따라 framework output dict로 정규화한다.

`warmup(inputs, num_runs)`는 아직 mode가 없으면 sync mode를 선택하고 같은 `run()`을
반복한다. async backend가 이미 만들어졌으면 sync runtime을 만들지 않고 backend의
blocking warmup helper를 사용해 같은 owner loop와 같은 `AsyncRuntime`에서
`async_run()`을 실행한다.

### 7.5 unload와 재시도 가능 cleanup

sync mode는 runtime reference를 해제하고 상태를 초기화한다. 공개 close API가 없는
SDK object에 private method를 호출하지 않는다.

async mode는 먼저 backend `shutdown(timeout=shutdown_timeout_sec)`을 호출한다. accepted
job, callback, owner thread가 모두 끝났다는 증명이 없으면 `AsyncRuntime` reference와
compiled-model state를 유지한 채 오류를 내고 cleanup pending으로 남긴다. 호출자는
`unload()`를 재시도할 수 있다. 안전 증명 없이 device object를 해제하지 않는다.

backend 초기화 도중 실패한 경우에도 owner thread나 생성된 SDK object의 cleanup
소유권을 잃지 않는다. 완전 rollback에 성공하면 `LOADED_NO_ENGINE`으로 돌아가 재시도를
허용하고, rollback이 불완전하면 cleanup pending으로 전환한다.

### 7.6 device spec

`get_device_spec()`은 작은 primitive-only dict를 반환한다.

```text
backend=rbln
device=0
device_id=0
accelerator_vendor=Rebellions
accelerator_name=RBLN-CA22              # 실제 조회값
execution_mode=loaded|sync|native_async
sdk_version=0.11.0                       # importlib.metadata
artifact_compiler_version=<inspect value>
artifact_npu=<inspect value>
artifact_uuid=<inspect value>
tensor_parallel_size=1
async_parallel=1
```

async details의 `_safe_runtime_diagnostics()`에 `rbln` backend와 위 필드의 명시적
type/length allowlist를 추가한다. 임의 object의 `str()`을 호출하지 않는다. monitor를
끄더라도 software/artifact provenance가 남아야 한다.

## 8. Native async와 queue 설계

### 8.1 동시성 parameter의 의미

세 parameter는 서로 다른 계층이며 합치지 않는다.

| parameter | 소유자 | 의미 |
|---|---|---|
| `queue_capacity` | framework | accepted됐지만 아직 terminal이 아닌 backlog 상한 |
| `worker_count` | framework/executor | 동시에 SDK에 제출할 수 있는 request 수와 `max_inflight` |
| `async_parallel` | RBLN SDK | 한 AsyncRuntime의 input buffer 준비 thread 수, 1 또는 2 |

초기 기본값은 `batch_size=1`, `async_parallel=1`이다. 공식 문서가 `parallel=2`의 효과와
안정성이 모델별이라고 경고하므로 자동 활성화하지 않는다. 사용자가 명시적으로 2를
선택한 benchmark는 결과 metadata에 반드시 남긴다.

구현 시 async CLI 조립은 유효 `worker_count`를 runtime option
`max_async_inflight`로도 전달한다. `RblnRuntime.max_concurrent_workers()`는 native async
mode에서 이 값을 반환하여 기존 `AsyncInferenceEngine` capability 검사를 통과시킨다.
이 값은 framework/executor의 inflight 허용량일 뿐이며 RBLN SDK constructor의
`parallel=async_parallel` 값은 바꾸지 않는다.

### 8.2 owner event loop

`RblnNativeBackend`는 daemon owner thread 하나와 그 thread 전용 asyncio event loop를
만든다. `rebel.AsyncRuntime`도 owner thread 안에서 생성한다.

```python
rebel.AsyncRuntime(
    str(artifact_path),
    device=device_id,
    tensor_type="np",
    parallel=async_parallel,
    timeout=runtime_timeout_sec,
)
```

초기화 caller는 bounded startup event를 기다린다. startup timeout, SDK constructor
오류, loop 조기 종료를 각각 구분해 보고한다. constructor exception은 원문 전체를
result artifact에 복사하지 않고 안전한 type과 고정된 adapter message만 전달한다.

### 8.3 submit_async 계약

framework executor가 요구하는 signature는 다음과 같다.

```python
submit_async(inputs: dict[str, np.ndarray], callback) -> str
```

동작 순서는 다음과 같다.

1. closing 여부와 input 계약을 동기적으로 검사한다.
2. monotonic job number로 `rbln-<n>` id를 만든다.
3. input reference와 concurrent future를 job registry에 등록한다.
4. `asyncio.run_coroutine_threadsafe()`로 owner loop에 coroutine을 게시한다.
5. 게시 성공 직후 job id를 반환한다.
6. coroutine은 `await async_runtime.async_run(*ordered_inputs)`를 실행한다.
7. 성공이면 output 정규화와 host elapsed time을 담은 `NativeAsyncOutcome`을 callback에
   전달한다.
8. SDK/output 오류면 bounded error type과 고정 message를 담은 failure outcome을
   callback에 전달한다.
9. callback이 예외를 내더라도 job registry와 input reference를 finally에서 해제한다.

accepted job 하나당 callback은 정확히 한 번 시도한다. callback 호출 자체가 실패해도
두 번째 callback을 보내지 않는다. `NativeAsyncRuntimeExecutor`가 duplicate/late
callback, logical timeout, acknowledge와 permit 반환을 계속 관리한다.

adapter는 별도 waiter thread를 job마다 만들지 않는다. owner event loop의 coroutine이
SDK completion을 기다린다. submission path에서는 blocking slot wait를 하지 않는다.
inflight 상한은 executor의 bounded semaphore가 이미 보장한다.

### 8.4 timeout, cancellation, shutdown

공개 문서에서 실행 중 `async_run()`의 물리 cancel 완료를 증명하는 계약을 확인할 수
없으므로 phase 1은 physical cancellation을 주장하지 않는다.

- request timeout은 framework의 logical terminal이다.
- late vendor completion은 callback을 통해 physical completion을 증명할 때까지
  dispatch와 input ownership을 유지한다.
- logical timeout request를 executor가 acknowledge해도 late callback 전에는 permit과
  runtime unload safety가 회복되지 않는다.
- shutdown은 새 submit을 막고 accepted coroutine이 자연 완료하기를 기다린다.
- deadline 안에 job이 남으면 `False`를 반환하고 loop/AsyncRuntime을 유지한다.
- 모든 job이 끝난 뒤에만 owner loop에서 AsyncRuntime reference를 해제하고 loop를
  stop한 다음 thread를 join한다.

`NativeAsyncRuntimeExecutor.shutdown()`과 `RblnNativeBackend.shutdown()`은 모두
idempotent해야 한다. main의 기존 async failure path가 outstanding zero를 증명하지
못하면 runtime unload를 건너뛰는 계약을 유지한다.

### 8.5 async warmup

executor factory는 `engine.run_async()`보다 먼저 `create_native_backend()`을 호출한다.
따라서 async warmup 시점에는 backend가 존재한다. backend는 owner loop에 동일한
`async_run()` coroutine을 게시하고 caller가 timeout 안에서 결과를 기다리는
`run_warmup_blocking()`을 제공한다. warmup request는 측정 queue, request trace,
evaluator와 async metric에 넣지 않는다.

## 9. input/output 계약

### 9.1 input

runtime과 native backend는 공통 private helper를 사용한다.

- input은 `dict[str, np.ndarray]`여야 한다.
- extra/missing input은 거부한다.
- artifact inspect 순서대로 정렬한다.
- single input만 이름 fallback을 허용한다.
- scalar, rank, fixed shape, dtype를 검증한다.
- non-contiguous array는 `np.ascontiguousarray()`로 변환한다.
- dtype cast, padding, truncation, layout transpose는 runtime에서 하지 않는다.

전처리와 layout은 기존 loader가 소유한다. RBLN-specific preprocessing이 실제 model
zoo artifact에 필요하면 generic runtime에 숨기지 않고 model profile/loader option으로
별도 추가한다.

### 9.2 output

공식 Runtime 반환형인 single ndarray 또는 ndarray list/tuple을 다음 규칙으로
정규화한다.

- single ndarray: spec output이 하나일 때 그 이름에 mapping
- list/tuple: spec output 순서와 count가 정확히 같아야 함
- dict: key가 spec output과 정확히 일치할 때만 defensive compatibility로 허용
- 모든 값은 NumPy array인지 검사
- output count/key mismatch는 즉시 오류

output shape는 evaluator가 사용하기 전에 artifact inspect descriptor와 spec을 비교한다.
동적 dimension/bucket을 의미하는 metadata는 이번 target에서 거부한다. output array를
불필요하게 복사하지 않지만 callback/ack가 끝나기 전 input/output ownership을 해제하지
않는다.

### 9.3 model별 초기 계약

| 모델 | static target 계약 | 주의점 |
|---|---|---|
| ResNet50 | image classification, NCHW `(1,3,224,224)` | model zoo artifact input name은 단일-input fallback 가능 |
| YOLOv5m | object detection, `(1,3,640,640)` -> raw `(1,25200,85)` | NMS 포함/다른 output layout artifact는 기존 decoder와 호환되지 않으므로 거부 |
| BERT base SST-2 | `input_ids`, `attention_mask`, length 128 | artifact가 `token_type_ids`를 추가 요구하면 별도 profile 변경 필요 |
| BERT base SQuAD | 두 input, length 384, start/end logits | input 순서를 이름으로 검증 |
| PatchTST FM R1 | `past_values` float32, `past_observed_mask` bool | bool mask를 float로 암묵 변환하지 않음 |
| Llama 3.2 3B / 3.1 8B | 이번 target에서 거부 | 후속 `rbln-vllm` target 사용 |

이 표의 shape와 실제 `.rbln` inspect 결과가 다르면 artifact를 억지로 실행하지 않는다.
model zoo artifact에 맞춘 새로운 profile 또는 decoder가 필요하다는 명시적 오류로
처리한다.

## 10. RblnCollector 설계

### 10.1 실행 방식

새 `framework/src/monitors/rbln_collector.py`는 shell 없이 다음 argv를 실행한다.

```text
rbln-smi -b -j -d 0
```

collector 자체에는 외부 Python dependency가 없으므로 `is_available()`은 `True`를
반환한다. `start()`가 `shutil.which("rbln-smi")`로 executable을 검증한다. 이렇게 해야
`create_hw_monitor()`가 vendor collector를 조용히 건너뛰고 system metric만 기록하는
일이 없다. 실제 snapshot은 `subprocess.run()`에 `capture_output=True`, `text=True`,
`check=True`, `timeout=2.0`을 사용한다. command, device id, timeout은 생성자에서
검증한다. shell expansion과 사용자 문자열 삽입은 허용하지 않는다.

HWMonitor 기본 poll이 0.2초여도 CLI process를 5 Hz로 만들지 않는다. collector 내부
monotonic clock으로 성공/실패 시도 간 최소 1.0초를 강제하고, 이보다 빠른 `collect()`는
빈 dict를 반환한다. start/stop boundary sample은 throttle과 별개로 한 번씩 강제한다.

### 10.2 JSON parsing

선택 장치는 `devices[*].npu == device_id`로 찾는다. device가 없거나 중복되면 start는
실패한다. 다음 변환을 수행한다.

| JSON | framework key | 변환 |
|---|---|---|
| `util` | `hw_accel_util` | float percent |
| `memory.used` | `hw_accel_mem_used_mb` | bytes / 1024² |
| `temperature` | `hw_accel_temp_c` | `38C` 또는 numeric 처리 |
| `card_power` | `hw_accel_power_w` | `18810987uW` -> `18.810987W`; numeric/unit 변형 허용 |
| matching context `memalloc` | `hw_accel_mem_proc_mb` | 현재 Python PID context만 합산, 식별 불가하면 key 생략 |

정적/last-known metadata는 `get_static_info()`에 둔다.

```text
hw_accel_vendor=Rebellions
hw_accel_name=RBLN-CA22
hw_accel_device_id=0
hw_accel_device_node=rbln0
hw_accel_uuid=<uuid>
hw_accel_serial_id=<sid>
hw_accel_status=normal
hw_accel_pstate=P14/P2/...
hw_accel_monitor_source=rbln-smi-json
hw_accel_kmd_version=3.2.2
hw_accel_firmware_version=3.2.2
hw_accel_pci_bus_id=0000:ab:00.0
hw_accel_pci_numa_node=1
hw_accel_pci_link_speed=32.0GT/s
hw_accel_pci_link_width=8
hw_accel_mem_total_mb=16096.0
```

JSON의 `KMD_version`과 version별 `driver_version` spelling을 모두 허용한다. metric
number/string, unit suffix와 missing field를 방어적으로 처리하되, malformed value를
0으로 만들지 않는다.

### 10.3 energy와 coverage

성공한 power sample의 monotonic timestamp 사이를 사다리꼴 적분한다.

```text
energy_j += (previous_w + current_w) / 2 * elapsed_seconds
```

요약에는 다음을 기록한다.

- `hw_accel_energy_j`: 유효 power sample이 2개 이상일 때만
- `hw_accel_power_samples`: 성공 횟수
- `hw_accel_monitor_attempts`: 실제 subprocess 시도 횟수
- `hw_accel_monitor_successes`: schema까지 유효한 snapshot 횟수
- `hw_accel_monitor_coverage`: successes / attempts

power는 card 전체 값이며 현재 process만의 에너지가 아니다. idle power도 포함한다.
warmup은 HWMonitor 시작 전이므로 energy interval에서 제외된다.

### 10.4 오류 정책

`start()`의 missing executable, command failure, invalid JSON, 선택 device 부재, device
status 비정상은 monitor를 요청한 실행의 setup 오류다. 측정 중 일시적 command timeout,
non-zero exit, malformed sample은 benchmark inference를 중단하지 않고 그 sample을
생략한다. 마지막 오류는 type이 sanitize된 고정 길이
`hw_accel_monitor_note`로 남긴다.

`stop()`은 마지막 boundary sample을 시도한다. sampling 오류가 있어도 persistent
resource는 없지만, 오류를 summary diagnostic에 보존한다. collector를 다시 start하면
energy, counters, static cache와 error state를 초기화한다.

## 11. 정상 데이터 흐름

### 11.1 E2E

```text
CLI validate artifact/profile
-> RblnRuntime.load: SDK/device/artifact inspect
-> warmup: lazy rebel.Runtime + fixed-shape run
-> monitor.start: initial rbln-smi snapshot
-> InferenceEngine.run_e2e
-> BlockingRuntimeExecutor host latency
-> existing decoder/evaluator
-> monitor.stop + summary
-> CSV save
-> runtime.unload
```

### 11.2 async offline/server-like

```text
CLI validate config and reserve result artifacts
-> RblnRuntime.load: inspect only
-> create_native_backend: owner loop + rebel.AsyncRuntime
-> InferenceEngine.run_async
   -> async warmup through same AsyncRuntime
   -> monitor.start
   -> producer issues request
   -> bounded queue admission/backpressure
   -> worker -> NativeAsyncRuntimeExecutor permit/token
   -> RblnNativeBackend coroutine submission
   -> async_run completion callback
   -> executor RuntimeExecution
   -> CompletionCoordinator exact-once evaluator
   -> ACK, dispatch retirement, permit release
-> flush and shutdown prove outstanding=0
-> monitor.stop + summary
-> JSON details/optional JSONL trace/CSV
-> backend drain + owner loop stop + runtime.unload
```

PR #24의 `GenerationObservation`, TTFT, TPOT, ITL은 static model에서 생성하지 않는다.
queue delay, end-to-end latency, runtime latency, throughput, outstanding, timeout과 validity는
현재 async metric schema를 그대로 사용한다.

## 12. failure와 cleanup 계약

| failure 지점 | terminal/cleanup 처리 |
|---|---|
| package import 실패 | load 실패, device allocation 없음, 설치 안내 |
| unavailable/wrong NPU | load 실패, artifact 실행 안 함 |
| artifact inspect/profile mismatch | load 실패, expected/actual descriptor 진단 |
| sync Runtime constructor 실패 | mode 미완료 상태에서 cleanup 시도, 원인 type 보존 |
| async owner loop startup 실패 | thread/loop rollback; 불완전하면 cleanup pending |
| submit 전 validation 실패 | SDK accepted job 없음, executor submit failure |
| `run_coroutine_threadsafe` 실패 | registry rollback, accepted job 없음으로 보고 |
| `async_run`/output normalize 실패 | failure `NativeAsyncOutcome` callback exactly once |
| logical request timeout | terminal timeout, vendor completion까지 ownership 유지 |
| callback consumer 예외 | 재호출하지 않고 adapter job cleanup, executor 진단 유지 |
| monitor sample 실패 | sample 생략, coverage/note 기록, inference 계속 |
| async drain timeout | invalid run, runtime unload unsafe, SDK object 해제 금지 |
| result persistence 실패 | 기존 reservation/failure artifact 정책 유지 |

오류 message에 artifact binary, tensor data, arbitrary exception `repr`, 전체 subprocess
stderr를 넣지 않는다. stderr는 bounded whitespace-normalized text로만 debug log에
사용하며 result에는 operation과 safe exception type을 기록한다.

## 13. 예상 변경 파일

Production:

- `framework/src/runtimes/rbln_rt.py` 신규
- `framework/src/runtimes/__init__.py` lazy registry/export
- `framework/src/monitors/rbln_collector.py` 신규
- `framework/src/monitors/__init__.py` lazy registry
- `framework/src/core/targets.py` `rbln-static` target
- `framework/src/main.py` backend/target resolution, artifact/task validation, selector lock,
  safe runtime diagnostics

Tests:

- `framework/tests/test_rbln_runtime.py` 신규
- `framework/tests/test_rbln_native_backend.py` 신규
- `framework/tests/test_rbln_collector.py` 신규
- `framework/tests/test_plugin_registry.py` target/runtime/collector graph
- `framework/tests/test_async_cli.py` native executor와 result metadata
- `framework/tests/test_main_paths.py` CLI help/artifact validation
- 기존 Mobilint selector tests: generalization 회귀 검증

Docs:

- `docs/rbln-setup.md` 서버 설치, artifact, 명령, 모니터링, 문제 해결
- `framework/src/runtimes/README.md` adapter lifecycle와 SDK optional dependency
- `framework/README.md` target/model 표와 CLI 예제
- `framework/CHANGELOG.md`

정확한 test 함수와 production edit 순서는 다음 단계의 implementation plan에서 작은
TDD task로 나눈다.

## 14. TDD와 검증 설계

### 14.1 SDK-free runtime tests

fake `rebel` module로 다음 RED 계약부터 작성한다.

1. module import는 `rebel` package가 없어도 성공한다.
2. `.rbln`/backend/device availability/target NPU/tensor parallel validation이 각각
   의도한 오류를 낸다.
3. inspect input name/shape/dtype mismatch를 runtime constructor 전에 거부한다.
4. load 직후 sync/async object가 아직 없다.
5. 첫 sync warmup/run이 `rebel.Runtime`을 정확히 한 번 만든다.
6. sync 뒤 async 또는 async 뒤 sync 전환을 거부한다.
7. multi-input ordering, contiguous conversion, no dtype cast를 검증한다.
8. single/list/dict output normalization과 count mismatch를 검증한다.
9. unload success, constructor rollback, cleanup retry ownership을 검증한다.
10. device spec은 primitive allowlisted provenance만 반환한다.

### 14.2 native backend race tests

fake asyncio AsyncRuntime으로 다음을 결정적으로 제어한다.

1. AsyncRuntime constructor와 async_run이 owner thread에서 실행된다.
2. submit은 coroutine 완료 전에 job id를 반환한다.
3. success/error/output-normalization failure마다 callback이 정확히 한 번 온다.
4. out-of-order completion도 job id와 input ownership을 섞지 않는다.
5. callback 자체가 raise해도 registry가 drain된다.
6. submit과 shutdown race에서 shutdown 이후 새 job은 accepted되지 않는다.
7. logical timeout 뒤 late completion 전에는 executor inflight와 unload safety가 남는다.
8. late completion + ACK 뒤 permit과 input이 해제된다.
9. shutdown timeout은 loop를 강제 종료하거나 runtime을 해제하지 않는다.
10. drain된 shutdown은 loop stop/thread join과 반복 호출에 안전하다.
11. async warmup은 measurement callback/metric을 만들지 않고 같은 runtime을 사용한다.

thread test는 1초 sleep에 의존하지 않고 Event/barrier/fake coroutine gate를 사용한다.

### 14.3 collector tests

subprocess runner와 clock을 주입해 다음을 검증한다.

1. 사용자가 제공한 `rbln-smi -j` JSON sample을 정확히 파싱한다.
2. uW/W, C, byte/MiB, numeric/string 변형을 처리한다.
3. device 0만 선택하고 다른 device/context를 섞지 않는다.
4. current PID context memory만 process metric에 포함한다.
5. 1초 throttle이 실제 subprocess 호출 수를 제한한다.
6. start/stop boundary와 사다리꼴 energy 값을 fake clock으로 검증한다.
7. missing field는 key 생략이며 0으로 채우지 않는다.
8. missing CLI/start failure와 measurement 중 transient failure 정책을 구분한다.
9. coverage와 bounded diagnostic을 검증한다.
10. 재시작 시 state가 완전히 초기화된다.

### 14.4 registry/CLI/integration tests

- registry graph strict validation에서 `rbln-static`이 runtime/monitor를 찾는다.
- SDK가 없어도 `list_targets/list_runtimes/list_collectors`가 동작한다.
- target이 backend/device/monitor/artifact format을 주입한다.
- runtime과 monitor device selector 불일치를 거부하고 Mobilint 기존 behavior를 보존한다.
- `.rbln` artifact 없음/wrong suffix/directory를 조립 전에 거부한다.
- RBLN generation task를 거부한다.
- async batch size 2를 generic native-async validation이 거부한다.
- async batch size 1은 `NativeAsyncRuntimeExecutor`를 만들고 max inflight가
  `min(worker_count, queue_capacity)`다.
- result details에 runtime provenance, queue metric, monitor summary가 직렬화된다.
- fake static model로 E2E와 async offline/server-like가 decoder/evaluator까지 통과한다.

### 14.5 개발 host 검증 command

implementation plan의 각 단계에서 focused tests를 실행하고 마지막에 다음을 실행한다.

```bash
cd /tmp/ml-hw-benchmark-rbln-runtime-monitor/framework
python -m pytest -q \
  tests/test_rbln_runtime.py \
  tests/test_rbln_native_backend.py \
  tests/test_rbln_collector.py \
  tests/test_plugin_registry.py \
  tests/test_async_cli.py \
  tests/test_main_paths.py

python -m pytest -q
git diff --check
python -m compileall -q src tests
```

전체 suite는 기준 12 failure와 새 failure를 구분해 보고한다. RBLN focused suite에는
known-failure 예외를 넣지 않는다.

## 15. 실제 RBLN-CA22 서버 승인 절차

실제 NPU 검증은 코드가 SDK-free suite를 통과한 뒤 수행한다. framework가 package 설치,
driver 변경이나 모델 다운로드를 자동 실행하지 않는다.

### 15.1 preflight

```bash
python3 --version
python3 -m pip show rebel-compiler
rbln-smi -j
python3 - <<'PY'
import rebel
print("available", rebel.npu_is_available(0))
print("name", rebel.get_npu_name(0))
print("count", rebel.device_count())
PY
```

기대값은 device 0 available, name `RBLN-CA22`, count 1 이상이다. benchmark 전후
`rbln-smi -j`에 유실된 context가 없는지도 확인한다.

### 15.2 artifact preflight

각 model zoo/manual artifact마다 다음을 먼저 저장한다.

```bash
python3 - <<'PY'
import json
from rebel import RBLNCompiledModel

path = "/path/to/model.rbln"
meta = RBLNCompiledModel.inspect(path)
selected = {
    key: meta.get(key)
    for key in (
        "compiler_version", "inputs", "outputs", "npu",
        "tensor_parallel_size", "uuid", "alloc_per_node",
    )
}
print(json.dumps(selected, indent=2, default=str))
PY
```

`npu=RBLN-CA22`, `tensor_parallel_size=1`, profile과 같은 input/output 계약이어야 한다.
raw Hugging Face directory나 ONNX file은 이 target의 artifact가 아니다.

### 15.3 model별 E2E smoke

framework root에서 model마다 작은 `--max-steps`로 먼저 실행한다.

```bash
python3 -m src.main \
  --model resnet50 \
  --target rbln-static \
  --artifact /path/to/resnet50.rbln \
  --dataset datasets/imagenet_1k \
  --inference-mode e2e \
  --batch-size 1 \
  --warmup 2 \
  --max-steps 10 \
  --monitor
```

같은 형태로 `yolov5m`, `bert-base-uncased`, `bert-base-uncased-squad-v1`,
`patchtst-fm-r1`을 실행한다. 기존 profile의 dataset과 tokenizer 경로를 사용한다.
prediction shape, evaluator count, monitor sample/coverage와 exit code 0을 확인한 뒤 전체
dataset run으로 늘린다.

### 15.4 async offline smoke와 sweep

```bash
python3 -m src.main \
  --model resnet50 \
  --target rbln-static \
  --artifact /path/to/resnet50.rbln \
  --dataset datasets/imagenet_1k \
  --inference-mode async_queue \
  --scenario offline \
  --batch-size 1 \
  --worker-count 4 \
  --queue-capacity 64 \
  --min-samples 100 \
  --warmup 2 \
  --flush-timeout-sec 300 \
  --save-request-trace \
  --monitor
```

한 번에 여러 parameter를 바꾸지 않는다.

1. `worker_count=1,2,4,8` sweep, `async_parallel=1` 고정
2. 최고 안정 worker count에서 `async_parallel=2` 별도 비교
3. queue capacity 16/64/256 비교
4. 각 설정을 최소 3회 반복

유효 run은 accepted/completed/evaluated count 일치, outstanding 0, duplicate callback 0,
timeout 0, monitor coverage 기록, unload 뒤 context 0을 만족해야 한다. throughput이 늘지
않거나 tail latency/error가 악화되면 더 높은 worker/parallel 설정을 기본값으로 채택하지
않는다.

### 15.5 server-like smoke

offline 안정 설정에서 낮은 QPS부터 올린다.

```bash
python3 -m src.main \
  --model resnet50 \
  --target rbln-static \
  --artifact /path/to/resnet50.rbln \
  --dataset datasets/imagenet_1k \
  --inference-mode async_queue \
  --scenario server_like \
  --target-qps 10 \
  --batch-size 1 \
  --worker-count 4 \
  --queue-capacity 64 \
  --min-duration-sec 30 \
  --min-samples 100 \
  --latency-slo-ms 100 \
  --monitor
```

QPS 증가 시 scheduled delay, queue delay, p95/p99 latency, timeout, utilization, power,
energy/sample을 함께 본다. 단순히 NPU util이 가장 높은 설정이 아니라 timeout 없이
SLO와 exact-count invariant를 만족하는 가장 높은 지속 QPS를 채택한다.

## 16. 완료 기준

이번 브랜치는 다음을 모두 만족할 때만 구현 완료다.

1. SDK가 없는 개발 host에서 모든 RBLN focused test가 통과한다.
2. 기존 target registry와 Mobilint selector behavior가 회귀하지 않는다.
3. static E2E는 sync runtime 하나만, async queue는 AsyncRuntime 하나만 만든다.
4. async queue가 별도 adapter queue 없이 기존 bounded queue/backpressure/metric을 쓴다.
5. accepted RBLN job의 callback, ACK, input ownership과 shutdown invariant가 race test로
   증명된다.
6. timeout 상태에서 physical completion 전 runtime unload를 하지 않는다.
7. monitor가 device 0만 수집하고 power 단위, energy, coverage, KMD/firmware/PCI metadata를
   정확히 기록한다.
8. ResNet50, YOLOv5m, BERT, PatchTST의 실제 artifact smoke가 각각 exit 0이다.
9. run 종료 뒤 `rbln-smi` context가 비고 memory가 baseline으로 돌아온다.
10. full suite에서 기준 12건 외 새 failure가 없다.
11. setup 문서에 raw HF/ONNX와 executable `.rbln`의 차이, 모델별 inspect 요구사항,
    실제 command와 troubleshooting이 포함된다.

## 17. 후속 확장 경계

`rbln-vllm`은 이 static runtime class를 상속하지 않는다. target/collector/result
metadata만 공유하고, vLLM-RBLN의 in-process async engine을 `create_native_backend()`에
맞추어 별도 generation runtime으로 구현한다. token stream은 PR #24의
`GenerationObservation`을 생성하고 vLLM continuous batching이 request batching을
소유한다. framework dynamic batch는 1로 유지한다.

compiler plugin은 static runtime이 안정된 뒤 `.rbln` artifact를 생산하는 별도
`Compiler` 구현으로 추가한다. compile cache, fixed shape/bucketing, target NPU, compiler
version과 compile time을 runtime benchmark 밖에서 관리한다. 이 순서를 지키면 현재
branch의 runtime/queue/monitor 결과가 compile latency나 vLLM cold compile과 섞이지
않는다.
