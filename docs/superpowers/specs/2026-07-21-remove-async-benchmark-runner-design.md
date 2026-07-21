# AsyncBenchmarkRunner 제거 설계

**상태:** 구현됨

**승인일:** 2026-07-21

## 1. 결정

`AsyncBenchmarkRunner`를 병합 전에 완전히 제거한다. deprecated alias나 별도
compatibility module은 남기지 않는다. CLI와 framework 내부 테스트는 public
`InferenceEngine.run_async()`를 직접 사용한다.

이 변경은 의도적인 공개 API 파기다. 다음 import는 더 이상 지원하지 않는다.

```python
from core.async_inference import AsyncBenchmarkRunner
from core.async_inference.runner import AsyncBenchmarkRunner
```

실제 비동기 실행 구현인 `_AsyncRunController`, `AsyncInferenceEngine`, producer,
bounded queue, worker, completion, metrics와 result schema는 유지한다.

## 2. 제거 이유

현재 `AsyncBenchmarkRunner`는 다음 작업만 수행한다.

- `InferenceEngine` 하나를 생성한다.
- 속성 접근과 변경을 `InferenceEngine`으로 전달한다.
- `run()`을 `InferenceEngine.run_async()`로 전달한다.
- active controller의 failure diagnostic을 다시 노출한다.

queue, worker, batching, runtime dispatch, completion, evaluator 또는 결과 저장을
직접 담당하지 않는다. 따라서 독립적인 추론 책임이 없는 public façade를 유지하면
공개 진입점이 두 개로 보이고, `InferenceEngine`이 추론 전체를 관리한다는 설계가
흐려진다.

기존 façade를 유지한 이유는 이전 공개 API 호환뿐이었다. 병합 전 단계이며 외부
호환을 보존하지 않기로 승인했으므로 지금 완전히 제거한다.

## 3. 목표 구조

```text
main.py
  ├─ model/runtime/component 조립
  ├─ async artifact 예약과 결과 저장
  └─ InferenceEngine 생성
           │
           └─ run_async(config, warmup_runs, monitor)
                         │
                         ▼
                _AsyncRunController
                  ├─ warmup
                  ├─ Offline/Server-like producer
                  ├─ monitor와 run lifecycle
                  ├─ flush/shutdown
                  └─ metric/details 조립
                         │
                         ▼
                AsyncInferenceEngine
                  ├─ bounded request queue
                  ├─ slot/backpressure
                  ├─ worker와 dynamic batch
                  ├─ RuntimeExecutor dispatch/ACK
                  └─ cancellation/shutdown
                         │
                         ▼
                CompletionCoordinator
                  └─ exact-once terminal → decoder/evaluator
```

공개 추론 진입점은 `InferenceEngine` 하나다. `_AsyncRunController`는 한 번의 async
run을 조율하는 private implementation이고, `AsyncInferenceEngine`은 queue와 worker를
수행하는 내부 실행기다. `main.py`는 component 조립과 artifact 저장을 계속 담당한다.

## 4. 책임과 공개 계약

### 4.1 InferenceEngine

다음을 소유한다.

- dataloader, runtime, evaluator, decoder dependency
- 공통 `InferencePipeline`
- 하나의 `RuntimeExecutor`
- `CompletionCoordinator`
- e2e/async mode claim과 single-run lifecycle
- active `_AsyncRunController`

기존 façade가 CLI에 제공하던 진단을 다음 read-only public property로 직접 제공한다.

```python
engine.failure_phase -> str
engine.runtime_unload_safe_after_failure -> bool
```

controller가 만들어지기 전 기본값은 각각 `"created"`, `True`다. controller가
만들어진 뒤에는 controller의 안전하게 동기화된 값을 반환한다. 이 property는 queue나
controller 자체를 외부에 노출하지 않는다.

dependency mutation은 기존 `InferenceEngine._set_dependency()` 규칙을 유지한다.
run claim 뒤 변경은 계속 거부한다. 이번 변경에서 새 public setter를 만들지 않는다.

### 4.2 main.py

async 경로에서 `AsyncBenchmarkRunner` 대신 `InferenceEngine`을 직접 생성한다.

```python
engine = InferenceEngine(
    loader,
    runtime,
    evaluator,
    decoder=decoder,
    max_new_tokens=args.max_new_tokens,
    trace_callback=trace_callback,
    lifecycle_callback=lifecycle_callback,
)

async_result = engine.run_async(
    config,
    warmup_runs=args.warmup,
    monitor=hw_monitor,
)
```

failure handling에서 사용하던 `runner.failure_phase`와
`runner.runtime_unload_safe_after_failure`는 동일한 engine property로 교체한다.
artifact reservation, trace writer, runtime unload, details/CSV persistence와 terminal
marker 규칙은 변경하지 않는다.

### 4.3 async_inference package

`core.async_inference.__all__`과 package import에서 `AsyncBenchmarkRunner`를 제거한다.
`AsyncInferenceConfig`, `AsyncBenchmarkResult`, `AsyncScenario`와 기존 result/request
type은 그대로 공개한다.

`runner.py`에는 private `_AsyncRunController`와 그 helper가 남는다. 이 파일의 이름
변경은 불필요한 diff와 import churn을 늘리므로 이번 범위에서 하지 않는다.

## 5. 데이터·오류 흐름

정상 흐름은 façade 한 단계를 제외하고 완전히 동일하다.

```text
CLI → InferenceEngine.run_async()
→ _AsyncRunController → producer → bounded queue
→ RuntimeExecutor → CompletionCoordinator → evaluator
→ AsyncBenchmarkResult → CLI persistence
```

validation, warmup, engine start, producer, flush, shutdown 또는 persistence 실패의
우선순위와 cleanup 계약도 변경하지 않는다. CLI는 engine의 공개 진단 property를 통해
기존과 동일한 failure phase와 runtime unload 안전성을 판단한다.

다음 불변식을 보존한다.

- validation 전 실패에서는 runtime unload가 안전하다.
- engine start 뒤 소유권이 남을 수 있는 실패에서는 controller 판단을 따른다.
- async invalid run은 성공으로 저장하지 않는다.
- reserved/final run ID, CSV/details/trace 연결은 유지한다.
- request ID, submission token, dispatch token과 ACK authority는 변경하지 않는다.

## 6. TDD 설계

Production 변경 전에 다음 RED 계약을 먼저 추가한다.

1. `core.async_inference`가 `AsyncBenchmarkRunner`를 export하지 않는다.
2. CLI async 경로가 `InferenceEngine`을 생성하고 `run_async()`를 호출한다.
3. `InferenceEngine.failure_phase`가 controller 생성 전 `"created"`, validation 이후
   실제 failure phase를 반환한다.
4. `InferenceEngine.runtime_unload_safe_after_failure`가 controller 생성 전 `True`이고,
   engine lifecycle 이후 controller의 값을 반환한다.
5. validation/warmup/start/shutdown failure에서 CLI가 engine 진단을 사용하여 기존
   unload와 failure artifact 규칙을 보존한다.

위 테스트가 현재 façade 사용 때문에 실패하는 것을 확인한 뒤 production을 수정한다.

기존 `test_async_runner.py`의 테스트는 두 종류로 분류한다.

- 실제 async behavior test: `InferenceEngine(...).run_async(...)` 기준으로 이전한다.
- façade forwarding/export만 검증하는 test: 삭제하거나 위 public engine diagnostic
  계약 test로 대체한다.

CLI test에서 `AsyncBenchmarkRunner`를 monkeypatch하던 부분은 `InferenceEngine` test
double로 이전한다. test double은 production의 private controller를 직접 모사하지 않고
`run_async()`, `failure_phase`, `runtime_unload_safe_after_failure` public contract만
제공한다.

## 7. 검증 기준

- 새 RED test가 production 변경 전에 의도한 이유로 실패한다.
- InferenceEngine/async orchestration/CLI focused suite가 통과한다.
- native async executor와 shutdown/cancellation race suite가 통과한다.
- 전체 `framework/tests`가 통과한다.
- 실제 ONNX Runtime CPU async CLI가 exit 0, `CPUExecutionProvider`, valid count
  invariant, outstanding 0을 만족한다.
- CSV, JSON details, 선택적 JSONL trace와 reserved/final run ID 연결이 유지된다.
- production import/export와 현재 구조도·공개 API 설명에서
  `AsyncBenchmarkRunner` 참조가 0이다. 과거 계획과 제거 이력을 설명하는 문서는 해당
  이름을 역사적 문맥으로 유지할 수 있다.
- `git diff --check`와 Python compile 검증이 통과한다.

## 8. 변경 파일 범위

예상 production 변경:

- `framework/src/main.py`
- `framework/src/core/inference_engine.py`
- `framework/src/core/async_inference/runner.py`
- `framework/src/core/async_inference/__init__.py`

예상 test 변경:

- `framework/tests/test_inference_engine.py`
- `framework/tests/test_async_runner.py`
- `framework/tests/test_async_cli.py`
- `framework/tests/test_async_onnx_cpu.py`
- `framework/tests/test_object_detection_loader_async.py`
- `framework/tests/_async_hostile_result_process.py`

예상 문서 변경:

- `docs/unified-inference-engine-design.md`
- `docs/superpowers/specs/2026-07-14-async-inference-queue-design.md`
- `framework/src/core/README.md`
- `framework/CHANGELOG.md`

정확한 파일 목록은 상세 구현 계획에서 `rg` 결과를 기준으로 확정한다.

## 9. 비범위

- e2e `BenchmarkRunner` 제거
- `_AsyncRunController` 파일 이동 또는 rename
- `AsyncInferenceEngine` rename
- queue, worker, producer, metric 또는 result schema 변경
- 실제 NPU vendor adapter 추가
- MLPerf LoadGen integration, API/log compatibility, submission/compliance 대응

## 10. 승인 기준

`AsyncBenchmarkRunner`가 production import/export와 현재 구조 설명에서 제거되고, CLI와
모든 실제 async behavior test가 `InferenceEngine`을 직접 사용해야 한다. 과거 계획과 제거
이력은 역사적 설명으로 남길 수 있지만 외부 compatibility alias는 남기지 않는다. 이
변경으로 실행 결과, 오류 처리, artifact 또는 측정 의미가 달라져서는 안 된다.
