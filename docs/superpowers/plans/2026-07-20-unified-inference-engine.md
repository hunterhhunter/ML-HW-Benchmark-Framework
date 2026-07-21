# Unified InferenceEngine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 동기 `e2e`와 `async_queue`가 하나의 `InferenceEngine`, 데이터 파이프라인, completion 계약을 공유하고, blocking runtime과 vendor-native async runtime을 교체 가능한 `RuntimeExecutor`로 실행하게 한다.

**Architecture:** `BenchmarkRunner`와 호환용 `AsyncBenchmarkRunner`는 warmup·monitor·CLI 호환 경계를 담당하고 실제 추론 lifecycle은 `InferenceEngine`에 위임한다. `InferenceEngine`은 하나의 `InferencePipeline`, `CompletionCoordinator`, `RuntimeExecutor`를 소유한다. 동기는 inline completion으로 queue/thread를 만들지 않고 비동기는 기존 bounded queue와 exact-once 로직을 보존한다.

**Tech Stack:** Python 3.12, `dataclasses`, `threading`, `queue`, NumPy, pytest, ONNX, ONNX Runtime `CPUExecutionProvider`.

## Global Constraints

- 모든 production code는 그 동작을 요구하는 실패 테스트와 예상한 RED 로그가 먼저 있어야 한다. RED를 확인하지 못한 production change는 삭제하고 테스트부터 다시 시작한다.
- 각 작업은 `RED → GREEN → REFACTOR → focused regression → commit → task review` 순서로 실행한다.
- 동기 `e2e`의 output, evaluator metric, runtime-only latency 입력, monitor 경계, CLI 기본값, CSV/details 결과 계약을 보존한다.
- 동기 `e2e`는 Framework Queue, completion queue, worker thread, native in-flight registry를 만들지 않는다.
- 비동기 경로의 bounded Framework Queue, bounded Completion Queue, backpressure, timing, counter, exact-once terminal, failure truth 계약을 변경하지 않는다.
- `submitted = accepted + rejected`와 `accepted = completed + failed + outstanding`을 모든 async 종료 경로에서 보존한다.
- request ID, submission token, executor dispatch token, vendor job ID는 서로 다른 identity다. vendor job ID를 framework terminal truth로 사용하지 않는다.
- native async input/output buffer와 in-flight permit은 `CompletionCoordinator` terminal ACK 전에는 해제하지 않는다.
- MLPerf LoadGen은 신뢰성 레퍼런스일 뿐이다. `mlperf_loadgen` 의존성, SUT/QSL API, 로그·제출·compliance 호환을 추가하지 않는다.
- CI와 실제 인수 대상은 ONNX Runtime CPU 및 fake native async다. 실제 NPU vendor adapter, Backend, Frontend, 자동 executor 전환은 범위 밖이다.
- test double은 실제 protocol을 구현하는 작은 fake로 작성한다. mock 호출 자체가 아니라 output, terminal state, queue ownership, buffer lifetime을 검증한다.
- 기존 공개 `BenchmarkRunner`, `AsyncBenchmarkRunner`, CLI option, async result schema는 호환 façade로 보존한다.

---

### Task 1: Blocking RuntimeExecutor 실행 경계

**Files:**
- Create: `framework/src/core/runtime_executor.py`
- Modify: `framework/src/core/inference_pipeline.py`
- Create: `framework/tests/test_runtime_executor.py`
- Modify: `framework/tests/test_inference_pipeline.py`

**Interfaces:**
- Consumes: `Runtime.run()`, `Runtime.generate()`, `Runtime.supports_generate()`.
- Produces: `RuntimeExecution`, `RuntimeExecutor.execute(inputs, timeout=None)`, `acknowledge(execution)`, `shutdown(timeout)`, `BlockingRuntimeExecutor`.
- Preserves: `InferencePipeline.invoke()` as a compatibility delegate; 실행 로직은 `BlockingRuntimeExecutor` 한 곳에만 둔다.

`BlockingRuntimeExecutor.__init__(runtime, *, is_llm: bool, max_new_tokens: int = 256, stop_token_ids=None)` is the exact constructor used by every later task.

- [ ] **Step 1: Write failing blocking and generation tests**

Create `framework/tests/test_runtime_executor.py`:

```python
def test_blocking_executor_runs_array_runtime_and_reports_latency():
    runtime = ArrayRuntime()
    executor = BlockingRuntimeExecutor(runtime, is_llm=False)
    inputs = {"input": np.array([[1.0]], dtype=np.float32)}
    execution = executor.execute(inputs)
    np.testing.assert_array_equal(execution.outputs["output"], [[2.0]])
    assert execution.timing_ms >= 0.0
    assert execution.generated_tokens == 0
    assert execution.dispatch_token is None
    assert execution.error_type is None
    assert runtime.calls == [inputs]
    executor.acknowledge(execution)
    assert executor.shutdown(timeout=0.0) is True


def test_blocking_executor_preserves_generation_result_mapping():
    runtime = GenerationRuntime()
    executor = BlockingRuntimeExecutor(
        runtime,
        is_llm=True,
        max_new_tokens=17,
        stop_token_ids=[2, 3],
    )
    execution = executor.execute(
        {"input_ids": np.array([[1, 2]], dtype=np.int64)}
    )
    np.testing.assert_array_equal(execution.outputs["generated_ids"], [[4, 5]])
    np.testing.assert_array_equal(execution.outputs["generated_lengths"], [2])
    assert execution.generated_tokens == 2
    assert execution.timing_ms["total_ms"] == 3.0
    assert execution.timing_ms["timing_source"] == "measured"
```

Use these fixture returns so the assertions are reproducible:

```python
class ArrayRuntime:
    def __init__(self):
        self.calls = []

    def run(self, inputs):
        self.calls.append(inputs)
        return {"output": inputs["input"] + 1}


class GenerationRuntime:
    def generate(self, inputs, max_new_tokens, stop_token_ids):
        return GenerationResult(
            generated_ids=np.array([[4, 5]], dtype=np.int64),
            generated_lengths=np.array([2], dtype=np.int64),
            total_ms=3.0,
            ttft_ms=1.0,
            tpot_ms=2.0,
            num_tokens=2,
            timing_mode="native",
            uses_kv_cache=True,
            timing_source="measured",
        )
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=framework/src /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_runtime_executor.py -q
```

Expected: collection FAIL with `ModuleNotFoundError: No module named 'core.runtime_executor'`.

- [ ] **Step 3: Implement the minimal executor module**

Create these exact public types in `runtime_executor.py`:

```python
@dataclass(frozen=True)
class RuntimeExecution:
    outputs: Optional[Dict[str, Any]]
    timing_ms: float | Dict[str, Any] | None
    generated_tokens: int = 0
    dispatch_token: Optional[int] = None
    vendor_job_id: Any = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class RuntimeExecutor(ABC):
    @abstractmethod
    def execute(self, inputs, timeout=None) -> RuntimeExecution:
        raise NotImplementedError

    @abstractmethod
    def acknowledge(self, execution: RuntimeExecution) -> None:
        raise NotImplementedError

    @abstractmethod
    def shutdown(self, timeout: float) -> bool:
        raise NotImplementedError
```

`BlockingRuntimeExecutor.execute()` copies the current `InferencePipeline.invoke()` behavior exactly: `perf_counter()` around `runtime.run()`, or `runtime.generate()` mapped to `generated_ids`, optional `generated_lengths`, token count, and the six existing generation timing fields. `acknowledge()` is a no-op and `shutdown()` returns `True`.

- [ ] **Step 4: Remove duplicate invocation logic**

In `InferencePipeline.__init__` create one compatibility executor:

```python
self._compat_executor = BlockingRuntimeExecutor(
    runtime,
    is_llm=self.is_llm,
    max_new_tokens=max_new_tokens,
    stop_token_ids=self.stop_token_ids,
)
```

Alias `RuntimeInvocation = RuntimeExecution` and replace `invoke()` with:

```python
def invoke(self, runtime_input):
    return self._compat_executor.execute(runtime_input)
```

- [ ] **Step 5: Verify GREEN and commit**

```bash
PYTHONPATH=framework/src /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_runtime_executor.py framework/tests/test_inference_pipeline.py -q
git add framework/src/core/runtime_executor.py framework/src/core/inference_pipeline.py \
  framework/tests/test_runtime_executor.py framework/tests/test_inference_pipeline.py
git commit -m "refactor(framework): introduce runtime executor boundary"
```

Expected: both files PASS without a new warning.

---

### Task 2: Inline CompletionCoordinator와 동기 InferenceEngine

**Files:**
- Create: `framework/src/core/inference_engine.py`
- Modify: `framework/src/core/async_inference/completion.py`
- Modify: `framework/src/core/inference_pipeline.py`
- Create: `framework/tests/test_inference_engine.py`
- Modify: `framework/tests/test_async_completion.py`

**Interfaces:**
- Produces: `InferenceEngine.warmup(runs, batch_size)`, `run_e2e(batch_size=1, max_steps=None)`.
- Produces: `CompletionCoordinator(..., queue_capacity=None, raise_callback_errors=True)` with `queue is None`, `thread is None`, caller-thread completion, exact membership/terminal commit.

The exact engine constructor carried through Tasks 2–7 is:

```python
InferenceEngine(
    dataloader,
    runtime,
    evaluator,
    *,
    decoder=None,
    max_new_tokens: int = 256,
    runtime_executor: RuntimeExecutor | None = None,
    trace_callback=None,
    lifecycle_callback=None,
)
```

`test_inference_engine.py` uses these concrete fixtures: a two-sample loader containing `[1, 2] → 3` and `[3, 4] → 7`, a runtime whose `run()` reduces each row and whose async capability methods return one worker/no dynamic batching, and an evaluator that extends `(prediction, label)` pairs and reports `Total Samples`. Its loader implements both `load_batch()` and `load_by_index()` so the same fixture is valid in Task 5 async tests. `FailingEvaluator(primary)` raises that exact object from `add_batch()`. `RecordingMonitor(events, summary)` appends `"start"`/`"stop"` and returns a copy of `summary`.

- [ ] **Step 1: Write RED inline-completion tests**

Append:

```python
def test_inline_completion_uses_membership_without_queue_or_thread():
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(), evaluator=RecordingEvaluator(), decoder=None,
        metrics=AsyncMetricsCollector(0, 1), queue_capacity=None,
        raise_callback_errors=True,
    )
    req = request(0)
    coordinator.start()
    coordinator.register(req)
    coordinator.submit(completion(req))
    assert coordinator.queue is None
    assert coordinator.thread is None
    assert coordinator.snapshot_outstanding() == ()
    assert coordinator.stop(timeout=0.0) is True


def test_inline_completion_commits_failure_then_reraises_same_error():
    primary = ValueError("quality failure")
    evaluator = FailingEvaluator(primary)
    metrics = AsyncMetricsCollector(0, 1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(), evaluator=evaluator, decoder=None,
        metrics=metrics, queue_capacity=None, raise_callback_errors=True,
    )
    req = request(0)
    coordinator.register(req)
    with pytest.raises(ValueError) as raised:
        coordinator.submit(completion(req))
    assert raised.value is primary
    assert coordinator.snapshot_outstanding() == ()
    assert metrics.finalize(10)["summary"]["async_failed_requests"] == 1
```

- [ ] **Step 2: Write RED synchronous-engine tests**

Create `test_inference_engine.py` with real fake loader/runtime/evaluator fixtures and:

```python
def test_e2e_engine_uses_inline_completion_and_no_async_threads():
    before = {thread.ident for thread in threading.enumerate()}
    engine = InferenceEngine(FakeLoader(), FakeRuntime(), FakeEvaluator())
    result = engine.run_e2e(batch_size=2)
    assert result == {"pairs": [(3.0, 3), (7.0, 7)], "Total Samples": 2}
    assert engine.completion.queue is None
    assert engine.completion.thread is None
    created = [t for t in threading.enumerate() if t.ident not in before]
    assert not [t for t in created if t.name.startswith("async-")]


def test_e2e_engine_warmup_resets_loader_before_measurement():
    loader, runtime, evaluator = FakeLoader(), FakeRuntime(), FakeEvaluator()
    engine = InferenceEngine(loader, runtime, evaluator)
    engine.warmup(runs=1, batch_size=1)
    result = engine.run_e2e(batch_size=2)
    assert runtime.warmup_calls == 1
    assert result["Total Samples"] == 2
```

- [ ] **Step 3: Run RED**

```bash
PYTHONPATH=framework/src /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_async_completion.py framework/tests/test_inference_engine.py \
  -k 'inline_completion or e2e_engine' -q
```

Expected: FAIL because `queue_capacity=None`, `raise_callback_errors`, and `core.inference_engine` are unsupported.

- [ ] **Step 4: Implement inline delivery**

Make `queue_capacity: int | None` and `raise_callback_errors: bool = False`. In inline mode create neither queue nor thread; `start()` is a no-op; `submit()` rejects `operation_key` and calls `_handle()` inline; `stop()` succeeds only with no reservation/outstanding item. `_handle()` retains the exact decoder/evaluator exception, commits failure metrics/trace/removal first, then re-raises only for inline `raise_callback_errors=True`. Queued defaults do not change.

- [ ] **Step 5: Implement minimal InferenceEngine**

The constructor creates one `InferencePipeline` and one default `BlockingRuntimeExecutor`. `run_e2e()` lazily creates exactly one inline `CompletionCoordinator` and assigns it to `engine.completion`; `run_async()` in Task 5 instead assigns the queued coordinator owned by its private run controller. This avoids an unused second completion service. `warmup()` loads/collates/prepares input, calls `runtime.warmup()`, and resets the loader in `finally`.

`run_e2e()` must per batch create monotonic request IDs, register them, execute once, convert `RuntimeExecution` to `BatchCompletion`, submit inline, acknowledge only after terminal commit, and return `evaluator.compute()`. Add shared `InferencePipeline.batch_size(collated)` and `reset_dataloader_cursor()` helpers for static/non-static loaders.

- [ ] **Step 6: Verify GREEN and commit**

```bash
PYTHONPATH=framework/src /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_inference_engine.py framework/tests/test_async_completion.py \
  framework/tests/test_inference_pipeline.py -q
git add framework/src/core/inference_engine.py framework/src/core/inference_pipeline.py \
  framework/src/core/async_inference/completion.py framework/tests/test_inference_engine.py \
  framework/tests/test_async_completion.py
git commit -m "feat(framework): add synchronous unified inference engine"
```

---

### Task 3: BenchmarkRunner를 얇은 e2e façade로 전환

**Files:**
- Modify: `framework/src/core/benchmarkrunner.py`
- Modify: `framework/tests/test_inference_engine.py`
- Modify: `framework/tests/test_inference_pipeline.py`
- Modify: `framework/tests/test_main_paths.py`

**Interfaces:**
- Preserves: `BenchmarkRunner(...).run(warmup_runs=1, batch_size=1, max_steps=None)`.
- Produces: `BenchmarkRunner.engine: InferenceEngine`; runner owns warmup/monitor, engine owns inference/completion/evaluator.

- [ ] **Step 1: Write RED façade and cleanup tests**

```python
def test_benchmark_runner_owns_monitor_and_delegates_inference():
    events = []
    monitor = RecordingMonitor(events, {"hw_samples": 1})
    runner = BenchmarkRunner(FakeLoader(), FakeRuntime(), FakeEvaluator(), monitor=monitor)
    result = runner.run(warmup_runs=1, batch_size=2)
    assert isinstance(runner.engine, InferenceEngine)
    assert events == ["start", "stop"]
    assert result["Total Samples"] == 2
    assert result["hw_samples"] == 1


def test_benchmark_runner_stops_monitor_when_engine_raises():
    events = []
    runner = BenchmarkRunner(
        FakeLoader(), FakeRuntime(), FailingEvaluator(RuntimeError("failed")),
        monitor=RecordingMonitor(events, {}),
    )
    with pytest.raises(RuntimeError, match="failed"):
        runner.run(warmup_runs=0, batch_size=2)
    assert events == ["start", "stop"]
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=framework/src /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_inference_engine.py -k benchmark_runner -q
```

Expected: FAIL because `runner.engine` does not exist and monitor stop is not protected from evaluator failure.

- [ ] **Step 3: Delegate to the engine**

Construct the engine with the existing dependencies:

```python
self.engine = InferenceEngine(
    dataloader,
    runtime,
    evaluator,
    decoder=decoder,
    max_new_tokens=max_new_tokens,
)
```

Keep legacy private helper methods as delegates to `engine.pipeline`. Implement `run()` as warmup, monitor start, `engine.run_e2e()` in `try`, monitor stop in `finally`, and monitor summary merge after successful inference. Retain current user-facing log messages; do not keep a second inference loop.

- [ ] **Step 4: Verify GREEN, e2e regressions, commit**

```bash
PYTHONPATH=framework/src /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_inference_engine.py framework/tests/test_inference_pipeline.py \
  framework/tests/test_main_paths.py framework/tests/test_llama_e2e.py \
  framework/tests/test_object_detection_e2e.py framework/tests/test_patchtst_e2e.py -q
git add framework/src/core/benchmarkrunner.py framework/tests/test_inference_engine.py \
  framework/tests/test_inference_pipeline.py framework/tests/test_main_paths.py
git commit -m "refactor(framework): delegate e2e runs to inference engine"
```

---

### Task 4: 기존 async queue에 RuntimeExecutor와 terminal ACK 연결

**Files:**
- Modify: `framework/src/core/async_inference/engine.py`
- Modify: `framework/src/core/async_inference/runner.py`
- Modify: `framework/src/core/async_inference/types.py`
- Modify: `framework/tests/test_async_engine.py`
- Modify: `framework/tests/test_async_runner.py`

**Interfaces:**
- Consumes: `RuntimeExecutor.execute()` and `RuntimeExecutor.acknowledge()`.
- Produces: `AsyncInferenceEngine(..., executor=None)`; omitted executor creates one blocking executor for compatibility.
- Contract: `RuntimeExecution` is acknowledged only after coordinator handoff ACK and dequeue retirement.

- [ ] **Step 1: Write RED terminal-ACK tests**

Add a protocol-complete `GatedExecutor` and gated evaluator to `test_async_engine.py`:

```python
def test_runtime_execution_is_acknowledged_only_after_terminal_handoff():
    executor = GatedExecutor(dispatch_token=41)
    evaluator = BlockingEvaluator()
    config = AsyncInferenceConfig(
        queue_capacity=1, worker_count=1,
        max_batch_size=1, min_samples=1,
        flush_timeout_sec=1.0,
    )
    engine, runtime, evaluator, metrics = build(
        config, evaluator=evaluator, executor=executor,
    )
    engine.start()
    assert engine.submit(make_request(0), block=True) is True
    assert evaluator.entered.wait(timeout=1.0)
    assert executor.acknowledged == []
    evaluator.release.set()
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True
    assert [item.dispatch_token for item in executor.acknowledged] == [41]
    summary = metrics.finalize(time.monotonic_ns())["summary"]
    assert summary["async_outstanding_requests"] == 0


def test_executor_failure_execution_is_one_failed_terminal_then_acked():
    executor = FailureExecutor("DeviceError", "failed", dispatch_token=42)
    config = AsyncInferenceConfig(
        queue_capacity=1, worker_count=1,
        max_batch_size=1, min_samples=1,
        flush_timeout_sec=1.0,
    )
    engine, runtime, evaluator, metrics = build(config, executor=executor)
    engine.start()
    assert engine.submit(make_request(0), block=True) is True
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True
    summary = metrics.finalize(time.monotonic_ns())["summary"]
    assert summary["async_completed_requests"] == 0
    assert summary["async_failed_requests"] == 1
    assert summary["async_outstanding_requests"] == 0
    assert len(executor.acknowledged) == 1
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=framework/src /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_async_engine.py \
  -k 'runtime_execution_is_acknowledged or executor_failure_execution' -q
```

Expected: FAIL because the queue engine does not accept an executor and still calls `pipeline.invoke()`.

Extend the existing `build()` test helper with `executor=None` and pass it to `AsyncInferenceEngine`; do not add a second helper. `GatedExecutor` implements all three `RuntimeExecutor` methods, returns a `RuntimeExecution(outputs={"output": np.array([[0.0]])}, timing_ms=1.0, dispatch_token=41)`, records ACK objects, and returns `True` from shutdown. `FailureExecutor` uses the same implementation with `outputs=None` and the specified failure fields.

- [ ] **Step 3: Inject executor and bind execution lifetime to handoff**

Retain `runtime` only for capability validation. Add `executor=None`, defaulting to `BlockingRuntimeExecutor(runtime, is_llm=pipeline.is_llm, max_new_tokens=pipeline.max_new_tokens, stop_token_ids=pipeline.stop_token_ids)`.

Replace worker invocation with:

```python
execution = self.executor.execute(
    runtime_input,
    timeout=self.config.flush_timeout_sec,
)
```

Build `BatchCompletion` from `RuntimeExecution`, including failure fields. Add `_execution_by_handoff` under `_handoff_retirement_lock`, bind it to the existing `completion_operation_key`, and call `executor.acknowledge()` exactly once only after coordinator retirement is proven. On ACK exception, retain the mapping, mark `request_failed`, and make shutdown fail. Call `executor.shutdown(remaining_deadline)` after all completion handoffs retire and before `STOPPED`.

- [ ] **Step 4: Pass one shared executor from async runner setup**

Add `runtime_executor=None` to the async runner/controller constructor, create one default blocking executor, and pass it to `AsyncInferenceEngine`. After this step no async worker may call `InferencePipeline.invoke()`.

- [ ] **Step 5: Verify GREEN and all queue reliability tests**

```bash
PYTHONPATH=framework/src /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_async_engine.py framework/tests/test_async_completion.py \
  framework/tests/test_async_runner.py -q
```

Expected: PASS including capacity, backpressure, duplicate/unknown/stale completion, shutdown, counter, timing, and failure-truth tests.

- [ ] **Step 6: Commit Task 4**

```bash
git add framework/src/core/async_inference/engine.py \
  framework/src/core/async_inference/runner.py \
  framework/src/core/async_inference/types.py \
  framework/tests/test_async_engine.py framework/tests/test_async_runner.py
git commit -m "refactor(framework): execute async batches through runtime executor"
```

---

### Task 5: InferenceEngine의 async 소유권과 호환 façade

**Files:**
- Modify: `framework/src/core/inference_engine.py`
- Modify: `framework/src/core/async_inference/runner.py`
- Modify: `framework/src/core/async_inference/__init__.py`
- Modify: `framework/tests/test_inference_engine.py`
- Modify: `framework/tests/test_async_runner.py`
- Modify: `framework/tests/test_async_onnx_cpu.py`

**Interfaces:**
- Produces: `InferenceEngine.run_async(config, warmup_runs=1, monitor=None) -> AsyncBenchmarkResult`.
- Produces: public `AsyncBenchmarkRunner.engine: InferenceEngine` compatibility façade.
- Private: current async orchestration body becomes `_AsyncRunController`, created only by `InferenceEngine.run_async` and never exported.

- [ ] **Step 1: Write RED unified ownership tests**

```python
def test_same_engine_type_runs_e2e_and_async_with_same_quality():
    e2e = InferenceEngine(FakeLoader(), FakeRuntime(), FakeEvaluator()).run_e2e(
        batch_size=2
    )
    async_result = InferenceEngine(
        FakeLoader(), FakeRuntime(), FakeEvaluator()
    ).run_async(
        AsyncInferenceConfig(
            queue_capacity=2, worker_count=1,
            max_batch_size=1, min_samples=1,
        ),
        warmup_runs=0,
    )
    assert e2e["Total Samples"] == async_result.metrics["Total Samples"] == 2
    assert async_result.metrics["pairs"] == e2e["pairs"]
    assert async_result.metrics["async_outstanding_requests"] == 0


def test_async_runner_is_compatibility_facade_over_inference_engine():
    runner = AsyncBenchmarkRunner(FakeLoader(), FakeRuntime(), FakeEvaluator())
    result = runner.run(
        AsyncInferenceConfig(queue_capacity=2, min_samples=1),
        warmup_runs=0,
    )
    assert isinstance(runner.engine, InferenceEngine)
    assert runner.failure_phase == "complete"
    assert result.metrics["async_outstanding_requests"] == 0
```

Add a direct ONNX CPU test to `test_async_onnx_cpu.py` using the existing tiny sum model:

```python
def test_unified_engine_direct_onnx_cpu_e2e_async_parity(tmp_path):
    model_path = tmp_path / "tiny-sum.onnx"
    _create_sum_model(model_path)
    e2e_runtime = _load_cpu_runtime(model_path)
    async_runtime = _load_cpu_runtime(model_path)
    try:
        e2e = InferenceEngine(
            TinySumLoader(), e2e_runtime, SumEvaluator()
        ).run_e2e(batch_size=2)
        async_result = InferenceEngine(
            TinySumLoader(), async_runtime, SumEvaluator()
        ).run_async(
            AsyncInferenceConfig(
                queue_capacity=4, max_batch_size=2,
                batch_timeout_ms=10, min_samples=1,
            ),
            warmup_runs=0,
        )
    finally:
        e2e_runtime.unload()
        async_runtime.unload()
    assert e2e["accuracy"] == async_result.metrics["accuracy"] == 1.0
    assert e2e["Total Samples"] == async_result.metrics["Total Samples"] == 4
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=framework/src /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_inference_engine.py framework/tests/test_async_onnx_cpu.py \
  -k 'same_engine_type or compatibility_facade or unified_engine_direct' -q
```

Expected: FAIL because `InferenceEngine.run_async()` and `AsyncBenchmarkRunner.engine` do not exist.

- [ ] **Step 3: Move public async ownership**

Rename the existing implementation class to `_AsyncRunController`. It receives the already-created `pipeline` and `runtime_executor`; delete its creation of a second pipeline/executor.

Implement `InferenceEngine.run_async()` with a lazy import, a single-run guard, construction of `_AsyncRunController` with the engine's dataloader/runtime/evaluator/decoder/pipeline/executor/callbacks, and delegation to `controller.run()`. Save `_async_controller` for failure diagnostics. Apply the same single-run guard to `run_e2e`; `warmup()` remains valid before a run.

- [ ] **Step 4: Add the compatibility façade**

Define a new public `AsyncBenchmarkRunner` with the existing signature. It creates one `InferenceEngine`, delegates `run()` to `engine.run_async(config, warmup_runs, monitor=self.monitor)`, and delegates `failure_phase` and `runtime_unload_safe_after_failure` to the active controller. Keep `async_inference.__all__` unchanged.

- [ ] **Step 5: Verify GREEN and commit**

```bash
PYTHONPATH=framework/src /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_inference_engine.py framework/tests/test_async_runner.py \
  framework/tests/test_async_onnx_cpu.py framework/tests/test_inference_pipeline.py -q
git add framework/src/core/inference_engine.py framework/src/core/async_inference/runner.py \
  framework/src/core/async_inference/__init__.py framework/tests/test_inference_engine.py \
  framework/tests/test_async_runner.py framework/tests/test_async_onnx_cpu.py
git commit -m "refactor(framework): unify sync and async inference ownership"
```

---

### Task 6: Fake SDK로 검증하는 NativeAsyncRuntimeExecutor

**Files:**
- Modify: `framework/src/core/runtime_executor.py`
- Create: `framework/tests/test_native_async_runtime_executor.py`
- Modify: `framework/tests/test_async_engine.py`

**Interfaces:**
- Produces: `NativeAsyncOutcome`, `NativeAsyncExecutorSnapshot`, `NativeAsyncRuntimeExecutor`.
- Backend protocol: `submit_async(inputs, callback) -> vendor_job_id`; callback receives one `NativeAsyncOutcome` and may run inline before submit returns.
- Contract: dispatch token is framework-owned; vendor job ID is diagnostic only; first terminal outcome wins; `acknowledge()` is the normal buffer/permit release point.

The exact constructor is `NativeAsyncRuntimeExecutor(backend, *, max_inflight: int, completion_timeout_sec: float)`; its public methods are the three `RuntimeExecutor` methods plus `snapshot() -> NativeAsyncExecutorSnapshot`.

In this first implementation `execute()` is a callback-to-blocking bridge: each framework worker submits one native async job and waits on that job's event, while multiple workers allow up to `min(worker_count, max_inflight)` SDK jobs concurrently. A separate nonblocking dispatcher thread and more native jobs than framework workers are follow-up optimizations, not hidden scope in this plan.

- [ ] **Step 1: Write the RED fake-SDK matrix**

Create a real event/thread fake backend that stores full `(inputs, callback, vendor_job_id)` jobs and lets the test complete them in any order. Write:

The file starts with this protocol-complete fake (test-side `complete()` is valid because it controls the fake SDK, not a production object):

```python
class FakeNativeBackend:
    def __init__(self, *, inline_outcome=None, submit_error=None):
        self.inline_outcome = inline_outcome
        self.submit_error = submit_error
        self.condition = threading.Condition()
        self.jobs = {}
        self.submitted = []
        self.next_job = 1

    def submit_async(self, inputs, callback):
        if self.submit_error is not None:
            raise self.submit_error
        with self.condition:
            job_id = f"job-{self.next_job}"
            self.next_job += 1
            self.jobs[job_id] = (inputs, callback)
            self.submitted.append(job_id)
            self.condition.notify_all()
        if self.inline_outcome is not None:
            callback(self.inline_outcome)
        return job_id

    def wait_for_jobs(self, count, timeout=1.0):
        deadline = time.monotonic() + timeout
        with self.condition:
            while len(self.submitted) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("fake SDK submission not observed")
                self.condition.wait(remaining)
            return tuple(self.submitted)

    def complete(self, job_id, outcome):
        _, callback = self.jobs[job_id]
        callback(outcome)
```

The file must contain these concrete cases:

- inline callback: backend invokes `callback(NativeAsyncOutcome(outputs={"output": inputs["input"]}))` before returning job ID `"inline-1"`; assert returned output, non-`None` dispatch token, and vendor ID.
- out-of-order: start two `executor.execute()` calls in a two-thread pool, complete the second vendor job before the first, and assert each future receives the output for its own input and the two dispatch tokens differ.
- duplicate: complete one job twice with different outputs, assert the first output is returned and `duplicate_callbacks == 1`.
- submit failure: fake backend raises `DeviceSubmitError("submit failed")`; assert a failure `RuntimeExecution`, ACK it, then submit a succeeding job to prove the permit was released.
- timeout: keep one job incomplete past `completion_timeout_sec`; assert `error_type == "NativeAsyncTimeout"`, the input weak reference remains live before ACK, then ACK and assert `inflight == 0` and the reference becomes collectible.
- shutdown: before ACK assert `shutdown(timeout=0.0) is False`; after ACK assert it is `True`.
- queue integration: pass the native executor into the real `AsyncInferenceEngine`, reverse two callbacks, and assert exact outputs, two terminal traces, both counter equations, and zero outstanding/inflight.

The first and duplicate cases use these exact assertion shapes:

```python
def test_native_executor_accepts_inline_callback_before_vendor_id_return():
    backend = FakeNativeBackend(
        inline_outcome=NativeAsyncOutcome(outputs={"output": np.array([[7]])}, timing_ms=1.0)
    )
    executor = NativeAsyncRuntimeExecutor(
        backend, max_inflight=1, completion_timeout_sec=1.0
    )
    execution = executor.execute({"input": np.array([[7]])})
    np.testing.assert_array_equal(execution.outputs["output"], [[7]])
    assert execution.vendor_job_id == "job-1"
    assert execution.dispatch_token is not None
    executor.acknowledge(execution)
    assert executor.snapshot().inflight == 0


def test_native_executor_ignores_duplicate_callback_and_keeps_first_result():
    backend = FakeNativeBackend()
    executor = NativeAsyncRuntimeExecutor(
        backend, max_inflight=1, completion_timeout_sec=1.0
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(executor.execute, {"input": np.array([[1]])})
        job_id = backend.wait_for_jobs(1)[0]
        backend.complete(job_id, NativeAsyncOutcome(outputs={"output": np.array([[1]])}, timing_ms=1.0))
        backend.complete(job_id, NativeAsyncOutcome(outputs={"output": np.array([[99]])}, timing_ms=2.0))
        execution = future.result(timeout=1.0)
    np.testing.assert_array_equal(execution.outputs["output"], [[1]])
    assert executor.snapshot().duplicate_callbacks == 1
    executor.acknowledge(execution)
```

Assertions must check actual output ordering, unique dispatch tokens, diagnostic counts, `snapshot().inflight`, strong input liveness before ACK and collection after ACK, terminal trace uniqueness, both counter equations, and zero outstanding. Do not assert only that a fake method was called.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=framework/src /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_native_async_runtime_executor.py -q
```

Expected: collection FAIL because native executor types do not exist.

- [ ] **Step 3: Add native outcome and snapshot types**

```python
@dataclass(frozen=True)
class NativeAsyncOutcome:
    outputs: Optional[Dict[str, Any]] = None
    timing_ms: float | Dict[str, Any] | None = None
    generated_tokens: int = 0
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class NativeAsyncExecutorSnapshot:
    inflight: int
    duplicate_callbacks: int
    late_callbacks: int
    submit_failures: int
    timeouts: int
```

Use private `_NativeDispatch` records holding dispatch token, strong input reference, event, vendor ID, first outcome, terminal flag, and ACK flag.

- [ ] **Step 4: Implement dispatch/callback/ACK lifecycle**

`execute()` must acquire a bounded semaphore; allocate a monotonic dispatch token under lock; publish the record before `submit_async`; accept inline callback; store returned vendor ID without keying on it; wait for callback/deadline; install a single `NativeAsyncTimeout` failure on timeout; and return `RuntimeExecution` with dispatch token/vendor ID.

The callback commits only the first outcome. Callback after timeout increments `late_callbacks`; another callback after a normal outcome increments `duplicate_callbacks`; neither overwrites output. Synchronous submit exception becomes a normalized failure `RuntimeExecution` and increments `submit_failures`; traceback/tensors never enter diagnostics.

`acknowledge()` idempotently removes the exact record, clears strong input/outcome refs, and releases one permit. Unknown non-`None` token raises `RuntimeError`. `shutdown(timeout)` waits for registry emptiness. `snapshot()` returns primitive copies under lock.

- [ ] **Step 5: Exercise native executor through the real queue**

Use two workers and two requests, complete callbacks in reverse order, then assert:

```python
assert summary["async_submitted_requests"] == 2
assert summary["async_accepted_requests"] == 2
assert summary["async_completed_requests"] == 2
assert summary["async_failed_requests"] == 0
assert summary["async_outstanding_requests"] == 0
assert native_executor.snapshot().inflight == 0
```

Add duplicate, timeout, and submit-failure queue variants; each must assert exact-once trace membership and both counter equations.

- [ ] **Step 6: Verify GREEN repeatedly and commit**

```bash
PYTHONPATH=framework/src /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_native_async_runtime_executor.py framework/tests/test_async_engine.py -q
for run in 1 2 3 4 5; do
  PYTHONPATH=framework/src /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
    -m pytest framework/tests/test_native_async_runtime_executor.py -q || exit 1
done
git add framework/src/core/runtime_executor.py \
  framework/tests/test_native_async_runtime_executor.py framework/tests/test_async_engine.py
git commit -m "feat(framework): add native async runtime executor"
```

Expected: all repeated runs PASS and no `async-*` thread leaks.

---

### Task 7: 실제 ONNX CPU 인수, 문서, CHANGELOG

**Files:**
- Modify: `docs/unified-inference-engine-design.md`
- Modify: `docs/superpowers/specs/2026-07-14-async-inference-queue-design.md`
- Modify: `framework/src/core/README.md`
- Modify: `framework/CHANGELOG.md`
- Modify only for a discovered regression: `framework/tests/test_async_cli_onnx_cpu.py`

**Interfaces:**
- Acceptance: actual `python src/main.py` subprocess for e2e and async_queue on ONNX Runtime CPU, isolated results, exit 0, matching quality/sample count, async outstanding 0.
- Documents: class ownership, executor choice, ID mapping, TDD/CLI evidence, non-goals.

- [ ] **Step 1: Run focused architecture and real CLI acceptance**

```bash
HF_DATASETS_CACHE=/tmp/mlhw-hf-cache HF_DATASETS_OFFLINE=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=framework/src \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_inference_engine.py \
  framework/tests/test_runtime_executor.py \
  framework/tests/test_native_async_runtime_executor.py \
  framework/tests/test_async_onnx_cpu.py \
  framework/tests/test_async_cli_onnx_cpu.py -q
```

Expected: PASS. The CLI test creates fresh ONNX/data assets, invokes `python src/main.py` for both modes, verifies `CPUExecutionProvider`, run IDs, CSV/details/trace linkage, and zero outstanding.

- [ ] **Step 2: Update architecture docs**

Add this implemented structure:

```text
BenchmarkRunner / AsyncBenchmarkRunner (compatibility façades)
                       │
                       ▼
                InferenceEngine
          ┌────────────┴────────────┐
     e2e inline                async_queue
  no queue / no worker      bounded Framework Queue
          └──── RuntimeExecutor ────┘
                   ├─ BlockingRuntimeExecutor
                   └─ NativeAsyncRuntimeExecutor
                              │
                       vendor SDK queue

Both modes:
DataLoader → InferencePipeline → RuntimeExecution → CompletionCoordinator
→ Decoder/Postprocessor → Evaluator → Result
```

State that `_AsyncRunController` and native dispatch registry are private, and that no MLPerf code/API/log compatibility was added.

- [ ] **Step 3: Record implementation and tests in CHANGELOG**

Under `[Unreleased]` add:

```markdown
### Added
- Added a unified `InferenceEngine` and pluggable blocking/native-async runtime executor contract; native callback behavior is CI-tested with a fake SDK and no vendor dependency.

### Changed
- `e2e` and `async_queue` now share the inference pipeline and completion/evaluator path. E2E remains inline with no queue/worker, while async preserves bounded queues and exact-once terminal accounting.

### Tested
- Added TDD coverage for executor dispatch/ACK ownership, inline completion, sync/async parity, native callback ordering/duplicate/timeout/submit failure/shutdown, and actual ONNX Runtime CPU CLI runs.
```

- [ ] **Step 4: Run full regression**

```bash
HF_DATASETS_CACHE=/tmp/mlhw-hf-cache HF_DATASETS_OFFLINE=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=framework/src \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests -q
```

Expected: all PASS; only the pre-existing unregistered `integration` marker warning is acceptable. Any behavior bug found now gets a new RED regression test before its fix.

- [ ] **Step 5: Verify hygiene, commit, final review**

```bash
git diff --check
rg -n 'T[B]D|TO[D]O|FIX[M]E' docs/unified-inference-engine-design.md \
  framework/src/core/inference_engine.py framework/src/core/runtime_executor.py \
  framework/CHANGELOG.md
git status --short
git add docs/unified-inference-engine-design.md \
  docs/superpowers/specs/2026-07-14-async-inference-queue-design.md \
  framework/src/core/README.md framework/CHANGELOG.md \
  framework/tests/test_async_cli_onnx_cpu.py
git commit -m "docs(framework): record unified inference engine acceptance"
```

Generate a review package from branch merge-base to `HEAD` and dispatch the final whole-branch reviewer. Critical/Important findings go to one fix subagent; every behavior fix starts with a failing regression test, then focused/full tests and final review repeat. Do not claim completion or merge readiness before this gate is clean.
