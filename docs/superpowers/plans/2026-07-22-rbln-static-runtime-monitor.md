# Rebellions RBLN Static Runtime and Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a production-safe `rbln-static` target that runs precompiled `.rbln` artifacts synchronously or through the framework native-async queue and records RBLN-CA22 telemetry.

**Architecture:** Keep `InferenceEngine`, `NativeAsyncRuntimeExecutor`, bounded queues, completion coordination, and result persistence as the control plane. Add a lazy `RblnRuntime`, an owner-event-loop `RblnNativeBackend`, and a throttled `RblnCollector`; compose them through the existing runtime, monitor, and target registries. A loaded compiled model may create either one sync `Runtime` or one async `AsyncRuntime`, never both.

**Tech Stack:** Python 3.10, NumPy, `asyncio`, `threading`, Rebellions `rebel-compiler` Python API, `rbln-smi` JSON, pytest.

---

## Global implementation constraints

- Work only in `/tmp/ml-hw-benchmark-rbln-runtime-monitor` on `feat/rbln-runtime-monitor`.
- Treat `docs/superpowers/specs/2026-07-22-rbln-static-runtime-monitor-design.md` as the approved contract. Any behavioral change requires an explicit design amendment before code.
- Do not add `rebel-compiler`, `optimum-rbln`, or `vllm-rbln` to base requirements. Import `rebel` only inside runtime methods.
- Do not implement compilation, model downloading, Llama generation, vLLM, an external server, dynamic-shape bucketing, tensor parallelism, or multi-NPU execution in this branch.
- Lock target `rbln-static` to device 0, artifact suffix `.rbln`, backend `rbln`, tensor-parallel size 1, and request batch size 1.
- Reuse the framework bounded request queue. Do not add an adapter-side scheduling queue or one waiter thread per request.
- Never create sync and async RBLN runtimes for the same loaded compiled model.
- Logical request timeout is not physical cancellation. Retain request ownership and executor permits until a late SDK completion is observed and acknowledged.
- Defaults: `async_parallel=1`, framework `worker_count=1`, `runtime_timeout_sec=60`, `shutdown_timeout_sec=300.0`. RBLN SDK 0.11의 pybind runtime constructor 계약에 맞춰 `runtime_timeout_sec`는 signed 32-bit 범위의 양의 built-in integer seconds만 허용한다.
- Invoke monitoring as the exact argv `rbln-smi -b -j -d 0`, with `shell=False`, `check=True`, and a 2-second command timeout. Internally throttle vendor polling to at least one second.
- Omit absent telemetry fields; never synthesize zero for an unavailable sensor.
- Baseline before RBLN code: 1,359 passed, 13 skipped, 12 failed. Eleven failures are pre-existing Furiosa callback timeouts and one is an isolated Hugging Face DNS/download failure.
- For every code task: write RED tests, run them and inspect the expected failure, implement the smallest behavior, rerun the focused suite GREEN, then commit only that task.

## File responsibility map

| File | Responsibility |
|---|---|
| `framework/src/runtimes/rbln_rt.py` | Static artifact inspection, sync runtime, native async bridge, resource ownership, device/runtime provenance |
| `framework/src/runtimes/__init__.py` | Lazy RBLN registration and aliases; importing the registry must not import `rebel` |
| `framework/src/monitors/rbln_collector.py` | `rbln-smi` polling, normalization, throttling, process context, power integration, coverage |
| `framework/src/monitors/__init__.py` | RBLN collector registration and lazy construction |
| `framework/src/core/targets.py` | `rbln-static` target contract and fixed options |
| `framework/src/main.py` | CLI backend, artifact/task validation, locked target option merging, native async capacity, safe diagnostics |
| `framework/tests/test_rbln_runtime.py` | SDK-free sync runtime and artifact contract tests |
| `framework/tests/test_rbln_native_backend.py` | SDK-free owner-loop, callback, timeout, race, and shutdown tests |
| `framework/tests/test_rbln_collector.py` | JSON parsing, command safety, throttling, energy, and failure policy tests |
| `framework/tests/test_plugin_registry.py` | Runtime/collector/target registration and import-laziness graph |
| `framework/tests/test_main_paths.py` | CLI resolution, locked selectors, artifact and task rejection |
| `framework/tests/test_async_cli.py` | RBLN native executor construction and serialized diagnostics |
| `framework/docs/rbln-setup.md` | Operator preflight, artifact inspection, commands, tuning, telemetry, troubleshooting |
| `framework/README.md` | Supported target/model matrix and quick-start links |
| `framework/src/runtimes/README.md` | Runtime ownership and async behavior |
| `framework/CHANGELOG.md` | Added RBLN static runtime/monitor scope and explicit exclusions |

## Task 1: Static artifact inspection and synchronous runtime

**Files:**

- Create: `framework/src/runtimes/rbln_rt.py`
- Create: `framework/tests/test_rbln_runtime.py`
- Test: `framework/tests/test_runtime_factory.py`

### Step 1: Build one reusable fake SDK fixture

- [ ] Add fake tensor descriptors, inspected metadata, compiled model, sync runtime, async runtime, and device functions to `test_rbln_runtime.py`. The fake must record constructor calls, input order, timeout, device id, and destruction thread.

```python
class FakeTensor:
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype


class FakeInspect:
    compiler_version = "0.11.0"
    npu = "RBLN-CA22"
    tensor_parallel_size = 1
    uuid = "artifact-uuid"
    alloc_per_node = (4096,)
    inputs = (
        FakeTensor("input_ids", (1, 8), "int64"),
        FakeTensor("attention_mask", (1, 8), "int64"),
    )
    outputs = (FakeTensor("logits", (1, 2), "float32"),)


class FakeRBLNCompiledModel:
    inspect_calls = []

    @classmethod
    def inspect(cls, path):
        cls.inspect_calls.append(path)
        return FakeInspect()


class FakeRebel:
    RBLNCompiledModel = FakeRBLNCompiledModel

    def Runtime(self, path, **kwargs):
        self.runtime_calls.append((path, kwargs))
        instance = FakeSyncRuntime()
        self.sync_instances.append(instance)
        return instance

```

The fixture must expose the API names used by the installed `rebel-compiler==0.11.0` documentation: `RBLNCompiledModel.inspect(path)`, `Runtime(path, ...)`, `AsyncRuntime(path, ...)`, `npu_is_available`, and `get_npu_name`. If the real server probe in Task 8 shows a version-specific signature difference, isolate it in the one constructor/inspection helper responsible for that operation rather than spreading version branches.

### Step 2: Write RED load and compatibility tests

- [ ] Add tests for:

  - `.rbln` plus backend aliases `rbln`, `rebel`, `rbln-static` are compatible.
  - wrong suffix/backend is incompatible.
  - module import and object construction do not import `rebel`.
  - `load()` imports SDK lazily, checks device 0, inspects without creating `Runtime` or `AsyncRuntime`, and records metadata.
  - unavailable device, actual device-name mismatch, artifact NPU mismatch, explicitly reported tensor-parallel size other than 1, missing descriptors, dynamic dimensions, or zero outputs each fail before runtime allocation.
  - mapping-style and attribute-style inspect metadata normalize to the same contract.
  - input/output fixed shapes and input dtypes must match `CompiledModel.spec`; a single input may use positional name fallback, and one unnamed artifact output may bind to one profile output, while multi-input or multi-output name mismatches are rejected rather than guessed.
  - a second `load()` without `unload()` is rejected.

```python
def test_load_inspects_contract_without_allocating_runtime(
    tmp_path, monkeypatch, fake_rebel
):
    artifact = tmp_path / "bert.rbln"
    artifact.touch()
    monkeypatch.setattr(
        "runtimes.rbln_rt.import_module", lambda name: fake_rebel
    )
    runtime = RblnRuntime(device="0")

    runtime.load(_compiled_model(artifact, backend="rbln"))

    assert fake_rebel.runtime_calls == []
    assert runtime.get_device_spec()["artifact_compiler_version"] == "0.11.0"
    assert runtime.get_device_spec()["artifact_uuid"] == "artifact-uuid"
```

- [ ] Run and confirm RED due to the missing module:

```bash
cd framework
pytest -q tests/test_rbln_runtime.py -k 'load or compatible'
```

Expected: collection fails with `ModuleNotFoundError: runtimes.rbln_rt`.

### Step 3: Implement the inspection boundary

- [ ] Add `RblnRuntime(Runtime)` with these normalized constructor fields:

```python
class RblnRuntime(Runtime):
    def __init__(self, **runtime_options):
        self.device = str(runtime_options.get("device", "0"))
        self.device_id = _require_builtin_int(
            runtime_options.get("device_id", 0), "device_id", minimum=0
        )
        self.async_parallel = _require_builtin_int(
            runtime_options.get("async_parallel", 1),
            "async_parallel",
            minimum=1,
        )
        self.runtime_timeout_sec = _require_builtin_int(
            runtime_options.get("runtime_timeout_sec", 60),
            "runtime_timeout_sec",
            minimum=1,
            maximum=(1 << 31) - 1,
        )
        self.shutdown_timeout_sec = _require_positive_finite_number(
            runtime_options.get("shutdown_timeout_sec", 300.0),
            "shutdown_timeout_sec",
        )
        self.max_async_inflight = _require_builtin_int(
            runtime_options.get("max_async_inflight", 1),
            "max_async_inflight",
            minimum=1,
        )
        # Validate device_id == 0, async_parallel in {1, 2}, and timeout limits.
        # Initialize compiled/runtime/backend/metadata fields to unloaded values.
```

The numeric validators must reject booleans, non-finite values, and user-defined conversion objects rather than silently coercing them. `device_id` is valid only when exactly zero; `async_parallel` is valid only when exactly 1 or 2. `runtime_timeout_sec` must be a built-in integer in `[1, 2_147_483_647]` because RBLN SDK 0.11 requires an exact C++ signed-int-compatible Python value; do not truncate a fractional timeout. `shutdown_timeout_sec` remains a positive finite host-wait value and may be fractional.

- [ ] Implement these private helpers with bounded, user-facing errors:

```python
@staticmethod
def _load_rebel(): ...

@staticmethod
def _inspect_compiled_model(rebel, path): ...

@staticmethod
def _normalize_shape(raw_shape) -> tuple[int, ...]: ...

@staticmethod
def _normalize_dtype(raw_dtype) -> np.dtype: ...

def _inspect_contract(self, inspected) -> None: ...
```

`_inspect_contract()` must normalize mapping/attribute metadata and reject booleans as dimensions, non-integer/static dimensions, dimensions less than one, missing/duplicate input names, missing/duplicate multi-output names, target NPU not exactly equal to detected `RBLN-CA22`, and an explicitly reported `tensor_parallel_size != 1`. Treat an absent/`None` tensor-parallel field as unavailable provenance. Compare normalized descriptors with `CompiledModel.spec`; allow name fallback only when both sides have exactly one input or both sides have exactly one output and the artifact output name is absent.

- [ ] Implement `load()` in this order: validate `CompiledModel` compatibility and file existence → import SDK → `npu_is_available(0)` → `get_npu_name(0)` → open compiled model → inspect → validate → atomically publish loaded state. Constructor/runtime failure must leave the object unloadable and retryable.

- [ ] Implement `is_compatible()` and `get_device_spec()`. Device metadata must contain only JSON-safe scalar/list values and include:

```text
backend, device, device_id, accelerator_vendor, accelerator_name,
detected_npu, sdk_version, artifact_compiler_version, artifact_npu,
tensor_parallel_size, artifact_uuid, artifact_alloc_per_node,
input_names, input_shapes, input_dtypes, output_names, output_shapes,
output_dtypes, async_parallel, max_async_inflight
```

Read `sdk_version` with `importlib.metadata.version("rebel-compiler")`; if package metadata is unavailable in an injected fake, accept only a primitive bounded module version fallback and otherwise omit the field.

- [ ] Rerun the load tests GREEN:

```bash
cd framework
pytest -q tests/test_rbln_runtime.py -k 'load or compatible'
```

### Step 4: Write RED execution, input, output, and cleanup tests

- [ ] Add tests for:

  - first `run()` creates exactly one sync runtime for device 0 with the configured timeout;
  - `warmup()` reuses the same runtime and passes inputs in inspected descriptor order;
  - missing/extra input, dtype mismatch, static shape mismatch, scalar input, and batch dimension other than 1 fail before SDK invocation;
  - one SDK output maps to the inspected output name whether returned bare or in a list/tuple;
  - multiple outputs map by inspected order;
  - output count, dtype, or fixed shape mismatch raises a bounded error;
  - `native_async_max_batch_size()` returns exact built-in `int(1)`;
  - sync mode reports `max_concurrent_workers() == 1`;
  - `unload()` clears references, is idempotent, and allows a later reload.

```python
def test_sync_run_orders_inputs_and_normalizes_named_outputs(
    loaded_runtime, fake_rebel
):
    inputs = {
        "attention_mask": np.ones((1, 8), dtype=np.int64),
        "input_ids": np.arange(8, dtype=np.int64).reshape(1, 8),
    }

    outputs = loaded_runtime.run(inputs)

    assert list(outputs) == ["logits"]
    assert fake_rebel.sync_instances[0].calls[0][0] is inputs["input_ids"]
    assert fake_rebel.sync_instances[0].calls[0][1] is inputs["attention_mask"]
```

- [ ] Run and confirm RED at the first missing execution method:

```bash
cd framework
pytest -q tests/test_rbln_runtime.py -k 'run or warmup or unload or batch'
```

### Step 5: Implement synchronous execution and cleanup

- [ ] Implement:

```python
def _require_loaded(self): ...
def _ordered_inputs(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]: ...
def _normalize_outputs(self, raw_outputs) -> dict[str, np.ndarray]: ...
def _ensure_sync_runtime(self): ...
def run(self, inputs): ...
def warmup(self, inputs, num_runs=1): ...
def native_async_max_batch_size(self) -> int: return 1
def max_concurrent_workers(self) -> int: ...
def unload(self) -> None: ...
```

`max_concurrent_workers()` returns 1 until native async mode is selected; after native backend creation it returns `max_async_inflight`. `run()` and `warmup()` must reject use once async ownership has been selected. `unload()` must refuse to release the compiled model if async shutdown cannot prove that all accepted jobs physically completed.

- [ ] Run the complete Task 1 suite:

```bash
cd framework
pytest -q tests/test_rbln_runtime.py tests/test_runtime_factory.py
```

Expected: all pass without `rebel` installed.

### Step 6: Commit Task 1

- [ ] Inspect and commit only the runtime and its tests:

```bash
git diff --check
git status --short
git add framework/src/runtimes/rbln_rt.py framework/tests/test_rbln_runtime.py
git commit -m "feat: add RBLN static runtime"
```

## Task 2: Owner-loop native async backend

**Files:**

- Modify: `framework/src/runtimes/rbln_rt.py`
- Create: `framework/tests/test_rbln_native_backend.py`
- Test: `framework/tests/test_native_async_runtime_executor.py`

### Step 1: Write RED owner-loop and submission tests

- [ ] Reuse or move the fake SDK fixture into a local fixture module only if both RBLN test files need more than 30 duplicated lines. Do not introduce a production abstraction for test convenience.
- [ ] Add tests proving:

  - `create_native_backend()` creates one daemon owner thread and constructs `rebel.AsyncRuntime` on that thread.
  - constructor receives artifact path, `device=0`, `tensor_type="np"`, configured `parallel`, and timeout.
  - `submit_async()` validates input synchronously, returns a monotonic `rbln-1` job id without waiting for completion, and calls the callback once.
  - output normalization is shared with sync mode.
  - SDK, normalization, and callback exceptions are bounded and do not leak input/job references.
  - multiple submitted coroutines run on the same event-loop thread and may complete out of order.

```python
def test_native_backend_constructs_and_executes_on_owner_loop(
    loaded_runtime, fake_rebel
):
    loaded_runtime.async_parallel = 2
    backend = loaded_runtime.create_native_backend()
    outcomes = []

    job_id = backend.submit_async(valid_inputs(), outcomes.append)
    assert fake_rebel.release_one(job_id)
    assert fake_rebel.wait_for_callbacks(1)

    owner_ident = backend.owner_thread_ident
    assert fake_rebel.async_constructor_thread == owner_ident
    assert fake_rebel.async_run_threads == [owner_ident]
    assert job_id == "rbln-1"
    assert outcomes[0].error_type is None
```

- [ ] Run and confirm RED because native backend construction is not implemented:

```bash
cd framework
pytest -q tests/test_rbln_native_backend.py -k 'owner or submit or callback'
```

### Step 2: Implement the owner loop and exactly-once callback path

- [ ] Add a private job record that holds the job id, ordered input references, completion future, and terminal claim state:

```python
@dataclass
class _RblnAsyncJob:
    job_id: str
    inputs: list[np.ndarray]
    future: Future | None = None
    callback_claimed: bool = False
```

- [ ] Implement `RblnNativeBackend.__init__()` to:

  1. verify `RblnRuntime` is loaded and has no sync runtime;
  2. create a thread-safe condition, startup event, job mapping, monotonically increasing id, closing flag, and startup error field;
  3. start one daemon thread;
  4. create `asyncio.new_event_loop()` and `rebel.AsyncRuntime(...)` inside that thread;
  5. publish successful startup, then `loop.run_forever()`;
  6. wait no longer than a bounded constructor timeout and cleanly distinguish timeout from constructor failure.

- [ ] Implement submission as a direct cross-thread coroutine publication, with no extra queue and no per-request thread:

```python
def submit_async(self, inputs, callback):
    ordered = self.runtime._ordered_inputs(inputs)
    self._validate_single_batch(ordered)
    with self._condition:
        if self._closing:
            raise RuntimeError("RBLN native backend is shutting down.")
        job_id = f"rbln-{self._next_job_id}"
        self._next_job_id += 1
        job = _RblnAsyncJob(job_id=job_id, inputs=ordered)
        self._jobs[job_id] = job
    try:
        job.future = asyncio.run_coroutine_threadsafe(
            self._execute(job, callback), self._loop
        )
    except BaseException:
        with self._condition:
            self._jobs.pop(job_id, None)
            self._condition.notify_all()
        raise
    return job_id
```

- [ ] `_execute()` must capture start/end monotonic time, await `async_runtime.async_run(*job.inputs)`, normalize outputs, construct `NativeAsyncOutcome`, attempt callback exactly once, and release job/input ownership only in `finally`. Bound error type to 64 alphanumeric/underscore characters and use fixed adapter messages no longer than 512 characters.
- [ ] Rerun owner/submission tests GREEN.

### Step 3: Write RED mode-exclusivity, warmup, and shutdown tests

- [ ] Add tests for:

  - sync runtime active → native backend creation fails without allocating async runtime;
  - native backend active → `run()` fails without allocating sync runtime;
  - `run_warmup_blocking()` executes on the same async runtime, bypassing benchmark metrics;
  - shutdown rejects new submissions immediately;
  - shutdown waits for accepted jobs and returns `True` only after callback/finally cleanup, runtime release on owner loop, loop stop, and thread join;
  - shutdown deadline with an unfinished SDK coroutine returns `False`, leaves the owner loop/runtime alive, and later retry succeeds;
  - repeated successful `shutdown()` is idempotent;
  - `RblnRuntime.unload()` retains state when shutdown is unproven and succeeds after the late completion.

```python
def test_shutdown_timeout_retains_runtime_until_late_completion(
    loaded_runtime, fake_rebel
):
    backend = loaded_runtime.create_native_backend()
    job_id = backend.submit_async(valid_inputs(), lambda outcome: None)

    assert backend.shutdown(timeout=0.001) is False
    assert backend.owner_thread_alive
    with pytest.raises(RuntimeError, match="cleanup pending"):
        loaded_runtime.unload()

    fake_rebel.release_one(job_id)
    assert backend.shutdown(timeout=1.0) is True
    loaded_runtime.unload()
    assert loaded_runtime.execution_mode == "unloaded"
```

- [ ] Run and confirm RED at shutdown/warmup behavior:

```bash
cd framework
pytest -q tests/test_rbln_native_backend.py -k 'exclusive or warmup or shutdown or unload'
```

### Step 4: Implement warmup and two-phase shutdown

- [ ] Implement `run_warmup_blocking(inputs, timeout)` by publishing a private coroutine to the owner loop and waiting on its `concurrent.futures.Future.result(timeout=...)`. It must not enter measured callback/counter paths, but it must remain in a separate tracked warmup-future set until physical completion so a caller-side timeout cannot make unload unsafe.
- [ ] Implement `shutdown(timeout)` with one monotonic deadline:

  1. set `closing=True` under the condition;
  2. wait for `_jobs` and tracked warmup futures to empty, without cancelling running SDK coroutines;
  3. if deadline expires, return `False` with resources intact;
  4. schedule an owner-loop finalizer that clears `AsyncRuntime` then stops the loop;
  5. join the owner thread with remaining time;
  6. return `True` only if no jobs, no async runtime, stopped loop, and dead thread are all proven.

- [ ] Make `RblnRuntime.create_native_backend()` transactional. If startup rollback is complete, return to `loaded`; if incomplete, mark cleanup pending. Add an `execution_mode` field to `get_device_spec()`.
- [ ] Rerun complete Task 2 suites:

```bash
cd framework
pytest -q \
  tests/test_rbln_runtime.py \
  tests/test_rbln_native_backend.py \
  tests/test_native_async_runtime_executor.py
```

### Step 5: Commit Task 2

- [ ] Commit only native backend behavior and tests:

```bash
git diff --check
git add framework/src/runtimes/rbln_rt.py framework/tests/test_rbln_native_backend.py
git commit -m "feat: bridge RBLN native async runtime"
```

## Task 3: Integrate executor timeout and hostile race semantics

**Files:**

- Modify: `framework/tests/test_rbln_native_backend.py`
- Modify only if a new RBLN test exposes a generic defect: `framework/src/core/runtime_executor.py`
- Modify only if a new RBLN test exposes a generic defect: `framework/src/core/async_inference/engine.py`
- Test: `framework/tests/test_native_async_runtime_executor.py`
- Test: `framework/tests/test_async_completion.py`

### Step 1: Add integration tests around the existing native executor

- [ ] Construct `NativeAsyncRuntimeExecutor(backend, max_inflight=2, completion_timeout_sec=...)` around the fake RBLN backend and test:

  - two requests are accepted and a third nonblocking submission respects executor capacity;
  - reverse SDK completion still returns the correct output for each dispatch token;
  - success, SDK failure, malformed output, and callback failure each release exactly one adapter job;
  - executor logical timeout returns one timeout terminal but preserves inflight/permit ownership;
  - late callback after logical timeout increments late-callback accounting and only ACK plus physical completion returns inflight to zero;
  - duplicate SDK completion cannot produce a second terminal;
  - executor shutdown first blocks new submissions, then backend shutdown drains all accepted RBLN jobs.

```python
def test_logical_timeout_keeps_rbln_physical_ownership(
    loaded_runtime, fake_rebel
):
    backend = loaded_runtime.create_native_backend()
    executor = NativeAsyncRuntimeExecutor(
        backend, max_inflight=1, completion_timeout_sec=0.005
    )

    execution = executor.execute(valid_inputs(), timeout=0.001)
    assert execution.error_type == "TimeoutError"
    assert executor.snapshot().inflight == 1

    executor.acknowledge(execution)
    assert executor.snapshot().inflight == 1
    fake_rebel.release_all()
    assert wait_until(lambda: executor.snapshot().inflight == 0)
    assert executor.shutdown(timeout=1.0) is True
```

- [ ] Run RED/GREEN against existing executor semantics:

```bash
cd framework
pytest -q tests/test_rbln_native_backend.py -k 'executor or timeout or late or duplicate'
```

Expected RED is acceptable only for a precisely identified generic race. If tests already pass, do not edit generic executor code.

### Step 2: Add async-engine lifecycle coverage

- [ ] Use the smallest existing async test harness to verify one warmup and two measured requests use the same `AsyncRuntime`; warmup is absent from accepted/completed counts; queue capacity remains the framework queue; and no thread count grows per request.
- [ ] Add an assertion that `RblnRuntime.max_concurrent_workers()` equals the injected `max_async_inflight`, so `worker_count=4` is accepted while `max_batch_size=2` is rejected by `native_async_max_batch_size()==1`.
- [ ] Run the lifecycle regression suites:

```bash
cd framework
pytest -q \
  tests/test_rbln_native_backend.py \
  tests/test_native_async_runtime_executor.py \
  tests/test_async_completion.py \
  tests/test_inference_engine.py
```

### Step 3: Apply a generic fix only if evidence requires it

- [ ] If a test fails in shared code, first preserve the failing test and document the exact interleaving in its name/comments. Change only the lock/claim transition needed to restore exactly-once terminal and physical-ownership semantics. Rerun every existing native backend suite:

```bash
cd framework
pytest -q \
  tests/test_rbln_native_backend.py \
  tests/test_native_async_runtime_executor.py \
  tests/test_mobilint_native_backend.py \
  tests/test_furiosa_native_backend.py \
  tests/test_async_completion.py
```

### Step 4: Commit Task 3

- [ ] Commit tests and any evidence-backed generic fix:

```bash
git diff --check
git add framework/tests/test_rbln_native_backend.py
git add framework/src/core/runtime_executor.py framework/src/core/async_inference/engine.py
git commit -m "test: harden RBLN async lifecycle races"
```

Before committing, unstage either generic source file if `git diff --cached --name-only` shows it without an associated RED test and focused GREEN proof.

## Task 4: RBLN device telemetry collector

**Files:**

- Create: `framework/src/monitors/rbln_collector.py`
- Create: `framework/tests/test_rbln_collector.py`
- Test: `framework/tests/test_hw_monitor.py`

### Step 1: Define representative JSON fixtures and an injected runner

- [ ] Use the user-provided `rbln-smi -j` response as the main fixture, plus one payload with numeric JSON values and one payload with missing fields/contexts. The test runner records `args`, `shell`, `check`, and `timeout`, and returns scripted results/exceptions.

```python
class FakeRunner:
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        return SimpleNamespace(stdout=json.dumps(action))
```

### Step 2: Write RED command, parsing, and startup policy tests

- [ ] Add tests proving:

  - `is_available()` returns `True` without importing a Python SDK, allowing `start()` to own executable validation rather than silently dropping to system-only monitoring;
  - `start()` invokes exactly `['rbln-smi', '-b', '-j', '-d', '0']` with `shell=False`, `check=True`, `text=True`, and `timeout=2.0`;
  - missing executable, malformed initial JSON, missing device 0, or initial non-`normal` status is a setup error;
  - the provided string-valued payload normalizes utilization 0.0, used/total MiB, temperature 38.0 C, power 18.810987 W, P-state, KMD/firmware, PCI/NUMA/link, UUID, SID, and status;
  - only context rows matching `os.getpid()` contribute process-context allocated MiB.

```python
def test_start_uses_safe_device_scoped_command(user_payload):
    runner = FakeRunner([user_payload, user_payload])
    collector = RblnCollector(
        device_id=0,
        runner=runner,
        executable_resolver=lambda name: "/usr/bin/rbln-smi",
    )

    collector.start()
    metrics = collector.collect(force=True)

    args, kwargs = runner.calls[0]
    assert args == ["rbln-smi", "-b", "-j", "-d", "0"]
    assert kwargs == {
        "capture_output": True,
        "text": True,
        "check": True,
        "timeout": 2.0,
        "shell": False,
    }
    assert metrics["hw_accel_power_w"] == pytest.approx(18.810987)
```

- [ ] Run and confirm RED because the collector is absent:

```bash
cd framework
pytest -q tests/test_rbln_collector.py -k 'available or start or parse or context'
```

### Step 3: Implement safe snapshot normalization

- [ ] Add helpers that accept finite JSON numbers or suffixed strings and return `None` for missing/invalid values:

```python
def _number(value, suffix=None) -> float | None: ...
def _bytes(value) -> float | None: ...
def _power_w(value) -> float | None: ...
def _bounded_text(value, limit=128) -> str | None: ...
```

- [ ] Implement `RblnCollector(Collector)` with dependency injection for `runner`, `clock`, `executable_resolver=shutil.which`, and `process_id=None` (resolve `os.getpid()` inside `__init__`, not at module import). Keep these states:

```text
started, stopped, last_poll_at, last_snapshot, poll_attempts,
poll_successes, power_samples, energy_joules, last_power_w,
last_power_at, last_error_type, static_device_info
```

- [ ] `_snapshot()` must execute the exact argv, select device 0 even if more devices appear, parse contexts defensively, and split normalized data into current metrics and static device metadata. Do not emit a metric key when its source is absent.
- [ ] `start()` resolves and validates the executable through the initial snapshot. `stop()` takes one final forced snapshot when started, then becomes idempotent.
- [ ] Rerun command/parsing/startup tests GREEN.

### Step 4: Write RED throttling, energy, and transient-failure tests

- [ ] Add deterministic-clock tests proving:

  - `collect()` calls within one second return an empty dict without invoking `rbln-smi`;
  - the first power sample establishes a baseline but emits no energy;
  - two or more successful power samples integrate trapezoidal joules into the collector summary;
  - a post-start command timeout, nonzero exit, malformed JSON, missing device, or non-finite numeric field omits that sample, retains only a bounded summary note, and does not add energy;
  - coverage is `successful_polls / attempted_polls`; attempts skipped by throttling are not counted;
  - missing temperature/power/context fields remain absent rather than zero.

```python
def test_integrates_power_only_across_successful_samples(payload, fake_clock):
    runner = FakeRunner([payload(10.0), payload(14.0), payload(14.0)])
    collector = RblnCollector(
        runner=runner,
        clock=fake_clock,
        executable_resolver=lambda name: "/usr/bin/rbln-smi",
    )
    collector.start()             # t=0, 10 W baseline
    fake_clock.advance(2.0)

    metrics = collector.collect(force=True)  # t=2, 14 W

    assert metrics["hw_accel_power_w"] == pytest.approx(14.0)
    summary = collector.get_summary_metrics()
    assert summary["hw_accel_energy_j"] == pytest.approx(24.0)
    assert summary["hw_accel_monitor_coverage"] == pytest.approx(1.0)
```

- [ ] Run and confirm RED at throttle/energy/failure policy:

```bash
cd framework
pytest -q tests/test_rbln_collector.py -k 'throttle or energy or transient or missing or coverage'
```

### Step 5: Implement throttling, integration, and exported metrics

- [ ] Clamp `sample_interval_sec` to at least `1.0`; allow `force=True` only for lifecycle snapshots/tests. On successful polls, integrate energy using `(previous_w + current_w) / 2 * elapsed_seconds`. Do not bridge across a failed poll.
- [ ] Export current metrics with stable names:

```text
hw_accel_util, hw_accel_mem_used_mb, hw_accel_mem_proc_mb,
hw_accel_temp_c, hw_accel_power_w
```

- [ ] Export static info through the collector metadata path using:

```text
hw_accel_vendor, hw_accel_name, hw_accel_device_id,
hw_accel_device_node, hw_accel_uuid, hw_accel_serial_id,
hw_accel_status, hw_accel_pstate, hw_accel_monitor_source,
hw_accel_kmd_version, hw_accel_firmware_version,
hw_accel_pci_bus_id, hw_accel_pci_numa_node,
hw_accel_pci_link_speed, hw_accel_pci_link_width,
hw_accel_mem_total_mb
```

- [ ] Export collector-owned summaries through `get_summary_metrics()` as `hw_accel_energy_j` (only after two valid power samples), `hw_accel_power_samples`, `hw_accel_monitor_attempts`, `hw_accel_monitor_successes`, `hw_accel_monitor_coverage`, and an optional bounded `hw_accel_monitor_note`.

- [ ] Bound diagnostic error type/message; never serialize stdout/stderr wholesale. Rerun all collector tests:

```bash
cd framework
pytest -q tests/test_rbln_collector.py tests/test_hw_monitor.py
```

### Step 6: Commit Task 4

- [ ] Commit collector and tests:

```bash
git diff --check
git add framework/src/monitors/rbln_collector.py framework/tests/test_rbln_collector.py
git commit -m "feat: collect RBLN device telemetry"
```

## Task 5: Register the runtime, collector, and target without SDK imports

**Files:**

- Modify: `framework/src/runtimes/__init__.py`
- Modify: `framework/src/monitors/__init__.py`
- Modify: `framework/src/core/targets.py`
- Modify: `framework/tests/test_plugin_registry.py`

### Step 1: Write RED registry graph tests

- [ ] Add a subprocess or import-spy test that removes `rebel` from `sys.modules`, blocks any import whose root is `rebel`, imports all three registries, and asserts listing works.
- [ ] Add exact graph assertions:

```python
def test_rbln_static_target_graph_is_lazy_and_consistent():
    target = get_target("rbln-static")
    runtime = get_runtime_entry(target.runtime_name)
    collector = get_collector_entry(target.monitor_names[0])

    assert runtime.name == "rbln"
    assert runtime.aliases == ("rebel", "rbln-static")
    assert collector.name == "rbln"
    assert "rbln-smi" in collector.aliases
    assert target.device == "0"
    assert target.artifact_format == "rbln"
    assert target.monitor_names == ("rbln", "system")
    assert target.runtime_options == {
        "device_id": 0,
        "async_parallel": 1,
        "runtime_timeout_sec": 60,
        "shutdown_timeout_sec": 300.0,
    }
    assert target.monitor_options["rbln"] == {
        "device_id": 0,
        "sample_interval_sec": 1.0,
        "command_timeout_sec": 2.0,
    }
```

- [ ] Assert target capabilities contain exactly the agreed behavior classes: `rbln`, `sync`, `native_async`, `latency`, `throughput`, `monitor`, `npu`, `local`, `static_shape`; assert they do not contain `compile`, `generation`, `streaming`, `dynamic_batch`, or `multi_npu`.
- [ ] Assert legacy `resolve_target(None, "rbln", "0")` resolves to `rbln-static`, while explicit `--target` remains authoritative.
- [ ] Run and confirm RED on missing registrations:

```bash
cd framework
pytest -q tests/test_plugin_registry.py -k rbln
```

### Step 2: Add lazy registrations and target contract

- [ ] Register and export the runtime lazily:

```python
register_runtime(RuntimeEntry(
    name="rbln",
    module="runtimes.rbln_rt",
    class_name="RblnRuntime",
    aliases=("rebel", "rbln-static"),
    description="Rebellions runtime for precompiled static RBLN artifacts",
))
```

Add `RblnRuntime` to `__getattr__` and `__all__`; do not import `runtimes.rbln_rt` eagerly.

- [ ] Register the collector lazily:

```python
register_collector(CollectorEntry(
    name="rbln",
    module="monitors.rbln_collector",
    class_name="RblnCollector",
    aliases=("rbln-smi", "rebel"),
    description="Rebellions NPU telemetry through rbln-smi JSON",
))
```

- [ ] Register `rbln-static` in `core/targets.py` exactly as asserted by the tests, with `accelerator_vendor="Rebellions"`, `accelerator_name="RBLN NPU"`, `device_selector="0"`, and no compiler.
- [ ] Add the minimal legacy resolution mapping required by the existing `resolve_target()` scheme. Do not create a generic RBLN target for arbitrary devices.
- [ ] Run registry suites GREEN:

```bash
cd framework
pytest -q tests/test_plugin_registry.py tests/test_runtime_factory.py tests/test_hw_monitor.py
```

### Step 3: Commit Task 5

- [ ] Commit only registry and target changes:

```bash
git diff --check
git add \
  framework/src/runtimes/__init__.py \
  framework/src/monitors/__init__.py \
  framework/src/core/targets.py \
  framework/tests/test_plugin_registry.py
git commit -m "feat: register RBLN static target"
```

## Task 6: Wire CLI validation, native capacity, and safe result diagnostics

**Files:**

- Modify: `framework/src/main.py`
- Modify: `framework/tests/test_main_paths.py`
- Modify: `framework/tests/test_async_cli.py`
- Test: `framework/tests/test_plugin_registry.py`
- Test: `framework/tests/test_mobilint_runtime.py`
- Test: `framework/tests/test_mobilint_native_backend.py`

### Step 1: Write RED parser and early-rejection tests

- [ ] Add parser tests that `--backend rbln` is accepted and the `--target` help names `rbln-static`.
- [ ] Add direct helper/main-path tests proving:

  - target requires `--artifact`;
  - artifact must exist, be a regular file, and have case-insensitive `.rbln` suffix;
  - directories, symlinks resolving to directories, `.onnx`, raw Hugging Face directories, and missing files fail before runtime creation/model preparation;
  - RBLN does not invoke a compiler or a model download/prepare script;
  - `Task.NLP_GENERATION` fails before load with a message directing Llama to the future `rbln-vllm` target;
  - image classification, object detection, NLP classification/QA, and time-series forecast are accepted by the target-level task gate;
  - batch size other than 1 fails before load for both e2e and async modes.

```python
@pytest.mark.parametrize("task", [Task.NLP_GENERATION])
def test_rbln_static_rejects_generation_before_runtime_creation(task):
    with pytest.raises(ValueError, match="rbln-vllm"):
        _validate_target_task(_rbln_target(), task, batch_size=1)
```

- [ ] Run and confirm RED:

```bash
cd framework
pytest -q tests/test_main_paths.py -k 'rbln or locked_target'
```

### Step 2: Generalize locked target options without changing Mobilint behavior

- [ ] Replace Mobilint-only constants/helper with a data-driven selector contract:

```python
_LOCKED_TARGET_OPTIONS = {
    "mobilint-aries": ("mobilint", ("device_id", "expected_family")),
    "mobilint-regulus": ("mobilint", ("device_id", "expected_family")),
    "rbln-static": ("rbln", ("device_id",)),
}
```

- [ ] Keep exact type comparisons for integer `device_id`, case-insensitive comparison only for string `expected_family`, target definition consistency checks between runtime and monitor options, and source-specific error messages. Rename helpers generically but preserve existing Mobilint messages where tests assert them.
- [ ] Add tests for loader and CLI attempts to override RBLN `device_id=1`; equal exact override `device_id=0` is accepted and normalized. A boolean `True` must not equal integer 1 or 0.

### Step 3: Implement artifact/task validation and parser wiring

- [ ] Add small pure helpers near existing vendor validation:

```python
def _validate_precompiled_artifact(target, artifact_value: str | None) -> Path:
    # Require value, resolve path, require is_file(), and check target suffix.
    ...


def _validate_target_task(target, task: Task, batch_size: int) -> None:
    # RBLN static: batch exactly 1 and no generation.
    ...
```

- [ ] Call these immediately after target/profile resolution and option parsing, before automatic model preparation and runtime construction. Preserve existing Hailo, Furiosa, DeepX, Mobilint, ONNX, and HF artifact branches.
- [ ] Add `rbln`, `rebel`, and `rbln-static` to parser backend choices only to the extent aliases are already supported by registry resolution. Set `args.artifact` as the compiled artifact path; do not populate `args.onnx` or `args.model_path` from it.
- [ ] Run parser/artifact/task tests GREEN.

### Step 4: Write RED async-capacity and result-sanitization tests

- [ ] Extend `test_async_cli.py` with a fake `RblnRuntime` and RBLN target. Assert `_enable_native_async_pipeline()` injects `max_async_inflight` from effective `worker_count` (default 1, explicit 4) but leaves SDK `async_parallel` unchanged.
- [ ] Assert `_build_async_runtime_executor()` creates one `NativeAsyncRuntimeExecutor` with `max_inflight=min(worker_count, queue_capacity)` and requires `native_async_max_batch_size()==1` as an exact built-in positive integer.
- [ ] Assert `worker_count=4` passes `AsyncInferenceEngine` runtime-capability validation because runtime `max_concurrent_workers()==4`; `max_batch_size=2` still fails.
- [ ] Add hostile diagnostics values and require `_safe_runtime_diagnostics()` to emit only whitelisted primitive RBLN fields with bounded lengths:

```text
backend, device, device_id, accelerator_vendor, accelerator_name,
detected_npu, execution_mode, sdk_version,
artifact_compiler_version, artifact_npu,
tensor_parallel_size, artifact_uuid,
async_parallel, max_async_inflight
```

Large descriptor arrays and arbitrary objects must be omitted from async details even if present in `get_device_spec()`.

- [ ] Run and confirm RED:

```bash
cd framework
pytest -q tests/test_async_cli.py -k rbln
```

### Step 5: Implement async capacity selection and safe fields

- [ ] Generalize `_enable_native_async_pipeline()` while preserving Mobilint behavior:

```python
if args.inference_mode != "async_queue" or "native_async" not in target.capabilities:
    return
if target.runtime_name == "mobilint":
    runtime_kwargs["async_pipeline_enabled"] = True
if target.runtime_name == "rbln":
    runtime_kwargs["max_async_inflight"] = (
        1 if args.worker_count is None else args.worker_count
    )
```

The framework `worker_count` controls accepted concurrent submissions; `async_parallel` remains a separately supplied RBLN SDK input-preparation setting.

- [ ] Add `rbln` to `_SAFE_RUNTIME_BACKENDS` and use exact-type field allowlists: bounded identifiers for strings, non-boolean nonnegative integers for ids/limits, and finite numeric values only. Never call `str()` on an arbitrary object.
- [ ] Run focused CLI and regression suites:

```bash
cd framework
pytest -q \
  tests/test_main_paths.py \
  tests/test_async_cli.py \
  tests/test_plugin_registry.py \
  tests/test_mobilint_runtime.py \
  tests/test_mobilint_native_backend.py
```

### Step 6: Commit Task 6

- [ ] Commit CLI/result integration:

```bash
git diff --check
git add framework/src/main.py framework/tests/test_main_paths.py framework/tests/test_async_cli.py
git commit -m "feat: wire RBLN target into benchmark CLI"
```

## Task 7: SDK-free end-to-end integration and operator documentation

**Files:**

- Modify: `framework/tests/test_rbln_native_backend.py`
- Create: `framework/docs/rbln-setup.md`
- Modify: `framework/README.md`
- Modify: `framework/src/runtimes/README.md`
- Modify: `framework/CHANGELOG.md`

### Step 1: Write one complete SDK-free async integration test

- [ ] Add a test that assembles the real `RblnRuntime`, fake SDK, real `NativeAsyncRuntimeExecutor`, real async inference engine, bounded queue, minimal loader/evaluator, and two measured requests. Assert all lifecycle invariants in one place:

```text
one inspected compiled model
zero sync Runtime instances
one AsyncRuntime instance
one owner event-loop thread
warmup on that AsyncRuntime but absent from measured counters
accepted == completed == evaluated == 2
outstanding == 0
timeouts == duplicate_callbacks == late_callbacks == 0
maximum framework queue depth <= configured queue_capacity
adapter has no scheduling queue and no per-request thread
shutdown true before unload
zero fake SDK contexts after unload
```

- [ ] Add a corresponding sync smoke that proves one sync runtime and zero async runtimes for one warmup plus two measured calls.
- [ ] Run the integration tests RED/GREEN. Fix only adapter wiring exposed by these tests:

```bash
cd framework
pytest -q tests/test_rbln_native_backend.py -k 'full_lifecycle or sdk_free_e2e'
```

### Step 2: Write the operator guide with exact contracts and commands

- [ ] Create `framework/docs/rbln-setup.md` with these sections and concrete content:

  1. Scope table: static vision/BERT/PatchTST supported; Llama/vLLM, compile, multi-NPU, dynamic shape excluded.
  2. Verified starting environment: Ubuntu 22.04.5, Python 3.10.12, `rebel-compiler==0.11.0`, KMD/FW 3.2.2, RBLN-CA22 device 0, 16,877,879,296 bytes.
  3. Preflight commands: package versions, `/etc/os-release`, `rbln-smi -q`, `rbln-smi -j`, device node/permissions, and context absence.
  4. Artifact layout convention under `framework/models/rbln/{model-name}/model.rbln`, followed by the five concrete model paths used in Task 8 and the artifact inspection command.
  5. Model contract table for `resnet50`, `yolov5m`, `bert-base-uncased`, BERT SQuAD profile, and `patchtst-fm-r1`, explicitly requiring inspected shape/dtype to match the model profile.
  6. E2E smoke and full-run commands.
  7. Async offline and server-like commands.
  8. Tuning order: worker count 1/2/4/8 with `async_parallel=1`; compare 2 only at the best worker count; then queue 16/64/256; three repetitions.
  9. Metric meanings, whole-card energy caveat, and monitor coverage.
  10. Failure matrix for package, device, artifact target, tensor parallel, shape/dtype, monitor, timeout/drain, and cleanup-pending errors.
  11. Post-run verification using `rbln-smi -j` to require no leaked context.

Use this exact first smoke command from the framework directory:

```bash
python3 -m src.main \
  --model resnet50 \
  --target rbln-static \
  --artifact models/rbln/resnet50/model.rbln \
  --dataset datasets/imagenet_1k \
  --inference-mode e2e \
  --batch-size 1 \
  --warmup 2 \
  --max-steps 10 \
  --monitor \
  --results-path results/rbln-resnet50-e2e.csv
```

Use this exact initial async command:

```bash
python3 -m src.main \
  --model resnet50 \
  --target rbln-static \
  --artifact models/rbln/resnet50/model.rbln \
  --dataset datasets/imagenet_1k \
  --inference-mode async_queue \
  --scenario offline \
  --batch-size 1 \
  --worker-count 1 \
  --queue-capacity 16 \
  --min-samples 100 \
  --warmup 2 \
  --flush-timeout-sec 300 \
  --save-request-trace \
  --monitor \
  --results-path results/rbln-resnet50-async-w1.csv
```

Show SDK input preparation tuning only as `--runtime-option async_parallel=2`; do not equate it with `--worker-count`.

### Step 3: Update public support matrices and change log

- [ ] In `framework/README.md`, add `rbln-static` to the target/backend table, link the guide, list the four supported task families, and mark Llama generation as planned `rbln-vllm` rather than supported.
- [ ] In `framework/src/runtimes/README.md`, document lazy optional dependency loading, inspect-before-allocation, sync/async exclusivity, owner-loop bridge, framework queue ownership, and timeout/drain semantics.
- [ ] In `framework/CHANGELOG.md`, add an Unreleased entry for RBLN static runtime, native async, and telemetry, with explicit exclusions for compilation and vLLM.

### Step 4: Verify docs and all RBLN-focused tests

- [ ] Run:

```bash
cd framework
python3 -m src.main --help
pytest -q \
  tests/test_rbln_runtime.py \
  tests/test_rbln_native_backend.py \
  tests/test_rbln_collector.py \
  tests/test_plugin_registry.py \
  tests/test_main_paths.py \
  tests/test_async_cli.py
cd ..
git diff --check
```

Expected: help lists `rbln`/`rbln-static`; all focused tests pass; no whitespace errors.

### Step 5: Commit Task 7

- [ ] Commit integration tests and documentation:

```bash
git add \
  framework/tests/test_rbln_native_backend.py \
  framework/docs/rbln-setup.md \
  framework/README.md \
  framework/src/runtimes/README.md \
  framework/CHANGELOG.md
git commit -m "docs: add RBLN benchmark operations guide"
```

## Task 8: Full regression verification and CA22 server acceptance

**Files:**

- Verify only; modify a file only when a failing test demonstrates an RBLN defect.
- Record evidence in the implementation handoff/PR body, not in generated repository log files.

### Step 1: Run static and registry safety checks

- [ ] From the worktree root, run:

```bash
git status --short
git diff --check
python3 -m compileall -q framework/src
cd framework
python3 -c 'from runtimes import list_runtimes; from monitors import list_collectors; from core.targets import list_targets; print(len(list_runtimes()), len(list_collectors()), len(list_targets()))'
```

Expected: no syntax error; registry listing succeeds without importing or installing `rebel` on the development host.

### Step 2: Run complete focused regression suites

- [ ] Run:

```bash
cd framework
pytest -q \
  tests/test_rbln_runtime.py \
  tests/test_rbln_native_backend.py \
  tests/test_rbln_collector.py \
  tests/test_plugin_registry.py \
  tests/test_runtime_factory.py \
  tests/test_hw_monitor.py \
  tests/test_main_paths.py \
  tests/test_async_cli.py \
  tests/test_native_async_runtime_executor.py \
  tests/test_async_completion.py \
  tests/test_inference_engine.py \
  tests/test_mobilint_runtime.py \
  tests/test_mobilint_native_backend.py \
  tests/test_furiosa_native_backend.py
```

Expected: all focused tests pass. A pre-existing Furiosa flake must be rerun in isolation and reported rather than hidden.

### Step 3: Run the full suite and compare to the recorded baseline

- [ ] Run from `framework`:

```bash
pytest -q
```

Record passed/skipped/failed counts and exact failing node ids. Acceptance requires no new non-RBLN failure and no RBLN failure. Compare any Furiosa timeout or Hugging Face network failure to the recorded 1,359 passed / 13 skipped / 12 failed baseline; do not claim a clean full suite unless it is actually clean.

### Step 4: Verify the final diff and commit topology

- [ ] Run:

```bash
cd ..
git status --short
git diff --stat origin/main...HEAD
git log --oneline --decorate origin/main..HEAD
git diff --check origin/main...HEAD
```

Expected: changes are limited to the file map in this plan, task commits are reviewable, and no model binaries, result CSV/JSON, credentials, cache files, or vendor packages are tracked.

### Step 5: Run CA22 preflight on the user's NPU server

- [ ] In the deployed branch/worktree, record:

```bash
python3 --version
python3 -m pip list --format=freeze | grep -Ei '^(rebel|optimum-rbln|vllm-rbln|torch|transformers|tokenizers|vllm)'
cat /etc/os-release
rbln-smi -q
rbln-smi -j
```

Expected starting facts: Python 3.10.12, Ubuntu 22.04.5, `rebel-compiler==0.11.0`, KMD/FW 3.2.2, device 0 `RBLN-CA22`, status `normal`, 16,877,879,296 total bytes, and no context before load.

### Step 6: Inspect every installed artifact before execution

- [ ] Place artifacts at these concrete convention paths:

```text
framework/models/rbln/resnet50/model.rbln
framework/models/rbln/yolov5m/model.rbln
framework/models/rbln/bert-base-uncased/model.rbln
framework/models/rbln/bert-base-uncased-squad-v1/model.rbln
framework/models/rbln/patchtst-fm-r1/model.rbln
```

- [ ] For each file, run `RBLNCompiledModel.inspect()` and record only compiler version, target NPU, tensor-parallel size, UUID, allocation per node, and input/output name/shape/dtype. Require target `RBLN-CA22`, any explicitly reported tensor-parallel size to be 1, fixed positive dimensions, and exact shape/dtype agreement with the selected framework model profile. Permit only the SDK 0.11 single unnamed-output fallback defined above. Reject an incompatible artifact; do not cast or reshape around it.

### Step 7: Run one-model sync and async acceptance gates

- [ ] Run the exact ResNet50 E2E and initial async commands from Task 7. After each run, assert from console/result artifacts:

```text
exit code 0
correct target/runtime/device and artifact provenance
warmup completed before measurement
accepted == completed == evaluated
outstanding == 0
timeouts == duplicate_callbacks == late_callbacks == 0
monitor attempts >= monitor successes >= 1
monitor coverage present
temperature/power/utilization/memory present only when reported
runtime unload succeeds
rbln-smi contexts empty after process exit
```

- [ ] If either gate fails, keep `worker_count=1`, `async_parallel=1`, and queue 16 while diagnosing. Do not start throughput tuning with a lifecycle failure.

### Step 8: Run model-family smoke tests

- [ ] Run ten-sample E2E smoke tests for ResNet50, YOLOv5m, BERT classification, BERT SQuAD, and PatchTST with their existing datasets/tokenizer profiles. Check output keys/shapes, evaluator count, and task-specific decoding. For YOLOv5m, confirm the `.rbln` returns the raw tensor form expected by the existing decoder; do not add hidden RBLN post-processing.
- [ ] Run a 100-sample async smoke for each artifact that passes E2E. Generation profiles must remain rejected with the `rbln-vllm` guidance.

### Step 9: Tune concurrency one variable at a time

- [ ] For a stable representative model, run each setting three times:

  1. `worker_count=1,2,4,8`, queue 64, `async_parallel=1`;
  2. at the best stable worker count only, compare `async_parallel=1` and 2;
  3. at the best stable worker/parallel pair, compare queue 16, 64, and 256;
  4. use the best offline configuration in server-like mode, raising target QPS gradually.

For every run record throughput, p50/p95/p99 end-to-end and queue latency, timeout count, outstanding, duplicate/late callbacks, NPU utilization, memory, temperature, power, energy, and monitor coverage. Select a default only when exact-count invariants hold, timeout is zero, contexts clean up, and the result repeats. High utilization alone is not sufficient.

### Step 10: Final acceptance and defect policy

- [ ] The branch is implementation-complete only when SDK-free checks and the CA22 sync/async gate both pass. If the server exposes an SDK API mismatch, add the smallest compatibility shim plus a fake regression test, rerun Tasks 1–8, and commit the exact affected files with a message describing the verified mismatch. Do not make an empty “verification” commit.
