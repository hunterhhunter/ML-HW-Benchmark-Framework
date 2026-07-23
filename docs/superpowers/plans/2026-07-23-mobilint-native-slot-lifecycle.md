# Mobilint Native Async Slot Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Release Mobilint SDK execution capacity when a Future becomes terminal instead of retaining it until the framework callback returns.

**Architecture:** Keep `_slots` as the Mobilint SDK admission boundary and `_jobs` as the framework callback/shutdown boundary. Add idempotent per-job slot retirement after Future completion and output normalization, while retaining the job through callback return.

**Tech Stack:** Python 3.12, `threading`, dataclasses, pytest, NumPy, fake qb Runtime Futures

## Global Constraints

- Modify only the Mobilint native backend, its focused tests, and Mobilint runtime documentation.
- Do not change the common async request queue, completion coordinator, native executor, metrics collector, worker count, queue capacity, timeout policy, or public error contract.
- Acquire once and release exactly once for every SDK slot.
- Do not release a slot before `Future.get()` is terminal and output normalization finishes.
- Keep `_jobs[job_id]` and `job.inputs` through framework callback return so shutdown remains unload-safe.
- Keep `claim_lock` and `claimed` as the exact-once Future consumption guard.
- Preserve direct slot release when closing or synchronous `infer_async()` failure occurs before a job is registered.
- Do not add unconditional output copies; verify retained result integrity during hardware acceptance.
- Do not modify user-owned changes in other worktrees.

---

## File Structure

- Modify `framework/src/runtimes/mobilint_rt.py`: separate per-job SDK slot retirement from callback/job retirement.
- Modify `framework/tests/test_mobilint_native_backend.py`: reproduce the race and cover terminal failure, callback failure, shutdown, and pending-Future invariants.
- Modify `framework/src/runtimes/README.md`: document the two lifetimes and hardware acceptance checks.
- Preserve `docs/superpowers/specs/2026-07-23-mobilint-native-slot-lifecycle-design.md`: approved problem definition and design rationale.

### Task 1: Reproduce terminal-Future/callback slot races

**Files:**
- Modify: `framework/tests/test_mobilint_native_backend.py:174-245`

**Interfaces:**
- Consumes: `MobilintNativeBackend.submit_async(inputs, callback) -> str`
- Produces: regression requirements for SDK slot availability and callback-owned job retention

- [ ] **Step 1: Add deterministic success, Future-error, callback-error, and shutdown tests**

Add these tests after `test_waiter_capacity_is_bounded_without_an_extra_request_queue`:

```python
def test_terminal_future_releases_slot_before_callback_returns():
    callback_entered = threading.Event()
    release_callback = threading.Event()
    second_done = threading.Event()
    first = FakeFuture([np.array([[1]])])
    second = FakeFuture([np.array([[2]])])
    backend = MobilintNativeBackend(
        _runtime(FakeModel([first, second]), slots=1)
    )

    def blocked_callback(outcome):
        callback_entered.set()
        assert release_callback.wait(timeout=2.0)

    first_job = backend.submit_async(_inputs(1), blocked_callback)
    assert callback_entered.wait(timeout=1.0)
    assert first_job in backend._jobs

    second_job = backend.submit_async(
        _inputs(2), lambda outcome: second_done.set()
    )

    assert second_done.wait(timeout=1.0)
    assert first_job in backend._jobs
    release_callback.set()
    assert backend.shutdown(timeout=1.0) is True
    assert (first_job, second_job) == ("mobilint-1", "mobilint-2")
    assert first.get_calls == 1
    assert second.get_calls == 1


def test_failed_future_releases_slot_before_error_callback_returns():
    callback_entered = threading.Event()
    release_callback = threading.Event()
    second_done = threading.Event()
    first = FakeFuture(error=RuntimeError("private SDK failure"))
    second = FakeFuture([np.array([[2]])])
    backend = MobilintNativeBackend(
        _runtime(FakeModel([first, second]), slots=1)
    )
    outcomes = []

    def blocked_callback(outcome):
        outcomes.append(outcome)
        callback_entered.set()
        assert release_callback.wait(timeout=2.0)

    backend.submit_async(_inputs(1), blocked_callback)
    assert callback_entered.wait(timeout=1.0)

    backend.submit_async(_inputs(2), lambda outcome: second_done.set())

    assert second_done.wait(timeout=1.0)
    assert outcomes[0].error_type == "RuntimeError"
    assert outcomes[0].error_message == "Mobilint asynchronous inference failed."
    release_callback.set()
    assert backend.shutdown(timeout=1.0) is True


def test_callback_failure_does_not_delay_next_sdk_submission():
    callback_entered = threading.Event()
    release_callback = threading.Event()
    second_done = threading.Event()
    first = FakeFuture([np.array([[1]])])
    second = FakeFuture([np.array([[2]])])
    backend = MobilintNativeBackend(
        _runtime(FakeModel([first, second]), slots=1)
    )

    def failing_callback(outcome):
        callback_entered.set()
        assert release_callback.wait(timeout=2.0)
        raise RuntimeError("consumer failed")

    backend.submit_async(_inputs(1), failing_callback)
    assert callback_entered.wait(timeout=1.0)

    backend.submit_async(_inputs(2), lambda outcome: second_done.set())

    assert second_done.wait(timeout=1.0)
    release_callback.set()
    assert backend.shutdown(timeout=1.0) is True
    assert first.get_calls == 1
    assert second.get_calls == 1


def test_shutdown_waits_for_callback_after_sdk_slot_is_available():
    callback_entered = threading.Event()
    release_callback = threading.Event()
    backend = MobilintNativeBackend(
        _runtime(
            FakeModel([FakeFuture([np.array([[1]])])]),
            slots=1,
        )
    )

    def blocked_callback(outcome):
        callback_entered.set()
        assert release_callback.wait(timeout=2.0)

    backend.submit_async(_inputs(), blocked_callback)
    assert callback_entered.wait(timeout=1.0)
    assert backend._slots.acquire(blocking=False) is True
    backend._slots.release()

    assert backend.shutdown(timeout=0.01) is False
    release_callback.set()
    assert backend.shutdown(timeout=1.0) is True
```

- [ ] **Step 2: Run the focused tests and verify RED plus the shutdown characterization guard**

Run:

```bash
UV_CACHE_DIR=/tmp/codex-mobilint-slot-uv-cache uv run --with pytest --with numpy \
  pytest framework/tests/test_mobilint_native_backend.py \
  -k 'terminal_future_releases_slot or failed_future_releases_slot or callback_failure_does_not_delay or shutdown_waits_for_callback_after' -vv
```

Expected: the first three tests fail with `Mobilint native async waiter capacity is exhausted`; the shutdown test fails at the direct `_slots.acquire(blocking=False)` assertion. These failures prove the old implementation retains SDK capacity through callback execution.

- [ ] **Step 3: Commit the RED tests**

```bash
git add framework/tests/test_mobilint_native_backend.py
git commit -m "test: reproduce Mobilint callback slot race"
```

### Task 2: Split SDK slot retirement from callback job retirement

**Files:**
- Modify: `framework/src/runtimes/mobilint_rt.py:22-202`
- Test: `framework/tests/test_mobilint_native_backend.py`

**Interfaces:**
- Produces: `_MobilintAsyncJob.slot_released: bool`
- Produces: `MobilintNativeBackend._release_job_slot(job: _MobilintAsyncJob) -> bool`
- Preserves: `MobilintNativeBackend.submit_async(inputs, callback) -> str`

- [ ] **Step 1: Add explicit per-job slot state**

Extend `_MobilintAsyncJob`:

```python
@dataclass
class _MobilintAsyncJob:
    future: Any
    inputs: list[np.ndarray]
    thread: threading.Thread | None = None
    claim_lock: Any = field(default_factory=threading.Lock, repr=False)
    claimed: bool = False
    slot_released: bool = False
```

- [ ] **Step 2: Add idempotent SDK slot retirement**

Add this method after `_thread_is_alive`:

```python
    def _release_job_slot(self, job: _MobilintAsyncJob) -> bool:
        with self._condition:
            if job.slot_released:
                return False
            job.slot_released = True
            self._slots.release()
            self._condition.notify_all()
            return True
```

- [ ] **Step 3: Release terminal SDK capacity before callback and retain job cleanup after callback**

Replace `_wait_for_job` with:

```python
    def _wait_for_job(
        self,
        job_id: str,
        job: _MobilintAsyncJob,
        callback: Callable[[NativeAsyncOutcome], None],
    ) -> None:
        with job.claim_lock:
            if job.claimed:
                return
            job.claimed = True
        started_ns = time.perf_counter_ns()
        try:
            try:
                outputs = self.runtime._normalize_outputs(job.future.get())
                outcome = NativeAsyncOutcome(
                    outputs=outputs,
                    timing_ms=(time.perf_counter_ns() - started_ns)
                    / 1_000_000.0,
                )
            except BaseException as exc:
                outcome = NativeAsyncOutcome(
                    error_type=self._error_type(exc),
                    error_message="Mobilint asynchronous inference failed.",
                )
            finally:
                self._release_job_slot(job)
            try:
                callback(outcome)
            except BaseException:
                # Consumer failures must not strand accepted SDK work during unload.
                pass
        finally:
            with self._condition:
                self._jobs.pop(job_id, None)
                job.inputs = []
                self._condition.notify_all()
```

The outer finalizer keeps job cleanup correct even if terminal outcome creation
itself raises. Do not call `_slots.release()` from the job cleanup finalizer.

- [ ] **Step 4: Run the focused race tests and verify GREEN**

Run the Task 1 Step 2 command again.

Expected: `4 passed`.

- [ ] **Step 5: Run the complete Mobilint native backend suite**

Run:

```bash
UV_CACHE_DIR=/tmp/codex-mobilint-slot-uv-cache uv run --with pytest --with numpy \
  pytest framework/tests/test_mobilint_native_backend.py -q
```

Expected: `19 passed` with no thread exception warnings.

- [ ] **Step 6: Commit the implementation**

```bash
git add framework/src/runtimes/mobilint_rt.py
git commit -m "fix: separate Mobilint SDK slot lifetime"
```

### Task 3: Document ownership and verify adjacent native-async behavior

**Files:**
- Modify: `framework/src/runtimes/README.md:52-74`
- Verify: `framework/tests/test_native_async_runtime_executor.py`
- Verify: `framework/tests/test_mobilint_runtime.py`

**Interfaces:**
- Documents: SDK slot lifetime, framework job lifetime, and hardware acceptance evidence

- [ ] **Step 1: Extend the Mobilint async runtime documentation**

Add this paragraph to the end of `### 비동기 실행 구조`:

```markdown
Mobilint adapter의 SDK 실행 slot과 framework callback job은 수명이 다릅니다.
SDK slot은 `Future.get()`이 terminal이 되고 raw output 정규화가 끝난 직후
반환합니다. `_jobs`와 input 참조는 framework callback이 완전히 반환될 때까지
유지하므로, 다음 SDK 요청은 callback 반환 지연과 무관하게 제출할 수 있지만
shutdown과 model dispose는 callback/waiter 종료 전에는 완료되지 않습니다. Slot
반환은 job별 exact-once 상태로 보호하며, pending Future가 있는 동안에는 기존의
nonblocking capacity 제한을 유지합니다.
```

Add this item under `### 실제 하드웨어 인수 점검`:

```markdown
- native async를 1,000건과 3,000건 연속 실행해
  `submitted == accepted == completed`,
  `failed == timed_out == outstanding == 0`을 확인하고, 다음 추론 중에도 보관한
  직전 output의 hash가 변하지 않는지 검증합니다. 수정 전후 QPS와 p95/p99도 함께
  기록합니다.
```

- [ ] **Step 2: Run the adjacent CPU regression suites**

Run:

```bash
UV_CACHE_DIR=/tmp/codex-mobilint-slot-uv-cache uv run --with pytest --with numpy \
  pytest framework/tests/test_mobilint_native_backend.py \
  framework/tests/test_native_async_runtime_executor.py \
  framework/tests/test_mobilint_runtime.py -q
```

Expected: all selected tests pass with zero failures and zero thread exception warnings.

- [ ] **Step 3: Run syntax compilation for changed Python files**

Run:

```bash
UV_CACHE_DIR=/tmp/codex-mobilint-slot-uv-cache uv run --with numpy \
  python -m py_compile framework/src/runtimes/mobilint_rt.py \
  framework/tests/test_mobilint_native_backend.py
```

Expected: exit code `0` with no output.

- [ ] **Step 4: Review the final diff and ownership invariants**

Run:

```bash
git diff origin/main...HEAD --check
git diff origin/main...HEAD -- framework/src/runtimes/mobilint_rt.py \
  framework/tests/test_mobilint_native_backend.py \
  framework/src/runtimes/README.md \
  docs/superpowers/specs/2026-07-23-mobilint-native-slot-lifecycle-design.md \
  docs/superpowers/plans/2026-07-23-mobilint-native-slot-lifecycle.md
```

Expected: no whitespace errors; the slot moves before callback, `_jobs` cleanup remains after callback, and no unrelated subsystem changes appear.

- [ ] **Step 5: Commit runtime documentation**

```bash
git add framework/src/runtimes/README.md
git commit -m "docs: explain Mobilint async slot ownership"
```

## Hardware Acceptance Command Shape

Run the repository's Mobilint native-async benchmark twice with identical
model, dataset, device, worker, queue, timeout, and monitor options, changing
only the request count from 1,000 to 3,000. Store the exact commands and JSON
reports with SDK, driver, firmware, device, artifact SHA-256, QPS, p95, p99,
submitted, accepted, completed, failed, timed_out, and outstanding fields.

Hardware is required for release acceptance but is not available in the CPU
unit-test environment. Do not describe the hardware criterion as passed until
those reports exist.
