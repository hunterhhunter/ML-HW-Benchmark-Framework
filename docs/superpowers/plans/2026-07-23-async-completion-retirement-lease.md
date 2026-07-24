# Async Completion Retirement Lease Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent the second native-async request from stalling behind an acknowledged-but-unretired first dispatch by transferring a one-shot retirement capability to the completion thread.

**Architecture:** `AsyncInferenceEngine` creates a private `_RetirementLease` for each normal worker completion and publishes it with the coordinator handoff. `CompletionCoordinator` invokes the generic lease after terminal state is committed and the handoff is marked `ACKED`; the lease ACKs the runtime execution and releases native capacity. Existing worker/flush paths continue retiring dequeue and coordinator handoff journals so queue recovery remains in its original fault domain.

**Tech Stack:** Python 3.12, `threading`, dataclasses, pytest, NumPy

## Global Constraints

- Keep request admission, batching, workers, cancellation, flush, and shutdown in `AsyncInferenceEngine`.
- Keep terminal membership, decoder/evaluator execution, metrics, and trace publication in `CompletionCoordinator`.
- `CompletionCoordinator` must not import `RuntimeExecutor`, Mobilint, or request-queue implementation types.
- Do not replace the framework request queue or make `RuntimeExecutor.execute()` nonblocking.
- Preserve the native buffer lifetime rule: physical SDK completion and logical framework acknowledgement are both required before permit release.
- Preserve inline/e2e completion behavior and all vendor backend interfaces.
- Do not modify the unrelated user-owned `.superpowers/sdd/task-5-report.md` worktree change.

---

## Execution Correction

The initial Task 3 draft moved both runtime ACK and dequeue retirement into the
completion lease. The full engine suite exposed 28 queue fault-domain and
recovery regressions. The verified implementation therefore narrows the lease
to runtime execution ACK, restores worker-local pending handoff retirement, and
serializes both paths with the existing handoff retirement lock. This preserves
the two-request deadlock fix while keeping queue recovery semantics unchanged.

## File Structure

- Modify `framework/src/core/async_inference/engine.py`: define the one-shot lease, create one for each successful worker handoff, and bind it to the existing retirement operation.
- Modify `framework/src/core/async_inference/completion.py`: accept an optional generic lease on journaled completion handoffs and invoke it after terminal commit.
- Modify `framework/tests/test_async_engine.py`: verify lease exact-once and failure semantics.
- Modify `framework/tests/test_async_completion.py`: verify coordinator ordering and generic lease invocation outside its condition lock.
- Modify `framework/tests/test_native_async_runtime_executor.py`: reproduce and prevent the two-request, one-worker, one-native-slot stall.
- Modify `framework/src/core/README.md`: document terminal retirement capability ownership.

### Task 1: One-shot retirement primitive

**Files:**
- Modify: `framework/tests/test_async_engine.py:1-20`
- Modify: `framework/src/core/async_inference/engine.py:25-70`

**Interfaces:**
- Produces: `_RetirementLease(callback: Callable[[], None])`
- Produces: `_RetirementLease.retire() -> bool`
- Produces: `_RetirementLease.state -> str`

- [ ] **Step 1: Write failing exact-once and failure tests**

Add `_RetirementLease` to the engine test import and add these tests near the other engine helper tests:

```python
from core.async_inference.engine import (
    AsyncInferenceEngine,
    _RequestQueue,
    _RetirementLease,
)


def test_retirement_lease_serializes_racing_retire_calls_exactly_once():
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def retire_callback():
        calls.append(threading.get_ident())
        entered.set()
        assert release.wait(timeout=2.0)

    lease = _RetirementLease(retire_callback)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(lease.retire)
        assert entered.wait(timeout=1.0)
        second = pool.submit(lease.retire)
        release.set()
        assert first.result(timeout=1.0) is True
        assert second.result(timeout=1.0) is True

    assert len(calls) == 1
    assert lease.state == "RETIRED"


def test_retirement_lease_failure_is_stable_and_not_reexecuted():
    failure = RuntimeError("planned retirement failure")
    calls = []

    def fail_retirement():
        calls.append(True)
        raise failure

    lease = _RetirementLease(fail_retirement)
    with pytest.raises(RuntimeError) as first:
        lease.retire()
    with pytest.raises(RuntimeError) as second:
        lease.retire()

    assert first.value is failure
    assert second.value is failure
    assert calls == [True]
    assert lease.state == "FAILED"
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run pytest framework/tests/test_async_engine.py \
  -k 'retirement_lease_' -vv
```

Expected: collection fails because `_RetirementLease` is not defined.

- [ ] **Step 3: Implement `_RetirementLease`**

Add this private primitive before `_SlotLeasePool` in `engine.py`:

```python
class _RetirementLease:
    def __init__(self, callback):
        if not callable(callback):
            raise ValueError("retirement callback must be callable")
        self._callback = callback
        self._condition = threading.Condition()
        self._state = "PENDING"
        self._error = None

    @property
    def state(self):
        with self._condition:
            return self._state

    def retire(self) -> bool:
        with self._condition:
            while self._state == "RETIRING":
                self._condition.wait()
            if self._state == "RETIRED":
                return True
            if self._state == "FAILED":
                raise self._error
            self._state = "RETIRING"
            callback = self._callback

        try:
            callback()
        except BaseException as exc:
            with self._condition:
                self._callback = None
                self._error = exc
                self._state = "FAILED"
                self._condition.notify_all()
            raise

        with self._condition:
            self._callback = None
            self._state = "RETIRED"
            self._condition.notify_all()
        return True
```

- [ ] **Step 4: Run tests and verify GREEN**

Run the Step 2 command again. Expected: `2 passed`.

- [ ] **Step 5: Commit the primitive**

```bash
git add framework/src/core/async_inference/engine.py \
  framework/tests/test_async_engine.py
git commit -m "feat: add one-shot async retirement lease"
```

### Task 2: Completion coordinator lease handoff

**Files:**
- Modify: `framework/tests/test_async_completion.py:1960-2120`
- Modify: `framework/src/core/async_inference/completion.py:45-70`
- Modify: `framework/src/core/async_inference/completion.py:580-675`
- Modify: `framework/src/core/async_inference/completion.py:947-985`

**Interfaces:**
- Consumes: any internal object exposing callable `retire() -> bool`
- Produces: `CompletionCoordinator.submit(completion, timeout=None, *, operation_key=None, retirement_lease=None) -> None`

- [ ] **Step 1: Write the failing coordinator ordering test**

Add:

```python
def test_completion_handoff_retires_lease_after_terminal_commit_outside_lock():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )
    req = request(211)
    operation_key = object()
    observations = []
    retired = threading.Event()

    class Lease:
        def retire(self):
            observations.append(
                (
                    coordinator.condition._is_owned(),
                    coordinator.completion_handoff_state(operation_key),
                    req.request_id in coordinator.outstanding,
                )
            )
            retired.set()
            return True

    lease = Lease()
    coordinator.register(req)
    coordinator.start()
    coordinator.submit(
        completion(req),
        timeout=1.0,
        operation_key=operation_key,
        retirement_lease=lease,
    )

    assert retired.wait(timeout=1.0)
    assert observations == [(False, "ACKED", False)]
    assert coordinator.acknowledge_completion_handoff(operation_key) is True
    assert coordinator.stop(timeout=1.0) is True
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
uv run pytest framework/tests/test_async_completion.py::test_completion_handoff_retires_lease_after_terminal_commit_outside_lock -vv
```

Expected: FAIL because `submit()` does not accept `retirement_lease`.

- [ ] **Step 3: Store and validate the generic lease**

Extend `_CompletionHandoff`:

```python
@dataclass
class _CompletionHandoff:
    completion: BatchCompletion
    queued: object
    retirement_lease: object | None = None
    state: str = "ENQUEUING"
    producer_active: bool = False
```

Extend `submit()` and its creation/retry checks:

```python
def submit(
    self,
    completion: BatchCompletion,
    timeout: float | None = None,
    *,
    operation_key=None,
    retirement_lease=None,
) -> None:
    if retirement_lease is not None and not callable(
        getattr(retirement_lease, "retire", None)
    ):
        raise ValueError("retirement_lease must provide callable retire()")
    if self.queue is None:
        if operation_key is not None or retirement_lease is not None:
            raise ValueError(
                "operation_key and retirement_lease are not supported "
                "by inline completion"
            )
        # existing inline path
```

When allocating a handoff, pass `retirement_lease`; when retrying an existing
operation key, reject a different lease object with
`RuntimeError("completion handoff retirement ownership changed")`.

- [ ] **Step 4: Invoke the lease after ACK commit and outside the lock**

In `_run()`, immediately after `_mark_completion_handoff_acked_locked()` and
after leaving `with self.condition`, invoke:

```python
retirement_lease = None
with self.condition:
    self._mark_completion_handoff_acked_locked(
        queued_handoff.operation_key
    )
    handoff = self._completion_handoffs.get(
        queued_handoff.operation_key
    )
    if handoff is not None:
        retirement_lease = handoff.retirement_lease
if retirement_lease is not None:
    retirement_lease.retire()
```

Keep `handoff_ack_callback` invocation after lease retirement so legacy deferred
recovery observes already-retired normal handoffs.

- [ ] **Step 5: Run coordinator handoff tests**

Run:

```bash
uv run pytest framework/tests/test_async_completion.py \
  -k 'handoff or completion_thread_failure' -vv
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit coordinator support**

```bash
git add framework/src/core/async_inference/completion.py \
  framework/tests/test_async_completion.py
git commit -m "feat: retire completion leases after terminal commit"
```

### Task 3: Transfer normal worker retirement to the lease

**Files:**
- Modify: `framework/tests/test_native_async_runtime_executor.py:1560-1600`
- Modify: `framework/src/core/async_inference/engine.py:3119-3185`
- Modify: `framework/src/core/async_inference/engine.py:3585-3820`

**Interfaces:**
- Consumes: `_RetirementLease`
- Consumes: `CompletionCoordinator.submit(..., retirement_lease=lease)`
- Produces: `_retire_completion_lease(operation_key) -> None`

- [ ] **Step 1: Write the two-request regression test**

Add before the existing reverse-completion integration test:

```python
def test_native_executor_real_queue_retires_first_slot_before_second_dispatch():
    backend = FakeNativeBackend()
    engine, executor, evaluator, metrics, traces = build_native_engine(
        backend,
        worker_count=1,
        max_inflight=1,
    )
    engine.start()
    assert engine.submit(make_request(0), block=True) is True
    assert engine.submit(make_request(1), block=True) is True

    first_job = backend.wait_for_jobs(1)[0]
    first_inputs = backend.inputs_for(first_job)
    backend.complete(
        first_job,
        NativeAsyncOutcome(
            outputs={"output": first_inputs["input"] * 10},
            timing_ms=1.0,
        ),
    )

    second_job = backend.wait_for_jobs(2, timeout=0.5)[1]
    second_inputs = backend.inputs_for(second_job)
    backend.complete(
        second_job,
        NativeAsyncOutcome(
            outputs={"output": second_inputs["input"] * 10},
            timing_ms=1.0,
        ),
    )

    observed = traces.wait_for(2)
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True

    assert [trace.request_id for trace in observed] == [0, 1]
    assert evaluator.pairs == [(0.0, 0.0), (10.0, 1.0)]
    assert executor.snapshot().inflight == 0
    assert_accounting(metrics, completed=2, failed=0)
```

- [ ] **Step 2: Run the regression and verify RED**

Run:

```bash
uv run pytest framework/tests/test_native_async_runtime_executor.py::test_native_executor_real_queue_retires_first_slot_before_second_dispatch -vv
```

Expected: FAIL in `backend.wait_for_jobs(2)` because the first dispatch still
owns the sole permit.

- [ ] **Step 3: Pass the lease through completion submission**

Extend `_submit_completion_handoff()`:

```python
def _submit_completion_handoff(
    self,
    completion,
    operation_key,
    timeout: float,
    *,
    wait_for_ack: bool = True,
    retirement_lease=None,
) -> None:
    # existing deadline logic
    self.coordinator.submit(
        completion,
        timeout=max(0.0, deadline - time.monotonic()),
        operation_key=operation_key,
        retirement_lease=retirement_lease,
    )
```

Add the focused adapter to the existing retirement journal:

```python
def _retire_completion_lease(self, operation_key) -> None:
    remaining = self._retire_worker_handoffs(
        [operation_key],
        deadline=time.monotonic(),
    )
    if remaining:
        raise RuntimeError("completion lease retirement is incomplete")
```

- [ ] **Step 4: Create and transfer the lease in the normal worker path**

After `_register_worker_local_handoff(completion_operation_key)`, create:

```python
retirement_lease = _RetirementLease(
    lambda operation_key=completion_operation_key: (
        self._retire_completion_lease(operation_key)
    )
)
```

Pass it to `_submit_completion_handoff(..., wait_for_ack=False,
retirement_lease=retirement_lease)`. Remove the successful-path
`pending_handoffs.append(...)` and immediate `_retire_worker_handoffs(...)`
calls; retain the existing pending/deferred structures for exceptional recovery.

- [ ] **Step 5: Run the regression and verify GREEN**

Run the Step 2 command again. Expected: `1 passed` and completion under one
second, with two completed requests and zero native inflight dispatches.

- [ ] **Step 6: Run engine retirement and native integration tests**

Run:

```bash
uv run pytest framework/tests/test_async_engine.py \
  framework/tests/test_native_async_runtime_executor.py -vv
```

Expected: all tests pass. If a recovery test asserts the old successful-path
polling detail, update only that assertion to verify exact-once retirement and
zero outstanding operations; do not weaken failure-path assertions.

- [ ] **Step 7: Commit worker integration**

```bash
git add framework/src/core/async_inference/engine.py \
  framework/tests/test_native_async_runtime_executor.py
git commit -m "fix: retire native dispatches from completion terminal"
```

### Task 4: Failure contract, documentation, and verification

**Files:**
- Modify: `framework/tests/test_async_completion.py`
- Modify: `framework/src/core/README.md:70-85`

**Interfaces:**
- Verifies: lease failures make coordinator shutdown fail without executing the lease twice.
- Documents: terminal completion owns invocation of the generic retirement capability.

- [ ] **Step 1: Add a failing-retirement coordinator test**

Add:

```python
def test_completion_retirement_failure_fails_coordinator_without_retry():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )
    req = request(212)
    operation_key = object()

    class FailingLease:
        def __init__(self):
            self.calls = 0

        def retire(self):
            self.calls += 1
            raise RuntimeError("planned lease retirement failure")

    lease = FailingLease()
    coordinator.register(req)
    coordinator.start()
    coordinator.submit(
        completion(req),
        timeout=1.0,
        operation_key=operation_key,
        retirement_lease=lease,
    )

    coordinator.thread.join(timeout=1.0)
    assert not coordinator.thread.is_alive()
    assert "planned lease retirement failure" in coordinator.thread_error
    assert lease.calls == 1
    assert coordinator.stop(timeout=1.0) is False
```

- [ ] **Step 2: Run the failure test**

Run:

```bash
uv run pytest framework/tests/test_async_completion.py::test_completion_retirement_failure_fails_coordinator_without_retry -vv
```

Expected: PASS with the Task 2 implementation. If it fails, adjust only the
coordinator error propagation so the original retirement exception reaches the
existing outer `_run()` failure handler; never retry the lease in coordinator.

- [ ] **Step 3: Document lease ownership**

Add this paragraph to the native async section in `framework/src/core/README.md`:

```markdown
Queued completion은 terminal commit 뒤 generic retirement lease를 실행합니다.
lease는 completion coordinator가 runtime이나 vendor type을 알지 않고도 dequeue
소유권과 `RuntimeExecutor` ACK를 exact-once로 정리하는 capability입니다. 정상
worker 경로의 native permit은 이 terminal retirement에서 반환되고, engine의
handoff journal은 publication 또는 cleanup 실패 복구용으로 유지됩니다.
```

- [ ] **Step 4: Run focused verification**

```bash
uv run pytest framework/tests/test_async_completion.py \
  framework/tests/test_async_engine.py \
  framework/tests/test_native_async_runtime_executor.py \
  framework/tests/test_mobilint_native_backend.py -q
```

Expected: zero failures and zero leaked `async-*` test threads.

- [ ] **Step 5: Run broader async regression verification**

```bash
uv run pytest framework/tests/test_async_runner.py \
  framework/tests/test_async_cli.py \
  framework/tests/test_async_onnx_cpu.py \
  framework/tests/test_runtime_executor.py -q
```

Expected: zero failures.

- [ ] **Step 6: Run static checks and inspect scope**

```bash
uv run python -m compileall -q framework/src/core/async_inference \
  framework/src/core/runtime_executor.py
git diff --check HEAD~3..HEAD
git status --short
```

Expected: compile and diff checks exit 0. Status contains only the pre-existing
`.superpowers/sdd/task-5-report.md` modification.

- [ ] **Step 7: Commit documentation and failure contract**

```bash
git add framework/src/core/README.md \
  framework/tests/test_async_completion.py
git commit -m "test: cover async retirement failure contract"
```

- [ ] **Step 8: Prepare Aries server validation command**

Provide the existing one-worker command with `--max-samples 2`,
`--queue-capacity 1`, `--worker-count 1`, `--flush-timeout-sec 15`, and
`--request-timeout-ms 5000`. Acceptance is two completed requests, zero failed
or timed-out requests, zero outstanding requests, and `async_run_status=valid`.
