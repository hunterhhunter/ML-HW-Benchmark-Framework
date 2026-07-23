# Async Metrics Incremental Accounting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the O(N^3) full-history rebuild from normal async metrics accounting while preserving recovery and result invariants.

**Architecture:** Keep `outcomes` and `terminal_times` as authoritative evidence. Maintain accepted/rejected counters, queue transitions, and inflight state incrementally under the existing sealed lock, mark interrupted projections dirty, and rebuild canonically only during exceptional recovery or finalization.

**Tech Stack:** Python 3.12, pytest 9, NumPy 2.4, existing `core.async_inference` sealed accounting primitives.

## Global Constraints

- Scope is P0 metrics scalability only.
- Do not change request queue, worker, batching, completion, native dispatch, vendor runtime, activation-slot, or flush behavior.
- Preserve `outcomes` attempt-token authority and accepted/rejected idempotency.
- Preserve queue sequence evidence, time-weighted inflight metrics, exact-once terminal, timeout, late-completion, dispatch-ACK, and unload contracts.
- Normal acceptance, rejection, and first terminal inflight updates must be O(1).
- Full reconstruction is allowed only for dirty recovery and `finalize()`, and must be O(N log N) or better.
- Do not add wall-time thresholds to CI tests.
- Do not modify or stage `.superpowers/sdd/task-5-report.md`.

---

### Task 1: Lock the hot-path and interruption contracts with RED tests

**Files:**
- Modify: `framework/tests/test_async_metrics.py`
- Modify: `framework/tests/test_async_engine.py:3690`
- Test: `framework/tests/test_async_metrics.py`
- Test: `framework/tests/test_async_engine.py`

**Interfaces:**
- Consumes: existing `_commit_acceptance_internal`, `_record_rejected_internal`, `AsyncMetricsCollector.record_terminal`, and `_resolve_accounting_internal`.
- Produces: structural tests for zero hot-path rebuilds and recovery tests for `_apply_accepted_outcome_locked`, `_apply_rejected_outcome_locked`, and `_apply_terminal_inflight_locked`.

- [ ] **Step 1: Add imports for synthetic queue transitions and sealed primitives**

Add to `framework/tests/test_async_metrics.py`:

```python
from types import SimpleNamespace

from core.async_inference.metrics import (
    AsyncMetricsCollector,
    _SEALED_ACCOUNTING_REGISTRY,
    _commit_acceptance_internal,
    _record_queue_sequence_allocated,
    _record_rejected_internal,
)
```

- [ ] **Step 2: Add the structural 3,000-request test**

Add after `test_outcome_identity_is_normalized_before_sealed_lock`:

```python
def test_outcome_accounting_hot_path_rebuilds_only_at_finalize(monkeypatch):
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    original = metrics_module._rebuild_outcome_accounting_locked
    rebuild_calls = 0
    rebuild_allowed = False

    def guarded_rebuild(state):
        nonlocal rebuild_calls
        rebuild_calls += 1
        assert rebuild_allowed is True
        return original(state)

    monkeypatch.setattr(
        metrics_module,
        "_rebuild_outcome_accounting_locked",
        guarded_rebuild,
    )

    for index in range(2_000):
        observed_ns = (index + 1) * 10
        metrics.record_submitted()
        _commit_acceptance_internal(
            metrics,
            now_ns=observed_ns,
            queue_depth=1,
            queue_transition=SimpleNamespace(
                sequence=index + 1,
                depth=1,
                now_ns=observed_ns,
            ),
            attempt_token=index,
            request_id=index,
        )
        metrics.record_terminal(
            make_trace(
                index,
                observed_ns,
                observed_ns + 1,
                observed_ns + 2,
                observed_ns + 3,
            )
        )

    for index in range(2_000, 3_000):
        metrics.record_submitted()
        _record_rejected_internal(
            metrics,
            "queue_full",
            attempt_token=index,
            request_id=index,
        )

    assert rebuild_calls == 0
    rebuild_allowed = True
    result = metrics.finalize(end_ns=30_010)

    assert rebuild_calls == 1
    assert result["summary"]["async_accepted_requests"] == 2_000
    assert result["summary"]["async_completed_requests"] == 2_000
    assert result["summary"]["async_rejected_requests"] == 1_000
    assert result["summary"]["async_outstanding_requests"] == 0
    assert result["details"]["counter_invariants"]["valid"] is True
```

- [ ] **Step 3: Add dirty projection recovery tests**

Add to `framework/tests/test_async_metrics.py`:

```python
@pytest.mark.parametrize("fault_timing", ["before", "after"])
@pytest.mark.parametrize(
    ("helper_name", "operation", "expected_summary_key"),
    [
        (
            "_apply_accepted_outcome_locked",
            "accepted",
            "async_accepted_requests",
        ),
        (
            "_apply_rejected_outcome_locked",
            "rejected",
            "async_rejected_requests",
        ),
        (
            "_apply_terminal_inflight_locked",
            "terminal",
            "async_completed_requests",
        ),
    ],
)
def test_dirty_outcome_projection_recovers_from_incremental_fault(
    monkeypatch,
    fault_timing,
    helper_name,
    operation,
    expected_summary_key,
):
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    state = metrics_module._sealed_accounting(metrics)
    original = getattr(metrics_module, helper_name)

    if operation == "terminal":
        metrics.record_submitted()
        _commit_acceptance_internal(
            metrics,
            now_ns=1,
            queue_depth=1,
            queue_transition=SimpleNamespace(sequence=1, depth=1, now_ns=1),
            attempt_token=1,
            request_id=1,
        )
    else:
        metrics.record_submitted()

    def interrupt(*args):
        if fault_timing == "before":
            raise KeyboardInterrupt("before incremental projection")
        original(*args)
        raise KeyboardInterrupt("after incremental projection")

    monkeypatch.setattr(metrics_module, helper_name, interrupt)

    with pytest.raises(KeyboardInterrupt, match="incremental projection"):
        if operation == "accepted":
            _commit_acceptance_internal(
                metrics,
                now_ns=1,
                queue_depth=1,
                queue_transition=SimpleNamespace(
                    sequence=1,
                    depth=1,
                    now_ns=1,
                ),
                attempt_token=1,
                request_id=1,
            )
        elif operation == "rejected":
            _record_rejected_internal(
                metrics,
                "queue_full",
                attempt_token=1,
                request_id=1,
            )
        else:
            metrics.record_terminal(make_trace(1, 1, 2, 3, 4))

    assert state.outcome_accounting_dirty is True
    monkeypatch.setattr(metrics_module, helper_name, original)
    metrics_module._resolve_accounting_internal(metrics)
    assert state.outcome_accounting_dirty is False

    result = metrics.finalize(end_ns=10)
    assert result["summary"][expected_summary_key] == 1
    assert result["details"]["counter_invariants"]["valid"] is True
```

- [ ] **Step 4: Move the engine recovery injection to the incremental projection**

In `framework/tests/test_async_engine.py`, rename
`test_rejection_outcome_rebuild_restores_reason_and_evidence` to
`test_rejection_outcome_projection_recovers_reason_and_evidence`. Replace the
monkeypatch target and wrapper with:

```python
    original = metrics_module._apply_rejected_outcome_locked
    injected = False

    def interrupt(state, record):
        nonlocal injected
        if not injected and fault_timing == "before":
            injected = True
            raise WorkerAbort("before outcome projection")
        original(state, record)
        if not injected and fault_timing == "after":
            injected = True
            raise WorkerAbort("after outcome projection")

    monkeypatch.setattr(
        metrics_module,
        "_apply_rejected_outcome_locked",
        interrupt,
    )
```

- [ ] **Step 5: Run the RED tests**

Run:

```bash
PYTHONPATH=framework/src \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest \
  framework/tests/test_async_metrics.py::test_outcome_accounting_hot_path_rebuilds_only_at_finalize \
  framework/tests/test_async_metrics.py::test_dirty_outcome_projection_recovers_from_incremental_fault \
  framework/tests/test_async_engine.py::test_rejection_outcome_projection_recovers_reason_and_evidence \
  -q
```

Expected: FAIL because the hot path calls `_rebuild_outcome_accounting_locked`
and the three `_apply_*` helpers and dirty flag do not exist.

---

### Task 2: Implement O(1) projections and canonical dirty recovery

**Files:**
- Modify: `framework/src/core/async_inference/metrics.py:196-288`
- Modify: `framework/src/core/async_inference/metrics.py:539-764`
- Modify: `framework/src/core/async_inference/metrics.py:819-840`
- Modify: `framework/src/core/async_inference/metrics.py:1078-1160`
- Test: `framework/tests/test_async_metrics.py`
- Test: `framework/tests/test_async_engine.py`

**Interfaces:**
- Consumes: sealed state lock, outcome journal records, terminal timestamp evidence, `_record_queue_depth_event_locked`, `_update_inflight_locked`.
- Produces: `state.outcome_accounting_dirty`, `_apply_accepted_outcome_locked(state, record)`, `_apply_rejected_outcome_locked(state, record)`, `_apply_terminal_inflight_locked(state, completed_ns)`, and conditional `_resolve_accounting_internal(metrics)`.

- [ ] **Step 1: Add and initialize the sealed dirty flag**

Add `"outcome_accounting_dirty"` to `_SealedAccountingState.__slots__`, and
initialize it immediately before `outcomes`:

```python
        self.outcome_accounting_dirty = False
        self.outcomes = {}
```

Reset it in `_try_begin_measurement()`:

```python
            state.outcome_accounting_dirty = False
```

- [ ] **Step 2: Add the three O(1) projection helpers**

Add immediately before `_rebuild_outcome_accounting_locked`:

```python
def _apply_accepted_outcome_locked(state, record) -> None:
    (
        _request_id,
        now_ns,
        queue_depth,
        sequence,
        depth,
        observed_ns,
    ) = record
    _increment(state.counters, "accepted")
    if sequence is None:
        state.legacy_queue_events += 1
        _update_queue_depth_locked(state, queue_depth, now_ns)
    else:
        _record_queue_depth_event_locked(
            state,
            depth,
            observed_ns,
            sequence,
        )
    _update_inflight_locked(
        state,
        state.inflight_value + 1,
        now_ns,
    )


def _apply_rejected_outcome_locked(state, record) -> None:
    _request_id, reason, evidence = record
    _increment(state.counters, "rejected")
    _increment(state.counters, f"rejected:{reason}")
    state.invalid_reasons.add(evidence)


def _apply_terminal_inflight_locked(state, completed_ns: int) -> None:
    _update_inflight_locked(
        state,
        state.inflight_value - 1,
        completed_ns,
    )
```

- [ ] **Step 3: Replace the quadratic rebuild with local canonical reconstruction**

Replace `_rebuild_outcome_accounting_locked` with:

```python
def _rebuild_outcome_accounting_locked(state) -> None:
    accepted = []
    rejected = []
    for kind, record in state.outcomes.values():
        if kind == "accepted":
            accepted.append(record)
        else:
            rejected.append(record)

    counters = {
        key: value
        for key, value in state.counters.items()
        if key not in {"accepted", "rejected"}
        and not key.startswith("rejected:")
    }
    if accepted:
        counters["accepted"] = len(accepted)
    if rejected:
        counters["rejected"] = len(rejected)

    invalid_reasons = set(state.invalid_reasons)
    invalid_reasons.discard("request_rejected")
    for _request_id, reason, evidence in rejected:
        _increment(counters, f"rejected:{reason}")
        invalid_reasons.add(evidence)

    accepted_sequences = {
        record[3] for record in accepted if record[3] is not None
    }
    queue_transitions = {
        sequence: transition
        for sequence, transition in state.queue_transitions.items()
        if sequence not in accepted_sequences
    }
    queue_transitions.update(
        {
            sequence: (depth, observed_ns)
            for (
                _request_id,
                _now_ns,
                _queue_depth,
                sequence,
                depth,
                observed_ns,
            ) in accepted
            if sequence is not None
        }
    )
    queue_sequence_high_water = state.queue_sequence_high_water
    if accepted_sequences:
        queue_sequence_high_water = max(
            queue_sequence_high_water,
            max(accepted_sequences),
        )

    inflight_last_ns = state.started_ns
    inflight_value = 0
    inflight_area = 0
    inflight_minimum = 0
    inflight_maximum = 0
    events = [(record[1], 1) for record in accepted]
    events.extend((when, -1) for when in state.terminal_times.values())
    for when, delta in sorted(events, key=lambda item: (item[0], -item[1])):
        effective_ns = max(when, inflight_last_ns)
        inflight_area += inflight_value * (effective_ns - inflight_last_ns)
        inflight_value += delta
        inflight_last_ns = effective_ns
        inflight_minimum = min(inflight_minimum, inflight_value)
        inflight_maximum = max(inflight_maximum, inflight_value)

    state.counters = counters
    state.invalid_reasons = invalid_reasons
    state.queue_transitions = queue_transitions
    state.queue_sequence_high_water = queue_sequence_high_water
    state.inflight_last_ns = inflight_last_ns
    state.inflight_value = inflight_value
    state.inflight_area = inflight_area
    state.inflight_minimum = inflight_minimum
    state.inflight_maximum = inflight_maximum
    state.outcome_accounting_dirty = False
```

- [ ] **Step 4: Make recovery conditional**

Replace `_resolve_accounting_internal` with:

```python
def _resolve_accounting_internal(metrics) -> None:
    state = _sealed_accounting(metrics)
    with state.lock:
        if state.outcome_accounting_dirty:
            _rebuild_outcome_accounting_locked(state)
```

- [ ] **Step 5: Replace acceptance and rejection rebuilds with journal-first projections**

In `_commit_acceptance_internal`, keep record construction unchanged but replace
the existing mutation tail with:

```python
            state.outcome_accounting_dirty = True
            state.outcomes[key] = ("accepted", record)
            _apply_accepted_outcome_locked(state, record)
            state.outcome_accounting_dirty = False
```

Remove the unconditional `_rebuild_outcome_accounting_locked(state)` call.

In `_record_rejected_internal`, use:

```python
        if existing is None:
            record = (effective_request_id, reason, "request_rejected")
            state.outcome_accounting_dirty = True
            state.outcomes[key] = ("rejected", record)
            _apply_rejected_outcome_locked(state, record)
            state.outcome_accounting_dirty = False
```

Remove its unconditional rebuild call.

- [ ] **Step 6: Make terminal inflight accounting incremental**

Replace the terminal timestamp/rebuild pair in `record_terminal()` with:

```python
            if request_id not in state.terminal_times:
                state.outcome_accounting_dirty = True
                state.terminal_times[request_id] = completed_ns
                _apply_terminal_inflight_locked(state, completed_ns)
                state.outcome_accounting_dirty = False
            else:
                state.terminal_times[request_id] = completed_ns
                state.outcome_accounting_dirty = True
```

Keep the unconditional rebuild in `finalize()` so each normal run receives one
canonical projection.

- [ ] **Step 7: Run focused GREEN tests**

Run the Task 1 command again.

Expected: `9 passed` because the dirty recovery test expands to six cases and
the engine recovery test expands to two cases.

- [ ] **Step 8: Run the full metrics file and focused engine recovery set**

Run:

```bash
PYTHONPATH=framework/src \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest \
  framework/tests/test_async_metrics.py \
  framework/tests/test_async_engine.py::test_rejection_outcome_projection_recovers_reason_and_evidence \
  framework/tests/test_async_engine.py::test_terminal_bitmap_completes_accepted_recovery_without_masking_interrupt \
  -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit the P0 implementation**

```bash
git add \
  framework/src/core/async_inference/metrics.py \
  framework/tests/test_async_metrics.py \
  framework/tests/test_async_engine.py
git commit -m "fix: make async metrics accounting incremental"
```

---

### Task 3: Verify scaling and the complete async contract

**Files:**
- Modify: `docs/superpowers/plans/2026-07-23-async-metrics-incremental-accounting.md` only to check completed steps and record no new requirements.
- Test: `framework/tests/test_async_metrics.py`
- Test: `framework/tests/test_async_engine.py`
- Test: `framework/tests/test_async_completion.py`
- Test: `framework/tests/test_native_async_runtime_executor.py`
- Test: `framework/tests/test_async_runner.py`
- Test: `framework/tests/test_async_result_artifacts.py`

**Interfaces:**
- Consumes: completed P0 implementation.
- Produces: focused/full regression evidence and before/after scaling measurements.

- [ ] **Step 1: Run the CPU-only scaling benchmark**

```bash
PYTHONPATH=framework/src \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -c 'from types import SimpleNamespace; from time import perf_counter; from core.async_inference.metrics import AsyncMetricsCollector,_commit_acceptance_internal; counts=(500,1000,3000); results=[]
for n in counts:
 m=AsyncMetricsCollector(started_ns=0,worker_count=1); t=perf_counter()
 for i in range(n):
  s=i+1; _commit_acceptance_internal(m,now_ns=s,queue_depth=min(s,16),queue_transition=SimpleNamespace(sequence=s,depth=min(s,16),now_ns=s),attempt_token=i,request_id=i)
 m.finalize(n+1); results.append((n,perf_counter()-t))
print("\n".join(f"count={n:4d} elapsed={elapsed:.6f}s per_request_us={elapsed*1e6/n:.3f}" for n,elapsed in results))'
```

Expected: 3,000 requests finish promptly and per-request cost remains in the
same order of magnitude instead of increasing cubically. Record values in the
completion report, not as test assertions.

- [ ] **Step 2: Run the async core regression suite**

```bash
PYTHONPATH=framework/src \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest \
  framework/tests/test_async_metrics.py \
  framework/tests/test_async_engine.py \
  framework/tests/test_async_completion.py \
  framework/tests/test_native_async_runtime_executor.py \
  framework/tests/test_async_runner.py \
  framework/tests/test_async_result_artifacts.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 3: Run the complete framework suite**

```bash
cd framework
PYTHONPATH=src \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest tests -q
```

Expected: all tests pass. Vendor SDK tests that are explicitly skipped without
hardware remain skipped.

- [ ] **Step 4: Check scope and worktree cleanliness**

```bash
git diff --check
git status --short
git diff --stat origin/fix/mobilint-image-preprocess..HEAD
```

Expected: no whitespace errors; only the approved spec, plan, P0 metrics source,
and P0 tests are committed. `.superpowers/sdd/task-5-report.md` remains modified
and unstaged as user-owned content.

- [ ] **Step 5: Commit only a plan checkbox update if one is needed**

```bash
git add docs/superpowers/plans/2026-07-23-async-metrics-incremental-accounting.md
git commit -m "docs: record async metrics verification"
```

Do not create this commit when the plan file did not need an update.

---

### Review hardening amendment

Code review identified two recovery boundaries that the initial implementation
tests did not expose because `finalize()` repaired them:

- a new acceptance, rejection, or terminal mutation must resolve a pre-existing
  dirty projection before it can clear the dirty flag for its own update;
- canonical recovery needs ordered authoritative evidence for legacy
  `sequence=None` queue events, including both accepted outcomes and explicit
  depth events.

The implementation therefore journals legacy queue events in O(1), replays them
during the existing O(N log N) canonical rebuild, and adds projection-level
tests for chained dirty mutations, duplicate terminal timestamps, explicit and
accepted legacy interruption recovery, idempotency, and incremental/canonical
equivalence.
