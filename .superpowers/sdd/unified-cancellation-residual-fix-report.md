# Unified active-cancellation residual Phase 4 report

Date: 2026-07-20

Base: `adbcb3d`

Scope:

- `framework/src/core/async_inference/engine.py`
- `framework/tests/test_async_engine.py`
- this report

No completion coordinator, unified `InferenceEngine`, runner, runtime executor,
native SDK adapter, CLI, result, evaluator, or generation code was changed.

## TDD evidence

Production code was unchanged while the tests were written.

The first deterministic request-0..3 regression failed after late coordinator
terminalization because `_active_cancellation_operation` remained the exact
canonical request-3 operation. The gated test held `_active_drain_lock` from a
different thread for the entire terminal callback, reproducing the diagnosed
lost-wakeup window.

The expanded RED matrix was then run before implementation:

```text
test_terminal_cancellation_retires_while_active_drain_lock_is_busy       FAIL
test_deferred_cancellation_immediately_retires_terminal_before_transfer FAIL
test_normal_cancellation_ack_stays_with_original_producer_exactly_once  FAIL
test_deferred_cancellation_retirement_fault_keeps_retry_owner           FAIL
test_old_cancellation_reaper_cannot_clear_new_generation                FAIL

5 failed
```

The failures were the expected missing behavior: the cancellation operation
had no phase, and the engine had no deferred cancellation transfer/reaper.

After the minimum implementation, the same matrix passed:

```text
5 passed in 0.35s
```

## Implementation

`_CancellationOperation` now has the explicit private lifecycle:

```text
ACTIVE_PRODUCER
  -> RETIRED
  -> DEFERRED_TERMINAL_WAIT
       -> RETIRING
       -> RETIRED
```

The engine owns an exact-operation deferred cancellation journal guarded by a
dedicated cancellation-retirement lock. On a submission timeout/error whose
existing handoff is not ACKED, the original producer transfers the canonical
operation to `DEFERRED_TERMINAL_WAIT` and immediately makes one non-waiting
retirement pass. This handles terminal-before-transfer.

The existing coordinator terminal callback now makes one non-waiting sweep of
that journal after the deferred worker-handoff sweep. This handles
terminal-after-transfer. The reaper:

- never calls `_cancel_queued`;
- never submits or resubmits a completion;
- never waits for `_active_drain_lock` or completion;
- queries only the exact existing handoff's terminal state;
- reuses the existing dequeue, drain-journal, transition, handoff, and runtime
  acknowledgement stages;
- records retry stage completion so partial retirement remains idempotent;
- restores `DEFERRED_TERMINAL_WAIT` plus diagnostics after an injected stage
  exception;
- clears active scalar fields only when the active operation is the same object
  generation.

Normal synchronous cancellation ACK remains owned by its producer and becomes
`RETIRED` without entering the deferred journal. A later explicit control call
may reclaim a still-nonterminal deferred operation, preserving the existing
canonical full-queue retry behavior; the callback itself never becomes a
second submitter.

Reads and writes of active cancellation lifecycle fields now share the
dedicated retirement-lock discipline. The callback never acquires the active
drain lock, and the retirement lock is not held while querying/acknowledging a
coordinator handoff or while doing queue cleanup.

## Test coverage

The new tests cover:

- deterministic request-0..3 full-completion-queue shutdown;
- `_active_drain_lock` busy for the entire late terminal callback;
- terminal callback before deferred-owner transfer;
- terminal callback after transfer;
- normal producer ACK without deferred ownership;
- duplicate terminal callback idempotence;
- retry ownership retained after an injected retirement-stage exception;
- an old reaper finishing after a newer cancellation generation is installed;
- one canonical cancellation submission and one exact handoff ACK;
- unchanged metrics after duplicate callback;
- exact queue task balance, slot release, transition retirement, worker/runtime
  ACK cleanup, and completion handoff cleanup;
- weak-reference collection of the drain, cancellation, completion, request,
  and queue entry after exact retirement;
- finite failed shutdown with no second user/control call.

The prior late-worker-handoff regression was updated to require the request-3
drain and transition residual to be empty, rather than documenting that known
defect as an expected leftover.

## Verification

Independent-process race stress (10 fresh pytest processes, three ordering
tests per process):

```text
10/10 processes passed; 30/30 test executions passed
```

Async engine suite:

```text
226 passed in 4.77s
```

Focused async/completion/runtime/native/runner/CLI/result suite:

```text
861 passed in 14.72s
```

Full framework suite:

```text
1120 passed, 13 skipped, 1 warning in 55.80s
```

The warning is the pre-existing unregistered `pytest.mark.integration` marker
in `test_ettm_loader.py`.

`git diff --check` also completed without errors.

## Residual concern

No functional blocker remains in this scope. The weak-reference integration
test suppresses the expected worker exception log at the logger boundary,
because pytest's log capture intentionally retains `exc_info` and therefore
the terminated worker frame until the test itself ends. A `gc.get_referrers`
diagnostic confirmed that this was test-runner ownership; the other four
canonical objects already cleared, and the request cleared when the captured
traceback was not retained.
