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

## Independent review follow-up

An independent review of commit `abba3b5` requested three Important
corrections. Each correction received a new failing test before production was
changed.

### RED evidence

The review matrix failed in all six collected cases:

```text
test_old_pending_only_cancellation_cannot_clear_new_drain_journal       FAIL
test_deferred_cancellation_reconciles_ack_commit_then_raise             FAIL
test_deferred_cancellation_clear_fault_never_sticks_retiring[before]    FAIL
test_deferred_cancellation_clear_fault_never_sticks_retiring[partial]   FAIL
test_cancel_submit_error_persists_deferred_owner_before_diagnostics[
  failure_sink]                                                        FAIL
test_cancel_submit_error_persists_deferred_owner_before_diagnostics[
  handoff_query]                                                       FAIL

6 failed
```

The failures proved three distinct gaps:

1. A pending-only cancellation owned a retained active-drain scalar key but no
   `_DrainOperation`. A new real drain journal could materialize under that key
   while the old reaper was gated immediately before handoff ACK. The old
   identity-only clear then erased the scalar and orphaned the new journal and
   transition allocation.
2. If handoff acknowledgement removed the coordinator journal and then raised,
   retry observed state `None` and never reconciled the committed removal. A
   clear exception before or after partial mutations could also leave
   `RETIRING` or lose the explicit journal owner.
3. Submission-error reconciliation called the handoff query and invalid-reason
   sink before durable ownership transfer. Either diagnostic exception left the
   canonical operation in `ACTIVE_PRODUCER` with no scheduled owner.

### Corrections

- Every canonical cancellation now captures an exact cancellation generation
  plus the active-drain scalar key and generation, including the empty retained
  key used by pending-only cancellation. Materializing a new drain journal
  advances the drain generation even if it reuses the same scalar key. A late
  old reaper clears only an exact matching key and generation.
- Handoff retirement records `handoff_ack_started`. If an ACK call removed the
  coordinator handoff and then raised, the next pass treats exact state `None`
  as committed only for that started stage and does not issue a second ACK.
- Cancellation clear is now a reconciled stage transaction. Pre-mutation and
  partial-mutation failures restore `DEFERRED_TERMINAL_WAIT` and the exact
  journal owner; an exception after all exact clear mutations is recognized as
  already committed. A newer cancellation/drain generation is never cleared.
- Submit failure conservatively transfers the canonical operation before
  failure diagnostics. Handoff-query failure is treated as nonterminal for
  ownership purposes. Diagnostic or reaper errors may still propagate, but the
  operation is already durably journaled.
- Retirement diagnostics are exception-safe and bounded to 128 characters for
  the type and 512 characters for the message.

### Follow-up verification

The six-test review matrix passed after the minimum implementation:

```text
6 passed in 0.52s
```

The original plus review cancellation matrix passed:

```text
11 passed in 0.35s
```

The six review cases were then run in ten fresh pytest processes:

```text
10/10 processes passed; 60/60 test executions passed
```

Final suites:

```text
async engine: 232 passed in 4.63s
focused async/completion/runtime/native/runner/CLI/result: 867 passed in 16.49s
full framework: 1126 passed, 13 skipped, 1 warning in 55.42s
```

The single warning remains the pre-existing unregistered integration marker.

## Second independent review follow-up

A second independent review found three narrower publication and partial-commit
windows. New tests were again added before production changes.

### RED evidence

The new matrix collected six cases. The two empty/error publication cleanup
controls already passed, while the four affected paths failed:

```text
drain journal materialized before drain_requests returned                FAIL
explicit cancel reclaim after ACK pop-commit/raise                       FAIL
canonical drain finish failure before pop                               FAIL
canonical drain finish failure after pop plus same-key replacement      FAIL

4 failed, 2 passed
```

The failures showed:

1. The drain generation advanced only after `drain_requests()` returned. A
   queue journal was therefore visible internally while the old generation was
   still published, allowing an old reaper to clear the scalar in that gap.
2. Explicit `_cancel_queued()` immediately reclaimed every deferred operation.
   An operation whose ACK had already committed and whose coordinator journal
   was gone was therefore made an active producer again before terminal
   reconciliation.
3. Deferred drain retirement re-looked up only by key. After the canonical
   drain was popped and the pop wrapper raised, a replacement drain created
   under the same key could be mutated and retired by the old operation.

### Corrections

- When no drain journal currently exists, `_drain_request_queue` reserves a new
  publication generation under the cancellation-retirement lock before calling
  the queue journal API. A materialized journal keeps that generation. Empty or
  double-failure paths restore the prior generation, or clear the scalar if the
  prior owner retired while publication was gated.
- Explicit cancellation reclaim first invokes the non-waiting deferred reaper
  outside the retirement lock, then re-reads the exact operation phase. An ACK
  pop-commit represented by `handoff_ack_started` plus coordinator state `None`
  retires without any second submit. Only a still-nonterminal deferred
  operation can return to `ACTIVE_PRODUCER`.
- `_CancellationOperation` now owns the canonical `_DrainOperation` object and
  a `drain_retirement_started` stage. Before-pop failure retries only that same
  object. Once retirement started, canonical absence or replacement proves the
  old pop committed; the reaper marks only the old stage complete and never
  mutates or finishes the replacement journal.

### Follow-up verification

```text
new review matrix: 6 passed in 0.10s
cumulative cancellation matrix: 17 passed in 0.37s
async engine: 238 passed in 4.58s
independent-process stress: 10/10 processes, 60/60 executions
focused async/completion/runtime/native/runner/CLI/result: 873 passed in 14.26s
full framework: 1132 passed, 13 skipped, 1 warning in 54.39s
```

`git diff --check` remained clean. The one warning is still the pre-existing
unregistered integration marker.
