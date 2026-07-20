# Unified async shutdown race fix report

Date: 2026-07-20

Base: `001bb62d1acb334f985aaa050af879ebc7fce69e`

Scope:

- `framework/src/core/async_inference/engine.py`
- `framework/src/core/async_inference/completion.py`
- `framework/tests/test_async_engine.py`
- `framework/tests/test_async_completion.py`

The implementation follows the confirmed root cause in
`unified-shutdown-race-diagnosis.md`: a dying worker used to discard handoff
keys that were still nonterminal after its bounded recovery wait, leaving no
live owner to retry dequeue/execution retirement after a late coordinator ACK.

## TDD evidence

Production files were unchanged when the new tests were first run.

The deterministic regression held completion 0's handler, waited for
`shutdown() == False`, joined the worker so all one-shot recovery had ended,
then released the handler. Current production failed deterministically:

```text
test_late_completion_ack_retires_handoffs_after_worker_exit
expected unfinished_tasks == 0
observed unfinished_tasks == 3
```

Six additional focused contracts were added before production changes. The
initial focused run was:

```text
7 failed
```

The failures covered:

1. deterministic late-terminal cleanup after worker exit;
2. ACK immediately before deferred-owner transfer;
3. ACK immediately after deferred-owner transfer;
4. flush retirement winning before deferred transfer;
5. deferred reaper exception containment and retained ownership;
6. coordinator failure finalization notification outside its condition lock;
7. callback exception containment without killing the completion thread.

After the minimum ownership implementation, the same run was:

```text
7 passed in 0.68s
```

## Implementation

`AsyncInferenceEngine` now owns a distinct
`_deferred_worker_handoffs` journal protected by the existing
`_handoff_retirement_lock`.

- Normal live workers continue to own `_worker_local_handoffs`.
- Worker exception recovery accumulates only the handoffs that remain after
  its existing bounded retirement attempts, then transfers those keys to the
  deferred journal before exit.
- Transfer atomically removes worker-local ownership and records engine
  ownership, then immediately performs a nonblocking retirement pass. This
  closes the ACK-before-transfer lost-wakeup window.
- Later coordinator terminal notifications sweep only the deferred journal;
  normal worker-local handoffs are not swept.
- Retirement reuses `_retire_worker_handoffs`,
  `_finalize_dequeue_handoff`, and `_acknowledge_completion_handoff`. No queue
  counter or runtime-ACK path was duplicated.
- Completed keys are removed from worker, flush, deferred, completion, dequeue,
  and execution journals through the existing idempotent path.
- Reaper failures leave keys in deferred ownership, mark `request_failed`, and
  do not escape into the completion thread.
- The deferred journal participates in shutdown diagnostics.

`CompletionCoordinator` now sends handoff-terminal notifications:

- after an ordinary handoff becomes ACKED;
- after failure or normal-stop finalization has committed outstanding terminal
  states and the final coordinator state;
- outside `coordinator.condition`;
- behind exception containment so a callback cannot kill the completion
  thread.

No reaper thread, polling loop, fixed retry count, arbitrary sleep, longer
timeout, raw task-counter mutation, or unbounded wait was added. The
deterministic regression still requires the original finite
`shutdown() == False` result before the blocked handler is released.

## Existing flaky-test observation boundary

The old flaky test released the completion handler and joined only the worker
before asserting cleanup. Under the new event-driven ownership contract, the
worker may correctly transfer a nonterminal handoff and exit before the
coordinator has finished terminalization and invoked the reaper. That assertion
could therefore run before the event that authorizes eventual cleanup.

The test still proves `shutdown() == False` before handler release. After
release it now joins the coordinator before inspecting late-terminal cleanup.
This adds no production wait and does not weaken the finite shutdown contract.

## Characterized residual outside this fix

The deterministic request-0..3 scenario retains exactly one pre-existing
request-3 cancellation `_DrainOperation` after the failed shutdown. The test
characterizes it explicitly:

- the residual contains request ID 3 only;
- every physical queue task entry is already balanced;
- `unfinished_tasks == 0` and `live_task_entry_count == 0`;
- all queue slots are released;
- no dequeue, stop, worker, flush, deferred, or execution handoff remains.

This is the separate active-cancellation issue already isolated by the
diagnosis. Cancellation resume was deliberately not added to the coordinator
callback, preserving the single-root-cause correction boundary.

## Verification

Focused async engine/completion/runtime executor:

```text
258 passed in 4.51s
```

Original flaky test in independent pytest processes, after synchronizing its
observation with coordinator terminalization:

```text
independent_processes=50 failures=0
```

New deterministic regression in independent pytest processes:

```text
deterministic_processes=20 failures=0
```

Full framework suite with writable Hugging Face cache:

```text
1095 passed, 13 skipped, 1 warning in 53.42s
```

The warning is the pre-existing unregistered `pytest.mark.integration` marker
in `framework/tests/test_ettm_loader.py`.

`git diff --check` completed with no output.

## Remaining concern

The characterized request-3 cancellation journal should receive its own
systematic diagnosis and TDD task. It is not a live task-token, slot, worker
handoff, or runtime-buffer leak from this change.
