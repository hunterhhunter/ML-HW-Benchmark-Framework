# Generic async handoff acknowledgement fix report

## Scope

This follow-up fixes the generic async-engine capacity-reuse race exposed by
the SDK-free RBLN full-lifecycle test. The production change is confined to
`framework/src/core/async_inference/engine.py`; the deterministic generic
regression is in
`framework/tests/test_native_async_runtime_executor.py`. Task 7's RBLN test
and documentation were not edited by this task.

The commit subject is `fix: retire async handoffs before worker reuse`.

## Root cause and exact interleaving

With one engine worker and `NativeAsyncRuntimeExecutor(max_inflight=1)`, the
following order was possible:

1. Request 1 physically completed and returned a `RuntimeExecution` while its
   native dispatch continued to own the single executor permit until ACK.
2. The worker submitted request 1's completion handoff without waiting for the
   coordinator ACK.
3. Before the completion coordinator committed that handoff, the worker's
   immediate and loop-top retirement checks both observed a non-ACKED state.
4. The same worker dequeued request 2 and entered `executor.execute()`.
5. Request 2 blocked acquiring the permit still owned by request 1.
6. The completion coordinator then committed request 1 and marked its handoff
   ACKED, but only the blocked worker normally retired that worker-local
   handoff and acknowledged its execution.

No thread remained able to release request 1's permit before request 2's
backpressure timeout. RBLN tests using a larger inflight capacity masked this
cycle.

## TDD evidence

The regression uses the real `AsyncInferenceEngine`,
`CompletionCoordinator`, and `NativeAsyncRuntimeExecutor` with one worker and
one permit. It gates the first evaluator call so the worker deterministically
enters the second permit acquisition before the coordinator marks the first
handoff ACKED. Events and the observed semaphore's synchronous release count
establish the order; the test does not sleep or poll.

Final RED verification temporarily removed only the new terminal-callback
execution-ACK call:

```text
tests/test_native_async_runtime_executor.py::
  test_acked_handoff_retires_executor_before_single_worker_reuses_capacity
1 failed in 0.10s
assert capacity_reused_without_rescue is True
```

The test's cleanup-only rescue ACK allowed both completions and engine
shutdown to finish before reporting RED, so the failure did not leak an async
thread. Restoring the callback call produced:

```text
1 passed in 0.09s
```

## Implementation

When a completion handoff becomes terminal, the coordinator callback first
keeps the existing deferred worker and cancellation retirement order. It then
scans bound executions whose coordinator state is ACKED and acknowledges only
their runtime execution.

This early acknowledgement releases native physical capacity after the
completion is durably committed, but deliberately leaves these owners intact:

- coordinator handoff retirement;
- dequeue-operation finalization;
- worker-local, flush, and deferred retirement markers.

Those owners continue through the existing worker/flush/cancellation paths.
The engine records the early acknowledgement result by handoff key:

- success suppresses a duplicate executor ACK when the normal handoff owner
  later retires the coordinator/dequeue state;
- failure marks the engine failed, suppresses a second acknowledgement
  attempt, and retains `_execution_by_handoff` so shutdown cannot report a
  clean result.

An initial implementation retired the whole coordinator/dequeue handoff from
the callback. The full suite rejected that broader change: 20 existing
flush/deferred race cases could no longer observe the ACKED coordinator
handoff, and one recovery case lost its expected ownership window. The final
implementation therefore moves only executor capacity retirement earlier.

## Concurrency and lifecycle audit

- `_handoff_retirement_lock` remains the single guard for execution bindings,
  early-ACK results, and worker/flush retirement markers. It is an `RLock`,
  matching the existing nested retirement paths.
- The completion callback runs after the coordinator has marked the handoff
  ACKED and invokes this code outside the coordinator condition critical
  section. Lock ordering remains the existing handoff-retirement lock to
  coordinator-state/executor locks ordering.
- A worker/flush retirement racing the callback is exactly once: either the
  full retirement removes the coordinator state before the scan, or the scan
  records its result before full retirement consumes it.
- Deferred cancellations retire before the generic early-ACK scan, preserving
  their journal and drain ownership.
- Successful early-ACK records are removed with their execution binding at
  normal handoff retirement. Failed records intentionally remain paired with
  the retained execution binding and force failed shutdown accounting.
- Duplicate terminal notifications find either no ACKED bound execution or an
  existing result and do not acknowledge again.

## Verification

Focused and generic async suites on the final implementation:

```text
targeted generic capacity regression                    1 passed
prior failing flush/deferred/ACK-failure matrix         22 passed
tests/test_async_engine.py                              239 passed
test_async_completion + test_inference_engine
  + test_native_async_runtime_executor                  207 passed
```

Vendor suites:

```text
RBLN native/runtime/collector                           192 passed
Mobilint native/runtime/device/collector                 77 passed
Furiosa native/runtime                                   16 passed
```

The ordinary filesystem sandbox blocks the Unix-socket send used by
asyncio's self-pipe (`EPERM`), which caused Furiosa callback timeouts there.
The Furiosa suite and final full framework suite were therefore run outside
that sandbox without changing any Furiosa timeout:

```text
1631 passed, 13 skipped, 1 warning in 36.18s
```

The warning is the existing unregistered `integration` pytest mark in
`test_ettm_loader.py`.

## Files and residual boundary

- `framework/src/core/async_inference/engine.py`
- `framework/tests/test_native_async_runtime_executor.py`
- `.superpowers/sdd/rbln-generic-ack-fix-report.md`

No RBLN SDK or CA22 device is available on this host. The full-lifecycle RBLN
test is SDK-free; real-device acceptance remains Task 8's hardware boundary.
An independent reviewer agent could not be started because all four agent
slots were occupied, so review consisted of the focused exactly-once suites,
the full async race suite, the three vendor suites, and the complete framework
suite.
