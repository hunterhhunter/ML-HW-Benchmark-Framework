# Async Completion Retirement Lease Design

## Context

`AsyncInferenceEngine` currently publishes a `BatchCompletion` without waiting
for the completion thread to acknowledge it. The engine retains the related
runtime execution and dequeue operation in several registries until the
`CompletionCoordinator` commits the terminal state.

For a native async executor with `max_inflight=1`, the worker can start its next
iteration before the previous handoff is acknowledged. The previous native
dispatch still owns the sole permit, but the completion terminal callback only
retries deferred handoffs, not the worker-local handoff. The next execution can
therefore wait for the permit until the flush timeout.

## Goals

- Retire one completion's dequeue ownership and runtime execution exactly once.
- Release the native executor permit promptly after terminal completion.
- Keep request admission, batching, workers, cancellation, flush, and shutdown
  in `AsyncInferenceEngine`.
- Keep terminal membership, decoder/evaluator execution, metrics, and trace
  publication in `CompletionCoordinator`.
- Prevent `CompletionCoordinator` from depending on `RuntimeExecutor`, Mobilint,
  or queue implementation details.
- Preserve blocking runtime and inline completion behavior.

## Non-goals

- Replacing the framework request queue with a vendor SDK queue.
- Making `RuntimeExecutor.execute()` nonblocking.
- Changing batching, queue capacity, worker count, or native inflight policy.
- Refactoring cancellation and drain journals that are unrelated to normal
  worker completion.
- Adding vendor-specific behavior to the common completion coordinator.

## Considered Approaches

### 1. Let the worker wait synchronously for every completion ACK

This removes the deadlock but serializes worker execution with decoding,
evaluation, metrics, and trace writes. It also discards the existing completion
queue pipeline and reduces throughput.

### 2. Add an engine-owned handoff retirement manager

This centralizes the existing maps but adds another lifecycle object owned and
coordinated by `AsyncInferenceEngine`. Completion notification still requires a
reverse callback into the engine.

### 3. Transfer a one-shot retirement lease with the completion

This is the selected approach. The worker creates a lease containing the
normal-completion cleanup capability and transfers it with the completion
handoff. `CompletionCoordinator` invokes the generic lease only after it has
committed the handoff terminal state. The coordinator does not know which
resources the lease releases.

## Design

### Retirement lease

Introduce a private one-shot `_RetirementLease` in the async engine module. It
owns a callback and a small thread-safe state machine:

- `PENDING`: the callback has not been claimed.
- `RETIRING`: one caller is running the callback.
- `RETIRED`: cleanup completed successfully.
- `FAILED`: cleanup raised; the same failure is observable without executing
  the callback again.

`retire()` is idempotent and safe when completion, shutdown, or recovery paths
race. Only the first caller executes cleanup.

### Completion payload

Extend the private completion handoff record, rather than the public request
identity, with an optional generic retirement lease. The coordinator treats it
as a capability exposing `retire()` and does not import executor or queue types.

The existing operation key remains the journal identity used to make completion
publication retryable. The lease replaces the normal worker-path execution and
dequeue retirement bookkeeping; it does not replace the operation key.

### Terminal ordering

For a queued completion, the completion thread performs these steps:

1. Validate completion membership.
2. Decode outputs and update the evaluator.
3. Commit request terminal records, metrics, and trace callbacks.
4. Mark the completion handoff `ACKED`.
5. Invoke the attached retirement lease.
6. Notify waiters.

The lease callback finalizes the matching dequeue operations, retires the
coordinator's acknowledged handoff record, and calls
`RuntimeExecutor.acknowledge(execution)`. Native permits therefore remain held
until the framework has consumed the result, preserving the current buffer
lifetime contract. A caller that already holds the lease uses the lease state,
rather than absence of the retired coordinator record, to distinguish success
from an unknown operation key.

If lease retirement fails, the coordinator records a completion-thread failure
and wakes engine waiters. It must not publish a successful shutdown proof while
runtime or dequeue ownership is unresolved.

### Engine ownership

`AsyncInferenceEngine` continues to own the high-level concurrency lifecycle.
For the normal worker path, it no longer needs to poll a worker-local pending
handoff before accepting the next request. The lease owns the exact cleanup
operation once publication succeeds.

Existing recovery, cancellation, and drain paths remain unchanged unless a
normal completion lease is already attached. This keeps the first change
focused on the reproduced two-request deadlock.

### Compatibility

- Inline/e2e completion does not use an operation key and does not attach a
  retirement lease.
- `BlockingRuntimeExecutor.acknowledge()` remains a no-op.
- `NativeAsyncRuntimeExecutor` continues requiring both physical callback
  completion and logical acknowledgement before releasing its permit.
- Mobilint and other native backends require no lease-specific code.

## Error Handling

- A lease callback exception is captured as a terminal retirement failure and
  marks the run invalid.
- Duplicate completion delivery cannot execute cleanup twice.
- A completion thread failure before terminal commitment leaves the lease
  available to bounded engine recovery/shutdown handling.
- Shutdown must still report failure when the native executor snapshot contains
  an unretired dispatch.

## Tests

1. A focused regression uses a native async executor with `max_inflight=1` and
   submits two requests with one worker. Both requests must complete without a
   flush timeout.
2. The regression verifies the second native dispatch is accepted only after
   the first completion has terminally retired its lease.
3. A unit test races or repeats `retire()` and proves the callback runs once.
4. A retirement callback failure produces an invalid/failure result and does
   not rerun the callback.
5. Existing async engine, completion coordinator, native executor, e2e parity,
   and Mobilint runtime tests remain green.

## Acceptance Criteria

- The previously failing two-request, one-worker, one-native-slot scenario
  completes with two accepted and two completed requests and zero timeouts.
- No synchronous completion wait is added to the worker's successful path.
- `CompletionCoordinator` has no dependency on runtime or vendor classes.
- Native dispatch permits are released only after terminal completion.
- No unrelated queue, cancellation, or monitor behavior changes.
