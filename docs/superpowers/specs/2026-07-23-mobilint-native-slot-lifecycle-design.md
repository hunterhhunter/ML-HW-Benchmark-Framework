# Mobilint Native Async Slot Lifecycle Design

## Context

After incremental async metrics accounting removed an O(n^3) aggregation cost,
a 1,000-request Mobilint native-async run reached roughly 486 QPS and exposed a
single submission failure:

```text
submitted: 1000
accepted: 1000
completed: 999
failed: 1
timed_out: 0
outstanding: 0
```

The failed request reported `RuntimeError: native async submission failed`.
The native executor emitted that generic boundary error because
`MobilintNativeBackend.submit_async()` raised before the SDK accepted the next
request.

The failure is reproducible without Mobilint hardware. With one backend slot,
an immediately completed first Future, and a callback held on an Event, a
second `submit_async()` deterministically raises:

```text
Mobilint native async waiter capacity is exhausted.
```

## Root Cause

`MobilintNativeBackend` acquires `_slots` before calling
`Model.infer_async()`. The semaphore represents the number of SDK executions
that may be outstanding, but the waiter currently releases it only after the
framework callback returns:

```text
Future.get() terminal
-> normalize outputs
-> framework callback
-> remove job and input references
-> release SDK slot
```

The native executor callback publishes the result and wakes the framework
worker before returning. Once the completion thread consumes the result and
acknowledges the dispatch, the executor permit becomes available. A newly
permitted worker can therefore call the Mobilint backend while the previous
waiter is still returning from its callback and still owns `_slots`.

The race is:

```text
Mobilint waiter                        Framework worker/completion
─────────────────────────────────────────────────────────────────
Future.get() returns
outputs are normalized
callback(outcome) enters
  dispatch event is set
                                      worker publishes completion
                                      completion commits terminal state
                                      dispatch is acknowledged
                                      executor permit is released
                                      next execute() calls submit_async()
                                      backend slot acquisition fails
callback(outcome) returns
backend slot is released
```

The metrics optimization did not create the race. It shortened the surrounding
work enough to make the existing scheduling window observable.

## Ownership Model

The backend must represent two independent lifetimes.

### SDK execution-slot ownership

- Starts immediately before `infer_async()` submission.
- Continues while the Mobilint Future is pending.
- Ends after `Future.get()` returns or raises and output normalization finishes.
- Does not depend on framework callback latency.

### Framework job/callback ownership

- Starts when `_jobs[job_id]` is registered.
- Retains the Future and input arrays while the accepted job is tracked.
- Continues until the framework callback fully returns.
- Keeps shutdown and model unload from declaring quiescence during a callback.

The defect is caused by ending both lifetimes at the framework callback
boundary.

## Selected Design

Split slot retirement from job retirement:

```text
infer_async() accepted
-> Future.get() terminal
-> normalize outputs or construct failure outcome
-> release SDK slot exactly once
-> invoke framework callback
-> callback returns
-> remove _jobs entry and clear input references
```

Add `slot_released: bool` to `_MobilintAsyncJob` and a private
`_release_job_slot(job) -> bool` operation serialized by the backend condition.
The operation is idempotent, releases the bounded semaphore at most once, and
notifies shutdown/capacity waiters.

The waiter calls this operation after it has constructed the terminal outcome
and before it invokes the framework callback. Its existing finalizer continues
to remove `_jobs[job_id]` and clear `job.inputs`, but no longer releases the
slot.

Submission failures before a job exists keep their current direct-release
behavior:

- closing detected after slot acquisition;
- `infer_async()` raises before returning a Future.

The existing `claim_lock` and `claimed` guard remain the single owner of
`Future.get()`, including thread-start failure and inline fallback paths.

## Required Invariants

- Every successful semaphore acquisition has exactly one release.
- No slot is released while its Future is pending.
- Output normalization finishes before slot release.
- Callback latency and callback exceptions do not control SDK capacity.
- `_jobs` retains every accepted job until its callback returns.
- `shutdown()` waits for accepted callbacks and waiter threads.
- `Future.get()` is called no more than once per accepted job.
- A synchronous `infer_async()` exception releases the pre-job slot once.
- Thread construction, start, and start-then-raise fallbacks use the same slot
  and job retirement rules.
- Native executor permits continue to remain held until framework terminal
  acknowledgement, preserving framework-visible result lifetime.

## Output Buffer Safety

qb Runtime documents `Future.get()` as the terminal operation that returns a
list of NumPy arrays. Its official async examples submit multiple Futures and
retain their results while later inference is active. The framework also keeps
the native executor permit until completion processing has consumed the result.

The implementation will not add an unconditional output copy because that
would copy large CNN tensors on every request and could undo the throughput
improvement. Hardware acceptance must nevertheless retain and hash one result,
run the following request, and prove the retained result is unchanged. If the
installed SDK violates this result-lifetime assumption, a contiguous owned
copy before slot release is the compatibility fallback and must be measured as
a separate performance decision.

References:

- <https://docs.mobilint.com/v1.1/doxygen/html_en/classqbruntime_1_1future_1_1Future.html>
- <https://docs.mobilint.com/v1.2/kr/advanced_usage.html>

## Rejected Alternatives

### Remove `_slots`

The common executor normally bounds dispatches, but the backend is also a
standalone SDK protection boundary. Removing `_slots` would make direct or
misconfigured callers depend on undocumented SDK saturation behavior.

### Block while acquiring `_slots`

This converts a short ownership race into latency or a deadlock and makes the
backend's nonblocking submission contract ambiguous.

### Retry failed submissions

An exception does not universally prove that a vendor request was not accepted.
Blind retry can duplicate inference.

### Add another request queue or reduce workers

The framework already owns request queuing and backpressure. Another queue or a
lower worker count masks this boundary bug and reduces throughput without
correcting ownership.

### Change executor permit retirement

Executor permits correctly protect framework result consumption through
terminal acknowledgement. Releasing them earlier would weaken the existing
buffer-lifetime contract and is unrelated to this backend-local slot race.

## Test Strategy

CPU tests use real backend synchronization with fake SDK Futures:

1. With one slot, hold the first callback and prove a second submission is
   accepted before that callback returns.
2. Repeat with a failing first Future and verify the sanitized error callback
   does not retain SDK capacity.
3. Repeat with a callback that eventually raises and prove no slot leak.
4. Hold a callback, prove `shutdown()` still reports non-quiescence, then release
   the callback and prove shutdown completes.
5. Preserve the parameterized thread construction/start/start-then-raise tests
   and their exact-once Future consumption checks.
6. Preserve the pending-Future capacity test: a second request must still be
   rejected while the first Future is physically pending.

Hardware acceptance runs 1,000 and 3,000 requests and records:

- `submitted == accepted == completed`;
- `failed == timed_out == outstanding == 0`;
- SDK, driver, firmware, device, artifact hash, and effective runtime options;
- QPS, p95, and p99 before and after the change;
- retained output integrity across the following inference;
- zero outstanding Futures before model disposal.

## Scope

This change modifies only the Mobilint native backend, its focused tests, and
runtime documentation. It does not change the common async request queue,
completion coordinator, native executor, metrics collector, worker count,
queue capacity, timeout policy, or error reporting contract. A vendor-specific
capacity exception type or counter belongs in a later diagnostics change.

## Acceptance Criteria

- The deterministic CPU reproducer fails on the old implementation and passes
  after the change.
- A pending Future still prevents over-capacity submission.
- A terminal Future frees SDK capacity before callback return.
- Shutdown does not complete while a callback is active.
- Mobilint native backend and native executor regression suites pass.
- Hardware runs complete with no submission failure, timeout, or outstanding
  request and without material throughput regression.
