# Async Metrics Incremental Accounting Design

**Status:** Approved

**Date:** 2026-07-23

**Scope:** P0 async metrics scalability only

## Problem

The async request queue itself uses bounded `append`/`popleft` operations and is
not the source of the observed nonlinear slowdown. The dominant cost is
`_rebuild_outcome_accounting_locked()` in
`framework/src/core/async_inference/metrics.py`.

Every accepted, rejected, and terminal request currently rebuilds accounting
from the complete outcome history while holding the sealed metrics lock. The
rebuild also recreates an accepted-sequence set inside a transition
comprehension and sorts all inflight events. A rebuild at history size `k` is
therefore approximately O(k^2), and repeating it for every request produces an
O(N^3) measurement path.

A CPU-only acceptance benchmark on the existing implementation reproduced the
growth:

| Requests | Elapsed |
|---:|---:|
| 50 | 0.001572 s |
| 100 | 0.007890 s |
| 200 | 0.053831 s |
| 400 | 0.389317 s |
| 800 | 3.258113 s |

The final doubling took 8.37 times longer. Because the rebuild holds the
metrics lock, producer acceptance and completion accounting serialize behind
this CPU work and can starve the accelerator of new submissions.

## Goals

- Make normal acceptance, rejection, and terminal inflight accounting O(1).
- Keep `state.outcomes` as the authoritative accepted/rejected journal.
- Keep terminal timestamps as the authoritative inflight-completion evidence.
- Preserve attempt-token idempotency and accepted/rejected conflict detection.
- Preserve queue transition evidence, missing-sequence detection, and
  time-weighted inflight metrics.
- Recover correct derived accounting after interruption at any tested mutation
  boundary.
- Limit full-history work to recovery and finalization, with O(N log N) worst
  case from event sorting.

## Non-goals

- Do not change `RequestQueue`, worker scheduling, batching, or completion
  coordination.
- Do not introduce a non-blocking native dispatch interface.
- Do not change Mobilint Future polling, RBLN owner-loop behavior, activation
  slots, or framework worker counts.
- Do not optimize `CompletionCoordinator.wait_for_requests()` in this change.
- Do not weaken exact-once terminal, timeout, late completion, dispatch ACK, or
  runtime unload contracts.
- Do not split the large async engine files during the P0 fix.

## State Model

The sealed accounting state has two layers:

1. Authoritative evidence
   - `outcomes`: one immutable accepted or rejected outcome per attempt token
   - `terminal_times`: completion timestamp evidence keyed by request ID
   - explicit queue transition and failed-sequence evidence
2. Derived projection
   - accepted/rejected and rejected-reason counters
   - queue transition projection and high-water state
   - time-weighted inflight gauge

Add one sealed boolean, `outcome_accounting_dirty`, initialized to `False`.
It indicates that authoritative evidence may be newer than its derived
projection.

## Normal Mutation Protocol

All steps occur under the existing sealed accounting lock.

### Acceptance

1. Normalize untrusted values before acquiring the lock, as today.
2. Resolve the attempt token and inspect the existing outcome.
3. Reject an existing rejected outcome.
4. Return without mutation for an identical accepted outcome.
5. Set `outcome_accounting_dirty = True`.
6. Store the accepted journal record.
7. Increment the accepted counter once.
8. Record the enqueue queue transition once, or update the legacy queue gauge.
9. Advance the inflight gauge by one at the acceptance timestamp.
10. Set `outcome_accounting_dirty = False`.

### Rejection

1. Resolve the attempt token and inspect the existing outcome.
2. Reject an existing accepted outcome.
3. Return without mutation for an identical rejected outcome.
4. Set `outcome_accounting_dirty = True`.
5. Store the rejected journal record.
6. Increment `rejected`, `rejected:<reason>`, and rejection evidence once.
7. Set `outcome_accounting_dirty = False`.

### Terminal inflight update

The existing terminal counters, errors, and timing distributions remain in
their current code path. For outcome-derived inflight accounting:

1. Set `outcome_accounting_dirty = True` before storing a new terminal time.
2. Store the terminal timestamp evidence.
3. Decrement the inflight gauge once at that timestamp.
4. Set `outcome_accounting_dirty = False`.

The single completion coordinator already serializes normal terminal recording,
so normal completion timestamps are monotonic. Finalization still rebuilds from
sorted evidence and is the canonical result if recovery created an unusual
terminal-before-acceptance ordering.

## Recovery and Finalization

`_resolve_accounting_internal()` checks the dirty flag. It runs a full rebuild
only when the flag is set, then clears the flag after a successful rebuild. An
interrupt during rebuild leaves the flag set so a later recovery or finalize
can retry.

`finalize()` performs one canonical full rebuild regardless of the dirty flag.
This validates the incremental projection and preserves current snapshot
semantics. Normal per-request methods never call the full rebuild.

The rebuild is rewritten to compute local values before replacing derived
state:

- scan `outcomes` once to separate accepted and rejected records;
- construct `accepted_sequences` once;
- rebuild accepted/rejected and rejected-reason counters;
- merge accepted queue transitions with explicit non-acceptance transition
  evidence without a nested set reconstruction;
- construct and sort inflight events once;
- replace the derived projection only after the local reconstruction succeeds;
- clear `outcome_accounting_dirty` last.

The rebuild remains under the sealed lock. Moving reconstruction outside the
lock would require versioned snapshots and retry logic, which adds complexity
without benefiting the normal hot path.

## Failure Semantics

The dirty protocol makes every tested interruption boundary recoverable:

- interruption after dirty but before journal storage rebuilds the old journal;
- interruption after journal storage but before projection update rebuilds the
  new journal;
- interruption during projection update discards partial derived state during
  rebuild;
- interruption after projection update but before clearing dirty rebuilds an
  equivalent canonical projection;
- interruption during recovery leaves dirty set for another retry.

Existing engine recovery continues to query `state.outcomes` for authoritative
accepted/rejected ownership. The P0 change does not alter request ownership,
queue rollback, slot release, coordinator membership, or terminal retirement.

## Test Strategy

Tests are added before production changes.

1. Hot-path structure
   - Spy on `_rebuild_outcome_accounting_locked`.
   - Record 3,000 accepted requests and terminal traces.
   - Assert zero rebuilds before finalize and exactly one during finalize.
2. Idempotency
   - Repeat the same accepted and rejected attempt tokens.
   - Assert counters, queue transitions, and inflight values change once.
   - Assert accepted/rejected conflicts still raise.
3. Fault recovery
   - Inject failures before journal storage, after journal storage, during the
     derived update, and after the update before dirty clear.
   - Resolve or finalize and assert the canonical counters and queue/inflight
     summaries.
4. Differential projection
   - Feed deterministic accepted, rejected, queue, and terminal evidence.
   - Compare the incremental snapshot with a forced canonical rebuild.
5. Existing contracts
   - Run `test_async_metrics.py`.
   - Run the async engine accounting/recovery tests.
   - Run the full framework test suite after focused tests pass.
6. Performance evidence
   - Re-run 500, 1,000, and 3,000 request CPU-only benchmarks.
   - Record elapsed time and normalized time per request without using a flaky
     wall-time threshold as a CI assertion.
   - Re-run the RBLN 3,000-request hardware command outside CI.

## Acceptance Criteria

- No full accounting rebuild occurs in normal acceptance, rejection, or
  terminal hot paths.
- A normal run performs one canonical rebuild during finalize.
- Normal hot-path accounting is O(1) per request.
- Finalize and exceptional recovery are O(N log N) or better.
- The 500/1,000/3,000 CPU benchmark grows approximately linearly rather than
  cubically.
- All prior accounting, queue sequence, fault-injection, timeout, retirement,
  and result-schema tests pass unchanged unless a test is explicitly rewritten
  to assert the new hot-path structure.
- The RBLN 3,000-request run completes without metrics-driven accelerator
  starvation and reports accepted/completed/evaluator counts of 3,000 with zero
  outstanding requests.

## Rollout

Deliver this as a standalone P0 metrics commit. Do not combine it with native
async dispatch or queue architecture changes. If hardware validation exposes a
new bottleneck after P0, profile that result before approving a separate P1
design.
