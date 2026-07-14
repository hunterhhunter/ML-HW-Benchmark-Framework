# Task 5 implementation report

## Scope

- Base: `4fcda7ebec6da92b01a0eade875bd16bd8f7f1ae`
- Branch: `feat/async-inference-queue`
- Worktree: `/tmp/ml-hw-benchmark-async-worktree`
- Added the bounded async inference engine and its focused tests.
- Made three small, backward-compatible `CompletionCoordinator` extensions that
  are required for finite engine cleanup and exact flush/register semantics:
  atomic registration callbacks, optional completion-submit timeout, and
  outstanding snapshot waits.
- No subagents were used.

## RED / GREEN record

1. Missing engine
   - RED: `tests/test_async_engine.py` failed during collection with
     `ModuleNotFoundError: No module named 'core.async_inference.engine'`.
   - GREEN: the initial bounded engine implementation passed the first 10 engine
     tests, then the engine/completion/metrics focus set passed 60 tests.

2. Full completion queue during shutdown
   - RED: an event-gated completion handler filled the bounded completion queue;
     `shutdown()` remained blocked and the test's one-second future timed out.
   - Root cause: `CompletionCoordinator.submit()` only had an unbounded condition
     wait when its queue was full, so interrupt/shutdown cancellation could not
     honor the engine deadline.
   - GREEN: optional submit timeouts preserve the existing no-timeout behavior
     for prior callers while engine paths use `flush_timeout_sec`; the regression
     finishes in about 0.10 seconds.

3. Registration versus coordinator crash
   - RED: a condition/event-controlled crash between `register()` and accepted
     accounting produced `inflight_min == -1` and could let crash cleanup observe
     an incompletely committed request.
   - Root cause: outstanding registration, accepted metrics, and request-queue
     publication were three separately visible operations.
   - GREEN: `CompletionCoordinator.register(..., on_registered=...)` now commits
     accepted accounting and bounded-queue publication while holding the
     coordinator lifecycle condition. The same test reports non-negative inflight
     and exactly one terminal record.

4. Flush invocation boundary
   - RED: after `flush()` began waiting for request 0, request 1 was accepted and
     held in a second runtime gate; the flush future incorrectly waited for
     request 1 and timed out.
   - Root cause: `wait_for_all()` waited for the entire outstanding mapping to
     become empty instead of the accepted set captured at flush invocation.
   - GREEN: the engine snapshots bounded outstanding request IDs and waits only
     for that set. `close_submission()` and the registration commit share the
     engine state lock, so shutdown cannot miss a post-close acceptance.

5. Batch-assembly ownership failure
   - RED: an input object raising `BaseException` during compatibility-key
     calculation left the just-dequeued candidate outside both `owned` and
     `pending`; flush timed out with one outstanding request.
   - Root cause: ownership was recorded after compatibility inspection.
   - GREEN: every request becomes worker-owned immediately after dequeue, before
     queue-depth accounting or batch-key inspection. Both requests receive one
     canonical `WorkerAbort` terminal completion and queue unfinished-task count
     reaches zero.

## Design decisions

- Request memory remains bounded by `queue.Queue(maxsize=queue_capacity)`, a
  reservation semaphore, fixed worker count, a maximum batch size, one pending
  incompatible request per worker, and the coordinator's
  `queue.Queue(maxsize=worker_count)`.
- A slot is reserved before registration. Registration, accepted accounting, and
  queue visibility form one lifecycle commit, so a zero-latency worker or a
  coordinator crash cannot complete before accepted accounting.
- A completion-monitor thread waits on the coordinator condition (not polling)
  and drains abandoned request-queue payloads on coordinator failure, releasing
  slots so blocked submitters wake and reject against the failed engine state.
- Runtime/collation exceptions become one failed `BatchCompletion` for the entire
  batch. Unexpected worker `BaseException` terminalizes every worker-owned and
  still-queued request once.
- Completion submission is bounded by `flush_timeout_sec`; a stalled evaluator or
  full completion queue cannot make worker or shutdown cleanup infinite.
- Dynamic batching respects `max_batch_size`, `batch_timeout_ms`, input
  name/dtype/shape compatibility, runtime dynamic-batch capability and maximum,
  runtime worker capability, static-batched atomic requests, and the LLM batch
  generation capability check.
- A single sentinel is passed between workers. Shutdown uses one absolute
  deadline across flush, worker join, and coordinator stop; daemon workers remain
  the documented fallback for an uninterruptible runtime call.
- `cancel_queued()` first removes all bounded queue payloads and releases their
  slots, then submits one bounded cancellation completion for the drained set.

## Tests added

- Dynamic batching and exact drain with non-negative inflight.
- Runtime error completion without flush deadlock.
- Runtime dynamic-batch, maximum-batch, and worker-count capability validation.
- Invalid engine lifecycle transitions.
- Queue-full nonblocking rejection and duplicate-ID counter invariants.
- Static-batched atomic requests and incompatible-shape batch separation.
- Queued interrupt cancellation with slot/task cleanup.
- Worker `BaseException` and batch-key failure ownership cleanup.
- Completion crash versus blocked submitter and atomic registration commit.
- Permanently blocked runtime finite flush/shutdown behavior.
- Flush invocation snapshot semantics.
- Full completion queue finite shutdown behavior.

All concurrency tests use events, conditions, or futures. No arbitrary sleeps are
used.

## Self-review

- Removed a double-rejection counter path for registration errors and added a
  duplicate request-ID invariant test.
- Changed `_mark_failed()` to avoid nested metrics/state lock ordering, preventing
  a registration-commit versus failure-monitor lock inversion.
- Checked every `queue.get()` path for matching `task_done()` and every request
  dequeue/drain path for exactly one semaphore release.
- Checked coordinator crash, stopped coordinator, full completion queue, runtime
  exception, worker `BaseException`, cancellation, and blocked runtime paths for
  bounded return and canonical terminal ownership.
- Preserved existing coordinator APIs: registration callback and submit timeout
  are optional, and all previous call sites retain their behavior.

## Verification

- Focused:
  `HF_DATASETS_CACHE=/tmp/ml-hw-hf-datasets .../.venv/bin/python -m pytest tests/test_async_engine.py tests/test_async_completion.py tests/test_async_metrics.py -q`
  - Result before final commit: `68 passed in 0.46s`.
- Concurrency repetition:
  `tests/test_async_engine.py` ran 20 consecutive times successfully before the
  final edge-case additions.
- Preliminary full suite:
  `300 passed, 13 skipped, 1 warning in 26.76s`.
  The warning is the pre-existing unknown `integration` mark in
  `tests/test_ettm_loader.py`.
- Final fresh focused:
  `68 passed in 0.42s`.
- Final fresh full suite:
  `305 passed, 13 skipped, 1 warning in 26.90s`.
  The sole warning is the same pre-existing unknown `integration` mark.
