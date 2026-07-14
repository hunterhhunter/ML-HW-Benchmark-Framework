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

## Review revision

### Scope and feasibility interpretation

- Revision parent: `08f88b986f197ae908fca8b354bf18f3d79b89b8`.
- Applied every item in `.superpowers/sdd/task-5-review-findings.md` and updated
  the global design specification where its unconditional payload-cleanup and
  callback-error wording was impossible for an uninterruptible external Python
  callback.
- The supported callback-hang contract is now explicit: `shutdown()` returns
  `False` by its deadline, the engine remains `FAILED`, nonterminal request IDs
  remain in the outstanding diagnostic, and the enclosing CLI must report a
  non-zero result. The coordinator does not invent a terminal outcome or try to
  mutate an arbitrary blocked callback stack. If the callback later returns,
  the existing coordinator commits that request exactly once and releases its
  framework-owned references. Process isolation and cooperative callback
  cancellation remain outside the current core scope.
- No subagents were used.

### Review RED / GREEN record

1. Idle payload references
   - RED: after a successful terminal flush, weak references to the request,
     collated runtime input, and runtime output remained alive.
   - Root cause: the worker's `for _ in batch` loop variable retained the final
     request and the coordinator's `_run.item` retained the final completion
     while both threads idled.
   - GREEN: worker payload locals and the coordinator item are explicitly
     cleared after ownership/terminal handoff; the weak-reference regression
     observes all four payload references collected before shutdown.

2. Worker-owned incompatible pending cancellation
   - RED: while request 0 was blocked in the runtime, incompatible request 1 had
     left the request queue for the worker's private pending slot;
     `cancel_queued()` returned 0 and request 1 later completed successfully.
   - GREEN: each worker publishes its pending request in a lock-protected,
     bounded registry. Cancellation and the worker atomically compete to claim
     it, and cancellation now returned 1 with one completed and one canonical
     `CancelledError` terminal request. Queue task and semaphore counts balance.

3. Complete dynamic-batch compatibility
   - RED: requests could not represent task, generation options, or a declared
     batch axis; differing LLM generation options coalesced. Inputs with shapes
     `(1, 3)` and `(2, 3)` failed in `np.stack`, and two two-sample requests
     produced a runtime batch of 4 despite `max_batch_size=3`.
   - GREEN: compatibility includes normalized task, recursively frozen
     generation options, input names, dtype, declared batch axis, and shape with
     only that axis removed. Different LLM options seal separate batches.
     Compatible prebatched arrays concatenate on the declared axis and seal when
     the sum of actual `sample_count` would exceed `max_batch_size`.

4. Publication timestamp and queue-depth linearization
   - RED: a gated registration proved that `enqueued_ns` was assigned before
     actual bounded-queue publication, and queue depth was predicted outside the
     queue's publication critical section.
   - GREEN: `_RequestQueue.publish()` assigns the monotonic enqueue timestamp,
     inserts the request, derives exact depth, and records accepted metrics under
     the queue mutex. The coordinator registration condition still surrounds
     that publication, preserving accepted-before-terminal ordering for a
     zero-latency worker. Callback failure rolls back the invisible tail item.

5. Cancel/shutdown sentinel race
   - RED: an event-gated idle worker let shutdown enqueue `_STOP`; concurrent
     `cancel_queued()` removed it and shutdown timed out.
   - GREEN: a control lock publishes shutdown start before sentinel insertion;
     external cancellation thereafter returns 0 and cannot drain the sentinel.
     The deterministic race now reaches `STOPPED` with zero unfinished tasks.

6. Completion queue contract
   - RED: the engine accepted both an unbounded completion queue (`maxsize=0`)
     and a capacity smaller than `worker_count`.
   - GREEN: construction rejects non-positive capacity and requires the
     completion queue capacity to equal the configured worker count.

7. Actual static-batched size
   - RED: a static request representing two samples recorded batch metric and
     trace sizes of 1.
   - GREEN: worker busy metrics, `BatchCompletion`, and request traces use the
     sum of request `sample_count`; two atomic requests now produce runtime,
     evaluator, metric, and trace sizes `[2, 2]`.

8. Multi-worker and completion-stall semantics
   - A deterministic two-worker characterization releases request 1 before
     request 0 and observes terminal trace order `[1, 0]`, two exact terminals,
     and zero outstanding requests.
   - RED: the blocked evaluator contract had no public bounded snapshot of the
     outstanding IDs.
   - GREEN: while the evaluator gate is closed, shutdown returns `False`, state
     is `FAILED`, outstanding IDs are `(0,)`, and completed/failed counters stay
     at zero. Releasing the gate produces exactly one completed terminal and an
     empty outstanding set, without a fabricated failure terminal.

### Review verification

- Final focused command:
  `HF_DATASETS_CACHE=/tmp/ml-hw-hf-datasets .../framework/.venv/bin/python -m pytest tests/test_async_engine.py tests/test_async_completion.py tests/test_async_metrics.py -q`
  - Result: `80 passed in 0.74s`.
- Final concurrency repetition:
  `tests/test_async_engine.py` ran 10 consecutive times after all revisions.
  - Result: `300 passed` total; every run reported `30 passed`.
- Final full suite:
  `HF_DATASETS_CACHE=/tmp/ml-hw-hf-datasets .../framework/.venv/bin/python -m pytest tests -q`
  - Result: `317 passed, 13 skipped, 1 warning in 27.27s`.
  - The only warning remains the pre-existing unknown `integration` mark in
    `tests/test_ettm_loader.py`.

## Review round 2 revision

### Scope

- Revision parent: `b2dbf0b34891c2f146a5a2493d29056fba69a22c`.
- Applied every item in `.superpowers/sdd/task-5-review-r2-findings.md` using
  deterministic event/condition/fake-clock regressions and no arbitrary sleeps.
- Extended the design specification only to make the resulting queue-depth,
  deadline, request-validation, and accepted-metric claim contracts explicit.
- No subagents were used.

### Round 2 RED / GREEN record

1. Candidate ownership from dequeue
   - RED: request 1 left the queue and blocked in `np.asarray()` while its
     compatibility key was computed. `cancel_queued()` returned 0 because the
     request existed only in the worker's local `candidate`; it later remained
     eligible for successful execution.
   - Root cause: the worker published the request to its pending registry only
     after compatibility/cap calculations, leaving cancellation no claimable
     owner during those calculations.
   - GREEN: `_RequestQueue.get_candidate()` moves a dequeued request into the
     worker's bounded claimable registry while still holding the queue mutex.
     Worker commit and cancellation now atomically compete to pop that registry
     entry. The gated regression reports one completed request and one canonical
     cancellation terminal, with zero outstanding/unfinished tasks and all
     reservation slots returned.

2. Actual per-request batch validation
   - RED: a declared axis length of 2 with `sample_count=1`, a three-sample
     request under `max_batch_size=2`, and a static three-sample request under a
     runtime cap of 2 were all accepted and reached the worker/runtime path.
   - GREEN: submission validates positive integral `sample_count`, declared
     input-axis lengths, static leading batch lengths, configured maximum, and
     runtime maximum before slot reservation or coordinator registration.
     Invalid requests become one `invalid_request` rejection: runtime/evaluator
     counts remain zero and submitted/accepted/rejected/outstanding invariants
     remain valid. Valid prebatched requests still report the same runtime,
     evaluator, worker, trace, and completed-sample counts.

3. Linearized dequeue depth
   - RED: the request queue had no operation that combined removal and exact
     post-removal depth observation (`take` was absent); engine workers used
     separate `get()` and `qsize()` calls, so a concurrent publication could
     hide the intervening zero/lower-depth transition.
   - GREEN: queue `take` performs removal and its depth callback under the same
     mutex. A concurrent publisher is deterministically held until that callback
     finishes, producing the exact event sequence `1 -> 0 -> 1`. Candidate claim
     registration shares the same removal critical section.

4. One shutdown deadline without cancel serialization
   - RED: with a fake clock, shutdown waited for the control lock and produced
     deadline 201 instead of the invocation deadline 101. A separately gated
     cancel held that lock through completion submission, preventing shutdown
     from reaching flush; a cancel that passed the check first also conflicted
     with later sentinel publication.
   - GREEN: shutdown creates its absolute deadline on its first line. External
     cancellation holds the control lock only for the `_shutdown_started` check,
     then drains/submits outside it. Queue drain atomically retains `_STOP`, so a
     pre-existing cancel cannot steal a later shutdown sentinel. Fake-clock,
     non-serialization, and sentinel interleaving regressions all pass without
     adding two timeout windows serially.

5. Accepted metrics partial-mutation safety
   - RED: a collector incremented `accepted` and then raised before updating its
     gauges. Queue/coordinator publication rolled back and submission rejected
     the same request, leaving accepted plus rejected counts for one submitted
     request and a false positive outstanding count.
   - GREEN: metrics opens an explicit per-thread acceptance claim with the
     pre-publication accepted snapshot. Normal `record_accepted()` marks commit;
     claim resolution also detects an accepted counter delta from a partially
     mutating collector. Once committed, the engine never decrements accepted or
     records a rejection: it preserves `counter_invariant_failed`, keeps the
     request published, and terminalizes it exactly once. Failure before any
     accepted change rolls queue/coordinator state back and records one
     `metrics_unavailable` rejection. No unsafe metric rollback is used.

6. Actual failure/cancel batch size
   - RED: cancellation of requests representing two and three samples produced
     trace batch sizes `[2, 2]`, the count of request objects.
   - GREEN: every engine-created failure/cancellation `BatchCompletion` now uses
     `sum(request.sample_count)`. The same regression records `[5, 5]`, two
     failed requests, five failed samples, and zero outstanding requests.

### Round 2 verification

- Focused Task 4/5 set:
  `tests/test_async_types.py tests/test_async_engine.py tests/test_async_completion.py tests/test_async_metrics.py`
  - Result: `97 passed in 0.80s`.
- Final concurrency repetition:
  `tests/test_async_engine.py` ran 10 consecutive times after the R2 changes.
  - Result: `400 passed` total; every run reported `40 passed`.
- Full framework suite:
  `HF_DATASETS_CACHE=/tmp/ml-hw-hf-datasets .../framework/.venv/bin/python -m pytest tests -q`
  - Result: `327 passed, 13 skipped, 1 warning in 27.35s`.
  - The warning remains the pre-existing unknown `integration` mark in
    `tests/test_ettm_loader.py`.

## Review round 3 revision

### Scope

- Revision parent: `16a14e4f76cbdcc94331742123497c38be9d16a1`.
- Applied every item in `.superpowers/sdd/task-5-review-r3-findings.md` with
  event-gated exception/block/re-entry tests, explicit queue transition
  timestamps and sequences, and a gated late-sentinel regression.
- Updated only the async engine/completion/metrics implementation and tests,
  plus the approved design specification and this report. No subagents were
  used.

### Round 3 RED / GREEN record

1. Queue transition callbacks outside the queue mutex
   - RED: the new focused engine regressions produced six failures. `take()`
     had no callback-free transition result; a gated dequeue metric held the
     queue mutex long enough for shutdown to exceed its deadline; first and
     candidate metric exceptions lost worker ownership and left flush false;
     and a drain metric exception escaped before slot/cancellation cleanup.
   - Root cause: `_RequestQueue._take()` and `drain_requests()` invoked
     failure-prone metrics callbacks while holding the queue's non-reentrant
     mutex. The first request had not yet been published into worker-local
     ownership, and the candidate's worker state had not observed its pending
     claim when the callback raised.
   - GREEN: request removal/drain now captures an immutable depth,
     `monotonic_ns`, and sequence under the mutex and returns it without
     invoking metrics. Workers establish first/pending ownership and release
     reservation slots before recording the transition. Drain completes task
     accounting and slot release before metrics, converting a metrics failure
     to `metrics_unavailable` without dropping canonical cancellation.
     Callback exceptions terminalize every owned request once; callback block
     and queue re-entry no longer retain the queue mutex.

2. Exact queue-depth time and order
   - RED: moving a callback outside the mutex would otherwise allow a later
     publication callback to reach the collector first, while recording the
     callback execution time would lengthen the preceding queue state.
   - GREEN: every request queue transition receives a mutex-linearized
     sequence and actual transition timestamp. The collector buffers
     out-of-order sequence observations and applies them in transition order.
     A deterministic reverse-delivery test reconstructs the expected
     time-weighted depth mean of `0.3` from events at 2 ms and 5 ms.

3. Late sentinel ownership after shutdown cleanup
   - RED: a worker dequeued `_STOP` and paused before `_pass_stop_token()`;
     shutdown returned with an empty queue but `unfinished_tasks == 1`, after
     which the worker could repost a new sentinel into the cleaned queue.
   - GREEN: `_RequestQueue.take()` balances a dequeued sentinel immediately.
     Shutdown marks a terminal epoch under the control lock before final drain,
     and `_pass_stop_token()` checks that epoch while serialized with repost.
     The gated regression observes an empty queue and zero unfinished tasks
     both at shutdown return and after the worker resumes and exits.

4. Synthetic failure trace batch size
   - RED: stop and completion-thread crash traces for requests representing
     multiple samples still reported `batch_size=1`.
   - GREEN: coordinator-synthesized failure traces use each request's
     `sample_count`. Stop now reports 3, and crash cleanup reports `[2, 3, 4]`.

### Round 3 verification

- Focused Task 4/5 set:
  `framework/tests/test_async_types.py framework/tests/test_async_engine.py framework/tests/test_async_completion.py framework/tests/test_async_metrics.py`
  - Result: `105 passed in 0.94s`.
- Final concurrency repetition:
  `framework/tests/test_async_engine.py` was collected and run 10 times in one
  pytest process with `--keep-duplicates`.
  - Result: `470 passed in 7.34s`.
- Full framework suite:
  `HF_DATASETS_CACHE=/tmp/ml-hw-hf-datasets .../framework/.venv/bin/python -m pytest tests -q`
  - Result: `335 passed, 13 skipped, 1 warning in 27.42s`.
  - The warning remains the pre-existing unknown `integration` mark in
    `tests/test_ettm_loader.py`.

## Review round 4 revision

### Scope

- Revision parent: `df045f60c1d95885c38d2b48b4787e59f80ad285`.
- Applied every item in `.superpowers/sdd/task-5-review-r4-findings.md` with
  deterministic event/future/fake-clock tests and no arbitrary sleeps.
- Replaced wait-for-next queue metric buffering, introduced an explicit
  submission reservation transaction, added terminal queue closure/broadcast,
  and moved candidate transition capture ahead of pending ownership locking.
- Updated the approved design specification and this report. The temporary R4
  execution plan was intentionally excluded from the delivered diff.
- No subagents were used.

### Round 4 RED / GREEN record

1. Independent queue transition evidence
   - RED: four new metric regressions failed because the collector had no
     missing/failed/duplicate diagnostics and retained every later transition
     in a wait-for-next buffer after a missing sequence. Finalization could
     therefore expose a numeric queue-depth result without proving that the
     sequence was complete.
   - Root cause: transition delivery order and transition completeness were
     conflated in `_pending_queue_transitions`; no explicit record survived a
     callback exception.
   - GREEN: transitions are stored once by sequence and sorted only at
     finalization. Missing ranges, failed sequences, identical duplicates, and
     conflicting duplicates are reported separately. Missing, failed, mixed,
     or conflicting evidence marks `metrics_unavailable` and returns `None` for
     depth min/max/mean. One hundred events following a failed sequence occupy
     exactly one hundred event entries rather than an additional blocked
     backlog. The complete metric module reports `21 passed`.

2. Acceptance preflight and bounded submission transaction
   - RED: three deterministic acceptance regressions either deadlocked when the
     accepted callback re-entered `request_queue.qsize()` or could not let
     close/shutdown honor their deadline while that callback was event-blocked.
   - Root cause: the failure-prone accepted callback ran while engine state,
     coordinator, and non-reentrant request queue locks were held.
   - GREEN: a slot-bounded transaction reserves the coordinator request, then
     runs a non-mutating metrics preflight outside every lifecycle lock. A
     short commit revalidates RUNNING ownership, captures the actual visible
     publication timestamp/depth/sequence, commits internal accepted accounting
     and coordinator outstanding ownership, and only then relinquishes the
     transaction. Shutdown can cancel a blocked preflight's slot/reservation;
     its late return becomes one stale rejection. Queue re-entry, blocked
     preflight versus shutdown, and close versus stale submit all pass.

3. Acceptance rollback sequence evidence
   - RED: a failure before accepted accounting rolled back the queue item and
     also decremented the queue transition sequence. The next accepted request
     reused sequence 1, so finalization falsely declared a complete queue-depth
     history with no evidence of the failed publication attempt.
   - GREEN: publication rollback never reuses a captured sequence. It records
     the failed sequence explicitly, so a later successful request leaves
     `failed_sequences == [1]`, `missing_sequence_ranges == [[1, 1]]`, and no
     misleading queue-depth statistic. Both the pre-commit rollback and
     post-mutation accepted-accounting regression pass.

4. Terminal broadcast for multiple idle workers
   - RED: with two workers, one worker could own the physical sentinel and pause
     after shutdown's terminal epoch. The second idle worker remained blocked
     forever because the late owner was correctly forbidden to repost.
   - Root cause: worker termination depended on serial sentinel handoff, but the
     terminal epoch deliberately disabled the only wakeup path left for other
     waiters.
   - GREEN: terminal shutdown closes the request queue and broadcasts its
     `not_empty` condition. A closed empty queue returns a virtual terminal
     signal without task accounting, so every waiter exits independently. The
     late owner, second waiter, queue emptiness, and zero unfinished-task
     assertions all pass.

5. Candidate transition timestamp
   - RED: a fake-clock test observed candidate removal at time 20 but received
     transition time 30 after its pending ownership callback was released.
   - Root cause: `_capture_transition()` ran after the callback acquired the
     separately contended pending lock.
   - GREEN: the transition timestamp and sequence are captured immediately
     after `_get()` while still under the queue mutex, before publishing pending
     ownership. The regression now retains time 20.

### Round 4 self-review

- Rechecked the submission lock order (`state -> coordinator -> queue`) and
  verified that failure-prone preflight, rejection accounting, queue-depth
  delivery, and exception logging occur after lifecycle locks are released.
  The publication critical section contains only the collector's internal
  accepted commit required for worker-visibility atomicity.
- Rechecked every transaction exit for exactly one reservation abort/commit and
  semaphore release/transfer, including shutdown cancellation followed by a
  late preflight return.
- Rechecked worker terminal paths: a physical sentinel is accounted at dequeue,
  virtual closure has no unfinished task, and terminal cleanup cannot repost.
- Rechecked queue metric failure paths: a captured sequence is never reused;
  callback failures record explicit evidence and later events remain
  independently collectable.
- Rechecked the changed tests for time sleeps; all synchronization uses events,
  conditions, futures, or fake clocks.

### Round 4 verification

- Final engine module:
  `.../.venv/bin/python -m pytest -q framework/tests/test_async_engine.py`
  - Result: `53 passed in 0.89s`.
- Focused Task 4/5 set:
  `.../.venv/bin/python -m pytest -q framework/tests/test_async_types.py framework/tests/test_async_engine.py framework/tests/test_async_completion.py framework/tests/test_async_metrics.py`
  - Result: `115 passed in 0.95s`.
- Concurrency repetition:
  `framework/tests/test_async_engine.py` was collected and run 10 times in one
  pytest process with `--keep-duplicates`.
  - Result: `530 passed in 8.53s`.
- Full framework suite:
  `HF_DATASETS_CACHE=/tmp/ml-hw-hf-datasets .../framework/.venv/bin/python -m pytest tests -q`
  - Result: `345 passed, 13 skipped, 1 warning in 27.48s`.
  - The sole warning remains the pre-existing unknown `integration` mark in
    `tests/test_ettm_loader.py`.
