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

## Review round 5 revision

### Scope

- Revision parent: `fc8c9f79fae3d76d10c82f8dc13a58f4486d4a50`.
- Applied every item in `.superpowers/sdd/task-5-review-r5-findings.md` with
  deterministic event/condition tests and no arbitrary sleeps.
- Replaced subclass-dispatch acceptance accounting with sealed module-private
  primitives, recorded queue sequence high-water at allocation, latched
  finalize-time missing evidence, and made shutdown cancellation commit an
  immediate exact-once rejection.
- Audited the broader lock rule and moved pre-existing completion timeout,
  stop, crash, and membership diagnostic callbacks outside the coordinator
  condition as well.
- Updated only the approved design specification and this report in addition
  to the implementation/tests. The temporary R5 execution plan was deleted
  before delivery. No subagents were used.

### Round 5 RED / GREEN record

1. Sealed acceptance accounting
   - RED: the initial seven-test R5 target run produced seven failures. A
     collector overriding public `claim_acceptance`, `commit_acceptance`, and
     `finish_acceptance` entered the hook while state/coordinator/queue locks
     were held; its event gate prevented submit completion and queue `qsize()`
     re-entry waited behind the non-reentrant mutex. Existing partial/failing
     public commit overrides also changed publication outcomes.
   - Root cause: R4 called methods named as an internal commit but still used
     ordinary subclass dispatch for all three acceptance operations.
   - GREEN: queue publication now calls a metrics-module free function directly.
     It commits accepted, inflight, exact queue transition, and expected
     sequence state without public methods, callbacks, logging, or override
     lookup. Public metrics APIs retain compatibility wrappers but engine code
     never calls them in the lifecycle commit. The override/re-entry target set
     reports `4 passed` and both former failure-injection overrides remain
     uncalled while their requests complete normally.

2. Allocation-time queue sequence high-water
   - RED: an allocated-only metrics test and an engine whose final dequeue
     delivery was event-blocked both returned `sequence_valid == True` with
     numeric depth statistics. With no later sequence, R4 had no evidence that
     the trailing callback was missing.
   - Root cause: maximum expected sequence was derived only from delivered or
     explicitly failed events rather than the request queue's allocation point.
   - GREEN: every dequeue/drain sequence allocation records high-water through a
     sealed primitive before failure-prone delivery. Accepted publication
     commits high-water and its event atomically in the same metrics critical
     section, avoiding a false transient gap. Finalize compares all events to
     high-water, latches any observed missing ranges, adds
     `metrics_unavailable`, and keeps min/max/mean `None` on repeated finalize
     even after late delivery. The complete metrics plus blocked engine target
     reports `23 passed`.

3. Immediate shutdown rejection
   - RED: while preflight was blocked, shutdown reclaimed its reservation and
     slot but returned with submitted 1, accepted 0, and rejected 0. The counter
     invariant stayed invalid until the callback returned; a permanently
     blocked callback could never repair it.
   - Root cause: R4 stored only a cancellation flag and delegated rejection to
     the late submitter path instead of making cancellation terminal.
   - GREEN: each bounded transaction has a one-way `pending -> accepted` or
     `pending -> rejected` state. Shutdown changes every remaining preflight to
     rejected, removes registry ownership, aborts the reservation, releases the
     slot, and commits the sealed rejected counter exactly once before return.
     Late callback return sees terminal state and cannot publish or reject
     again. Blocked, permanently blocked, and close/late-return targets report
     `3 passed`; shutdown snapshots satisfy submitted=accepted+rejected.

4. Public completion metrics outside coordinator condition
   - The first test attempt used nonblocking acquisition and passed because
     Python `Condition` defaults to a reentrant lock; this did not test the
     requirement and was immediately replaced rather than treated as RED.
   - RED: the corrected collector checked condition ownership and failed at
     `wait_for_requests()` timeout because `add_invalid_reason()` was invoked
     while the coordinator condition was owned.
   - GREEN: timeout/error state is captured under the condition and metrics are
     recorded after release. The same treatment covers stop queue failure,
     coordinator crash, claimed-terminal collision, and duplicate/unknown
     completion membership. The completion module reports `35 passed`.

5. Exception-path ownership cleanup
   - Evidence from the initial blocked public-hook RED exposed a second R4
     issue: the outer submit exception handler released the slot directly and
     then `_abort_submission()` released the same transaction-owned slot again,
     producing `Semaphore released too many times`.
   - GREEN: all post-reservation failures now share the terminal rejection
     helper, which transfers reservation and slot ownership once. The outer
     handler derives accepted/rejected state from the transaction and never
     performs a second release.

### Round 5 self-review

- Rechecked every engine state, coordinator condition, and request queue mutex
  region. Locked metrics writes are limited to the three imported module-private
  primitives; every public/subclass-dispatch call and logger is outside those
  lifecycle locks.
- Rechecked sequence allocation paths: accepted publication records high-water
  atomically with its event; first/candidate dequeue and drain record it before
  their public delivery callback. Physical/virtual stop signals allocate no
  request-depth sequence.
- Rechecked transaction races: shutdown and late submit atomically compete on
  `terminal_state`; only the winner from pending may release ownership and
  increment rejected. Accepted transactions cannot be cancelled.
- Rechecked repeated finalize behavior: detected missing ranges are merged into
  bounded range evidence and never removed, while independently delivered event
  storage remains one entry per sequence.
- Rechecked all changed concurrency tests for sleeps; synchronization uses
  events, conditions, futures, or lock ownership checks. No subagents were used.

### Round 5 verification

- Final engine module:
  `.../.venv/bin/python -m pytest -q framework/tests/test_async_engine.py`
  - Result: `56 passed in 0.97s`.
- Completion module after the lock audit:
  `.../.venv/bin/python -m pytest -q framework/tests/test_async_completion.py`
  - Result: `35 passed in 0.10s`.
- Final focused Task 4/5 set:
  `.../.venv/bin/python -m pytest -q framework/tests/test_async_types.py framework/tests/test_async_engine.py framework/tests/test_async_completion.py framework/tests/test_async_metrics.py`
  - Result: `120 passed in 1.05s`.
- Concurrency repetition:
  `framework/tests/test_async_engine.py` was collected and run 10 times in one
  pytest process with `--keep-duplicates`.
  - Result: `560 passed in 9.16s`.
- Full framework suite:
  `HF_DATASETS_CACHE=/tmp/ml-hw-hf-datasets .../framework/.venv/bin/python -m pytest tests -q`
  - Result: `350 passed, 13 skipped, 1 warning in 27.49s`.
  - The sole warning remains the pre-existing unknown `integration` mark in
    `tests/test_ettm_loader.py`.

## Review round 6 revision

### Scope

- Revision parent: `c8f4993f7d381d8c2e1c73451b8fd4b8c1e7ac9d`.
- Applied every item in `.superpowers/sdd/task-5-review-r6-findings.md`
  with deterministic event/future/lock-ownership tests and no arbitrary
  sleeps.
- Replaced collector-owned accounting with an identity-keyed module registry,
  a private per-collector lock, sealed counters/invalid evidence, primitive
  inflight and legacy queue gauges, and sealed queue transition/high-water
  evidence.
- Moved stop-enqueue and duplicate first-token diagnostics outside their
  lifecycle locks. Updated only the approved design specification and this
  report in addition to implementation/tests. No temporary plan file was
  created and no subagents were used.

### Round 6 RED / GREEN record

1. Sealed accounting state and private lock
   - RED: the five-test R6 target produced five failures. Shutdown exceeded its
     deadline while a preflight held `metrics.lock`; accepted publication
     dispatched to an injected `metrics.inflight.update()` and raised after
     entering the queue critical section; mutating the public counter object
     changed accepted from 1 to 100; stop-enqueue diagnostics re-entered a held
     control lock; and duplicate first-token diagnostics kept the tracker lock
     while gated.
   - Root cause: the R5 module functions bypassed method overrides but still
     stored their lock, counters, and inflight gauge on the extensible collector
     object. The accepted transaction therefore retained replaceable dispatch
     and a partial-mutation boundary between the accepted counter and gauge.
   - GREEN: a weak identity registry owns one private state and lock per live
     collector. Accepted/rejected/submitted/terminal counters, invalid evidence,
     inflight arithmetic, queue arithmetic, transitions, failed/duplicate
     evidence, latched missing ranges, and sequence high-water live only in
     that state. Public counter access returns an isolated snapshot, while the
     public lock and replaceable inflight object are never read by internal
     accounting, base methods, or finalize. The registry keeps no strong
     collector reference and verifies the weak reference before identity reuse.

2. Transactional direct arithmetic
   - The first GREEN implementation still kept queue transition/high-water
     fields on the collector. Self-review caught that accepted publication
     could therefore increment sealed accepted/inflight state and then touch a
     replaceable queue-evidence object.
   - The final implementation moved all queue evidence into the same sealed
     state before final verification. Internal acceptance now resolves state,
     takes its private lock, and directly updates only module-owned primitive
     values and built-in collections. Rejection has the same property. There is
     no public method, logger, collector field, extension lock, or replaceable
     gauge call in either lifecycle primitive.

3. Public lock and inflight adversaries
   - A preflight acquires the public collector lock and holds it across the
     complete shutdown/finalize assertion. Shutdown still returns by its 50 ms
     configured deadline, commits submitted=1/rejected=1/outstanding=0, and
     finalize returns while that public lock remains owned.
   - A separate accepted path replaces the public inflight object with one that
     raises from both `update()` and `summary()`. Submit, terminal completion,
     shutdown, and finalize complete with accepted=completed=1, inflight max=1,
     and zero calls to the injected object.
   - A public counter snapshot is mutated to accepted=terminal=99 before normal
     base-method accounting. Finalization still returns accepted=completed=1
     with valid invariants.

4. Remaining lifecycle callbacks
   - RED: `_enqueue_stop()` called `add_invalid_reason()` while `_control_lock`
     was held; a deterministic override could not acquire that lock
     nonblocking. `FirstTokenTracker.record()` called the same public method
     while its tracker lock was held; a concurrent valid finalize remained
     blocked behind the gated override.
   - GREEN: `_enqueue_stop()` returns only success/failure evidence, shutdown
     closes the queue and releases `_control_lock`, then records
     `worker_shutdown_failed`. First-token record captures invalidity under its
     lock and records the diagnostic after release. The control-lock re-entry
     succeeds and tracker finalize completes while the diagnostic remains
     gated.

### Round 6 self-review

- Rechecked lock order. Registry lookup holds only the registry lock and
  releases it before callers acquire the sealed state lock. Queue publication
  and allocation use `request queue mutex -> sealed state`; no sealed-state
  path acquires the request queue, public collector, engine control, tracker,
  or coordinator lock in reverse.
- Rechecked the internal accepted/rejected functions structurally: both resolve
  module state before mutation; neither reads `metrics.lock`,
  `metrics.counters`, `metrics.inflight`, another collector field, a public
  method, or a logger. Queue transition/high-water and accepted/inflight commit
  under one private lock.
- Rechecked weak registry lifetime and identity reuse: state holds no collector
  reference, the cleanup callback captures only integer identity, and cleanup
  removes an entry only when its exact weak-reference object is still current.
- Rechecked public base methods and finalize: all counters, invalid evidence,
  measurement start, queue evidence, and inflight values use the same sealed
  state. Public counter/invalid-reason reads are copies rather than mutation
  handles.
- Rechecked changed concurrency tests for sleeps. Synchronization uses events,
  futures, nonblocking lock acquisition, or configured queue/deadline waits.
  The code-review checklist was performed locally because the task explicitly
  prohibited subagents; no critical or important issue remained before final
  verification.

### Round 6 verification

- Final engine module:
  `.../.venv/bin/python -m pytest -q framework/tests/test_async_engine.py`
  - Result: `59 passed in 1.14s`.
- Completion module:
  `.../.venv/bin/python -m pytest -q framework/tests/test_async_completion.py`
  - Result: `36 passed in 0.09s`.
- Final focused Task 4/5 set:
  `.../.venv/bin/python -m pytest -q framework/tests/test_async_types.py framework/tests/test_async_engine.py framework/tests/test_async_completion.py framework/tests/test_async_metrics.py`
  - Result: `125 passed in 1.09s`.
- Concurrency repetition:
  `framework/tests/test_async_engine.py` was collected and run 10 times in one
  pytest process with `--keep-duplicates`.
  - Result: `590 passed in 9.79s`.
- Full framework suite:
  `HF_DATASETS_CACHE=/tmp/ml-hw-hf-datasets .../framework/.venv/bin/python -m pytest tests -q`
  - Result: `355 passed, 13 skipped, 1 warning in 27.53s`.
  - The sole warning remains the pre-existing unknown `integration` mark in
    `tests/test_ettm_loader.py`.

## Review round 7 revision

### Scope and RED / GREEN evidence

- Revision parent: `233e24256ce903b80d12094fdde299cc54615fd6`.
- Four deterministic R7 targets initially failed: an accepted commit
  `BaseException` rolled the queue back and added rejected after accepted had
  changed; a rejection exception escaped after transaction ownership was
  already released; a self-referencing `str` subclass retained its collector
  through registry evidence; and replaceable worker aggregates dispatched
  while the sealed lock was held.
- Accepted/rejected now first install one authoritative request-ID outcome in
  sealed membership. Derived counters, inflight history, queue transitions and
  reason counts rebuild idempotently from that membership. Queue publication
  queries the outcome after any `BaseException`: committed acceptance keeps the
  queue item and completes coordinator/transaction ownership, while absence of
  acceptance permits rollback. Rejection records and resolves its outcome
  before reservation/slot/transaction ownership is released; retry uses the
  same request ID and cannot double count.
- Every externally supplied value is converted before the sealed lock to exact
  built-in `int`, `float`, or `str`. Registry state uses plain exact
  `dict`/`list`/`set`/`tuple` containers rather than `Counter` or collector-owned
  aggregates. A self-referencing string plus numeric subclasses no longer keep
  the collector alive, and the weakref callback removes its identity entry.
- Worker, terminal, batch, timing, generation and failure aggregates moved to
  module-owned primitives. Finalize performs only module-private primitive
  calculations and immutable built-in snapshot capture under the sealed lock;
  percentile and schema formatting run after release. Replaced public timing,
  batch, worker and error aggregate objects are never dispatched.
- The approved design spec and this report were appended. No temporary plan
  file or subagent was used.

### Round 7 lock and exception review

- Registry lookup releases its global lock before the per-collector lock.
  Queue publication remains `queue mutex -> sealed state`; no sealed aggregate
  or finalize path calls the queue, public collector lock, subclass method, or
  replaceable summary object in reverse.
- Outcome insertion is the single commit point. Normalization happens before
  it; later rebuild operations are repeatable from exact records. Engine
  recovery never maps an accepted outcome to rejected or terminalizes a
  rejection without matching accounting membership.
- Sealed state holds no collector reference. Weakref cleanup compares the exact
  weakref object before deleting an identity, protecting ID reuse.
- Local requirements-focused review was used because R7 explicitly prohibited
  subagents. No critical or important issue remained before the fresh final
  matrix.

### Round 7 verification

- Final focused Task 4/5 set:
  `.../.venv/bin/python -m pytest -q framework/tests/test_async_types.py framework/tests/test_async_engine.py framework/tests/test_async_completion.py framework/tests/test_async_metrics.py`
  - Result: `129 passed in 1.32s`.
- Concurrency repetition: the engine module was collected ten times in one
  process with `--keep-duplicates`.
  - Result: `610 passed in 9.91s`.
- Full framework suite:
  `HF_DATASETS_CACHE=/tmp/ml-hw-hf-datasets .../framework/.venv/bin/python -m pytest tests -q`
  - Result: `359 passed, 13 skipped, 1 warning in 27.83s`.
  - The sole warning remains the pre-existing unknown `integration` mark in
    `tests/test_ettm_loader.py`.

## Review round 8 revision

### Scope and RED / GREEN evidence

- Revision parent: `1a0a1c16ffff2d66515af508b16158976c4a2e08`.
- The first R8 target exposed all three requested accounting regressions:
  two rejected attempts with request ID 7 produced rejected=1 instead of 2;
  a rejected request ID 9 could not be accepted on its next attempt; and an
  empty metrics snapshot emitted ten internal zero-default keys in
  `details.counts` instead of the prior `{}` shape.
- GREEN: the engine allocates a monotonic submission-attempt token while it
  admits each submitter. Sealed outcomes use that token as their key and keep
  request ID as normalized payload data. Accepted and rejected attempts with
  the same request ID no longer collide, while coordinator membership still
  rejects an actually reserved, outstanding, or terminal duplicate as its own
  attempt. Finalize copies emitted counters before adding private zero defaults
  for invariant arithmetic.
- Rejected outcome records now contain the normalized reason and exact
  `request_rejected` evidence. Every rebuild clears and reconstructs rejected
  counts and that evidence from authoritative outcome membership together.
  Deterministic before/after rebuild faults prove both the reason counter and
  invalid evidence are restored.
- A post-publication fault initially reproduced an `AttributeError` while
  accepted recovery tried to coordinator-commit `None`: the queue item was
  visible and accounting was accepted, but the external wrapper raised before
  returning the queued payload to the engine. GREEN stores the exact queued
  payload on the transaction inside the queue critical section, before the
  accepted commit can become externally ambiguous.
- Transaction terminal-stage tests initially failed because terminal flag and
  registry removal were one inline block with no independently retryable
  stages. GREEN adds idempotent terminal-mark and registry-removal operations,
  along with explicit coordinator-commit, reservation-abort, and slot-release
  stage flags. The outer `BaseException` handler always queries the attempt
  outcome first, then completes only matching accepted or rejected stages.
- Self-review found a further duplicate-ownership bug: when an original
  request was paused in metrics preflight with a reservation, a second attempt
  using the same request ID was rejected but aborted the original reservation.
  Its RED assertion observed an empty reservation map. GREEN records whether
  this attempt saw registration availability before reserve and only adopts an
  ambiguous reservation when this attempt could have created it. The duplicate
  remains rejected, the owner remains reserved, and the owner later accepts and
  completes.
- The complete deterministic fault matrix covers before and after queue
  publication, coordinator commit, transaction terminal mark, transaction
  registry removal on both accepted and rejected paths, reservation abort,
  slot release, and rejected-outcome rebuild/diagnostics. It also covers
  reject/reject, reject/accept, accepted duplicate, and reserved duplicate
  request IDs. Tests use events, futures, lock membership, and injected
  exceptions without arbitrary sleeps.
- The approved design specification and this report were appended. No
  temporary plan file or subagent was used.

### Round 8 lock and exception review

- Attempt-token allocation and transaction registry access remain under the
  engine state condition. Registration availability is captured while holding
  `state condition -> coordinator condition`, the existing submit lock order;
  the coordinator uses its reentrant condition for the reserve operation.
- Queue publication remains `state condition -> coordinator condition -> queue
  mutex -> sealed metrics state`. The queue stores transaction payload evidence
  before the sealed accepted outcome can be observed. Recovery never acquires
  those locks in reverse: it queries sealed outcome without an engine lock,
  resolves coordinator membership, then terminalizes under the state condition.
- Reservation abort ambiguity is resolved from coordinator membership and the
  attempt's pre-reserve availability. Slot release, terminal marking, and
  registry removal each have an idempotent transaction flag; a fault before an
  operation retries it and a fault after an operation observes the completed
  flag or authoritative coordinator membership.
- Rebuild uses only sealed exact built-in records under the private metrics
  lock. It derives accepted/rejected counts, per-reason rejected counts,
  inflight history, queue transitions, and `request_rejected` evidence without
  calling an extensible collector method.
- The first focused run caught one compatibility mistake in slot recovery: an
  unbound `BoundedSemaphore.release` bypassed the existing `TrackingSlots`
  wrapper and failed one prior crash test. The final implementation calls the
  configured slot object's normal release inside the idempotent stage; the
  prior test and all before/after slot faults then passed.

### Round 8 verification

- Changed-file compile check:
  `.../.venv/bin/python -m compileall -q framework/src/core/async_inference/engine.py framework/src/core/async_inference/metrics.py framework/tests/test_async_engine.py framework/tests/test_async_metrics.py`
  - Result: exit 0.
- Final engine module:
  `.../.venv/bin/pytest -q framework/tests/test_async_engine.py`
  - Result: `82 passed in 1.07s`.
- Final focused Task 4/5 set:
  `.../.venv/bin/pytest -q framework/tests/test_async_types.py framework/tests/test_async_engine.py framework/tests/test_async_completion.py framework/tests/test_async_metrics.py`
  - Result: `150 passed in 1.37s`.
- Concurrency repetition: the engine module was collected ten times in one
  process with `--keep-duplicates`.
  - Result: `820 passed in 10.27s`.
- Full framework suite:
  `HF_DATASETS_CACHE=/tmp/ml-hw-hf-datasets .../.venv/bin/pytest -q tests`
  - Result: `380 passed, 13 skipped, 1 warning in 27.92s`.
  - The sole warning remains the pre-existing unknown `integration` mark in
    `tests/test_ettm_loader.py`.

## Review round 9 revision

### Scope and RED / GREEN evidence

- Revision parent: `dcce005f0dc1e7199353a5353d48337a6a0ae38f`.
- Queue RED cases injected faults after `_put`, after transaction payload
  evidence, and after transition allocation. They exposed retained items,
  unfinished-task drift, and missing failed-sequence evidence. GREEN wraps the
  complete mutation window under the queue mutex. A pre-outcome fault removes
  the exact queued object, restores task accounting, clears payload evidence,
  and marks every allocated sequence failed; a committed accepted outcome
  preserves the item and task. Direct publication and accepted publication use
  the same rollback evidence.
- Accepted recovery initially raised an ownership-missing error when a fast
  completion had already set terminal membership and popped outstanding.
  GREEN treats any non-zero coordinator terminal bitmap entry as authoritative
  completed-registration evidence. Recovery finishes transaction terminal and
  registry stages while the original `BaseException` remains the exception
  observed by the caller.
- Slot RED cases injected after held membership removal and called rejection
  cleanup concurrently. GREEN replaces semaphore count plus release flags with
  a token-keyed lease pool. Capacity is `capacity - len(held)`, and release is
  one membership removal; retry and concurrent cleanup therefore cannot add
  capacity or retain a lease. The legacy `acquire`/`release` facade delegates
  to that same authoritative pool.
- Reservation ABA RED installed a replacement reservation with the same
  request ID before stale cleanup. GREEN stores the attempt token in each
  reservation and requires matching tokens for validate, commit, and abort.
  Accepted recovery also refuses to commit a replacement reservation.
- Shared-collector RED used two engines whose local attempt counters collided.
  GREEN moves attempt allocation into the collector's sealed accounting state.
  Both engines now commit distinct accepted outcomes with valid terminal and
  counter invariants.
- Numeric subclass guards proved request ID and attempt-token conversions no
  longer dispatch under engine/coordinator lifecycle locks or the sealed
  metrics lock. All outcome identity values are converted to exact built-in
  integers before those locks are acquired.
- Interruption tests cover `KeyboardInterrupt` and `SystemExit` at preflight,
  coordinator commit, reservation abort, slot removal, terminal mark, and
  registry pop boundaries. Cleanup resolves authoritative membership and
  retries remaining stages, then bare re-raises the original exception rather
  than returning `False`, swallowing it, or replacing it with a recovery
  exception.
- The approved design specification and this report were appended. No
  temporary plan file or subagent was used.

### Round 9 ownership and exception review

- Lock order remains lifecycle condition -> coordinator condition -> request
  queue mutex -> sealed metrics lock during publication. Outcome queries,
  reservation refresh, terminal marking, and lease release do not acquire that
  chain in reverse. Numeric extension conversion occurs before every sealed or
  coordinator lock that consumes identity values.
- Queue object identity, coordinator token membership, terminal bitmap,
  slot-token membership, and sealed outcome membership are the authoritative
  facts. Transaction fields retain progress and payload evidence but no longer
  duplicate slot capacity truth.
- Rejected cleanup can be entered concurrently. Token comparison protects a
  replacement reservation, lease membership makes duplicate release harmless,
  and terminal/registry transitions are idempotent under the lifecycle
  condition. Accepted cleanup cannot synthesize ownership from a reservation
  belonging to another attempt.
- The requesting-review checklist was performed locally because R9 explicitly
  prohibited subagents. The final audit added two more RED/GREEN guards for
  direct-publication sequence failure and pre-sealed-lock outcome identity
  normalization; no critical or important issue remained.

### Round 9 verification

- Changed-file compile check:
  `.../.venv/bin/python -m compileall -q` over all changed async source and test
  modules.
  - Result: exit 0.
- Final focused Task 4/5 set:
  `.../.venv/bin/pytest -q framework/tests/test_async_types.py framework/tests/test_async_engine.py framework/tests/test_async_completion.py framework/tests/test_async_metrics.py`
  - Result: `167 passed in 1.38s`.
- Concurrency repetition: the 96-test engine module was collected ten times in
  one pytest process with `--keep-duplicates`; terminal output was disabled to
  avoid progress-render overhead.
  - Result: 960 test executions, exit 0 in 10.5s.
- Full framework suite:
  `HF_DATASETS_CACHE=/tmp/ml-hw-hf-datasets .../.venv/bin/pytest -q tests`
  - Result: `397 passed, 13 skipped, 1 warning in 27.95s`.
  - The sole warning remains the pre-existing unknown `integration` mark in
    `tests/test_ettm_loader.py`.

## Review round 10 revision

### Scope and RED / GREEN evidence

- Revision parent: `788b15e64c47748caa810c08db938b202c4c6ba7`.
- The first secondary-query RED test raised the original publication
  `WorkerAbort`, then a `SystemExit` from the outcome query. The query failure
  was silently converted to `None`, so the queue item, unfinished task, and
  transaction payload were rolled back while the outcome was unknown. GREEN
  introduces an explicit `_OUTCOME_UNKNOWN` result. UNKNOWN preserves all
  queue ownership under the mutex and rethrows the original exception; engine
  recovery re-queries only after lifecycle and queue locks have been released.
- A transient absent result now identity-removes the preserved item, decrements
  exactly one unfinished task, records allocated sequences as failed, and only
  then commits rejection. A transient accepted result completes token-matching
  coordinator registration before restoring queue visibility. Deterministic
  tests cover both directions and prove the secondary exception never masks
  the original one.
- Persistent query and visibility faults initially left the engine RUNNING or
  removed its transaction, allowing a later successful shutdown. GREEN leaves
  the transaction marked `recovery_unresolved`, changes engine state to
  `FAILED`, exposes its request ID through outstanding diagnostics, and forces
  every shutdown attempt to return false while that evidence remains. Metrics
  diagnostic callbacks are best effort and cannot replace the original
  exception.
- Acquire-after-add RED injected immediately after `_held.add()` but before
  `acquire()` returned. The local `acquired` variable remained false and leaked
  one lease. GREEN queries authoritative held-token membership whenever no
  transaction owns recovery, then performs the same idempotent release used by
  transaction cleanup.
- Outstanding ABA RED installed a replacement request under the same request
  ID with a different submission token. GREEN accepts outstanding evidence
  only when its exact built-in token equals the transaction attempt token; the
  replacement remains untouched and the old transaction remains unresolved.
- Legacy unregister RED showed an unconditional token-less reservation and
  outstanding pop. GREEN requires an expected token, normalizes request ID and
  token before the coordinator condition, and compare-and-removes only matching
  reservation or outstanding membership.
- Focused verification exposed a deterministic weakref registry deadlock:
  lookup dereferenced a weakref under a non-reentrant registry lock, and GC ran
  its cleanup callback on the same thread. A RED nonblocking re-entry guard
  reproduced the lock contract. GREEN uses a reentrant registry lock while
  retaining exact weakref identity checks, eliminating callback self-deadlock.
- The approved design specification and this report were appended. No
  temporary plan file or subagent was used.

### Round 10 ownership and exception review

- Publication still follows lifecycle condition -> coordinator condition ->
  queue mutex -> sealed metrics lock. The UNKNOWN result crosses out of that
  chain as inert evidence; outcome re-query, deferred rollback, unresolved
  marking, and diagnostics run only after the publication locks are released.
- Queue rollback is permitted only after an explicit absent outcome. Accepted
  and UNKNOWN paths never remove the item or decrement task ownership.
  Registration recovery precedes accepted visibility, preventing a worker from
  completing before outstanding membership exists.
- Lease release, reservation abort, outstanding acceptance, and legacy cleanup
  all compare authoritative token membership. No local acquire flag or
  request-ID-only entry decides ownership.
- Any two unsuccessful recovery attempts leave `recovery_unresolved` state and
  force FAILED shutdown. This covers persistent query failure and secondary
  cleanup failure without swallowing or replacing the first exception.
- The requesting-review checklist was performed locally because R10 explicitly
  prohibited subagents. No critical or important issue remained before the
  final verification matrix.

### Round 10 verification

- Changed-file compile check:
  `.../.venv/bin/python -m compileall -q` over all changed async source and test
  modules.
  - Result: exit 0.
- Final focused Task 4/5 set:
  `.../.venv/bin/pytest -q framework/tests/test_async_types.py framework/tests/test_async_engine.py framework/tests/test_async_completion.py framework/tests/test_async_metrics.py`
  - Result: `175 passed in 1.41s`.
- Concurrency repetition: the 102-test engine module was collected ten times in
  one pytest process with `--keep-duplicates`; terminal output was disabled.
  - Result: 1,020 test executions, exit 0 in 10.6s.
- Full framework suite:
  `HF_DATASETS_CACHE=/tmp/ml-hw-hf-datasets .../.venv/bin/pytest -q tests`
  - Result: `405 passed, 13 skipped, 1 warning in 28.01s`.
  - The sole warning remains the pre-existing unknown `integration` mark in
    `tests/test_ettm_loader.py`.

## Review round 11 revision

### Scope and RED / GREEN evidence

- Revision parent: `23a9b36f1a6bec0854c9c479bafb7666b628de09`.
- Acquire-query RED raised after held-token insertion, then injected a
  `SystemExit` from the secondary membership query. The old public `contains()`
  path released the lease and lost all diagnostics. GREEN queries the held set
  through a protected non-dispatch internal operation. UNKNOWN retains the
  lease and a token-keyed unresolved transaction, marks the engine `FAILED`,
  exposes the request ID, forces shutdown false, and re-raises the original
  `WorkerAbort`.
- Prepared-visibility RED proved both a spurious-wakeup take and a real worker
  could consume a published item before coordinator registration committed.
  GREEN tracks identity-preserving `PREPARED` and `ACCEPTED_PREPARED` state in
  the bounded queue. `_take` waits on visible head state, and exact-token
  accepted recovery changes only that item to visible and notifies after
  coordinator commit. Explicit absent removes it; UNKNOWN retains it prepared.
- Capacity and depth tests define the split contract: all prepared and visible
  request items occupy physical `maxsize`; logical depth transitions include
  visible plus known-accepted prepared items and exclude outcome-unknown
  prepared items and stop tokens. Two accepted-prepared entries therefore
  report depths `1, 2`, remain unclaimable, and dequeue to `1, 0` only after
  visibility commits.
- Failed-sequence RED raised after transition allocation and made every sealed
  failure-evidence write raise. The old queue swallowed that fault, deleted the
  transaction evidence, and completed rejection. GREEN stages physical removal
  separately, retains the exact payload/sequence evidence until the idempotent
  sealed write succeeds, and leaves persistent failure unresolved without
  masking the original publication exception.
- Terminal ABA RED installed a committed terminal bitmap for a replacement
  token. The old non-zero-only check completed the stale transaction. GREEN
  stores the registration token alongside terminal state and requires exact
  transaction-token equality for terminal-only recovery.
- A real weakref/GC regression now verifies both exact collector-entry removal
  and the identity guard that preserves a replacement registry entry. No
  temporary plan file or subagent was used.

### Round 11 ownership and exception review

- Publication still enters the physical queue under its mutex, but accepted
  publication no longer emits consumer visibility. Queue state, unfinished-task
  ownership, transition sequence, and transaction payload are all staged before
  locks are released. Visibility is a separate retryable commit after exact
  coordinator evidence.
- A prepared head blocks later items, preserving FIFO. Close wakes consumers but
  retains unresolved prepared payloads for diagnostics; drain and stop-token
  cleanup remove only visible requests/control tokens and cannot cancel an
  ambiguous publication.
- Deferred absent cleanup may mutate physical queue/task ownership once, then
  retry sealed failed-sequence evidence idempotently. Transaction fields remain
  authoritative until that evidence commits, so rejection cannot become
  externally successful early.
- Terminal state and terminal token are extended together at reservation and
  the token is fixed at registration commit. Outstanding, reservation, and
  terminal recovery therefore all use the same exact attempt namespace.

### Round 11 verification

- Focused Task 4/5 set after implementation:
  `tests/test_async_types.py tests/test_async_engine.py tests/test_async_completion.py tests/test_async_metrics.py`
  - Result: `184 passed in 3.50s`.
- Concurrency repetition: the 108-test engine module was collected ten times in
  one process with `--keep-duplicates`.
  - Result: `1,080 passed in 31.29s`.
- Final full framework suite:
  `HF_DATASETS_CACHE=/tmp/ml-hw-hf-datasets .../.venv/bin/python -m pytest -q framework/tests`
  - Result: `414 passed, 13 skipped, 1 warning in 30.23s`.
  - The sole warning remains the pre-existing unknown `integration` mark in
    `tests/test_ettm_loader.py`.
- Final changed-file compile check and `git diff --check` both exited 0.

## Review round 12 revision

### Scope and RED / GREEN evidence

- Revision parent: `672747544ce86422d16f8f4c2794e79225ad7eb0`.
- Head-wakeup RED placed a prepared request before a visible request and parked
  a real waiter on the visibility predicate. Identity rollback removed the
  prepared head but only notified `not_full`, so the waiter timed out beside a
  runnable item. A second RED made two accepted-prepared entries visible in
  reverse order with two waiters; one waiter removed the first head while the
  other remained asleep beside the second. GREEN re-evaluates head visibility
  after absent removal and every dequeue, then notifies all waiters under the
  queue mutex. The deterministic transitions finish at depths `1, 0`.
- Registration RED parameterized before/after `BaseException` at terminal
  record allocation, token binding, outstanding insertion, and reservation
  removal. The parallel bitmap/token implementation had no independently
  retryable boundaries, and a fault before reservation pop left stale
  ownership even when outstanding evidence existed. GREEN replaces both arrays
  with one token-bound `_TerminalRecord` authority and read-only indexed views.
  Reservation and each commit stage are idempotent; accepted recovery reruns
  the sealed reconciliation from matching reservation, outstanding, or
  terminal evidence and always removes its matching reservation.
- A persistent reservation with no outstanding request initially allowed
  engine shutdown to report success. GREEN makes coordinator stop and repeated
  stop observation fail with `counter_invariant_failed` while any reservation
  remains, so engine shutdown finishes `FAILED` rather than hiding incomplete
  registration.
- Completion-membership RED submitted an old token under a replacement
  request ID. The old ID-only lookup evaluated stale payload metadata,
  terminalized the replacement, and popped its outstanding entry. GREEN
  normalizes incoming identity outside the coordinator condition and compares
  both the bound terminal-record token and outstanding request token before any
  evaluator or terminal action. The stale completion is diagnosed and ignored;
  the correct token later evaluates and terminalizes exactly once.
- Existing duplicate/unknown batch behavior, legacy token-less registration,
  terminal crash cleanup, and R9-R11 recovery paths remain covered. No
  temporary plan file or subagent was used.

### Round 12 ownership and exception review

- `_TerminalRecord` co-locates token binding and terminal state; compatibility
  views expose indexed reads but reject assignment. No internal path reads or
  mutates the compatibility views.
- Commit order is record allocation, token bind, outstanding publication, then
  reservation removal. A retry validates exact token ownership before every
  idempotent stage. A committed terminal record suppresses outstanding
  resurrection while still permitting stale-reservation removal.
- Queue wakeups occur immediately after head mutation, before transition or
  claim callbacks can fault. A safe extra `notify_all()` is allowed when an
  arbitrary identity rollback reveals an already-visible head.
- Completion membership uses normalized incoming values and only the stored
  outstanding object reaches evaluation. Stale input objects cannot supply
  labels, remove replacement ownership, or claim its terminal record.

### Round 12 verification

- Focused Task 4/5 set after implementation:
  `tests/test_async_types.py tests/test_async_engine.py tests/test_async_completion.py tests/test_async_metrics.py`
  - Final fresh result: `196 passed in 3.52s`.
- Concurrency repetition: the 119-test engine module was collected ten times in
  one process with `--keep-duplicates`.
  - Final fresh result: `1,190 passed in 31.41s`.
- Final fresh full framework suite:
  - Result: `426 passed, 13 skipped, 1 warning in 30.24s`.
  - The warning is the pre-existing unknown `integration` mark in
    `tests/test_ettm_loader.py`.
- Changed-file bytecode compilation, `git diff --check`, and the scoped
  changed-file review all exited cleanly after the final edits.

## Review round 13 revision

### Scope and RED / GREEN evidence

- Revision parent: `914a424321c14cb11044c94f73922ccbfd1d8bbe`.
- Terminal-before-visibility RED committed registration, crashed the real
  completion thread, waited deterministically for its exact-token terminal
  record to reach committed state, and only then allowed submission recovery.
  The old path attempted reconciliation against a failed coordinator and left
  the accepted-prepared item unresolved. GREEN reads exact token plus state
  under the coordinator condition. A claimed/committed record removes the
  exact prepared identity, balances unfinished ownership, emits the accepted
  depth-zero transition, releases the attempt slot, and finalizes/removes the
  accepted transaction without runtime or evaluator work.
- RuntimeError RED injected the same exception object immediately before and
  after outstanding publication. Both old paths tried rejected accounting
  against a sealed accepted outcome and replaced the primary with `request
  already has accepted accounting`. GREEN queries the authoritative attempt
  outcome first. Accepted/UNKNOWN faults enter the common recovery and bare
  re-raise path; rejected/absent faults receive only matching cleanup. The
  existing absent coordinator-unavailable path retains its `False` contract.
- FAILED-stop RED left a token-bound registration reservation while crashing
  the coordinator. The old early return reported only completion-thread
  failure. GREEN snapshots reservations in every failed-stop exit and records
  the counter invariant after releasing a deliberately non-reentrant
  coordinator condition.
- The initial RED selection produced four expected failures. The same
  selection passed after implementation, and the prior crash-before-acceptance
  regression was also rerun after narrowing the RuntimeError branch. No sleep,
  temporary plan file, or subagent was used.

### Round 13 ownership and exception review

- Terminal state inspection and queue mutation share the existing
  coordinator-condition to queue-mutex order, so a terminal claim cannot land
  between the state decision and visibility. A pending record becomes visible;
  any non-pending exact record is physically removed instead.
- Terminal removal uses the normal accepted dequeue sequence path for depth
  metrics and an idempotent attempt-token lease release. Transaction fields
  retain the removed item and transition long enough for cleanup retry, then
  exact registry identity removal completes accepted ownership.
- Publication RuntimeError handling no longer infers accounting from exception
  class. Accepted cleanup cannot call rejected accounting, and cleanup faults
  remain secondary to the original stage exception in the outer recovery
  loop. Expected outcome-absent coordinator failure remains a normal rejected
  submit rather than changing the established public API.
- FAILED stop reads residual reservation state only under the coordinator
  condition and invokes metrics callbacks only after releasing it. The scoped
  requesting-review checklist was performed locally because R13 prohibited
  subagents; no critical or important issue remained.

### Round 13 verification

- Final focused Task 4/5 set:
  `tests/test_async_types.py tests/test_async_engine.py tests/test_async_completion.py tests/test_async_metrics.py`
  - Result: `200 passed in 3.49s`.
- Concurrency repetition: the 122-test engine module was collected ten times in
  one process with `--keep-duplicates`.
  - Result: `1,220 passed in 31.49s`.
- Final full framework suite with the existing Hugging Face model cache and a
  temporary datasets cache:
  - Result: `430 passed, 13 skipped, 1 warning in 30.48s`.
  - The warning remains the pre-existing unknown `integration` mark in
    `tests/test_ettm_loader.py`.
  - A preliminary invocation incorrectly redirected `HF_HOME` to an empty
    directory and failed only the unrelated cached-tokenizer test; restoring
    the established cache produced the clean final result above.
- Final changed-file bytecode compilation and `git diff --check` exited 0 after
  the report/spec append. The scoped diff contains only the two async source
  files, their two test modules, this report, and the existing design spec.

## Review round 14 revision

### Scope and RED / GREEN evidence

- Revision parent: `91b3f365c159d9fba0342ee70fbf394293eedf77`.
- The FAILED/PENDING RED commits an exact-token accepted registration, crashes
  the real completion thread, gates `_fail_outstanding()`, and gates recovery
  until the coordinator has published `FAILED`. While the terminal record is
  still `PENDING`, the submitter remains blocked, the accepted item remains
  prepared, and runtime/evaluator calls remain zero. Releasing finalization
  changes the exact terminal record and wakes recovery; recovery removes the
  prepared identity and completes terminal cleanup without making it visible.
- A bounded-wait companion keeps `_fail_outstanding()` gated beyond the engine
  flush deadline. Recovery preserves the prepared item, exact attempt lease,
  and transaction in the unresolved registry; the engine remains `FAILED` and
  shutdown returns `False`. Releasing the gate later cannot make the item
  worker-visible.
- Fourteen parameterized RED cases inject `BaseException` immediately before
  and after each terminal cleanup stage: physical identity removal,
  unfinished-task balance, transition capture, depth evidence, lease release,
  transaction terminal mark, and registry removal. Two additional cases fault
  `_capture_transition()` before or after allocation. The old cleanup either
  lacked those stages or lost queue ownership after a partial mutation. GREEN
  retains per-stage evidence and completes every retry with one logical
  transition, no slot leak, no unfinished task, and no registry residue.
- The initial 18-case RED selection failed all 18 cases. After implementation
  and correction of the lifecycle test gate so it deterministically reached
  `FAILED/PENDING`, the same selection passed all 18. No sleep, temporary plan
  file, or subagent was used.

### Round 14 ownership and exception review

- Visibility remains a coordinator-condition decision: only an exact-token
  `PENDING` record while lifecycle is `RUNNING` can clear the prepared marker.
  A non-running coordinator with an exact pending record waits on the same
  condition using one transaction-owned deadline. Terminal-state mutation in
  `_fail_outstanding()` now uses the notifying state helper, so waiters do not
  poll and do not observe a prepared item as runnable.
- Terminal prepared cleanup is ordered as identity removal, unfinished-task
  balance, transition allocation/evidence, depth delivery, authoritative slot
  release, accepted transaction mark, and exact registry pop. Queue removal and
  the transition object are saved on the transaction before a fallible return.
  Every later stage has an explicit committed flag or an authoritative state
  query, making both before- and after-mutation retries idempotent.
- Depth evidence precedes slot release as required. Repeated transition
  allocation reuses the saved sequence and timestamp; repeated metric
  allocation is a max operation, and repeated depth delivery is sequence-
  idempotent. Slot cleanup verifies held-token membership after release rather
  than trusting a potentially interrupted return value.
- The requesting-review checklist was performed locally because R14 prohibited
  subagents. The scoped lock-order, exact-token ownership, primary-exception,
  retry-deadline, queue-depth, task-balance, and lease-release review found no
  critical or important issue.

### Round 14 verification

- New lifecycle and cleanup fault selection:
  - Result: `18 passed, 122 deselected in 0.50s`.
- Complete engine module:
  - Result: `140 passed in 3.41s`.
- Final focused Task 4/5 set:
  `tests/test_async_types.py tests/test_async_engine.py tests/test_async_completion.py tests/test_async_metrics.py`
  - Result: `218 passed in 3.76s`.
- Concurrency repetition: the 140-test engine module was collected ten times in
  one process with `--keep-duplicates`.
  - Result: `1,400 passed in 33.92s`.
- Final fresh full framework suite with the existing Hugging Face model cache
  and a temporary datasets cache:
  - Result: `448 passed, 13 skipped, 1 warning in 30.64s`.
  - The warning remains the pre-existing unknown `integration` mark in
    `tests/test_ettm_loader.py`.

## Review round 15 revision

### Scope and RED / GREEN evidence

- Revision parent: `048a7d916cb0048ae4f7a3b533aa460ed9ae544b`.
- The terminal inner-mutation RED faults `_clear_entry_state()` after its map
  mutation. The previous cleanup had already deleted the deque identity but
  had not committed `terminal_queue_removed`, so both retries reported missing
  ownership. GREEN first commits an exact request/token tombstone in the state
  map, then treats physical deletion, state cleanup, and task balance as
  separate stages. A missing deque identity under the exact tombstone proves
  physical removal; absent state under that same staged ownership proves state
  cleanup. The retry completes one depth transition, one slot release, and one
  registry removal while preserving the primary exception.
- A gated terminal-removal RED leaves the tombstoned identity physically at
  the queue head. The exact tombstone remains non-visible, so the worker cannot
  claim it and runtime/evaluator calls stay zero until removal is released.
- Clock and transition-construction RED cases fault before allocation
  membership. The old eager counter consumed sequence 1 and the retry returned
  sequence 2. GREEN normalizes depth/time and constructs the transition before
  committing one operation-key mapping; the sequence is derived from mapping
  membership, so an absent pre-commit fault retries sequence 1. A companion
  after-membership case proves that retrying the same operation key returns the
  original depth, timestamp, object record, and sequence before the following
  operation receives sequence 2.
- Concurrent FAILED/PENDING shutdown RED holds accepted submission recovery
  and coordinator finalization past the shutdown deadline. The old shutdown
  cancellation attempted rejected accounting against the accepted token and
  raised `request already has accepted accounting`. GREEN classifies each
  active transaction through sealed outcome state; accepted and UNKNOWN or
  unresolved transactions remain owned and force bounded shutdown `False`.
  Runtime remains unentered and accepted/rejected accounting remains 1/0.
- A final classifier RED gives an explicit-absent transaction publication
  recovery evidence. It proved that `None` outcome alone is insufficient:
  shutdown may reject only an absent transaction that is still structurally in
  preflight. Publication recovery, accepted, UNKNOWN, and unresolved ownership
  are retained.
- The first R15 selection produced six expected failures. The added
  absent-nonpreflight case also failed before its guard was implemented. All
  seven executions pass after GREEN. No sleep, temporary plan file, or
  subagent was used.

### Round 15 ownership and exception review

- `_TerminalQueueTombstone` co-locates exact request ID and attempt token in the
  existing prepared-state authority before deque mutation. `take()` already
  requires absent entry state for visibility, so tombstones cannot execute.
  State cleanup happens only after physical removal, and unfinished-task
  balance happens only after state cleanup. Each retry validates the preceding
  authoritative stage rather than inferring ownership from request ID alone.
- `_transition_allocations` is the sole sequence authority. All production
  publish, dequeue, drain, and terminal paths supply an opaque operation key.
  Fallible clock reads, integer normalization, and transition construction
  happen before the mapping assignment. Once membership commits, retries query
  the same key and reuse the immutable transition. Publication rollback derives
  failed-sequence evidence from that record rather than a counter range.
- Shutdown snapshots transaction identities, queries the sealed attempt
  outcome, and rejects only explicit-absent transactions whose fields still
  prove preflight. It performs no rejected accounting for accepted, UNKNOWN,
  unresolved, or publication-recovery transactions. Existing permanently
  blocked preflight cases still commit one rejection and release their lease.
- The requesting-review checklist was performed locally because R15 prohibited
  subagents. The scoped tombstone identity, queue lock order, transition
  membership, failed-sequence evidence, outcome race, deadline, primary-
  exception, and lease/task/registry review found no critical or important
  issue.

### Round 15 verification

- Initial R15 lifecycle/allocation selection:
  - Result after GREEN: `6 passed, 140 deselected in 2.24s`.
- Shutdown classifier selection, including the three prior preflight cases:
  - Result: `5 passed, 142 deselected in 0.35s`.
- Complete engine module:
  - Result: `147 passed in 3.60s`.
- Final focused Task 4/5 set:
  `tests/test_async_types.py tests/test_async_engine.py tests/test_async_completion.py tests/test_async_metrics.py`
  - Result: `225 passed in 3.89s`.
- Concurrency repetition: the 147-test engine module was collected ten times in
  one process with `--keep-duplicates`.
  - Result: `1,470 passed in 35.45s`.
- Final fresh full framework suite with the existing Hugging Face model cache
  and a temporary datasets cache:
  - Result: `455 passed, 13 skipped, 1 warning in 30.85s`.
  - The warning remains the pre-existing unknown `integration` mark in
    `tests/test_ettm_loader.py`.

## Review round 16 revision

### Scope and RED / GREEN evidence

- Revision parent: `bcd5d2113361420e4eed6f296017ea002212ca8b`.
- Six deterministic RED cases gate a real worker before its first dequeue and
  inject `BaseException` at the transition clock, transition constructor, or
  immediately after the physical deque removal. The same three faults are
  injected through public `cancel_queued()` while its accepted request remains
  visible. Before GREEN, all worker cases left the exact request outstanding
  and all cancel cases propagated the interruption after losing queue
  ownership. The initial selection therefore failed all six cases.
- `_DequeueOperation` now records the opaque operation key, exact request
  object, normalized request ID, exact attempt token, worker owner, and the
  immutable post-dequeue transition before `_get()` can mutate the deque. The
  physical removal, prepared-state cleanup, pending/owned handoff, slot
  release, depth delivery, and unfinished-task balance are independent commit
  stages. Worker exception cleanup queries its persistent records, resumes any
  interrupted removal, deduplicates already-published pending/owned payloads,
  and terminalizes the exact attempts with the original exception type.
- `_DrainOperation` similarly snapshots every visible request object, request
  ID, attempt-token lease, and the post-drain transition before the first
  physical removal. Retry uses the same caller operation key and exact object
  tuple. Removal, prepared-state cleanup, aggregate task balance, per-request
  lease release, depth evidence, failure submission, and cancellation
  completion each have idempotent evidence. A second failure marks the engine
  `FAILED` while retaining the operation record; a one-shot clock,
  constructor, or post-remove failure resumes within the public call.
- An event-gated concurrency case pauses cancel after physical removal, starts
  shutdown while the request is still outstanding, and then releases recovery.
  Shutdown observes the exact cancellation completion, the worker consumes
  only its stop token, and both calls finish without stealing or duplicating
  the removed payload.
- A follow-up handoff RED faults the candidate callback after it has published
  `_pending_by_worker` but before `_take()` can commit its return. It initially
  left that exact payload retained in the pending map even though the dequeue
  record had terminalized it. GREEN always claims the authoritative worker
  pending entry during exception cleanup, independently of the caller's local
  `has_pending` flag, then deduplicates it against persistent dequeue records.
- A follow-up drain RED faults slot release after the exact lease mutation.
  GREEN re-enters the engine-side drain stages with the same operation record,
  observes absent token membership, and continues through depth and
  cancellation completion. All nine new cases pass after GREEN. No sleep,
  temporary plan file, or subagent was used.

### Round 16 ownership and exception review

- Queue mutation is ordered after request/worker/token normalization,
  monotonic clock acquisition, immutable transition construction, operation
  record construction, and operation-map publication. Constructor and clock
  failures therefore leave the request visible; an interruption after
  `_get()` leaves an exact worker-owned record whose absent deque identity is
  authoritative removal evidence.
- Normal completion balances each dequeue operation once and retires its
  payload record only after coordinator submission and task balance. Worker
  cleanup retains the operation through slot/depth recovery and task balance,
  so a retry cannot double-release a lease or decrement unfinished tasks
  twice. Exact object identity deduplication prevents a candidate present in
  both the pending map and its dequeue record from being terminalized twice.
- Drain retry never cycles retained prepared entries or stop tokens through
  `_get()`. It removes only the snapshotted visible identities, so a fault
  cannot drop a sentinel or non-visible accepted-prepared payload. Slot and
  depth callbacks execute outside the queue mutex, preserving existing
  reentrancy and shutdown-deadline contracts.
- The requesting-review checklist was performed locally because R16
  prohibited subagents. The scoped lock-order, exact-token/object ownership,
  operation retirement, task/lease idempotence, transition ordering,
  cancellation/shutdown race, payload lifetime, and primary-exception review
  found no critical or important issue.

### Round 16 verification

- Initial clock/constructor/post-remove RED selection:
  - Result before GREEN: `6 failed, 147 deselected in 6.60s`.
  - Result after GREEN: `6 passed, 147 deselected in 0.08s`.
- Follow-up pending-handoff and post-slot-mutation RED cases each failed once
  before their remaining stage recovery was added and passed in isolation
  after GREEN.
- Complete engine module:
  - Result: `156 passed in 3.64s`.
- Final focused Task 4/5 set:
  `tests/test_async_types.py tests/test_async_engine.py tests/test_async_completion.py tests/test_async_metrics.py`
  - Result: `234 passed in 3.97s`.
- Concurrency repetition: the 156-test engine module was collected ten times in
  one process with `--keep-duplicates`.
  - Result: `1,560 passed in 35.80s`.
- Final fresh full framework suite with the existing Hugging Face model cache
  and a temporary datasets cache:
  - Result: `464 passed, 13 skipped, 1 warning in 30.78s`.
  - The warning remains the pre-existing unknown `integration` mark in
    `tests/test_ettm_loader.py`.
