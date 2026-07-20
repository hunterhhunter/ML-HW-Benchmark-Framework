# Unified Task 6 implementation report

## Scope and base

- Worktree: `/tmp/ml-hw-benchmark-async-worktree`
- Base HEAD: `0fa491c`
- Implemented production scope: `framework/src/core/runtime_executor.py`
- Test scope: `framework/tests/test_native_async_runtime_executor.py`
- MLPerf LoadGen was not imported, wrapped, exposed, or claimed compatible.
- `InferenceEngine`, `_AsyncRunController`, and compatibility facade ownership
  files were not modified.

Commit `fe600c9` had imported a native-executor-like implementation before a
documented TDD cycle. I inspected its isolated native delta, added the missing
tests first, removed only its native production delta with `apply_patch`, and
then implemented the contract again from the approved Task 6 brief.

## RED evidence

Before removing production, the newly added tests exposed real defects in the
imported implementation:

```text
8 failed, 10 passed, 7 deselected in 0.18s
```

The failures were:

- seven invalid setting cases that were accepted or raised incidental
  `TypeError` instead of deliberate `ValueError` (bool, fractional/string
  inflight values, bool/string/non-finite timeout values);
- callback success followed by `submit_async()` raising was overwritten by a
  submit failure, violating first-terminal-wins.

After removing `NativeAsyncOutcome`, `NativeAsyncExecutorSnapshot`,
`_NativeDispatch`, and `NativeAsyncRuntimeExecutor` while retaining the tests,
the required clean RED was:

```text
ImportError: cannot import name 'NativeAsyncOutcome' from
'core.runtime_executor'
1 error in 0.10s
exit code 2
```

No production change was made after the removal until this RED was captured.

## Implementation

The fresh `NativeAsyncRuntimeExecutor` is a callback-to-blocking bridge: each
framework worker submits one SDK job and waits on that dispatch's event. It
does not create a framework dispatcher thread.

Implemented guarantees:

- constructor settings reject bool, malformed, non-finite, and non-positive
  values; per-call and shutdown timeouts deliberately allow zero but reject
  bool, malformed, non-finite, and negative values;
- a bounded permit is acquired before a monotonic framework dispatch token is
  published;
- the dispatch record is in the registry before SDK submission, so inline
  callbacks are safe;
- callback, invalid payload, timeout, and submit exception all commit through
  one lock-protected first-terminal-wins state;
- callback-then-submit-raise preserves the callback result and does not count a
  false submit-failure terminal;
- framework dispatch tokens own lifecycle truth; vendor IDs are diagnostic and
  sanitized to primitive bounded values rather than used as registry keys;
- submit exceptions are reduced to type name plus a fixed message, retaining
  no exception, traceback, or tensor in diagnostics;
- duplicate and post-timeout callbacks cannot overwrite the first outcome and
  increment separate counters;
- inputs, outputs, the dispatch record, and its bounded permit remain owned
  until `acknowledge()`;
- exact-token duplicate ACK is idempotent, unknown tokens fail without
  releasing a permit, and tokenless normalized failures ACK as no-ops;
- shutdown closes submission, waits on registry emptiness with a deadline, and
  wakes on ACK;
- snapshots copy primitive counters while holding the registry condition lock.

## Test coverage

`framework/tests/test_native_async_runtime_executor.py` uses condition/event
synchronization and actual NumPy arrays. It contains:

- inline callback before vendor-ID return;
- two concurrent submissions completed in reverse order;
- duplicate callback first-result preservation;
- timeout followed by late callback;
- submit failure, ACK, and subsequent success;
- callback-then-submit-raise first-terminal-wins;
- invalid callback payload followed by a valid duplicate;
- weak-reference proof that input and output buffers plus the permit remain
  live before ACK and are collectable after ACK and fake-SDK ownership release;
- idempotent ACK, unknown-token rejection without permit release;
- bounded-slot timeout and shutdown-before-submit normalized failures;
- sanitized submit diagnostics;
- real `AsyncInferenceEngine` reverse completion, duplicate, timeout/late, and
  submit-failure variants;
- exact output identity, one trace per request, both counter equations, zero
  outstanding, and zero native inflight;
- an autouse assertion that every test leaves no new `async-*` thread alive.

## GREEN and regression evidence

Native file:

```text
25 passed in 0.15s
```

Five independent native pytest processes:

```text
run 1: 25 passed in 0.13s
run 2: 25 passed in 0.13s
run 3: 25 passed in 0.13s
run 4: 25 passed in 0.12s
run 5: 25 passed in 0.13s
```

Focused native/runtime/async engine regression:

```text
225 passed in 4.03s
```

Framework full suite:

```text
1039 passed, 13 skipped, 1 warning in 52.40s
```

The warning is the pre-existing unknown `pytest.mark.integration` warning in
`framework/tests/test_ettm_loader.py`.

## Concurrency diagnostic observed during verification

The first focused run had one failure in the pre-existing blocking-executor
test `test_full_completion_queue_cannot_make_shutdown_infinite`; a direct
rerun failed once, then five current-tree repetitions produced four passes and
one failure. An exported clean `0fa491c` base produced six consecutive passes.
The current focused suite subsequently passed all 225 tests. No Task 6 code is
used by that test, and the Task 6 production diff leaves
`BlockingRuntimeExecutor` unchanged, so I did not make an out-of-scope queue
engine change. This remains a scheduling-sensitive existing shutdown-test
concern worth tracking separately.

Running pytest from repository root also collects backend tests outside the
approved framework scope and hung at the first test,
`backend/tests/test_results_api.py::TestGetResults::test_empty_results`, inside
the FastAPI TestClient request before any native test ran. That process was
interrupted with exit 130. The scoped `framework/tests` full suite above is the
project baseline-compatible verification command and passed.

## Imported generation timing scope audit

`fe600c9` also changed:

- `framework/src/core/generation_result.py`;
- `framework/src/evaluators/llama_evaluator.py`;
- their timing tests and the blocking executor's optional timing mapping.

Those changes make unavailable TTFT/TPOT values optional and exclude missing
or non-finite timing samples. They are coherent and tested, but they are not
required by `NativeAsyncRuntimeExecutor`, whose outcome contract already
allows independent timing payloads. Per the Task 6 brief, they were neither
removed nor expanded.

## Remaining limits

- CI evidence covers fake callback SDKs and the real framework async queue, not
  a physical NPU vendor SDK.
- Vendor-specific callback threading, cancellation, and device teardown must be
  validated when each adapter is introduced.
- The unrelated blocking shutdown stress-test flake and root-level backend
  TestClient hang should be handled as separate diagnostics rather than folded
  into the native executor change.
