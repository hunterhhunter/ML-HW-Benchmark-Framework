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

## Independent review remediation

An independent review of commit `295dd70` identified four important findings
and one minor finding. Each was handled as a separate test-first cycle before
the next production change.

### 1. Platform timeout ceiling

Tests were added for `threading.TIMEOUT_MAX + 1` at constructor, execute, and
shutdown boundaries. The execute case also proves rejection happens before SDK
publication and leaves zero registry entries, then runs a valid dispatch to
prove the permit is usable.

RED:

```text
3 failed, 25 deselected in 1.10s
```

All three boundaries accepted the oversized timeout. `_finite_timeout()` now
rejects values above `threading.TIMEOUT_MAX` with deliberate `ValueError`.

GREEN:

```text
3 passed, 25 deselected in 0.06s
```

### 2. One logical deadline across permit, submit, and callback

A gated backend holds `submit_async()` beyond a 10 ms deadline. When its
callback arrived after 30 ms, the imported behavior incorrectly returned
success. A paired boundary test calls back before the deadline but delays only
the submit return; that already-terminal callback must remain successful.

RED:

```text
1 failed, 1 passed, 28 deselected in 0.16s
```

The callback now captures its observation time. Permit acquisition, callback,
submit return, submit exception, and final event wait compare against the same
deadline. If no terminal exists after the deadline, `NativeAsyncTimeout` wins;
the callback that arrived late increments `late_callbacks`. A callback observed
before the deadline remains first-terminal even if submit returns later.

Self-review then added a deterministic fake-clock/fake-permit assertion that
the semaphore receives the deadline remainder rather than the original
timeout.

RED:

```text
assert [0.01] == [0.006 +/- 6.0e-09]
1 failed, 50 deselected in 0.12s
```

GREEN:

```text
3 passed, 27 deselected in 0.12s
1 passed, 50 deselected in 0.06s
```

The class docstring now states the backend protocol explicitly:
`submit_async()` must publish work and return promptly. Framework timeout is a
logical terminal decision; physical vendor-job cancellation is adapter follow-up
scope. No submit thread was added.

### 3. Bounded ACK state

A 10,000-dispatch test measured all list/set/dict entries directly retained by
the executor without depending on a private tombstone name.

RED:

```text
assert 10000 == 0
1 failed, 30 deselected in 0.18s
```

The unbounded acknowledged-token set was removed. Because tokens are monotonic
and a dispatch leaves the live registry only through ACK, registry absence plus
`1 <= token < next_dispatch_token` proves an issued token is already ACKed.
Tokens outside that range remain unknown errors.

GREEN:

```text
3 passed, 28 deselected in 0.14s
```

### 4. Native timing trust boundary

The test matrix covers bool, negative, non-finite, ndarray, traceback-bearing
exception, nested value, mapping ndarray/exception, oversized mapping, long
key, and long text. It also checks a flat LLM timing dictionary, NumPy scalar
copy, and weak-reference collection of the original outcome and timing map.

RED:

```text
16 failed, 31 deselected in 0.20s
```

The accepted timing contract is now:

- `None` or a finite, nonnegative real scalar copied to built-in `float`;
- a flat mapping of at most 32 items;
- nonempty exact-string keys up to 128 characters;
- values limited to `None`, exact bool, strings up to 512 characters, or
  finite nonnegative real numbers;
- no ndarray, nested mapping, exception, traceback-bearing object, negative,
  or non-finite value.

Accepted mappings, diagnostic strings, output mappings, token counts, and the
outcome container are copied into primitive/new containers. Invalid payloads
become `NativeAsyncProtocolError` without retaining the original timing or
outcome object. The existing flat LLM timing keys remain compatible.

GREEN:

```text
16 passed, 31 deselected in 0.07s
```

### 5. Bounded vendor diagnostics

Tests use a 100,001-digit-equivalent integer and both plain and adversarial
string subclasses.

RED:

```text
2 failed, 1 passed, 47 deselected in 0.13s
```

Integers up to 128 bits remain exact. Larger integers become a bounded sign and
bit-count summary without decimal conversion. Every vendor string, including
subclasses overriding slicing, becomes an exact built-in string capped at 512
characters. Type summaries are capped to the same length.

GREEN:

```text
3 passed, 47 deselected in 0.06s
```

### Review-fix verification

Native file:

```text
51 passed in 0.35s
```

Five independent native processes after the final production change:

```text
run 1: 51 passed in 0.31s
run 2: 51 passed in 0.31s
run 3: 51 passed in 0.31s
run 4: 51 passed in 0.32s
run 5: 51 passed in 0.32s
```

Focused native/runtime/async-engine regression:

```text
251 passed in 4.23s
```

The first full run reproduced only the already documented, out-of-scope
blocking shutdown stress-test flake:

```text
1 failed, 1064 passed, 13 skipped, 1 warning in 53.12s
```

No queue-engine change was made. A fresh full framework process completed:

```text
1065 passed, 13 skipped, 1 warning in 53.14s
```

## Re-review trust-boundary remediation

This round supersedes the earlier callback wording that said the callback
captured its observation time before payload handling. A callback is now
observed only when its fully validated, detached payload reaches the
lock-protected terminal commit point. The callback may make a quick terminal
check before validation, but it normalizes outside the condition lock and then
rechecks terminal state and the shared deadline atomically under that lock.

### 1. Untrusted protocol and diagnostic values

Tests were written first for a mapping whose `len()` lies while `items()`
contains 33 entries, 10,000-character string subclasses in error/timing
payloads, primitive-looking vendor-ID subclasses, a vendor integer subclass
whose conversion raises, and a 10,000-character submit exception class name.

RED:

```text
7 failed, 50 deselected in 0.17s
```

The implementation no longer trusts mapping length and rejects the enumerated
33rd item. Protocol text accepts only exact built-in strings. Vendor IDs retain
only exact, bounded built-in primitive values; subclasses become bounded type
markers through an exception-safe sanitizer. Vendor-ID and submit-exception
sanitization occurs outside the condition lock, and submit exception type names
are capped at 256 characters. All error paths remain ACK-able and release the
inflight permit exactly once.

GREEN:

```text
7 passed, 50 deselected in 0.06s
```

### 2. Validated callback commit deadline

A gated timing mapping deliberately held payload validation past a 10 ms
deadline. The pre-fix implementation returned success because it timestamped
the callback before validation.

RED:

```text
1 failed, 57 deselected in 0.15s
```

The deadline gate now occurs after normalization, while holding the condition
lock at the terminal commit point. Validation that crosses the deadline commits
`NativeAsyncTimeout` and counts the callback as late. The paired existing case
still proves that a callback committed before the deadline remains successful
when only `submit_async()` return is delayed.

GREEN:

```text
3 passed, 55 deselected in 0.16s
```

### 3. Fixed unknown-ACK failure

Unknown tokens containing a 100,000-digit integer or an adversarial long string
previously triggered unbounded conversion or leaked attacker-controlled text.

RED:

```text
2 failed, 58 deselected in 0.13s
```

Every unknown token now raises the same bounded
`RuntimeError("unknown native async dispatch token")`; no token value is
interpolated. Existing duplicate-ACK behavior remains idempotent.

GREEN:

```text
3 passed, 57 deselected in 0.10s
```

### 4. Exception-safe, single-conversion timeouts

Constructor, execute, and shutdown tests use float and registered `Real`
objects whose `__float__()` raises. A one-shot float additionally fails if it
is converted twice.

RED:

```text
7 failed, 60 deselected in 0.24s
```

`_finite_timeout()` now performs exactly one conversion inside an exception
boundary and converts every conversion failure into the documented
`ValueError`. Follow-up coverage also includes a huge integer whose conversion
overflows; it required no further production change.

GREEN:

```text
10 passed, 60 deselected in 0.11s
```

### Re-review verification

Native file:

```text
70 passed in 0.39s
```

Five independent native processes after the final production change:

```text
run 1: 70 passed in 0.35s
run 2: 70 passed in 0.34s
run 3: 70 passed in 0.36s
run 4: 70 passed in 0.35s
run 5: 70 passed in 0.35s
```

Focused native/runtime/async-engine regression:

```text
270 passed in 4.23s
```

Fresh full framework regression with all Hugging Face integrations forced
offline:

```text
1084 passed, 13 skipped, 1 warning in 30.24s
```

The single warning is the pre-existing unregistered `integration` marker.
