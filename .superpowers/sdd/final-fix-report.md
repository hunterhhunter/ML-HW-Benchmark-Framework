# Async Final Review Fix Report

Date: 2026-07-15 (Asia/Seoul)

Worktree: `/tmp/ml-hw-benchmark-async-worktree`

Starting HEAD: `638a153bcc2e4d69e3358526930a2c23f04c862a`

Implementation commit: `01a8c46 fix(framework): preserve async failure truth`

## Outcome

All requested final-review findings were implemented and verified. Normal async
JSON/CSV artifacts are never rewritten to tell later failure truth. Fatal
post-commit or persistence-stage failures instead use the deterministic,
immutable `details/{run_id}.failure.json` recovery record, with a CSV
`failure_details_path` link whenever the reservation is still writable and a
deterministic stderr run-id/path link when the CSV is already consumed.

There are no known blocking failures. The final committed tree passes the full
framework test suite.

## Finding-to-fix matrix

| Finding | Implemented contract | Primary regression evidence |
|---|---|---|
| A later fatal error could not be represented without colliding with committed normal details | Added strict `save_async_failure_details()` and `RunArtifactReservation.failure_details_path`; normal bytes are preserved and recovery uses `{run_id}.failure.json` | `test_committed_normal_sidecar_baseexception_gets_recovery_record`, `test_committed_normal_sidecar_and_csv_baseexception_gets_recovery_record` |
| Runtime unload happened after normal terminal publication | When `async_outstanding_requests == 0`, unload now runs before normal sidecar and CSV publication; unload is not retried after an attempted fatal unload | `test_async_success_unloads_before_terminal_artifacts`, unload-failure actual-artifact regressions |
| CSV append errors had ambiguous active/pending/consumed state | Added strict `get_reserved_result_state()`; only an exact pending normal row is retried, consumed rows are accepted as committed, and active reservations remain available for an invalid failure row | actual-commit CSV, writable CSV, and consumed-artifact regressions |
| Failure recovery could overwrite or duplicate normal artifacts | Both normal and failure sidecars use trusted-directory, strict normalization, hard-link no-overwrite publication; CSV retains one reserved run-id transaction | strict artifact tests and actual artifact matrices below |
| Cleanup/persistence diagnostics could carry unsafe exception content | Recovery snapshots only exact built-in containers, allowlisted phases/types, and generic messages; hostile/raw messages are excluded from artifacts | `test_failure_recovery_records_persistence_error_and_csv_link`, cleanup-warning regressions |
| Async `--debug` enabled evaluator/decoder sample output | Async debug now wires coarse lifecycle output only; e2e keeps its existing evaluator/decoder debug behavior | real subprocess `test_async_debug_emits_lifecycle_without_sample_debug_output` |
| Empty runtime diagnostics and empty counts were ambiguous | Empty device snapshots emit `runtime_device_spec_unavailable`; `details.counts.outstanding` is always present, including zero | CLI warning regressions and `test_details_counts_include_terminal_outstanding_snapshot` |
| Loaded ONNX provider evidence could differ from configured providers | Loaded sessions continue to report `session.get_providers()` as the active provider source | `test_loaded_onnx_device_spec_prefers_active_session_providers` |
| A lifecycle observer failure could abort the benchmark | Lifecycle callback failures remain diagnostic-only and do not change run validity | `test_lifecycle_callback_failure_in_every_phase_is_non_fatal` |

## Root causes

1. `{run_id}.json` and the reserved CSV row were correctly no-overwrite, but
   fatal recovery had no separate immutable namespace after either artifact had
   committed.
2. Normal terminal ordering published details/CSV before `runtime.unload()`, so
   an unload fatal could occur after a valid result was visible.
3. The CLI did not query the reservation transaction's durable
   active/pending/consumed evidence before deciding whether a normal CSV retry
   was safe.
4. Cleanup diagnostics attached to the primary exception were safe enough for
   stderr but lacked a second strict snapshot boundary before artifact storage.
5. The common debug flag was passed into evaluator and decoder construction in
   both inference modes, enabling sample-level output for async runs.

## Artifact outcome matrix

| Durable state at fatal boundary | Normal details | CSV | Failure recovery | Completion evidence |
|---|---|---|---|---|
| Reservation active; no normal artifact | Failure details may use the normal `{run_id}.json` slot | One invalid row consumes the reservation | Added only if persistence itself needs separate recovery | `RUN_ID` only after CSV plus failure truth commit |
| Normal details committed; CSV active | Original normal JSON bytes unchanged | One invalid row with `failure_details_path` | Immutable `{run_id}.failure.json` | Reserved and final run IDs each emitted once |
| Normal details committed; CSV pending | Original normal JSON bytes unchanged | Exact original pending row fingerprint is retried; no different row may replace it | Immutable recovery record if the fatal remains | No duplicate run-id row |
| Normal details and CSV consumed | Both original byte sequences unchanged | Existing row unchanged | Immutable `{run_id}.failure.json`; stderr records run ID and path because CSV cannot be amended | Final run ID may be emitted once after recovery truth commits |
| Normal sidecar commit followed by `BaseException` | Original normal JSON bytes unchanged | One invalid row; `details_path` is not falsely asserted and `failure_details_path` links recovery | Immutable recovery record with exact `sidecar_save` phase | Original fatal re-raised |
| Normal CSV commit followed by `BaseException` | Original normal JSON bytes unchanged | Existing valid row unchanged; one row only | Immutable recovery record with exact `csv_save` phase | Original fatal re-raised |
| Safe unload fails before normal publication | No valid normal artifact is newly published | Invalid failure row when writable | Failure truth records `runtime_unload` | Original fatal re-raised; unload called once |

## Strict recovery API matrix

`save_async_failure_details()` was exercised with:

- active, pending-compatible, and consumed reservation authority;
- deterministic `{run_id}.failure.json` naming;
- the same strict NumPy/Path/enum/container normalizer as normal details;
- hostile-object rejection before final publication;
- exact results-root binding;
- hard-link no-overwrite behavior;
- repeated publication preserving the original bytes and raising
  `FileExistsError`;
- simulated `EXDEV` capability failure with no final or temporary artifact
  left behind.

The API intentionally permits a verified pending reservation as well as active
and consumed states. Pending is required for the durable window where a normal
CSV transaction has published its pending fingerprint but commit recovery has
not yet cleared it.

## TDD evidence

Baseline before changes:

```text
932 passed, 13 skipped, 1 warning in 48.51s
```

Selected RED observations before implementation:

- strict recovery API import failed because `save_async_failure_details` did
  not exist;
- the empty-count regression expected `{"outstanding": 0}` and received `{}`;
- unload-order and missing-runtime-warning regressions failed;
- cleanup secondary warnings were absent from recovery details;
- actual post-commit regressions had no immutable recovery file and no
  committed-state arguments;
- the real async debug subprocess printed evaluator prediction/label/score
  samples;
- an active CSV fatal produced a normal valid row instead of a linked invalid
  recovery row;
- final self-review RED reproduced a committed normal sidecar followed by a
  custom `BaseException`: `DID NOT RAISE`, proving the fatal was swallowed.

Selected GREEN results:

```text
strict failure-detail API tests:       5 passed
real async debug + e2e results tests:  2 passed
test_async_cli.py:                    51 passed
focused aggregate:                  332 passed in 8.26s
```

The final self-review regression became GREEN after narrowing the normal
sidecar persistence catch boundary so custom `BaseException` reaches the outer
recovery path and is re-raised after immutable failure persistence.

## Files changed

- `framework/src/core/artifact_reservation.py`
  - deterministic failure-details path on the reservation.
- `framework/src/core/result_store.py`
  - `failure_details_path` CSV metadata;
  - strict immutable failure-details publication;
  - strict reserved CSV transaction-state query.
- `framework/src/core/async_inference/metrics.py`
  - unconditional terminal `outstanding` detail count.
- `framework/src/main.py`
  - safe recovery snapshots, unload-before-publication ordering, exact CSV
    recovery, normal-artifact preservation, recovery linking, runtime warning,
    and async lifecycle-only debug configuration.
- `framework/tests/test_async_result_artifacts.py`
  - strict failure-sidecar authority, safety, no-overwrite, and capability
    regressions.
- `framework/tests/test_async_cli.py`
  - ordering, warnings, privacy, actual active/pending/consumed artifact
    recovery, and post-commit fatal regressions.
- `framework/tests/test_async_cli_onnx_cpu.py`
  - real async debug isolation and e2e `--results-path` coverage.
- `framework/tests/test_async_metrics.py`
  - empty terminal outstanding count.
- `framework/tests/test_async_onnx_cpu.py`
  - loaded-session provider precedence.
- `framework/tests/test_async_runner.py`
  - non-fatal lifecycle callback failures across every phase.
- `docs/superpowers/specs/2026-07-14-async-inference-queue-design.md`
  - Section 42 and counter-contract consistency.
- `docs/async-inference-queue.md`
  - operator-facing failure recovery, debug, warning, and unload contracts.
- `framework/CHANGELOG.md`
  - Unreleased behavior changes.

## Verification

Pre-commit focused aggregate:

```bash
HF_DATASETS_CACHE=/tmp/mlhw-hf-cache-final-focused-2 \
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest \
  framework/tests/test_async_cli.py \
  framework/tests/test_async_cli_onnx_cpu.py \
  framework/tests/test_async_result_artifacts.py \
  framework/tests/test_async_metrics.py \
  framework/tests/test_async_runner.py \
  framework/tests/test_async_onnx_cpu.py -q
```

Result: `332 passed in 8.26s`.

Pre-commit full suite: `947 passed, 13 skipped, 1 warning in 25.69s`.

Post-commit full suite on `01a8c46`:

```bash
HF_DATASETS_CACHE=/tmp/mlhw-hf-cache-postcommit \
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests -q
```

Result: `947 passed, 13 skipped, 1 warning in 25.90s`.

Additional checks:

- `git diff --check`: pass before commit;
- `python -m py_compile` for all changed production Python modules: pass;
- staged scope inspection: exactly the 13 files listed above;
- commit created from starting HEAD with no unrelated file changes.

The one warning is pre-existing:
`PytestUnknownMarkWarning` for `pytest.mark.integration` in
`framework/tests/test_ettm_loader.py:223`.

## Self-review and remaining boundaries

- No blocking correctness, privacy, or no-overwrite concern remains in the
  reviewed scope.
- Failure CSV and recovery JSON are two durable publications, not one atomic
  filesystem transaction. The CLI emits no terminal `RUN_ID` until both are
  present. A process crash between them can therefore leave a consumed invalid
  row that points to the deterministic recovery path before that path exists;
  the reserved run ID and durable transaction state remain recovery evidence.
- Real-device automated coverage is ONNX Runtime CPU. GPU/NPU runtime behavior
  remains device-specific follow-up validation, as required by Section 42.
- The worktree is host-managed and intentionally preserved; no merge, push, or
  worktree cleanup was performed.

## Final re-review follow-up

Starting HEAD for this follow-up:
`01a8c46631d193619f0d1b60cc1fad9f04d0af61`.

### Important 1: terminal gating requires committed failure truth

Root cause: the exception boundary emitted `RUN_ID` when either
`failure_csv_saved` was true or a normal CSV was already committed. A consumed
normal CSV (including an exact retry from pending state) did not prove that the
immutable failure recovery record had committed.

Two actual-artifact regressions were added:

- a normal JSON and normal CSV are fully consumed, the original CSV
  `BaseException` is preserved, and failure recovery publication is forced to
  fail;
- the first normal CSV write leaves exact pending provenance, the CLI retries
  the same row to consumed, and failure recovery publication is forced to
  fail.

Both assert one `RUN_ID_RESERVED`, zero terminal `RUN_ID`, exact primary
identity, no recovery file, and unchanged normal sidecar bytes. The pending
case also proves that the consumed fingerprint equals the captured pending
fingerprint.

RED command:

```bash
HF_DATASETS_CACHE=/tmp/mlhw-hf-rereview-red-terminal \
/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_async_cli.py \
  -k 'consumed_normal_csv_without_recovery or pending_normal_csv_without_recovery' \
  -q
```

RED result: `2 failed, 51 deselected`; both failures observed one unexpected
`RUN_ID=async001`.

GREEN command:

```bash
HF_DATASETS_CACHE=/tmp/mlhw-hf-rereview-green-terminal \
/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_async_cli.py \
  -k 'consumed_normal_csv_without_recovery or pending_normal_csv_without_recovery' \
  -q
```

GREEN result: `2 passed, 51 deselected in 0.15s`.

Minimal fix: terminal emission now requires `failure_csv_saved` itself. A
normal committed/consumed CSV cannot substitute for committed failure truth.

### Important 2: ordinary sidecar exceptions enter immutable recovery

Root cause: the normal `save_async_details` `Exception` handler captured commit
evidence and diagnostics but continued result shaping. This converted the
exception into a nonzero return and bypassed the outer `BaseException` recovery
boundary.

Actual pre-commit and proven-post-commit regressions assert:

- the exact original `OSError` object is re-raised;
- a pre-commit failure can use the still-empty normal details slot for failure
  truth and link one invalid CSV row;
- a proven committed normal sidecar retains identical bytes;
- a writable invalid CSV links both the preserved normal details and
  `failure_details_path`;
- `{run_id}.failure.json` contains generic failure truth without the raw secret
  exception message.

RED command:

```bash
HF_DATASETS_CACHE=/tmp/mlhw-hf-rereview-red-sidecar \
/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_async_cli.py \
  -k 'precommit_normal_sidecar_exception or committed_normal_sidecar_exception' \
  -q
```

RED result: `2 failed, 53 deselected`; both failed with `DID NOT RAISE`.

GREEN command:

```bash
HF_DATASETS_CACHE=/tmp/mlhw-hf-rereview-green-sidecar \
/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_async_cli.py \
  -k 'precommit_normal_sidecar_exception or committed_normal_sidecar_exception' \
  -q
```

GREEN result: `2 passed, 53 deselected in 0.15s`.

The two pre-existing mock sidecar tests were updated to the Section 42
exception-preservation contract. Their focused result was
`2 passed, 53 deselected in 0.17s`.

Minimal fix: after safe diagnostic capture and optional committed-sidecar
evidence recording, the handler uses a bare `raise`, preserving exact exception
identity for the outer recovery path.

### Minor findings

The `--debug` help test first failed because the help text mentioned sample
output only:

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest \
  framework/tests/test_async_cli.py::test_debug_help_distinguishes_e2e_samples_from_async_lifecycle \
  -q
```

RED result: `1 failed`; `e2e` was absent. After documenting e2e sample debug
versus async coarse lifecycle-only debug, the same command returned
`1 passed in 0.10s`.

A direct strict result-store pending-authority characterization was added:

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest \
  framework/tests/test_async_result_artifacts.py::test_save_async_failure_details_accepts_and_preserves_pending_authority \
  -q
```

Result on first run: `1 passed in 0.22s`. No production change was needed:
strict failure-detail publication already accepted verified pending authority
and preserved its pending fingerprint/state. This test closes the direct API
coverage gap that had previously been exercised only through CLI recovery.

### Follow-up verification

CLI module:

```bash
HF_DATASETS_CACHE=/tmp/mlhw-hf-rereview-cli-final \
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_async_cli.py -q
```

Result: `56 passed in 0.39s`.

Focused aggregate:

```bash
HF_DATASETS_CACHE=/tmp/mlhw-hf-rereview-focused \
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest \
  framework/tests/test_async_cli.py \
  framework/tests/test_async_cli_onnx_cpu.py \
  framework/tests/test_async_result_artifacts.py \
  framework/tests/test_async_metrics.py \
  framework/tests/test_async_runner.py \
  framework/tests/test_async_onnx_cpu.py -q
```

Result: `338 passed in 8.34s`.

Full suite:

```bash
HF_DATASETS_CACHE=/tmp/mlhw-hf-rereview-full \
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests -q
```

Result: `953 passed, 13 skipped, 1 warning in 25.75s`.

Additional checks:

- `git diff --check`: pass;
- `python -m py_compile framework/src/main.py`: pass;
- warning remains the pre-existing unregistered `integration` mark at
  `framework/tests/test_ettm_loader.py:223`.
