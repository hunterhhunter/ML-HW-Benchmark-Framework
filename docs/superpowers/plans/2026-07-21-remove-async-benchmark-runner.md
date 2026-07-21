# AsyncBenchmarkRunner Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the public `AsyncBenchmarkRunner` façade before merge so the CLI and all active async consumers use `InferenceEngine.run_async()` directly without changing async execution, result, or failure semantics.

**Architecture:** `InferenceEngine` becomes the only public inference owner for the async path. It exposes the two read-only failure diagnostics formerly forwarded by the façade, while private `_AsyncRunController`, `AsyncInferenceEngine`, bounded queues, completion, and artifact persistence remain unchanged. The removal is an intentional breaking API change with no deprecated alias.

**Tech Stack:** Python 3.12, pytest, ONNX Runtime CPU, Git worktree, Markdown documentation

## Global Constraints

- Work only in `/tmp/ml-hw-benchmark-async-worktree` on `feat/async-inference-queue`.
- Follow strict TDD for every production change: write and run the specified failing test before modifying production code.
- Do not add a deprecated alias, compatibility module, factory, or replacement runner class.
- Keep `_AsyncRunController`, `AsyncInferenceEngine`, producer, queue, worker, completion, metrics, and result schemas behaviorally unchanged.
- Do not remove or redesign the e2e `BenchmarkRunner` in this plan.
- Preserve CLI options, reserved/final run IDs, CSV/details/trace linkage, failure persistence, and runtime-unload safety.
- Do not add MLPerf LoadGen code, API/log compatibility, submission, or compliance behavior.
- Use `HF_DATASETS_CACHE=/tmp/mlhw-remove-runner-hf-cache`, `HF_DATASETS_OFFLINE=1`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` for framework regression commands.
- Baseline at plan creation: `1155 passed, 13 skipped`, with the pre-existing unknown `integration` marker warning.

---

### Task 1: Move async failure diagnostics onto InferenceEngine

**Files:**
- Modify: `framework/tests/test_inference_engine.py`
- Modify: `framework/src/core/inference_engine.py`

**Interfaces:**
- Consumes: existing private `InferenceEngine._async_controller`
- Produces: public read-only `InferenceEngine.failure_phase -> str`
- Produces: public read-only `InferenceEngine.runtime_unload_safe_after_failure -> bool`

- [ ] **Step 1: Write failing public-diagnostic tests**

Add these tests beside the existing async ownership tests in
`framework/tests/test_inference_engine.py`:

```python
def test_inference_engine_exposes_safe_async_diagnostics_before_run():
    engine = InferenceEngine(FakeLoader(), FakeRuntime(), FakeEvaluator())

    assert engine.failure_phase == "created"
    assert engine.runtime_unload_safe_after_failure is True


def test_inference_engine_exposes_controller_diagnostics_after_validation():
    engine = InferenceEngine(FakeLoader(), FakeRuntime(), FakeEvaluator())

    with pytest.raises(ValueError, match="warmup_runs"):
        engine.run_async(
            AsyncInferenceConfig(min_samples=1),
            warmup_runs=-1,
        )

    assert engine.failure_phase == "validation"
    assert engine.runtime_unload_safe_after_failure is True


def test_inference_engine_exposes_controller_diagnostics_after_success():
    engine = InferenceEngine(FakeLoader(), FakeRuntime(), FakeEvaluator())

    result = engine.run_async(
        AsyncInferenceConfig(batch_timeout_ms=0, min_samples=1),
        warmup_runs=0,
    )

    assert result.status is RunStatus.VALID
    assert engine.failure_phase == "complete"
    assert engine.runtime_unload_safe_after_failure is True
```

Import `RunStatus` from `core.async_inference.types` in the existing import
block if it is not already imported.

- [ ] **Step 2: Run the tests and verify RED**

```bash
PYTHONPATH=framework/src \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_inference_engine.py \
  -k 'exposes_safe_async_diagnostics or exposes_controller_diagnostics' -q
```

Expected: three failures with `AttributeError` because the public properties do
not exist. Do not touch production before recording this output in
`.superpowers/sdd/remove-async-benchmark-runner-task-1.md`.

- [ ] **Step 3: Implement the two read-only properties**

Add immediately after `_prepare_async_diagnostics()` in
`framework/src/core/inference_engine.py`:

```python
    @property
    def failure_phase(self) -> str:
        controller = self._async_controller
        return "created" if controller is None else controller.failure_phase

    @property
    def runtime_unload_safe_after_failure(self) -> bool:
        controller = self._async_controller
        if controller is None:
            return True
        return controller.runtime_unload_safe_after_failure
```

Do not expose `_async_controller`, add setters, or copy the diagnostic values
onto the engine.

- [ ] **Step 4: Run focused GREEN and lifecycle regression**

```bash
PYTHONPATH=framework/src \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_inference_engine.py \
  framework/tests/test_async_runner.py \
  -k 'diagnostic or failure_phase or runtime_unload_safe or unified or ownership' -q
```

Expected: PASS. Confirm that validation failure still leaves the controller
available for diagnostics and does not claim a runnable engine.

- [ ] **Step 5: Commit Task 1**

```bash
git add framework/src/core/inference_engine.py \
  framework/tests/test_inference_engine.py
git commit -m "feat(framework): expose async engine diagnostics"
```

Dispatch a fresh spec reviewer, then a quality reviewer. Fix every
Critical/Important finding with a new RED regression before starting Task 2.

---

### Task 2: Make the CLI call InferenceEngine directly

**Files:**
- Modify: `framework/tests/test_async_cli.py`
- Modify: `framework/src/main.py`

**Interfaces:**
- Consumes: `InferenceEngine(loader, runtime, evaluator, *, decoder, max_new_tokens, trace_callback, lifecycle_callback)`
- Consumes: `InferenceEngine.run_async(config, warmup_runs=1, monitor=None) -> AsyncBenchmarkResult`
- Consumes: `InferenceEngine.failure_phase`
- Consumes: `InferenceEngine.runtime_unload_safe_after_failure`
- Produces: unchanged `execute_benchmark(...) -> int` CLI contract

- [ ] **Step 1: Convert the shared CLI fake into an engine and force RED**

In `_execute()` in `framework/tests/test_async_cli.py`, replace the local
`Runner` with this engine test double:

```python
    class Engine:
        def __init__(self, **kwargs):
            events.append(("engine_init", kwargs))
            self.failure_phase = "created"
            self.runtime_unload_safe_after_failure = True

        def run_async(self, config, warmup_runs, monitor):
            events.append(("async_run", config, warmup_runs, monitor))
            self.failure_phase = "complete"
            return result or _result()

    class ForbiddenRunner:
        def __init__(self, **kwargs):
            del kwargs
            raise AssertionError("AsyncBenchmarkRunner must not be constructed")
```

Patch both symbols so the current implementation fails for the intended
reason:

```python
    monkeypatch.setattr(
        benchmark_main,
        "InferenceEngine",
        Engine,
        raising=False,
    )
    monkeypatch.setattr(
        benchmark_main,
        "AsyncBenchmarkRunner",
        ForbiddenRunner,
        raising=False,
    )
```

Update lifecycle-callback assertions to look for `"engine_init"` instead of
`"async_init"`, and rename the test to
`test_async_engine_lifecycle_callback_is_debug_only`.

- [ ] **Step 2: Run one CLI test and verify RED**

```bash
PYTHONPATH=framework/src \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest \
  framework/tests/test_async_cli.py::test_async_branch_reserves_before_measurement_and_propagates_token \
  -q
```

Expected: FAIL with
`AssertionError: AsyncBenchmarkRunner must not be constructed`. Record the
tests-only diff and output in
`.superpowers/sdd/remove-async-benchmark-runner-task-2.md` before editing
`main.py`.

- [ ] **Step 3: Change main.py imports and async construction**

Replace the current import with:

```python
from core.inference_engine import InferenceEngine
from core.async_inference import (
    AsyncInferenceConfig,
    AsyncScenario,
    RunStatus,
)
```

In `execute_benchmark()`, rename the async local owner from `runner` to
`engine`, initialize it to `None`, and construct it directly:

```python
        engine = InferenceEngine(
            dataloader=loader,
            runtime=runtime,
            evaluator=evaluator,
            max_new_tokens=args.max_new_tokens,
            decoder=decoder,
            trace_callback=(
                trace_writer.write if trace_writer is not None else None
            ),
            lifecycle_callback=(
                (
                    lambda lifecycle_phase: _debug_lifecycle(
                        args,
                        lifecycle_phase,
                        "start",
                        reservation,
                    )
                )
                if args.debug
                else None
            ),
        )
        async_result = engine.run_async(
            config,
            warmup_runs=args.warmup,
            monitor=hw_monitor,
        )
```

Use `engine.failure_phase` in the exception path. Rename
`_cleanup_async_run_failure(primary, runner, ...)`'s parameter to `engine` and
read `engine.runtime_unload_safe_after_failure`. Preserve the exact existing
fallback default `False`, secondary-error attachment, trace close, unload, and
failure-persistence order.

- [ ] **Step 4: Migrate all CLI test doubles to the engine contract**

For every custom patch currently shaped as:

```python
monkeypatch.setattr(benchmark_main, "AsyncBenchmarkRunner", Runner)
```

use:

```python
monkeypatch.setattr(benchmark_main, "InferenceEngine", Engine)
```

Each test double must expose this exact public surface:

```python
class Engine:
    failure_phase = "created"
    runtime_unload_safe_after_failure = True

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run_async(self, config, warmup_runs, monitor):
        del config, warmup_runs, monitor
        return async_result
```

When a test injects a failure phase or unsafe unload state, retain those exact
values. Do not emulate private controller fields. Keep existing
`async_runner_module.AsyncInferenceEngine` fault injection because the private
controller remains in that module. After the RED evidence has been recorded
and `main.py` uses `InferenceEngine`, delete `ForbiddenRunner` and its
`AsyncBenchmarkRunner` monkeypatch so the active CLI tests contain no façade
reference.

- [ ] **Step 5: Run CLI GREEN and exact failure-boundary tests**

```bash
PYTHONPATH=framework/src \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_async_cli.py -q
```

Expected: all CLI tests PASS, including reservation, validation/warmup/start/
shutdown failure persistence, runtime unload safety, CSV recovery, and
no-success-on-failure cases.

- [ ] **Step 6: Commit Task 2**

```bash
git add framework/src/main.py framework/tests/test_async_cli.py
git commit -m "refactor(framework): call inference engine from async cli"
```

Dispatch fresh spec and quality reviewers. Do not remove the façade until this
CLI review is approved.

---

### Task 3: Migrate behavior tests and non-CLI consumers to InferenceEngine

**Files:**
- Modify: `framework/tests/test_async_runner.py`
- Modify: `framework/tests/test_async_onnx_cpu.py`
- Modify: `framework/tests/test_object_detection_loader_async.py`
- Modify: `framework/tests/_async_hostile_result_process.py`

**Interfaces:**
- Consumes: public `InferenceEngine.run_async(config, warmup_runs=1, monitor=None)`
- Produces: no production interface; active tests stop depending on the façade

- [ ] **Step 1: Replace the common test consumer pattern**

In all four files import:

```python
from core.inference_engine import InferenceEngine
```

Replace one-shot calls of this form:

```python
result = AsyncBenchmarkRunner(
    loader,
    runtime,
    evaluator,
    decoder=decoder,
).run(config, warmup_runs=0)
```

with:

```python
result = InferenceEngine(
    loader,
    runtime,
    evaluator,
    decoder=decoder,
).run_async(config, warmup_runs=0)
```

For monitor cases move monitor ownership to the run call:

```python
engine = InferenceEngine(loader, runtime, evaluator)
result = engine.run_async(
    config,
    warmup_runs=0,
    monitor=monitor,
)
```

For tests that inspect post-run diagnostics, assert
`engine.failure_phase` and `engine.runtime_unload_safe_after_failure` directly.

- [ ] **Step 2: Remove façade-only tests without deleting engine coverage**

Delete tests whose only contract is façade forwarding or setter mutation:

```text
test_async_runner_is_compatibility_facade_over_inference_engine
test_async_runner_public_constructor_fields_delegate_to_engine
test_async_runner_public_setters_update_the_executed_engine
test_materialized_default_dependencies_are_rebuilt_before_run
test_explicit_executor_survives_dependency_invalidation
test_dependency_mutation_is_rejected_after_run_claim_before_setup
test_dependency_mutation_is_rejected_during_active_run
test_dependency_mutation_is_rejected_after_completed_run
```

Keep and migrate actual executor identity, default executor materialization,
single-run, validation, warmup, producer, monitor, result, fault, flush,
shutdown, and callback tests. Where a kept test references `runner.engine`,
replace it with the direct `engine` variable. Do not change the positive
package export test yet; Task 4 replaces it with the removal RED test.

- [ ] **Step 3: Run the migrated consumer suite**

```bash
PYTHONPATH=framework/src \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_async_runner.py \
  framework/tests/test_async_onnx_cpu.py \
  framework/tests/test_object_detection_loader_async.py -q
```

Expected: PASS while the production façade still exists but is unused by
these behavior tests.

- [ ] **Step 4: Run the hostile-result subprocess directly**

```bash
PYTHONPATH=framework/src \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  framework/tests/_async_hostile_result_process.py
```

Expected: exit 0 with the same bounded serialized result and no reference to
`AsyncBenchmarkRunner`.

- [ ] **Step 5: Verify active consumer migration**

```bash
rg -n '\bAsyncBenchmarkRunner\b' \
  framework/src/main.py \
  framework/tests/test_async_cli.py \
  framework/tests/test_async_onnx_cpu.py \
  framework/tests/test_object_detection_loader_async.py \
  framework/tests/_async_hostile_result_process.py
```

Expected: no matches. `test_async_runner.py` may still contain only the one
positive export test reserved for Task 4.

- [ ] **Step 6: Commit Task 3**

```bash
git add framework/tests/test_async_runner.py \
  framework/tests/test_async_onnx_cpu.py \
  framework/tests/test_object_detection_loader_async.py \
  framework/tests/_async_hostile_result_process.py
git commit -m "test(framework): migrate async consumers to inference engine"
```

Dispatch fresh spec and quality reviewers. Characterization migrations do not
need an artificial RED because this task has no production change, but every
migrated behavior must be green before Task 4.

---

### Task 4: Delete the façade/export and record the breaking change

**Files:**
- Modify: `framework/tests/test_async_runner.py`
- Modify: `framework/src/core/async_inference/runner.py`
- Modify: `framework/src/core/async_inference/__init__.py`
- Modify: `docs/unified-inference-engine-design.md`
- Modify: `docs/superpowers/specs/2026-07-14-async-inference-queue-design.md`
- Modify: `docs/superpowers/specs/2026-07-21-remove-async-benchmark-runner-design.md`
- Modify: `framework/src/core/README.md`
- Modify: `framework/CHANGELOG.md`

**Interfaces:**
- Removes: `core.async_inference.AsyncBenchmarkRunner`
- Removes: `core.async_inference.runner.AsyncBenchmarkRunner`
- Preserves: `InferenceEngine.run_async(...) -> AsyncBenchmarkResult`
- Preserves: all public async config/result/request/status types

- [ ] **Step 1: Replace the positive export test with the removal contract**

Replace `test_runner_is_exported_from_async_inference_package` with:

```python
def test_async_runner_facade_is_removed():
    facade_name = "AsyncBenchmark" + "Runner"

    assert facade_name not in async_inference.__all__
    assert not hasattr(async_inference, facade_name)
    assert not hasattr(async_runner_module, facade_name)
```

Remove any top-level `from ... import AsyncBenchmarkRunner` statement first so
the test module can collect while production still exposes the class.

- [ ] **Step 2: Run the removal test and verify RED**

```bash
PYTHONPATH=framework/src \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest \
  framework/tests/test_async_runner.py::test_async_runner_facade_is_removed \
  -q
```

Expected: FAIL because the class remains in the package export and runner
module. Record this tests-only RED in
`.superpowers/sdd/remove-async-benchmark-runner-task-4.md`.

- [ ] **Step 3: Remove the production class and package export**

Delete the complete `class AsyncBenchmarkRunner:` block from
`framework/src/core/async_inference/runner.py`. Do not alter
`_AsyncRunController` or any helper above it.

Change `framework/src/core/async_inference/__init__.py` so it contains no
runner import and the export list begins:

```python
__all__ = [
    "AsyncBenchmarkResult",
    "AsyncInferenceConfig",
    "AsyncScenario",
    "BatchCompletion",
    "EngineState",
    "FirstTokenEvent",
    "InferenceRequest",
    "RequestTrace",
    "RunStatus",
    "TerminalStatus",
]
```

- [ ] **Step 4: Verify GREEN and absence in production/tests**

```bash
PYTHONPATH=framework/src \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_async_runner.py \
  framework/tests/test_inference_engine.py \
  framework/tests/test_async_cli.py -q
```

Expected: PASS.

```bash
rg -n '\bAsyncBenchmarkRunner\b' framework/src framework/tests
```

Expected: no matches.

- [ ] **Step 5: Update current architecture docs and CHANGELOG**

Make these exact semantic changes:

- `docs/unified-inference-engine-design.md`: draw `main.py -> InferenceEngine`
  for async; retain only `BenchmarkRunner` as an e2e compatibility façade.
- `docs/superpowers/specs/2026-07-14-async-inference-queue-design.md`: mark
  `AsyncBenchmarkRunner` as historical and replace current architecture and
  file-boundary sections with direct `InferenceEngine.run_async()`.
- `framework/src/core/README.md`: remove the async façade row and show direct
  CLI-to-engine flow.
- `framework/CHANGELOG.md`: under `[Unreleased]`, record the intentional
  removal and direct engine migration; do not claim backward compatibility.
- `docs/superpowers/specs/2026-07-21-remove-async-benchmark-runner-design.md`:
  change status from `승인됨` to `구현됨` only after focused tests pass.

Historical plans remain unchanged. The removal design and CHANGELOG may use
the deleted name when documenting history.

- [ ] **Step 6: Run native/race regression**

```bash
PYTHONPATH=framework/src \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_native_async_runtime_executor.py \
  framework/tests/test_async_engine.py \
  framework/tests/test_async_completion.py -q
```

Expected: PASS with no queue, ACK, shutdown, or cancellation changes.

Repeat the key ownership set in ten fresh pytest processes:

```bash
for run in 1 2 3 4 5 6 7 8 9 10; do
  PYTHONPATH=framework/src \
    /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
    -m pytest \
    framework/tests/test_async_engine.py::test_late_completion_ack_retires_handoffs_after_worker_exit \
    framework/tests/test_async_engine.py::test_drain_generation_is_reserved_before_queue_journal_publication \
    framework/tests/test_async_engine.py::test_deferred_reaper_retries_only_exact_canonical_drain \
    -q || exit 1
done
```

Expected: every process passes.

- [ ] **Step 7: Run actual ONNX Runtime CPU acceptance**

```bash
HF_DATASETS_CACHE=/tmp/mlhw-remove-runner-hf-cache \
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=framework/src \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_async_cli_onnx_cpu.py \
  framework/tests/test_async_onnx_cpu.py \
  --basetemp=/tmp/remove-async-runner-onnx -q
```

Expected: actual `python src/main.py` subprocess exits 0, uses
`CPUExecutionProvider`, writes the selected path, reports valid counter
invariants and zero outstanding, and links the same run ID across
CSV/details/trace.

- [ ] **Step 8: Run full verification and hygiene**

```bash
HF_DATASETS_CACHE=/tmp/mlhw-remove-runner-hf-cache \
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=framework/src \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests -q -ra
```

Expected: all available tests PASS; only the pre-existing unknown
`integration` marker warning is acceptable.

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m py_compile framework/src/main.py \
  framework/src/core/inference_engine.py \
  framework/src/core/async_inference/runner.py \
  framework/src/core/async_inference/__init__.py
git diff --check
git status --short
```

Expected: compile and diff check succeed. Before committing, status contains
only intended production, test, and documentation files.

- [ ] **Step 9: Commit Task 4 and request final whole-branch review**

```bash
git add framework/src/core/async_inference/runner.py \
  framework/src/core/async_inference/__init__.py \
  framework/tests/test_async_runner.py \
  docs/unified-inference-engine-design.md \
  docs/superpowers/specs/2026-07-14-async-inference-queue-design.md \
  docs/superpowers/specs/2026-07-21-remove-async-benchmark-runner-design.md \
  framework/src/core/README.md framework/CHANGELOG.md
git commit -m "refactor(framework): remove async benchmark runner facade"
```

Generate a review package from `f2769a4` to `HEAD`. The final reviewer must
check the removal spec, all task reports, active import/export absence, CLI
failure cleanup, direct engine ownership, actual ONNX CPU evidence, and full
regression result. Every behavior fix starts with a new failing regression;
do not claim merge readiness until the re-review is approved.
