> **Historical implementation snapshot — superseded.** The runner-removal
> design and plan supersede this report. `AsyncBenchmarkRunner` has since been
> removed; current public async ownership belongs to `InferenceEngine`.

# Unified Task 5 recovery report

## Status and scope

- Status: DONE
- Recovery base/merge: `714801f0638037fc948cc0b68074f5cd8a0815df`
- Worktree: `/tmp/ml-hw-benchmark-async-worktree`
- Task 5 source scope: `framework/src/core/inference_engine.py` and
  `framework/src/core/async_inference/runner.py`
- Test scope: `test_inference_engine.py`, `test_async_runner.py`, and
  `test_async_onnx_cpu.py`
- `framework/src/core/async_inference/__init__.py` was inspected and intentionally
  left unchanged so its public `__all__` contract remains unchanged.
- No Task 6 runtime executor, generation, evaluator, or async engine files were
  modified. No subagents were used.

## External-overlap audit and TDD recovery

The required isolated delta was inspected with:

```bash
git diff fe600c9..aeaea92 -- \
  framework/src/core/inference_engine.py \
  framework/src/core/async_inference/runner.py \
  framework/tests/test_inference_engine.py \
  framework/tests/test_async_runner.py
```

The imported production delta was then removed with `apply_patch`. A comparison
against `fe600c9` confirmed that both production files exactly matched their
pre-Task-5 shape while the Task 4 executor injection and all imported tests were
retained. The missing ownership tests were added before production was
reimplemented.

### RED evidence

The ownership/parity selection failed exactly on the absent public API:

```text
3 failed, 94 deselected in 0.27s
- InferenceEngine.run_async: AttributeError (quality parity)
- InferenceEngine.run_async: AttributeError (direct ONNX CPU parity)
- AsyncBenchmarkRunner.engine: AttributeError (compatibility facade)
```

The shared-identity and one-mode/one-run selection independently failed on the
same missing ownership behavior:

```text
3 failed, 19 deselected in 0.09s
- shared async pipeline/executor: run_async AttributeError
- e2e then async guard: run_async AttributeError
- async then any later mode guard: run_async AttributeError
```

## Implementation

- `InferenceEngine` lazily owns exactly one `InferencePipeline` and its compatible
  runtime executor. Lazy construction preserves validation-before-loader-side-
  effects behavior for the compatibility runner.
- `InferenceEngine.run_async()` validates public inputs, claims the engine's
  single execution mode, lazily imports and creates `_AsyncRunController`, saves
  it for diagnostics, and delegates the run.
- `run_e2e()` and `run_async()` share one lock-protected one-run/one-mode claim;
  `warmup()` remains usable before a run.
- `_AsyncRunController` receives the engine-owned pipeline and executor. Its
  former fallback `InferencePipeline` construction was removed, so async
  orchestration cannot silently create a second ownership graph.
- Public `AsyncBenchmarkRunner` keeps its constructor and `run()` signatures,
  creates one `InferenceEngine`, forwards the monitor, and delegates
  `failure_phase` and `runtime_unload_safe_after_failure` to the active private
  controller.
- `_AsyncRunController` has one construction call site and is not re-exported by
  the async inference package.

## Tests added or strengthened

- Same public engine type produces equal E2E/async quality and zero outstanding
  async requests.
- Engine and private controller share the identical pipeline and injected
  executor.
- An async run prevents both a second async run and a later E2E run; the existing
  opposite-direction and concurrent facade guards remain covered.
- Compatibility facade exposes its engine, initial/completed failure phases,
  unload-safety diagnostic, and zero-outstanding result.
- Direct tiny-model ONNX CPU E2E/async parity covers accuracy and sample count.

The parity fixture returns JSON-stable list pairs because async metrics are
intentionally total-serialized while direct E2E returns evaluator values as-is.
This retains the literal equality assertion without changing the established
async result/CLI serialization contract.

## GREEN and verification evidence

Narrow ownership/guard suite:

```text
6 passed, 91 deselected in 0.25s
```

Focused Task 5 suite:

```text
101 passed in 2.97s
```

Command:

```bash
PYTHONPATH=framework/src .../framework/.venv/bin/python -m pytest \
  framework/tests/test_inference_engine.py \
  framework/tests/test_async_runner.py \
  framework/tests/test_async_onnx_cpu.py \
  framework/tests/test_inference_pipeline.py -q
```

Full framework suite with the datasets cache redirected to a writable sandbox
path while retaining the existing tokenizer cache:

```text
1000 passed, 13 skipped, 1 warning in 50.40s
```

Command:

```bash
HF_DATASETS_CACHE=/tmp/ml-hw-hf-datasets PYTHONPATH=framework/src \
  .../framework/.venv/bin/python -m pytest framework/tests -q
```

The sole warning is the pre-existing unknown `integration` mark in
`test_ettm_loader.py`. Two preliminary full runs each reached 999 passes before
the unrelated tokenizer test failed: the first encountered a read-only default
datasets-cache lock; the second over-broad `HF_HOME` override hid the cached
tokenizer and attempted unavailable network access. The scoped cache override
above resolves both environmental constraints without a source change.

## Self-review

- `git diff --check` is clean.
- The net diff contains one Task 5 production correction, three Task 5 test
  files, and this new report. `inference_engine.py` and the package initializer
  were revalidated through the mandated recovery but need no net change from the
  imported implementation.
- The async package export list is unchanged and contains only the public runner.
- `_AsyncRunController` contains no `InferencePipeline` construction; the only
  Task 5 pipeline construction is the engine-owned lazy property.
- Existing lifecycle, validation ordering, monitor forwarding, result schema,
  serialization, CLI, and native executor tests pass in the full suite.
- No reset, revert, checkout, or native executor behavior change was used.

## Independent review fixes

An independent review did not approve the initial recovery commit
`1f7aae88f779ba0da418495b8624be0e69682391`. Its three Important findings were
handled sequentially with a separate RED/GREEN cycle for each item.

### 1. Engine ownership of async completion diagnostics

RED tests covered both a successful run and a runtime-failure result. Both
finished with `InferenceEngine.completion is None`:

```text
2 failed, 22 deselected in 0.13s
```

`InferenceEngine.run_async()` now creates the async metrics collector and
`CompletionCoordinator`, retains the coordinator as `engine.completion`, and
injects the same metrics/coordinator objects into `_AsyncRunController`. The
controller no longer constructs or owns an independent coordinator.

GREEN proves identity and an empty outstanding snapshot after both outcomes:

```text
2 passed, 22 deselected in 0.11s
```

### 2. Validation and lazy-construction failure diagnostics

RED covered invalid config, invalid warmup count, and dataloader metadata failure:

```text
3 failed, 70 deselected in 0.16s
- public validation failures retained failure_phase="created"
- metadata failure left engine._async_controller unset
```

The engine now prepares a diagnostic controller before validation. That
controller emits and retains `validation`, then performs config/warmup
validation without claiming the engine run. Only successful validation reaches
the lock-protected one-mode claim and subsequent pipeline/executor/coordinator
construction. The successful claimant is reinstalled as the active controller,
so a concurrent losing attempt cannot replace active diagnostics.

GREEN includes the new diagnostics plus the prior no-side-effect and
invalid-input-does-not-consume-one-shot contracts:

```text
5 passed, 68 deselected in 0.08s
```

Metadata/pipeline construction failure now preserves the controller, the
`validation` phase/callback, and unload-safe diagnostic while correctly leaving
`engine.completion` unset because coordinator construction was never reached.

### 3. Public compatibility properties without shadow ownership

RED showed that constructor fields were absent and assignments created dead
façade attributes ignored by execution:

```text
2 failed, 73 deselected in 0.18s
```

The façade now exposes engine-delegating properties, with setters, for
`dataloader`, `runtime`, `evaluator`, `max_new_tokens`, `decoder`,
`trace_callback`, `lifecycle_callback`, and `runtime_executor`. It stores no
duplicate copies of those values. A first GREEN attempt found an invalid test
assumption that offline indexed loading advances `current_idx`; the test was
corrected to assert the replacement loader's exact `load_by_index` events.

Final GREEN proves constructor identity and that pre-run mutations drive the
actual loader, runtime capability queries, evaluator, decoder, executor,
callbacks, and pipeline token configuration:

```text
2 passed, 73 deselected in 0.15s
```

### Review-fix verification

Focused ownership/facade/ONNX/pipeline suite:

```text
108 passed in 3.02s
```

Full framework suite:

```text
1007 passed, 13 skipped, 1 warning in 50.66s
```

The warning remains the pre-existing unknown `integration` mark. The review-fix
diff is limited to Task 5's engine, async runner, their two focused test files,
and this report; no Task 6/native executor, generation, evaluator, or async
engine implementation file was changed.

## Dependency-mutation re-review fix

The follow-up review found that the compatibility setters could leave an
already-materialized pipeline and default executor bound to the constructor
dependencies. It also required a deterministic policy for concurrent and
post-run mutation.

### RED evidence

The regression selection was added before the production correction. It covered
replacement of materialized default dependencies, preservation of an explicitly
injected executor, mutation after claim and during an active run, and every
public dependency setter after completion:

```text
13 failed, 75 deselected in 0.46s
- 2 stale materialized-pipeline/default-executor failures
- 1 proposed side-effect-free default-executor getter failure
- 2 claim/active mutation-policy failures
- 8 completed-run mutation-policy failures
```

The proposed side-effect-free getter was then rejected as an unsafe contract,
not implemented: `BlockingRuntimeExecutor` needs pipeline-derived LLM state,
including dataloader `stop_token_ids`, before it is executable. The regression
was changed to characterize the required behavior instead: the getter
materializes the compatible pipeline metadata and returns that pipeline's exact
executor. This preserves the existing immediately-correct public getter.

### Implementation

- `InferenceEngine._set_dependency()` is the single internal mutation path used
  by all compatibility-facade setters. It acquires the run-claim lock before the
  pipeline lock and rejects every dependency change once a run has been claimed,
  including active and completed runs.
- The engine tracks whether the executor is explicit or automatically derived.
  Changes to `dataloader`, `runtime`, or `max_new_tokens` invalidate the cached
  pipeline and clear only an automatically derived executor. An explicitly
  injected executor survives invalidation and is installed into the rebuilt
  pipeline.
- The default-executor getter and pipeline property share lock-protected
  materialization, so concurrent reads cannot create incompatible ownership
  graphs.
- After the successful run claim, the controller is rebound to the engine's
  current dependencies together with the rebuilt pipeline, executor, metrics,
  and completion coordinator. Thus a legal pre-claim mutation drives the actual
  run rather than the controller's initial validation snapshot.
- Evaluator, decoder, trace, lifecycle, and executor setters follow the same
  one-shot mutation policy even though only the pipeline-defining dependencies
  require cache invalidation.

### GREEN and verification evidence

Selected dependency regressions:

```text
13 passed, 75 deselected in 0.25s
```

Focused Task 5 engine/runner/ONNX/pipeline suite:

```text
121 passed in 3.28s
```

Full framework suite:

```text
1020 passed, 13 skipped, 1 warning
```

The full run used `pytest framework/tests -qq`, returned exit code 0, and a
separate collection check reported 1033 tests. The warning remains the
pre-existing unknown `integration` mark in `test_ettm_loader.py`.

Final scope checks report a clean `git diff --check`. The re-review delta touches
only `inference_engine.py`, the compatibility runner, its focused regression
tests, and this report; Task 6/native executor files remain untouched.
