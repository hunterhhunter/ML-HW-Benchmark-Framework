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
