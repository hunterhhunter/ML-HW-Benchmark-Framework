# Async CLI Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the async queue executable diagnosable before and during measurement, isolate result output for CI, preserve malformed historical results without guessing their schema, and prove the real ONNX Runtime CPU CLI path works.

**Architecture:** Keep `RUN_ID=<id>` as the existing single completion signal and add `RUN_ID_RESERVED=<id>` immediately after async artifact reservation. Extend the current CLI orchestration with coarse lifecycle callbacks, safe runtime/run metadata snapshots, and best-effort failure artifact persistence that never replaces the original exception. Keep `result_store` strict; archive the malformed tracked CSV and let each results path start with a newly generated schema.

**Tech Stack:** Python 3.12, argparse, pytest, subprocess, ONNX, ONNX Runtime CPUExecutionProvider, Pillow, CSV/JSON/JSONL artifacts.

## Global Constraints

- Scope remains `framework` core, CLI, result storage, tests, and documentation only.
- Do not add or integrate `mlperf_loadgen`; MLPerf LoadGen remains a reliability reference only.
- Preserve the default `e2e` inference mode and the Backend's single `RUN_ID=<id>` completion-line contract.
- `RUN_ID_RESERVED=<id>` is diagnostic identity, not a successful-result signal.
- Do not auto-pad, truncate, or otherwise reinterpret malformed historical CSV rows.
- Do not persist traceback, input, label, output tensor, or prompt content in failure artifacts.
- Debug logging is phase-level only; do not synchronously print per-request lifecycle events.
- ONNX Runtime CPU with one worker is the required real-device CI target; other devices remain follow-up validation work.

---

### Task 1: Isolated result paths and legacy result preservation

**Files:**
- Modify: `framework/src/main.py`
- Modify: `.gitignore`
- Rename: `framework/results/benchmark_results.csv` to `framework/results/benchmark_results.legacy.csv`
- Create: `framework/results/README.md`
- Test: `framework/tests/test_async_cli.py`

**Interfaces:**
- Consumes: existing `execute_benchmark(args, *, loader, runtime, evaluator, decoder, hw_monitor, task_name, target_meta, results_path=None) -> int`.
- Produces: CLI argument `--results-path PATH`; `main()` passes `Path(args.results_path)` to `execute_benchmark`; generated default results are untracked while the original malformed bytes remain tracked as a legacy archive.

- [ ] **Step 1: Write failing CLI parsing tests**

Add parser assertions to `framework/tests/test_async_cli.py`:

```python
def test_results_path_is_common_and_defaults_to_none(tmp_path):
    assert parse([]).results_path is None
    chosen = tmp_path / "isolated" / "results.csv"
    assert parse(["--results-path", str(chosen)]).results_path == str(chosen)


def test_results_path_is_not_rejected_in_e2e_mode(tmp_path):
    args = parse(["--results-path", str(tmp_path / "results.csv")])
    benchmark_main.validate_async_args(args)
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
HF_DATASETS_CACHE=/tmp/mlhw-hf-cache \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_async_cli.py \
  -k 'results_path' -q
```

Expected: FAIL because the parser does not recognize or populate `results_path`.

- [ ] **Step 3: Add the common CLI option and pass it to execution**

Add to `build_parser()` beside the other common output/debug arguments:

```python
parser.add_argument(
    "--results-path",
    type=str,
    default=None,
    help=(
        "결과 CSV 경로. async details/trace도 이 CSV의 parent 아래에 저장됩니다. "
        "기본: framework/results/benchmark_results.csv"
    ),
)
```

Pass the parsed value at the bottom of `main()`:

```python
results_path = Path(args.results_path) if args.results_path else None
return execute_benchmark(
    args,
    loader=loader,
    runtime=runtime,
    evaluator=evaluator,
    decoder=decoder,
    hw_monitor=hw_monitor,
    task_name=task_enum.name,
    target_meta=target_meta,
    results_path=results_path,
)
```

When printing the result location, use the actual path selected for that run instead of the hard-coded `results/benchmark_results.csv` text in both e2e and async branches.

- [ ] **Step 4: Preserve the legacy bytes and ignore newly generated artifacts**

Rename the tracked CSV without changing its bytes:

```bash
git mv framework/results/benchmark_results.csv \
  framework/results/benchmark_results.legacy.csv
```

Append these repository-relative rules to `.gitignore`:

```gitignore
# Generated benchmark result artifacts
framework/results/benchmark_results.csv
framework/results/details/
framework/results/traces/
framework/results/.run_artifacts/
```

Create `framework/results/README.md` explaining that `benchmark_results.legacy.csv` is an immutable, structurally inconsistent historical archive, is not read by default, and must not be automatically padded or truncated. State that a new `benchmark_results.csv` is created on first execution.

- [ ] **Step 5: Verify parser behavior, archive identity, and clean generation**

Run the focused tests from Step 2 and:

```bash
git show HEAD:framework/results/benchmark_results.csv \
  | sha256sum
sha256sum framework/results/benchmark_results.legacy.csv
git check-ignore -v framework/results/benchmark_results.csv
```

Expected: tests PASS; both SHA-256 values match; the new runtime filename is ignored.

- [ ] **Step 6: Commit Task 1**

```bash
git add .gitignore framework/src/main.py framework/tests/test_async_cli.py
git add -A framework/results
git commit -m "feat(framework): isolate benchmark result paths"
```

---

### Task 2: Runtime diagnostics and runner lifecycle phases

**Files:**
- Modify: `framework/src/runtimes/onnx_rt.py`
- Modify: `framework/src/core/async_inference/runner.py`
- Modify: `framework/src/main.py`
- Test: `framework/tests/test_async_onnx_cpu.py`
- Test: `framework/tests/test_async_runner.py`
- Test: `framework/tests/test_async_cli.py`

**Interfaces:**
- Consumes: `Runtime.get_device_spec() -> Dict[str, Any]` and `AsyncBenchmarkRunner.run(config, warmup_runs=1)`.
- Produces: loaded ONNX sessions report their actual providers; `AsyncBenchmarkRunner(dataloader, runtime, evaluator, max_new_tokens=256, monitor=None, decoder=None, trace_callback=None, lifecycle_callback=None)` emits coarse phase strings and exposes `failure_phase`; normal and failure sidecars use `_safe_runtime_diagnostics(runtime) -> dict` and `_async_run_metadata(args, task_name, target_meta, runtime_diagnostics) -> dict`.

- [ ] **Step 1: Write RED tests for actual providers and lifecycle phases**

Extend the existing real ONNX test after `_load_cpu_runtime()`:

```python
def test_loaded_onnx_device_spec_reports_actual_cpu_provider(tmp_path):
    model_path = tmp_path / "tiny-sum.onnx"
    _create_sum_model(model_path)
    runtime = _load_cpu_runtime(model_path)
    try:
        assert runtime.get_device_spec()["active_providers"] == [
            "CPUExecutionProvider"
        ]
    finally:
        runtime.unload()
```

Add runner tests using the existing deterministic loader/runtime fixtures:

```python
def test_runner_emits_coarse_lifecycle_phases():
    phases = []
    runner = AsyncBenchmarkRunner(
        Loader(),
        Runtime(),
        Evaluator(),
        lifecycle_callback=phases.append,
    )
    result = runner.run(
        AsyncInferenceConfig(
            queue_capacity=4,
            max_batch_size=2,
            batch_timeout_ms=0,
            min_samples=1,
        ),
        warmup_runs=0,
    )
    assert result.status is RunStatus.VALID
    assert phases == [
        "validation",
        "engine_setup",
        "engine_start",
        "measurement",
        "finalization",
        "complete",
    ]
    assert runner.failure_phase == "complete"
```

Add this warmup-failure variant:

```python
def test_runner_failure_phase_identifies_warmup():
    primary = RuntimeError("warmup failed")
    phases = []

    class PhaseFailingWarmupRuntime(Runtime):
        def warmup(self, inputs, num_runs=1):
            del inputs, num_runs
            raise primary

    runner = AsyncBenchmarkRunner(
        Loader(),
        PhaseFailingWarmupRuntime(),
        Evaluator(),
        lifecycle_callback=phases.append,
    )
    with pytest.raises(RuntimeError) as raised:
        runner.run(
            AsyncInferenceConfig(
                queue_capacity=4,
                max_batch_size=1,
                min_samples=1,
            ),
            warmup_runs=1,
        )
    assert raised.value is primary
    assert phases[-1] == "warmup"
    assert runner.failure_phase == "warmup"
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
HF_DATASETS_CACHE=/tmp/mlhw-hf-cache \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_async_onnx_cpu.py \
  framework/tests/test_async_runner.py \
  -k 'device_spec_reports_actual or lifecycle_phases or failure_phase' -q
```

Expected: FAIL because lifecycle callback/failure phase do not exist. The provider test protects the post-load contract even if it happens to pass with the current CPU configuration.

- [ ] **Step 3: Make ONNX provider reporting reflect the loaded session**

Replace the fixed provider value in `OnnxRuntime.get_device_spec()` with:

```python
active_providers = (
    list(self.session.get_providers())
    if self.session is not None
    else list(self.providers)
)
return {
    "backend": "onnxruntime",
    "device": self.device,
    "active_providers": active_providers,
}
```

- [ ] **Step 4: Add non-invasive runner phase reporting**

Extend `AsyncBenchmarkRunner.__init__` with `lifecycle_callback=None`, initialize `_failure_phase = "created"`, and add:

```python
@property
def failure_phase(self):
    return self._failure_phase

def _set_phase(self, phase):
    self._failure_phase = phase
    if self.lifecycle_callback is None:
        return
    try:
        self.lifecycle_callback(phase)
    except Exception:
        return
```

Call `_set_phase()` only at coarse boundaries: before validation, after pipeline/metrics/coordinator/engine construction (`engine_setup`), before warmup, before `engine.start()`, before producer submission (`measurement`), before final metric/evaluator assembly (`finalization`), and immediately before returning (`complete`). Do not call it from request submission, worker, completion, metrics, or trace paths.

- [ ] **Step 5: Add safe run metadata helpers and attach them to normal async details**

In `framework/src/main.py`, add helpers with exact built-in outputs:

```python
def _safe_runtime_diagnostics(runtime) -> dict:
    try:
        value = runtime.get_device_spec()
    except BaseException as exc:
        return {"error": _safe_persistence_error("runtime_device_spec", exc)}
    return value if type(value) is dict else {
        "error": {
            "phase": "runtime_device_spec",
            "error_type": "TypeError",
            "error_message": "get_device_spec() did not return dict",
        }
    }


def _async_run_metadata(
    args, task_name, target_meta, runtime_diagnostics
) -> dict:
    artifact = args.onnx or args.hef or args.artifact or args.model_path or ""
    return {
        "model_name": args.model,
        "task": task_name,
        "backend": args.backend,
        "device": args.device,
        "batch_size": args.batch_size,
        "warmup_runs": args.warmup,
        "target_id": target_meta.get("target_id", ""),
        "dataset_path": str(args.dataset or ""),
        "model_artifact_path": str(artifact),
        "runtime_device_spec": runtime_diagnostics,
    }
```

Use `_async_run_metadata(args, task_name, target_meta, _safe_runtime_diagnostics(runtime))` for the existing normal `async_result.details["run"]` assignment. Pass a phase callback to the runner only when `args.debug` is true.

- [ ] **Step 6: Run focused and regression tests**

Run the command from Step 2 plus:

```bash
HF_DATASETS_CACHE=/tmp/mlhw-hf-cache \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_async_cli.py -q
```

Expected: all selected tests PASS and existing runner construction tests remain compatible because the new callback defaults to `None`.

- [ ] **Step 7: Commit Task 2**

```bash
git add framework/src/runtimes/onnx_rt.py \
  framework/src/core/async_inference/runner.py framework/src/main.py \
  framework/tests/test_async_onnx_cpu.py framework/tests/test_async_runner.py \
  framework/tests/test_async_cli.py
git commit -m "feat(framework): expose async lifecycle diagnostics"
```

---

### Task 3: Reserved run identity and failure artifacts

**Files:**
- Modify: `framework/src/main.py`
- Test: `framework/tests/test_async_cli.py`

**Interfaces:**
- Consumes: `RunArtifactReservation`, `save_async_details`, `save_result`, `_async_run_metadata`, `AsyncBenchmarkRunner.failure_phase`.
- Produces: one early `RUN_ID_RESERVED=<id>` line; debug phase/path messages; `_persist_async_failure(args, config, reservation, primary, phase, measurement_started, runtime_diagnostics, task_name, target_meta) -> bool` that writes a minimal invalid sidecar and CSV without replacing the original exception; final `RUN_ID=<id>` only when the failure CSV is committed.

- [ ] **Step 1: Write RED tests for identity, debug paths, and warmup failure persistence**

Update the successful async branch test to assert:

```python
lines = capsys.readouterr().out.splitlines()
assert lines.count("RUN_ID_RESERVED=async001") == 1
assert lines.count("RUN_ID=async001") == 1
assert lines.index("RUN_ID_RESERVED=async001") < lines.index("RUN_ID=async001")
```

Add a `--debug` test that asserts stderr contains `phase=reservation`, the selected results path, details path, and trace path, but does not contain request IDs.

Extend the existing `test_real_runner_warmup_failure_unloads_and_preserves_primary` with monkeypatched `save_async_details` and `save_result` collectors, then assert:

```python
assert raised.value is primary
assert saved_details["status"] == "invalid"
assert saved_details["run"]["measurement_started"] is False
assert saved_details["failure"] == {
    "phase": "warmup",
    "error_type": "RuntimeError",
    "error_message": "warmup failed",
}
assert saved_csv["async_run_status"] == "invalid"
assert saved_csv["async_invalid_reasons"] == "benchmark_exception"
assert captured.out.splitlines().count("RUN_ID=async001") == 1
```

Add failure-in-failure tests proving sidecar/CSV persistence errors are attached as secondary diagnostics and the original warmup exception object is re-raised. If CSV persistence fails, assert no final `RUN_ID=` line is printed and `RUN_ID_RESERVED=` remains available.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
HF_DATASETS_CACHE=/tmp/mlhw-hf-cache \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_async_cli.py \
  -k 'reserved or debug_paths or warmup_failure or failure_persistence' -q
```

Expected: FAIL because the reserved identity and structured failure persistence do not exist.

- [ ] **Step 3: Add coarse debug formatting and reserved identity**

Add:

```python
def _debug_lifecycle(args, phase, event, reservation=None, **fields):
    if not args.debug:
        return
    parts = [f"phase={phase}", f"event={event}"]
    if reservation is not None:
        parts.append(f"run_id={reservation.run_id}")
    parts.extend(f"{key}={value}" for key, value in fields.items())
    print("[AsyncDebug] " + " ".join(parts), file=sys.stderr, flush=True)
```

Immediately after successful reservation:

```python
print(f"RUN_ID_RESERVED={reservation.run_id}", flush=True)
_debug_lifecycle(
    args,
    "reservation",
    "complete",
    reservation,
    results_path=reservation.results_path,
    details_path=reservation.details_path,
    trace_path=reservation.trace_path,
)
```

Wrap trace start, runner run, trace close, sidecar save, CSV save, and runtime unload with start/complete/failed phase messages. Pass `lambda phase: _debug_lifecycle(args, phase, "start", reservation)` as the runner lifecycle callback.

- [ ] **Step 4: Implement safe exception extraction and minimal failure persistence**

Add helpers that use `_safe_persistence_error()` and built-in container access only:

```python
def _failure_diagnostic(primary, phase):
    diagnostic = _safe_persistence_error(phase, primary)
    return {
        "phase": phase,
        "error_type": diagnostic["error_type"],
        "error_message": diagnostic["error_message"],
    }


def _persist_async_failure(
    *, args, config, reservation, primary, phase, measurement_started,
    runtime_diagnostics, task_name, target_meta,
):
    run = _async_run_metadata(
        args,
        task_name,
        target_meta,
        runtime_diagnostics,
    )
    run["measurement_started"] = bool(measurement_started)
    details = {
        "status": RunStatus.INVALID.value,
        "invalid_reasons": ["benchmark_exception"],
        "warnings": [],
        "run": run,
        "failure": _failure_diagnostic(primary, phase),
        "counts": (
            {
                "submitted": 0,
                "accepted": 0,
                "completed": 0,
                "failed": 0,
                "rejected": 0,
                "outstanding": 0,
            }
            if not measurement_started
            else None
        ),
        "counts_available": not measurement_started,
    }
```

Complete `_persist_async_failure()` by saving `details` first and computing its relative reference. Then call `save_result()` with empty quality metrics, the reservation, async configuration fields, `async_run_status="invalid"`, and `async_invalid_reasons="benchmark_exception"`. Return `True` only if the reserved CSV save returns the same run ID. On a sidecar error call `_attach_secondary(primary, "failure_sidecar", exc)`; on a CSV error call `_attach_secondary(primary, "failure_csv", exc)`; print each through `_render_persistence_error()` without interpolating the raw exception.

- [ ] **Step 5: Integrate failure persistence without changing exception identity**

Track `reservation`, `runner`, `phase`, and a pre-cleanup runtime diagnostic snapshot in `execute_benchmark`. For any exception after reservation:

1. determine `phase` from `runner.failure_phase` when available;
2. snapshot runtime diagnostics before unload;
3. perform the existing bounded trace/runtime cleanup;
4. call `_persist_async_failure()`;
5. print `RUN_ID=<id>` exactly once only when failure CSV persistence succeeds;
6. use bare `raise` so the original exception object and traceback propagate.

Reservation failure has no reserved identity and keeps the current cleanup/re-raise behavior. Derive `measurement_started` from phases `measurement`, `finalization`, and `complete`; warmup is false. Do not fabricate completed request counts.

- [ ] **Step 6: Run focused tests and the full async CLI test module**

Run the command from Step 2, then:

```bash
HF_DATASETS_CACHE=/tmp/mlhw-hf-cache \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_async_cli.py -q
```

Expected: all tests PASS; existing `RUN_ID=` completion tests still observe exactly one final line.

- [ ] **Step 7: Commit Task 3**

```bash
git add framework/src/main.py framework/tests/test_async_cli.py
git commit -m "feat(framework): persist async startup failures"
```

---

### Task 4: Real CLI acceptance coverage and operator documentation

**Files:**
- Create: `framework/tests/test_async_cli_onnx_cpu.py`
- Modify: `docs/async-inference-queue.md`
- Modify: `framework/CHANGELOG.md`

**Interfaces:**
- Consumes: actual `framework/src/main.py`, ONNX/Pillow dependencies, `--results-path`, `--debug`, `--save-request-trace`.
- Produces: offline success and warmup-failure subprocess acceptance tests with no network/model download; documented debug workflow and Unreleased changelog entry.

- [ ] **Step 1: Write an actual subprocess acceptance test**

Create helpers in `framework/tests/test_async_cli_onnx_cpu.py` that:

- build a dynamic `["batch", 3, 224, 224] -> ["batch", 1000]` ONNX graph with `GlobalAveragePool`, `Flatten`, and `Gemm`;
- set class 0 bias above every other class;
- generate four RGB PNG files with Pillow under `dataset/val/`;
- write `val_labels.txt` lines as `sample_0.png 0` through `sample_3.png 0`;
- run `sys.executable src/main.py` with cwd=`framework`, `--model resnet50`, `--target cpu`, `--inference-mode async_queue`, `--scenario offline`, `--max-samples 4`, `--min-samples 4`, `--batch-size 2`, `--queue-capacity 4`, `--worker-count 1`, `--batch-timeout-ms 20`, `--warmup 0`, `--save-request-trace`, `--debug`, and a temp `--results-path`.

The success test must assert:

```python
assert completed.returncode == 0, completed.stderr
reserved = re.findall(r"^RUN_ID_RESERVED=(\w+)$", completed.stdout, re.M)
finished = re.findall(r"^RUN_ID=(\w+)$", completed.stdout, re.M)
assert len(reserved) == len(finished) == 1
assert reserved == finished
assert rows[0]["async_run_status"] == "valid"
assert details["counts"]["submitted"] == 4
assert details["counts"]["accepted"] == 4
assert details["counts"]["completed"] == 4
assert details["counts"]["outstanding"] == 0
assert details["batch_size"]["max"] == 2.0
assert details["run"]["runtime_device_spec"]["active_providers"] == [
    "CPUExecutionProvider"
]
assert len(trace_rows) == 4
assert {row["status"] for row in trace_rows} == {"completed"}
```

Run:

```bash
HF_DATASETS_CACHE=/tmp/mlhw-hf-cache \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_async_cli_onnx_cpu.py -q
```

Expected after Tasks 1-3: PASS if the separately tested pieces compose correctly through the real CLI. If it fails, keep the observed failing acceptance test as RED evidence, isolate the failing contract with the smallest focused test, and fix only that contract before rerunning this test.

- [ ] **Step 2: Add the warmup-failure subprocess case**

Create a second model whose input is fixed to `[1, 3, 1, 1]` while the resnet loader supplies `[1, 3, 224, 224]`. Run with `--warmup 1` and assert non-zero exit, identical reserved/final run IDs, one invalid CSV row, a details sidecar with `run.measurement_started is False`, `failure.phase == "warmup"`, and a safe ONNX Runtime shape error type/message. Assert the sidecar does not contain the substrings `traceback`, `input_tensor`, `output_tensor`, or `prompt`.

- [ ] **Step 3: Run real CLI smoke tests and focused regression tests**

Run:

```bash
HF_DATASETS_CACHE=/tmp/mlhw-hf-cache \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_async_cli_onnx_cpu.py \
  framework/tests/test_async_onnx_cpu.py \
  framework/tests/test_async_cli.py -q
```

Expected: all tests PASS with no network access and no writes under the tracked default results path.

- [ ] **Step 4: Document the debug workflow and changelog**

Add an `Unreleased` section at the top of `framework/CHANGELOG.md` with:

- the `async_queue` core/CLI/result artifact feature;
- `--results-path` output isolation;
- `RUN_ID_RESERVED`, coarse `--debug` lifecycle output, actual provider/run metadata, and startup failure artifacts;
- legacy CSV archive preservation;
- ONNX Runtime CPU real CLI tests.

Update `docs/async-inference-queue.md` with a copy-pastable command using `--results-path /tmp/mlhw-results/benchmark_results.csv --debug --save-request-trace`. Explain that `RUN_ID_RESERVED` identifies a started run, `RUN_ID` identifies a persisted terminal record, lifecycle logs are coarse and outside per-request measurement, and request postmortem data requires the JSONL trace.

Add `benchmark_exception` to the documented framework-owned invalid reasons. State that a fatal exception after measurement begins records `counts: null` and `counts_available: false` when no trustworthy terminal counter snapshot exists.

- [ ] **Step 5: Run the whole framework suite**

Run:

```bash
HF_DATASETS_CACHE=/tmp/mlhw-hf-cache \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests -q
```

Expected: PASS; only explicitly documented pre-existing skips/warnings may remain.

- [ ] **Step 6: Commit Task 4**

```bash
git add framework/tests/test_async_cli_onnx_cpu.py \
  docs/async-inference-queue.md framework/CHANGELOG.md
git commit -m "test(framework): exercise async CLI on ONNX CPU"
```

---

### Task 5: Manual acceptance runs and final evidence

**Files:**
- No product files unless verification exposes a defect; any defect follows a new RED test before its fix.

**Interfaces:**
- Consumes: completed Tasks 1-4.
- Produces: fresh manual Offline and Server-like execution evidence, clean worktree status, and a final implementation/test/changelog briefing.

- [ ] **Step 1: Generate fresh temporary smoke assets**

Reuse the test helper or its exact graph/data construction in a temporary directory. Do not use or overwrite `framework/results/benchmark_results.legacy.csv` or the default generated result path.

- [ ] **Step 2: Run Offline through the actual CLI**

Run the same command as the subprocess success test with a dedicated `/tmp` results path. Expected: exit 0; four completed requests; batch count 2 with max batch size 2; valid CSV/details/trace; actual `CPUExecutionProvider`; one reserved and one final run ID.

- [ ] **Step 3: Run Server-like through the actual CLI**

Use the same assets and:

```text
--scenario server_like --target-qps 20 --min-duration-sec 0
```

Expected: exit 0; four submitted/accepted/completed requests; zero failed/rejected/outstanding; valid linked artifacts. Batch size may remain 1 because 50 ms mean inter-arrival exceeds the 20 ms batch timeout; report this as expected scheduling behavior, not a regression.

- [ ] **Step 4: Inspect artifact invariants programmatically**

For both runs verify CSV run ID, details filename, trace filename, count equations, `counter_invariants` values, active provider, trace line count, and terminal statuses. Record the exact run IDs and commands in the final briefing.

- [ ] **Step 5: Run final verification from a clean test cache**

Run:

```bash
git diff --check
HF_DATASETS_CACHE=/tmp/mlhw-hf-cache-final \
  /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests -q
git status --short
```

Expected: no whitespace errors; full suite PASS; worktree contains no uncommitted generated result artifacts.

- [ ] **Step 6: Request final code review and address only evidence-backed findings**

Dispatch a whole-branch reviewer against the design's Section 42 and this plan. If a defect is found, reproduce it with a failing test, implement the smallest fix, rerun focused and full verification, and commit it separately.

- [ ] **Step 7: Deliver the briefing**

Report:

- how execution and debug paths were implemented;
- what structured data is available for normal, request-failure, and pre-measurement-failure cases;
- exact automated and manual commands/results;
- Offline and Server-like artifact/count evidence;
- the `framework/CHANGELOG.md` Unreleased entries;
- remaining device-validation scope and known thread/runtime limitations.
