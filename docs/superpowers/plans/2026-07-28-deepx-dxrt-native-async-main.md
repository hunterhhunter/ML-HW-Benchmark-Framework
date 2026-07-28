# DEEPX DX-RT Native Async Main Forward-Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add DX-RT 3.3 callback-based native async inference and async-safe warmup to latest main without changing other accelerator runtimes or the common async engine.

**Architecture:** Keep synchronous E2E execution in DeepXRuntime and introduce a runtime-owned DeepXNativeBackend implementing main's submit_async/shutdown contract. The backend exists only for async_queue; after creation, DeepXRuntime.warmup delegates to the same native callback path used for measurement.

**Tech Stack:** Python 3.12, NumPy, pytest, DEEPX dx_engine 3.3, NativeAsyncRuntimeExecutor.

## Global Constraints

- Base implementation on origin/main SHA ad9be09 or a later fast-forward.
- Change only DeepX runtime, DeepX target metadata, DeepX tests, and DeepX documentation.
- Do not change Hailo, Furiosa, RBLN, Mobilint, or common async runner/executor behavior.
- Preserve synchronous E2E run and warmup behavior.
- Async measurement and warmup submit exactly one sample per DX-RT job.
- Copy callback-owned outputs before returning from the SDK callback.
- Keep input payloads alive until physical callback completion.
- buffer_count defaults to 6 and accepts only integers 1 through 100.
- async_completion_timeout_sec defaults to 30 and must be finite and positive.
- Known pre-change baseline: 11 unrelated Furiosa timeout failures, 1963 passed, 13 skipped.

## File Map

- Create framework/tests/test_deepx_native_backend.py for fake DX-RT callback and lifecycle tests.
- Modify framework/src/runtimes/deepx_rt.py for DeepXNativeBackend and runtime ownership.
- Modify framework/src/core/targets.py only in the DeepX target block.
- Modify framework/tests/test_plugin_registry.py for DeepX target assertions.
- Modify docs/deepx-setup.md and framework/README.md for Jetson commands.

---

### Task 1: Reproduce Warmup Contamination and Establish Mode Ownership

**Files:**
- Create: framework/tests/test_deepx_native_backend.py
- Modify: framework/src/runtimes/deepx_rt.py
- Test: framework/tests/test_plugin_registry.py

**Interfaces:**
- Consumes: NativeAsyncOutcome and existing DeepX input/output normalization.
- Produces: create_native_backend(), native_async_max_batch_size(), and run_warmup_blocking().

- [ ] **Step 1: Add a fake DX-RT engine reproducing the Jetson callback**

The fake synchronous run invokes a registered callback with user_arg=None. The async method invokes it with the supplied token:

    class FakeInferenceEngine:
        def register_callback(self, callback):
            state.callback = callback

        def run(self, input_data):
            state.sync_calls += 1
            outputs = [np.asarray([[3.0, 4.0]], dtype=np.float32)]
            if state.callback is not None:
                state.callback(outputs, None)
            return outputs

        def run_async(self, input_data, user_arg=None, output_buffer=None):
            state.async_calls.append((input_data, user_arg))
            state.callback(
                [np.asarray([[7.0, 8.0]], dtype=np.float32)],
                user_arg,
            )
            return len(state.async_calls)

Expose one loaded native fixture for the tests in this file:

    def make_compiled_model(artifact):
        spec = Model_Spec(
            name="deepx-test",
            task=Task.IMAGE_CLASSIFICATION,
            input_shapes={"input": (1, 3, 4, 4)},
            input_dtype={"input": "float32"},
            output_shapes={"output": (1, 2)},
            model_paths={},
        )
        return CompiledModel(
            spec=spec,
            backend_name="deepx",
            artifact_path=artifact,
        )

    def install_fake_dx_engine(monkeypatch):
        state = FakeDXRTState()
        module = types.SimpleNamespace(
            InferenceEngine=FakeInferenceEngine,
            InferenceOption=FakeInferenceOption,
            __version__="3.3.2-test",
        )
        monkeypatch.setitem(sys.modules, "dx_engine", module)
        return state

    @pytest.fixture
    def loaded_native_runtime(monkeypatch, tmp_path):
        state = install_fake_dx_engine(monkeypatch)
        artifact = tmp_path / "model.dxnn"
        artifact.write_bytes(b"DXNN-test")
        runtime = DeepXRuntime(buffer_count=6)
        runtime.load(make_compiled_model(artifact))
        backend = runtime.create_native_backend()
        inputs = {"input": np.zeros((1, 3, 4, 4), dtype=np.float32)}
        return state, runtime, backend, inputs

- [ ] **Step 2: Write the failing warmup regression**

    def test_native_async_warmup_never_calls_sync_run(loaded_native_runtime):
        state, runtime, backend, inputs = loaded_native_runtime

        runtime.warmup(inputs, num_runs=2)
        outcomes = []
        backend.submit_async(inputs, outcomes.append)

        assert state.sync_calls == 0
        assert len(state.async_calls) == 3
        assert outcomes[0].error_type is None

- [ ] **Step 3: Run RED**

    /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests/test_deepx_native_backend.py::test_native_async_warmup_never_calls_sync_run -q

Expected: FAIL because main has no DeepX native backend and always uses synchronous warmup.

- [ ] **Step 4: Implement minimal ownership**

Add validated async configuration and mode state:

    self.buffer_count = self._parse_buffer_count(
        runtime_options.get("buffer_count", 6)
    )
    self.async_completion_timeout_sec = self._parse_positive_timeout(
        runtime_options.get("async_completion_timeout_sec", 30.0),
        "async_completion_timeout_sec",
    )
    self._execution_mode = None
    self._native_backend = None

Add:

    def native_async_max_batch_size(self) -> int:
        return 1

    def create_native_backend(self) -> DeepXNativeBackend:
        if self._engine is None:
            raise RuntimeError("DeepXRuntime is not loaded. Call load() first.")
        if self._execution_mode == "sync":
            raise RuntimeError("DeepX native async is unavailable in sync mode.")
        if self._native_backend is None:
            self._native_backend = DeepXNativeBackend(self)
        self._execution_mode = "native_async"
        return self._native_backend

DeepXRuntime.run rejects native_async mode and otherwise claims sync mode. warmup delegates to backend.run_warmup_blocking only when the backend exists.

- [ ] **Step 5: Run GREEN**

    /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests/test_deepx_native_backend.py::test_native_async_warmup_never_calls_sync_run framework/tests/test_plugin_registry.py::test_deepx_runtime_run_and_warmup_never_use_async_api -q

Expected: 2 passed. The existing plugin test proves E2E warmup stays synchronous.

- [ ] **Step 6: Commit**

    git add framework/src/runtimes/deepx_rt.py framework/tests/test_deepx_native_backend.py
    git commit -m "fix: isolate DeepX native async warmup"

---

### Task 2: Complete the DX-RT Callback Protocol

**Files:**
- Modify: framework/tests/test_deepx_native_backend.py
- Modify: framework/src/runtimes/deepx_rt.py

**Interfaces:**
- Consumes: Task 1 DeepXNativeBackend and runtime normalization helpers.
- Produces: submit_async(inputs, callback) returning a vendor job ID with exact token ownership.

- [ ] **Step 1: Add failing output-copy and out-of-order tests**

    def test_callback_outputs_are_copied_before_sdk_reuses_them(
        loaded_native_runtime,
    ):
        state, runtime, backend, inputs = loaded_native_runtime
        outcomes = []
        job_id = backend.submit_async(inputs, outcomes.append)
        sdk_output = np.asarray([[5.0, 6.0]], dtype=np.float32)
        state.complete(job_id, [sdk_output])
        sdk_output.fill(-1)
        np.testing.assert_array_equal(
            outcomes[0].outputs["output"], [[5.0, 6.0]]
        )

    def test_callbacks_match_out_of_order_tokens(loaded_native_runtime):
        state, runtime, backend, inputs = loaded_native_runtime
        completion_order = []
        first = backend.submit_async(
            {"input": np.full((1, 3, 4, 4), 1, dtype=np.float32)},
            lambda outcome: completion_order.append(1),
        )
        second = backend.submit_async(
            {"input": np.full((1, 3, 4, 4), 2, dtype=np.float32)},
            lambda outcome: completion_order.append(2),
        )
        state.complete(second, [np.asarray([[2.0]])])
        state.complete(first, [np.asarray([[1.0]])])
        assert completion_order == [2, 1]

- [ ] **Step 2: Run RED**

    /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests/test_deepx_native_backend.py -k "copied or out_of_order" -q

- [ ] **Step 3: Implement job/token state**

Publish this record before run_async so inline callbacks are valid:

    job = {
        "callback": callback,
        "input_payload": sdk_payload,
        "started_ns": time.perf_counter_ns(),
        "completion_started": False,
        "completion_finished": False,
        "submission_finished": False,
    }
    token = self._next_token
    self._next_token += 1
    self._jobs[token] = job
    vendor_job_id = submit(sdk_payload, user_arg=token)

_handle_completion accepts only an exact integer live token, normalizes outputs using runtime output names, recursively copies arrays/lists/tuples, publishes one NativeAsyncOutcome, and retires only after callback and submission finish.

- [ ] **Step 4: Add failing inline and named multi-input tests**

Add loaded_multi_input_runtime by passing input names ("left", "right") to
install_fake_dx_engine and by constructing a Model_Spec with matching input
shapes. Return state, runtime, backend, left, and right from the fixture.

    def test_inline_callback_before_job_id_return_completes_once(
        loaded_native_runtime,
    ):
        state, runtime, backend, inputs = loaded_native_runtime
        outcomes = []
        state.inline_outputs = [np.asarray([[9.0]], dtype=np.float32)]
        assert backend.submit_async(inputs, outcomes.append) == 1
        assert len(outcomes) == 1

    def test_named_multi_input_uses_named_async_api(loaded_multi_input_runtime):
        state, runtime, backend, left, right = loaded_multi_input_runtime
        outcomes = []
        backend.submit_async({"left": left, "right": right}, outcomes.append)
        assert state.calls[0][0] == "run_async_multi_input"
        assert list(state.calls[0][1]) == ["left", "right"]

- [ ] **Step 5: Run RED, implement routing, run GREEN**

    /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests/test_deepx_native_backend.py -k "inline or multi_input" -q

Use run_async for one input and run_async_multi_input for named multi-input, falling back to an ordered list only when the named API is unavailable.

- [ ] **Step 6: Add protocol failure tests**

Unknown or non-integer tokens must fail safely with DeepXAsyncProtocolError. Empty, batched, or wrong-count callback outputs must publish DeepXAsyncCompletionError. Synchronous submission exceptions must remove unpublished jobs without fabricating callback completion.

- [ ] **Step 7: Verify and commit**

    /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests/test_deepx_native_backend.py -q
    git add framework/src/runtimes/deepx_rt.py framework/tests/test_deepx_native_backend.py
    git commit -m "feat: bridge DeepX DX-RT native callbacks"

---

### Task 3: Enforce Warmup Timeout and Shutdown Safety

**Files:**
- Modify: framework/tests/test_deepx_native_backend.py
- Modify: framework/src/runtimes/deepx_rt.py

**Interfaces:**
- Consumes: Task 2 physical job records.
- Produces: bounded run_warmup_blocking(), shutdown(timeout), and safe runtime.unload().

- [ ] **Step 1: Add failing timeout and post-warmup tests**

    def test_warmup_timeout_keeps_physical_job_tracked(
        loaded_native_runtime,
    ):
        state, runtime, backend, inputs = loaded_native_runtime
        with pytest.raises(TimeoutError, match="warmup timed out"):
            backend.run_warmup_blocking(inputs, timeout=0.001)
        assert backend.shutdown(timeout=0.001) is False
        state.complete_pending()
        assert backend.shutdown(timeout=1.0) is True

    def test_measurement_succeeds_after_two_warmups(loaded_native_runtime):
        state, runtime, backend, inputs = loaded_native_runtime
        outcomes = []
        runtime.warmup(inputs, num_runs=2)
        backend.submit_async(inputs, outcomes.append)
        assert outcomes[0].error_type is None

- [ ] **Step 2: Run RED**

    /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests/test_deepx_native_backend.py -k "warmup_timeout or after_two_warmups" -q

- [ ] **Step 3: Implement blocking native warmup**

Submit through submit_async, wait on a local event for the bounded timeout, and never delete a physically pending job on logical timeout. Return copied outputs on success; raise a sanitized RuntimeError for callback failure.

- [ ] **Step 4: Add failing shutdown tests**

Assert live jobs prevent shutdown, completion permits shutdown, callback unregister happens before dispose, unload from a callback is rejected, and callbacks dispatched during unregister are drained.

- [ ] **Step 5: Implement bounded shutdown and unload**

Mark closing, reject new jobs, wait for jobs and active callbacks, call register_callback(None), wait again, and return False at deadline. DeepXRuntime.unload calls backend.shutdown with async_completion_timeout_sec and leaves the engine intact if shutdown is not proven.

- [ ] **Step 6: Verify and commit**

    /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests/test_deepx_native_backend.py -q
    git add framework/src/runtimes/deepx_rt.py framework/tests/test_deepx_native_backend.py
    git commit -m "fix: drain DeepX async warmup and callbacks"

---

### Task 4: Register DeepX Native Async on Main

**Files:**
- Modify: framework/src/core/targets.py
- Modify: framework/tests/test_plugin_registry.py
- Modify: framework/tests/test_deepx_native_backend.py

**Interfaces:**
- Consumes: main _build_async_runtime_executor and Task 1 native factory.
- Produces: DeepX native_async capability and full SDK-free lifecycle with warmup=2.

- [ ] **Step 1: Add failing registry assertions**

    assert "native_async" in target.capabilities
    assert target.runtime_options["buffer_count"] == 6
    assert target.runtime_options["async_completion_timeout_sec"] == 30.0

- [ ] **Step 2: Run RED**

    /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests/test_plugin_registry.py::test_builtin_registries_expose_deepx -q

- [ ] **Step 3: Update only the DeepX TargetSpec**

Add native_async and the two defaults to the existing DeepX target block. Do not edit another target or main.py.

- [ ] **Step 4: Add full SDK-free async lifecycle**

Build NativeAsyncRuntimeExecutor with max_inflight=4, InferenceEngine, and AsyncInferenceConfig with worker_count=4, max_batch_size=1, min_samples=8, max_samples=8. Run warmup_runs=2 and assert:

    assert result.status.value == "valid"
    assert result.metrics["async_submitted_requests"] == 8
    assert result.metrics["async_completed_requests"] == 8
    assert result.metrics["async_failed_requests"] == 0
    assert result.metrics["async_outstanding_requests"] == 0
    assert state.sync_calls == 0

- [ ] **Step 5: Run focused integration suites**

    /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests/test_deepx_native_backend.py framework/tests/test_plugin_registry.py framework/tests/test_deepx_dxnn_metadata.py framework/tests/test_native_async_runtime_executor.py -q

- [ ] **Step 6: Commit**

    git add framework/src/core/targets.py framework/tests/test_plugin_registry.py framework/tests/test_deepx_native_backend.py
    git commit -m "feat: register DeepX native async target"

---

### Task 5: Update Jetson Guide and Verify

**Files:**
- Modify: docs/deepx-setup.md
- Modify: framework/README.md

**Interfaces:**
- Consumes: final DeepX options and target behavior.
- Produces: copy-paste ResNet50 and YOLOv5M E2E/async validation commands.

- [ ] **Step 1: Document checkout and preflight**

Use branch agent/deepx-dxrt-native-async-main, DX-RT 3.3.2, /dev/dxrt0, run_async/register_callback checks, and the actual model/dataset paths.

- [ ] **Step 2: Document four commands**

Include E2E and async_queue for both models. Async uses warmup=2, batch-size=1, worker-count=4, buffer_count=6, result sidecars, and request traces.

- [ ] **Step 3: Document acceptance**

Require submitted=accepted=completed, failed=rejected=timed_out=outstanding=0, and async_run_status=valid. Explain that the unmatched-callback recovery message indicates a warmup ownership regression.

- [ ] **Step 4: Run final verification**

    /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests/test_deepx_native_backend.py framework/tests/test_plugin_registry.py framework/tests/test_deepx_dxnn_metadata.py framework/tests/test_native_async_runtime_executor.py -q
    HF_DATASETS_CACHE=/tmp/ml-hw-deepx-hf-cache /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests -q -k "not TestCudaRuntimePhysical"
    git diff --check
    git status --short

Focused tests must all pass. The full suite must add no failures beyond the recorded 11 Furiosa baseline failures.

- [ ] **Step 5: Commit docs**

    git add docs/deepx-setup.md framework/README.md
    git commit -m "docs: add DeepX native async device validation"

- [ ] **Step 6: Review branch scope and push**

    git diff --stat origin/main...HEAD
    git diff --name-only origin/main...HEAD
    git log --oneline origin/main..HEAD
    git push -u origin agent/deepx-dxrt-native-async-main

Expected scope: design/plan documents plus DeepX runtime, DeepX target block, DeepX tests, and DeepX operational docs only.
