# Rebellions vLLM Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an in-process Rebellions vLLM target for Llama 3.2 3B and Llama 3.1 8B, with synchronous E2E, native async streaming, hardware preflight, and an executable server runbook.

**Architecture:** Keep the framework async queue as the single scheduler and adapt only vLLM RBLN engine calls. A lazy `RblnVllmRuntime` selects exactly one sync or async engine per load, scopes the RBLN device-count environment at construction, validates prepared model directories and hardware support, and emits the framework's existing generation contracts.

**Tech Stack:** Python 3.10+, NumPy, pytest, vLLM 0.22.0, vllm-rbln 0.11.0, rebel-compiler 0.11.0.post1, optimum-rbln 0.11.0.post1.

## Global Constraints

- Work on branch `feat/rbln-vllm` in `/tmp/ml-hw-benchmark-rbln-vllm`.
- Use a local precompiled Optimum RBLN model directory; compile-on-load is out of scope.
- Keep `tensor_parallel_size=1`; use `num_devices` through `VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK`.
- Require an explicit opt-in for unsupported single-NPU Llama 3.2 3B.
- Reject single-NPU Llama 3.1 8B before engine initialization.
- Do not add a second request queue, HTTP server, or result schema.
- Add tests before each production change and observe the intended failure.

---

### Task 1: Register the target and CLI artifact contract

**Files:**
- Modify: `framework/tests/test_plugin_registry.py`
- Modify: `framework/tests/test_main_paths.py`
- Modify: `framework/src/core/targets.py`
- Modify: `framework/src/runtimes/__init__.py`
- Modify: `framework/src/main.py`

**Interfaces:**
- Consumes: `--target rbln-vllm`, local `--model-path`, generation profile.
- Produces: runtime `rbln_vllm`, artifact format `rbln_llm_dir`, tokenizer default, and RBLN runtime options.

- [ ] Add RED tests for target metadata, runtime lazy registration, parser backend choice, generation-only validation, local prepared-directory validation, and auto-prepare bypass.
- [ ] Run the focused registry/main tests and confirm failures are missing target/runtime behavior.
- [ ] Register the target/runtime and implement the `rbln_llm_dir` CLI helpers without weakening existing `hf_model` targets.
- [ ] Forward `--max-model-len` to `rbln_vllm` and merge CLI runtime options through existing precedence rules.
- [ ] Run the focused tests to GREEN.

### Task 2: Implement sync runtime and hardware preflight

**Files:**
- Create: `framework/tests/test_rbln_vllm_runtime.py`
- Create: `framework/src/runtimes/rbln_vllm_rt.py`

**Interfaces:**
- Consumes: `CompiledModel`, prepared RBLN directory, runtime options, NumPy prompt tensors.
- Produces: `GenerationResult`, compatibility/device metadata, and deterministic preflight errors.

- [ ] Add RED tests for option validation, required files, config parsing, official device counts, single-NPU opt-in, 8B memory rejection, device inventory, and scoped environment restoration.
- [ ] Implement pure validation helpers and lazy SDK imports.
- [ ] Add RED tests for sync engine kwargs, left/right padding, batch bounds, stop IDs, timing metrics, and idempotent unload.
- [ ] Implement deferred `LLM` construction, sync generation, normalization, timing extraction, and engine shutdown.
- [ ] Run the full runtime test file to GREEN.

### Task 3: Implement native async streaming

**Files:**
- Modify: `framework/tests/test_rbln_vllm_runtime.py`
- Modify: `framework/tests/test_async_cli.py`
- Modify: `framework/src/runtimes/rbln_vllm_rt.py`

**Interfaces:**
- Consumes: one request per `submit_async`, vLLM async generator, common callback executor.
- Produces: exactly one `NativeAsyncOutcome` and cumulative token events.

- [ ] Add RED tests for async engine construction, multiple concurrent requests, cumulative token events, output normalization, abort-on-error, callback exactly once, shutdown timeout, and sync/async mode exclusion.
- [ ] Implement the owned event-loop thread and `AsyncLLMEngine.from_engine_args` path.
- [ ] Implement request submission/consumption, abort, retirement, engine shutdown, and thread join.
- [ ] Add CLI executor tests proving the common native-async path is selected without a second scheduler.
- [ ] Run runtime and async CLI tests to GREEN.

### Task 4: Add manual model preparation and server runbook

**Files:**
- Create: `framework/tests/test_prepare_rbln_vllm_model.py`
- Create: `framework/tools/prepare_rbln_vllm_model.py`
- Create: `framework/docs/rbln-vllm-setup.md`

**Interfaces:**
- Consumes: model alias/ID, output directory, context/block size, device count, single-NPU opt-in.
- Produces: saved Optimum RBLN directory plus reproducibility manifest and copy-paste run commands.

- [ ] Add RED tests for alias resolution, compile parameter validation, 8B single-NPU rejection, 3B opt-in, and deterministic manifest hashing.
- [ ] Implement the lazy Optimum compiler utility using `RBLNLlamaForCausalLM.from_pretrained(..., export=True)` and `save_pretrained()`.
- [ ] Document uv environment creation, exact SDK packages, portal/Hugging Face authentication, device checks, official eight-NPU preparation, experimental one-NPU 3B preparation, E2E/async commands, metrics, and context cleanup.
- [ ] Run utility tests and `--help` without vendor packages.

### Task 5: Regression and release verification

**Files:**
- Modify: `README.md` only if its target matrix has a natural RBLN vLLM entry.

**Interfaces:**
- Consumes: completed target/runtime/tool/docs.
- Produces: verified branch ready for hardware validation.

- [ ] Run formatting/static syntax checks for all changed Python files.
- [ ] Run focused RBLN vLLM, registry, main, async, RBLN static runtime, collector, and plugin tests.
- [ ] Run the broad framework test suite with required test dependencies and record any environment-only exclusions.
- [ ] Inspect `git diff --check`, `git status`, and the complete diff for unrelated changes.
- [ ] Provide server acceptance commands for Llama 3.2 3B one-NPU experiment and official eight-NPU runs; mark Llama 3.1 8B one-NPU as intentionally blocked.
