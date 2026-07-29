# Mobilint ARIES Transformer and LLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run BERT SST-2, BERT SQuAD, PatchTST ETTh1, Llama 3.1 8B, and Llama 3.2 3B through the existing ARIES benchmark paths without adding compiler integration.

**Architecture:** Generalize the existing Mobilint vision contract into an ordered tensor contract for static MXQ models, select SDK native async per loaded artifact rather than solely per target, and extend the existing Model Zoo LLM runtime to validate standard/Batch16/Batch32 capacities while continuing to execute blocking `generate()` through the framework async queue.

**Tech Stack:** Python 3.12, NumPy, pytest, Mobilint qb Runtime/Model Zoo v1.3, Hugging Face Transformers and Hub.

## Global Constraints

- Work on branch `feat/mobilint-aries-transformers` in `/tmp/ml-hw-benchmark-mobilint-transformers`.
- Preserve all ResNet50, YOLOv5m, ARIES/REGULUS monitoring, and result-schema behavior.
- Do not add compiler integration or claim hardware verification from fake-SDK tests.
- Do not route BERT, PatchTST, or Llama through qb Runtime native async; the v1.3 SDK excludes those model classes.
- `Batch16` and `Batch32` are maximum artifact capacities, not required actual batch sizes.
- Use failing behavior tests before each production change.

### Task 1: Generalize the static Mobilint tensor contract

**Files:**
- Create: `framework/src/core/mobilint_tensor_contracts.py`
- Create: `framework/tests/test_mobilint_tensor_contracts.py`
- Modify: `framework/src/runtimes/mobilint_rt.py`
- Modify: `framework/tests/test_mobilint_runtime.py`

- [x] Add failing tests that derive ordered BERT SST-2, BERT SQuAD, and PatchTST contracts from literal ModelSpecs.
- [x] Add failing runtime tests for multi-input dtype/shape validation, leading singleton SDK shapes, output count mapping, and native-async disabled contracts.
- [x] Implement immutable tensor-contract DTOs and ModelSpec derivation.
- [x] Normalize legacy vision contracts and new tensor contracts into one internal runtime representation.
- [x] Preserve legacy diagnostics while adding artifact profile and ordered tensor diagnostics.
- [x] Run `test_mobilint_tensor_contracts.py`, `test_mobilint_runtime.py`, and `test_mobilint_native_backend.py` to green.

### Task 2: Select native async per artifact

**Files:**
- Modify: `framework/tests/test_async_cli.py`
- Modify: `framework/tests/test_main_paths.py`
- Modify: `framework/src/main.py`

- [x] Add failing tests proving a static Transformer Mobilint runtime returns intentional blocking fallback while a vision runtime still selects the native executor.
- [x] Inject the ModelSpec-derived tensor contract for Mobilint non-vision static tasks.
- [x] Enable the qbruntime async pipeline only when that contract permits native async.
- [x] Treat `native_async_max_batch_size() is None` as blocking fallback and retain strict validation for malformed non-null declarations.
- [x] Run async CLI and main-path tests to green.

### Task 3: Add official Llama acquisition and capacity validation

**Files:**
- Create: `framework/models/prepare_mobilint_llm.py`
- Create: `framework/tests/test_prepare_mobilint_llm.py`
- Modify: `framework/src/runtimes/mobilint_llm_rt.py`
- Modify: `framework/tests/test_mobilint_llm_runtime.py`

- [x] Add failing downloader tests for the two model families and standard/Batch16/Batch32 repository mappings.
- [x] Add failing runtime tests for missing/malformed `config.json`, positive `max_batch_size`, and safe diagnostics.
- [x] Implement deterministic full-repository `snapshot_download()` into `framework/models/mobilint/<model>/<variant>`.
- [x] Read and validate artifact capacity before device/model acquisition.
- [x] Expose dynamic and generation batch capabilities only when capacity exceeds one; keep concurrent model calls at one.
- [x] Run downloader and LLM runtime tests to green.

### Task 4: Support grouped Llama generation up to artifact capacity

**Files:**
- Modify: `framework/tests/test_mobilint_llm_runtime.py`
- Modify: `framework/src/runtimes/mobilint_llm_rt.py`

- [x] Add failing tests for actual batches 1, less than capacity, equal to capacity, and above capacity.
- [x] Add failing tests for padded prompts, per-row continuation lengths, EOS truncation, and aggregate grouped streamer events.
- [x] Implement single-request prompt compaction and rectangular grouped generation without confusing capacity with actual batch.
- [x] Return 1-D IDs for batch one and 2-D padded IDs plus `generated_lengths` for grouped batches.
- [x] Run LLM runtime, executor, evaluator, and async-generation metric tests to green.

### Task 5: Add MXQ inspection and operator documentation

**Files:**
- Create: `framework/tools/inspect_mobilint_mxq.py`
- Create: `framework/tests/test_inspect_mobilint_mxq.py`
- Create: `docs/mobilint-aries-transformers.md`
- Modify: `framework/src/runtimes/README.md`

- [x] Add failing tool tests for JSON metadata output, SDK absence, and guaranteed model disposal.
- [x] Implement an SDK-lazy inspection CLI for variants, input dtypes/shapes, and output shapes.
- [x] Document SDK checks, Llama downloads, external MXQ placement, artifact inspection, dataset preparation, and E2E/async commands for every requested model.
- [x] Clearly label hardware-only verification steps and the lack of public task-specific BERT/PatchTST MXQs.

### Task 6: Regression verification and handoff

- [x] Run focused Mobilint, BERT, PatchTST, Llama, async CLI, main-path, and registry tests.
- [x] Run the complete test suite in the temporary uv environment.
- [x] Review the diff for secret/model payload leakage and unrelated changes.
- [ ] Commit the coherent implementation, push `feat/mobilint-aries-transformers`, and provide exact server fetch/switch/smoke commands.
