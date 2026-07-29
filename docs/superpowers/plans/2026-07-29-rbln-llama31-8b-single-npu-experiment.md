# RBLN Llama 3.1 8B Single-NPU Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicitly opted-in, batch-one, 512-token Llama 3.1 8B experiment on one RBLN-CA22 without changing the behavior of Llama 3.2 3B or any `rbln-static` model.

**Architecture:** Extend the existing `allow_unsupported_single_npu` policy at the two existing contract boundaries: model preparation and runtime load. Keep model-specific limits so 3B retains its 1024-token ceiling while 8B receives a 512-token ceiling and a 15 GiB inventory prerequisite. Continue using the existing manifest, model-identity, lifecycle, monitoring, and async execution paths.

**Tech Stack:** Python 3.10, pytest, Optimum RBLN 0.11, vLLM RBLN 0.11, Rebellions `rbln-smi` inventory JSON.

## Global Constraints

- Do not modify `rbln-static`, its artifact contracts, model profiles, collectors, or async backend.
- Do not change Llama 3.2 3B one-NPU behavior: explicit opt-in, batch 1, maximum 1024 tokens.
- Do not change official eight-NPU defaults for Llama 3.2 3B or Llama 3.1 8B.
- Do not add quantization or claim official support.
- The one-NPU 8B contract is `num_devices=1`, `batch_size=1`, `max_seq_len<=512`, `max_num_seqs=1`, and decoder batch `[1]`.
- Record the experiment as `unsupported_single_npu_experiment`.
- Never overwrite an existing prepared-model directory.
- Preserve the original vLLM/RBLN engine exception if actual device allocation fails.

---

### Task 1: Preparation contract for one-NPU Llama 3.1 8B

**Files:**
- Modify: `framework/tests/test_prepare_rbln_vllm_model.py:75-112`
- Modify: `framework/tools/prepare_rbln_vllm_model.py:55-157`

**Interfaces:**
- Consumes: `resolve_compile_contract(...) -> dict[str, Any]` and the existing `--allow-unsupported-single-npu` CLI flag.
- Produces: a manifest contract classified as `unsupported_single_npu_experiment` for opted-in one-NPU Llama 3.1 8B at no more than 512 tokens.

- [ ] **Step 1: Replace the unconditional-rejection test with explicit policy tests**

```python
def test_single_npu_llama_3_1_8b_requires_opt_in():
    with pytest.raises(ValueError, match="allow-unsupported-single-npu"):
        prepare.resolve_compile_contract(
            model="llama-3.1-8b",
            model_id=None,
            num_devices=1,
            max_seq_len=512,
            block_size=512,
            batch_size=1,
            allow_unsupported_single_npu=False,
        )


def test_single_npu_llama_3_1_8b_accepts_opted_short_context():
    contract = prepare.resolve_compile_contract(
        model="llama-3.1-8b",
        model_id=None,
        num_devices=1,
        max_seq_len=512,
        block_size=512,
        batch_size=1,
        allow_unsupported_single_npu=True,
        decoder_batch_sizes="1",
    )

    assert contract == {
        "model": "llama-3.1-8b",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "num_devices": 1,
        "max_seq_len": 512,
        "block_size": 512,
        "batch_size": 1,
        "decoder_batch_sizes": [1],
        "support_classification": "unsupported_single_npu_experiment",
    }


def test_single_npu_llama_3_1_8b_rejects_context_over_512():
    with pytest.raises(ValueError, match="at most 512"):
        prepare.resolve_compile_contract(
            model="llama-3.1-8b",
            model_id=None,
            num_devices=1,
            max_seq_len=1024,
            block_size=512,
            batch_size=1,
            allow_unsupported_single_npu=True,
        )


def test_single_npu_llama_3_1_8b_rejects_batch_greater_than_one():
    with pytest.raises(ValueError, match="batch_size=1"):
        prepare.resolve_compile_contract(
            model="llama-3.1-8b",
            model_id=None,
            num_devices=1,
            max_seq_len=512,
            block_size=512,
            batch_size=2,
            decoder_batch_sizes="1,2",
            allow_unsupported_single_npu=True,
        )
```

- [ ] **Step 2: Run the new preparation tests and verify RED**

Run:

```bash
python -m pytest -q \
  framework/tests/test_prepare_rbln_vllm_model.py \
  -k 'single_npu_llama_3_1_8b'
```

Expected: the opt-in acceptance test fails because production still raises `cannot fit`; the missing-opt-in and 512-boundary expectations also do not yet match the new policy.

- [ ] **Step 3: Implement model-specific experimental limits**

Replace the unconditional 8B rejection and 3B-only experiment branch with a shared single-NPU experiment branch. Keep the model-specific ceiling literal and do not change official defaults:

```python
    if selected_num_devices == 8:
        support_classification = "official"
    elif selected_num_devices == 1 and model in {
        "llama-3.2-3b",
        "llama-3.1-8b",
    }:
        if not allow_unsupported_single_npu:
            raise ValueError(
                f"the one-NPU {model} experiment requires "
                "--allow-unsupported-single-npu"
            )
        support_classification = "unsupported_single_npu_experiment"
    else:
        raise ValueError(
            f"{model} is officially supported with 8 NPU chips"
        )

    if (
        support_classification == "unsupported_single_npu_experiment"
        and selected_batch_size != 1
    ):
        raise ValueError(
            f"the one-NPU {model} experiment requires batch_size=1"
        )
```

Use `512` as the default experimental sequence length. Validate the model-specific maximum after shape divisibility checks:

```python
    if support_classification == "unsupported_single_npu_experiment":
        single_npu_max_seq_len = (
            512 if model == "llama-3.1-8b" else 1024
        )
        if selected_max_seq_len > single_npu_max_seq_len:
            raise ValueError(
                f"the one-NPU {model} experiment allows at most "
                f"{single_npu_max_seq_len} tokens"
            )
```

Update the CLI help so the existing flag describes unsupported one-NPU Llama experiments rather than claiming it never enables 8B.

- [ ] **Step 4: Run the preparation suite and verify GREEN**

Run:

```bash
python -m pytest -q framework/tests/test_prepare_rbln_vllm_model.py
```

Expected: all tests pass, including unchanged official-model and Llama 3.2 3B tests.

- [ ] **Step 5: Commit the preparation contract**

```bash
git add \
  framework/tools/prepare_rbln_vllm_model.py \
  framework/tests/test_prepare_rbln_vllm_model.py
git commit -m "feat: allow opted single-NPU Llama 3.1 8B preparation"
```

---

### Task 2: Runtime load contract and memory prerequisite

**Files:**
- Modify: `framework/tests/test_rbln_vllm_runtime.py:295-330`
- Modify: `framework/tests/test_rbln_vllm_runtime.py:390-445`
- Modify: `framework/src/runtimes/rbln_vllm_rt.py:704-742`
- Modify: `framework/src/runtimes/rbln_vllm_rt.py:821-858`

**Interfaces:**
- Consumes: prepared-directory manifest fields, resolved model identity, device inventory, and `allow_unsupported_single_npu`.
- Produces: a loaded runtime classified as `unsupported_single_npu_experiment`, or a pre-engine validation error for missing opt-in, invalid sequence/batch, unreadable memory, or less than 15 GiB total memory.

- [ ] **Step 1: Write runtime acceptance and guard tests**

Replace the unconditional 8B rejection test and add boundary coverage:

```python
def test_load_rejects_unopted_single_npu_llama_3_1_8b_before_sdk_import(
    monkeypatch, tmp_path
):
    monkeypatch.setitem(sys.modules, "vllm", None)
    model_dir = _prepared_model(
        tmp_path,
        "llama-3.1-8b",
        manifest_num_devices=1,
    )
    runtime = _runtime(allow_single=False)

    with pytest.raises(ValueError, match="allow_unsupported_single_npu"):
        runtime.load(_compiled(model_dir, "llama-3.1-8b"))


def test_load_accepts_explicit_single_npu_llama_3_1_8b_experiment(tmp_path):
    model_dir = _prepared_model(
        tmp_path,
        "llama-3.1-8b",
        manifest_num_devices=1,
    )
    runtime = _runtime(allow_single=True)

    runtime.load(_compiled(model_dir, "llama-3.1-8b"))

    spec = runtime.get_device_spec()
    assert spec["model_kind"] == "llama-3.1-8b"
    assert spec["support_classification"] == (
        "unsupported_single_npu_experiment"
    )


def test_load_rejects_single_npu_llama_3_1_8b_context_over_512(tmp_path):
    model_dir = _prepared_model(
        tmp_path,
        "llama-3.1-8b",
        manifest_num_devices=1,
        manifest_max_seq_len=1024,
        manifest_block_size=512,
    )
    runtime = _runtime(
        allow_single=True,
        max_model_len=1024,
    )

    with pytest.raises(ValueError, match="max_model_len.*512"):
        runtime.load(_compiled(model_dir, "llama-3.1-8b"))


def test_load_rejects_single_npu_llama_3_1_8b_below_15_gib(tmp_path):
    model_dir = _prepared_model(
        tmp_path,
        "llama-3.1-8b",
        manifest_num_devices=1,
    )
    runtime = _runtime(
        allow_single=True,
        inventory_provider=lambda: _inventory(1, memory_bytes=14 * GIB),
    )

    with pytest.raises(ValueError, match="at least 15 GiB"):
        runtime.load(_compiled(model_dir, "llama-3.1-8b"))


def test_load_rejects_single_npu_llama_3_1_8b_batch_greater_than_one(
    tmp_path,
):
    model_dir = _prepared_model(tmp_path, "llama-3.1-8b")
    runtime = _runtime(
        allow_single=True,
        max_num_seqs=2,
    )

    with pytest.raises(ValueError, match="max_num_seqs.*exactly 1"):
        runtime.load(_compiled(model_dir, "llama-3.1-8b"))
```

- [ ] **Step 2: Run the new runtime tests and verify RED**

Run:

```bash
python -m pytest -q \
  framework/tests/test_rbln_vllm_runtime.py \
  -k 'single_npu_llama_3_1_8b'
```

Expected: the acceptance test fails with the current `cannot fit` rejection, and boundary tests fail because the 8B-specific experimental policy does not exist.

- [ ] **Step 3: Implement model-specific runtime policy**

In `_validate_model_device_contract`, retain the official branch and allow both known Llama models through the existing opt-in. Use a model-specific context limit:

```python
        if model_kind in {"llama-3.1-8b", "llama-3.2-3b"}:
            if self.num_devices == 8:
                return "official"
            if self.num_devices == 1 and self.allow_unsupported_single_npu:
                if self.max_num_seqs != 1:
                    raise ValueError(
                        f"single-NPU {model_kind} requires max_num_seqs "
                        "exactly 1"
                    )
                max_single_npu_context = (
                    512 if model_kind == "llama-3.1-8b" else 1024
                )
                if (
                    resolved_max_model_len is None
                    or resolved_max_model_len > max_single_npu_context
                ):
                    raise ValueError(
                        f"single-NPU {model_kind} max_model_len must be "
                        "explicitly resolved and at most "
                        f"{max_single_npu_context}"
                    )
                return "unsupported_single_npu_experiment"
            raise ValueError(
                f"{model_kind} is officially supported with 8 NPUs; set "
                "allow_unsupported_single_npu=true for the bounded "
                "one-NPU experiment"
            )
```

Generalize the existing single-NPU memory inventory check without changing its 3B threshold:

```python
        if model_kind in {"llama-3.2-3b", "llama-3.1-8b"} and self.num_devices == 1:
            minimum_gib = 15 if model_kind == "llama-3.1-8b" else 8
            total_bytes = _memory_bytes(
                selected[0].get("memory", {}).get("total")
                if isinstance(selected[0].get("memory"), Mapping)
                else None
            )
            if total_bytes is None:
                raise ValueError(
                    f"single-NPU {model_kind} requires readable device "
                    "memory.total"
                )
            if total_bytes < minimum_gib * _GIB:
                raise ValueError(
                    f"single-NPU {model_kind} requires at least "
                    f"{minimum_gib} GiB for weights and runtime reserve"
                )
```

- [ ] **Step 4: Run runtime and integration suites and verify GREEN**

Run:

```bash
python -m pytest -q \
  framework/tests/test_rbln_vllm_runtime.py \
  framework/tests/test_main_paths.py
```

Expected: all tests pass, including existing 3B sync, async, warmup, identity, and cleanup tests.

- [ ] **Step 5: Commit the runtime contract**

```bash
git add \
  framework/src/runtimes/rbln_vllm_rt.py \
  framework/tests/test_rbln_vllm_runtime.py
git commit -m "feat: allow opted single-NPU Llama 3.1 8B runtime"
```

---

### Task 3: Operator runbook and regression verification

**Files:**
- Modify: `framework/docs/rbln-vllm-setup.md:12-34`
- Modify: `framework/docs/rbln-vllm-setup.md:190-290`
- Modify: `framework/docs/rbln-vllm-setup.md:290-430`
- Modify: `framework/README.md:29-45`

**Interfaces:**
- Consumes: the new preparation and runtime contracts.
- Produces: exact non-overwriting server commands for compile, artifact inspection, one-request sync, four-request async, and context verification.

- [ ] **Step 1: Update support classification without changing other model rows**

Change only the Llama 3.1 8B one-NPU row from `실행 차단` to a non-official 512-token experiment. State that default Optimum precision is used and actual runtime capacity remains unverified until the hardware gate passes. Keep the eight-NPU row unchanged.

- [ ] **Step 2: Add exact one-NPU 8B preparation command**

Document this separate output directory and command:

```bash
export RBLN_LLAMA31_DIR="$HOME/rebelion/rbln-model-zoo/custom/framework-contracts/llama-3.1-8b-npu1-seq512"

test ! -e "$RBLN_LLAMA31_DIR"

"$RBLN_VLLM_PY" tools/prepare_rbln_vllm_model.py \
  --model llama-3.1-8b \
  --output-dir "$RBLN_LLAMA31_DIR" \
  --num-devices 1 \
  --max-seq-len 512 \
  --block-size 512 \
  --batch-size 1 \
  --decoder-batch-sizes 1 \
  --allow-unsupported-single-npu
```

Precede it with `df -BG`, `free -h`, and `rbln-smi -j` checks, requiring at least 30 GiB free disk and no active contexts. Explain that partial output is retained on failure.

- [ ] **Step 3: Add sync and async acceptance commands**

The sync command uses one sample and one generated token. The async command uses four samples, one worker, one queue slot, and the runtime option workaround `decoder_batch_sizes=1,` so the shared CLI coercer retains a string:

```bash
"$RBLN_RUN_PY" -m src.main \
  --model llama-3.1-8b \
  --target rbln-vllm \
  --model-path "$RBLN_LLAMA31_DIR" \
  --tokenizer-path "$RBLN_LLAMA31_DIR" \
  --dataset "$RBLN_DATASET" \
  --inference-mode async_queue \
  --scenario offline \
  --batch-size 1 \
  --max-model-len 512 \
  --max-new-tokens 1 \
  --worker-count 1 \
  --queue-capacity 1 \
  --min-samples 4 \
  --max-samples 4 \
  --warmup 1 \
  --flush-timeout-sec 600 \
  --runtime-option block_size=512 \
  --runtime-option num_devices=1 \
  --runtime-option max_num_seqs=1 \
  --runtime-option decoder_batch_sizes=1, \
  --runtime-option allow_unsupported_single_npu=true \
  --save-request-trace \
  --monitor \
  --debug \
  --results-path results/rbln-llama31-8b-npu1-async-smoke.csv
```

Document exact valid-run counters and the `compiled_but_single_npu_runtime_capacity_failed` outcome for allocation failure.

- [ ] **Step 4: Run focused RBLN and CLI regression suites**

Run:

```bash
python -m pytest -q \
  framework/tests/test_prepare_rbln_vllm_model.py \
  framework/tests/test_rbln_vllm_runtime.py \
  framework/tests/test_rbln_runtime.py \
  framework/tests/test_rbln_native_backend.py \
  framework/tests/test_rbln_collector.py \
  framework/tests/test_plugin_registry.py \
  framework/tests/test_main_paths.py
```

Expected: zero failures. This is the explicit regression gate for Llama 3.2 3B and static RBLN paths used by ResNet50, YOLOv5m, PatchTST, and BERT.

- [ ] **Step 5: Run the complete framework test suite**

Run:

```bash
python -m pytest -q framework/tests
```

Expected: zero failures; environment-dependent skips are reported separately rather than counted as passes.

- [ ] **Step 6: Commit documentation and verified runbook**

```bash
git add framework/docs/rbln-vllm-setup.md framework/README.md
git commit -m "docs: add single-NPU Llama 3.1 8B experiment runbook"
```

- [ ] **Step 7: Review the final branch diff**

Run:

```bash
git status --short
git diff --check HEAD~3..HEAD
git diff --stat HEAD~3..HEAD
git log --oneline -4
```

Expected: only the preparation tool/tests, RBLN vLLM runtime/tests, Llama documentation, design, and plan files changed. No `rbln-static` production file or model profile appears in the diff.
