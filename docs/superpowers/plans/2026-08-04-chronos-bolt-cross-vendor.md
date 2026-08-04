# Chronos-Bolt Cross-Vendor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compile and device-verify the same split `amazon/chronos-bolt-tiny` Transformer core on Rebellions CA22, Furiosa RNGD, and Mobilint ARIES without accepting CPU fallback or unsupported partitions as success.

**Architecture:** A vendor-independent host adapter retains Chronos-Bolt normalization, missing-value semantics, static patching, learned input embedding, decoder-start embedding, and inverse normalization on CPU. A tensor-only `ChronosBoltTransformerCore` receives three fixed FP32 inputs and returns normalized quantiles `(1, 9, 64)`; vendor tools compile that unchanged Torch module and record a uniform artifact/result manifest.

**Tech Stack:** Python 3.10–3.12, PyTorch 2.x, `chronos-forecasting==2.3.1`, Transformers 4.57–5.x, NumPy, pytest, Rebellions `rebel-compiler==0.11.0`, Furiosa Torch 2026.3.0, Mobilint `qbcompiler` for `aries-rb`.

## Global Constraints

- The exact first checkpoint is `amazon/chronos-bolt-tiny`; write its resolved revision and file SHA-256 to every run manifest.
- The external contract is FP32 context `(1, 512)` to FP32 quantiles `(1, 9, 64)`, with quantiles `0.1` through `0.9`, batch one, patch size/stride 16, and prediction length 64. The actual `amazon/chronos-bolt-tiny` checkpoint has maximum context 2048, but its fixed 512-point benchmark input is valid.
- Tiny uses a learned REG token. Therefore the core ABI has 32 data-patch positions plus REG: `input_embeds=(1,33,256)` and `attention_mask=(1,33)`. The learned REG embedding is generated on the CPU host adapter; the Transformer core receives it as a normal tensor input.
- CPU execution owns normalization, NaN/mask handling, patch construction, `input_patch_embedding`, decoder-start embedding, and inverse normalization; those operations are not silently removed.
- The NPU core accepts `input_embeds`, `attention_mask`, and `decoder_input_embeds`, all FP32 and fixed-shape; it exposes no `ModelOutput`, Python dict, or dynamic input length.
- Compile success requires process exit zero, a nonempty artifact, contract inspection, no CPU/eager fallback or unsupported partition, artifact hash, and compiler version evidence.
- Device verification additionally requires artifact load, first inference, finite `(1, 9, 64)` output, CPU-core parity evidence, and device-utilization evidence.
- Artifact binaries, model weights, datasets, raw compiler logs, wheel files, tokens, and credentials remain outside Git.
- Vendor SDK imports are lazy so the ordinary Python test suite runs without Rebellions, Furiosa, or Mobilint software.
- Do not modify the existing PatchTST workflow, model profiles, generic runtime factories, or the main benchmark CLI in this implementation. Framework E2E integration follows only after all core compile gates are evidence-backed.

---

## File Map

- Create: `framework/src/chronos_bolt/__init__.py` — public split-core API only.
- Create: `framework/src/chronos_bolt/contracts.py` — immutable fixed ABI, validation, and JSON serialization.
- Create: `framework/src/chronos_bolt/host_adapter.py` — CPU preprocessing/postprocessing and deterministic context preparation.
- Create: `framework/src/chronos_bolt/core.py` — tensor-only Transformer-core wrapper around loaded Chronos modules.
- Create: `framework/src/chronos_bolt/evidence.py` — artifact hashes, package versions, safe status/result JSON.
- Create: `framework/tools/chronos_bolt_compile.py` — common reference, describe, and compiler dispatch CLI.
- Create: `framework/tools/chronos_bolt_vendors/__init__.py` — lazy vendor compiler package.
- Create: `framework/tools/chronos_bolt_vendors/rbln.py` — CA22 offline `.rbln` compile and inspect.
- Create: `framework/tools/chronos_bolt_vendors/furiosa.py` — strict RNGD first-call compilation evidence.
- Create: `framework/tools/chronos_bolt_vendors/mobilint.py` — `aries-rb` MBLT/MXQ compile and inspection, imported only when `qbcompiler` is installed.
- Create: `framework/tests/test_chronos_bolt_contracts.py` — ABI and result-status tests.
- Create: `framework/tests/test_chronos_bolt_host_adapter.py` — static patches, masks, and inverse-normalization tests.
- Create: `framework/tests/test_chronos_bolt_core.py` — fake-component tensor-path tests.
- Create: `framework/tests/test_chronos_bolt_compile_cli.py` — lazy imports, parser, overwrite, and manifest tests.
- Create: `framework/tests/test_chronos_bolt_rbln.py` — fake Rebellions compile/inspect contract tests.
- Create: `framework/tests/test_chronos_bolt_furiosa.py` — strict compiler configuration tests.
- Create: `framework/tests/test_chronos_bolt_mobilint.py` — fake `qbcompiler` target/calibration/offload tests.
- Create: `framework/docs/chronos-bolt-cross-vendor.md` — operator runbook and exact acceptance criteria.

## Task 1: Fixed Contract and Result Evidence

**Files:**
- Create: `framework/src/chronos_bolt/contracts.py`
- Create: `framework/src/chronos_bolt/evidence.py`
- Create: `framework/tests/test_chronos_bolt_contracts.py`

**Interfaces:**
- Produces `ChronosBoltContract`, `TensorContract`, `CompileStatus`, `validate_core_inputs()`, and `write_result()`.
- Every vendor consumes `ChronosBoltContract.tensors` in the declared order: `input_embeds`, `attention_mask`, `decoder_input_embeds`.

- [ ] **Step 1: Write failing contract tests**

```python
from chronos_bolt.contracts import ChronosBoltContract, CompileStatus


def test_tiny_contract_has_fixed_external_and_core_abi():
    contract = ChronosBoltContract.tiny(d_model=128)
    assert contract.external_input.shape == (1, 512)
    assert contract.external_output.shape == (1, 9, 64)
    assert [tensor.name for tensor in contract.core_inputs] == [
        "input_embeds", "attention_mask", "decoder_input_embeds",
    ]
    assert [tensor.shape for tensor in contract.core_inputs] == [
        (1, 33, 128), (1, 33), (1, 1, 128),
    ]
    assert contract.quantile_levels == tuple(index / 10 for index in range(1, 10))


def test_terminal_success_status_requires_device_evidence():
    assert CompileStatus.COMPILED.value == "compiled"
    assert CompileStatus.DEVICE_VERIFIED.value == "device_verified"
```

- [ ] **Step 2: Verify RED**

Run from `framework/`:

```bash
python -m pytest tests/test_chronos_bolt_contracts.py -q
```

Expected: collection fails because `chronos_bolt` does not exist.

- [ ] **Step 3: Implement the immutable ABI and result writer**

```python
@dataclass(frozen=True)
class TensorContract:
    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"


@dataclass(frozen=True)
class ChronosBoltContract:
    d_model: int
    external_input: TensorContract
    external_output: TensorContract
    core_inputs: tuple[TensorContract, ...]
    core_output: TensorContract
    quantile_levels: tuple[float, ...]

    @classmethod
    def tiny(cls, d_model: int) -> "ChronosBoltContract":
        ...
```

Reject non-FP32, non-batch-one, or non-32-data-patch-plus-REG core tensors with `ValueError`; make the result writer atomically create a new JSON file and reject an existing destination.

- [ ] **Step 4: Verify GREEN and commit**

```bash
python -m pytest tests/test_chronos_bolt_contracts.py -q
git add framework/src/chronos_bolt framework/tests/test_chronos_bolt_contracts.py
git commit -m "feat: define Chronos-Bolt fixed core contract"
```

## Task 2: CPU Host Adapter

**Files:**
- Create: `framework/src/chronos_bolt/host_adapter.py`
- Create: `framework/tests/test_chronos_bolt_host_adapter.py`

**Interfaces:**
- Consumes `context: torch.Tensor` with shape `(1, T)` and optional observed mask.
- Produces `PreparedChronosBoltInputs(input_embeds, attention_mask, decoder_input_embeds, loc, scale)` and `restore_quantiles(normalized)`.

- [ ] **Step 1: Write failing static-patch and NaN tests**

```python
def test_adapter_left_pads_short_context_and_emits_32_patches(fake_model):
    adapter = ChronosBoltHostAdapter(fake_model)
    prepared = adapter.prepare(torch.arange(20, dtype=torch.float32).reshape(1, 20))
    assert prepared.input_embeds.shape == (1, 33, fake_model.config.d_model)
    assert prepared.attention_mask.shape == (1, 33)
    assert prepared.decoder_input_embeds.shape == (1, 1, fake_model.config.d_model)


def test_adapter_uses_explicit_mask_and_restores_all_quantiles(fake_model):
    context = torch.tensor([[1.0, float("nan"), 3.0]])
    prepared = ChronosBoltHostAdapter(fake_model).prepare(
        context, observed_mask=torch.tensor([[True, False, True]])
    )
    restored = prepared.restore(torch.ones((1, 9, 64)))
    assert restored.shape == (1, 9, 64)
    assert torch.isfinite(restored).all()
```

- [ ] **Step 2: Verify RED**

```bash
python -m pytest tests/test_chronos_bolt_host_adapter.py -q
```

Expected: import fails because the host adapter is absent.

- [ ] **Step 3: Implement semantics without `unfold`**

Accept a checkpoint maximum context of at least 512, then right-align crop/left NaN pad the benchmark input to 512 positions. Calculate mean and scale in FP32 from the NaN pattern exactly as Chronos `InstanceNorm` does; an explicit observed mask affects only patch contents and attention, not `loc`/`scale`. Replace patch values whose mask is zero by zero after normalization. Construct `(1, 32, 16)` values and masks with a fixed `torch.stack` of 32 slices, concatenate them to `(1, 32, 32)`, invoke the loaded model's learned `input_patch_embedding`, append the learned REG embedding, and get the decoder-start embedding through `shared` exactly once. Do not use `Tensor.unfold`, `nanmean`, or a dynamic Python-length loop in the produced core ABI.

- [ ] **Step 4: Verify GREEN and commit**

```bash
python -m pytest tests/test_chronos_bolt_host_adapter.py -q
git add framework/src/chronos_bolt/host_adapter.py framework/tests/test_chronos_bolt_host_adapter.py
git commit -m "feat: add Chronos-Bolt CPU host adapter"
```

## Task 3: Tensor-Only Transformer Core and Reference Parity CLI

**Files:**
- Create: `framework/src/chronos_bolt/core.py`
- Create: `framework/tools/chronos_bolt_compile.py`
- Create: `framework/tests/test_chronos_bolt_core.py`
- Create: `framework/tests/test_chronos_bolt_compile_cli.py`

**Interfaces:**
- `ChronosBoltTransformerCore(model: torch.nn.Module).forward(input_embeds, attention_mask, decoder_input_embeds) -> torch.Tensor`
- CLI command: `python tools/chronos_bolt_compile.py --vendor reference --model-path PATH --output-dir PATH`.

- [ ] **Step 1: Write failing core and CLI tests**

```python
def test_core_returns_only_fixed_quantile_tensor(fake_chronos_model):
    core = ChronosBoltTransformerCore(fake_chronos_model)
    output = core(
        torch.zeros((1, 33, 8)),
        torch.ones((1, 33)),
        torch.zeros((1, 1, 8)),
    )
    assert type(output) is torch.Tensor
    assert output.shape == (1, 9, 64)


def test_describe_does_not_import_vendor_sdk(capsys):
    assert main(["--vendor", "reference", "--describe"]) == 0
    assert json.loads(capsys.readouterr().out)["external_output"]["shape"] == [1, 9, 64]
```

- [ ] **Step 2: Verify RED**

```bash
python -m pytest tests/test_chronos_bolt_core.py tests/test_chronos_bolt_compile_cli.py -q
```

Expected: imports fail because the core and CLI are absent.

- [ ] **Step 3: Implement core and CPU parity path**

Call the T5 encoder and decoder with `return_dict=False`; select tensors positionally, apply Chronos's output patch embedding, and use a fixed reshape to `(1, 9, 64)`. The reference command must load local snapshot files only, compare full-model output against adapter/core/restore output on finite and NaN synthetic contexts, and write only a new result JSON. It must reject a failed all-quantile `torch.testing.assert_close(rtol=1e-5, atol=1e-6)` gate rather than relaxing tolerances.

- [ ] **Step 4: Verify GREEN and commit**

```bash
python -m pytest tests/test_chronos_bolt_core.py tests/test_chronos_bolt_compile_cli.py -q
git add framework/src/chronos_bolt framework/tools/chronos_bolt_compile.py framework/tests/test_chronos_bolt_core.py framework/tests/test_chronos_bolt_compile_cli.py
git commit -m "feat: add Chronos-Bolt split Transformer core"
```

## Task 4: Rebellions CA22 Offline Compiler Adapter

**Files:**
- Create: `framework/tools/chronos_bolt_vendors/__init__.py`
- Create: `framework/tools/chronos_bolt_vendors/rbln.py`
- Create: `framework/tests/test_chronos_bolt_rbln.py`

**Interfaces:**
- Consumes the shared core, `ChronosBoltContract`, three deterministic examples, and `--artifact OUTPUT.rbln`.
- Produces a CA22 `.rbln`, independent inspection data, and a result JSON with `compiled` or a precise failure status.

- [ ] **Step 1: Write failing fake-SDK tests**

```python
def test_rbln_adapter_compiles_exact_three_input_contract(fake_rebel, tiny_core):
    result = compile_rbln(tiny_core, ChronosBoltContract.tiny(d_model=8), tmp_path / "tiny.rbln")
    assert fake_rebel.calls[0]["method"] == "compile_from_torch"
    assert [item[0] for item in fake_rebel.calls[0]["inputs"]] == [
        "input_embeds", "attention_mask", "decoder_input_embeds"
    ]
    assert result.status.value == "compiled"


def test_rbln_adapter_rejects_empty_or_wrong_abi_artifact(fake_rebel, tiny_core):
    fake_rebel.inspection["outputs"][0]["shape"] = [1, 9, 96]
    with pytest.raises(ValueError, match="shape"):
        compile_rbln(tiny_core, ChronosBoltContract.tiny(d_model=8), tmp_path / "bad.rbln")
```

- [ ] **Step 2: Verify RED**

```bash
python -m pytest tests/test_chronos_bolt_rbln.py -q
```

- [ ] **Step 3: Implement lazy offline compilation**

Import `rebel` only inside `compile_rbln()`. Invoke `rebel.compile_from_torch` with the shared core and all three named FP32 fixed inputs; save only to a nonexistent `.rbln`; then use `RBLNCompiledModel.inspect()` to require `RBLN-CA22`, three ordered input descriptors, and `(1, 9, 64)` output. Record artifact SHA-256, byte size, `rebel-compiler` version, and model snapshot evidence. Do not load the hardware in this task.

- [ ] **Step 4: Verify GREEN and commit**

```bash
python -m pytest tests/test_chronos_bolt_rbln.py -q
git add framework/tools/chronos_bolt_vendors/rbln.py framework/tests/test_chronos_bolt_rbln.py
git commit -m "feat: compile Chronos-Bolt core for CA22"
```

## Task 5: Furiosa Strict RNGD Compiler Adapter

**Files:**
- Create: `framework/tools/chronos_bolt_vendors/furiosa.py`
- Create: `framework/tests/test_chronos_bolt_furiosa.py`
- Modify: `framework/tools/chronos_bolt_compile.py`

**Interfaces:**
- Consumes the shared core and one fixed input triple.
- Produces first-call compiler evidence and `device_verified` only after finite NPU output matches CPU core output within an explicit recorded tolerance.

- [ ] **Step 1: Write failing strict-configuration tests**

```python
def test_furiosa_adapter_disables_eager_fallback_and_dynamic_shapes(fake_furiosa, tiny_core):
    compile_furiosa(tiny_core, examples(), device="furiosa:0")
    assert fake_furiosa.backend_config["eager_fallback"] is False
    assert fake_furiosa.compile_kwargs == {"fullgraph": True, "dynamic": False}


def test_furiosa_adapter_does_not_call_cpu_output_a_success(fake_furiosa, tiny_core):
    fake_furiosa.raise_on_first_npu_call = RuntimeError("eager fallback is not allowed")
    result = try_compile_furiosa(tiny_core, examples(), device="furiosa:0")
    assert result.status.value == "compile_failed"
```

- [ ] **Step 2: Verify RED**

```bash
python -m pytest tests/test_chronos_bolt_furiosa.py -q
```

- [ ] **Step 3: Implement strict first-call compilation**

Create the Furiosa backend with `CompilerConfig(TacticHintConfig.Default)` and `eager_fallback=False`; call `torch.compile(core, backend=..., fullgraph=True, dynamic=False)`, move exactly the three inputs to `furiosa:0`, synchronize before/after the timed first call, and require returned FP32 finite output. Store first-call compile duration separately from steady-state inference timing. SDK absence is `prerequisite_missing`; missing device is `compile_blocked_no_device`; a compiler exception is `compile_failed`.

- [ ] **Step 4: Verify GREEN and commit**

```bash
python -m pytest tests/test_chronos_bolt_furiosa.py -q
git add framework/tools/chronos_bolt_vendors/furiosa.py framework/tools/chronos_bolt_compile.py framework/tests/test_chronos_bolt_furiosa.py
git commit -m "feat: add strict RNGD Chronos-Bolt compile"
```

## Task 6: Mobilint ARIES Compiler Adapter and Operator Runbook

**Files:**
- Create: `framework/tools/chronos_bolt_vendors/mobilint.py`
- Create: `framework/tests/test_chronos_bolt_mobilint.py`
- Create: `framework/docs/chronos-bolt-cross-vendor.md`
- Modify: `framework/tools/chronos_bolt_compile.py`

**Interfaces:**
- Consumes the identical Torch core, target `aries-rb`, fixed input ABI, and a declared deterministic calibration directory.
- Produces an `.mblt` intermediate, an `.mxq` artifact, metadata inspection, and explicit rejection of CPU-offloaded/partitioned compilation.

- [ ] **Step 1: Write failing fake-SDK tests**

```python
def test_mobilint_adapter_targets_aries_without_cpu_offload(fake_qbcompiler, tiny_core):
    compile_mobilint(tiny_core, contract(), calibration_dir=tmp_path, target_device="aries-rb")
    assert fake_qbcompiler.mblt_kwargs["target_device"] == "aries-rb"
    assert fake_qbcompiler.mblt_kwargs["cpu_offload"] is False
    assert fake_qbcompiler.mxq_kwargs["target_device"] == "aries-rb"


def test_mobilint_adapter_rejects_wrong_target_before_sdk_import():
    with pytest.raises(ValueError, match="aries-rb"):
        compile_mobilint(object(), contract(), calibration_dir=Path("."), target_device="regulus-rb")
```

- [ ] **Step 2: Verify RED**

```bash
python -m pytest tests/test_chronos_bolt_mobilint.py -q
```

- [ ] **Step 3: Implement MBLT/MXQ compilation and documentation**

Import `qbcompiler` only after checking `target_device == "aries-rb"`. Feed the three fixed tensors through its Torch wrappers, compile MBLT with `cpu_offload=False`, reject any compiler report that declares a host partition, generate deterministic calibration `.npy` inputs from adapter outputs, compile MXQ, then inspect loadable metadata using `qbruntime`. The runbook must state that `qbcompiler` is a separate vendor wheel, typically requiring its supported Python ABI, and document exact upload-free commands to run against a local checkout.

- [ ] **Step 4: Verify GREEN, run regressions, and commit**

```bash
python -m pytest tests/test_chronos_bolt_mobilint.py tests/test_mobilint_tensor_contracts.py -q
python -m pytest tests/test_chronos_bolt_contracts.py tests/test_chronos_bolt_host_adapter.py tests/test_chronos_bolt_core.py tests/test_chronos_bolt_compile_cli.py tests/test_chronos_bolt_rbln.py tests/test_chronos_bolt_furiosa.py tests/test_chronos_bolt_mobilint.py -q
git add framework/tools/chronos_bolt_vendors/mobilint.py framework/tools/chronos_bolt_compile.py framework/tests/test_chronos_bolt_mobilint.py framework/docs/chronos-bolt-cross-vendor.md
git commit -m "feat: add ARIES Chronos-Bolt compile adapter"
```

## Task 7: Hardware Evidence and Model-Size Expansion

**Files:**
- Modify: `framework/docs/chronos-bolt-cross-vendor.md`
- Runtime outputs: ignored `framework/results/chronos-bolt-cross-vendor/<run-id>/`

**Interfaces:**
- Consumes compiled Tiny artifacts and the exact reference snapshot.
- Produces per-vendor `compiled`/`device_verified` results and repeatable commands for mini, small, and base.

- [ ] **Step 1: Run Tiny reference parity on the actual reference environment**

```bash
REF_PY=/home/etri_ecas/ML-HW-Benchmark-Framework/.venv-chronos-reference/bin/python
cd /home/etri_ecas/ML-HW-Benchmark-Framework/framework
"$REF_PY" tools/chronos_bolt_compile.py \
  --vendor reference \
  --model-path models/amazon_chronos-bolt-tiny \
  --output-dir results/chronos-bolt-cross-vendor/reference-tiny
```

- [ ] **Step 2: Run each vendor only after its compiler precondition passes**

```bash
"$RBLN_PY" tools/chronos_bolt_compile.py --vendor rbln --model-path "$MODEL" --artifact "$RUN/tiny.rbln" --output-dir "$RUN"
"$FURIOSA_PY" tools/chronos_bolt_compile.py --vendor furiosa --model-path "$MODEL" --device furiosa:0 --output-dir "$RUN"
"$MOBILINT_PY" tools/chronos_bolt_compile.py --vendor mobilint --model-path "$MODEL" --target-device aries-rb --calibration-dir "$RUN/calibration" --artifact "$RUN/tiny.mxq" --output-dir "$RUN"
```

- [ ] **Step 3: Record only evidence-backed outcomes and extend sizes**

For every Tiny result, record status, artifact hash, compiler/runtime versions, output shape/dtype, max/mean CPU delta, and NPU utility sample. Repeat the same command for `amazon/chronos-bolt-mini`, `small`, and `base` only after Tiny is recorded; never infer a larger model's support from a smaller model's result. Commit only the final Markdown summary, never artifacts or raw logs.

## Plan Self-Review

- Shared source-of-truth ABI, CPU semantics, all three vendors, validation, manifests, and device proof are covered by Tasks 1–7.
- The plan contains no fallback-to-success path: CPU partitions, callable-only compilation, and hardware-less output are rejected or recorded as non-success.
- Package and artifact operations are lazy and vendor-isolated, so ordinary tests do not depend on NPU SDKs.
- Mobilint compilation is explicitly gated on obtaining a vendor `qbcompiler` wheel; ARIES runtime availability alone is not treated as compiler availability.
