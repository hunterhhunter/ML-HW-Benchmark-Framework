# TTM-R1 Cross-Vendor Compilation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a reproducible strict CPU-reference, compilation, execution, and evidence path for `ibm-granite/granite-timeseries-ttm-r1` on CA22, RNGD, and ARIES.

**Architecture:** A TTM-specific package owns the fixed public contract, CPU scaling/restore adapter, and tensor-only model wrapper. A command-line driver runs a shared reference preflight before dispatching to per-vendor strict runners; it writes the same evidence structure used by Chronos-Bolt without refactoring that existing implementation.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Transformers/Hub, NumPy, pytest, rebel-compiler, furiosa-torch, qbcompiler/qbruntime.

## Global Constraints

- Use the local `ibm-granite/granite-timeseries-ttm-r1` checkpoint only; execution never downloads or changes weights.
- Keep the public ABI exactly `context: float32 [1,512,1]` and `forecast: float32 [1,96,1]`.
- Scale and restore on CPU; assert finite and NaN host-path parity before calling a vendor compiler.
- Compile a tensor-only static core; reject dynamic axes, eager fallback, or a CPU substitute for device execution.
- Preserve a result JSON for every terminal outcome; use `device_verified` only after a physical device execution passes the common numeric gate.
- Do not refactor the Chronos-Bolt modules or add TTM-R2 in this implementation.

---

### Task 1: Define the fixed TTM-R1 ABI and CPU data boundary

**Files:**
- Create: `framework/src/ttm_r1/__init__.py`
- Create: `framework/src/ttm_r1/contracts.py`
- Create: `framework/src/ttm_r1/host_adapter.py`
- Create: `framework/tests/test_ttm_r1_contracts.py`
- Create: `framework/tests/test_ttm_r1_host_adapter.py`

**Interfaces:**
- Produces `TTMR1Contract.fixed() -> TTMR1Contract`, with `external_input`, `external_output`, `core_inputs`, and `core_output` tensor contracts.
- Produces `TTMR1HostAdapter.prepare(context: torch.Tensor) -> PreparedTTMR1Inputs` and `PreparedTTMR1Inputs.restore(normalized_forecast: torch.Tensor) -> torch.Tensor`.
- The initial core input is `past_values: float32 [1,512,1]`; a later static-patch boundary may only replace this after a passing CPU equivalence test and a recorded ABI change.

- [ ] **Step 1: Write the failing ABI tests**

```python
def test_fixed_contract_exposes_univariate_512_to_96_abi():
    contract = TTMR1Contract.fixed()
    assert contract.external_input == TensorContract("context", (1, 512, 1), "float32")
    assert contract.external_output == TensorContract("forecast", (1, 96, 1), "float32")
    assert contract.core_inputs == (TensorContract("past_values", (1, 512, 1), "float32"),)
    assert contract.core_output == TensorContract("forecast", (1, 96, 1), "float32")
```

```python
def test_adapter_standardizes_finite_values_and_restores_original_scale():
    context = torch.arange(512, dtype=torch.float32).reshape(1, 512, 1)
    prepared = TTMR1HostAdapter().prepare(context)
    assert torch.allclose(prepared.past_values.mean(dim=1), torch.zeros((1, 1)))
    assert torch.allclose(prepared.restore(prepared.past_values[:, :96]), context[:, :96])
```

```python
def test_adapter_rejects_wrong_shape_and_all_missing_context():
    with pytest.raises(ValueError, match="shape"):
        TTMR1HostAdapter().prepare(torch.zeros((1, 512), dtype=torch.float32))
    with pytest.raises(ValueError, match="observed"):
        TTMR1HostAdapter().prepare(torch.full((1, 512, 1), torch.nan))
```

- [ ] **Step 2: Run tests to verify the expected missing-module failure**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_ttm_r1_contracts.py framework/tests/test_ttm_r1_host_adapter.py`

Expected: collection fails because `ttm_r1` does not exist.

- [ ] **Step 3: Implement the contract and adapter**

```python
@dataclass(frozen=True)
class TTMR1Contract:
    external_input: TensorContract
    external_output: TensorContract
    core_inputs: tuple[TensorContract, ...]
    core_output: TensorContract

    @classmethod
    def fixed(cls) -> "TTMR1Contract":
        context = TensorContract("context", (1, 512, 1), "float32")
        forecast = TensorContract("forecast", (1, 96, 1), "float32")
        return cls(context, forecast, (TensorContract("past_values", (1, 512, 1), "float32"),), forecast)
```

`TTMR1HostAdapter.prepare` validates FP32 `[1,512,1]`, derives mean and population standard deviation only from finite values, requires at least one observation, replaces NaNs with the mean, and returns normalized `past_values`, `loc`, and nonzero `scale`. `restore` requires FP32 `[1,96,1]` and computes `forecast * scale + loc`.

- [ ] **Step 4: Run the two test files and verify they pass**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_ttm_r1_contracts.py framework/tests/test_ttm_r1_host_adapter.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the ABI boundary**

```bash
git add framework/src/ttm_r1 framework/tests/test_ttm_r1_contracts.py framework/tests/test_ttm_r1_host_adapter.py
git commit -m "feat: define TTM-R1 fixed host contract"
```

### Task 2: Add a tensor-only TTM wrapper and CPU reference preflight

**Files:**
- Create: `framework/src/ttm_r1/core.py`
- Create: `framework/src/ttm_r1/reference.py`
- Create: `framework/tests/test_ttm_r1_core.py`
- Create: `framework/tests/test_ttm_r1_reference.py`

**Interfaces:**
- Produces `TTMR1Core(model: torch.nn.Module)`, whose `forward(past_values)` returns only FP32 `[1,96,1]`.
- Produces `load_ttm_r1_model(model_path: str) -> torch.nn.Module`, using `TinyTimeMixerForPrediction.from_pretrained(..., local_files_only=True)`.
- Produces `run_preflight(model_path: Path) -> TTMR1Preflight`, with the core, contract, finite/NaN core inputs and outputs, and host/reference parity metadata.

- [ ] **Step 1: Write failing wrapper and preflight tests**

```python
def test_core_unwraps_prediction_outputs_to_a_single_forecast_tensor():
    core = TTMR1Core(_FakeTTM())
    output = core(torch.zeros((1, 512, 1), dtype=torch.float32))
    assert output.shape == (1, 96, 1)
    assert output.dtype == torch.float32
```

```python
def test_preflight_requires_exact_host_and_reference_forecasts(monkeypatch, tmp_path):
    monkeypatch.setattr(reference, "load_ttm_r1_model", lambda _: _FakeTTM())
    result = reference.run_preflight(tmp_path)
    assert set(result.core_inputs) == {"finite", "nan"}
    assert result.host_parity["finite"]["max_abs_error"] == 0.0
```

```python
def test_core_rejects_a_model_output_without_a_96_step_prediction():
    with pytest.raises(ValueError, match="forecast"):
        TTMR1Core(_BadTTM())(torch.zeros((1, 512, 1), dtype=torch.float32))
```

- [ ] **Step 2: Run tests to verify the wrapper is absent**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_ttm_r1_core.py framework/tests/test_ttm_r1_reference.py`

Expected: collection fails because `ttm_r1.core` and `ttm_r1.reference` do not exist.

- [ ] **Step 3: Implement the minimal wrapper and preflight**

```python
class TTMR1Core(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.contract = TTMR1Contract.fixed()

    def forward(self, past_values: torch.Tensor) -> torch.Tensor:
        _validate_tensor(past_values, self.contract.core_inputs[0])
        output = self.model(past_values=past_values, return_dict=True)
        forecast = _extract_forecast(output)
        _validate_tensor(forecast, self.contract.core_output)
        return forecast
```

`run_preflight` creates deterministic finite and NaN contexts, uses `TTMR1HostAdapter`, executes the unwrapped checkpoint directly on the prepared values, restores both reference and wrapper outputs, and applies `torch.testing.assert_close(..., rtol=1e-5, atol=1e-6)` before returning any compiler inputs.

- [ ] **Step 4: Run the core and reference tests**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_ttm_r1_core.py framework/tests/test_ttm_r1_reference.py`

Expected: all tests pass with no model download.

- [ ] **Step 5: Commit the CPU-reference path**

```bash
git add framework/src/ttm_r1/core.py framework/src/ttm_r1/reference.py framework/tests/test_ttm_r1_core.py framework/tests/test_ttm_r1_reference.py
git commit -m "feat: add TTM-R1 strict reference preflight"
```

### Task 3: Provide checkpoint acquisition and a reproducible CPU CLI

**Files:**
- Create: `framework/tools/acquire_ttm_r1.py`
- Create: `framework/tools/ttm_r1_compile.py`
- Create: `framework/tests/test_acquire_ttm_r1.py`
- Create: `framework/tests/test_ttm_r1_compile_cli.py`

**Interfaces:**
- Produces `acquire(model_id: str, destination: Path) -> Path`, which calls `snapshot_download` once, rejects a nonempty destination, and writes a SHA-256 manifest for `config.json` and weight files.
- Produces `main(argv) -> int` with `--vendor {reference,rbln,furiosa,mobilint}`, `--model-path`, `--output-dir`, and `--describe`.
- `--describe` needs no checkpoint or vendor SDK; every execution mode requires both local paths.

- [ ] **Step 1: Write failing acquisition and dispatch tests**

```python
def test_acquire_rejects_nonempty_destination(tmp_path):
    destination = tmp_path / "model"
    destination.mkdir()
    (destination / "stale").write_text("x")
    with pytest.raises(FileExistsError, match="nonempty"):
        acquire_ttm_r1.acquire("ibm-granite/granite-timeseries-ttm-r1", destination)
```

```python
def test_describe_reports_the_fixed_public_contract(capsys):
    assert ttm_r1_compile.main(["--vendor", "reference", "--describe"]) == 0
    assert json.loads(capsys.readouterr().out)["external_input"]["shape"] == [1, 512, 1]
```

```python
def test_mobilint_dispatch_requires_model_and_output_paths():
    with pytest.raises(ValueError, match="--model-path"):
        ttm_r1_compile.main(["--vendor", "mobilint"])
```

- [ ] **Step 2: Run the tests to verify the command modules are absent**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_acquire_ttm_r1.py framework/tests/test_ttm_r1_compile_cli.py`

Expected: collection fails because both tool modules do not exist.

- [ ] **Step 3: Implement acquisition, reference evidence, and CLI dispatch**

`acquire_ttm_r1.py` uses `huggingface_hub.snapshot_download` with the immutable model id, `local_dir=destination`, and no pattern that excludes `config.json` or `model.safetensors`. The command writes `ttm-r1-manifest.json` with the resolved directory and SHA-256 values.

`ttm_r1_compile.py` serializes `TTMR1Contract.fixed()`, calls `run_preflight`, writes `reference-result.json` through `chronos_bolt.evidence.write_result`, and dispatches vendor modes only after that preflight succeeds.

- [ ] **Step 4: Run the acquisition and CLI tests**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_acquire_ttm_r1.py framework/tests/test_ttm_r1_compile_cli.py`

Expected: all tests pass using monkeypatched Hub and vendor functions.

- [ ] **Step 5: Commit reproducible CPU automation**

```bash
git add framework/tools/acquire_ttm_r1.py framework/tools/ttm_r1_compile.py framework/tests/test_acquire_ttm_r1.py framework/tests/test_ttm_r1_compile_cli.py
git commit -m "feat: automate TTM-R1 acquisition and reference checks"
```

### Task 4: Add strict CA22, RNGD, and ARIES runners

**Files:**
- Create: `framework/tools/ttm_r1_vendors/__init__.py`
- Create: `framework/tools/ttm_r1_vendors/rbln.py`
- Create: `framework/tools/ttm_r1_vendors/furiosa.py`
- Create: `framework/tools/ttm_r1_vendors/mobilint.py`
- Create: `framework/tests/test_ttm_r1_rbln.py`
- Create: `framework/tests/test_ttm_r1_furiosa.py`
- Create: `framework/tests/test_ttm_r1_mobilint.py`
- Modify: `framework/tools/ttm_r1_compile.py`

**Interfaces:**
- Produces `compile_rbln(core, contract, artifact) -> dict` and `run_rbln(artifact, inputs, contract) -> np.ndarray`.
- Produces `run_furiosa(core, inputs, contract) -> torch.Tensor` with `fullgraph=True`, `dynamic=False`, and `eager_fallback=False`.
- Produces `export_core_onnx(core, inputs, contract, path) -> Path`, `compile_mblt(onnx, artifact) -> dict`, and `run_mblt(artifact, inputs, contract) -> np.ndarray` targeting `aries-rb`.

- [ ] **Step 1: Write the failing strict-runner tests**

```python
def test_furiosa_runner_forbids_eager_fallback():
    dependencies = _FakeFuriosaDependencies()
    run_furiosa(_IdentityCore(), (torch.zeros((1, 512, 1)),), TTMR1Contract.fixed(), dependencies=dependencies)
    assert dependencies.compile_kwargs == {"fullgraph": True, "dynamic": False, "options": {"eager_fallback": False}}
```

```python
def test_mobilint_export_has_static_axes_and_aries_rb_target(monkeypatch, tmp_path):
    onnx_path = export_core_onnx(_IdentityCore(), (torch.zeros((1, 512, 1)),), TTMR1Contract.fixed(), tmp_path / "core.onnx")
    compiled = compile_mblt(onnx_path, tmp_path / "core.mblt", qbcompiler_module=_FakeQbCompiler())
    assert compiled["target_device"] == "aries-rb"
```

```python
def test_rbln_runner_rejects_a_device_output_with_the_wrong_contract_shape():
    with pytest.raises(ValueError, match="output"):
        run_rbln(_FakeArtifact(), (np.zeros((1, 512, 1), np.float32),), TTMR1Contract.fixed())
```

- [ ] **Step 2: Run runner tests to verify imports fail**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_ttm_r1_rbln.py framework/tests/test_ttm_r1_furiosa.py framework/tests/test_ttm_r1_mobilint.py`

Expected: collection fails because `tools.ttm_r1_vendors` does not exist.

- [ ] **Step 3: Implement vendor runners and CLI evidence**

Port the validated Chronos-Bolt vendor mechanics into `ttm_r1_vendors`, changing only module names and the one-input TTM contract. The ONNX exporter uses input name `past_values`, output name `forecast`, opset 17, `dynamic_axes=None`, and `dynamo=False`. The Mobilint compiler calls `mblt_compile_V2(..., target_device="aries-rb", backend="onnx", device="cpu", cpu_offload=False)`.

Add CLI vendor handlers that run finite and NaN cases, compute the shared `rtol=1e-3, atol=1e-3` parity record, write artifact and compiler inspection metadata, then raise after writing a `parity_failed` result when the numeric gate fails.

- [ ] **Step 4: Run strict-runner tests and the complete TTM unit suite**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_ttm_r1_*.py`

Expected: all TTM tests pass without a physical NPU.

- [ ] **Step 5: Commit cross-vendor runners**

```bash
git add framework/tools/ttm_r1_vendors framework/tools/ttm_r1_compile.py framework/tests/test_ttm_r1_rbln.py framework/tests/test_ttm_r1_furiosa.py framework/tests/test_ttm_r1_mobilint.py
git commit -m "feat: add strict TTM-R1 vendor runners"
```

### Task 5: Document user-run device commands and verify the repository suite

**Files:**
- Create: `framework/docs/ttm-r1-cross-vendor.md`
- Create: `framework/tests/test_ttm_r1_runbook.py`
- Modify: `framework/README.md`
- Test: `framework/tests/test_ttm_r1_*.py`

**Interfaces:**
- Documents one copyable command group per existing RBLN, Furiosa, and Mobilint virtual environment.
- Documents that an artifact or device execution failure is an expected evidence result, not an instruction to relax compilation.

- [ ] **Step 1: Write a failing documentation-presence test**

```python
def test_ttm_r1_runbook_mentions_all_vendor_modes():
    text = Path("framework/docs/ttm-r1-cross-vendor.md").read_text()
    assert "--vendor rbln" in text
    assert "--vendor furiosa" in text
    assert "--vendor mobilint" in text
```

- [ ] **Step 2: Run the test to verify the runbook is absent**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_ttm_r1_runbook.py`

Expected: FAIL because the runbook path does not exist.

- [ ] **Step 3: Add the runbook and README link**

The runbook contains exact activation, checkpoint acquisition, reference, and one-command-per-vendor invocations. It records the expected model directory as `framework/models/ibm-granite_granite-timeseries-ttm-r1`, creates timestamped result directories, and names the result JSON files to inspect after every run.

- [ ] **Step 4: Run all TTM tests and the existing Chronos tests**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_ttm_r1_*.py framework/tests/test_chronos_bolt_*.py`

Expected: all tests pass; the existing Chronos suite proves no accidental cross-model modification.

- [ ] **Step 5: Commit documentation and verified test suite**

```bash
git add framework/docs/ttm-r1-cross-vendor.md framework/README.md framework/tests/test_ttm_r1_runbook.py
git commit -m "docs: add TTM-R1 cross-vendor runbook"
```

## Plan self-review

- Spec coverage: Tasks 1–2 implement the fixed ABI and CPU parity; Task 3 covers acquisition and CLI; Task 4 covers all three strict vendor paths and evidence; Task 5 supplies the user-run commands and regression verification.
- Placeholder scan: the plan has no deferred behavior labels; the only future conditional is the explicitly tested static-patch boundary rule from the approved design.
- Type consistency: every runner accepts `TTMR1Contract.fixed()` and one ordered `past_values` tensor; every CLI invocation begins with `run_preflight` and emits `[1,96,1]` forecasts.
