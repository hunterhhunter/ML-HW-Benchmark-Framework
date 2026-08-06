# TTM-R2 Cross-Vendor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the official TTM-R2 `main` (`512-96-r2`) checkpoint on CA22, RNGD, and ARIES with the same fixed ABI and ETTh1 comparison as TTM-R1.

**Architecture:** Create a small `ttm_r2` package with its own immutable checkpoint identity, fixed `[1,512,1] -> [1,96,1]` contract, CPU scaler/patchify split, and preflight. Reuse the proven vendor semantics from TTM-R1, but keep artifact and evidence names R2-specific. Compile ARIES MXQ locally and execute only transferred artifact/fixture on the remote runtime host.

**Tech Stack:** Python 3.10/3.12, PyTorch, Transformers/`granite-tsfm`, safetensors, Rebel SDK 0.11.0, Furiosa Torch, ONNX Runtime, qbcompiler 1.2.0, qbruntime 1.3.2.

## Global Constraints

- Checkpoint is exactly `ibm-granite/granite-timeseries-ttm-r2`, revision `main` (`512-96-r2`); do not use automatic model selection or a frequency-prefix branch.
- Device contract is fixed FP32 input `[1,512,1]` and output `[1,96,1]`.
- Keep standard scaling, NaN fill, and output restoration on CPU; device graph receives the same prepared core input on every vendor.
- CPU public/split gate is `torch.testing.assert_close(rtol=1e-5, atol=1e-6)` for finite and NaN fixtures.
- Strict device parity gate is `rtol=1e-3, atol=1e-3`; preserve a `parity_failed` JSON rather than hiding an executed artifact.
- Furiosa must use `fullgraph=True`, `dynamic=False`, and `eager_fallback=False`.
- ARIES compilation happens locally; the remote ARIES host receives only MXQ, fixture, and result files.
- Never overwrite prior result JSON or artifact paths.

---

### Task 1: Acquire and identify the fixed R2 checkpoint

**Files:**
- Create: `framework/src/ttm_r2/__init__.py`
- Create: `framework/src/ttm_r2/download.py`
- Create: `framework/tools/acquire_ttm_r2.py`
- Test: `framework/tests/test_acquire_ttm_r2.py`

**Interfaces:**
- Produces `TTM_R2_REPOSITORY = "ibm-granite/granite-timeseries-ttm-r2"`, `TTM_R2_REVISION = "main"`.
- Produces `download_checkpoint(output_dir: Path) -> Path` and `write_checkpoint_manifest(checkpoint: Path) -> Path`.
- The manifest contains repository, revision, top-level file SHA256s, Python version, and Transformers version.

- [ ] **Step 1: Write the failing acquisition test**

```python
from ttm_r2.download import TTM_R2_REPOSITORY, TTM_R2_REVISION, write_checkpoint_manifest

def test_r2_identity_and_manifest_hashes_checkpoint_files(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    manifest = write_checkpoint_manifest(checkpoint)
    payload = json.loads(manifest.read_text())
    assert TTM_R2_REPOSITORY == "ibm-granite/granite-timeseries-ttm-r2"
    assert TTM_R2_REVISION == "main"
    assert payload["files"]["model.safetensors"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_acquire_ttm_r2.py`

Expected: collection fails because `ttm_r2` does not exist.

- [ ] **Step 3: Implement download and immutable manifest**

```python
snapshot_download(
    repo_id=TTM_R2_REPOSITORY,
    revision=TTM_R2_REVISION,
    local_dir=str(output_dir),
)
```

Reject a nonempty `output_dir`; write `ttm-r2-manifest.json` only once; hash regular top-level files except the manifest itself.

- [ ] **Step 4: Run focused tests**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_acquire_ttm_r2.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add framework/src/ttm_r2 framework/tools/acquire_ttm_r2.py framework/tests/test_acquire_ttm_r2.py
git commit -m "feat: acquire fixed TTM-R2 checkpoint"
```

### Task 2: Implement the fixed R2 CPU split and reference gate

**Files:**
- Create: `framework/src/ttm_r2/contracts.py`
- Create: `framework/src/ttm_r2/host_adapter.py`
- Create: `framework/src/ttm_r2/core.py`
- Create: `framework/src/ttm_r2/reference.py`
- Test: `framework/tests/test_ttm_r2_contracts.py`
- Test: `framework/tests/test_ttm_r2_core.py`
- Test: `framework/tests/test_ttm_r2_reference.py`

**Interfaces:**
- Produces `TTMR2Contract.fixed()` with one `past_values: float32 [1,512,1]` core input and `forecast: float32 [1,96,1]` output.
- Produces `load_ttm_r2_model(model_path: str) -> torch.nn.Module`.
- Produces `TTMR2Core(model)` and `run_preflight(model_path) -> TTMR2Preflight`.
- `TTMR2Preflight` exposes `core`, `contract`, `core_inputs["finite"|"nan"]`, `core_outputs`, and `host_parity`.

- [ ] **Step 1: Write failing ABI and loader tests**

```python
def test_r2_contract_is_the_fixed_512_to_96_single_series_abi():
    contract = TTMR2Contract.fixed()
    assert contract.core_inputs[0].shape == (1, 512, 1)
    assert contract.core_output.shape == (1, 96, 1)

def test_r2_loader_restores_every_safetensor_key(monkeypatch, checkpoint):
    model = load_ttm_r2_model(str(checkpoint))
    assert model.config.context_length == 512
    assert model.config.prediction_length == 96
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_ttm_r2_contracts.py framework/tests/test_ttm_r2_core.py framework/tests/test_ttm_r2_reference.py`

Expected: collection fails because the R2 modules do not exist.

- [ ] **Step 3: Implement the R2 core by preserving R1 lowering semantics**

```python
class TTMR2Core(torch.nn.Module):
    def forward(self, past_values: torch.Tensor) -> torch.Tensor:
        self._validate("past_values", past_values, (1, 512, 1))
        forecast = self.model(past_values=past_values, return_dict=True).prediction_outputs
        self._validate("forecast", forecast, (1, 96, 1))
        return forecast
```

Load with `TinyTimeMixerForPrediction.from_pretrained(..., local_files_only=True)`, add the compatibility metadata when absent, then restore the full safetensors state dict with `strict=True`. Replace only the fixed std scaler and non-overlapping 64-step patchifier with the same CPU/static modules proven for R1. Reject configurations other than context 512, horizon 96, one input channel, patch length 64, stride 64, and eight patches.

- [ ] **Step 4: Implement preflight with finite and NaN fixtures**

```python
for name, context in {"finite": finite, "nan": nan_context}.items():
    prepared = adapter.prepare(context)
    public = original(past_values=prepared.reference_past_values, return_dict=True).prediction_outputs
    split = prepared.restore(core(prepared.past_values))
    torch.testing.assert_close(split, prepared.restore_reference(public), rtol=1e-5, atol=1e-6)
```

- [ ] **Step 5: Run focused tests**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_ttm_r2_contracts.py framework/tests/test_ttm_r2_core.py framework/tests/test_ttm_r2_reference.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add framework/src/ttm_r2 framework/tests/test_ttm_r2_contracts.py framework/tests/test_ttm_r2_core.py framework/tests/test_ttm_r2_reference.py
git commit -m "feat: add TTM-R2 fixed CPU split"
```

### Task 3: Add strict R2 compiler dispatch and vendor adapters

**Files:**
- Create: `framework/tools/ttm_r2_compile.py`
- Create: `framework/tools/ttm_r2_vendors/__init__.py`
- Create: `framework/tools/ttm_r2_vendors/rbln.py`
- Create: `framework/tools/ttm_r2_vendors/furiosa.py`
- Create: `framework/tools/ttm_r2_vendors/mobilint.py`
- Test: `framework/tests/test_ttm_r2_compile_cli.py`
- Test: `framework/tests/test_ttm_r2_rbln.py`
- Test: `framework/tests/test_ttm_r2_furiosa.py`
- Test: `framework/tests/test_ttm_r2_mobilint.py`

**Interfaces:**
- Produces `python tools/ttm_r2_compile.py --vendor {reference,rbln,furiosa,mobilint} --model-path PATH --output-dir PATH`.
- RBLN exposes `compile_rbln(core, contract, artifact)` and `run_rbln_artifact(artifact, inputs, contract)`.
- Furiosa exposes `run_furiosa_core(core, inputs, contract)`.
- Mobilint exposes `export_core_onnx`, `run_onnx_reference`, `compile_mxq`, and `run_mxq` with the R2 contract.

- [ ] **Step 1: Write failing strict-vendor tests**

```python
def test_r2_rbln_uses_one_fixed_float32_512_input(monkeypatch, tmp_path):
    report = compile_rbln(torch.nn.Identity(), TTMR2Contract.fixed(), tmp_path / "core.rbln")
    assert report["inspection"]["inputs"][0]["shape"] == [1, 512, 1]

def test_r2_furiosa_disables_eager_fallback(monkeypatch):
    runner = run_furiosa_core(_Core(), (torch.zeros((1, 512, 1)),), TTMR2Contract.fixed())
    assert captured["fullgraph"] is True
    assert captured["eager_fallback"] is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_ttm_r2_compile_cli.py framework/tests/test_ttm_r2_rbln.py framework/tests/test_ttm_r2_furiosa.py framework/tests/test_ttm_r2_mobilint.py`

Expected: collection fails because the R2 CLI and vendor modules do not exist.

- [ ] **Step 3: Implement the CLI and evidence behavior**

```python
try:
    artifact = compile_rbln(preflight.core, preflight.contract, output_dir / "ttm-r2-core.rbln")
except Exception as error:
    write_result(output_dir / "rbln-result.json", {
        "status": "compile_failed", "vendor": "rbln", "error": {"type": type(error).__name__, "message": str(error)}
    })
    raise
```

Use a fresh output directory; validate each artifact ABI; compare finite and NaN core outputs; preserve JSON on any compiler or runtime exception. RBLN must target CA22 device zero. Furiosa must make exactly one strict compiled runner and execute both fixtures through it. Mobilint must require CPU ONNX parity before calling qbcompiler and must target `aries-rb`.

- [ ] **Step 4: Run focused tests**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_ttm_r2_compile_cli.py framework/tests/test_ttm_r2_rbln.py framework/tests/test_ttm_r2_furiosa.py framework/tests/test_ttm_r2_mobilint.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add framework/tools/ttm_r2_compile.py framework/tools/ttm_r2_vendors framework/tests/test_ttm_r2_compile_cli.py framework/tests/test_ttm_r2_rbln.py framework/tests/test_ttm_r2_furiosa.py framework/tests/test_ttm_r2_mobilint.py
git commit -m "feat: add TTM-R2 strict vendor dispatch"
```

### Task 4: Produce local R2 evidence and ARIES transfer bundle

**Files:**
- Create: `framework/tools/ttm_r2_mobilint_prepare.py`
- Create: `framework/tools/ttm_r2_mobilint_run.py`
- Modify: `framework/docs/ttm-r1-cross-vendor.md` to link the R2 runbook only if the R2 commands are verified.
- Test: `framework/tests/test_ttm_r2_mobilint_prepare.py`

**Interfaces:**
- Local prepare CLI writes `ttm-r2-core.onnx`, `ttm-r2-inference-fixture.npz`, calibration tensors, and `prepare-result.json` to a new directory.
- ARIES runner accepts `--artifact`, `--fixture`, and `--output-dir`; writes `mobilint-result.json` after `qbruntime` device-zero execution.

- [ ] **Step 1: Write failing prepare test**

```python
def test_prepare_refuses_mxq_when_onnx_cpu_parity_misses_gate(tmp_path, monkeypatch):
    with pytest.raises(DeviceParityError, match="ONNX CPU parity"):
        prepare_r2_bundle(model_path, tmp_path / "bundle")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_ttm_r2_mobilint_prepare.py`

Expected: collection fails because the R2 ARIES preparation module does not exist.

- [ ] **Step 3: Implement local and remote ARIES boundaries**

Save finite and NaN prepared inputs and expected core outputs in one compressed fixture. Generate deterministic, finite R2-core calibration tensors locally. The remote runner must require ARIES device zero, inspect MXQ ABI, use `infer_to_float`, compare both outputs to the fixture, and save saturation count plus numeric metrics.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_ttm_r2_mobilint_prepare.py framework/tests/test_ttm_r2_mobilint.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add framework/tools/ttm_r2_mobilint_prepare.py framework/tools/ttm_r2_mobilint_run.py framework/tests/test_ttm_r2_mobilint_prepare.py
git commit -m "feat: add TTM-R2 ARIES transfer validation"
```

### Task 5: Execute the real vendor matrix and measure ETTh1 where possible

**Files:**
- Create: `framework/docs/ttm-r2-cross-vendor.md`
- Modify: `framework/README.md`

**Interfaces:**
- Documents exact RBLN, Furiosa, local-Docker ARIES, SSH transfer, and ARIES runtime commands.
- Result directories hold command output, device status before/after, checkpoint manifest, artifact hash, and result JSON.

- [ ] **Step 1: Write the runbook assertions as a checklist**

```markdown
- [ ] RBLN result contains compiler version, CA22 inspection, artifact hash, and finite/NaN parity.
- [ ] Furiosa result states fullgraph, static, no-fallback mode and contains finite/NaN parity.
- [ ] ARIES result identifies `aries-rb`, MXQ hash, ABI, quantization saturation, and parity.
- [ ] ETTh1 task quality is emitted only after an artifact executes.
```

- [ ] **Step 2: Execute CPU reference locally**

Run: `PYTHONPATH=framework/src python framework/tools/ttm_r2_compile.py --vendor reference --model-path framework/models/ibm-granite_granite-timeseries-ttm-r2 --output-dir framework/results/ttm-r2/reference-<timestamp>`

Expected: `reference-result.json` with both finite and NaN CPU-split parity.

- [ ] **Step 3: Execute real vendor attempts**

Run the documented commands in the matching RBLN, Furiosa, and local QBC/remote ARIES environments. Capture failure JSON as an outcome; do not relabel it as device execution.

- [ ] **Step 4: Execute ETTh1 quality only for successful runtimes**

Use `datasets/etth1/ETTh1.csv`, column `OT`, the existing 8640/2880/2880 split, and 240 test windows. Record CPU and device MAE/RMSE, forecast delta, runtime success, and strict parity status.

- [ ] **Step 5: Verify and commit documentation**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_ttm_r2_*.py && python framework/tools/ttm_r2_compile.py --help`

Expected: all R2 tests pass; runbook never claims a compile or runtime that lacks evidence.

```bash
git add framework/docs/ttm-r2-cross-vendor.md framework/README.md
git commit -m "docs: add TTM-R2 cross-vendor runbook"
```
