# TimesFM 2.5 Cross-Vendor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible fixed-shape validation of `google/timesfm-2.5-200m-transformers` on Rebellions CA22, Furiosa RNGD, and Mobilint ARIES with the public TimesFM result as the FP32 behavioral authority.

**Architecture:** The public `TimesFm2_5ModelForPrediction` path is evaluated at a fixed 1,024-point ETTh1 context and 128-point horizon.  A CPU adapter reproduces list packing, global normalization, flip-invariance combination, inverse normalization, and conditional non-negative clamping.  A single static point core contains the model's patch-wise normalization, embedding, 20-layer decoder, point projection, and median extraction; the same artifact runs twice, on normalized `x` and `-x`, to preserve the public flip-invariance behavior without duplicating model weights in the graph.

**Tech Stack:** Python 3.10/3.12, PyTorch, Transformers 5.14.1, safetensors, ONNX Runtime, Rebel compiler 0.11.0, Furiosa Torch 2026.3, qbcompiler 1.2.0, qbruntime 1.3.2, ETTh1 CSV.

## Global Constraints

- Checkpoint: `google/timesfm-2.5-200m-transformers`, local files only after download; write a SHA-256 manifest.
- Fixed public contract: one finite `float32` context `[1, 1024]` to point forecast `[1, 128]`.
- Use `forecast_context_len=1024`, `window_size=None`, and the checkpoint defaults for `force_flip_invariance` and `truncate_negative`.
- Preserve public semantics before attempting any NPU compilation; static split parity is a prerequisite.
- Do not globally upgrade vendor SDK environments.  Treat missing TimesFM 2.5 import support as environment incompatibility and record it.
- Compile the complete static core first; stage splitting is diagnostic-only.
- Furiosa first-call compilation uses `fullgraph=True`, `dynamic=False`, and `eager_fallback=False`.
- ARIES compiles locally to MXQ and runs remotely through qbruntime; never send MBLT to qbruntime.
- Every output directory must be new and preserve artifacts, fixtures, compiler/runtime metadata, and exceptions.

---

### Task 1: Establish the TimesFM 2.5 reference environment and checkpoint evidence

**Files:**
- Create: `framework/src/timesfm25/__init__.py`
- Create: `framework/src/timesfm25/download.py`
- Create: `framework/tools/timesfm25_download.py`
- Create: `framework/tests/test_timesfm25_download.py`
- Create: `framework/docs/timesfm25-cross-vendor.md`

**Interfaces:**
- Produces `download_checkpoint(output_dir: Path) -> Path`.
- Produces `<model-dir>/timesfm25-manifest.json` containing repository id, file hashes, reference Python, and Transformers version.
- Requires `transformers>=5.14.1` and `TimesFm2_5ModelForPrediction`.

- [ ] **Step 1: Write the failing manifest test**

```python
def test_write_manifest_hashes_every_regular_checkpoint_file(tmp_path: Path):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    result = write_checkpoint_manifest(checkpoint, "google/timesfm-2.5-200m-transformers")
    assert result["repository"] == "google/timesfm-2.5-200m-transformers"
    assert result["files"]["config.json"] == hashlib.sha256(b"{}").hexdigest()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_timesfm25_download.py`

Expected: FAIL because `timesfm25.download` and `write_checkpoint_manifest` do not exist.

- [ ] **Step 3: Implement local-only download and manifest writing**

Create `timesfm25/download.py` with `write_checkpoint_manifest(directory, repository)` and make `timesfm25_download.py` call `huggingface_hub.snapshot_download` only when the destination has no checkpoint files.  Reject an existing destination without a valid manifest instead of overwriting it.  In the manifest, hash each regular checkpoint file in sorted filename order.

- [ ] **Step 4: Run focused tests**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_timesfm25_download.py`

Expected: PASS.

- [ ] **Step 5: Add the reference-environment runbook section**

Document the exact reference command:

```bash
source /home/swlab-youngjin/ML-HW-Benchmark-Framework/.venv-mobilint-compile/bin/activate
python -c 'from transformers import TimesFm2_5ModelForPrediction; print("TimesFM 2.5 available")'
PYTHONPATH=framework/src python framework/tools/timesfm25_download.py \
  --output-dir framework/models/google_timesfm-2.5-200m-transformers
```

- [ ] **Step 6: Commit**

```bash
git add framework/src/timesfm25/download.py framework/tools/timesfm25_download.py \
  framework/tests/test_timesfm25_download.py framework/docs/timesfm25-cross-vendor.md
git commit -m "feat: add TimesFM 2.5 checkpoint evidence"
```

### Task 2: Define the immutable contract and load the public model

**Files:**
- Create: `framework/src/timesfm25/contracts.py`
- Create: `framework/src/timesfm25/model.py`
- Create: `framework/tests/test_timesfm25_contracts.py`
- Create: `framework/tests/test_timesfm25_model.py`

**Interfaces:**
- `TimesFM25Contract.fixed() -> TimesFM25Contract` exposes `external_input`, `core_input`, and `point_output` descriptors.
- `load_timesfm25_model(model_path: str) -> torch.nn.Module` loads only local weights and returns FP32 eval mode with gradients disabled.
- Fixed descriptors are `context: float32 [1, 1024]`, `normalized_context: float32 [1, 1024]`, and `point_forecast: float32 [1, 128]`.

- [ ] **Step 1: Write contract and loader tests**

```python
def test_fixed_contract_uses_model_card_context_and_native_horizon():
    contract = TimesFM25Contract.fixed()
    assert contract.external_input.shape == (1, 1024)
    assert contract.core_output.shape == (1, 128)

def test_loader_rejects_nonlocal_checkpoint(monkeypatch):
    monkeypatch.setattr(model, "_timesfm_model_class", lambda: FakePredictionModel)
    with pytest.raises(ValueError, match="local checkpoint"):
        load_timesfm25_model("missing")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_timesfm25_contracts.py framework/tests/test_timesfm25_model.py`

Expected: FAIL because the `timesfm25` package is absent.

- [ ] **Step 3: Implement the smallest fixed contract and loader**

Implement `TensorContract`/`TimesFM25Contract` locally rather than importing the Chronos-Bolt contract.  `load_timesfm25_model` must verify the directory and `model.safetensors`, import `TimesFm2_5ModelForPrediction`, call `from_pretrained(..., local_files_only=True)`, cast to `float32`, call `eval()`, and call `requires_grad_(False)`.  Validate `patch_length==32`, `horizon_length==128`, `num_hidden_layers==20`, and `hidden_size==1280`.

- [ ] **Step 4: Run focused tests**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_timesfm25_contracts.py framework/tests/test_timesfm25_model.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add framework/src/timesfm25 framework/tests/test_timesfm25_contracts.py framework/tests/test_timesfm25_model.py
git commit -m "feat: add TimesFM 2.5 fixed contract"
```

### Task 3: Build and prove the CPU adapter/static point core split

**Files:**
- Create: `framework/src/timesfm25/core.py`
- Create: `framework/src/timesfm25/host_adapter.py`
- Create: `framework/src/timesfm25/reference.py`
- Create: `framework/tests/test_timesfm25_core.py`
- Create: `framework/tests/test_timesfm25_host_adapter.py`
- Create: `framework/tests/test_timesfm25_reference.py`

**Interfaces:**
- `TimesFM25PointCore(model).forward(normalized_context) -> Tensor[1,128]`.
- `TimesFM25HostAdapter.prepare(context) -> PreparedTimesFM25Inputs` with `normal_context`, `flipped_context`, `loc`, `scale`, and `input_was_nonnegative`.
- `PreparedTimesFM25Inputs.restore(normal, flipped) -> Tensor[1,128]`.
- `run_preflight(model_path) -> TimesFM25Preflight` returns the public output, split output, fixed fixtures, and parity metrics.

- [ ] **Step 1: Write the failing split-parity tests**

```python
def test_restore_matches_flip_invariance_and_global_denormalization():
    prepared = PreparedTimesFM25Inputs(
        normalized_context=torch.zeros((1, 1024)),
        flipped_context=torch.zeros((1, 1024)),
        loc=torch.tensor([[2.0]]),
        scale=torch.tensor([[3.0]]),
        input_was_nonnegative=torch.tensor(False),
    )
    assert torch.equal(
        prepared.restore(torch.full((1, 128), 4.0), torch.full((1, 128), -2.0)),
        torch.full((1, 128), 11.0),
    )

def test_preflight_requires_public_and_split_point_forecasts_to_match(monkeypatch):
    monkeypatch.setattr(reference, "load_timesfm25_model", lambda _: FakeTimesFM())
    with pytest.raises(AssertionError):
        reference.run_preflight("unused")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_timesfm25_core.py framework/tests/test_timesfm25_host_adapter.py framework/tests/test_timesfm25_reference.py`

Expected: FAIL because no static point-core split exists.

- [ ] **Step 3: Implement the exact fixed public-path decomposition**

`TimesFM25HostAdapter.prepare` must reproduce the public path for a full 1,024-value input: construct zero padding, compute `mu_global` and `sigma_global` using the same PyTorch operators as the public model, and form `normalized_context`/`flipped_context`.

`TimesFM25PointCore` must use a registered all-zero long padding buffer with shape `[1,1024]`, invoke the model backbone, apply `output_projection_point`, apply the backbone patch-wise reverse RevIN, select only the final patch, reshape to `[1,128,10]`, and extract `config.decode_index`.  It must not include a Python sequence loop, a dynamic horizon, or a quantile head.

`restore(normal, flipped)` computes `(normal - flipped) / 2`, applies the global inverse RevIN, and only then applies the public conditional non-negative clamp.  `run_preflight` calls the public model with a one-element list and `forecast_context_len=1024`, calls the static core twice, and asserts close at `rtol=1e-5, atol=1e-5` before exposing any fixture.

- [ ] **Step 4: Run focused tests and a real local checkpoint preflight**

Run:

```bash
PYTHONPATH=framework/src pytest -q framework/tests/test_timesfm25_core.py framework/tests/test_timesfm25_host_adapter.py framework/tests/test_timesfm25_reference.py
PYTHONPATH=framework/src python framework/tools/timesfm25_compile.py \
  --vendor reference \
  --model-path framework/models/google_timesfm-2.5-200m-transformers \
  --output-dir framework/results/timesfm25/reference-$(date -u +%Y%m%dT%H%M%SZ)
```

Expected: tests PASS and the real preflight reports zero public-vs-split mismatch within the explicit tolerance.

- [ ] **Step 5: Commit**

```bash
git add framework/src/timesfm25 framework/tests/test_timesfm25_core.py \
  framework/tests/test_timesfm25_host_adapter.py framework/tests/test_timesfm25_reference.py
git commit -m "feat: split TimesFM 2.5 static point core"
```

### Task 4: Add common evidence and strict RBLN/Furiosa compilation

**Files:**
- Create: `framework/tools/timesfm25_compile.py`
- Create: `framework/tools/timesfm25_vendors/__init__.py`
- Create: `framework/tools/timesfm25_vendors/rbln.py`
- Create: `framework/tools/timesfm25_vendors/furiosa.py`
- Create: `framework/tests/test_timesfm25_rbln.py`
- Create: `framework/tests/test_timesfm25_furiosa.py`

**Interfaces:**
- CLI: `--vendor {reference,rbln,furiosa,mobilint}`, `--model-path`, `--output-dir`, and `--describe`.
- `compile_rbln(core, contract, artifact)` and `run_rbln_artifact(...) -> np.ndarray`.
- `run_furiosa_core(core, inputs, contract) -> torch.Tensor`.
- Result evidence uses `compiled`, `device_verified`, `parity_failed`, or `compile_failed` without suppressing compiler exceptions.

- [ ] **Step 1: Write vendor contract tests with fake SDKs**

```python
def test_furiosa_requests_strict_static_compilation(fake_dependencies):
    run_furiosa_core(torch.nn.Identity(), (torch.zeros((1, 1024)),), TimesFM25Contract.fixed(), dependencies=fake_dependencies)
    assert fake_dependencies.backend.eager_fallback is False
    assert fake_dependencies.torch.compile_calls == [{"fullgraph": True, "dynamic": False}]

def test_rbln_rejects_artifact_with_wrong_fixed_input_shape(fake_rebel, tmp_path):
    with pytest.raises(ValueError, match="shape mismatch"):
        compile_rbln(torch.nn.Identity(), TimesFM25Contract.fixed(), tmp_path / "core.rbln")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_timesfm25_rbln.py framework/tests/test_timesfm25_furiosa.py`

Expected: FAIL because vendor helpers and CLI are absent.

- [ ] **Step 3: Implement strict vendor paths by adapting TTM-R1 helpers**

Copy only the tested RBLN/Furiosa mechanisms; parameterize them with `TimesFM25Contract`, do not refactor existing TTM helpers.  The CLI must run reference preflight first, produce separate normal/flipped NPU invocations, restore them on CPU, and record both core-output parity and restored point-forecast parity.  Persist a JSON failure result before re-raising exceptions.

- [ ] **Step 4: Run unit tests and vendor import preflight**

Run:

```bash
PYTHONPATH=framework/src pytest -q framework/tests/test_timesfm25_rbln.py framework/tests/test_timesfm25_furiosa.py
python -c 'from transformers import TimesFm2_5ModelForPrediction; print("TimesFM 2.5 import available")'
```

Expected: unit tests PASS.  A missing class in an existing RBLN/Furiosa environment is written as an environment-preflight failure; no SDK package is modified automatically.

- [ ] **Step 5: Commit**

```bash
git add framework/tools/timesfm25_compile.py framework/tools/timesfm25_vendors \
  framework/tests/test_timesfm25_rbln.py framework/tests/test_timesfm25_furiosa.py
git commit -m "feat: add TimesFM 2.5 RBLN and Furiosa validation"
```

### Task 5: Add local ARIES MXQ compilation and remote runtime verification

**Files:**
- Create: `framework/src/timesfm25/mobilint_aries.py`
- Create: `framework/tools/timesfm25_mobilint_calibrate.py`
- Create: `framework/tools/timesfm25_mobilint_run.py`
- Create: `framework/tests/test_timesfm25_mobilint.py`

**Interfaces:**
- Local compiler input: TimesFM static point-core ONNX and 256 train-only normalized `[1,1024]` calibration tensors.
- Local artifact: `timesfm25-point-core.mxq` targeted to `aries-rb`.
- Remote runner: accepts MXQ and a fixture, discovers runtime input/output ABI and scale, performs only documented FP32-to-int8 conversion, runs `qbruntime`, and records dequantized point-core output.

- [ ] **Step 1: Write failing ABI/quantization tests**

```python
def test_quantize_core_input_rejects_a_runtime_abi_with_wrong_element_count():
    with pytest.raises(ValueError, match="1024"):
        quantize_core_input(np.zeros((1, 1024), np.float32), (1, 16, 32), FakeScale())

def test_restore_core_output_requires_128_values():
    with pytest.raises(ValueError, match="128"):
        restore_core_output(np.zeros((1, 127), np.float32), (1, 127))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_timesfm25_mobilint.py`

Expected: FAIL because ARIES conversion helpers do not exist.

- [ ] **Step 3: Implement local ONNX/MXQ and remote execution boundaries**

Export the core with static FP32 `[1,1024]` input and `[1,128]` output, run ONNX Runtime CPU parity before QBC, and compile with `mxq_compile_V2(... target_device="aries-rb", device="cpu", cpu_offload=False, use_random_calib=False)`.  Store each normalized training sample in its own `.npy` file.  The remote runner must not assume that QBC preserves the ONNX rank; it validates element counts, scale axis length, runtime dtypes, saturation count, and output element count before comparing to the fixture.

- [ ] **Step 4: Run focused tests and local compiler smoke test**

Run:

```bash
PYTHONPATH=framework/src pytest -q framework/tests/test_timesfm25_mobilint.py
PYTHONPATH=framework/src python framework/tools/timesfm25_mobilint_calibrate.py \
  --model-path framework/models/google_timesfm-2.5-200m-transformers \
  --dataset-path framework/datasets/etth1/ETTh1.csv \
  --output-dir /home/swlab-youngjin/timesfm25-aries-$(date -u +%Y%m%dT%H%M%SZ)
```

Expected: ABI tests PASS; compiler either writes a nonempty MXQ and compile evidence or preserves a classified compiler failure.

- [ ] **Step 5: Commit**

```bash
git add framework/src/timesfm25/mobilint_aries.py framework/tools/timesfm25_mobilint_calibrate.py \
  framework/tools/timesfm25_mobilint_run.py framework/tests/test_timesfm25_mobilint.py
git commit -m "feat: add TimesFM 2.5 ARIES validation"
```

### Task 6: Add 128-step ETTh1 quality evaluation and runbooks

**Files:**
- Create: `framework/src/timesfm25/etth1_quality.py`
- Create: `framework/tools/timesfm25_rbln_etth1_quality.py`
- Create: `framework/tools/timesfm25_furiosa_etth1_quality.py`
- Create: `framework/tools/timesfm25_mobilint_etth1_quality.py`
- Create: `framework/tests/test_timesfm25_etth1_quality.py`
- Modify: `framework/docs/timesfm25-cross-vendor.md`

**Interfaces:**
- `TimesFM25ETTh1Config` fixes column `OT`, context 1,024, horizon 128, and chronological test windows.
- `evaluate_windows(...)` returns CPU/NPU MAE, RMSE, prediction delta, and saved `[windows,128]` predictions.
- Each quality tool writes an immutable result JSON and NPZ, or a result JSON with `task_quality_status="not_measured"` on failure.

- [ ] **Step 1: Write the failing no-leakage/window-shape test**

```python
def test_etth1_windows_use_1024_context_and_128_future_steps(csv_path: Path):
    contexts, targets, split = load_etth1_windows(TimesFM25ETTh1Config(csv_path, windows=2))
    assert contexts.shape == (2, 1024)
    assert targets.shape == (2, 128)
    assert split["test_start"] == 11520
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_timesfm25_etth1_quality.py`

Expected: FAIL because the 1,024/128 evaluator is absent.

- [ ] **Step 3: Implement quality evaluation from the same host/core boundary**

For each window, use the same adapter and execute the normal and flipped core input on CPU and device.  Restore point forecasts on CPU and calculate MAE/RMSE against the 128 future `OT` values.  Record strict parity status separately from task quality.  ARIES inference uses the remote MXQ runner; RBLN and Furiosa execute the in-process artifact/compiled graph.

- [ ] **Step 4: Run focused tests and all applicable vendor quality commands**

Run: `PYTHONPATH=framework/src pytest -q framework/tests/test_timesfm25_etth1_quality.py`

Then execute each vendor command documented in `framework/docs/timesfm25-cross-vendor.md` only after that vendor's strict artifact has run.  Use a new timestamped result directory each time.

- [ ] **Step 5: Run the complete TimesFM 2.5 test set and commit**

Run:

```bash
PYTHONPATH=framework/src pytest -q framework/tests/test_timesfm25_*.py
git add framework/src/timesfm25/etth1_quality.py framework/tools/timesfm25_*etth1_quality.py \
  framework/tests/test_timesfm25_etth1_quality.py framework/docs/timesfm25-cross-vendor.md
git commit -m "feat: add TimesFM 2.5 ETTh1 quality evaluation"
```

Expected: all TimesFM 2.5 unit tests PASS.  Existing full-suite collection failures outside this feature must be reported separately rather than hidden.
