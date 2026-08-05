# TTM-R1 ARIES ETTh1 Quality Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a train-calibrated ARIES MXQ locally and measure 240 ETTh1 OT forecasts on the remote device.

**Architecture:** A pure module creates 256 train-only standard-scaled calibration inputs and converts between the core ABI and runtime MXQ ABI. A local Docker-only CLI exports/quantizes an MXQ. A remote qbruntime-only CLI loads that MXQ once and compares its restored ETTh1 forecasts to CPU.

**Tech Stack:** PyTorch, NumPy, pandas, pytest, qbcompiler 1.2.0 Docker, qbruntime 1.3.2.

## Global Constraints

- Calibration uses exactly 256 evenly selected train origins in `[512,8640]`, each with only its preceding 512 OT values.
- Evaluation uses the first 240 test origins, OT, context 512, and horizon 96.
- QBC must use `target_device="aries-rb"`, `device="cpu"`, `cpu_offload=False`, `use_random_calib=False`.
- Test context/target must never appear in calibration and no fine-tuning is allowed.
- Keep compile success, runtime success, saturation, and task quality as separate result fields.

---

### Task 1: Calibration and runtime conversion primitives

**Files:**

- Modify: `framework/src/ttm_r1/etth1_quality.py`
- Create: `framework/src/ttm_r1/mobilint_aries.py`
- Modify: `framework/tests/test_ttm_r1_etth1_quality.py`
- Create: `framework/tests/test_ttm_r1_mobilint_aries.py`

**Interfaces:**

- `load_train_calibration_contexts(config, samples=256) -> tuple[torch.Tensor, dict]`
- `write_calibration_inputs(adapter, contexts, directory) -> dict`
- `quantize_core_input(core_input, artifact_shape, scale) -> tuple[np.ndarray, int]`
- `restore_artifact_output(raw_output, artifact_shape) -> np.ndarray`

- [ ] **Step 1: Write failing behavior tests**

```python
def test_train_calibration_uses_only_train_observations(tmp_path):
    csv = write_etth1_csv(tmp_path / "ETTh1.csv")
    contexts, metadata = load_train_calibration_contexts(ETTh1QualityConfig(csv), samples=3)
    assert metadata["origins"] == [512, 4576, 8640]
    assert contexts.shape == (3, 512, 1)
    assert contexts[-1, -1, 0].item() == 8639

def test_quantization_matches_aries_layout_and_reports_clipping():
    scale = SimpleNamespace(scale=0.0, is_uniform=False, scale_list=[100.0] * 64,
                            zero_point=0, is_asymmetric=False, zero_points=[])
    value, clipped = quantize_core_input(np.full((1, 512, 1), 2.0, np.float32), (1, 8, 64), scale)
    assert value.shape == (1, 8, 64)
    assert value.dtype == np.int8
    assert clipped == 512
    assert restore_artifact_output(np.zeros((1, 1, 96), np.float32), (1, 1, 96)).shape == (1, 96, 1)
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest framework/tests/test_ttm_r1_etth1_quality.py framework/tests/test_ttm_r1_mobilint_aries.py -q`

Expected: import failure for the new calibration/conversion interfaces.

- [ ] **Step 3: Implement minimal fixed behavior**

Select `numpy.linspace(512, 8640, num=samples, dtype=int)` after rejecting duplicate/out-of-range origins. Save each `TTMR1HostAdapter.prepare(...).past_values` as `calibration-000.npy` through `calibration-255.npy` and write an origins/hashes manifest. Require runtime input element count 512, reshape to its discovered ABI, apply uniform scale or a last-axis `scale_list`, round/clip to int8, and count pre-clip values beyond `[-128,127]`. Require output element count 96 and transpose `[1,1,96]` to `[1,96,1]`.

- [ ] **Step 4: Verify green and commit**

Run: `python -m pytest framework/tests/test_ttm_r1_etth1_quality.py framework/tests/test_ttm_r1_mobilint_aries.py -q`

```bash
git add framework/src/ttm_r1/etth1_quality.py framework/src/ttm_r1/mobilint_aries.py framework/tests/test_ttm_r1_etth1_quality.py framework/tests/test_ttm_r1_mobilint_aries.py
git commit -m "feat: add TTM-R1 ARIES calibration primitives"
```

### Task 2: Local calibrated MXQ compiler

**Files:**

- Create: `framework/tools/ttm_r1_mobilint_calibrate.py`
- Create: `framework/tests/test_ttm_r1_mobilint_calibrate_cli.py`

**Interfaces:**

- Required CLI paths: `--model-path`, `--dataset-path`, `--output-dir`.
- Optional `--calibration-samples` defaults to `256`.
- Writes ONNX, `calibration/`, MXQ, and `local-aries-compile-result.json`.

- [ ] **Step 1: Write failing parser and compiler-contract tests**

```python
def test_calibration_cli_defaults_to_256():
    module = runpy.run_path("framework/tools/ttm_r1_mobilint_calibrate.py", run_name="not_main")
    args = module["build_parser"]().parse_args(["--model-path", "/m", "--dataset-path", "/d", "--output-dir", "/o"])
    assert args.calibration_samples == 256

def test_mxq_compile_disables_random_calibration_and_cpu_offload(tmp_path):
    captured = {}
    compile_mxq(Path("core.onnx"), tmp_path / "calibration", tmp_path / "core.mxq",
                np.zeros((1, 512, 1), np.float32), fake_qbcompiler(captured))
    assert captured["target_device"] == "aries-rb"
    assert captured["device"] == "cpu"
    assert captured["cpu_offload"] is False
    assert captured["use_random_calib"] is False
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest framework/tests/test_ttm_r1_mobilint_calibrate_cli.py -q`

Expected: file-not-found failure.

- [ ] **Step 3: Implement local-only compiler**

Load the local official model, make `TTMR1Core` plus split host adapter, call Task 1 calibration writer, export static ONNX with `export_core_onnx`, and lazily call `qbcompiler.mxq_compile_V2` with `calib_data_path`, first NPY as `feed_dict["past_values"]`, target `aries-rb`, CPU, no CPU offload, and random calibration disabled. Reject pre-existing output paths. Persist input hashes, calibration manifest hash, options, ONNX/MXQ hashes, and `compiled_unvalidated` status.

- [ ] **Step 4: Verify green and commit**

Run: `python -m pytest framework/tests/test_ttm_r1_mobilint_calibrate_cli.py -q && python framework/tools/ttm_r1_mobilint_calibrate.py --help`

```bash
git add framework/tools/ttm_r1_mobilint_calibrate.py framework/tests/test_ttm_r1_mobilint_calibrate_cli.py
git commit -m "feat: add train-calibrated TTM-R1 ARIES compiler"
```

### Task 3: Remote ARIES quality evaluator and runbook

**Files:**

- Create: `framework/tools/ttm_r1_mobilint_etth1_quality.py`
- Create: `framework/tests/test_ttm_r1_mobilint_etth1_quality_cli.py`
- Modify: `framework/docs/ttm-r1-cross-vendor.md`

**Interfaces:**

- Required CLI paths: `--model-path`, `--dataset-path`, `--artifact`, `--output-dir`.
- Optional `--windows` defaults to `240`; `--compile-result` links compile evidence.
- Writes result JSON and CPU/ARIES/target NPZ arrays.

- [ ] **Step 1: Write failing runtime-boundary test**

```python
def test_aries_runner_quantizes_and_restores_the_runtime_abi():
    model = fake_model(input_shape=(1, 8, 64), output_shape=(1, 1, 96), scale_list=[1.0] * 64)
    runner = build_aries_runner(model)
    output, clipped = runner(np.ones((1, 512, 1), np.float32))
    assert model.received.dtype == np.int8
    assert model.received.shape == (1, 8, 64)
    assert output.shape == (1, 96, 1)
    assert clipped == 0
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest framework/tests/test_ttm_r1_mobilint_etth1_quality_cli.py -q`

Expected: file-not-found failure.

- [ ] **Step 3: Implement one-load qbruntime evaluator**

Require ARIES device 0, load MXQ once, read real shapes/dtypes/scales, and use Task 1 conversion before `infer_to_float`. Feed the shared `evaluate_prepared_windows` closure after converting its prepared tensor to NumPy; collect saturation across all 240 windows; dispose in `finally`. Save NPZ, then immutable JSON with ABI/scales summary, compile-result link, task metrics, prediction delta, degradation percent, `runtime_success`, and `quantization_status` (`saturated` iff clipping is nonzero).

- [ ] **Step 4: Document local Docker, SCP, remote execution**

Use the original wheel filename `/home/swlab-youngjin/Downloads/qbcompiler-1.2.0-py3-none-any.whl` mounted into `mobilint/qbcompiler:1.2-cpu-ubuntu22.04`; do not install qbcompiler on the ARIES server. Document transfer of MXQ and compile JSON, then remote qbruntime command and `mobilint-cli status` capture.

- [ ] **Step 5: Verify green and commit**

Run: `python -m pytest framework/tests/test_ttm_r1_etth1_quality.py framework/tests/test_ttm_r1_mobilint_aries.py framework/tests/test_ttm_r1_mobilint_calibrate_cli.py framework/tests/test_ttm_r1_mobilint_etth1_quality_cli.py -q && python framework/tools/ttm_r1_mobilint_etth1_quality.py --help`

```bash
git add framework/tools/ttm_r1_mobilint_etth1_quality.py framework/tests/test_ttm_r1_mobilint_etth1_quality_cli.py framework/docs/ttm-r1-cross-vendor.md
git commit -m "feat: add ARIES TTM-R1 ETTh1 quality evaluation"
```

### Task 4: Actual local compile and remote ARIES measurement

**Files:**

- Runtime local: `framework/results/ttm-r1/aries-etth1-calibrated-<timestamp>/`
- Runtime remote: `/home/etri_ecas/ttm-r1-aries-etth1-<timestamp>/`

- [ ] **Step 1: Compile in Docker**

```bash
OUT=framework/results/ttm-r1/aries-etth1-calibrated-$(date -u +%Y%m%dT%H%M%SZ)
test ! -e "$OUT"
docker run --rm -it -v "$PWD:/workspace/repo" -v /home/swlab-youngjin/Downloads/qbcompiler-1.2.0-py3-none-any.whl:/workspace/qbcompiler-1.2.0-py3-none-any.whl -w /workspace/repo/framework mobilint/qbcompiler:1.2-cpu-ubuntu22.04 bash -lc 'pip install --no-deps /workspace/qbcompiler-1.2.0-py3-none-any.whl granite-tsfm==0.2.27 pandas && python3 tools/ttm_r1_mobilint_calibrate.py --model-path models/ibm-granite_granite-timeseries-ttm-r1 --dataset-path datasets/etth1/ETTh1.csv --output-dir "'$OUT'" --calibration-samples 256'
```

- [ ] **Step 2: Transfer and execute on ARIES**

```bash
scp -P 20022 "$OUT/ttm-r1-core.mxq" "$OUT/local-aries-compile-result.json" etri_ecas@203.255.250.85:/home/etri_ecas/ttm-r1-aries-etth1/
```

Then run the Task 3 CLI on the remote server with the transferred artifact, `qbruntime`, and 240 windows. Confirm the saved prediction arrays all have shape `(240,96,1)` and report saturation alongside MAE/RMSE degradation.
