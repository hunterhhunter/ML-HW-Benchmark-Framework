# TTM-R1 Furiosa ETTh1 Quality Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure IBM Granite TTM-R1 zero-shot OT forecasts on ETTh1 for CPU and strict Furiosa RNGD execution, then persist comparable task-quality evidence.

**Architecture:** Add a pure ETTh1 evaluator module that owns CSV validation, standard split/window selection, prediction restoration, and metrics. Add a thin CLI that loads the existing TTM-R1 core, runs the CPU path and a one-time-compiled Furiosa path over exactly the same prepared inputs, then writes one JSON result. Keep the existing strict tensor-parity status intact rather than converting task-quality measurements into a replacement pass/fail gate.

**Tech Stack:** Python 3.12/3.10, PyTorch, pandas, existing `TTMR1HostAdapter`, `furiosa.torch`, pytest.

## Global Constraints

- Use only `ETTh1.csv` column `OT`, float32, `context_length=512`, and `prediction_length=96`.
- Evaluate the first 240 origins of test split `(train=8640, validation=2880, test=2880)` in chronological order.
- Context may precede the test boundary; target values must never enter input preparation or calibration.
- Furiosa execution must retain `fullgraph=True`, `dynamic=False`, and `eager_fallback=False`.
- Dataset and checkpoint must already exist locally; neither evaluator nor CLI may download them.
- Persist CPU task metrics, RNGD task metrics, CPU↔RNGD prediction deltas, MAE/RMSE degradation percentage, hashes, contracts, and distinct runtime/parity/task statuses.
- `task_quality_status` is `measured`, not a threshold-derived pass/fail label.

---

### Task 1: Pure ETTh1 windowing and metric module

**Files:**
- Create: `framework/src/ttm_r1/etth1_quality.py`
- Test: `framework/tests/test_ttm_r1_etth1_quality.py`

**Interfaces:**
- Produces `ETTh1QualityConfig(dataset_path: Path, column: str = "OT", context_length: int = 512, prediction_length: int = 96, windows: int = 240)`.
- Produces `load_etth1_windows(config: ETTh1QualityConfig) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | str]]`, returning contexts `[windows,512,1]`, targets `[windows,96,1]`, and split metadata.
- Produces `forecast_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]` with `mae` and `rmse`.
- Produces `prediction_delta_metrics(cpu: torch.Tensor, rngd: torch.Tensor) -> dict[str, float]` with `mae`, `rmse`, and `max_abs_error`.
- Produces `percentage_degradation(cpu_metric: float, rngd_metric: float) -> float | None`.

- [x] **Step 1: Write the failing window and metric tests**

```python
def test_load_etth1_windows_uses_first_test_origins_and_only_past_context(tmp_path):
    csv_path = tmp_path / "ETTh1.csv"
    values = list(range(8640 + 2880 + 2880))
    pd.DataFrame({"date": range(len(values)), "OT": values}).to_csv(csv_path, index=False)

    contexts, targets, split = load_etth1_windows(
        ETTh1QualityConfig(csv_path, windows=2)
    )

    assert split == {"train": 8640, "validation": 2880, "test": 2880,
                     "test_start": 11520, "windows": 2}
    assert contexts.shape == (2, 512, 1)
    assert targets.shape == (2, 96, 1)
    assert contexts[0, -1, 0].item() == 11519
    assert targets[0, 0, 0].item() == 11520
    assert targets[1, 0, 0].item() == 11521

def test_metrics_and_degradation_are_deterministic():
    cpu = torch.tensor([[[1.0], [3.0]]])
    rngd = torch.tensor([[[2.0], [5.0]]])
    target = torch.tensor([[[0.0], [1.0]]])

    assert forecast_metrics(cpu, target) == {"mae": 1.5, "rmse": 1.5811388300841898}
    assert prediction_delta_metrics(cpu, rngd) == {
        "mae": 1.5, "rmse": 1.5811388300841898, "max_abs_error": 2.0
    }
    assert percentage_degradation(2.0, 2.5) == 25.0
    assert percentage_degradation(0.0, 1.0) is None
```

- [x] **Step 2: Run the tests to verify failure**

Run: `python -m pytest framework/tests/test_ttm_r1_etth1_quality.py -q`

Expected: FAIL because `ttm_r1.etth1_quality` does not exist.

- [x] **Step 3: Implement the minimal, strict data module**

```python
_TRAIN, _VALIDATION, _TEST = 8640, 2880, 2880

def load_etth1_windows(config: ETTh1QualityConfig):
    if not config.dataset_path.is_file():
        raise ValueError(f"ETTh1 CSV is missing: {config.dataset_path}")
    frame = pandas.read_csv(config.dataset_path)
    if config.column not in frame:
        raise ValueError(f"ETTh1 CSV has no {config.column!r} column")
    values = torch.tensor(frame[config.column].to_numpy(), dtype=torch.float32)
    test_start = _TRAIN + _VALIDATION
    required = test_start + config.windows + config.prediction_length
    if values.numel() < required:
        raise ValueError("ETTh1 CSV does not contain enough requested test windows")
    starts = range(test_start, test_start + config.windows)
    contexts = torch.stack([values[i-config.context_length:i] for i in starts]).unsqueeze(-1)
    targets = torch.stack([values[i:i+config.prediction_length] for i in starts]).unsqueeze(-1)
    return contexts, targets, {"train": _TRAIN, "validation": _VALIDATION,
        "test": _TEST, "test_start": test_start, "windows": config.windows}
```

Validate all produced tensors are finite and exact expected shapes before returning. Implement metrics using float32 absolute differences and `torch.sqrt(torch.mean(delta.square()))`.

- [x] **Step 4: Run the focused tests**

Run: `python -m pytest framework/tests/test_ttm_r1_etth1_quality.py -q`

Expected: PASS.

- [x] **Step 5: Commit the independently testable module**

```bash
git add framework/src/ttm_r1/etth1_quality.py framework/tests/test_ttm_r1_etth1_quality.py
git commit -m "feat: add TTM-R1 ETTh1 quality primitives"
```

### Task 2: CPU and injectable strict-Furiosa evaluation runner

**Files:**
- Modify: `framework/src/ttm_r1/etth1_quality.py`
- Test: `framework/tests/test_ttm_r1_etth1_quality.py`

**Interfaces:**
- Produces `evaluate_prepared_windows(cpu_core, adapter, contexts, targets, device_runner) -> dict[str, object]`.
- `device_runner` has signature `Callable[[tuple[torch.Tensor]], torch.Tensor]`, closes over a separately compiled Furiosa core, and receives a prepared `[1,512,1]` input.
- Returns CPU and RNGD restored predictions `[windows,96,1]` plus the three metric groups.

- [x] **Step 1: Write a failing same-input/restoration test with an injected runner**

```python
def test_evaluator_restores_cpu_and_device_forecasts_from_identical_prepared_inputs():
    seen = []

    def device_runner(inputs):
        seen.append(inputs[0].clone())
        return inputs[0][:, -96:, :]

    result = evaluate_prepared_windows(
        cpu_core=_Last96Core(), adapter=TTMR1HostAdapter(),
        contexts=torch.arange(2 * 512, dtype=torch.float32).reshape(2, 512, 1),
        targets=torch.zeros((2, 96, 1), dtype=torch.float32),
        device_runner=device_runner,
    )

    assert len(seen) == 2
    assert result["cpu_predictions"].shape == (2, 96, 1)
    assert torch.equal(result["cpu_predictions"], result["rngd_predictions"])
    assert result["prediction_delta"] == {"mae": 0.0, "rmse": 0.0, "max_abs_error": 0.0}
```

- [x] **Step 2: Run the single test to verify failure**

Run: `python -m pytest framework/tests/test_ttm_r1_etth1_quality.py::test_evaluator_restores_cpu_and_device_forecasts_from_identical_prepared_inputs -q`

Expected: FAIL because `evaluate_prepared_windows` is absent.

- [x] **Step 3: Implement sequential prepared-window evaluation**

```python
def evaluate_prepared_windows(cpu_core, adapter, contexts, targets, device_runner):
    cpu_predictions, rngd_predictions = [], []
    for context in contexts:
        prepared = adapter.prepare(context.unsqueeze(0))
        with torch.inference_mode():
            cpu_output = cpu_core(prepared.past_values)
            rngd_core = device_runner((prepared.past_values,))
        cpu_predictions.append(prepared.restore(cpu_output).squeeze(0).cpu())
        rngd_predictions.append(prepared.restore(rngd_core.detach().cpu()).squeeze(0))
    cpu = torch.stack(cpu_predictions)
    rngd = torch.stack(rngd_predictions)
    return {"cpu_predictions": cpu, "rngd_predictions": rngd,
            "cpu_task": forecast_metrics(cpu, targets),
            "rngd_task": forecast_metrics(rngd, targets),
            "prediction_delta": prediction_delta_metrics(cpu, rngd)}
```

Before metric computation, reject non-finite inputs, outputs, or targets. Keep the runner injectable; do not import Furiosa in this module.

- [x] **Step 4: Run all quality tests**

Run: `python -m pytest framework/tests/test_ttm_r1_etth1_quality.py -q`

Expected: PASS.

- [x] **Step 5: Commit the evaluator**

```bash
git add framework/src/ttm_r1/etth1_quality.py framework/tests/test_ttm_r1_etth1_quality.py
git commit -m "feat: evaluate TTM-R1 ETTh1 CPU and device quality"
```

### Task 3: Furiosa one-time compilation and evidence CLI

**Files:**
- Create: `framework/tools/ttm_r1_furiosa_etth1_quality.py`
- Create: `framework/tests/test_ttm_r1_furiosa_etth1_quality_cli.py`
- Modify: `framework/docs/ttm-r1-cross-vendor.md`

**Interfaces:**
- CLI arguments: required `--model-path PATH`, `--dataset-path PATH`, `--output-dir PATH`; optional `--windows INT` defaulting to `240` and `--strict-parity-result PATH`.
- CLI writes `furiosa-etth1-quality-result.json` and `furiosa-etth1-quality-predictions.npz` beneath `--output-dir`.
- Result fields: `status`, `vendor`, `dataset`, `contract`, `compile_mode`, `runtime_success`, `strict_parity_status`, `task_quality_status`, `cpu_task`, `rngd_task`, `prediction_delta`, and `degradation_percent`.

- [x] **Step 1: Write a failing CLI contract test**

```python
def test_quality_cli_parser_requires_local_model_dataset_and_output_paths(monkeypatch):
    module = runpy.run_path("tools/ttm_r1_furiosa_etth1_quality.py", run_name="not_main")
    parser = module["build_parser"]()
    args = parser.parse_args([
        "--model-path", "/models/ttm", "--dataset-path", "/data/ETTh1.csv",
        "--output-dir", "/results/out"
    ])
    assert args.windows == 240
```

- [x] **Step 2: Run the test to verify failure**

Run: `python -m pytest framework/tests/test_ttm_r1_furiosa_etth1_quality_cli.py -q`

Expected: FAIL because the CLI file does not exist.

- [x] **Step 3: Implement the CLI with a compile-once runner**

Implement a `compile_furiosa_runner(core, contract)` helper in the tool. It must deep-copy the supplied CPU core before moving the copy to `furiosa:0`; construct the existing `CompilerConfig(TacticHintConfig.Default)`; call `furiosa.torch.backend.with_config(..., eager_fallback=False)`; call `torch.compile(device_core.eval().to("furiosa:0"), backend=..., fullgraph=True, dynamic=False)` once; and return a closure that moves its one input to `furiosa:0`, executes it under `torch.inference_mode()`, validates `[1,96,1]` float32 finite output, and returns it. The CLI must load one CPU `TTMR1Core(load_ttm_r1_model(...))`, create `TTMR1HostAdapter(split_ttm_scaler=True)`, call `load_etth1_windows`, compile a copy through `compile_furiosa_runner`, then call `evaluate_prepared_windows`.

Write predictions with:

```python
numpy.savez_compressed(
    output_dir / "furiosa-etth1-quality-predictions.npz",
    cpu_predictions=result["cpu_predictions"].numpy(),
    rngd_predictions=result["rngd_predictions"].numpy(),
    targets=targets.numpy(),
)
```

Compute `degradation_percent` from the two task metrics. Read the existing strict result only when an explicit `--strict-parity-result PATH` is supplied; otherwise store `"unknown"`, never infer strict parity success from task metrics. Hash dataset and manifest/weights where available with `hashlib.sha256` streamed in 1 MiB chunks. Catch exceptions only to serialize `{ "status": "failed", "error": {"type": ..., "message": ...} }` before re-raising.

- [x] **Step 4: Add the runbook section**

Add a Furiosa section that prepares ETTh1 without downloading in the evaluator, runs focused tests, uses a new timestamped output directory, invokes the CLI, and pretty-prints the result:

```bash
DATASET=datasets/etth1/ETTh1.csv
OUT=results/ttm-r1/furiosa-etth1-$(date -u +%Y%m%dT%H%M%SZ)
"$FURIOSA_PY" datasets/prepare_etth1.py --output-dir datasets/etth1
"$FURIOSA_PY" -m pytest tests/test_ttm_r1_etth1_quality.py tests/test_ttm_r1_furiosa_etth1_quality_cli.py -q
STRICT=results/ttm-r1/furiosa-rerun/furiosa-result.json
"$FURIOSA_PY" tools/ttm_r1_furiosa_etth1_quality.py --model-path "$MODEL" --dataset-path "$DATASET" --output-dir "$OUT" --windows 240 --strict-parity-result "$STRICT"
"$FURIOSA_PY" -m json.tool "$OUT/furiosa-etth1-quality-result.json"
```

- [x] **Step 5: Run unit tests and static CLI help**

Run: `python -m pytest framework/tests/test_ttm_r1_etth1_quality.py framework/tests/test_ttm_r1_furiosa_etth1_quality_cli.py -q && python framework/tools/ttm_r1_furiosa_etth1_quality.py --help`

Expected: tests PASS and help lists the three required paths plus `--windows` and `--strict-parity-result`.

- [x] **Step 6: Commit the CLI and runbook**

```bash
git add framework/tools/ttm_r1_furiosa_etth1_quality.py framework/tests/test_ttm_r1_furiosa_etth1_quality_cli.py framework/docs/ttm-r1-cross-vendor.md
git commit -m "feat: add Furiosa TTM-R1 ETTh1 quality evaluation"
```

### Task 4: Actual RNGD quality measurement and evidence review

**Files:**
- Create at runtime: `framework/results/ttm-r1/furiosa-etth1-<timestamp>/furiosa-etth1-quality-result.json`
- Create at runtime: `framework/results/ttm-r1/furiosa-etth1-<timestamp>/furiosa-etth1-quality-predictions.npz`

**Interfaces:**
- Consumes the CLI from Task 3 and the existing remote Furiosa environment.
- Produces one immutable run directory for the user to inspect; do not add runtime artifacts to git.

- [ ] **Step 1: Verify remote prerequisites**

Run on the Furiosa server:

```bash
test -d "$MODEL"
test -f datasets/etth1/ETTh1.csv || "$FURIOSA_PY" datasets/prepare_etth1.py --output-dir datasets/etth1
furiosa-smi status
```

Expected: local model directory, ETTh1 CSV, and live `rngd npu0`.

- [ ] **Step 2: Run the immutable quality measurement**

```bash
OUT=results/ttm-r1/furiosa-etth1-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
furiosa-smi status > "$OUT/furiosa-before.txt"
STRICT=results/ttm-r1/furiosa-rerun/furiosa-result.json
"$FURIOSA_PY" tools/ttm_r1_furiosa_etth1_quality.py --model-path "$MODEL" --dataset-path datasets/etth1/ETTh1.csv --output-dir "$OUT" --windows 240 --strict-parity-result "$STRICT"
furiosa-smi status > "$OUT/furiosa-after.txt"
"$FURIOSA_PY" -m json.tool "$OUT/furiosa-etth1-quality-result.json"
```

Expected: `runtime_success: true`, `task_quality_status: "measured"`, and recorded CPU/RNGD metrics. `strict_parity_status` may remain `parity_failed`.

- [ ] **Step 3: Verify saved predictions agree with summary metrics**

```bash
"$FURIOSA_PY" - <<PY
import json, numpy as np
from pathlib import Path
root = Path("$OUT")
result = json.loads((root / "furiosa-etth1-quality-result.json").read_text())
data = np.load(root / "furiosa-etth1-quality-predictions.npz")
print(data["cpu_predictions"].shape, data["rngd_predictions"].shape, data["targets"].shape)
print(result["cpu_task"], result["rngd_task"], result["degradation_percent"])
PY
```

Expected: all prediction arrays have shape `(240, 96, 1)` and the printed metric fields are present.

- [ ] **Step 4: Record the interpretation without changing the strict result**

State in the run result/report that the model has an actual RNGD task-quality measurement, while strict device parity retains its independently observed status. Do not label the model cross-vendor verified until ARIES calibration and all three vendor quality evaluations are complete.
