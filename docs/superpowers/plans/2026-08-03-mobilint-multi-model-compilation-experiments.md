# Mobilint Multi-Model Compilation Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build reproducible qbcompiler 1.2 experiments for BERT, PatchTST ETTh1, ResNet50, and YOLOv5m, execute the new recipes on the compiler server, validate generated MXQs on ARIES, and record both successes and failures.

**Architecture:** Keep the proven BERT compiler package intact and add a separate `mobilint_compile_recipes` package for shared contracts, attempt recording, and the three new recipes. A shell entrypoint creates immutable attempts and runs compiler stages in child processes; a runtime verifier extends the same result record with ARIES evidence. Generated artifacts and full logs remain outside Git, while commands, hashes, errors, and stage results are curated in one runbook.

**Tech Stack:** Python 3.10/3.12, PyTorch 2.7.1, TorchVision 0.22.1, Transformers 4.57.1, NumPy 1.26.0, qbcompiler 1.2.0, qbruntime 1.3.2, Bash, pytest

## Global Constraints

- Compiler host: Ubuntu 22.04, Intel x86-64, CPython 3.10.
- Compiler wheel: `qbcompiler-1.2.0-py3-none-any.whl`, SHA256 `28f276baef1bff86ed313cb819b53d8abb684a7555cf4c81c459edc09abf1b4b`.
- Target device: `aries-rb`.
- BERT retains the existing compiler implementation and `single`; the three new recipes use `global8`.
- Existing runtime ABI is required; byte-identical output to vendor MXQs is not required.
- Every attempt records source revision/checksum, calibration manifest, command, options, elapsed time, exit code, stage result, artifact size, and SHA256.
- A retry never overwrites its parent attempt.
- `use_random_calib` cannot produce a quality-capable compile pass.
- `.mblt`, `.mxq`, calibration arrays, weights, datasets, venvs, and full logs are never committed.
- Automated tests never count as actual qbcompiler or ARIES validation.

## File Map

- `framework/tools/mobilint_compile_recipes/contracts.py`: immutable tensor/model contracts.
- `framework/tools/mobilint_compile_recipes/compiler.py`: exact qbcompiler call boundary.
- `framework/tools/mobilint_compile_recipes/attempt.py`: attempt state, logging, elapsed time, exit/signal capture.
- `framework/tools/mobilint_compile_recipes/patchtst_etth1.py`: PatchTST stock/static variants.
- `framework/tools/mobilint_compile_recipes/resnet50.py`: uint8 NHWC ResNet recipe.
- `framework/tools/mobilint_compile_recipes/yolov5m.py`: pinned YOLO raw-head recipe.
- `framework/tools/mobilint_compile_recipes/bert_bridge.py`: existing BERT report importer.
- `framework/tools/mobilint_compile_recipes/runtime_verify.py`: ARIES verifier.
- `framework/scripts/run_mobilint_compile_experiment.sh`: bootstrap and orchestration.
- `framework/tests/test_mobilint_compile_*.py`: focused contract, attempt, recipe, and runtime tests.
- `docs/mobilint-compilation-experiments.md`: commands and observed attempt ledger.

---

### Task 1: Shared contracts and qbcompiler boundary

**Files:**
- Create: `framework/tools/mobilint_compile_recipes/__init__.py`
- Create: `framework/tools/mobilint_compile_recipes/contracts.py`
- Create: `framework/tools/mobilint_compile_recipes/compiler.py`
- Create: `framework/tests/test_mobilint_compile_contracts.py`

**Interfaces:**
- Produces: `TensorContract`, `CompileRecipe`, `get_recipe(model, variant)`, `contract_to_dict(recipe)`, `select_even_indices(total, count)`, `sha256_file(path)`, `run_mblt_compile(...)`, `run_mxq_compile(...)`.
- Constraint: importing these modules must not import qbcompiler.

- [ ] **Step 1: Write failing contract tests**

```python
def test_resnet_recipe_preserves_existing_runtime_abi():
    recipe = get_recipe("resnet50", "default")
    assert recipe.target_device == "aries-rb"
    assert recipe.inference_scheme == "global8"
    assert [(x.name, x.shape, x.dtype) for x in recipe.runtime_inputs] == [
        ("input_np", (1, 224, 224, 3), "uint8")
    ]
    assert recipe.outputs[0].shape == (1, 1000)


def test_yolov5m_recipe_preserves_three_raw_heads():
    recipe = get_recipe("yolov5m", "default")
    assert [x.shape for x in recipe.outputs] == [
        (1, 20, 20, 255),
        (1, 40, 40, 255),
        (1, 80, 80, 255),
    ]


def test_patchtst_variants_share_external_contract():
    stock = contract_to_dict(get_recipe("patchtst-etth1", "stock"))
    compat = contract_to_dict(get_recipe("patchtst-etth1", "compat-static-patchifier"))
    for key in ("target_device", "inference_scheme", "runtime_inputs", "outputs"):
        assert stock[key] == compat[key]


def test_even_indices_are_deterministic_and_include_endpoints():
    assert select_even_indices(100, 4) == (0, 33, 66, 99)
```

- [ ] **Step 2: Run RED test**

Run: `/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests/test_mobilint_compile_contracts.py -q`

Expected: collection fails because `tools.mobilint_compile_recipes` does not exist.

- [ ] **Step 3: Implement immutable contracts**

```python
@dataclass(frozen=True)
class TensorContract:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class CompileRecipe:
    model: str
    variant: str
    source_id: str
    target_device: str
    inference_scheme: str
    compiler_inputs: tuple[TensorContract, ...]
    runtime_inputs: tuple[TensorContract, ...]
    outputs: tuple[TensorContract, ...]
    calibration_samples: int = 32
    config_preset: str | None = None
    yolo_decode_include: bool | None = None
```

Register exact keys `patchtst-etth1/stock`, `patchtst-etth1/compat-static-patchifier`, `resnet50/default`, and `yolov5m/default`. Vision compiler inputs are float32 unit-range NHWC; their runtime inputs are uint8 NHWC. PatchTST compiler and runtime inputs are identical.

- [ ] **Step 4: Add failing exact-call tests**

Use fake compiler functions that capture kwargs and create the requested file. Assert `.mblt`/`.mxq` suffix enforcement, existing-path rejection, non-empty artifact enforcement, `target_device="aries-rb"`, `backend="torch"`, recipe core mode, copied `feed_dict`, explicit `CalibrationConfig`, and this vision config:

```python
assert captured["uint8_input_config"].kwargs == {
    "apply": True,
    "inputs": ["input_np"],
    "division_factor": 255.0,
}
```

- [ ] **Step 5: Implement compiler functions**

`run_mblt_compile` passes `model`, `mblt_save_path`, `target_device`, `backend`, `feed_dict`, and `cpu_offload=True`. `run_mxq_compile` passes the recipe's scheme, preset, optional YOLO decode flag and calibration path plus `CalibrationConfig(method=1, output=0, mode=1, MaxPercentile(percentile=0.999, topk_ratio=0.01))`. Construct `Uint8InputConfig` only where runtime and compiler dtypes differ.

- [ ] **Step 6: Verify and commit**

Run: `/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests/test_mobilint_compile_contracts.py -q`

Commit: `feat: add Mobilint multi-model compile contracts`

### Task 2: Immutable attempt recording

**Files:**
- Create: `framework/tools/mobilint_compile_recipes/attempt.py`
- Create: `framework/tests/test_mobilint_compile_attempt.py`

**Interfaces:**
- Produces: `STAGES`, `create_attempt(...) -> Path`, `execute_stage(attempt_root, stage, command) -> int`, `record_artifact(...)`, `record_quality_csv(...)`, `record_quality_failure(...)`, and CLI subcommands `create`, `run`, `artifact`, `quality`, `quality-failure`, `show`.
- Consumes: `sha256_file` from Task 1.

- [ ] **Step 1: Write failing real-process tests**

```python
def test_execute_stage_records_output_time_and_success(tmp_path):
    root = create_attempt(tmp_path, "fixed", "resnet50", "default", {})
    code = execute_stage(root, "SOURCE_SMOKE", [sys.executable, "-c", "print('SOURCE_OK')"])
    result = json.loads((root / "result.json").read_text())
    assert code == 0
    assert result["stages"]["SOURCE_SMOKE"]["status"] == "pass"
    assert result["stages"]["SOURCE_SMOKE"]["elapsed_seconds"] >= 0
    assert "SOURCE_OK" in (root / "compile.log").read_text()


def test_execute_stage_preserves_first_failure(tmp_path):
    root = create_attempt(tmp_path, "failed", "patchtst-etth1", "stock", {})
    code = execute_stage(root, "MBLT_COMPILE", [sys.executable, "-c", "import sys; print('bad op'); sys.exit(7)"])
    result = json.loads((root / "result.json").read_text())
    assert code == 7
    assert result["failed_at"] == "MBLT_COMPILE"
    assert result["stages"]["MBLT_COMPILE"]["status"] == "fail"
    assert result["stages"]["MXQ_COMPILE"]["status"] == "not_run"
```

Also test duplicate attempt rejection, unknown stage rejection, artifact hash recording, and that environment reports cannot contain arbitrary environment variables.

Add a quality CSV test using literal columns `total_samples`, `accuracy`, `f1`, `MSE`, `Top-1`, and `mAP@0.5`. `record_quality_csv` must retain only present allowlisted metric columns, record the CSV SHA256, and set `quality_status="pass"`; it must reject a CSV with no sample-count field. Add a separate test proving `record_quality_failure` stores the nonzero E2E exit code and log path, sets only `quality_status="fail"`, and does not rewrite `TASK_SMOKE` or compiler/runtime/contract status.

- [ ] **Step 2: Run RED test**

Run: `/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests/test_mobilint_compile_attempt.py -q`

Expected: import failure for `attempt.py`.

- [ ] **Step 3: Implement stage state and process logging**

```python
STAGES = (
    "SOURCE_PREPARE", "SOURCE_SMOKE", "CALIBRATION_PREPARE",
    "MBLT_COMPILE", "MXQ_COMPILE", "ARIES_LOAD",
    "CONTRACT_CHECK", "TASK_SMOKE",
)
```

`execute_stage` uses `subprocess.Popen` with combined stdout/stderr, writes every line to the terminal and `compile.log`, records negative return codes as a signal, and atomically replaces `result.json`. Store only allowlisted OS, architecture, Python, package, and wheel fields; never serialize `os.environ`.

`record_quality_csv` reads the final CSV row, accepts `total_samples`, `samples`, or `Total Samples` as the sample count, copies only the allowlisted task metrics, records the result CSV path and SHA256, and never changes compiler/runtime/contract stages. `record_quality_failure` records only the failed framework E2E subprocess evidence; a failed quality run must not erase a successful one-inference `TASK_SMOKE`.

- [ ] **Step 4: Verify and commit**

Run: `/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests/test_mobilint_compile_attempt.py -q`

Commit: `feat: record Mobilint compiler attempts`

### Task 3: PatchTST stock and static-patchifier recipes

**Files:**
- Create: `framework/tools/mobilint_compile_recipes/patchtst_etth1.py`
- Create: `framework/tests/test_mobilint_patchtst_compile.py`

**Interfaces:**
- Consumes: Task 1 APIs and `ETTmLoader` using `split="val"`, `split_boundaries=(8640, 11520)`, context 512, prediction 96, stride 12.
- Produces: `static_patchify`, `build_patchtst_wrapper`, `write_multi_input_calibration`, `prepare_calibration`, `source_smoke`, and `--stage describe|prepare|source-smoke|mblt|mxq`.

- [ ] **Step 1: Write failing fixed-patch and input-order tests**

```python
def test_static_patchifier_matches_unfold_layout():
    values = torch.arange(1 * 512 * 7, dtype=torch.float32).reshape(1, 512, 7)
    expected = values.unfold(1, 12, 12).permute(0, 2, 1, 3).contiguous()
    actual = static_patchify(values)
    assert actual.shape == (1, 7, 42, 12)
    torch.testing.assert_close(actual, expected)


def test_calibration_json_orders_equal_shape_inputs_by_contract(tmp_path):
    samples = [{
        "past_values": np.ones((1, 512, 7), dtype=np.float32),
        "past_observed_mask": np.ones((1, 512, 7), dtype=np.bool_),
    }]
    path = write_multi_input_calibration(samples, tmp_path)
    payload = json.loads(path.read_text())
    assert payload["info"]["input names"] == ["past_values", "past_observed_mask"]
    assert payload["calib paths"][0][0].endswith("past_values.npy")
    assert payload["calib paths"][0][1].endswith("past_observed_mask.npy")
```

Use a small fake PatchTST with a real `unfold` patchifier. Verify the compat wrapper keeps output `(1,96,7)` and CPU-equivalent values, stock never replaces the patchifier, and compat does.

- [ ] **Step 2: Run RED test**

Run: `/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests/test_mobilint_patchtst_compile.py -q`

Expected: import failure for `patchtst_etth1`.

- [ ] **Step 3: Implement deterministic preparation**

Resolve the requested Hugging Face revision to a commit SHA and record it. Load the existing `patchtst-etth1` model profile and `ETTmLoader`; select 32 evenly spaced validation windows. Write sample directories with the two named `.npy` files, an explicit ordered calibration JSON, and a separate manifest containing ETTh1 SHA256, indices, shape, dtype, normalization metadata, source ID, and resolved revision. Reject an existing task root.

- [ ] **Step 4: Implement stock and compat source smoke**

Load `PatchTSTForPrediction` at the resolved revision. Require context 512, prediction 96, seven channels, patch length 12, and stride 12. The compat variant replaces only the patchifier and casts the boolean mask to the values dtype; compare stock and compat outputs at `rtol=1e-5`, `atol=1e-6` before compiler import.

- [ ] **Step 5: Implement compiler stages**

Reload the source model in each MBLT/MXQ process, use one manifest sample as feed input, pass the ordered multi-input calibration JSON to MXQ, and write `source-manifest.json` plus `compile-report.json`. Use the Task 1 `global8` contract unchanged for both variants.

- [ ] **Step 6: Verify and commit**

Run: `/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests/test_mobilint_patchtst_compile.py framework/tests/test_mobilint_compile_contracts.py -q`

Commit: `feat: add Mobilint PatchTST compile experiments`

### Task 4: ResNet50 uint8 NHWC recipe

**Files:**
- Create: `framework/tools/mobilint_compile_recipes/resnet50.py`
- Create: `framework/tests/test_mobilint_resnet50_compile.py`

**Interfaces:**
- Consumes: Task 1 APIs and `MLPerfResNet50RawPreprocess`.
- Produces: `ResNet50SourceWrapper`, `preprocess_calibration_image`, `prepare_calibration`, `source_smoke`, and the same stage CLI.

- [ ] **Step 1: Write failing wrapper and calibration tests**

```python
def test_resnet_wrapper_normalizes_unit_nhwc_to_nchw():
    wrapper = ResNet50SourceWrapper(torch.nn.Identity())
    output = wrapper(torch.zeros((1, 224, 224, 3), dtype=torch.float32))
    assert output.shape == (1, 3, 224, 224)
    expected = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225])
    torch.testing.assert_close(output[0, :, 0, 0], expected)


def test_resnet_calibration_is_raw_rgb_uint8_nhwc():
    value = preprocess_calibration_image(Image.new("RGB", (320, 240), (10, 20, 30)))
    assert value.shape == (1, 224, 224, 3)
    assert value.dtype == np.uint8
```

Also test deterministic sorted file selection, non-image rejection, input file hashes, and source smoke output `(1,1000)` with finite float values.

- [ ] **Step 2: Run RED test**

Run: `/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests/test_mobilint_resnet50_compile.py -q`

- [ ] **Step 3: Implement wrapper and preparation**

The wrapper accepts float32 unit-range NHWC after the compiler's uint8 division, permutes NCHW, applies ImageNet mean `(0.485,0.456,0.406)` and std `(0.229,0.224,0.225)`, then calls frozen TorchVision `IMAGENET1K_V2`. Calibration uses raw resize/center-crop, transposes back to NHWC, casts uint8, and records TorchVision version, weight enum, weight file and SHA256.

- [ ] **Step 4: Implement compiler stages**

Use `config_preset="classification_torchvision"`, the Task 1 uint8 config, and `global8`. Save the resolved preset dump. Compare wrapper logits with the official weight transform path at `rtol=1e-5`, `atol=1e-6` before compile.

- [ ] **Step 5: Verify and commit**

Run: `/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests/test_mobilint_resnet50_compile.py framework/tests/test_mobilint_compile_contracts.py -q`

Commit: `feat: add Mobilint ResNet50 compile experiment`

### Task 5: YOLOv5m pinned raw-head recipe

**Files:**
- Create: `framework/tools/mobilint_compile_recipes/yolov5m.py`
- Create: `framework/tests/test_mobilint_yolov5m_compile.py`

**Interfaces:**
- Consumes: Task 1 APIs and existing Mobilint YOLOv5 letterbox constants.
- Produces: `validate_sources`, `YoloV5RawHeadWrapper`, `preprocess_calibration_image`, `source_smoke`, and the same stage CLI.

- [ ] **Step 1: Write failing source and head-contract tests**

Test wrong Git revision, missing required source files, and empty weights. Use this real tensor fake:

```python
class FakeYolo(torch.nn.Module):
    def forward(self, value):
        heads = [
            torch.zeros((1, 3, 80, 80, 85)),
            torch.zeros((1, 3, 40, 40, 85)),
            torch.zeros((1, 3, 20, 20, 85)),
        ]
        return torch.zeros((1, 25200, 85)), heads


def test_yolo_wrapper_returns_existing_head_order_and_shape():
    outputs = YoloV5RawHeadWrapper(FakeYolo())(
        torch.zeros((1, 640, 640, 3), dtype=torch.float32)
    )
    assert [tuple(x.shape) for x in outputs] == [
        (1, 20, 20, 255), (1, 40, 40, 255), (1, 80, 80, 255)
    ]
```

Add a non-square image letterbox test asserting raw RGB uint8 NHWC `(1,640,640,3)` and padding value 114.

- [ ] **Step 2: Run RED test**

Run: `/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests/test_mobilint_yolov5m_compile.py -q`

- [ ] **Step 3: Implement pinned source and raw-head wrapper**

Require commit `86fd1ab270cb2f7e53ee7412cd4a0650bf4bcc51`, `models/experimental.py`, `models/yolo.py`, and non-empty `yolov5m.pt`. Load with `attempt_load(..., map_location="cpu").fuse().eval()`. The wrapper accepts unit-range NHWC, permutes NCHW, extracts undecoded heads, sorts by ascending spatial size, and reshapes anchor/channel axes to NHWC 255. Reject decoded-only output.

- [ ] **Step 4: Implement calibration and compile stages**

Select 32 sorted COCO128 images evenly, apply the existing RGB letterbox, and save raw uint8 NHWC arrays. Use `config_preset="yolo_640"`, uint8 config, `yolo_decode_include=False`, and `global8`. Record Git revision, weight SHA256, anchors, strides, output shapes, and preset dump.

- [ ] **Step 5: Verify and commit**

Run: `/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests/test_mobilint_yolov5m_compile.py framework/tests/test_mobilint_compile_contracts.py -q`

Commit: `feat: add Mobilint YOLOv5m compile experiment`

### Task 6: BERT result bridge and experiment shell

**Files:**
- Create: `framework/tools/mobilint_compile_recipes/bert_bridge.py`
- Create: `framework/scripts/run_mobilint_compile_experiment.sh`
- Modify: `framework/tests/test_mobilint_compile_attempt.py`

**Interfaces:**
- Consumes: existing BERT calibration/compile reports and Tasks 2–5 stage CLIs.
- Produces: `import_bert_compile_result(task_root, output)` and one shell entrypoint for all model names.

- [ ] **Step 1: Write failing bridge and shell-help tests**

```python
def test_bert_bridge_does_not_infer_hardware_success(tmp_path):
    result = import_bert_compile_result(write_fake_bert_reports(tmp_path), tmp_path / "result.json")
    assert result["compile_status"] == "pass"
    assert result["runtime_status"] == "not_run"
    assert result["contract_status"] == "not_run"
    assert result["stages"]["MBLT_COMPILE"]["status"] == "pass"
    assert result["stages"]["MXQ_COMPILE"]["status"] == "pass"


def test_experiment_help_lists_every_model():
    completed = subprocess.run(
        ["bash", "framework/scripts/run_mobilint_compile_experiment.sh", "--help"],
        check=True, text=True, capture_output=True,
    )
    for name in ("bert-sst2", "bert-squad1", "patchtst-etth1", "resnet50", "yolov5m"):
        assert name in completed.stdout
```

- [ ] **Step 2: Run RED test**

Run: `/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests/test_mobilint_compile_attempt.py -q`

- [ ] **Step 3: Implement strict BERT import**

Require the existing manifest, compile report, non-empty MBLT/MXQ, and matching stored hashes. Map only compiler stages to pass. Leave runtime, contract, and quality `not_run` until actual evidence is merged.

- [ ] **Step 4: Implement shell bootstrap and orchestration**

Support `--wheel`, `--python`, `--venv`, `--model`, `--variant`, `--output-root`, `--dataset`, `--model-revision`, `--yolov5-root`, `--weights`, and `--parent-attempt`. Reuse the BERT script's exact OS/arch/Python/wheel guards, dependency versions, `ensurepip`, `pip check`, and signature check. Create the attempt before model work, run each stage through `attempt run`, stop after the first failure, and always print `ATTEMPT_ROOT` plus `EXPERIMENT_EXIT_CODE`. Never use `eval`.

- [ ] **Step 5: Verify and commit**

Run the attempt tests and `shellcheck framework/scripts/run_mobilint_compile_experiment.sh` when ShellCheck exists.

Commit: `feat: orchestrate Mobilint compiler experiments`

### Task 7: ARIES contract and smoke verifier

**Files:**
- Create: `framework/tools/mobilint_compile_recipes/runtime_verify.py`
- Create: `framework/tests/test_mobilint_compile_runtime_verify.py`

**Interfaces:**
- Consumes: attempt `result.json`, recipe contract, saved smoke inputs, generated MXQ.
- Produces: `verify_runtime(attempt_root, artifact, qbruntime_module=None)` and a CLI that updates hardware stages.

- [ ] **Step 1: Write failing complete-fake SDK tests**

The fake implements `ModelConfig`, explicit core setters, `Model`, metadata calls, `launch`, `infer`, and `dispose`.

```python
def test_runtime_verify_updates_all_hardware_stages(tmp_path):
    attempt = prepared_resnet_attempt(tmp_path)
    sdk = FakeQbRuntime(
        input_dtypes=["Uint8"], input_shapes=[(224, 224, 3)],
        output_shapes=[(1, 1, 1000)],
        outputs=[np.zeros((1, 1, 1, 1000), dtype=np.float32)],
    )
    result = verify_runtime(attempt, attempt / "mxq/resnet50.mxq", sdk)
    assert result["runtime_status"] == "pass"
    assert result["contract_status"] == "pass"
    assert result["quality_status"] == "not_run"
    assert result["stages"]["TASK_SMOKE"]["status"] == "pass"
    assert sdk.models[0].disposed is True
```

Add cases for dtype mismatch, output count mismatch, permitted singleton normalization, non-finite outputs, core setter rejection, inference exception, and disposal after failure.

- [ ] **Step 2: Run RED test**

Run: `/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests/test_mobilint_compile_runtime_verify.py -q`

- [ ] **Step 3: Implement v1.3.2 runtime validation**

Use `get_model_input_data_type`/`get_input_dtypes` compatibility, input/output shape compatibility, explicit `CoreId(Cluster0, Core0)` for single, and `set_global8_core_mode()` for global8. Launch once, infer once in contract order, validate output count/logical shape/dtype/finite values, and dispose in `finally`. Update `ARIES_LOAD`, `CONTRACT_CHECK`, `TASK_SMOKE`, `runtime_status`, and `contract_status` independently. Leave `quality_status` unchanged because one synthetic inference is not task-quality evidence.

- [ ] **Step 4: Verify and commit**

Run: `/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -m pytest framework/tests/test_mobilint_compile_runtime_verify.py -q`

Commit: `feat: verify compiled Mobilint models on ARIES`

### Task 8: Runbook, ledger, and generated-output safety

**Files:**
- Create: `docs/mobilint-compilation-experiments.md`
- Modify: `docs/mobilint-bert-compilation.md`
- Modify: `docs/mobilint-aries-transformers.md`
- Modify: `docs/mobilint-aries-troubleshooting.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: Tasks 1–7 CLI contracts.
- Produces: one canonical runbook and attempt ledger. No prose-string pytest is added.

- [ ] **Step 1: Ignore all generated roots**

Add anchored ignore rules for `mobilint-compile-attempts*/` and the new compiler venv/output names. Verify representative paths with `git check-ignore -v` and ensure source directories remain tracked.

- [ ] **Step 2: Write the runbook**

Document the fixed host/wheel, stage and final-status meanings, exact commands for all five task names and both PatchTST variants, source/revision rules, calibration selection/order, ABI/core mode, result inspection, artifact hashing, log/kernel/device inspection, and immutable retries. Initialize the observed ledger from completed BERT evidence and mark the three new models `not_run` until Task 9.

- [ ] **Step 3: Add guide links without duplicating commands**

Link the new runbook from the BERT compiler, transformer runtime, and ARIES troubleshooting introductions. Preserve historical result tables unchanged.

- [ ] **Step 4: Validate and commit**

Run a Python relative-link existence check over the four changed documents, `git diff --check`, and `git check-ignore`. Human prose is not frozen with text-presence tests.

Commit: `docs: add Mobilint compiler experiment runbook`

### Task 9: Local verification and server attempts

**Files:**
- Verify: every Task 1–8 file
- Update after actual runs: `docs/mobilint-compilation-experiments.md`

**Interfaces:**
- Consumes: completed implementation and server datasets/source weights.
- Produces: immutable PatchTST, ResNet50, and YOLOv5m compiler attempts plus ARIES results for every generated MXQ.

- [ ] **Step 1: Run focused local verification**

```bash
PY=/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python
"$PY" -m pytest framework/tests/test_mobilint_compile_contracts.py framework/tests/test_mobilint_compile_attempt.py framework/tests/test_mobilint_patchtst_compile.py framework/tests/test_mobilint_resnet50_compile.py framework/tests/test_mobilint_yolov5m_compile.py framework/tests/test_mobilint_compile_runtime_verify.py framework/tests/test_mobilint_bert_compile.py framework/tests/test_mobilint_bert_profiles.py framework/tests/test_mobilint_vision_profiles.py framework/tests/test_patchtst_etth1_profile.py -q
git diff --check origin/main...HEAD
```

Expected: all selected tests pass; diff check is silent.

- [ ] **Step 2: Push the implementation so the server can fetch it**

Run: `git push origin feat/mobilint-bert-mxq-benchmark`

- [ ] **Step 3: Create an exact detached server worktree**

```bash
REPO="$HOME/ML-HW-Benchmark-Framework"
TEST_WT="$HOME/ml-hw-mobilint-compile-pr47"
BRANCH="feat/mobilint-bert-mxq-benchmark"
git -C "$REPO" fetch origin "$BRANCH:refs/remotes/origin/$BRANCH"
git -C "$REPO" worktree add --detach "$TEST_WT" "refs/remotes/origin/$BRANCH"
git -C "$TEST_WT" log -1 --oneline
```

- [ ] **Step 4: Run PatchTST stock and capture its real exit**

```bash
cd "$TEST_WT"
ETTH1="$REPO/framework/datasets/etth1/ETTh1.csv"
PATCHTST_LOG="$HOME/mobilint-patchtst-stock-launch.log"
test -s "$ETTH1"

bash framework/scripts/run_mobilint_compile_experiment.sh \
  --wheel "$HOME/Downloads/qbcompiler-1.2.0-py3-none-any.whl" \
  --python "$(command -v python3.10)" \
  --model patchtst-etth1 \
  --variant stock \
  --dataset "$ETTH1" \
  --model-revision main \
  --output-root "$HOME/mobilint-compile-attempts" \
  |& tee "$PATCHTST_LOG"
PATCHTST_EXIT=${PIPESTATUS[0]}
echo "PATCHTST_EXIT=$PATCHTST_EXIT"

PATCHTST_ATTEMPT="$(sed -n 's/^ATTEMPT_ROOT=//p' "$PATCHTST_LOG" | tail -n 1)"
test -n "$PATCHTST_ATTEMPT"
test -s "$PATCHTST_ATTEMPT/result.json"
python3 -m json.tool "$PATCHTST_ATTEMPT/result.json"
```

If the recorded first failure is patchification or boolean-mask lowering, run `compat-static-patchifier` with the stock attempt ID in `--parent-attempt`. Preserve the stock directory.

Use the resolved commit SHA from the stock `source-manifest.json` as `--model-revision`; do not resolve `main` again:

```bash
PATCHTST_SHA="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["resolved_revision"])' "$PATCHTST_ATTEMPT/source-manifest.json")"
PATCHTST_COMPAT_LOG="$HOME/mobilint-patchtst-compat-launch.log"

bash framework/scripts/run_mobilint_compile_experiment.sh \
  --wheel "$HOME/Downloads/qbcompiler-1.2.0-py3-none-any.whl" \
  --python "$(command -v python3.10)" \
  --model patchtst-etth1 \
  --variant compat-static-patchifier \
  --dataset "$ETTH1" \
  --model-revision "$PATCHTST_SHA" \
  --parent-attempt "$PATCHTST_ATTEMPT" \
  --output-root "$HOME/mobilint-compile-attempts" \
  |& tee "$PATCHTST_COMPAT_LOG"
PATCHTST_COMPAT_EXIT=${PIPESTATUS[0]}
echo "PATCHTST_COMPAT_EXIT=$PATCHTST_COMPAT_EXIT"

PATCHTST_COMPAT_ATTEMPT="$(sed -n 's/^ATTEMPT_ROOT=//p' "$PATCHTST_COMPAT_LOG" | tail -n 1)"
test -n "$PATCHTST_COMPAT_ATTEMPT"
test -s "$PATCHTST_COMPAT_ATTEMPT/result.json"
```

Do not run this compat command for an unrelated stock failure. Diagnose that first failure and add a focused test under Task 10 instead.

- [ ] **Step 5: Run ResNet50 and YOLOv5m attempts**

Use the repository's established dataset roots and a separately pinned YOLO checkout. Stop before compilation if any prerequisite check fails:

```bash
IMAGENET="$REPO/datasets/imagenet_1k"
COCO128="$REPO/framework/datasets/coco128"
YOLO_ROOT="$HOME/mobillint/yolov5"
YOLO_WEIGHTS="$HOME/Downloads/yolov5m.pt"
YOLO_REVISION="86fd1ab270cb2f7e53ee7412cd4a0650bf4bcc51"

test -d "$IMAGENET/val"
test -d "$COCO128/images/train2017"
test -d "$YOLO_ROOT/.git"
test "$(git -C "$YOLO_ROOT" rev-parse HEAD)" = "$YOLO_REVISION"
test -s "$YOLO_WEIGHTS"

RESNET_LOG="$HOME/mobilint-resnet50-launch.log"
bash framework/scripts/run_mobilint_compile_experiment.sh \
  --wheel "$HOME/Downloads/qbcompiler-1.2.0-py3-none-any.whl" \
  --python "$(command -v python3.10)" \
  --model resnet50 \
  --variant default \
  --dataset "$IMAGENET" \
  --output-root "$HOME/mobilint-compile-attempts" \
  |& tee "$RESNET_LOG"
RESNET_EXIT=${PIPESTATUS[0]}
echo "RESNET_EXIT=$RESNET_EXIT"
RESNET_ATTEMPT="$(sed -n 's/^ATTEMPT_ROOT=//p' "$RESNET_LOG" | tail -n 1)"
test -n "$RESNET_ATTEMPT"
test -s "$RESNET_ATTEMPT/result.json"

YOLO_LOG="$HOME/mobilint-yolov5m-launch.log"
bash framework/scripts/run_mobilint_compile_experiment.sh \
  --wheel "$HOME/Downloads/qbcompiler-1.2.0-py3-none-any.whl" \
  --python "$(command -v python3.10)" \
  --model yolov5m \
  --variant default \
  --dataset "$COCO128" \
  --yolov5-root "$YOLO_ROOT" \
  --weights "$YOLO_WEIGHTS" \
  --model-revision "$YOLO_REVISION" \
  --output-root "$HOME/mobilint-compile-attempts" \
  |& tee "$YOLO_LOG"
YOLO_EXIT=${PIPESTATUS[0]}
echo "YOLO_EXIT=$YOLO_EXIT"
YOLO_ATTEMPT="$(sed -n 's/^ATTEMPT_ROOT=//p' "$YOLO_LOG" | tail -n 1)"
test -n "$YOLO_ATTEMPT"
test -s "$YOLO_ATTEMPT/result.json"
```

A non-zero code is an observed result, not permission to overwrite the attempt.

- [ ] **Step 6: Verify generated MXQs on ARIES**

Run only when `MXQ_COMPILE` is `pass`:

```bash
PY="$HOME/ML-HW-Benchmark-Framework/.venv-mobilint/bin/python"
for ATTEMPT_ROOT in \
  "$PATCHTST_ATTEMPT" \
  "${PATCHTST_COMPAT_ATTEMPT:-}" \
  "$RESNET_ATTEMPT" \
  "$YOLO_ATTEMPT"
do
  test -n "$ATTEMPT_ROOT" || continue
  MXQ_STATUS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["stages"]["MXQ_COMPILE"]["status"])' "$ATTEMPT_ROOT/result.json")"
  if [ "$MXQ_STATUS" != pass ]; then
    echo "SKIP_RUNTIME=$ATTEMPT_ROOT MXQ_COMPILE=$MXQ_STATUS"
    continue
  fi
  MXQ="$(find "$ATTEMPT_ROOT/mxq" -maxdepth 1 -type f -name '*.mxq' -print -quit)"
  test -s "$MXQ"
  PYTHONPATH="$TEST_WT/framework:$TEST_WT/framework/src" \
    "$PY" -m tools.mobilint_compile_recipes.runtime_verify \
    --attempt-root "$ATTEMPT_ROOT" \
    --artifact "$MXQ" \
    |& tee "$ATTEMPT_ROOT/runtime-verify.log"
  VERIFY_EXIT=${PIPESTATUS[0]}
  echo "VERIFY_EXIT=$VERIFY_EXIT ATTEMPT_ROOT=$ATTEMPT_ROOT"
done
mobilint-cli status
```

- [ ] **Step 7: Run framework E2E and record quality evidence**

Before returning evidence, run each framework E2E command independently after its runtime and contract checks pass. Write the CSV and E2E log inside that attempt directory:

```bash
PY="$HOME/ML-HW-Benchmark-Framework/.venv-mobilint/bin/python"
PATCHTST_E2E_ATTEMPT="${PATCHTST_COMPAT_ATTEMPT:-$PATCHTST_ATTEMPT}"
is_verified () {
  python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); raise SystemExit(0 if r["runtime_status"] == r["contract_status"] == "pass" else 1)' "$1/result.json"
}
record_quality () {
  ATTEMPT_ROOT="$1"
  E2E_EXIT="$2"
  if [ "$E2E_EXIT" -eq 0 ]; then
    PYTHONPATH="$TEST_WT/framework:$TEST_WT/framework/src" \
      "$PY" -m tools.mobilint_compile_recipes.attempt quality \
      --attempt-root "$ATTEMPT_ROOT" \
      --result-csv "$ATTEMPT_ROOT/quality.csv"
  else
    PYTHONPATH="$TEST_WT/framework:$TEST_WT/framework/src" \
      "$PY" -m tools.mobilint_compile_recipes.attempt quality-failure \
      --attempt-root "$ATTEMPT_ROOT" \
      --exit-code "$E2E_EXIT" \
      --log "$ATTEMPT_ROOT/framework-e2e.log"
  fi
}

if is_verified "$PATCHTST_E2E_ATTEMPT"; then
  PATCHTST_MXQ="$(find "$PATCHTST_E2E_ATTEMPT/mxq" -maxdepth 1 -type f -name '*.mxq' -print -quit)"
  PYTHONPATH="$TEST_WT/framework/src" "$PY" "$TEST_WT/framework/src/main.py" \
    --model patchtst-etth1 --target mobilint-aries \
    --artifact "$PATCHTST_MXQ" --dataset "$ETTH1" \
    --inference-mode e2e --batch-size 1 --warmup 2 --max-steps 64 \
    --runtime-option core_mode=global8 --no-compile \
    --results-path "$PATCHTST_E2E_ATTEMPT/quality.csv" \
    |& tee "$PATCHTST_E2E_ATTEMPT/framework-e2e.log"
  PATCHTST_E2E_EXIT=${PIPESTATUS[0]}
  record_quality "$PATCHTST_E2E_ATTEMPT" "$PATCHTST_E2E_EXIT"
else
  echo "SKIP_QUALITY=$PATCHTST_E2E_ATTEMPT"
fi

if is_verified "$RESNET_ATTEMPT"; then
  RESNET_MXQ="$(find "$RESNET_ATTEMPT/mxq" -maxdepth 1 -type f -name '*.mxq' -print -quit)"
  PYTHONPATH="$TEST_WT/framework/src" "$PY" "$TEST_WT/framework/src/main.py" \
    --model resnet50 --target mobilint-aries \
    --artifact "$RESNET_MXQ" --dataset "$IMAGENET" \
    --image-preprocess-profile auto --layout NHWC \
    --inference-mode e2e --batch-size 1 --warmup 2 --max-steps 64 \
    --runtime-option core_mode=global8 --no-compile \
    --results-path "$RESNET_ATTEMPT/quality.csv" \
    |& tee "$RESNET_ATTEMPT/framework-e2e.log"
  RESNET_E2E_EXIT=${PIPESTATUS[0]}
  record_quality "$RESNET_ATTEMPT" "$RESNET_E2E_EXIT"
else
  echo "SKIP_QUALITY=$RESNET_ATTEMPT"
fi

if is_verified "$YOLO_ATTEMPT"; then
  YOLO_MXQ="$(find "$YOLO_ATTEMPT/mxq" -maxdepth 1 -type f -name '*.mxq' -print -quit)"
  PYTHONPATH="$TEST_WT/framework/src" "$PY" "$TEST_WT/framework/src/main.py" \
    --model yolov5m --target mobilint-aries \
    --artifact "$YOLO_MXQ" --dataset "$COCO128" \
    --image-preprocess-profile auto --layout NHWC \
    --inference-mode e2e --batch-size 1 --warmup 2 --max-steps 64 \
    --runtime-option core_mode=global8 \
    --runtime-option conf_threshold=0.001 \
    --runtime-option iou_threshold=0.65 \
    --no-compile --results-path "$YOLO_ATTEMPT/quality.csv" \
    |& tee "$YOLO_ATTEMPT/framework-e2e.log"
  YOLO_E2E_EXIT=${PIPESTATUS[0]}
  record_quality "$YOLO_ATTEMPT" "$YOLO_E2E_EXIT"
else
  echo "SKIP_QUALITY=$YOLO_ATTEMPT"
fi
```

`quality` and `quality-failure` update only the independent quality result. They do not change a successful one-inference `TASK_SMOKE`.

- [ ] **Step 8: Return compact evidence**

For each attempt print the result, artifact hashes when files exist, and log tail:

```bash
for ATTEMPT_ROOT in \
  "$PATCHTST_ATTEMPT" \
  "${PATCHTST_COMPAT_ATTEMPT:-}" \
  "$RESNET_ATTEMPT" \
  "$YOLO_ATTEMPT"
do
  test -n "$ATTEMPT_ROOT" || continue
  echo "===== $ATTEMPT_ROOT ====="
  python3 -m json.tool "$ATTEMPT_ROOT/result.json"
  find "$ATTEMPT_ROOT" -type f \
    \( -name '*.mblt' -o -name '*.mxq' -o -name 'quality.csv' \) \
    -exec sha256sum {} +
  tail -n 120 "$ATTEMPT_ROOT/compile.log"
done
```

Send these outputs back; do not transfer full artifacts unless a later diagnosis explicitly needs one.

### Task 10: Record outcomes, iterate scientifically, and update PR #47

**Files:**
- Modify: affected recipe/test only when a server failure justifies a new attempt
- Modify: `docs/mobilint-compilation-experiments.md`

**Interfaces:**
- Consumes: actual server `result.json`, hashes, log excerpts, and ARIES output.
- Produces: evidence-backed commits and Korean PR text.

- [ ] **Step 1: Enter actual outcomes without inference**

For every attempt copy source revision/checksum, environment, command, elapsed time, first failed stage, artifact hashes, runtime/contract statuses, and quality sample count into the ledger. Use only `pass`, `fail`, or `not_run` as recorded.

- [ ] **Step 2: Use TDD for each observed recipe correction**

Before changing a recipe, add a focused test that reproduces the observed exception or wrong compiler argument, verify RED, implement the smallest fix, verify GREEN, commit, push, and create a child attempt with `--parent-attempt`. Keep both ledger rows.

- [ ] **Step 3: Run final verification**

```bash
PY=/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python
"$PY" -m pytest framework/tests -q
git diff --check origin/main...HEAD
git status --short --branch
git log --oneline origin/main..HEAD
```

If the full suite repeats the known unrelated Furiosa async failure, record its exact test separately and require every focused Mobilint test to pass.

- [ ] **Step 4: Push and update the Korean draft PR**

Run `git push origin feat/mobilint-bert-mxq-benchmark`, then update PR #47 with `gh pr edit 47 --body-file /tmp/pr47-body-ko.md`.

The Korean body lists the four model families, explains immutable attempts, separates compiler/runtime/contract/quality outcomes, and names every remaining failed or unverified stage. It never claims all models compiled if the ledger says otherwise.
