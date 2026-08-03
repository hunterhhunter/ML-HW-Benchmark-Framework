# Furiosa RNGD Compile Failure Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a server-runnable, strict Furiosa Torch reproduction tool and a canonical document that records why ResNet50, YOLOv5m, and PatchTST-FM-r1 did not complete RNGD compilation.

**Architecture:** Keep model construction and strict first-call execution in a lazy-import core module, with a thin CLI that always runs each model case in an isolated subprocess. The parent process streams combined stdout/stderr into a text log, classifies stable historical signatures, and writes a JSON report even when the child compiler process fails. Existing supported BERT adapters remain unchanged.

**Tech Stack:** Python 3.12, PyTorch 2.10.0, Furiosa Torch 2026.3.0, TorchVision, Ultralytics 8.3.216, granite-tsfm 0.3.6, pytest 9, Markdown, Bash

## Global Constraints

- Keep `eager_fallback=False`, `fullgraph=True`, and `dynamic=False` for every RNGD compile attempt.
- A `torch.compile()` return is not success; the compiled callable must return its first output.
- Do not register ResNet50, YOLOv5m, or PatchTST in the supported `furiosa-rngd-torch` runtime.
- Do not commit model weights, datasets, FXB files, or generated reproduction logs.
- A matched historical failure still exits non-zero; `matched_known_signature` describes evidence, not success.
- Import model and Furiosa packages lazily so `--help` and pure unit tests run without vendor SDK packages.
- Run every requested case in a separate subprocess so one Rust panic cannot contaminate another case.
- Preserve the supported BERT and Furiosa-LLM behavior without modification.

---

## File map

- Create `framework/tools/furiosa_compile_repro.py`: pure result contracts, signature classification, environment capture, model loaders, CPU validation, and strict RNGD first-call execution.
- Create `framework/tools/reproduce_furiosa_compile_failures.py`: CLI, child-process isolation, terminal/log streaming, and final JSON assembly.
- Create `framework/tests/test_furiosa_compile_repro.py`: SDK-free unit tests for strict invocation, signatures, subprocess reporting, and documentation contracts.
- Create `docs/furiosa-rngd-compilation-troubleshooting.md`: canonical reproduction commands and evidence ledger.
- Modify `docs/furiosa-rngd-setup.md`: link the canonical document and classify the three failures as reproduced.
- Modify `docs/furiosa-rngd-troubleshooting.md`: replace stale `미검증` rows and link exact evidence.
- Modify `.gitignore`: exclude `framework/results/furiosa-compile-repro/`.

---

### Task 1: Pure result contracts and known-signature classification

**Files:**
- Create: `framework/tools/furiosa_compile_repro.py`
- Test: `framework/tests/test_furiosa_compile_repro.py`

**Interfaces:**
- Produces: `StageResult`, `CaseResult`, `KNOWN_SIGNATURES`, `match_known_signature(text: str) -> str | None`, `safe_error_line(exc: BaseException) -> str`, and `write_json(path: Path, payload: Mapping[str, object]) -> None`
- Consumes: no vendor SDK packages at import time

- [ ] **Step 1: Write failing tests for stable signature selection and sanitized errors**

```python
from tools import furiosa_compile_repro as repro


def test_known_signature_prefers_specific_compiler_message():
    text = """
    furiosa.UnsupportedOpError: failed to compile the graph
    EdgeIndex(162) has empty transition cost table
    """
    assert repro.match_known_signature(text) == (
        "EdgeIndex(162) has empty transition cost table"
    )


def test_safe_error_line_keeps_only_exception_type_and_first_line():
    exc = RuntimeError("first line\nsecret prompt and path")
    assert repro.safe_error_line(exc) == "RuntimeError: first line"
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
cd framework
python -m pytest tests/test_furiosa_compile_repro.py \
  -k 'known_signature or safe_error_line' -q
```

Expected: collection fails because `tools.furiosa_compile_repro` does not exist.

- [ ] **Step 3: Implement immutable result contracts and ordered signatures**

Add these public contracts and ordered signature data:

```python
@dataclass(frozen=True)
class StageResult:
    name: str
    status: str
    detail: str | None = None


@dataclass(frozen=True)
class CaseResult:
    case: str
    status: str
    stages: tuple[StageResult, ...]
    error_type: str | None = None
    error_line: str | None = None
    matched_known_signature: str | None = None


KNOWN_SIGNATURES = (
    "align_up_required (true) != false (false)",
    "EinsumByDpe should be given only a single pass",
    "called `Option::unwrap()` on a `None` value",
    "EdgeIndex(162) has empty transition cost table",
    "mutable op violation",
    "aten._native_batch_norm_legit",
    "Tensor device mismatch! Expected: furiosa:0, Got: cpu",
    "Cannot view a tensor with shape torch.Size([7, 512, 16, 64])",
)
```

`match_known_signature()` scans the signature tuple before considering generic `UnsupportedOpError` text. `safe_error_line()` returns only `<type>: <first non-empty line>`, truncated to 500 characters. `write_json()` creates its parent directory and uses `dataclasses.asdict()` through a `_json_default()` helper for dataclasses, `Path`, tuple, and package-version values.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
cd framework
python -m pytest tests/test_furiosa_compile_repro.py \
  -k 'known_signature or safe_error_line' -q
```

Expected: 2 passed.

- [ ] **Step 5: Commit the pure contracts**

```bash
git add framework/tools/furiosa_compile_repro.py \
  framework/tests/test_furiosa_compile_repro.py
git commit -m "test: Furiosa 컴파일 오류 서명 계약 추가"
```

---

### Task 2: Strict case execution and lazy model loaders

**Files:**
- Modify: `framework/tools/furiosa_compile_repro.py`
- Modify: `framework/tests/test_furiosa_compile_repro.py`

**Interfaces:**
- Produces: `CaseConfig`, `CaseDefinition`, `CASE_DEFINITIONS`, `run_case(config: CaseConfig, *, dependencies: Dependencies | None = None, emit: Callable[[str], None] = print) -> CaseResult`
- Consumes: `StageResult`, `CaseResult`, `safe_error_line()` from Task 1

- [ ] **Step 1: Write failing tests for stage order and strict backend configuration**

Use fake dependencies whose tensors and models only record calls. Assert this exact order:

```python
assert events == [
    "load:resnet50",
    "cpu:first_call",
    "device:furiosa:0",
    "model_to:furiosa:0",
    "input_to:furiosa:0",
    "backend:eager_fallback=False",
    "compile:fullgraph=True,dynamic=False",
    "rngd:first_call",
]
assert [stage.status for stage in result.stages] == [
    "passed", "passed", "passed", "passed"
]
```

Add a second test that raises `RuntimeError("compiler panic")` from the compiled first call and asserts:

```python
assert result.status == "failed"
assert result.error_type == "RuntimeError"
assert result.stages[-1].name == "rngd_first_inference"
assert result.stages[-1].status == "failed"
```

- [ ] **Step 2: Run the strict-execution tests and verify RED**

Run:

```bash
cd framework
python -m pytest tests/test_furiosa_compile_repro.py \
  -k 'stage_order or compiler_failure' -q
```

Expected: FAIL because `CaseConfig` and `run_case` do not exist.

- [ ] **Step 3: Implement case definitions and shared strict execution**

Use these contracts:

```python
@dataclass(frozen=True)
class CaseConfig:
    case: str
    model_path: Path | None
    device: str = "furiosa:0"
    seed: int = 0


@dataclass(frozen=True)
class CaseDefinition:
    expected_shapes: tuple[tuple[int, ...], ...]
    loader: Callable[[CaseConfig, object], tuple[object, tuple[object, ...]]]
```

`run_case()` must:

1. load the case and mark `model_load`;
2. run CPU inference under `torch.inference_mode()` and validate output shapes and finite floating values;
3. move the model and all inputs to `config.device`;
4. create `CompilerConfig(tactic_hint=TacticHintConfig.Default)` and the Furiosa backend with `eager_fallback=False`;
5. call `torch.compile(..., fullgraph=True, dynamic=False)`;
6. call the compiled model once and validate outputs;
7. return a failed `CaseResult` instead of swallowing the original exception.

Add `_load_dependencies()` with only function-local imports of `torch`, `furiosa.torch`, `CompilerConfig`, and `TacticHintConfig`.

- [ ] **Step 4: Implement the ResNet50 diagnostic loader**

Use TorchVision `resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`. Recursively replace adjacent eval `Conv2d`/`BatchNorm2d` child pairs with `torch.nn.utils.fusion.fuse_conv_bn_eval()` and `torch.nn.Identity()` without fusing or reusing ReLU modules. Generate one seeded `(1, 3, 224, 224)` FP32 input. Expected output is `(1, 1000)`.

- [ ] **Step 5: Implement the YOLOv5m diagnostic loader**

Require an existing file named exactly `yolov5mu.pt`, then load it with `ultralytics.YOLO`, call `yolo.fuse()`, wrap `yolo.model.eval()`, and return only element zero when the model produces a tuple/list. Generate one `(1, 3, 640, 640)` FP32 zero input. Expected output is `(1, 84, 8400)`.

- [ ] **Step 6: Implement the PatchTST-FM-r1 diagnostic loader**

Require an existing local model directory. Import `PatchTSTFMForPrediction` and `modeling_patchtst_fm` from `tsfm_public.models.patchtst_fm`, add only `modeling_patchtst_fm.logger.info` to `torch._dynamo.config.ignore_logger_methods`, and load with `local_files_only=True`. The wrapper passes `prediction_length=96`, `return_dict=True` and returns `prediction_outputs`. Inputs are FP32 zeros `(1, 512, 7)` and bool ones `(1, 512, 7)`. Expected output is `(1, 96, 7)`.

- [ ] **Step 7: Run all core tests and verify GREEN**

Run:

```bash
cd framework
python -m pytest tests/test_furiosa_compile_repro.py -q
```

Expected: all Task 1-2 tests pass without importing Furiosa, Ultralytics, TorchVision, or TSFM at test collection.

- [ ] **Step 8: Commit strict model case execution**

```bash
git add framework/tools/furiosa_compile_repro.py \
  framework/tests/test_furiosa_compile_repro.py
git commit -m "feat: Furiosa 실패 모델 strict 재현 코어 추가"
```

---

### Task 3: Subprocess-isolated CLI and durable reports

**Files:**
- Create: `framework/tools/reproduce_furiosa_compile_failures.py`
- Modify: `framework/tests/test_furiosa_compile_repro.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `build_parser()`, `run_parent(args) -> int`, `run_child(args) -> int`, and executable CLI `python tools/reproduce_furiosa_compile_failures.py`
- Consumes: `CaseConfig`, `CaseResult`, `run_case()`, `match_known_signature()`, `write_json()` from Tasks 1-2

- [ ] **Step 1: Write failing CLI tests**

Assert parser defaults and explicit model path behavior:

```python
args = cli.build_parser().parse_args(["--case", "yolov5m"])
assert args.device == "furiosa:0"
assert args.output_dir == Path("results/furiosa-compile-repro")
assert args.yolov5_path == Path("models/yolov5m/yolov5mu.pt")
```

Add a parent-runner test with an injected fake `Popen` that emits the known YOLO panic and returns 1. Assert that the text log contains the raw line and the JSON report contains:

```python
assert report["status"] == "failed"
assert report["matched_known_signature"] == (
    "EdgeIndex(162) has empty transition cost table"
)
assert report["exit_code"] == 1
```

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```bash
cd framework
python -m pytest tests/test_furiosa_compile_repro.py \
  -k 'parser or parent_runner' -q
```

Expected: FAIL because the CLI module does not exist.

- [ ] **Step 3: Implement parser and hidden child mode**

Parser options:

```text
--case {resnet50,yolov5m,patchtst,all}    required
--device DEVICE                           default furiosa:0
--output-dir PATH                         default results/furiosa-compile-repro
--yolov5-path PATH                        default models/yolov5m/yolov5mu.pt
--patchtst-path PATH                      default models/ibm-research_patchtst-fm-r1
--seed INTEGER                            default 0
--_child                                  hidden boolean
--_child-result PATH                      hidden path
```

`run_child()` builds the selected `CaseConfig`, invokes `run_case()`, writes the child JSON even on failure, prints the final case status, and returns 0 only for `passed`.

- [ ] **Step 4: Implement parent subprocess isolation and streaming logs**

For each selected case, create timestamped paths:

```text
<output-dir>/<timestamp>-<case>.log
<output-dir>/<timestamp>-<case>.child.json
<output-dir>/<timestamp>-<case>.json
```

Invoke the current Python executable and current script with `--_child`. Use `subprocess.Popen(..., stdout=PIPE, stderr=STDOUT, text=True, bufsize=1)`; for each line, write to both terminal and the log. After exit, load the child JSON when present, scan the complete log with `match_known_signature()`, and write the final JSON including `exit_code`, `log_path`, `child_result_path`, and `matched_known_signature`.

`--case all` runs `resnet50`, `yolov5m`, and `patchtst` sequentially as three distinct child processes and returns 1 if any case fails.

- [ ] **Step 5: Ignore generated reproduction output**

Add exactly:

```gitignore
framework/results/furiosa-compile-repro/
```

- [ ] **Step 6: Run CLI tests and verify GREEN**

Run:

```bash
cd framework
python -m pytest tests/test_furiosa_compile_repro.py -q
python tools/reproduce_furiosa_compile_failures.py --help
```

Expected: tests pass; help lists the four public case choices and does not import Furiosa SDK.

- [ ] **Step 7: Commit the CLI**

```bash
git add .gitignore \
  framework/tools/reproduce_furiosa_compile_failures.py \
  framework/tests/test_furiosa_compile_repro.py
git commit -m "feat: Furiosa 컴파일 실패 재현 CLI 추가"
```

---

### Task 4: Canonical troubleshooting evidence and server commands

**Files:**
- Create: `docs/furiosa-rngd-compilation-troubleshooting.md`
- Modify: `docs/furiosa-rngd-setup.md`
- Modify: `docs/furiosa-rngd-troubleshooting.md`
- Modify: `framework/tests/test_furiosa_compile_repro.py`

**Interfaces:**
- Produces: one authoritative evidence table and copy-paste RNGD commands for each case
- Consumes: the CLI paths and signature names from Tasks 1-3

- [ ] **Step 1: Write failing documentation contract tests**

Add assertions that the canonical document contains:

```python
required = (
    "reproduce_furiosa_compile_failures.py",
    "align_up_required (true) != false (false)",
    "EdgeIndex(162) has empty transition cost table",
    "Cannot view a tensor with shape",
    "eager_fallback=False",
    "fullgraph=True",
    "dynamic=False",
    "실패 재현 완료",
)
for text in required:
    assert text in runbook
```

Also assert `docs/furiosa-rngd-setup.md` and
`docs/furiosa-rngd-troubleshooting.md` link to
`furiosa-rngd-compilation-troubleshooting.md` and no longer classify YOLOv5m or PatchTST as `미검증` in the Furiosa support table.

- [ ] **Step 2: Run documentation tests and verify RED**

Run:

```bash
cd framework
python -m pytest tests/test_furiosa_compile_repro.py \
  -k 'documentation' -q
```

Expected: FAIL because the canonical document does not exist.

- [ ] **Step 3: Write the canonical evidence document**

Include these sections:

1. purpose and strict success criteria;
2. verified server matrix: Ubuntu 22.04.5, kernel 6.8.0-124, driver 2026.3.0, firmware 1.11.0, Python 3.12.13, PyTorch 2.10.0+cpu, Furiosa Torch 2026.3.0, Transformers 5.1.0, Ultralytics 8.3.216;
3. one support table separating CPU forward, graph normalization, RNGD first call, and classification;
4. prerequisite/model download commands;
5. three individual CLI commands plus `--case all`;
6. exact ResNet50, YOLOv5m, and PatchTST error signatures and source locations from the archived logs;
7. root-cause boundary and why each workaround changed only one stage;
8. generated log/JSON interpretation;
9. SDK-upgrade retest checklist.

State explicitly that the current development host lacks RNGD and Furiosa SDK, so the committed tool is locally unit-tested while fresh hardware results must be generated on the connected server.

- [ ] **Step 4: Correct stale setup and troubleshooting status**

In the setup document, link the new canonical document immediately after the unsupported-model sentence. In the older troubleshooting support matrix, change:

```text
YOLOv5m: 미검증 -> 실패 재현 완료
PatchTST: 미검증 -> 실패 재현 완료
```

Keep their overall support result as unsupported, and point readers to the canonical document for exact logs.

- [ ] **Step 5: Run documentation tests and verify GREEN**

Run:

```bash
cd framework
python -m pytest tests/test_furiosa_compile_repro.py -q
```

Expected: all reproduction tests pass.

- [ ] **Step 6: Validate every shell block in the canonical document**

Extract each `bash` block to a temporary directory and run `bash -n`:

```bash
CHECK_DIR=$(mktemp -d /tmp/furiosa-compile-doc.XXXXXX)
awk -v output_dir="$CHECK_DIR" '
  /^```bash$/ {
    in_block = 1
    count += 1
    output = sprintf("%s/block-%02d.sh", output_dir, count)
    next
  }
  in_block && /^```$/ {
    in_block = 0
    close(output)
    next
  }
  in_block { print >> output }
' docs/furiosa-rngd-compilation-troubleshooting.md
for script in "$CHECK_DIR"/*.sh; do bash -n "$script"; done
```

Expected: every extracted shell block exits 0.

- [ ] **Step 7: Commit the evidence document**

```bash
git add docs/furiosa-rngd-compilation-troubleshooting.md \
  docs/furiosa-rngd-setup.md \
  docs/furiosa-rngd-troubleshooting.md \
  framework/tests/test_furiosa_compile_repro.py
git commit -m "docs: Furiosa 컴파일 실패 재현 절차 기록"
```

---

### Task 5: Regression verification and hardware handoff

**Files:**
- Verify: all files from Tasks 1-4
- Create locally only: `framework/results/furiosa-compile-repro/*` on an RNGD server

**Interfaces:**
- Produces: clean branch plus exact server handoff commands
- Consumes: completed reproduction CLI and documents

- [ ] **Step 1: Run the complete local reproduction tests**

```bash
cd framework
python -m pytest tests/test_furiosa_compile_repro.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run existing supported Furiosa tests that do not depend on sandbox cross-thread wakeups**

```bash
cd framework
python -m pytest \
  tests/test_furiosa_llm_runtime.py \
  tests/test_furiosa_server_metrics.py \
  tests/test_furiosa_torch_bert_integration.py \
  tests/test_furiosa_torch_bert_models.py \
  tests/test_furiosa_torch_environment_contract.py \
  tests/test_furiosa_torch_runtime.py \
  -q
```

Expected: pass with the documented hardware-only skip. Do not claim
`tests/test_furiosa_native_backend.py` locally passes: the current sandbox also times out for a pure Python cross-thread `asyncio.run_coroutine_threadsafe()` control test.

- [ ] **Step 3: Run static verification**

```bash
python -m py_compile \
  framework/tools/furiosa_compile_repro.py \
  framework/tools/reproduce_furiosa_compile_failures.py
git diff --check origin/main...HEAD
git status --short
```

Expected: both modules compile; no whitespace errors; only intentional files appear.

- [ ] **Step 4: Run each model on the RNGD server**

```bash
cd ~/ML-HW-Benchmark-Framework/framework
PY=../.venv-furiosa-torch/bin/python

timeout --signal=INT --kill-after=30s 45m \
  "$PY" tools/reproduce_furiosa_compile_failures.py \
  --case resnet50

timeout --signal=INT --kill-after=30s 45m \
  "$PY" tools/reproduce_furiosa_compile_failures.py \
  --case yolov5m \
  --yolov5-path models/yolov5m/yolov5mu.pt

timeout --signal=INT --kill-after=30s 45m \
  "$PY" tools/reproduce_furiosa_compile_failures.py \
  --case patchtst \
  --patchtst-path models/ibm-research_patchtst-fm-r1
```

Expected on SDK 2026.3.0: each exits 1, writes text and JSON evidence, and matches one of the model-specific signatures. If a case exits 0 after an SDK upgrade, run CPU/RNGD parity and the full framework benchmark before changing support status.

- [ ] **Step 5: Record final branch state**

```bash
git log --oneline --decorate origin/main..HEAD
git status --short --branch
```

Expected: the design, implementation, and documentation commits are present and the worktree is clean.
