# Furiosa Torch Model Graph Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve strict RNGD execution while normalizing BERT, YOLOv5m, and PatchTST-FM model graphs past their confirmed frontend failures.

**Architecture:** Keep every workaround inside the local model adapter that owns the incompatible graph. BERT selects the explicit eager attention graph, YOLO uses Ultralytics' official Conv-BN fusion, and PatchTST registers only its module logger as a TorchDynamo no-op; the shared runtime remains strict and unchanged.

**Tech Stack:** Python 3.12, PyTorch 2.10, Transformers 4.57.6, Ultralytics 8.3.216, granite-tsfm 0.3.6, Furiosa Torch 2026.3.0, pytest

## Global Constraints

- Keep `eager_fallback=False`, `fullgraph=True`, and `dynamic=False` unchanged.
- Do not enable CPU/eager fallback or suppress compiler errors.
- Preserve every adapter's existing input names, output names, shapes, and dtypes.
- Treat a post-normalization Furiosa compiler failure as `compiler-blocked` rather than silently changing execution mode.
- Do not claim RNGD compile success until the server E2E smoke test exits with status 0.

---

### Task 1: Normalize both BERT attention graphs

**Files:**
- Modify: `framework/tests/test_furiosa_torch_models.py`
- Modify: `framework/src/runtimes/furiosa_torch_models.py:75-122`

**Interfaces:**
- Consumes: `AutoModelForSequenceClassification.from_pretrained()` and `AutoModelForQuestionAnswering.from_pretrained()`.
- Produces: the existing wrapper interfaces, loaded with `attn_implementation="eager"`.

- [ ] **Step 1: Make both BERT loader tests require eager attention**

Change the expected load calls in both BERT tests to:

```python
assert load_calls == [
    {
        "path": model_path,
        "kwargs": {
            "local_files_only": True,
            "attn_implementation": "eager",
        },
    }
]
```

- [ ] **Step 2: Run the two tests and verify RED**

Run:

```bash
cd framework
python -m pytest \
  tests/test_furiosa_torch_models.py::test_bert_classification_loader_returns_only_logits \
  tests/test_furiosa_torch_models.py::test_bert_qa_loader_returns_raw_start_and_end_logits \
  -q
```

Expected: both fail because actual kwargs do not contain `attn_implementation`.

- [ ] **Step 3: Select eager attention in both production loaders**

Add the same keyword to both `from_pretrained()` calls:

```python
base = AutoModelForSequenceClassification.from_pretrained(
    path,
    local_files_only=True,
    attn_implementation="eager",
).eval()
```

```python
base = AutoModelForQuestionAnswering.from_pretrained(
    path,
    local_files_only=True,
    attn_implementation="eager",
).eval()
```

- [ ] **Step 4: Run the two tests and verify GREEN**

Run the Step 2 command again. Expected: `2 passed`.

- [ ] **Step 5: Commit the BERT normalization**

```bash
git add framework/tests/test_furiosa_torch_models.py \
  framework/src/runtimes/furiosa_torch_models.py
git commit -m "fix: Furiosa BERT attention 그래프 정규화"
```

### Task 2: Fuse YOLOv5m Conv-BatchNorm pairs

**Files:**
- Modify: `framework/tests/test_furiosa_torch_models.py`
- Modify: `framework/src/runtimes/furiosa_torch_models.py:46-73`

**Interfaces:**
- Consumes: `ultralytics.YOLO(path).fuse()` and the resulting `.model`.
- Produces: the existing YOLO wrapper returning raw detection tensor `(1, 84, 8400)`.

- [ ] **Step 1: Make the YOLO unit test observe official fusion**

Extend its fake loader and assertions:

```python
fuse_calls = []

class FakeYOLO:
    def __init__(self, path):
        load_calls.append(path)
        self.model = base

    def fuse(self):
        fuse_calls.append(self)
        return self

# existing wrapper invocation and output assertions
assert len(fuse_calls) == 1
```

- [ ] **Step 2: Run the YOLO loader test and verify RED**

Run:

```bash
cd framework
python -m pytest \
  tests/test_furiosa_torch_models.py::test_yolov5_loader_returns_raw_detection_tensor \
  -q
```

Expected: fail because `fuse_calls` remains empty.

- [ ] **Step 3: Fuse through the public Ultralytics model API**

Replace direct `.model` extraction with:

```python
yolo = YOLO(str(path))
yolo.fuse()
base = yolo.model.eval()
```

- [ ] **Step 4: Run the YOLO loader and CPU contract tests**

```bash
cd framework
python -m pytest \
  tests/test_furiosa_torch_models.py::test_yolov5_loader_returns_raw_detection_tensor \
  tests/test_furiosa_torch_models.py::test_yolov5mu_cpu_forward_matches_static_output_contract \
  -q
```

Expected: unit test passes; CPU contract passes when the checkpoint exists or skips otherwise.

- [ ] **Step 5: Commit the YOLO normalization**

```bash
git add framework/tests/test_furiosa_torch_models.py \
  framework/src/runtimes/furiosa_torch_models.py
git commit -m "fix: Furiosa YOLO Conv-BN 그래프 융합"
```

### Task 3: Remove the PatchTST logging side effect from capture

**Files:**
- Modify: `framework/tests/test_furiosa_torch_models.py`
- Modify: `framework/src/runtimes/furiosa_torch_models.py:150-187`

**Interfaces:**
- Consumes: `modeling_patchtst_fm.logger.info` and `torch._dynamo.config.ignore_logger_methods`.
- Produces: the unchanged PatchTST wrapper returning prediction tensor `(1, 96, 7)`.

- [ ] **Step 1: Make the fake PatchTST module expose a logger and assert registration**

Add `import logging` to the test file, then extend the PatchTST test:

```python
logger = logging.getLogger("test.furiosa.patchtst")
module.logger = logger
ignored = torch._dynamo.config.ignore_logger_methods
previous = set(ignored)
try:
    wrapper = get_torch_model_adapter("patchtst-fm-r1").loader(model_path)
    assert logger.info in ignored
finally:
    ignored.clear()
    ignored.update(previous)
```

Keep all existing model call and output assertions inside the `try` block so the global set is restored even when the test fails.

- [ ] **Step 2: Run the PatchTST loader test and verify RED**

```bash
cd framework
python -m pytest \
  tests/test_furiosa_torch_models.py::test_patchtst_fm_loader_uses_exact_tsfm_architecture \
  -q
```

Expected: fail because `logger.info` is not registered.

- [ ] **Step 3: Register only the PatchTST module logger before model load**

Use the module import so the logger instance is available:

```python
import torch
from tsfm_public.models.patchtst_fm import (
    PatchTSTFMForPrediction,
    modeling_patchtst_fm,
)

try:
    ignored_loggers = torch._dynamo.config.ignore_logger_methods
except AttributeError as exc:
    raise RuntimeError(
        "PatchTST-FM-R1 requires torch._dynamo.config."
        "ignore_logger_methods for strict fullgraph capture."
    ) from exc
ignored_loggers.add(modeling_patchtst_fm.logger.info)
```

Keep the existing optional-dependency error translation around both TSFM imports.

- [ ] **Step 4: Run both PatchTST tests and verify GREEN**

```bash
cd framework
python -m pytest \
  tests/test_furiosa_torch_models.py::test_patchtst_fm_loader_uses_exact_tsfm_architecture \
  tests/test_furiosa_torch_models.py::test_patchtst_fm_loader_explains_optional_dependency \
  -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit the PatchTST normalization**

```bash
git add framework/tests/test_furiosa_torch_models.py \
  framework/src/runtimes/furiosa_torch_models.py
git commit -m "fix: PatchTST fullgraph 로깅 부작용 제거"
```

### Task 4: Regression verification and server handoff

**Files:**
- Modify: `docs/furiosa-rngd-torch-multimodel.md`

**Interfaces:**
- Consumes: the three normalized adapters from Tasks 1-3.
- Produces: tested branch commits and exact server validation commands.

- [ ] **Step 1: Run the focused Furiosa Torch suite**

```bash
cd framework
python -m pytest \
  tests/test_furiosa_torch_environment_contract.py \
  tests/test_furiosa_torch_models.py \
  tests/test_furiosa_torch_runtime.py \
  tests/test_furiosa_torch_integration.py \
  -q
```

Expected: all tests pass; hardware/checkpoint-dependent tests may skip with an explicit reason.

- [ ] **Step 2: Document the server decision tree**

Add a troubleshooting section that records:

```text
BERT view failure -> verify attn_implementation=eager
YOLO mutable BatchNorm -> verify YOLO.fuse() and exported BatchNorm count 0
PatchTST logging.Logger -> verify only PatchTST logger is in ignore_logger_methods
Any later Furiosa UnsupportedOp/internal panic -> classify compiler-blocked
```

- [ ] **Step 3: Run the complete repository test suite available in the development environment**

```bash
cd framework
python -m pytest -q
```

Expected: all runnable tests pass; report exact pass/skip counts.

- [ ] **Step 4: Commit the verification documentation**

```bash
git add docs/furiosa-rngd-torch-multimodel.md
git commit -m "docs: Furiosa 텐서 그래프 문제 해결 절차 추가"
```

- [ ] **Step 5: Push the feature branch**

```bash
git push origin feat/furiosa-rngd-multimodel
```

- [ ] **Step 6: Validate on the RNGD server after pulling the branch**

Run each model in a separate process with `timeout`, starting with `e2e`, one warmup, and one measured sample. Confirm `eager_fallback=False` remains in the runtime and preserve the full traceback for any failure. Only models with `EXIT=0` proceed to single-worker `async_queue`.
