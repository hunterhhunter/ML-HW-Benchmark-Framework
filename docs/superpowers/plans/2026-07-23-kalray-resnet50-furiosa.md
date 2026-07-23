# Kalray ResNet50 Furiosa Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Furiosa Torch ResNet50 adapter load the exact Kalray FP32 ONNX artifact, convert it to a local PyTorch module, and preserve the common logits contract.

**Architecture:** The ResNet50 profile uses one ONNX path for both generic and Furiosa execution. The adapter validates the local artifact, lazily imports ONNX conversion dependencies, converts with `onnx2torch`, and returns an evaluation-mode wrapper without any alternate-weight fallback.

**Tech Stack:** Python 3.12, PyTorch 2.10.0, ONNX 1.20.1, onnx2torch 1.5.15, pytest

## Global Constraints

- Use only `models/Kalray_resnet50/resnet50-v1-7s.onnx` for initial FP32 integration.
- Do not download models at runtime.
- Do not fall back to torchvision, Microsoft ResNet50, random weights, or the INT8 ONNX file.
- Keep the input contract `[N, 3, 224, 224]` FP32 and output contract `[N, 1000]` logits.
- Import `onnx` and `onnx2torch` only when the ResNet50 loader is called.

---

### Task 1: Pin conversion dependencies and align the profile path

**Files:**
- Modify: `framework/requirements-furiosa-torch.txt`
- Modify: `framework/src/core/model_profiles.py`
- Test: `framework/tests/test_furiosa_torch_environment_contract.py`
- Test: `framework/tests/test_furiosa_torch_models.py`

**Interfaces:**
- Consumes: existing `SUPPORTED_PROFILES["resnet50"]` dictionary.
- Produces: identical `default_model_path` and `default_torch_model_path` values for the Kalray FP32 ONNX artifact.

- [ ] **Step 1: Write failing dependency and profile tests**

Update the requirements assertion to require all three exact lines and change the ResNet parameter row to use the ONNX path twice:

```python
assert requirements == [
    "furiosa-torch[vision,llm]==2026.3.0",
    "onnx==1.20.1",
    "onnx2torch==1.5.15",
]

(
    "resnet50",
    "models/Kalray_resnet50/resnet50-v1-7s.onnx",
    "models/Kalray_resnet50/resnet50-v1-7s.onnx",
),
```

- [ ] **Step 2: Run tests and verify the expected failures**

Run:

```bash
cd framework
../.venv-furiosa-torch/bin/python -m pytest \
  tests/test_furiosa_torch_environment_contract.py::test_furiosa_torch_requirements_are_isolated_and_pinned \
  tests/test_furiosa_torch_models.py::test_profiles_preserve_onnx_and_add_explicit_torch_source -q
```

Expected: both tests fail because the conversion dependencies are absent and the Torch profile still points to `models/microsoft_resnet-50`.

- [ ] **Step 3: Apply the minimal dependency and profile changes**

Set the requirement file to:

```text
furiosa-torch[vision,llm]==2026.3.0
onnx==1.20.1
onnx2torch==1.5.15
```

Set the ResNet profile field to:

```python
"default_torch_model_path": "models/Kalray_resnet50/resnet50-v1-7s.onnx",
```

- [ ] **Step 4: Run the focused tests and verify they pass**

Run the command from Step 2. Expected: `2 passed`.

- [ ] **Step 5: Commit the profile contract**

```bash
git add framework/requirements-furiosa-torch.txt \
  framework/src/core/model_profiles.py \
  framework/tests/test_furiosa_torch_environment_contract.py \
  framework/tests/test_furiosa_torch_models.py
git commit -m "build: Kalray ResNet 변환 의존성 고정"
```

### Task 2: Convert the local ONNX artifact in the ResNet adapter

**Files:**
- Modify: `framework/src/runtimes/furiosa_torch_models.py`
- Test: `framework/tests/test_furiosa_torch_models.py`

**Interfaces:**
- Consumes: `Path` to the local FP32 `.onnx` file.
- Produces: `torch.nn.Module` whose `forward(images)` returns one logits tensor.

- [ ] **Step 1: Replace the Hugging Face loader test with failing ONNX conversion tests**

Install fake `onnx` and `onnx2torch` modules in `sys.modules`, record calls, and assert:

```python
wrapper = get_torch_model_adapter("resnet50").loader(model_path)
images = torch.randn(1, 3, 224, 224)

assert wrapper(images) is logits
assert onnx_load_calls == [str(model_path)]
assert convert_calls == [onnx_graph]
assert wrapper.training is False
assert base.training is False
```

Add separate tests that assert a missing path and a directory ending in `.onnx` raise `FileNotFoundError`, and a regular `.pt` file raises `ValueError`. Guard imports so all validation failures happen before `onnx` or `onnx2torch` is imported.

- [ ] **Step 2: Run the ResNet tests and verify the expected failures**

Run:

```bash
cd framework
../.venv-furiosa-torch/bin/python -m pytest tests/test_furiosa_torch_models.py -k resnet -q
```

Expected: failures show the current loader importing Transformers and accepting a directory rather than requiring a local `.onnx` file.

- [ ] **Step 3: Implement local validation and lazy conversion**

Replace `_load_resnet` with:

```python
def _load_resnet(path: Path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Local ResNet50 ONNX model not found: {path}")
    if path.suffix.lower() != ".onnx":
        raise ValueError(f"ResNet50 Furiosa model must be an ONNX file: {path}")

    import torch
    import onnx
    from onnx2torch import convert

    base = convert(onnx.load(str(path))).eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, images):
            output = self.model(images)
            if isinstance(output, (tuple, list)):
                return output[0]
            return output

    return Wrapper(base).eval()
```

- [ ] **Step 4: Run the adapter tests and verify they pass**

Run:

```bash
cd framework
../.venv-furiosa-torch/bin/python -m pytest tests/test_furiosa_torch_models.py -q
```

Expected: all tests in the file pass.

- [ ] **Step 5: Commit the ONNX adapter**

```bash
git add framework/src/runtimes/furiosa_torch_models.py \
  framework/tests/test_furiosa_torch_models.py
git commit -m "feat: Kalray ResNet ONNX를 Furiosa Torch 모델로 변환"
```

### Task 3: Run regression checks and document server verification

**Files:**
- Modify: `docs/superpowers/specs/2026-07-23-kalray-resnet50-furiosa-design.md` only if verification exposes a contract correction.

**Interfaces:**
- Consumes: Tasks 1 and 2 commits.
- Produces: a regression-tested branch ready for actual model parity and RNGD hardware checks.

- [ ] **Step 1: Run focused Furiosa contract tests**

```bash
cd framework
../.venv-furiosa-torch/bin/python -m pytest \
  tests/test_furiosa_torch_environment_contract.py \
  tests/test_furiosa_torch_models.py -q
```

Expected: all selected tests pass with no new warning.

- [ ] **Step 2: Run the full test suite**

```bash
cd framework
HF_DATASETS_CACHE=/tmp/ml-hw-benchmark-hf-datasets-cache \
  ../.venv-furiosa-torch/bin/python -m pytest -q
```

Expected: no regression from the last `1464 passed, 14 skipped` branch baseline, with the new tests increasing the pass count.

- [ ] **Step 3: Check repository hygiene**

```bash
git diff --check
git status --short
```

Expected: no whitespace error and no uncommitted implementation file.

- [ ] **Step 4: Prepare exact server verification commands**

The handoff must include dependency installation, the existing Kalray downloader command, ONNX Runtime versus converted PyTorch tolerance check (`rtol=1e-3`, `atol=1e-4`), a single RNGD inference command, and `furiosa-smi status` observation. Model hashing is added when the Furiosa runtime/result metadata task is implemented; this adapter task must not invent a second result path.
