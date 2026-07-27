# Furiosa TorchVision ImageNet ResNet50 Compile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a server-runnable tool that downloads TorchVision ImageNet V2 ResNet50 weights and proves whether the whole model compiles and executes on RNGD without eager fallback.

**Architecture:** Keep this as an isolated diagnostic under `framework/tools`, not a benchmark profile. The tool lazily loads Torch, TorchVision, and Furiosa, deep-copies one pretrained CPU model for RNGD, compiles it with the strict backend contract, measures first and warm calls, and validates the completed outputs on CPU with NumPy.

**Tech Stack:** Python 3.12, PyTorch 2.10, TorchVision through `furiosa-torch[vision]`, Furiosa Torch 2026.3, NumPy, pytest

## Global Constraints

- Model is `torchvision.models.resnet50` with `ResNet50_Weights.IMAGENET1K_V2`.
- Input is static `(1, 3, 224, 224)`, `float32`, NCHW with deterministic seed `0` by default.
- Furiosa compilation always uses `eager_fallback=False`, `fullgraph=True`, and `dynamic=False`.
- A compile exception, invalid output, non-finite output, or CPU/NPU Top-1 mismatch must produce a non-zero process exit.
- The tool must not generate Warboy ENF, convert ONNX, save an FXB, or change the existing `resnet50` benchmark profile.
- Vendor imports remain lazy so test collection works without Furiosa SDK or RNGD hardware.

---

## File Structure

- Create `framework/tools/compile_furiosa_resnet50.py`: CLI, lazy vendor loading, strict compilation, timing, output validation, and status logging.
- Create `framework/tests/test_compile_furiosa_resnet50.py`: hardware-independent behavior and contract tests with fake vendor objects.
- Modify `docs/furiosa-rngd-torch-multimodel.md`: exact server command, timeout boundary, cache option, and result interpretation.

### Task 1: Strict ResNet50 compile tool

**Files:**
- Create: `framework/tests/test_compile_furiosa_resnet50.py`
- Create: `framework/tools/compile_furiosa_resnet50.py`

**Interfaces:**
- Consumes: TorchVision `resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)` and Furiosa `backend.with_config(CompilerConfig(tactic_hint=TacticHintConfig.Default), eager_fallback=False)`.
- Produces: `CompileCheckConfig`, `CompileCheckResult`, `build_parser()`, `run_compile_check(config, dependencies=None, timer=time.perf_counter, emit=print)`, and `main(argv=None) -> int`.

- [ ] **Step 1: Write failing tests for the default CLI and strict compiler contract**

Create fake tensor, model, Torch, TorchVision, and Furiosa dependencies. The central assertions must be:

```python
def test_parser_defaults_to_imagenet_v2_on_furiosa_zero():
    args = tool.build_parser().parse_args([])
    assert args.device == "furiosa:0"
    assert args.seed == 0
    assert args.torch_home is None


def test_compile_check_uses_imagenet_v2_and_strict_backend(fake_dependencies):
    result = tool.run_compile_check(
        tool.CompileCheckConfig(),
        dependencies=fake_dependencies,
        timer=iter((10.0, 14.0, 20.0, 20.5)).__next__,
        emit=lambda message: None,
    )

    assert fake_dependencies.resnet_calls == [
        fake_dependencies.imagenet_v2_weights
    ]
    assert fake_dependencies.backend_calls[0]["eager_fallback"] is False
    assert fake_dependencies.compile_calls[0]["fullgraph"] is True
    assert fake_dependencies.compile_calls[0]["dynamic"] is False
    assert result.first_call_seconds == 4.0
    assert result.warm_call_seconds == 0.5
    assert result.cpu_top1 == result.npu_top1
```

The fake model must record one CPU inference and two compiled NPU inferences. Its
`__deepcopy__` implementation must preserve the same state identifier so the test also
asserts that CPU and RNGD models share one downloaded set of weights.

- [ ] **Step 2: Run the focused test and observe the expected import failure**

Run:

```bash
cd framework
../.venv/bin/python -m pytest \
  tests/test_compile_furiosa_resnet50.py \
  -q
```

Expected: FAIL during collection because `tools.compile_furiosa_resnet50` does not exist.

- [ ] **Step 3: Implement the data types, parser, lazy imports, and strict compile path**

Create the tool with these public data types:

```python
@dataclass(frozen=True)
class CompileCheckConfig:
    device: str = "furiosa:0"
    seed: int = 0
    torch_home: Path | None = None


@dataclass(frozen=True)
class CompileCheckResult:
    first_call_seconds: float
    warm_call_seconds: float
    cpu_top1: int
    npu_top1: int
    max_abs_diff: float
    output_shape: tuple[int, ...]
```

The lazy dependency loader must import exactly:

```python
import torch
import furiosa.torch
from furiosa.torch.config import CompilerConfig, TacticHintConfig
from torchvision.models import ResNet50_Weights, resnet50
```

`run_compile_check` must configure and execute the strict path as follows:

```python
torch.manual_seed(config.seed)
cpu_model = dependencies.resnet50(
    weights=dependencies.imagenet_v2_weights
).eval()
cpu_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
with torch.inference_mode():
    cpu_output = cpu_model(cpu_input).detach().cpu().float()

npu_model = copy.deepcopy(cpu_model).to(torch.device(config.device))
npu_input = cpu_input.to(torch.device(config.device))
compiler_config = dependencies.CompilerConfig(
    tactic_hint=dependencies.TacticHintConfig.Default
)
backend = dependencies.furiosa_torch.backend.with_config(
    compiler_config,
    eager_fallback=False,
)
compiled = torch.compile(
    npu_model,
    backend=backend,
    fullgraph=True,
    dynamic=False,
)
```

Each timed NPU call must move the output to CPU before stopping the timer so the duration
represents completed work rather than asynchronous dispatch. Emit stage markers with
`flush=True` behavior through the supplied `emit` function. Set `TORCH_HOME` before loading
the dependencies when `config.torch_home` is not `None`.

- [ ] **Step 4: Add failing output and exception propagation tests**

Add tests that replace the compiled outputs or compile function:

```python
def test_compile_check_rejects_top1_mismatch(fake_dependencies):
    fake_dependencies.cpu_logits = np.zeros((1, 1000), dtype=np.float32)
    fake_dependencies.npu_logits = np.zeros((1, 1000), dtype=np.float32)
    fake_dependencies.cpu_logits[0, 1] = 1.0
    fake_dependencies.npu_logits[0, 2] = 1.0
    with pytest.raises(RuntimeError, match="Top-1 mismatch"):
        tool.run_compile_check(
            tool.CompileCheckConfig(),
            dependencies=fake_dependencies,
            emit=lambda message: None,
        )


def test_compile_check_propagates_compiler_failure(fake_dependencies):
    fake_dependencies.compile_error = RuntimeError("compiler panic")
    with pytest.raises(RuntimeError, match="compiler panic"):
        tool.run_compile_check(
            tool.CompileCheckConfig(),
            dependencies=fake_dependencies,
            emit=lambda message: None,
        )
```

Also cover output shape other than `(1, 1000)` and a non-finite NPU logit.

- [ ] **Step 5: Implement output validation and main**

Convert completed outputs using:

```python
cpu_array = np.asarray(cpu_output.numpy())
npu_array = np.asarray(npu_output.numpy())
```

Require both shapes to equal `(1, 1000)`, require `np.isfinite(npu_array).all()`, compute
Top-1 with `argmax(axis=1).item()`, and raise `RuntimeError` on a mismatch. Compute
`max_abs_diff = float(np.max(np.abs(cpu_array - npu_array)))` only after shape validation.

`main` must parse arguments, call `run_compile_check`, print a final compact summary, and
return `0`. Do not catch compiler or validation exceptions; the traceback and non-zero exit
are required diagnostic evidence.

- [ ] **Step 6: Run the focused tests until green**

Run:

```bash
cd framework
../.venv/bin/python -m pytest \
  tests/test_compile_furiosa_resnet50.py \
  -q
```

Expected: all tests PASS without importing a real Furiosa module.

- [ ] **Step 7: Commit the tested compile tool**

```bash
git add \
  framework/tools/compile_furiosa_resnet50.py \
  framework/tests/test_compile_furiosa_resnet50.py
git commit -m "feat: ImageNet ResNet50 RNGD 컴파일 검사 추가"
```

### Task 2: Server runbook

**Files:**
- Modify: `docs/furiosa-rngd-torch-multimodel.md`
- Modify: `framework/tests/test_compile_furiosa_resnet50.py`

**Interfaces:**
- Consumes: Task 1 CLI at `framework/tools/compile_furiosa_resnet50.py`.
- Produces: copy-pasteable server commands and unambiguous pass/fail interpretation.

- [ ] **Step 1: Write a failing runbook contract test**

Add:

```python
def test_runbook_documents_resnet50_compile_command():
    runbook = Path("../docs/furiosa-rngd-torch-multimodel.md").read_text()
    assert "tools/compile_furiosa_resnet50.py" in runbook
    assert "--signal=INT --kill-after=30s 45m" in runbook
    assert "IMAGENET1K_V2" in runbook
    assert "eager_fallback=False" in runbook
```

- [ ] **Step 2: Run the runbook test and observe failure**

Run:

```bash
cd framework
../.venv/bin/python -m pytest \
  tests/test_compile_furiosa_resnet50.py::test_runbook_documents_resnet50_compile_command \
  -q
```

Expected: FAIL because the new tool is not mentioned in the runbook.

- [ ] **Step 3: Document the exact server workflow**

Add a section before the model benchmark commands containing:

```bash
cd "$ROOT/framework"

timeout --signal=INT --kill-after=30s 45m \
  "$PY" tools/compile_furiosa_resnet50.py
echo "EXIT=$?"
```

Document optional isolated cache use:

```bash
mkdir -p "$ROOT/.cache/torch"
"$PY" tools/compile_furiosa_resnet50.py \
  --torch-home "$ROOT/.cache/torch"
```

State that the script downloads `IMAGENET1K_V2`, that the first duration includes compile
and load, and that only `EXIT=0` with matching CPU/NPU Top-1 is success. Preserve the full
traceback and `furiosa-smi info` when it fails.

- [ ] **Step 4: Run the focused test file**

Run:

```bash
cd framework
../.venv/bin/python -m pytest \
  tests/test_compile_furiosa_resnet50.py \
  -q
```

Expected: all tests PASS.

- [ ] **Step 5: Commit the runbook change**

```bash
git add \
  docs/furiosa-rngd-torch-multimodel.md \
  framework/tests/test_compile_furiosa_resnet50.py
git commit -m "docs: ResNet50 RNGD 컴파일 실행 절차 추가"
```

### Task 3: Regression verification and publication

**Files:**
- Verify only: all branch changes

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: a clean, pushed feature branch ready for RNGD server execution.

- [ ] **Step 1: Run syntax and focused verification**

```bash
cd framework
../.venv/bin/python -m py_compile tools/compile_furiosa_resnet50.py
../.venv/bin/python -m pytest \
  tests/test_compile_furiosa_resnet50.py \
  tests/test_furiosa_torch_environment_contract.py \
  tests/test_furiosa_torch_models.py \
  tests/test_furiosa_torch_runtime.py \
  tests/test_furiosa_torch_integration.py \
  -q
```

Expected: syntax compile succeeds and all selected tests PASS.

- [ ] **Step 2: Run the complete test suite**

```bash
cd framework
../.venv/bin/python -m pytest -q
```

Expected: all tests PASS, with only the repository's existing documented skips/warnings.

- [ ] **Step 3: Review the exact diff and repository state**

```bash
git diff --check
git status --short
git log -4 --oneline
```

Expected: no whitespace errors and no uncommitted files.

- [ ] **Step 4: Push the feature branch**

```bash
git push origin feat/furiosa-rngd-multimodel
```

Expected: remote branch advances to include the design, implementation, tests, and runbook.
