# Furiosa Hub Runtime Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the existing `furiosa-rngd` benchmark pipeline from a Furiosa Hugging Face model ID without an HTTP server or mandatory explicit FXB.

**Architecture:** Preserve `ModelSpec.model_paths["hf_model"]` as the model-reference source of truth, allow `CompiledModel.artifact_path` to be null for runtime-resolved artifacts, and add `fxb` to the Furiosa SDK constructor only when the user supplied one. The dataloader, generation calls, evaluators, async engine, and result store remain unchanged.

**Tech Stack:** Python 3.12, pytest, Furiosa-LLM 2026.3-compatible API, Hugging Face Transformers.

## Global Constraints

- Work on branch `feat/furiosa-hub-runtime-contract` in `/tmp/ml-hw-benchmark-furiosa-hub-runtime-contract`.
- `--model-path` accepts a non-empty Hub repository ID or local directory.
- `--fxb` is optional; `--artifact` remains its explicit-FXB alias.
- Existing local model plus explicit FXB behavior must remain unchanged.
- Do not change generation, async scheduling, evaluation, metrics, or result schemas.
- Run asyncio native-backend tests outside the sandbox because sandboxed Unix socketpair writes return `EPERM`.

---

### Task 1: Represent a runtime-resolved compiled artifact

**Files:**
- Create: `framework/tests/test_compiled_model.py`
- Modify: `framework/src/core/compiled_model.py:1-25`

**Interfaces:**
- Consumes: existing `CompiledModel(spec, backend_name, artifact_path)` construction.
- Produces: `CompiledModel.artifact_path: Path | None`; non-null paths retain existence validation and null means runtime resolution.

- [ ] **Step 1: Write the failing optional-artifact test**

```python
from pathlib import Path

import pytest

from core.compiled_model import CompiledModel
from core.model_spec import Model_Spec, Task


def _spec() -> Model_Spec:
    return Model_Spec(
        name="llama",
        task=Task.NLP_GENERATION,
        input_shapes={"input_ids": (1, 8)},
        input_dtype={"input_ids": "int64"},
        output_shapes={"generated_ids": (1, 4)},
        model_paths={"hf_model": "furiosa-ai/Llama-3.1-8B-Instruct"},
    )


def test_compiled_model_accepts_runtime_resolved_artifact():
    compiled = CompiledModel(_spec(), "furiosa_llm", None)
    assert compiled.artifact_path is None


def test_compiled_model_still_rejects_missing_local_artifact(tmp_path):
    with pytest.raises(FileNotFoundError, match="compiled artifact does not exist"):
        CompiledModel(_spec(), "furiosa_llm", tmp_path / "missing.fxb")
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_compiled_model.py -q
```

Expected: the null-artifact test fails because `None` has no `exists()` method; the missing-path test passes.

- [ ] **Step 3: Implement the optional path contract**

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .model_spec import Model_Spec


@dataclass(frozen=True)
class CompiledModel:
    spec: Model_Spec
    backend_name: str
    artifact_path: Optional[Path]

    def __post_init__(self):
        if self.artifact_path is not None and not self.artifact_path.exists():
            raise FileNotFoundError(
                f"[Compiler Data Error] The compiled artifact does not exist at "
                f"'{self.artifact_path}'. The compiler must ensure the binary is "
                "successfully generated before returning CompiledModel."
            )
```

Retain the existing explanatory docstring and update it to describe the null runtime-resolved state.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command. Expected: `2 passed`.

- [ ] **Step 5: Commit the DTO contract**

```bash
git add framework/src/core/compiled_model.py framework/tests/test_compiled_model.py
git commit -m "feat: allow runtime-resolved model artifacts"
```

### Task 2: Accept Hub model references and optional FXB at the CLI boundary

**Files:**
- Modify: `framework/tests/test_main_paths.py:81-154`
- Modify: `framework/src/main.py:170-190`
- Modify: `framework/src/main.py:1927-1992`

**Interfaces:**
- Consumes: `args.model_path: str`, `args.fxb: str | None`, and `args.artifact: str | None`.
- Produces: normalized `args.fxb`/`args.artifact` or null, default `args.tokenizer_path`, and a `CompiledModel` whose artifact is null when SDK resolution is selected.

- [ ] **Step 1: Write failing CLI tests for Hub and local automatic artifacts**

```python
def test_validate_furiosa_cli_accepts_hub_model_without_fxb():
    args = Namespace(
        model_path="furiosa-ai/Llama-3.1-8B-Instruct",
        fxb=None,
        artifact=None,
        tokenizer_path=None,
    )
    benchmark_main._validate_furiosa_cli(
        args, benchmark_main.Task.NLP_GENERATION
    )
    assert args.model_path == "furiosa-ai/Llama-3.1-8B-Instruct"
    assert args.fxb is None
    assert args.artifact is None
    assert args.tokenizer_path == args.model_path


def test_validate_furiosa_cli_accepts_local_model_without_fxb(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    args = Namespace(
        model_path=str(model_path),
        fxb=None,
        artifact=None,
        tokenizer_path=None,
    )
    benchmark_main._validate_furiosa_cli(
        args, benchmark_main.Task.NLP_GENERATION
    )
    assert args.fxb is None
    assert args.artifact is None
    assert args.tokenizer_path == str(model_path)
```

Add explicit empty-reference coverage and keep the existing local-file and
bad-FXB cases:

```python
@pytest.mark.parametrize("model_path", [None, "", "   "])
def test_validate_furiosa_cli_rejects_empty_model_reference(model_path):
    args = Namespace(
        model_path=model_path,
        fxb=None,
        artifact=None,
        tokenizer_path=None,
    )
    with pytest.raises(
        ValueError,
        match="repository ID or local model directory",
    ):
        benchmark_main._validate_furiosa_cli(
            args, benchmark_main.Task.NLP_GENERATION
        )
```

- [ ] **Step 2: Run the CLI tests and verify RED**

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_main_paths.py -q
```

Expected: both no-FXB tests fail under the current local-directory/FXB-required contract.

- [ ] **Step 3: Implement CLI validation and optional artifact construction**

```python
model_reference = args.model_path
if not isinstance(model_reference, str) or not model_reference.strip():
    raise ValueError(
        "furiosa_llm backend requires --model-path to be a Hugging Face "
        "repository ID or local model directory."
    )

model_path = Path(model_reference).expanduser()
if model_path.exists() and not model_path.is_dir():
    raise ValueError(
        "furiosa_llm backend requires a local --model-path to be a directory."
    )

selected_fxb = args.fxb or args.artifact
if selected_fxb:
    fxb_path = Path(selected_fxb).expanduser()
    if not fxb_path.is_file() or fxb_path.suffix.lower() != ".fxb":
        raise ValueError(
            "furiosa_llm --fxb (or --artifact) must be an existing .fxb file."
        )
    args.fxb = str(fxb_path)
    args.artifact = str(fxb_path)
else:
    args.fxb = None
    args.artifact = None

if not args.tokenizer_path:
    args.tokenizer_path = model_reference
```

Change Furiosa artifact assembly to:

```python
if args.backend == "furiosa_llm":
    artifact_path = Path(args.fxb) if args.fxb else None
```

- [ ] **Step 4: Run CLI and DTO tests and verify GREEN**

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_main_paths.py \
  framework/tests/test_compiled_model.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the CLI contract**

```bash
git add framework/src/main.py framework/tests/test_main_paths.py
git commit -m "feat: accept Furiosa Hub model references"
```

### Task 3: Delegate automatic artifact resolution to Furiosa-LLM

**Files:**
- Modify: `framework/tests/test_furiosa_llm_runtime.py:23-116`
- Modify: `framework/src/runtimes/furiosa_llm_rt.py:1-1`
- Modify: `framework/src/runtimes/furiosa_llm_rt.py:314-359`
- Modify: `framework/src/runtimes/furiosa_llm_rt.py:519-524`

**Interfaces:**
- Consumes: `CompiledModel.spec.model_paths["hf_model"]` and optional `CompiledModel.artifact_path`.
- Produces: `LLM(model_reference, **runtime_options)` without an `fxb` key for SDK resolution, or the existing explicit `fxb` key when a path is present.

- [ ] **Step 1: Write the failing automatic-resolution runtime test**

```python
def test_load_omits_fxb_for_hub_artifact_resolution(monkeypatch):
    state = _install_fake_sdk(monkeypatch)
    spec = Model_Spec(
        name="llama",
        task=Task.NLP_GENERATION,
        input_shapes={"input_ids": (1, 8)},
        input_dtype={"input_ids": "int64"},
        output_shapes={"generated_ids": (1, 4)},
        model_paths={
            "hf_model": "furiosa-ai/Llama-3.1-8B-Instruct"
        },
    )
    compiled = CompiledModel(spec, "furiosa_llm", None)
    runtime = FuriosaLlmRuntime(device="npu:0")
    runtime.load(compiled)

    model_reference, kwargs = state["llm_init"][0]
    assert model_reference == "furiosa-ai/Llama-3.1-8B-Instruct"
    assert "fxb" not in kwargs
    assert kwargs["devices"] == "npu:0"
    assert runtime.is_compatible(compiled) is True
```

Keep `test_load_passes_hf_fxb_device_and_scheduler_options` unchanged as the explicit-FXB regression test.

- [ ] **Step 2: Run the runtime test and verify RED**

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest \
  framework/tests/test_furiosa_llm_runtime.py::test_load_omits_fxb_for_hub_artifact_resolution \
  -q
```

Expected: failure because the current runtime passes `fxb="None"`.

- [ ] **Step 3: Implement conditional FXB forwarding and null-safe compatibility**

```python
llm_kwargs: Dict[str, Any] = {
    "devices": self.devices,
    "max_io_memory_mb": self.max_io_memory_mb,
}
if compiled_model.artifact_path is not None:
    llm_kwargs["fxb"] = str(compiled_model.artifact_path)
```

```python
def is_compatible(self, compiled_model: CompiledModel) -> bool:
    if compiled_model.backend_name.lower() in {
        "furiosa_llm", "furiosa", "rngd"
    }:
        return True
    artifact_path = compiled_model.artifact_path
    return (
        artifact_path is not None
        and artifact_path.suffix.lower() == ".fxb"
    )
```

Update the module docstring to describe explicit and SDK-resolved Furiosa artifacts.

- [ ] **Step 4: Run synchronous Furiosa tests and verify GREEN**

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_furiosa_llm_runtime.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit runtime delegation**

```bash
git add framework/src/runtimes/furiosa_llm_rt.py \
  framework/tests/test_furiosa_llm_runtime.py
git commit -m "feat: delegate Furiosa artifact resolution"
```

### Task 4: Document and verify the end-to-end contract

**Files:**
- Modify: `docs/furiosa-rngd-setup.md:1-58`
- Modify: `framework/README.md:146-148`
- Modify: `README.md:28-28` only if its link text still describes FXB as mandatory.

**Interfaces:**
- Consumes: the implemented Hub-ID and explicit-FXB CLI contracts.
- Produces: copy-paste E2E and async commands that do not launch a server.

- [ ] **Step 1: Update setup documentation**

Document this preferred command:

```bash
python src/main.py \
  --model llama-3.1-8b \
  --target furiosa-rngd \
  --model-path furiosa-ai/Llama-3.1-8B-Instruct \
  --dataset datasets/squad2/val.json \
  --inference-mode e2e \
  --warmup 2 \
  --max-new-tokens 32 \
  --max-steps 2
```

State that `furiosa-llm serve` is not used, the SDK selects its compatible Hub artifact, and `--fxb` remains an explicit override.

- [ ] **Step 2: Run formatting and focused contract checks**

```bash
git diff --check
/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_compiled_model.py \
  framework/tests/test_main_paths.py \
  framework/tests/test_furiosa_llm_runtime.py -q
```

Expected: no diff errors and all focused tests pass.

- [ ] **Step 3: Run the complete Furiosa regression set outside the sandbox**

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests/test_plugin_registry.py \
  framework/tests/test_main_paths.py \
  framework/tests/test_compiled_model.py \
  framework/tests/test_furiosa_llm_runtime.py \
  framework/tests/test_furiosa_native_backend.py -q
```

Expected: all tests pass outside the sandbox so asyncio thread wakeups can use their internal Unix socketpair.

- [ ] **Step 4: Run the full framework test suite outside the sandbox**

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python \
  -m pytest framework/tests -q
```

Expected: all framework tests pass.

- [ ] **Step 5: Review and commit documentation**

```bash
git diff --check
git status --short
git diff origin/main...HEAD --stat
git add docs/furiosa-rngd-setup.md framework/README.md README.md
git commit -m "docs: explain serverless Furiosa benchmarks"
```

If `README.md` is unchanged, omit it from `git add`.

## Hardware Verification Handoff

After the branch is available on the RNGD host, stop any active
`furiosa-llm serve` process and run the preferred Task 4 command. Success means
the log shows `BenchmarkRunner`, two measured SQuAD samples, final metrics,
`RUN_ID=...`, and a new row in `framework/results/benchmark_results.csv`.
