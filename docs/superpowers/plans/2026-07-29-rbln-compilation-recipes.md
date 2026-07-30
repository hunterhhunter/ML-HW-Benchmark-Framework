# RBLN Compilation Recipes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add repository-owned, copy-safe compile recipes for the five validated `rbln-static` models and one canonical Korean runbook covering those recipes plus the existing Llama 3.2 3B and Llama 3.1 8B preparation tool.

**Architecture:** Keep compilation outside the benchmark target registry. Five small Python modules own their model-specific wrapper and fixed ABI, while `tools.rbln_compile_recipes.common` provides only standard-library contract serialization, output overwrite protection, RBLN inspect normalization, ABI validation, and SHA256 reporting. The Llama models continue through `tools/prepare_rbln_vllm_model.py`; the new document links all seven flows without checking generated artifacts into Git.

**Tech Stack:** Python 3.10, `argparse`, `dataclasses`, `hashlib`, PyTorch/TorchVision, Transformers, Ultralytics YOLOv5 source, `rebel-compiler==0.11.0`, pytest, Markdown.

## Global Constraints

- Do not add automatic compilation to the `rbln-static` target.
- Do not modify model profiles, runtime, evaluator, async queue, monitor, or result semantics.
- Do not commit `.rbln`, model weights, tokenizers, datasets, caches, results, traces, logs, or credentials.
- Every recipe must process `--help` and `--describe` before importing `rebel`, Torch, TorchVision, Transformers, or YOLO source.
- Actual compilation requires an explicit `.rbln` `--output` path and refuses an existing path.
- Compiled artifacts target `RBLN-CA22` and fixed batch 1 ABIs only.
- The validated build baseline is Python 3.10.12, `rebel-compiler==0.11.0`, KMD/firmware 3.2.2.
- Llama single-NPU output remains `support_classification=unsupported_single_npu_experiment`.
- New local recipe code may be unit-tested without an NPU, but hardware compile success may only be claimed after rerunning on the ATOM server.

---

## File map

| File | Responsibility |
|---|---|
| `framework/tools/__init__.py` | Make the existing tools directory module-addressable without changing standalone scripts |
| `framework/tools/rbln_compile_recipes/common.py` | Pure contract types, parser helpers, lazy inspect, ABI validation, SHA256, safe save/finalize |
| `framework/tools/rbln_compile_recipes/resnet50/compile.py` | TorchVision ImageNet1K V2 ResNet50 recipe |
| `framework/tools/rbln_compile_recipes/yolov5m/compile.py` | Pinned YOLOv5m raw-head recipe using the local submodule and weight |
| `framework/tools/rbln_compile_recipes/bert_sst2/compile.py` | Two-input SST-2 classification recipe |
| `framework/tools/rbln_compile_recipes/bert_squad/compile.py` | Three-input QA recipe with positional start/end tuple |
| `framework/tools/rbln_compile_recipes/patchtst_etth1/compile.py` | Fixed static patchifier and bool-mask workaround recipe |
| `framework/tests/test_rbln_compile_recipes.py` | SDK-free CLI, contract, lazy import, output guard, and documentation tests |
| `framework/docs/rbln-compilation.md` | Canonical seven-model compile and artifact handoff runbook |
| `framework/docs/rbln-setup.md` | Clarify “no automatic compilation” and link the canonical runbook |
| `framework/docs/rbln-vllm-setup.md` | Link Llama preparation back to the canonical runbook |
| `README.md`, `framework/README.md` | Expose the new runbook from repository entrypoints |

---

### Task 1: Common recipe contract and safe finalization

**Files:**
- Create: `framework/tools/__init__.py`
- Create: `framework/tools/rbln_compile_recipes/__init__.py`
- Create: `framework/tools/rbln_compile_recipes/common.py`
- Create: `framework/tests/test_rbln_compile_recipes.py`

**Interfaces:**
- Produces: `TensorContract`, `RecipeContract`, `create_parser()`, `emit_description_or_require_output()`, `prepare_output_path()`, `save_and_validate()`, and `contract_to_dict()`.
- `save_and_validate(compiled_model: object, output: Path, contract: RecipeContract) -> dict[str, object]` saves once, inspects lazily, validates the fixed ABI, and returns the printable report.

- [ ] **Step 1: Write failing common-contract tests**

```python
from pathlib import Path

import pytest

from tools.rbln_compile_recipes.common import (
    RecipeContract,
    TensorContract,
    contract_to_dict,
    prepare_output_path,
)


def _contract():
    return RecipeContract(
        recipe="unit",
        model_id="owner/model",
        inputs=(TensorContract("x", (1, 3), "float32"),),
        outputs=(TensorContract("y", (1, 2), "float32"),),
        allow_unnamed_outputs=True,
        notes=("fixed batch one",),
    )


def test_contract_description_is_json_safe_and_stable():
    assert contract_to_dict(_contract()) == {
        "recipe": "unit",
        "model_id": "owner/model",
        "target_npu": "RBLN-CA22",
        "inputs": [{"name": "x", "shape": [1, 3], "dtype": "float32"}],
        "outputs": [{"name": "y", "shape": [1, 2], "dtype": "float32"}],
        "allow_unnamed_outputs": True,
        "notes": ["fixed batch one"],
    }


def test_prepare_output_path_requires_rbln_and_refuses_overwrite(tmp_path):
    with pytest.raises(ValueError, match=".rbln"):
        prepare_output_path(tmp_path / "model.bin")
    existing = tmp_path / "model.rbln"
    existing.write_bytes(b"do-not-overwrite")
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_output_path(existing)
```

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run:

```bash
cd framework
python -m pytest -q tests/test_rbln_compile_recipes.py
```

Expected: collection fails with `ModuleNotFoundError: tools.rbln_compile_recipes`.

- [ ] **Step 3: Implement the pure types and output guard**

Use frozen dataclasses and exact builtin normalization:

```python
@dataclass(frozen=True)
class TensorContract:
    name: str | None
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class RecipeContract:
    recipe: str
    model_id: str
    inputs: tuple[TensorContract, ...]
    outputs: tuple[TensorContract, ...]
    allow_unnamed_outputs: bool = False
    notes: tuple[str, ...] = ()
    target_npu: str = "RBLN-CA22"


def prepare_output_path(value: str | Path) -> Path:
    output = Path(value).expanduser().resolve()
    if output.suffix != ".rbln":
        raise ValueError("RBLN compile output must use the .rbln suffix")
    if output.exists():
        raise FileExistsError(f"RBLN compile output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
```

`create_parser()` adds `--output` and `--describe`; output is checked only after
`--describe` has emitted JSON. `common.py` must not import optional build packages at
module scope.

- [ ] **Step 4: Add inspect normalization and ABI validation tests with a fake SDK**

Test mapping and attribute descriptors, single unnamed output allowance, output count
mismatch, shape/dtype mismatch, NPU mismatch, and zero-byte save rejection. Inject a fake
`rebel` module through `sys.modules` so no real SDK is imported.

```python
def test_save_and_validate_accepts_mapping_inspect_and_reports_sha(monkeypatch, tmp_path):
    class Compiled:
        def save(self, path):
            Path(path).write_bytes(b"compiled")

    fake_rebel = types.SimpleNamespace(
        RBLNCompiledModel=types.SimpleNamespace(
            inspect=lambda path: {
                "npu": "RBLN-CA22",
                "compiler_version": "0.11.0",
                "inputs": [{"name": "x", "shape": [1, 3], "dtype": "float32"}],
                "outputs": [{"name": None, "shape": [1, 2], "dtype": "float32"}],
            }
        )
    )
    monkeypatch.setitem(sys.modules, "rebel", fake_rebel)
    report = save_and_validate(Compiled(), tmp_path / "model.rbln", _contract())
    assert report["size_bytes"] == 8
    assert len(report["sha256"]) == 64
```

- [ ] **Step 5: Implement lazy inspect, comparison, and reporting**

Inside `save_and_validate`, import `rebel` only after `compiled_model.save()`. Normalize
mapping/object fields with:

```python
def _field(value, name):
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)
```

Require exact tensor count, positive dimensions, shape and normalized lowercase dtype.
Require input names exactly. For outputs, accept `None` only when
`contract.allow_unnamed_outputs` is true; otherwise require the semantic name. Save to the
final explicit path once, reject empty output, calculate SHA256 in 1 MiB chunks, and print
the report as sorted JSON from each recipe.

- [ ] **Step 6: Run focused tests**

```bash
cd framework
python -m pytest -q tests/test_rbln_compile_recipes.py
```

Expected: all Task 1 tests pass without `rebel` installed.

- [ ] **Step 7: Commit**

```bash
git add framework/tools/__init__.py \
  framework/tools/rbln_compile_recipes \
  framework/tests/test_rbln_compile_recipes.py
git commit -m "feat: add safe RBLN compile recipe contracts"
```

---

### Task 2: ResNet50 and BERT SST-2 recipes

**Files:**
- Create: `framework/tools/rbln_compile_recipes/resnet50/__init__.py`
- Create: `framework/tools/rbln_compile_recipes/resnet50/compile.py`
- Create: `framework/tools/rbln_compile_recipes/bert_sst2/__init__.py`
- Create: `framework/tools/rbln_compile_recipes/bert_sst2/compile.py`
- Modify: `framework/tests/test_rbln_compile_recipes.py`

**Interfaces:**
- Produces module entrypoints `main(argv: Sequence[str] | None = None) -> int`.
- ResNet uses `TorchVision/ResNet50_Weights.IMAGENET1K_V2` and input name `input_np`.
- SST-2 defaults to `textattack/bert-base-uncased-SST-2` and returns logits only.

- [ ] **Step 1: Write failing describe and lazy-import tests**

Parameterize subprocess calls for both modules:

```python
@pytest.mark.parametrize(
    ("module", "model_id", "input_names", "output_shape"),
    [
        (
            "tools.rbln_compile_recipes.resnet50.compile",
            "torchvision/resnet50-imagenet1k-v2",
            ["input_np"],
            [1, 1000],
        ),
        (
            "tools.rbln_compile_recipes.bert_sst2.compile",
            "textattack/bert-base-uncased-SST-2",
            ["input_ids", "attention_mask"],
            [1, 2],
        ),
    ],
)
def test_recipe_describe_needs_no_optional_sdk(module, model_id, input_names, output_shape):
    result = subprocess.run(
        [sys.executable, "-m", module, "--describe"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert payload["model_id"] == model_id
    assert [item["name"] for item in payload["inputs"]] == input_names
    assert payload["outputs"][0]["shape"] == output_shape
```

Also run `--help` with an import guard that raises if `rebel`, `torch`, `torchvision`, or
`transformers` is imported.

- [ ] **Step 2: Confirm both module paths fail before implementation**

```bash
cd framework
python -m pytest -q tests/test_rbln_compile_recipes.py -k 'resnet or sst2 or optional_sdk'
```

Expected: module-not-found failures.

- [ ] **Step 3: Implement ResNet50**

The heavy build function is:

```python
def compile_model():
    import rebel
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    model.requires_grad_(False)
    return rebel.compile_from_torch(
        model,
        [("input_np", [1, 3, 224, 224], "float32")],
    )
```

The contract has one `input_np` and one semantic `output`, allows a single unnamed SDK
output, and records `weights=IMAGENET1K_V2` in notes. `main()` handles description first,
then prepares output, compiles, saves, validates, and prints the JSON report.

- [ ] **Step 4: Implement BERT SST-2**

Use a wrapper so only classification logits are exposed:

```python
class BertSst2(torch.nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).logits
```

Compile with `input_ids` and `attention_mask`, both `[1, 128] int64`. Add
`--model-id` defaulting to the validated ID, but record the selected value in the emitted
runtime report. The static `--describe` contract remains the validated default.

- [ ] **Step 5: Run focused tests and CLI help**

```bash
cd framework
python -m pytest -q tests/test_rbln_compile_recipes.py
python -m tools.rbln_compile_recipes.resnet50.compile --help
python -m tools.rbln_compile_recipes.bert_sst2.compile --describe
```

Expected: exit 0; no optional SDK import is needed for help/describe.

- [ ] **Step 6: Commit**

```bash
git add framework/tools/rbln_compile_recipes/resnet50 \
  framework/tools/rbln_compile_recipes/bert_sst2 \
  framework/tests/test_rbln_compile_recipes.py
git commit -m "feat: add ResNet and BERT RBLN compile recipes"
```

---

### Task 3: Pinned YOLOv5m raw-head recipe

**Files:**
- Create: `framework/tools/rbln_compile_recipes/yolov5m/__init__.py`
- Create: `framework/tools/rbln_compile_recipes/yolov5m/compile.py`
- Modify: `framework/tests/test_rbln_compile_recipes.py`

**Interfaces:**
- Consumes: `--yolov5-root PATH`, `--weights PATH`, explicit `--output PATH`.
- Produces: one raw prediction tensor `(1, 25200, 85)` compatible with the existing framework decoder.
- Enforces source revision `86fd1ab270cb2f7e53ee7412cd4a0650bf4bcc51`.

- [ ] **Step 1: Write failing source-preflight tests**

```python
def test_yolov5_preflight_names_missing_root_and_weight(tmp_path):
    module = importlib.import_module("tools.rbln_compile_recipes.yolov5m.compile")
    with pytest.raises(FileNotFoundError, match="YOLOv5 source root"):
        module.validate_sources(tmp_path / "missing", tmp_path / "yolov5m.pt")


def test_yolov5_describe_records_pinned_revision(run_recipe):
    payload = run_recipe("tools.rbln_compile_recipes.yolov5m.compile")
    assert "86fd1ab270cb2f7e53ee7412cd4a0650bf4bcc51" in payload["notes"]
    assert payload["outputs"][0]["shape"] == [1, 25200, 85]
```

Add a fake `git` executable or monkeypatch `subprocess.run` to prove a different revision is
rejected with a checkout command naming the expected SHA.

- [ ] **Step 2: Run the YOLO tests and confirm failure**

```bash
cd framework
python -m pytest -q tests/test_rbln_compile_recipes.py -k yolov5
```

Expected: missing module or missing `validate_sources`.

- [ ] **Step 3: Implement strict source and weight validation**

`validate_sources()` requires `models/experimental.py`, `models/yolo.py`, and a non-empty
weight file. Resolve the checkout with exact argv:

```python
result = subprocess.run(
    ["git", "-C", str(yolov5_root), "rev-parse", "HEAD"],
    check=True,
    text=True,
    capture_output=True,
)
revision = result.stdout.strip()
if revision != EXPECTED_YOLOV5_REVISION:
    raise RuntimeError(
        f"YOLOv5 source revision {revision} does not match validated "
        f"{EXPECTED_YOLOV5_REVISION}; run git -C {yolov5_root} checkout "
        f"{EXPECTED_YOLOV5_REVISION}"
    )
```

- [ ] **Step 4: Implement the raw-output wrapper and compile path**

After preflight, prepend the explicit root to `sys.path`, import
`models.experimental.attempt_load`, load/fuse/eval the local weight, and wrap the first
inference output:

```python
class YoloV5Raw(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_np):
        output = self.model(input_np)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output
```

Run one CPU zero input before compilation and require exact shape `(1, 25200, 85)`. Compile
with `[("input_np", [1, 3, 640, 640], "float32")]`. Do not include NMS or AutoShape.

- [ ] **Step 5: Run focused tests**

```bash
cd framework
python -m pytest -q tests/test_rbln_compile_recipes.py -k yolov5
python -m tools.rbln_compile_recipes.yolov5m.compile --describe
```

Expected: all tests and describe pass without YOLO/Torch imports.

- [ ] **Step 6: Commit**

```bash
git add framework/tools/rbln_compile_recipes/yolov5m \
  framework/tests/test_rbln_compile_recipes.py
git commit -m "feat: add pinned YOLOv5m RBLN compile recipe"
```

---

### Task 4: BERT SQuAD three-input recipe

**Files:**
- Create: `framework/tools/rbln_compile_recipes/bert_squad/__init__.py`
- Create: `framework/tools/rbln_compile_recipes/bert_squad/compile.py`
- Modify: `framework/tests/test_rbln_compile_recipes.py`

**Interfaces:**
- Defaults to `csarron/bert-base-uncased-squad-v1`.
- Compiles three named `int64 (1,384)` inputs.
- Returns tuple position 0 `start_logits`, position 1 `end_logits`; both names may inspect as `null`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_bert_squad_describe_has_three_inputs_and_two_ordered_outputs(run_recipe):
    payload = run_recipe("tools.rbln_compile_recipes.bert_squad.compile")
    assert [item["name"] for item in payload["inputs"]] == [
        "input_ids", "attention_mask", "token_type_ids"
    ]
    assert [item["name"] for item in payload["outputs"]] == [
        "start_logits", "end_logits"
    ]
    assert payload["allow_unnamed_outputs"] is True
```

- [ ] **Step 2: Confirm the missing recipe fails**

```bash
cd framework
python -m pytest -q tests/test_rbln_compile_recipes.py -k squad
```

- [ ] **Step 3: Implement the wrapper and compile entrypoint**

```python
class BertSquad(torch.nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_id).eval()

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        return outputs.start_logits, outputs.end_logits
```

Use the exact three-entry `input_info` from the design. Do not generate
`model.rbln.json` in this recipe, because the sidecar requires a subsequent real NPU mapping
check. Include that requirement in `notes` and the final report.

- [ ] **Step 4: Run focused tests and existing QA contract regression**

```bash
cd framework
python -m pytest -q \
  tests/test_rbln_compile_recipes.py \
  tests/test_bert_qa_contract.py \
  tests/test_rbln_runtime.py
```

Expected: all pass without altering the runtime/profile contract.

- [ ] **Step 5: Commit**

```bash
git add framework/tools/rbln_compile_recipes/bert_squad \
  framework/tests/test_rbln_compile_recipes.py
git commit -m "feat: add BERT SQuAD RBLN compile recipe"
```

---

### Task 5: PatchTST static-patch recipe

**Files:**
- Create: `framework/tools/rbln_compile_recipes/patchtst_etth1/__init__.py`
- Create: `framework/tools/rbln_compile_recipes/patchtst_etth1/compile.py`
- Modify: `framework/tests/test_rbln_compile_recipes.py`

**Interfaces:**
- Defaults to `ibm-granite/granite-timeseries-patchtst`.
- Produces a `(1,96,7)` tensor from `past_values float32 (1,512,7)` and
  `past_observed_mask bool (1,512,7)`.
- Exposes `static_patchify(past_values)` and `build_static_patchifier(torch_module)` for CPU tests without importing Torch during `--describe`.

- [ ] **Step 1: Write failing pure patchification tests guarded by Torch availability**

```python
torch = pytest.importorskip("torch")


def test_static_patchify_matches_unfold_without_aten_unfold():
    module = importlib.import_module(
        "tools.rbln_compile_recipes.patchtst_etth1.compile"
    )
    values = torch.arange(512 * 7, dtype=torch.float32).reshape(1, 512, 7)
    expected = values[:, 8:, :].unfold(1, 12, 12).transpose(2, 3).contiguous()
    actual = module.static_patchify(values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    traced = torch.jit.trace(module.build_static_patchifier(torch), (values,))
    assert "aten::unfold" not in str(traced.graph)
```

Also assert 42 patches and shape `(1,42,12,7)`.

- [ ] **Step 2: Run the PatchTST test and confirm failure**

```bash
cd framework
python -m pytest -q tests/test_rbln_compile_recipes.py -k patchtst
```

- [ ] **Step 3: Implement the fixed static patchifier**

```python
def static_patchify(past_values):
    import torch

    trimmed = past_values[:, 8:, :]
    return torch.stack(
        [trimmed[:, offset : offset + 12, :] for offset in range(0, 504, 12)],
        dim=1,
    )


def build_static_patchifier(torch_module):
    class StaticPatchifier(torch_module.nn.Module):
        def forward(self, past_values):
            return static_patchify(past_values)

    return StaticPatchifier()
```

Replace only `model.model.patchifier` after first collecting the original CPU output.

- [ ] **Step 4: Implement the mask wrapper and precompile equivalence gates**

Define the outer wrapper inside the heavy build function after importing Torch. It preserves
the artifact bool ABI and casts only before internal model math:

```python
class PatchTSTETTh1(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, past_values, past_observed_mask):
        observed_mask = past_observed_mask.to(dtype=past_values.dtype)
        return self.model(
            past_values=past_values,
            past_observed_mask=observed_mask,
            return_dict=True,
        ).prediction_outputs
```

With deterministic sample values and a 0/1 bool mask, require:

```python
torch.testing.assert_close(static_output, original_output, rtol=1e-5, atol=1e-6)
torch.testing.assert_close(float_mask_output, bool_mask_output, rtol=0, atol=0)
```

Trace the final wrapper and reject `aten::unfold` before calling:

```python
rebel.compile_from_torch(
    wrapper,
    [
        ("past_values", [1, 512, 7], "float32"),
        ("past_observed_mask", [1, 512, 7], "bool"),
    ],
    model_trace_method="jittrace",
)
```

- [ ] **Step 5: Run focused PatchTST and profile tests**

```bash
cd framework
python -m pytest -q \
  tests/test_rbln_compile_recipes.py \
  tests/test_ettm_loader.py \
  tests/test_rbln_runtime.py
```

Expected: all available tests pass; hardware compiler is not imported by describe tests.

- [ ] **Step 6: Commit**

```bash
git add framework/tools/rbln_compile_recipes/patchtst_etth1 \
  framework/tests/test_rbln_compile_recipes.py
git commit -m "feat: add PatchTST RBLN compile recipe"
```

---

### Task 6: Canonical seven-model compilation runbook

**Files:**
- Create: `framework/docs/rbln-compilation.md`
- Modify: `framework/docs/rbln-setup.md`
- Modify: `framework/docs/rbln-vllm-setup.md`
- Modify: `README.md`
- Modify: `framework/README.md`
- Modify: `framework/tests/test_rbln_compile_recipes.py`

**Interfaces:**
- Consumes the five module entrypoints and existing `tools/prepare_rbln_vllm_model.py`.
- Produces the canonical operator sequence from package preflight through artifact handoff.

- [ ] **Step 1: Add failing documentation coverage tests**

```python
def test_compilation_runbook_references_every_recipe_and_llama_tool():
    text = Path("docs/rbln-compilation.md").read_text(encoding="utf-8")
    for module in (
        "tools.rbln_compile_recipes.resnet50.compile",
        "tools.rbln_compile_recipes.yolov5m.compile",
        "tools.rbln_compile_recipes.bert_sst2.compile",
        "tools.rbln_compile_recipes.bert_squad.compile",
        "tools.rbln_compile_recipes.patchtst_etth1.compile",
    ):
        assert module in text
    assert text.count("tools/prepare_rbln_vllm_model.py") >= 2
    assert "unsupported_single_npu_experiment" in text
```

Also assert links to `rbln-compilation.md` from both RBLN setup documents and both README
entrypoints.

- [ ] **Step 2: Run and confirm the missing-document failure**

```bash
cd framework
python -m pytest -q tests/test_rbln_compile_recipes.py -k runbook
```

Expected: `FileNotFoundError` for `docs/rbln-compilation.md`.

- [ ] **Step 3: Write environment and provenance sections**

Document exact variables:

```bash
export RBLN_FW_ROOT="$HOME/ML-HW-Benchmark-Framework-rbln-vllm"
export RBLN_ZOO_ROOT="$HOME/rebelion/rbln-model-zoo"
export RBLN_BUILD_PY="$RBLN_ZOO_ROOT/.venv-rbln-zoo/bin/python"
export RBLN_VLLM_PY="$HOME/ML-HW-Benchmark-Framework-rbln/.venv-rbln/bin/python"
```

Include Python/package-origin inspection, `rbln-smi -q/-j`, Model Zoo clone, YOLO submodule
checkout at the validated SHA, Hugging Face/Meta access boundaries, the 401 guidance, and the
rule that credentials never appear in the repository or shell URL.

- [ ] **Step 4: Document the five static recipe calls and artifact handoff**

For every static model include:

1. model ID or weight provenance;
2. exact module command and output path under a non-overwriting Model Zoo/custom build area;
3. `--describe` output contract;
4. expected input/output ABI;
5. `RBLNCompiledModel.inspect()` and SHA256 command;
6. copy into `framework/models/rbln/<profile>/model.rbln` and source/destination hash equality;
7. relevant dataset preparation and sync smoke link;
8. model-specific failure and recovery.

Use the five exact module names from Step 1. For BERT SQuAD include the existing real CPU/NPU
mapping script and sidecar generation link; do not use strict `1e-3` allclose as the sole
mapping proof because the observed compiled precision had larger elementwise differences while
preserving the correct active-span argmax and answer.

- [ ] **Step 5: Document the two Llama calls and artifact directory contract**

Use the already validated commands:

```bash
"$RBLN_VLLM_PY" tools/prepare_rbln_vllm_model.py \
  --model llama-3.2-3b --output-dir "$RBLN_LLAMA32_DIR" \
  --num-devices 1 --max-seq-len 512 --block-size 512 \
  --batch-size 1 --decoder-batch-sizes 1 \
  --allow-unsupported-single-npu

"$RBLN_VLLM_PY" tools/prepare_rbln_vllm_model.py \
  --model llama-3.1-8b --output-dir "$RBLN_LLAMA31_DIR" \
  --num-devices 1 --max-seq-len 512 --block-size 512 \
  --batch-size 1 --decoder-batch-sizes 1 \
  --allow-unsupported-single-npu
```

State that tokenizer/config, manifest, prefill and decoder artifacts remain one directory;
record 3B observed `prefill.rbln` 7,238,844,846 bytes and `decoder_batch_1.rbln`
806,195,660 bytes as historical evidence, not universal expected sizes. Link the detailed
physical run IDs from `rbln-vllm-atom-validation.md`.

- [ ] **Step 6: Clarify existing docs and README links**

Change `rbln-setup.md` support wording from compile being excluded to:

> The runtime target consumes precompiled artifacts and never compiles automatically. Reproducible offline build recipes live in `rbln-compilation.md`.

Do not remove the existing runtime, monitoring, async, or troubleshooting content. Add concise
links from `rbln-vllm-setup.md`, root `README.md`, and `framework/README.md`.

- [ ] **Step 7: Run documentation and focused regression tests**

```bash
cd framework
python -m pytest -q \
  tests/test_rbln_compile_recipes.py \
  tests/test_prepare_rbln_vllm_model.py \
  tests/test_rbln_runtime.py \
  tests/test_rbln_vllm_runtime.py \
  tests/test_main_paths.py
```

Expected: all pass. Then run:

```bash
rg -n 'T[B]D|T[O]DO|F[I]XME|<{7}|={7}|>{7}' \
  docs/rbln-compilation.md \
  docs/rbln-setup.md \
  docs/rbln-vllm-setup.md
git diff --check
```

Expected: no placeholder/conflict match and exit 0 from `git diff --check`.

- [ ] **Step 8: Commit**

```bash
git add README.md framework/README.md \
  framework/docs/rbln-compilation.md \
  framework/docs/rbln-setup.md \
  framework/docs/rbln-vllm-setup.md \
  framework/tests/test_rbln_compile_recipes.py
git commit -m "docs: record reproducible RBLN model compilation"
```

---

### Task 7: Final regression, review, and PR update

**Files:**
- Verify only; modify a task-owned file only if a failing test proves a defect.

**Interfaces:**
- Produces a clean feature branch containing source recipes and documentation, not artifacts.

- [ ] **Step 1: Run the complete framework suite on the host**

```bash
cd framework
python -m pytest -q
```

Expected baseline: at least the existing `2219 passed, 17 skipped`; new recipe tests increase
the pass count, with no failures. The known unregistered `pytest.mark.integration` warning may
remain; any new warning fails the gate.

- [ ] **Step 2: Verify repository hygiene**

```bash
git diff --check
git status --short
git ls-files | rg '\.(rbln|pt|pth|safetensors)$' && exit 1 || true
find framework/tools/rbln_compile_recipes -type d -name __pycache__ -prune -print
```

Expected: no whitespace errors, only intended source/docs/tests before final commit, no tracked
binary artifact, and no new cache directory included.

- [ ] **Step 3: Review the complete range**

Review `6f89b22..HEAD` for:

- optional dependency lazy-import correctness;
- output overwrite protection;
- model ABI/profile agreement;
- YOLO raw output rather than NMS output;
- SQuAD token-type and output-position contract;
- PatchTST static equivalence and absence of `aten::unfold`;
- no change to production runtime/target behavior;
- all seven models present in the runbook.

- [ ] **Step 4: Push and update PR #40 without merging**

```bash
git push origin feat/rbln-vllm
```

Update the Korean PR body with the new compile recipe files, documentation link, exact test
result, and the statement that hardware recompile with the new checked-in scripts remains a
server verification gate. Do not include credentials or artifact hashes that were not observed.
Do not merge Main and do not change Ready/Draft state unless the user explicitly authorizes that
separate remote action.
