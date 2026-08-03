# Mobilint PatchTST Clamp Lowering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing compiler-only PatchTST compat variant lower unsupported `Tensor.clamp_min` to the mathematically equivalent qbcompiler-supported `Tensor.clamp`, while preserving the stock model, external ABI, immutable failure evidence, and explicit provenance.

**Architecture:** The framework benchmark path remains unchanged. The exact pinned Hugging Face checkpoint is wrapped only inside `framework/tools/mobilint_compile_recipes/patchtst_etth1.py`: the existing static patchifier and mask cast stay in place, and the checkpoint's inner std scaler is replaced with an equivalent module that uses `denominator.clamp(min=1.0)`. Calibration manifests and compile reports identify compat recipe revision 2, the exact rewrite list, and the recipe source SHA256.

**Tech Stack:** Python 3.10/3.12, PyTorch/TorchScript, Transformers PatchTST, pytest, qbcompiler 1.2 external compiler host

## Global Constraints

- Do not modify framework loaders, evaluators, runtime profiles, benchmark metrics, checkpoint weights, or the external `past_values float32 [1,512,7]` / `past_observed_mask bool [1,512,7]` ABI.
- Keep source model commit `7fe295d8bc8fbac8041b60ab351882634165517f` for the observed retry lineage; code must continue to accept any exact lowercase 40-character commit recorded by a stock parent.
- Preserve stock and compat revision 1 failed attempts; never overwrite or relabel them as successful.
- Compat CPU output must match stock at `rtol=1e-5`, `atol=1e-6` for all-observed, sparse, and zero-denominator masks.
- Compat graphs must contain neither `unfold` nor `clamp_min`; the external mask remains bool and the internal scaler denominator uses `clamp(min=1.0)`.
- Do not claim compiler or ARIES success from local tests. Only a fresh server attempt may change the observed status.

---

### Task 1: Add compiler-compatible std-scaler lowering and evidence

**Files:**
- Modify: `framework/tools/mobilint_compile_recipes/patchtst_etth1.py:20-145,252-430`
- Modify: `framework/tests/test_mobilint_patchtst_compile.py:20-210,248-400`
- Modify: `docs/mobilint-compilation-experiments.md:56-70,171-230,483-510`

**Interfaces:**
- Consumes: `build_patchtst_wrapper(model, variant)`, `prepare_calibration(...)`, `_compile_report(...)`, and exact stock-parent revision validation already used by `run_mobilint_compile_experiment.sh`.
- Produces: `COMPAT_RECIPE_REVISION = 2`; compat manifest/report field `compatibility` with `recipe_revision`, ordered `rewrites`, and `recipe_source_sha256`; a wrapper whose public call remains `wrapper(past_values, past_observed_mask) -> float32 [1,96,7]`.

- [ ] **Step 1: Extend the fake checkpoint with the real failing scaler pattern**

Add a fake inner std scaler and container to `framework/tests/test_mobilint_patchtst_compile.py`, and make `_FakePatchTST` use it before patchification:

```python
class _FakeStdScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 1
        self.keepdim = True
        self.minimum_scale = 1e-5

    def forward(self, data, observed_indicator):
        denominator = observed_indicator.sum(
            self.dim, keepdim=self.keepdim
        ).clamp_min(1.0)
        loc = (data * observed_indicator).sum(
            self.dim, keepdim=self.keepdim
        ) / denominator
        variance = (((data - loc) * observed_indicator) ** 2).sum(
            self.dim, keepdim=self.keepdim
        ) / denominator
        scale = torch.sqrt(variance + self.minimum_scale)
        return (data - loc) / scale, loc, scale


class _FakeScalerContainer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scaler = _FakeStdScaler()

    def forward(self, data, observed_indicator):
        return self.scaler(data, observed_indicator)
```

Set `self.model.scaler = _FakeScalerContainer()` and use its scaled tensor, loc, and scale in `_FakePatchTST.forward` so stock and compat exercise the same normalization path.

- [ ] **Step 2: Write RED graph and mask-equivalence tests**

Add these focused behaviors:

```python
@pytest.mark.parametrize("mask_kind", ["all", "sparse", "zero-channel"])
def test_compat_scaler_matches_stock_for_observation_masks(mask_kind):
    stock_model = _FakePatchTST().eval()
    compat_model = copy.deepcopy(stock_model).eval()
    values, sparse = _sample_inputs()
    if mask_kind == "all":
        mask = torch.ones_like(sparse)
    elif mask_kind == "zero-channel":
        mask = torch.ones_like(sparse)
        mask[:, :, 0] = False
    else:
        mask = sparse
    stock = build_patchtst_wrapper(stock_model, "stock")
    compat = build_patchtst_wrapper(
        compat_model, "compat-static-patchifier"
    )
    with torch.no_grad():
        expected = stock(values, mask)
        actual = compat(values, mask)
    assert actual.shape == expected.shape == (1, 96, 7)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_compat_trace_removes_unfold_and_clamp_min_but_keeps_bool_abi():
    values, mask = _sample_inputs()
    wrapper = build_patchtst_wrapper(
        _FakePatchTST().eval(), "compat-static-patchifier"
    )
    traced = torch.jit.trace(wrapper, (values, mask), strict=True)
    graph = str(traced.inlined_graph)
    assert "aten::unfold" not in graph
    assert "aten::clamp_min" not in graph
    assert "aten::clamp" in graph
    assert mask.dtype == torch.bool
```

Extend the compat prepare test to require:

```python
from tools.mobilint_compile_recipes import patchtst_etth1 as patchtst_module


assert manifest["compatibility"] == {
    "recipe_revision": 2,
    "rewrites": [
        "Tensor.unfold -> fixed slice/stack patchifier",
        "bool observation mask -> past_values dtype inside wrapper",
        "Tensor.clamp_min(1.0) -> Tensor.clamp(min=1.0)",
    ],
    "recipe_source_sha256": hashlib.sha256(
        Path(patchtst_module.__file__).read_bytes()
    ).hexdigest(),
}
```

Require the same mapping in `compile-report.json`. Also assert that a stock manifest/report has no `compatibility` field.

- [ ] **Step 3: Run the RED tests and confirm the observed defect**

Run:

```bash
PY=/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python
PYTHONPATH=framework:framework/src "$PY" -m pytest \
  framework/tests/test_mobilint_patchtst_compile.py \
  -k 'compat_scaler or compat_trace or compat_prepare' -q
```

Expected: FAIL because the compat graph still contains `aten::clamp_min`, the scaler is not replaced, and the compatibility provenance fields do not exist. Confirm the failure is not an import or fixture error.

- [ ] **Step 4: Implement the minimal compiler-only scaler replacement**

In `patchtst_etth1.py`, add exact immutable provenance constants:

```python
COMPAT_RECIPE_REVISION = 2
COMPAT_REWRITES = (
    "Tensor.unfold -> fixed slice/stack patchifier",
    "bool observation mask -> past_values dtype inside wrapper",
    "Tensor.clamp_min(1.0) -> Tensor.clamp(min=1.0)",
)


def _compatibility_provenance() -> dict[str, object]:
    return {
        "recipe_revision": COMPAT_RECIPE_REVISION,
        "rewrites": list(COMPAT_REWRITES),
        "recipe_source_sha256": sha256_file(Path(__file__).resolve()),
    }
```

Inside `build_patchtst_wrapper`, only for `compat-static-patchifier`, validate `model.model.scaler.scaler` exposes `dim`, `keepdim`, and `minimum_scale`, then replace that inner module with:

```python
class CompilerCompatibleStdScaler(torch.nn.Module):
    def __init__(self, stock_scaler):
        super().__init__()
        self.dim = stock_scaler.dim
        self.keepdim = stock_scaler.keepdim
        self.minimum_scale = stock_scaler.minimum_scale

    def forward(self, data, observed_indicator):
        denominator = observed_indicator.sum(
            self.dim, keepdim=self.keepdim
        ).clamp(min=1.0)
        loc = (data * observed_indicator).sum(
            self.dim, keepdim=self.keepdim
        ) / denominator
        variance = (((data - loc) * observed_indicator) ** 2).sum(
            self.dim, keepdim=self.keepdim
        ) / denominator
        scale = torch.sqrt(variance + self.minimum_scale)
        return (data - loc) / scale, loc, scale
```

Do not replace any scaler for stock. Add `_compatibility_provenance()` to both the compat source manifest and compat compile report; omit it entirely for stock.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
PY=/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python
PYTHONPATH=framework:framework/src "$PY" -m pytest \
  framework/tests/test_mobilint_patchtst_compile.py -q
```

Expected: all PatchTST compile recipe tests PASS. Inspect the traced graph assertion to confirm it fails if `clamp_min` is reintroduced.

- [ ] **Step 6: Update the canonical runbook with real evidence and compat revision 2**

In `docs/mobilint-compilation-experiments.md`:

- describe all three compiler-only rewrites and state that they do not change the external ABI, checkpoint, weights, loader, evaluator, or framework runtime;
- record stock attempt `20260803T102159225731801Z-806306` as `MBLT_COMPILE=fail` on boolean `clamp_min`;
- record compat revision 1 attempt `20260803T102423121257760Z-841876` as `MBLT_COMPILE=fail` on float32 `clamp_min`, after CPU equivalence/static patchification/mask cast passed;
- keep compat revision 2 as `not_run` until a fresh compiler-server attempt returns;
- state that all attempts use resolved model revision `7fe295d8bc8fbac8041b60ab351882634165517f` and that no MBLT/MXQ artifact exists for the two failed attempts.

- [ ] **Step 7: Run integration regression checks**

Run:

```bash
PY=/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python
PYTHONPATH=framework:framework/src "$PY" -m pytest \
  framework/tests/test_mobilint_patchtst_compile.py \
  framework/tests/test_mobilint_compile_attempt.py \
  framework/tests/test_mobilint_compile_runtime_verify.py \
  framework/tests/test_patchtst_etth1_profile.py -q
bash -n framework/scripts/run_mobilint_compile_experiment.sh
"$PY" -m compileall -q framework/tools/mobilint_compile_recipes
git diff --check
```

Expected: all selected tests, Bash syntax, Python compilation, and diff checks PASS. Local tests do not change the documented compiler/ARIES status.

- [ ] **Step 8: Commit the reviewed implementation**

```bash
git add \
  framework/tools/mobilint_compile_recipes/patchtst_etth1.py \
  framework/tests/test_mobilint_patchtst_compile.py \
  docs/mobilint-compilation-experiments.md
git commit -m "fix: lower PatchTST clamp for qbcompiler"
```

After task review and final branch review, push the branch. The server must fetch the new exact HEAD and create a fresh compat attempt; it must not resume either failed attempt root.
