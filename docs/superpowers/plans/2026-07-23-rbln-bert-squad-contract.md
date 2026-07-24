# RBLN BERT SQuAD Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run `bert-base-uncased-squad-v1` through the internal `rbln-static` engine with the correct three-input BERT contract and deterministic names for the SDK's two unnamed outputs.

**Architecture:** Keep the model profile, offline NumPy dataset, loader, and ONNX exporter on the same `input_ids`/`attention_mask`/`token_type_ids` contract. When `RBLNCompiledModel.inspect()` returns multiple unnamed outputs, load an adjacent `model.rbln.json` manifest whose SHA256 binds the declared positional names to that exact artifact; named outputs and the existing single-output fallback remain unchanged.

**Tech Stack:** Python 3.10, NumPy, pytest, Hugging Face Transformers/Datasets, Rebellions `rebel-compiler==0.11.0`, RBLN-CA22.

## Global Constraints

- Work only on `feat/rbln-runtime-monitor` in `/tmp/ml-hw-benchmark-rbln-runtime-monitor`.
- Do not add `rebel-compiler`, Transformers, or Datasets as mandatory runtime imports.
- Do not silently guess the order of multiple unnamed outputs.
- The sidecar path is the artifact path plus `.json`, for example `model.rbln.json`.
- The sidecar schema has exactly `schema_version`, `artifact_sha256`, and `output_names`; `schema_version` is exactly integer `1`.
- `artifact_sha256` is a lowercase 64-character SHA256 digest and must match the artifact bytes.
- `output_names` contains 1–256 character unique strings, has the inspected output count, and exactly matches the Model_Spec output-name set.
- Reject missing, malformed, oversized, stale, or ambiguous manifests before allocating an RBLN runtime.
- Keep sequence length `384`, batch size `1`, and all three BERT inputs as `int64`.
- Follow RED → GREEN → REFACTOR for every behavior change.

---

### Task 1: Validate SHA-bound RBLN output manifests

**Files:**
- Create: `framework/src/runtimes/rbln_manifest.py`
- Create: `framework/tests/test_rbln_manifest.py`

**Interfaces:**
- Consumes: an artifact `Path`, the inspected output count, and the Model_Spec output-name tuple.
- Produces: `load_output_names(artifact_path: Path, *, descriptor_count: int, expected_names: tuple[str, ...]) -> tuple[str, ...]`.

- [ ] **Step 1: Write failing parser tests**

```python
def test_load_output_names_accepts_sha_bound_manifest(tmp_path):
    artifact = tmp_path / "model.rbln"
    artifact.write_bytes(b"compiled")
    write_manifest(artifact, ["start_logits", "end_logits"])
    assert load_output_names(
        artifact,
        descriptor_count=2,
        expected_names=("start_logits", "end_logits"),
    ) == ("start_logits", "end_logits")


def test_load_output_names_requires_sidecar(tmp_path):
    artifact = tmp_path / "model.rbln"
    artifact.write_bytes(b"compiled")
    with pytest.raises(ValueError, match="sidecar manifest is required"):
        load_output_names(
            artifact,
            descriptor_count=2,
            expected_names=("start_logits", "end_logits"),
        )


def test_load_output_names_rejects_stale_hash(tmp_path):
    artifact = tmp_path / "model.rbln"
    artifact.write_bytes(b"compiled")
    write_manifest(
        artifact,
        ["start_logits", "end_logits"],
        artifact_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="SHA256 does not match"):
        load_output_names(
            artifact,
            descriptor_count=2,
            expected_names=("start_logits", "end_logits"),
        )
```

Add separate tests for wrong output count, duplicate names, names outside the Model_Spec set, unknown schema keys, invalid UTF-8/JSON, a non-regular sidecar, and a file larger than 64 KiB. Each case asserts its specific bounded `ValueError` message.

- [ ] **Step 2: Run the parser tests and verify RED**

Run: `/home/swlab-youngjin/runtime-test/.venv/bin/python -m pytest -q tests/test_rbln_manifest.py`

Expected: collection fails because `runtimes.rbln_manifest` does not exist.

- [ ] **Step 3: Implement the bounded manifest parser**

```python
_SCHEMA_KEYS = frozenset({"schema_version", "artifact_sha256", "output_names"})
_MAX_MANIFEST_BYTES = 64 * 1024


def load_output_names(
    artifact_path: Path,
    *,
    descriptor_count: int,
    expected_names: tuple[str, ...],
) -> tuple[str, ...]:
    manifest_path = Path(f"{artifact_path}.json")
    try:
        manifest_stat = manifest_path.lstat()
    except OSError as exc:
        raise ValueError("RBLN output sidecar manifest is required.") from exc
    if not stat.S_ISREG(manifest_stat.st_mode):
        raise ValueError("RBLN output sidecar manifest must be a regular file.")
    if manifest_stat.st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("RBLN output sidecar manifest must not exceed 64 KiB.")
    try:
        document = json.loads(manifest_path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("RBLN output sidecar manifest is invalid JSON.") from exc
    if type(document) is not dict or set(document) != _SCHEMA_KEYS:
        raise ValueError("RBLN output sidecar manifest must contain exactly the schema keys.")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise ValueError("RBLN output sidecar schema_version must be exactly 1.")
    digest = document["artifact_sha256"]
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("RBLN output sidecar artifact_sha256 must be lowercase SHA256.")
    names = document["output_names"]
    if type(names) is not list or len(names) != descriptor_count:
        raise ValueError("RBLN output sidecar output count does not match inspection.")
    if any(type(name) is not str or not name or len(name) > 256 for name in names):
        raise ValueError("RBLN output sidecar output names must be bounded strings.")
    if len(set(names)) != len(names):
        raise ValueError("RBLN output sidecar output names must be unique.")
    if set(names) != set(expected_names):
        raise ValueError("RBLN output sidecar names do not match Model_Spec.")
    artifact_digest = hashlib.sha256()
    try:
        with artifact_path.open("rb") as artifact_file:
            for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
                artifact_digest.update(chunk)
    except OSError as exc:
        raise ValueError("RBLN artifact SHA256 could not be computed.") from exc
    if not hmac.compare_digest(digest, artifact_digest.hexdigest()):
        raise ValueError("RBLN output sidecar SHA256 does not match the artifact.")
    return tuple(names)
```

- [ ] **Step 4: Run parser tests and verify GREEN**

Run: `/home/swlab-youngjin/runtime-test/.venv/bin/python -m pytest -q tests/test_rbln_manifest.py`

Expected: all parser tests pass.

- [ ] **Step 5: Commit the independently testable parser**

```bash
git add framework/src/runtimes/rbln_manifest.py framework/tests/test_rbln_manifest.py
git commit -m "feat: validate RBLN output manifests"
```

### Task 2: Bind multiple unnamed RBLN outputs through the manifest

**Files:**
- Modify: `framework/src/runtimes/rbln_rt.py`
- Modify: `framework/tests/test_rbln_runtime.py`

**Interfaces:**
- Consumes: `load_output_names(...)` from Task 1 when all inspected outputs are unnamed and output count is greater than one.
- Produces: ordered `_TensorDescriptor` objects with manifest-supplied names and device metadata key `output_binding_source="sha256-sidecar"`.

- [ ] **Step 1: Write failing runtime integration tests**

```python
def test_load_binds_multiple_unnamed_outputs_from_sha_sidecar(
    tmp_path, monkeypatch, fake_rebel
):
    fake_rebel.inspected.outputs = (
        FakeTensor(None, (1, 8), "float32"),
        FakeTensor(None, (1, 8), "float32"),
    )
    model = _compiled_model(
        tmp_path / "qa.rbln",
        output_shapes={"start_logits": (1, 8), "end_logits": (1, 8)},
    )
    write_manifest(model.artifact_path, ["start_logits", "end_logits"])
    runtime = _load_with_fake(monkeypatch, fake_rebel, model)
    assert runtime.get_device_spec()["output_names"] == [
        "start_logits", "end_logits"
    ]
    assert runtime.get_device_spec()["output_binding_source"] == "sha256-sidecar"
```

Also assert that no sidecar, a stale hash, partially named outputs, and a sidecar whose order conflicts with descriptor shapes are rejected before `Runtime(...)` or `AsyncRuntime(...)` allocation. Preserve tests for named multi-output descriptors and the single unnamed output fallback.

- [ ] **Step 2: Run the integration tests and verify RED**

Run: `/home/swlab-youngjin/runtime-test/.venv/bin/python -m pytest -q tests/test_rbln_runtime.py -k 'unnamed_outputs or sidecar'`

Expected: the valid two-output case fails with `missing output descriptor name`.

- [ ] **Step 3: Add positional fallbacks only for verified manifests**

```python
output_name_fallbacks = None
if len(raw_output_items) > 1 and all_output_names_missing:
    output_name_fallbacks = load_output_names(
        Path(compiled_model.artifact_path),
        descriptor_count=len(raw_output_items),
        expected_names=spec_output_names,
    )
output_descriptors = self._normalize_descriptors(
    raw_output_items,
    "output",
    positional_name_fallbacks=output_name_fallbacks,
    single_name_fallback=(
        spec_output_names[0] if len(spec_output_names) == 1 else None
    ),
)
```

Reject the partially named case without opening a sidecar. Store binding provenance in the pending contract and expose it only when a sidecar was used.

- [ ] **Step 4: Run all RBLN runtime tests and verify GREEN**

Run: `/home/swlab-youngjin/runtime-test/.venv/bin/python -m pytest -q tests/test_rbln_manifest.py tests/test_rbln_runtime.py tests/test_rbln_native_backend.py`

Expected: all tests pass.

- [ ] **Step 5: Commit runtime integration**

```bash
git add framework/src/runtimes/rbln_rt.py framework/tests/test_rbln_runtime.py
git commit -m "feat: bind unnamed RBLN outputs with sidecar"
```

### Task 3: Add `token_type_ids` to the complete BERT QA contract

**Files:**
- Modify: `framework/src/core/model_profiles.py`
- Modify: `framework/src/dataloader/bert_qa_loader.py`
- Modify: `framework/src/preprocessor/bert_qa_preprocessor.py`
- Modify: `framework/datasets/prepare_squad_numpy.py`
- Modify: `framework/models/prepare_bert_squad.py`
- Modify: `framework/tests/test_bert_qa_loader.py`
- Create: `framework/tests/test_bert_qa_contract.py`

**Interfaces:**
- Consumes: tokenizer result key `token_type_ids` with shape `(N, 384)`.
- Produces: every QA payload and model artifact with ordered inputs `input_ids`, `attention_mask`, `token_type_ids`, all `int64`.

- [ ] **Step 1: Write failing dataset and profile contract tests**

```python
def test_bert_qa_profile_declares_three_int64_inputs():
    profile = MODEL_PROFILES["bert-base-uncased-squad-v1"]
    assert profile["input_shapes"] == {
        "input_ids": (1, 384),
        "attention_mask": (1, 384),
        "token_type_ids": (1, 384),
    }
    assert profile["input_dtype"] == {
        "input_ids": "int64",
        "attention_mask": "int64",
        "token_type_ids": "int64",
    }


def test_loader_emits_token_type_ids(dummy_squad_data):
    loader = BertQALoader(Mock(spec=Model_Spec), dummy_squad_data[0])
    payload = loader.load_single()
    np.testing.assert_array_equal(
        payload["input"]["token_type_ids"],
        dummy_squad_data[1]["token_type_ids"][0],
    )
```

Add source-contract assertions that both dataset preparation paths save `token_type_ids.npy` and that ONNX export accepts and names the third input.

- [ ] **Step 2: Run QA contract tests and verify RED**

Run: `/home/swlab-youngjin/runtime-test/.venv/bin/python -m pytest -q tests/test_bert_qa_loader.py tests/test_bert_qa_contract.py`

Expected: fixture construction or assertions fail because `token_type_ids.npy` is not required or emitted.

- [ ] **Step 3: Implement the three-input dataset and model contract**

```python
all_token_type_ids.append(tokenized["token_type_ids"])
np_token_type_ids = np.asarray(all_token_type_ids, dtype=np.int64)
np.save(os.path.join(output_dir, "token_type_ids.npy"), np_token_type_ids)
```

Load this file with `mmap_mode="r"`, include it in `_build_payload`, and thread it through `load_single`, `load_batch`, and `load_by_index`. Export ONNX with:

```python
(
    inputs["input_ids"],
    inputs["attention_mask"],
    inputs["token_type_ids"],
)
```

and `input_names=["input_ids", "attention_mask", "token_type_ids"]`.

- [ ] **Step 4: Run QA tests and verify GREEN**

Run: `/home/swlab-youngjin/runtime-test/.venv/bin/python -m pytest -q tests/test_bert_qa_loader.py tests/test_bert_qa_evaluator.py tests/test_bert_qa_contract.py tests/test_plugin_registry.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the QA contract**

```bash
git add framework/src/core/model_profiles.py framework/src/dataloader/bert_qa_loader.py framework/src/preprocessor/bert_qa_preprocessor.py framework/datasets/prepare_squad_numpy.py framework/models/prepare_bert_squad.py framework/tests/test_bert_qa_loader.py framework/tests/test_bert_qa_contract.py
git commit -m "feat: complete BERT SQuAD input contract"
```

### Task 4: Document artifact creation and server verification

**Files:**
- Modify: `framework/docs/rbln-setup.md`
- Modify: `framework/README.md`

**Interfaces:**
- Consumes: three-input `.rbln` artifact and its SHA-bound sidecar.
- Produces: exact compile/copy/inspect/smoke/full/context-cleanup commands for the RBLN server.

- [ ] **Step 1: Update the contract table and sidecar schema**

Document the three BERT inputs and explain that the SDK 0.11 inspect result may contain two unnamed outputs. Include this exact sidecar example:

```json
{
  "schema_version": 1,
  "artifact_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "output_names": ["start_logits", "end_logits"]
}
```

- [ ] **Step 2: Add validation commands**

```bash
sha256sum models/rbln/bert-base-uncased-squad-v1/model.rbln
python -m src.main \
  --model bert-base-uncased-squad-v1 \
  --target rbln-static \
  --artifact models/rbln/bert-base-uncased-squad-v1/model.rbln \
  --dataset datasets/squad_numpy \
  --inference-mode e2e \
  --batch-size 1 \
  --warmup 2 \
  --max-steps 10 \
  --monitor \
  --results-path results/rbln-bert-squad-e2e-smoke.csv
rbln-smi -j
```

Require a CPU-vs-NPU one-sample comparison to confirm tuple position 0 is `start_logits` and position 1 is `end_logits` before creating the manifest. Require final `contexts: []` after unload.

- [ ] **Step 3: Verify documentation references**

Run: `rg -n 'bert-base-uncased-squad-v1|token_type_ids|artifact_sha256|output_names' framework/docs/rbln-setup.md framework/README.md`

Expected: the three-input contract, sidecar schema, smoke command, and context check are all present.

- [ ] **Step 4: Commit documentation**

```bash
git add framework/docs/rbln-setup.md framework/README.md
git commit -m "docs: add RBLN BERT SQuAD validation guide"
```

### Task 5: Regression verification and branch handoff

**Files:**
- Verify: all files changed in Tasks 1–4.

**Interfaces:**
- Consumes: completed implementation and tests.
- Produces: a pushed `feat/rbln-runtime-monitor` branch ready for RBLN-CA22 validation.

- [ ] **Step 1: Run the focused suite**

Run: `/home/swlab-youngjin/runtime-test/.venv/bin/python -m pytest -q tests/test_rbln_manifest.py tests/test_rbln_runtime.py tests/test_rbln_native_backend.py tests/test_bert_qa_loader.py tests/test_bert_qa_evaluator.py tests/test_bert_qa_contract.py tests/test_plugin_registry.py tests/test_main_paths.py`

Expected: all tests pass.

- [ ] **Step 2: Run the complete framework suite**

Run: `/home/swlab-youngjin/runtime-test/.venv/bin/python -m pytest -q`

Expected: all available tests pass with only existing documented skips.

- [ ] **Step 3: Inspect the final diff**

Run: `git status --short && git diff --check && git log --oneline origin/feat/rbln-runtime-monitor..HEAD`

Expected: no whitespace errors, only intended files changed, and all task commits are listed.

- [ ] **Step 4: Push the feature branch**

```bash
git push origin feat/rbln-runtime-monitor
```

- [ ] **Step 5: Validate on the RBLN server**

Recompile with three inputs, compare CPU/NPU output order, generate the SHA-bound sidecar, prepare SQuAD NumPy data, run 10-sample smoke and full validation, and confirm `rbln-smi -j` reports `contexts: []` after process exit.
