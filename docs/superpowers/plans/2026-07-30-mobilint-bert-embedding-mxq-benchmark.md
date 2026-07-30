# Mobilint BERT Embedding-MXQ Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the compiled SST-2 and SQuAD v1 embedding-input MXQ artifacts through the existing framework CLI on the `mobilint-aries` target while excluding host embedding construction from runtime latency.

**Architecture:** Keep the existing BERT loaders and inject one optional host input transform; do not wrap them. Select an explicit Mobilint BERT artifact profile that replaces the runtime-facing `Model_Spec`, then let the existing `MobilintRuntime` validate dynamic MXQ shapes, bind positional outputs, and normalize SDK singleton dimensions before the existing evaluators run.

**Tech Stack:** Python 3.12 runtime, NumPy, PyTorch, pytest, Mobilint qb Runtime v1.3.2, ARIES driver 1.13.0, qbcompiler 1.2 MXQ artifacts.

## Global Constraints

- Work only on `feat/mobilint-bert-mxq-benchmark` in `/home/swlab-youngjin/ML-HW-Benchmark-Framework/.worktrees/mobilint-bert-mxq-benchmark`.
- Preserve the existing ONNX/CPU/CUDA BERT token-input behavior when no input transform is supplied.
- Use `mobilint-aries`, not `mobilint-aries-llm` and not a Mobilint vision profile.
- Support both `bert-base-uncased` SST-2 and `bert-base-uncased-squad-v1` SQuAD v1.
- Accept exactly one MXQ input named `embeddings`, dtype `float32`, logical shape `(1, -1, 768)`.
- Keep SST-2 SDK output order `logits` and SQuAD SDK output order `end_logits`, `start_logits`.
- Support batch size exactly `1`; reject larger batches before qbruntime model allocation.
- Keep `native_async_supported=False` for both CPU-offload BERT artifacts.
- Perform token-to-embedding conversion in the loader path before `BlockingRuntimeExecutor` starts timing `runtime.run()`.
- Permit `-1` only as a single tensor dimension wildcard; concrete arrays must have positive dimensions.
- Do not add a wrapper BERT loader, a BERT-specific runtime, or a BERT-specific decoder.
- Do not commit MXQ files, embedding weights, datasets, or other model payloads.
- Follow RED → GREEN → REFACTOR for every behavior change.

---

### Task 1: Declare the two real Mobilint BERT artifact profiles

**Files:**
- Create: `framework/src/core/mobilint_bert_profiles.py`
- Create: `framework/tests/test_mobilint_bert_profiles.py`

**Interfaces:**
- Consumes: an existing token-input `Model_Spec`, exact model name, and task.
- Produces: `MobilintBertArtifactProfile`, `resolve_mobilint_bert_profile(model_name: str, task: Task) -> MobilintBertArtifactProfile | None`, and `apply_mobilint_bert_profile(spec: Model_Spec, profile: MobilintBertArtifactProfile) -> Model_Spec`.

- [ ] **Step 1: Write failing profile tests**

```python
def test_sst2_profile_declares_embedding_mxq_boundary():
    profile = resolve_mobilint_bert_profile(
        "bert-base-uncased", Task.NLP_CLASSIFICATION
    )
    adapted = apply_mobilint_bert_profile(_token_sst2_spec(), profile)

    assert profile.profile_id == "mobilint-bert-sst2-embedding-v1"
    assert profile.max_batch_size == 1
    assert profile.native_async_supported is False
    assert adapted.input_shapes == {"embeddings": (1, -1, 768)}
    assert adapted.input_dtype == {"embeddings": "float32"}
    assert adapted.output_shapes == {"logits": (1, 2)}


def test_squad_profile_binds_verified_sdk_output_order():
    profile = resolve_mobilint_bert_profile(
        "bert-base-uncased-squad-v1", Task.QUESTION_ANSWERING
    )
    adapted = apply_mobilint_bert_profile(_token_squad_spec(), profile)

    assert profile.profile_id == "mobilint-bert-squad1-embedding-v1"
    assert tuple(adapted.output_shapes) == ("end_logits", "start_logits")
    assert adapted.output_shapes == {
        "end_logits": (1, -1),
        "start_logits": (1, -1),
    }
```

Also test that a model/task mismatch raises `ValueError`, an unrelated model returns `None`, the adapted spec preserves `name`, `task`, and `model_paths`, and both profiles declare embedding width `768`.

- [ ] **Step 2: Run the tests and verify RED**

```bash
PYTHONPATH=framework/src /home/swlab-youngjin/runtime-test/.venv/bin/python \
  -m pytest framework/tests/test_mobilint_bert_profiles.py -q
```

Expected: collection fails because `core.mobilint_bert_profiles` does not exist.

- [ ] **Step 3: Implement immutable artifact profiles**

```python
@dataclass(frozen=True)
class MobilintBertArtifactProfile:
    model_name: str
    task: Task
    profile_id: str
    input_shapes: dict[str, tuple[int, ...]]
    input_dtype: dict[str, str]
    output_shapes: dict[str, tuple[int, ...]]
    embedding_width: int = 768
    max_batch_size: int = 1
    native_async_supported: bool = False


MOBILINT_BERT_SST2 = MobilintBertArtifactProfile(
    model_name="bert-base-uncased",
    task=Task.NLP_CLASSIFICATION,
    profile_id="mobilint-bert-sst2-embedding-v1",
    input_shapes={"embeddings": (1, -1, 768)},
    input_dtype={"embeddings": "float32"},
    output_shapes={"logits": (1, 2)},
)

MOBILINT_BERT_SQUAD1 = MobilintBertArtifactProfile(
    model_name="bert-base-uncased-squad-v1",
    task=Task.QUESTION_ANSWERING,
    profile_id="mobilint-bert-squad1-embedding-v1",
    input_shapes={"embeddings": (1, -1, 768)},
    input_dtype={"embeddings": "float32"},
    output_shapes={
        "end_logits": (1, -1),
        "start_logits": (1, -1),
    },
)
```

Return a new frozen `Model_Spec` from `apply_mobilint_bert_profile()`; never mutate the token-input spec or `SUPPORTED_PROFILES`.

- [ ] **Step 4: Run profile tests and verify GREEN**

Run the command from Step 2. Expected: all profile tests pass.

- [ ] **Step 5: Commit the profile boundary**

```bash
git add framework/src/core/mobilint_bert_profiles.py \
  framework/tests/test_mobilint_bert_profiles.py
git commit -m "feat: declare Mobilint BERT MXQ profiles"
```

### Task 2: Build the shared host embedding transform

**Files:**
- Create: `framework/src/preprocessor/mobilint_bert_embedding.py`
- Create: `framework/tests/test_mobilint_bert_embedding.py`

**Interfaces:**
- Consumes: `weight_dict.pth` with keys `word_embeddings`, `token_type_embeddings`, `position_embeddings`, `layernorm_weight`, `layernorm_bias`; token arrays named `input_ids`, `attention_mask`, and optional `token_type_ids`.
- Produces: `MobilintBertEmbeddingTransform(weights_path: str | Path, *, expected_width: int = 768)` callable returning `{"embeddings": contiguous_float32_array}`.

- [ ] **Step 1: Write failing numerical and validation tests**

```python
def test_transform_trims_padding_and_builds_float32_embedding(tmp_path):
    weights_path = _save_weights(tmp_path, width=4, max_positions=8)
    transform = MobilintBertEmbeddingTransform(
        weights_path, expected_width=4
    )
    result = transform({
        "input_ids": np.array([[2, 3, 0, 0]], dtype=np.int64),
        "attention_mask": np.array([[1, 1, 0, 0]], dtype=np.int64),
    })["embeddings"]

    assert result.shape == (1, 2, 4)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    expected = _reference_embedding(
        weights_path,
        input_ids=np.array([[2, 3]], dtype=np.int64),
        token_type_ids=np.array([[0, 0]], dtype=np.int64),
    )
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_transform_preserves_unbatched_loader_samples(tmp_path):
    transform = MobilintBertEmbeddingTransform(
        _save_weights(tmp_path, width=4), expected_width=4
    )
    result = transform({
        "input_ids": np.array([2, 3, 0], dtype=np.int64),
        "attention_mask": np.array([1, 1, 0], dtype=np.int64),
        "token_type_ids": np.array([0, 1, 0], dtype=np.int64),
    })["embeddings"]
    assert result.shape == (2, 4)
```

Add separate tests for missing weight keys, incompatible embedding widths, mismatched token array shapes, all-zero masks, non-prefix masks such as `[1, 0, 1]`, token IDs outside the word table, sequences longer than the position table, unexpected input keys, and batched input with `N=2`.

- [ ] **Step 2: Run transform tests and verify RED**

```bash
PYTHONPATH=framework/src /home/swlab-youngjin/runtime-test/.venv/bin/python \
  -m pytest framework/tests/test_mobilint_bert_embedding.py -q
```

Expected: collection fails because `preprocessor.mobilint_bert_embedding` does not exist.

- [ ] **Step 3: Implement one-time weight loading and batch-size-one conversion**

```python
class MobilintBertEmbeddingTransform:
    REQUIRED_KEYS = (
        "word_embeddings",
        "token_type_embeddings",
        "position_embeddings",
        "layernorm_weight",
        "layernorm_bias",
    )

    def __init__(self, weights_path, *, expected_width=768):
        import torch
        document = torch.load(
            Path(weights_path), map_location="cpu", weights_only=True
        )
        self._weights = self._validate_weights(document, expected_width)
        self.expected_width = expected_width

    def __call__(self, inputs):
        input_ids, attention_mask, token_type_ids, was_unbatched = (
            self._normalize_token_inputs(inputs)
        )
        valid_tokens = self._valid_prefix_length(attention_mask)
        input_ids = input_ids[:, :valid_tokens]
        token_type_ids = token_type_ids[:, :valid_tokens]
        positions = torch.arange(valid_tokens, dtype=torch.long).unsqueeze(0)
        embedded = (
            functional.embedding(input_ids, self._weights["word_embeddings"])
            + functional.embedding(token_type_ids, self._weights["token_type_embeddings"])
            + functional.embedding(positions, self._weights["position_embeddings"])
        )
        embedded = functional.layer_norm(
            embedded,
            (self.expected_width,),
            weight=self._weights["layernorm_weight"],
            bias=self._weights["layernorm_bias"],
            eps=1e-12,
        ).to(dtype=torch.float32)
        array = embedded.numpy()
        if was_unbatched:
            array = array[0]
        return {"embeddings": np.ascontiguousarray(array)}
```

Normalize one-dimensional samples to a temporary batch and restore the unbatched result. Validate all weight shapes before accepting the transform.

- [ ] **Step 4: Run transform tests and verify GREEN**

Run the command from Step 2. Expected: all transform tests pass.

- [ ] **Step 5: Commit the transform**

```bash
git add framework/src/preprocessor/mobilint_bert_embedding.py \
  framework/tests/test_mobilint_bert_embedding.py
git commit -m "feat: build Mobilint BERT host embeddings"
```

### Task 3: Add one optional transform hook to the existing BERT loaders

**Files:**
- Modify: `framework/src/dataloader/bert_classification_loader.py`
- Modify: `framework/src/dataloader/bert_qa_loader.py`
- Modify: `framework/tests/test_bert_classification_loader.py`
- Modify: `framework/tests/test_bert_qa_loader.py`

**Interfaces:**
- Consumes: optional keyword `input_transform: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]`.
- Produces: transformed `payload["input"]` from `load_single`, `load_batch`, and `load_by_index`; unchanged payloads when absent.

- [ ] **Step 1: Write failing loader-hook tests**

```python
@pytest.mark.parametrize("method,args", [
    ("load_single", ()),
    ("load_batch", (1,)),
    ("load_by_index", (0,)),
])
def test_classification_loader_applies_input_transform(
    dummy_bert_spec, tmp_path, method, args
):
    dataset_dir = create_mock_dataset(tmp_path, 2)
    calls = []

    def transform(inputs):
        calls.append(tuple(inputs))
        return {"embeddings": np.zeros((1, 2, 4), dtype=np.float32)}

    loader = BertClassificationLoader(
        model_spec=dummy_bert_spec,
        dataset_path=dataset_dir,
        input_transform=transform,
    )
    payload = getattr(loader, method)(*args)
    assert tuple(payload["input"]) == ("embeddings",)
    assert calls == [("input_ids", "attention_mask")]
```

Add equivalent QA cases expecting `input_ids`, `attention_mask`, `token_type_ids`. Preserve exact token dictionaries without a transform and reject a transform returning a non-dictionary or empty mapping.

- [ ] **Step 2: Run loader tests and verify RED**

```bash
PYTHONPATH=framework/src /home/swlab-youngjin/runtime-test/.venv/bin/python \
  -m pytest framework/tests/test_bert_classification_loader.py \
  framework/tests/test_bert_qa_loader.py -q
```

Expected: transformed payload assertions fail because the keyword is ignored.

- [ ] **Step 3: Implement the hook without a wrapper loader**

In both loaders, store `self.input_transform = kwargs.get("input_transform")` and centralize the call:

```python
def _transform_input(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if self.input_transform is None:
        return inputs
    transformed = self.input_transform(inputs)
    if not isinstance(transformed, dict) or not transformed:
        raise TypeError("input_transform must return a non-empty input dictionary")
    return transformed
```

Call `_transform_input()` inside the existing `_build_payload()` methods and leave cursor, labels, mmap, and metadata untouched.

- [ ] **Step 4: Run loader tests and verify GREEN**

Run the command from Step 2. Expected: all existing and new loader tests pass.

- [ ] **Step 5: Commit the loader extension point**

```bash
git add framework/src/dataloader/bert_classification_loader.py \
  framework/src/dataloader/bert_qa_loader.py \
  framework/tests/test_bert_classification_loader.py \
  framework/tests/test_bert_qa_loader.py
git commit -m "feat: transform BERT loader inputs"
```

### Task 4: Support dynamic named MXQ tensor contracts

**Files:**
- Modify: `framework/src/core/mobilint_tensor_contracts.py`
- Modify: `framework/src/runtimes/mobilint_rt.py`
- Modify: `framework/tests/test_mobilint_tensor_contracts.py`
- Modify: `framework/tests/test_mobilint_runtime.py`

**Interfaces:**
- Extends: `build_mobilint_tensor_contract(spec: Model_Spec, *, max_batch_size: int, profile_id: str | None = None, native_async_supported: bool = False) -> MobilintTensorContract`.
- Produces: exact-rank wildcard matching for `-1`, dynamic SDK metadata validation, and positive concrete runtime-input validation.

- [ ] **Step 1: Write failing dynamic-contract tests**

```python
def test_embedding_contract_preserves_dynamic_sequence_dimension():
    spec = Model_Spec(
        name="bert-base-uncased",
        task=Task.NLP_CLASSIFICATION,
        input_shapes={"embeddings": (1, -1, 768)},
        input_dtype={"embeddings": "float32"},
        output_shapes={"logits": (1, 2)},
    )
    contract = build_mobilint_tensor_contract(
        spec,
        max_batch_size=1,
        profile_id="mobilint-bert-sst2-embedding-v1",
    )
    assert contract.runtime_contract()["expected_unbatched_input_shapes"] == [
        [-1, 768]
    ]


def test_dynamic_runtime_contract_accepts_sdk_wildcard_and_concrete_input(
    monkeypatch, tmp_path
):
    state = _install_fake_qbruntime(monkeypatch)
    state["input_shapes"] = [(1, -1, 768)]
    state["input_dtypes"] = DataType.Float32
    state["output_shapes"] = [(1, 2)]
    artifact = tmp_path / "sst2.mxq"
    artifact.write_bytes(b"fake")
    spec = Model_Spec(
        name="bert-base-uncased",
        task=Task.NLP_CLASSIFICATION,
        input_shapes={"embeddings": (1, -1, 768)},
        input_dtype={"embeddings": "float32"},
        output_shapes={"logits": (1, 2)},
    )
    contract = build_mobilint_tensor_contract(
        spec,
        max_batch_size=1,
        profile_id="mobilint-bert-sst2-embedding-v1",
    )
    runtime = MobilintRuntime(
        expected_family="aries", **contract.runtime_contract()
    )
    runtime.load(CompiledModel(spec, "mobilint", artifact))
    state["models"][0].outputs = [
        np.array([[0.25, 0.75]], dtype=np.float32)
    ]
    outputs = runtime.run({
        "embeddings": np.zeros((1, 9, 768), dtype=np.float32)
    })
    assert "logits" in outputs
```

Add tests rejecting `0`, `-2`, dynamic batch axes, wrong embedding width, wrong runtime rank, and concrete zero-length dimensions. Assert existing static transformer and Mobilint vision contracts retain current behavior.

- [ ] **Step 2: Run focused contract tests and verify RED**

```bash
PYTHONPATH=framework/src /home/swlab-youngjin/runtime-test/.venv/bin/python \
  -m pytest framework/tests/test_mobilint_tensor_contracts.py \
  framework/tests/test_mobilint_runtime.py -k 'dynamic or tensor_contract' -q
```

Expected: the builder rejects `-1` as non-positive.

- [ ] **Step 3: Implement exact-rank dynamic matching**

```python
def _shape_matches(expected, actual):
    return len(expected) == len(actual) and all(
        expected_dimension == -1 or expected_dimension == actual_dimension
        for expected_dimension, actual_dimension in zip(expected, actual)
    )
```

Change `_unbatched_shape()` so the batch axis remains positive while following dimensions may be positive or exactly `-1`. Add the optional profile identifier and native-async flag to the contract builder and require a non-empty supplied profile identifier.

In `MobilintRuntime`, add `allow_dynamic=False` to `_normalize_shape()`. Use `allow_dynamic=True` only for named tensor contract declarations and their SDK metadata. Keep vision contracts strict. Replace direct tuple equality in metadata and concrete runtime-array checks with `_shape_matches()`.

- [ ] **Step 4: Run full Mobilint contract/runtime tests and verify GREEN**

```bash
PYTHONPATH=framework/src /home/swlab-youngjin/runtime-test/.venv/bin/python \
  -m pytest framework/tests/test_mobilint_tensor_contracts.py \
  framework/tests/test_mobilint_runtime.py -q
```

Expected: all tests pass, including existing static and vision cases.

- [ ] **Step 5: Commit dynamic contract support**

```bash
git add framework/src/core/mobilint_tensor_contracts.py \
  framework/src/runtimes/mobilint_rt.py \
  framework/tests/test_mobilint_tensor_contracts.py \
  framework/tests/test_mobilint_runtime.py
git commit -m "feat: validate dynamic Mobilint tensor shapes"
```

### Task 5: Normalize singleton-heavy outputs and configure qb Runtime v1.3 single-core mode

**Files:**
- Modify: `framework/src/runtimes/mobilint_rt.py`
- Modify: `framework/tests/test_mobilint_runtime.py`

**Interfaces:**
- Consumes: expected batch size, ordered named-tensor output shapes, and qb Runtime v1.3 `CoreId` enums.
- Produces: evaluator-ready output arrays and a valid Cluster0/Core0 single-core configuration.

- [ ] **Step 1: Write failing output and core-mode tests**

```python
def test_tensor_outputs_are_reshaped_to_logical_batched_shapes():
    runtime = MobilintRuntime(
        expected_family="aries",
        artifact_profile_id="mobilint-bert-sst2-embedding-v1",
        expected_input_names=["embeddings"],
        expected_input_dtypes=["float32"],
        expected_unbatched_input_shapes=[[-1, 768]],
        expected_output_names=["logits"],
        expected_unbatched_output_shapes=[[2]],
        max_input_batch_size=1,
        native_async_supported=False,
    )
    runtime._output_names = ("logits",)
    outputs = runtime._normalize_outputs(
        [np.zeros((1, 1, 1, 2), dtype=np.float32)],
        expected_batch_size=1,
    )
    assert outputs["logits"].shape == (1, 2)


def test_squad_outputs_bind_reversed_sdk_order_and_flatten():
    runtime = MobilintRuntime(
        expected_family="aries",
        artifact_profile_id="mobilint-bert-squad1-embedding-v1",
        expected_input_names=["embeddings"],
        expected_input_dtypes=["float32"],
        expected_unbatched_input_shapes=[[-1, 768]],
        expected_output_names=["end_logits", "start_logits"],
        expected_unbatched_output_shapes=[[-1], [-1]],
        max_input_batch_size=1,
        native_async_supported=False,
    )
    runtime._output_names = ("end_logits", "start_logits")
    outputs = runtime._normalize_outputs(
        [np.full((1, 1, 9, 1), 2.0), np.full((1, 1, 9, 1), 1.0)],
        expected_batch_size=1,
    )
    assert outputs["end_logits"].shape == (1, 9)
    assert outputs["start_logits"].shape == (1, 9)
    assert np.all(outputs["end_logits"] == 2.0)
```

Add a fake qb Runtime v1.3 test asserting default `core_mode="single"` calls `set_single_core_mode(None, [CoreId(Cluster0, Core0)])`. Add output failures for incompatible element counts, unresolvable multiple wildcards, and invalid requested batches. Preserve raw vision output behavior.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
PYTHONPATH=framework/src /home/swlab-youngjin/runtime-test/.venv/bin/python \
  -m pytest framework/tests/test_mobilint_runtime.py \
  -k 'logical_batched or reversed_sdk or cluster0_core0' -q
```

Expected: singleton-heavy arrays remain raw and the fake config records the old zero-argument single-core call.

- [ ] **Step 3: Implement ordered logical output reshaping**

For named tensor contracts only, resolve each expected `-1` from `array.size // batch_size`, validate divisibility, and reshape to `(batch_size, *resolved_unbatched_shape)` before creating the named dictionary. Validate outputs positionally so equal-shaped QA arrays retain declared order. Leave counter-based vision validation unchanged.

Configure default single-core mode with:

```python
core_id = qbruntime.CoreId(
    qbruntime.Cluster.Cluster0,
    qbruntime.Core.Core0,
)
result = config.set_single_core_mode(None, [core_id])
```

Retain existing explicit `num_cores` behavior when supplied.

- [ ] **Step 4: Run the Mobilint runtime suite and verify GREEN**

```bash
PYTHONPATH=framework/src /home/swlab-youngjin/runtime-test/.venv/bin/python \
  -m pytest framework/tests/test_mobilint_runtime.py \
  framework/tests/test_mobilint_native_backend.py -q
```

Expected: all tests pass with no vision or native-async regression.

- [ ] **Step 5: Commit runtime normalization**

```bash
git add framework/src/runtimes/mobilint_rt.py \
  framework/tests/test_mobilint_runtime.py
git commit -m "feat: normalize Mobilint tensor outputs"
```

### Task 6: Wire both BERT profiles into the framework CLI

**Files:**
- Modify: `framework/src/main.py`
- Modify: `framework/tests/test_main_paths.py`

**Interfaces:**
- Adds: CLI option `--mobilint-bert-weights PATH`.
- Produces: `_prepare_mobilint_bert_execution(args, target, spec) -> tuple[MobilintBertArtifactProfile | None, Model_Spec, MobilintBertEmbeddingTransform | None]` and the matching named tensor runtime contract.

- [ ] **Step 1: Write failing parser, validation, and wiring tests**

```python
def test_parser_accepts_mobilint_bert_weights():
    args = benchmark_main.build_parser().parse_args([
        "--model", "bert-base-uncased",
        "--mobilint-bert-weights", "/models/weight_dict.pth",
    ])
    assert args.mobilint_bert_weights == "/models/weight_dict.pth"


def test_mobilint_bert_validation_rejects_batch_two_before_transform(tmp_path):
    args = SimpleNamespace(
        model="bert-base-uncased",
        batch_size=2,
        mobilint_bert_weights=str(tmp_path / "weight_dict.pth"),
    )
    target = SimpleNamespace(target_id="mobilint-aries")
    spec = Model_Spec(
        name="bert-base-uncased",
        task=Task.NLP_CLASSIFICATION,
        input_shapes={
            "input_ids": (1, 128),
            "attention_mask": (1, 128),
        },
        input_dtype={
            "input_ids": "int64",
            "attention_mask": "int64",
        },
        output_shapes={"logits": (1, 2)},
    )
    with pytest.raises(ValueError, match="batch size exactly 1"):
        benchmark_main._prepare_mobilint_bert_execution(
            args, target, spec
        )


def assert_mobilint_squad_wiring(captured):
    spec = captured["compiled_model"].spec
    runtime_options = captured["runtime_request"][1]
    assert tuple(spec.input_shapes) == ("embeddings",)
    assert tuple(spec.output_shapes) == ("end_logits", "start_logits")
    assert callable(captured["loader_kwargs"]["input_transform"])
    assert runtime_options["artifact_profile_id"] == (
        "mobilint-bert-squad1-embedding-v1"
    )
    assert runtime_options["native_async_supported"] is False
```

Add the SST-2 main-path case, missing/non-file weight failures, rejection on `mobilint-regulus`, unchanged ONNX BERT wiring, and proof that validation finishes before runtime loading.

- [ ] **Step 2: Run focused main-path tests and verify RED**

```bash
PYTHONPATH=framework/src /home/swlab-youngjin/runtime-test/.venv/bin/python \
  -m pytest framework/tests/test_main_paths.py \
  -k 'mobilint_bert or mobilint_squad' -q
```

Expected: parser rejects the new option and current Mobilint BERT wiring still declares token inputs.

- [ ] **Step 3: Implement fail-fast CLI assembly**

```python
parser.add_argument(
    "--mobilint-bert-weights",
    type=str,
    default=None,
    help="Host BERT embedding weight_dict.pth for Mobilint embedding-input MXQ",
)
```

`_prepare_mobilint_bert_execution()` returns `(None, spec, None)` for unrelated paths. For a supported BERT model it requires `target.target_id == "mobilint-aries"`, batch size `1`, and a regular weight file before constructing `MobilintBertEmbeddingTransform`.

In `main()`, create the original token spec, apply the Mobilint BERT profile, build the tensor contract with its explicit profile ID and native-async flag, and set `loader_kwargs["input_transform"]`. Continue through the existing generic BERT loader and evaluator factories. Do not add dataloader or decoder factory branches.

- [ ] **Step 4: Run main-path and BERT component tests and verify GREEN**

```bash
PYTHONPATH=framework/src /home/swlab-youngjin/runtime-test/.venv/bin/python \
  -m pytest framework/tests/test_main_paths.py \
  framework/tests/test_bert_classification_loader.py \
  framework/tests/test_bert_qa_loader.py \
  framework/tests/test_bert_classification_evaluator.py \
  framework/tests/test_bert_qa_evaluator.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit CLI integration**

```bash
git add framework/src/main.py framework/tests/test_main_paths.py
git commit -m "feat: run Mobilint BERT MXQ benchmarks"
```

### Task 7: Update the ARIES runbook and perform complete verification

**Files:**
- Modify: `docs/mobilint-aries-transformers.md`
- Modify: `framework/README.md`
- Modify: `framework/tests/test_main_paths.py`

**Interfaces:**
- Produces: paste-safe server commands for both compiled artifacts and an explicit host/NPU timing boundary.

- [ ] **Step 1: Write failing documentation assertions**

```python
def test_mobilint_bert_runbook_documents_embedding_artifacts():
    runbook = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "mobilint-aries-transformers.md"
    ).read_text(encoding="utf-8")
    assert "mobilint-bert-sst2-embedding-v1" in runbook
    assert "mobilint-bert-squad1-embedding-v1" in runbook
    assert "--mobilint-bert-weights" in runbook
    assert "BERT SST-2 | `input_ids`, `attention_mask`" not in runbook
```

Keep the parser-help assertion in `test_parser_help_mentions_explicit_mobilint_targets_and_mxq_artifacts()` and require the new option's help to mention `weight_dict.pth`.

- [ ] **Step 2: Run documentation assertions and verify RED**

```bash
PYTHONPATH=framework/src /home/swlab-youngjin/runtime-test/.venv/bin/python \
  -m pytest framework/tests/test_main_paths.py -k 'runbook or parser_help' -q
```

Expected: the runbook still claims MXQ token inputs and lacks the weight option.

- [ ] **Step 3: Document real paths and commands**

```bash
REPO="$HOME/ML-HW-Benchmark-Framework"
FW="$REPO/framework"
PY="$REPO/.venv-mobilint/bin/python"
WORK="$REPO/.mobilint-bert-tasks-20260730-105143"

SST2_MXQ="$WORK/artifacts/sst2/mxq/sst2.mxq"
SST2_WEIGHTS="$WORK/artifacts/sst2/weights/weight_dict.pth"
SQUAD_MXQ="$WORK/artifacts/squad1/mxq/squad1.mxq"
SQUAD_WEIGHTS="$WORK/artifacts/squad1/weights/weight_dict.pth"
```

Document separate `main.py` commands with `--target mobilint-aries`, `--inference-mode e2e`, `--batch-size 1`, `--warmup 2`, `--max-steps 64`, `--runtime-option core_mode=single`, `--no-compile`, and matching `--mobilint-bert-weights`. State that runtime metrics include `MobilintRuntime.run()` but exclude loader-side embedding construction.

- [ ] **Step 4: Run the complete host regression suite**

```bash
PYTHONPATH=framework/src /home/swlab-youngjin/runtime-test/.venv/bin/python \
  -m pytest framework/tests/test_mobilint_bert_profiles.py \
  framework/tests/test_mobilint_bert_embedding.py \
  framework/tests/test_mobilint_tensor_contracts.py \
  framework/tests/test_mobilint_runtime.py \
  framework/tests/test_mobilint_native_backend.py \
  framework/tests/test_bert_classification_loader.py \
  framework/tests/test_bert_qa_loader.py \
  framework/tests/test_bert_classification_evaluator.py \
  framework/tests/test_bert_qa_evaluator.py \
  framework/tests/test_main_paths.py \
  framework/tests/test_plugin_registry.py -q
```

Expected: zero failures. Then run `git diff --check` and confirm only intended branch changes.

- [ ] **Step 5: Commit documentation and host verification**

```bash
git add docs/mobilint-aries-transformers.md framework/README.md \
  framework/tests/test_main_paths.py
git commit -m "docs: run Mobilint BERT benchmarks on ARIES"
```

- [ ] **Step 6: Run both 64-sample ARIES benchmarks**

On the ARIES server, check `mobilint-cli status`, then run the documented SST-2 command and require accuracy consistent with `59/64`. Run SQuAD and require non-degenerate start/end metrics with recorded output order `end_logits,start_logits`. After each run require exit code `0`, a committed result row, model disposal, and no remaining ARIES process.

- [ ] **Step 7: Record server evidence without generated payloads**

Record the server commit SHA, qbruntime `v1.3.2`, driver `1.13.0`, MXQ SHA256 values, sample counts, accuracy and latency metrics, and exit codes in the PR description or runbook. Do not add MXQ, `.pth`, `.npy`, result CSV, or log payloads to Git.
