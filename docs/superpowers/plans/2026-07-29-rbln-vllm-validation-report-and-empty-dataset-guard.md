# RBLN vLLM Validation Report and Empty Dataset Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent zero-sample benchmarks from being persisted as successful runs and record the reproducible single-ATOM Llama 3B/8B compilation, troubleshooting, and hardware validation evidence for PR #40.

**Architecture:** Add one postcondition at the dataset auto-preparation boundary and one capability-based validation immediately after dataloader construction. Keep loaders without a concrete `total_samples` count backward compatible. Store only commands, run IDs, and summarized hardware evidence in a dedicated validation report, and link it from the existing RBLN vLLM runbook.

**Tech Stack:** Python 3.10, pytest, argparse CLI, existing `framework/src/main.py` assembly path, Markdown documentation, Git/GitHub Draft PR #40.

## Global Constraints

- Preserve all existing static RBLN, Llama 3.2 3B, Mobilint, Furiosa, Hailo, ONNX, and streaming/custom loader behavior.
- Do not commit `.rbln` artifacts, model weights, tokenizers, datasets, caches, result CSVs, traces, logs, or credentials.
- Do not redirect an explicit missing dataset path to a guessed path or copy files between worktrees.
- Reject `total_samples == 0` only when `type(total_samples) is int`; loaders without a concrete count remain unchanged.
- Keep one-card Llama results classified as `unsupported_single_npu_experiment`, not official Rebellions support.
- Do not merge PR #40 without an explicit user instruction.
- Run local tests with `TEST_PY=/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python`; the isolated worktree intentionally has no duplicated virtual environment.

---

### Task 1: Enforce the dataset preparation postcondition

**Files:**
- Modify: `framework/src/main.py` (`run_auto_prepare` and a nearby helper)
- Test: `framework/tests/test_main_paths.py`

**Interfaces:**
- Consumes: `dataset_path: str | None` and `script: str` from `run_auto_prepare`.
- Produces: `_validate_prepared_dataset_path(dataset_path: str | None, script: str) -> None`.
- Raises: `FileNotFoundError` when a non-empty requested path still does not exist after the preparation script returns successfully.

- [ ] **Step 1: Add a failing test for a preparation script that writes to the wrong path**

Add this test next to the existing `run_auto_prepare` test in
`framework/tests/test_main_paths.py`:

```python
def test_run_auto_prepare_rejects_dataset_script_that_misses_requested_path(
    monkeypatch,
    tmp_path,
):
    requested = tmp_path / "other-worktree" / "squad2" / "val.json"
    generated = tmp_path / "current-worktree" / "squad2" / "val.json"

    def fake_prepare(script):
        generated.parent.mkdir(parents=True)
        generated.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(benchmark_main, "_run_prepare_script", fake_prepare)
    args = Namespace(
        backend="rbln_vllm",
        hef=None,
        artifact=None,
        compile=False,
        onnx=None,
        model_path=str(tmp_path / "prepared-model"),
        dataset=str(requested),
    )
    profile = {"prepare_dataset_script": "datasets/prepare_squad2.py"}
    target = SimpleNamespace(target_id="rbln-vllm")

    with pytest.raises(
        FileNotFoundError,
        match=r"prepare_squad2\.py.*requested dataset path.*other-worktree",
    ):
        benchmark_main.run_auto_prepare(profile, args, target)
```

- [ ] **Step 2: Run the failing postcondition test**

Run:

```bash
"$TEST_PY" -m pytest -q \
  framework/tests/test_main_paths.py::test_run_auto_prepare_rejects_dataset_script_that_misses_requested_path
```

Expected: FAIL because `run_auto_prepare` currently returns after the script
without verifying `args.dataset`.

- [ ] **Step 3: Add a failing success-path test**

```python
def test_run_auto_prepare_accepts_dataset_script_that_creates_requested_path(
    monkeypatch,
    tmp_path,
):
    requested = tmp_path / "squad2" / "val.json"

    def fake_prepare(script):
        requested.parent.mkdir(parents=True)
        requested.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(benchmark_main, "_run_prepare_script", fake_prepare)
    args = Namespace(
        backend="rbln_vllm",
        hef=None,
        artifact=None,
        compile=False,
        onnx=None,
        model_path=str(tmp_path / "prepared-model"),
        dataset=str(requested),
    )
    profile = {"prepare_dataset_script": "datasets/prepare_squad2.py"}
    target = SimpleNamespace(target_id="rbln-vllm")

    benchmark_main.run_auto_prepare(profile, args, target)

    assert requested.is_file()
```

- [ ] **Step 4: Implement the minimal postcondition helper**

Add near `run_auto_prepare` in `framework/src/main.py`:

```python
def _validate_prepared_dataset_path(
    dataset_path: str | None,
    script: str,
) -> None:
    if dataset_path and not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"dataset preparation script '{script}' completed but the "
            f"requested dataset path was not created: {dataset_path}"
        )
```

Then change the dataset branch in `run_auto_prepare` to:

```python
if "prepare_dataset_script" in profile and profile["prepare_dataset_script"]:
    if not dataset_path or not os.path.exists(dataset_path):
        script = profile["prepare_dataset_script"]
        print(
            "[*] 데이터셋 리소스 누락 감지. "
            f"자동 준비 스크립트 실행: {script}"
        )
        _run_prepare_script(script)
        _validate_prepared_dataset_path(dataset_path, script)
```

When `dataset_path` is `None`, preserve the existing behavior because there is
no exact requested path to validate.

- [ ] **Step 5: Run the focused and surrounding preparation tests**

```bash
"$TEST_PY" -m pytest -q \
  framework/tests/test_main_paths.py -k 'run_auto_prepare'
```

Expected: all selected tests PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add framework/src/main.py framework/tests/test_main_paths.py
git commit -m "fix: verify prepared dataset path"
```

---

### Task 2: Reject zero-sample dataloaders before runtime creation

**Files:**
- Modify: `framework/src/main.py` (new helper and call immediately after `create_dataloader`)
- Test: `framework/tests/test_main_paths.py`

**Interfaces:**
- Consumes: `loader: object`, `model_name: str`, `task_name: str`, and `dataset_path: str | None`.
- Produces: `_validate_dataloader_samples(loader, *, model_name: str, task_name: str, dataset_path: str | None) -> None`.
- Raises: `ValueError` only when `type(loader.total_samples) is int` and the value equals zero.

- [ ] **Step 1: Write helper-level failing tests**

```python
def test_validate_dataloader_samples_rejects_exact_zero():
    loader = SimpleNamespace(total_samples=0)

    with pytest.raises(
        ValueError,
        match=r"llama-3\.1-8b.*NLP_GENERATION.*dataset\.json.*zero samples",
    ):
        benchmark_main._validate_dataloader_samples(
            loader,
            model_name="llama-3.1-8b",
            task_name="NLP_GENERATION",
            dataset_path="/datasets/squad2/dataset.json",
        )


@pytest.mark.parametrize(
    "loader",
    [
        SimpleNamespace(total_samples=1),
        SimpleNamespace(total_samples=None),
        SimpleNamespace(),
        SimpleNamespace(total_samples=False),
    ],
)
def test_validate_dataloader_samples_preserves_supported_loader_contracts(
    loader,
):
    benchmark_main._validate_dataloader_samples(
        loader,
        model_name="model",
        task_name="TASK",
        dataset_path="/dataset",
    )
```

- [ ] **Step 2: Run the helper tests and verify RED**

```bash
"$TEST_PY" -m pytest -q \
  framework/tests/test_main_paths.py -k 'validate_dataloader_samples'
```

Expected: FAIL because `_validate_dataloader_samples` does not exist.

- [ ] **Step 3: Implement the helper**

Add near the dataloader assembly helpers in `framework/src/main.py`:

```python
def _validate_dataloader_samples(
    loader,
    *,
    model_name: str,
    task_name: str,
    dataset_path: str | None,
) -> None:
    total_samples = getattr(loader, "total_samples", None)
    if type(total_samples) is int and total_samples == 0:
        raise ValueError(
            f"dataloader for model={model_name}, task={task_name}, "
            f"dataset={dataset_path or '<unspecified>'} produced zero samples"
        )
```

- [ ] **Step 4: Call the helper before runtime creation**

Immediately after `loader = create_dataloader(...)` in `main()` add:

```python
_validate_dataloader_samples(
    loader,
    model_name=args.model,
    task_name=task_enum.name,
    dataset_path=args.dataset,
)
```

This call must remain before `create_runtime(...)`.

- [ ] **Step 5: Add an assembly-path test proving runtime creation is skipped**

Extend the existing RBLN vLLM Main test pattern:

```python
def test_rbln_vllm_main_rejects_empty_loader_before_runtime(
    monkeypatch,
    tmp_path,
):
    model_path = tmp_path / "prepared"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"model_type": "llama"}', encoding="utf-8"
    )
    (model_path / "decoder.rbln").write_bytes(b"compiled")
    (model_path / "tokenizer_config.json").write_text(
        "{}", encoding="utf-8"
    )
    (model_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_path / "rbln-vllm-manifest.json").write_text(
        json.dumps({"max_seq_len": 512}), encoding="utf-8"
    )
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("{}", encoding="utf-8")
    runtime_calls = []

    monkeypatch.setattr(
        benchmark_main,
        "create_model_spec",
        lambda *args, **kwargs: SimpleNamespace(
            task=benchmark_main.Task.NLP_GENERATION
        ),
    )
    monkeypatch.setattr(
        benchmark_main,
        "create_dataloader",
        lambda **kwargs: SimpleNamespace(
            total_samples=0,
            get_metadata=lambda: {},
        ),
    )
    monkeypatch.setattr(
        benchmark_main,
        "create_runtime",
        lambda *args, **kwargs: runtime_calls.append((args, kwargs)),
    )
    import utils.dataset_resolver as dataset_resolver
    monkeypatch.setattr(
        dataset_resolver,
        "resolve_dataset_paths",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--model",
            "llama-3.1-8b",
            "--target",
            "rbln-vllm",
            "--model-path",
            str(model_path),
            "--dataset",
            str(dataset_path),
            "--runtime-option",
            "block_size=512",
            "--runtime-option",
            "allow_unsupported_single_npu=true",
        ],
    )

    with pytest.raises(ValueError, match="produced zero samples"):
        benchmark_main.main()

    assert runtime_calls == []
```

If `main()` intentionally catches this assembly exception, assert
`SystemExit(1)` and capture the same public error message instead; do not move
the validation after runtime creation to satisfy the test.

- [ ] **Step 6: Run Main and result persistence regressions**

```bash
"$TEST_PY" -m pytest -q \
  framework/tests/test_main_paths.py \
  framework/tests/test_result_store.py
```

Expected: all tests PASS and no result-store positional compatibility failure.

- [ ] **Step 7: Commit Task 2**

```bash
git add framework/src/main.py framework/tests/test_main_paths.py
git commit -m "fix: reject empty benchmark dataloaders"
```

---

### Task 3: Record the physical ATOM Llama validation and troubleshooting

**Files:**
- Create: `framework/docs/rbln-vllm-atom-validation.md`
- Modify: `framework/docs/rbln-vllm-setup.md`

**Interfaces:**
- Consumes: verified commands and metrics from run IDs `a3168997` and `9dd3bf7a` plus the existing setup runbook.
- Produces: one evidence report linked from the operational runbook.

- [ ] **Step 1: Create the validation report with exact environment and scope**

Write `framework/docs/rbln-vllm-atom-validation.md` with these top-level
sections and exact facts:

```markdown
# Rebellions ATOM Llama vLLM 검증 보고서

## 1. 검증 범위
## 2. 서버와 Python 환경
## 3. 모델 준비 및 컴파일 계약
## 4. 실행 순서
## 5. 실제 검증 결과
## 6. 트러블슈팅 기록
## 7. 운영 및 병합 판정
```

Record:

- RBLN-CA22 one card, memory 16,096 MiB, KMD/firmware 3.2.2.
- Python 3.10.12, `rebel-compiler==0.11.0`,
  `optimum-rbln==0.11.0.post1`, `vllm-rbln==0.11.0`,
  `torch==2.11.0+cpu`, `transformers==5.8.1`,
  `tokenizers==0.22.1`.
- The global user-site supplies `rebel`; the project venv supplies the other
  Python packages; `PYTHONNOUSERSITE=1` must not be set for this server.
- The exact one-card compile command from the specification using sequence and
  block 512, batch and decoder batch 1, one device, and explicit opt-in.
- The exact sync and async commands, including `decoder_batch_sizes=1,` for the
  shared runtime-option parser.

- [ ] **Step 2: Add the measured result table**

Use a compact Markdown table containing:

| Mode | Run ID | Samples | Generated tokens | Engine latency/TTFT | Throughput | NPU memory peak | Process RAM peak | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| sync E2E | `a3168997` | 1 | 1 | 203.8591 ms | 4.9053 tokens/s | 14,630 MiB | 21,063.23 MiB | exit 0, contexts empty |
| async offline | `9dd3bf7a` | 4 | 4 | avg TTFT 202.7706 ms | 4.8801 tokens/s | 14,630 MiB | 21,260.79 MiB | valid, exit 0, contexts empty |

Also record async request E2E p99 604.8718 ms, queue wait p99 205.7612
ms, service p99 214.3395 ms, 4/4 completion, zero failure counters, monitor
coverage 1.0, and native async counters all zero.

- [ ] **Step 3: Record troubleshooting as cause → evidence → resolution**

Include these incidents without secrets:

1. Portal/wheel 401: authenticate to the authorized Rebellions index; do not
   commit credentials.
2. `rebel` absent from uv venv: preserve the validated hybrid user-site setup
   or install an authorized matching wheel; verify package origins explicitly.
3. uv dependency resolution for `vllm-rbln`: use the compatible RBLN/vLLM
   indices and the validated index strategy, then verify installed versions.
4. Single-NPU 8B initially rejected: require opt-in, manifest, 512 context,
   batch 1, decoder batch 1, and a 15 GiB readable-memory prerequisite.
5. Manifest-less or mismatched artifact: retain the complete prepared model
   directory and make compile/runtime contracts match exactly.
6. SQuAD path mismatch: the current-worktree script produced a file while the
   loader retained an old-worktree path; point `--dataset` at the exact file,
   assert a positive QA count, and rely on the new code guards.
7. `num_samples=0` false success evidence: NPU memory 0 MiB, P14, util 0, no
   EngineCore creation; mark run `4930bef2` invalid and exclude it.
8. TPOT/ITL zero or `None`: one generated token has no inter-token interval;
   use a multi-token run for TPOT benchmarking.
9. Short-run util 0: monitor sampling can miss a 204 ms burst; NPU memory,
   P-state, power, engine logs, exit status, and context cleanup provide the
   acceptance evidence.

- [ ] **Step 4: Update the operational runbook**

In `framework/docs/rbln-vllm-setup.md`:

- change the opening hardware status to state that one-card Llama 3.1 8B
  compile, sync, and async smoke passed;
- link `rbln-vllm-atom-validation.md` near that statement;
- note that one-token smoke validates capacity/lifecycle rather than TPOT or
  model quality;
- add the exact dataset-file and positive-QA preflight before smoke commands;
- retain the `unsupported_single_npu_experiment` wording and official 8-card
  distinction.

- [ ] **Step 5: Check documentation consistency and forbidden files**

```bash
rg -n '8B.*대기|8B.*미검증|rejected before engine' \
  framework/docs/rbln-vllm-setup.md \
  framework/docs/rbln-vllm-atom-validation.md
rg -n 'T[B]D|T[O]DO|F[I]XME' \
  framework/docs/rbln-vllm-setup.md \
  framework/docs/rbln-vllm-atom-validation.md
git status --short
```

Expected: no stale status or placeholders; no model, dataset, result, trace, or
log files are staged.

- [ ] **Step 6: Commit Task 3**

```bash
git add \
  framework/docs/rbln-vllm-setup.md \
  framework/docs/rbln-vllm-atom-validation.md
git commit -m "docs: record single-ATOM Llama validation"
```

---

### Task 4: Run final regression gates and update Draft PR #40

**Files:**
- Verify only: all files changed by Tasks 1–3
- External metadata update: GitHub PR #40 body

**Interfaces:**
- Consumes: clean local `feat/rbln-vllm` commits and passing tests.
- Produces: updated remote branch and Korean PR body matching the physical hardware evidence.

- [ ] **Step 1: Run focused RBLN vLLM and static-RBLN regressions**

```bash
"$TEST_PY" -m pytest -q \
  framework/tests/test_prepare_rbln_vllm_model.py \
  framework/tests/test_rbln_vllm_runtime.py \
  framework/tests/test_main_paths.py \
  framework/tests/test_result_store.py \
  framework/tests/test_rbln_runtime.py \
  framework/tests/test_rbln_native_backend.py \
  framework/tests/test_rbln_collector.py \
  framework/tests/test_plugin_registry.py
```

Expected: PASS. Run outside the filesystem/seccomp sandbox if the known Unix
socket restriction affects asyncio cross-thread tests.

- [ ] **Step 2: Run the complete framework suite**

```bash
"$TEST_PY" -m pytest -q framework/tests
```

Expected: PASS with only the pre-existing unknown `integration` mark warning.

- [ ] **Step 3: Verify repository scope**

```bash
git diff --check origin/feat/rbln-vllm...HEAD
git status --short
git diff --name-only origin/feat/rbln-vllm...HEAD
```

Expected: clean worktree after commits; only the design/plan, Main/tests, and
RBLN vLLM documentation files from this plan differ from the previous remote
head.

- [ ] **Step 4: Push the branch**

```bash
git push -u origin feat/rbln-vllm
```

Expected: remote head equals local `git rev-parse HEAD`.

- [ ] **Step 5: Update PR #40 in Korean without merging it**

Update the PR body to state:

- single-ATOM Llama 3.1 8B compile, sync E2E, and async queue smoke passed;
- run IDs `a3168997` and `9dd3bf7a` and the key memory/latency evidence;
- zero-sample false success is now blocked by post-prepare and loader guards;
- detailed evidence is in `framework/docs/rbln-vllm-atom-validation.md`;
- one-card support remains an unsupported experiment;
- all focused and full test counts from this final run.

Keep the PR in Draft state unless the user separately authorizes marking it
ready or merging it.

- [ ] **Step 6: Report the handoff**

Report the remote commit SHA, PR URL, focused/full test counts, documented run
IDs, and the remaining user decision: mark ready or merge to Main.
