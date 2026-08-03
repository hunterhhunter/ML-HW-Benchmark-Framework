# Mobilint Artifact Provenance Documentation Implementation Plan

> Superseded before implementation by
> `docs/superpowers/specs/2026-08-03-mobilint-multi-model-compilation-experiment-design.md`
> after the PR scope expanded to real PatchTST, ResNet50, and YOLOv5m compilation attempts.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one canonical document that shows how every Mobilint model used by the framework was compiled or obtained, while clearly marking compilation recipes that have not been reproduced.

**Architecture:** `docs/mobilint-artifacts.md` is the provenance index and links to existing model-specific compilation and runtime guides instead of duplicating them. A focused pytest file locks the model list, status terminology, repository identifiers, contracts, and cross-document links so an unverified artifact cannot silently be presented as reproducibly compiled.

**Tech Stack:** Markdown, Python 3.12, pytest, pathlib

## Global Constraints

- BERT SST-2 and SQuAD v1 are the only models marked as directly reproducible with `qbcompiler==1.2.0` in this change.
- PatchTST ETTh1, ResNet50, and YOLOv5m must be marked as having an existing MXQ/runtime contract but no repository-owned, ARIES-verified Mobilint compiler recipe.
- Llama 3.1 8B and Llama 3.2 3B must be described as official precompiled Mobilint Model Zoo downloads, not local qbcompiler outputs.
- No new PatchTST, ResNet50, or YOLOv5m compiler command may be inferred or documented as working until it has compiled on Ubuntu 22.04 and loaded on ARIES.
- Existing detailed commands stay in `docs/mobilint-bert-compilation.md`, `docs/mobilint-aries-transformers.md`, and `docs/mobilint-aries-troubleshooting.md`; the new index links to them.

---

### Task 1: Canonical provenance index and contract test

**Files:**
- Create: `docs/mobilint-artifacts.md`
- Create: `framework/tests/test_mobilint_artifact_docs.py`

**Interfaces:**
- Consumes: BERT recipe paths in `framework/scripts/compile_mobilint_bert.sh`, Mobilint Llama repository IDs in `framework/models/prepare_mobilint_llm.py`, vision profiles in `framework/src/dataloader/mobilint_vision_profiles.py`, and published contracts in the existing Mobilint guides.
- Produces: `docs/mobilint-artifacts.md`, the canonical human-readable provenance index linked by other Mobilint guides.

- [ ] **Step 1: Write the failing provenance test**

Create `framework/tests/test_mobilint_artifact_docs.py` with these checks:

```python
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = REPOSITORY_ROOT / "docs" / "mobilint-artifacts.md"


def _text() -> str:
    return DOCUMENT.read_text(encoding="utf-8")


def test_mobilint_artifact_index_covers_every_supported_model_family():
    text = _text()
    for value in (
        "BERT SST-2",
        "BERT SQuAD v1",
        "PatchTST ETTh1",
        "ResNet50 ImageNet1K V2",
        "YOLOv5m",
        "Llama 3.1 8B",
        "Llama 3.2 3B",
    ):
        assert value in text


def test_mobilint_artifact_index_separates_provenance_states():
    text = _text()
    for value in (
        "직접 컴파일·재현 가능",
        "컴파일 recipe 미검증",
        "공식 사전 컴파일 배포",
        "로컬 qbcompiler 대상 아님",
    ):
        assert value in text


def test_mobilint_artifact_index_records_sources_and_contracts():
    text = _text()
    for value in (
        "textattack/bert-base-uncased-SST-2",
        "csarron/bert-base-uncased-squad-v1",
        "ibm-granite/granite-timeseries-patchtst",
        "mobilint/YOLOv5m",
        "mobilint/Llama-3.1-8B-Instruct",
        "mobilint/Llama-3.2-3B-Instruct",
        "float32 (L, 768)",
        "float32 (512, 7)",
        "uint8 NHWC (224, 224, 3)",
        "uint8 NHWC (640, 640, 3)",
    ):
        assert value in text


def test_mobilint_artifact_index_links_existing_guides_and_scripts():
    text = _text()
    expected = {
        "mobilint-bert-compilation.md": REPOSITORY_ROOT
        / "docs"
        / "mobilint-bert-compilation.md",
        "mobilint-aries-transformers.md": REPOSITORY_ROOT
        / "docs"
        / "mobilint-aries-transformers.md",
        "mobilint-aries-troubleshooting.md": REPOSITORY_ROOT
        / "docs"
        / "mobilint-aries-troubleshooting.md",
        "../framework/scripts/compile_mobilint_bert.sh": REPOSITORY_ROOT
        / "framework"
        / "scripts"
        / "compile_mobilint_bert.sh",
        "../framework/models/prepare_mobilint_llm.py": REPOSITORY_ROOT
        / "framework"
        / "models"
        / "prepare_mobilint_llm.py",
    }
    for target, path in expected.items():
        assert f"]({target})" in text
        assert path.is_file()
```

- [ ] **Step 2: Run the test and confirm the missing document fails**

Run:

```bash
python -m pytest framework/tests/test_mobilint_artifact_docs.py -q
```

Expected: FAIL with `FileNotFoundError` for `docs/mobilint-artifacts.md`.

- [ ] **Step 3: Write the canonical provenance document**

Create `docs/mobilint-artifacts.md` with:

1. Definitions for `직접 컴파일·재현 가능`, `컴파일 recipe 미검증`, and `공식 사전 컴파일 배포`.
2. One summary table containing all seven named model/task rows.
3. BERT source IDs, `qbcompiler==1.2.0`, `aries-rb`, the one-shot script link, `float32 (L, 768)` input, and the observed SST-2/SQuAD output ordering.
4. PatchTST source checkpoint and existing `past_values float32 (512, 7)`, `past_observed_mask bool (512, 7)`, output `(96, 7)` contract, followed by an explicit statement that `prepare_patchtst.py` is an ONNX export helper and not a Mobilint compiler recipe.
5. ResNet50 and YOLOv5m artifact paths and exact vision contracts. State that YOLOv5m was downloaded from `mobilint/YOLOv5m`; state that the exact original qbcompiler commands for both vision MXQs are not present in the repository.
6. All six Llama standard/Batch16/Batch32 repository IDs from `prepare_mobilint_llm.py`, with a download command and `로컬 qbcompiler 대상 아님` wording.
7. A promotion gate requiring a pinned source, compiler version, target, calibration method, input/output contract, non-empty artifact hashes, MXQ inspection, ARIES launch, and task smoke result before changing `컴파일 recipe 미검증` to `직접 컴파일·재현 가능`.

Use this exact summary classification:

```markdown
| 모델·작업 | 생성·입수 방식 | 현재 상태 |
|---|---|---|
| BERT SST-2 | qbcompiler 1.2, `aries-rb` 직접 컴파일 | 직접 컴파일·재현 가능 |
| BERT SQuAD v1 | qbcompiler 1.2, `aries-rb` 직접 컴파일 | 직접 컴파일·재현 가능 |
| PatchTST ETTh1 | 기존 MXQ와 실행 계약 사용 | 컴파일 recipe 미검증 |
| ResNet50 ImageNet1K V2 | 기존 Model Zoo MXQ 사용 | 컴파일 recipe 미검증 |
| YOLOv5m | `mobilint/YOLOv5m`의 기존 MXQ 사용 | 컴파일 recipe 미검증 |
| Llama 3.1 8B | Mobilint Model Zoo snapshot 다운로드 | 공식 사전 컴파일 배포; 로컬 qbcompiler 대상 아님 |
| Llama 3.2 3B | Mobilint Model Zoo snapshot 다운로드 | 공식 사전 컴파일 배포; 로컬 qbcompiler 대상 아님 |
```

- [ ] **Step 4: Run the provenance test**

Run:

```bash
python -m pytest framework/tests/test_mobilint_artifact_docs.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit the canonical index**

```bash
git add docs/mobilint-artifacts.md framework/tests/test_mobilint_artifact_docs.py
git commit -m "docs: catalog Mobilint artifact provenance"
```

### Task 2: Cross-link the runtime guides

**Files:**
- Modify: `framework/tests/test_mobilint_artifact_docs.py`
- Modify: `docs/mobilint-aries-transformers.md`
- Modify: `docs/mobilint-aries-troubleshooting.md`

**Interfaces:**
- Consumes: the canonical `docs/mobilint-artifacts.md` created in Task 1.
- Produces: discoverable links from both existing Mobilint entry-point guides to the canonical provenance index.

- [ ] **Step 1: Add a failing cross-link test**

Append this test to `framework/tests/test_mobilint_artifact_docs.py`:

```python
def test_mobilint_guides_link_to_canonical_artifact_index():
    for relative_path in (
        "docs/mobilint-aries-transformers.md",
        "docs/mobilint-aries-troubleshooting.md",
    ):
        text = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
        assert "](mobilint-artifacts.md)" in text
```

- [ ] **Step 2: Run the cross-link test and confirm it fails**

Run:

```bash
python -m pytest \
  framework/tests/test_mobilint_artifact_docs.py::test_mobilint_guides_link_to_canonical_artifact_index \
  -q
```

Expected: FAIL because neither guide links `mobilint-artifacts.md` yet.

- [ ] **Step 3: Add concise links without copying the index**

In the introduction of `docs/mobilint-aries-transformers.md`, add:

```markdown
모델별 MXQ가 직접 컴파일된 것인지 공식 배포본인지와 현재 재현 수준은
[Mobilint 아티팩트 생성·입수 현황](mobilint-artifacts.md)에서 확인한다.
```

In the introduction of `docs/mobilint-aries-troubleshooting.md`, add the same two-line link after the purpose paragraph. Do not change historical result tables in this task.

- [ ] **Step 4: Run all documentation tests**

Run:

```bash
python -m pytest framework/tests/test_mobilint_artifact_docs.py -q
```

Expected: `5 passed`.

- [ ] **Step 5: Commit the cross-links**

```bash
git add \
  docs/mobilint-aries-transformers.md \
  docs/mobilint-aries-troubleshooting.md \
  framework/tests/test_mobilint_artifact_docs.py
git commit -m "docs: link Mobilint artifact provenance"
```

### Task 3: Repository verification and PR update

**Files:**
- Verify: `docs/mobilint-artifacts.md`
- Verify: `docs/mobilint-bert-compilation.md`
- Verify: `docs/mobilint-aries-transformers.md`
- Verify: `docs/mobilint-aries-troubleshooting.md`
- Verify: `framework/tests/test_mobilint_artifact_docs.py`

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: verified commits ready to push to `feat/mobilint-bert-mxq-benchmark` and an updated Korean PR description.

- [ ] **Step 1: Run focused Mobilint documentation and contract tests**

Run:

```bash
python -m pytest \
  framework/tests/test_mobilint_artifact_docs.py \
  framework/tests/test_mobilint_bert_compile.py \
  framework/tests/test_mobilint_bert_profiles.py \
  framework/tests/test_mobilint_vision_profiles.py \
  framework/tests/test_patchtst_etth1_profile.py \
  framework/tests/test_prepare_mobilint_llm.py \
  -q
```

Expected: all selected tests PASS.

- [ ] **Step 2: Check formatting, links, and prohibited claims**

Run:

```bash
git diff --check origin/main...HEAD
rg -n \
  '직접 컴파일·재현 가능|컴파일 recipe 미검증|공식 사전 컴파일 배포|로컬 qbcompiler 대상 아님' \
  docs/mobilint-artifacts.md
rg -n \
  'mobilint-artifacts.md' \
  docs/mobilint-aries-transformers.md \
  docs/mobilint-aries-troubleshooting.md
```

Expected: `git diff --check` has no output; the provenance status search shows BERT as reproducible, PatchTST/vision as unverified recipes, and Llama as official precompiled distribution; both existing guides show one canonical link.

- [ ] **Step 3: Review the final branch diff**

Run:

```bash
git status --short --branch
git diff --stat origin/main...HEAD
git log --oneline origin/main..HEAD
```

Expected: only intended PR files and commits appear; no generated `.mxq`, `.mblt`, weights, calibration arrays, datasets, logs, or virtual environments are tracked.

- [ ] **Step 4: Push and update the existing Korean draft PR**

Run:

```bash
git push origin feat/mobilint-bert-mxq-benchmark
gh pr edit 47 --body-file /tmp/pr47-body-ko.md
```

The Korean PR body must add these concise points under the documentation section:

```markdown
- Mobilint 모델별 아티팩트 생성·입수 방식과 재현 상태를 한 문서에 정리했습니다.
- BERT는 직접 컴파일 재현 가능, PatchTST·ResNet50·YOLOv5m은 recipe 미검증, Llama는 공식 사전 컴파일 배포본으로 구분했습니다.
- PatchTST·비전 모델의 컴파일 명령은 실제 컴파일 및 ARIES 검증 후 별도 변경으로 추가합니다.
```

Expected: the remote branch contains all local commits and PR #47 remains a draft targeting `main` with the expanded Korean summary.
