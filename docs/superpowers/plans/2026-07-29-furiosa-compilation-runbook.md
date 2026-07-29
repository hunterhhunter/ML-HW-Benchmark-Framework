# Furiosa RNGD Compilation Runbook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record the exact Furiosa RNGD FXB and first-call JIT compilation procedures, their success criteria, reusable artifacts, and the models that failed strict compilation.

**Architecture:** Extend the existing Furiosa setup runbook instead of creating a second operational guide. Keep executable commands, observed results, and unsupported-model evidence in one dedicated section, while preserving the existing environment and inference commands.

**Tech Stack:** Markdown, Bash, Python 3.12, Furiosa SDK 2026.3.0, `fxb`, `furiosa-torch`, `torch.compile`

## Global Constraints

- Document only compilation outcomes observed on the RNGD server.
- Distinguish FXB ahead-of-time compilation, first-call `torch.compile`, and vendor-precompiled artifact loading.
- Treat `--dry-run` as configuration validation, not compilation success.
- Treat Llama 3.2 3B as successful only after all 9 kernels complete and `Artifact Build Completed` is printed.
- Keep ResNet50, YOLOv5m, and PatchTST explicitly unsupported by the current strict full-model compile path.
- Do not add model binaries, FXB artifacts, dependencies, or runtime behavior in this documentation change.

---

### Task 1: Add the compilation and artifact reuse runbook

**Files:**
- Modify: `docs/furiosa-rngd-setup.md`

**Interfaces:**
- Consumes: the existing Furiosa-LLM and Furiosa Torch environment variables and model paths documented in `docs/furiosa-rngd-setup.md`
- Produces: one `컴파일과 artifact 재사용` section containing a status matrix, copy-paste commands, success criteria, reuse commands, and failure evidence

- [ ] **Step 1: Add the model compilation status matrix**

Insert a table with these exact classifications:

```markdown
| 모델 | 방식 | 결과 | 영구 artifact |
|---|---|---|---|
| Llama 3.2 3B Instruct | `fxb build` 사전 컴파일 | 성공 | `.fxb` |
| BERT SST-2 | 첫 추론 `torch.compile` | 성공 | 프레임워크가 별도 FXB를 만들지 않음 |
| BERT SQuAD v1 | 첫 추론 `torch.compile` | 성공 | 프레임워크가 별도 FXB를 만들지 않음 |
| Llama 3.1 8B Instruct | Furiosa 배포 artifact 로드 | 직접 컴파일하지 않음 | 모델 저장소의 배포 artifact |
| ResNet50 | strict `torch.compile` | 실패 | 없음 |
| YOLOv5m | strict `torch.compile` | 실패 | 없음 |
| PatchTST-FM-r1 | strict `torch.compile` | 실패 | 없음 |
```

- [ ] **Step 2: Add the Llama 3.2 3B config preparation command**

Document the exact config-only build directory and redundant `head_dim` removal:

```bash
cd ~/ML-HW-Benchmark-Framework

FURIOSA_BIN="$PWD/.venv-furiosa/bin"
BUILD_MODEL_DIR="$PWD/framework/models/llama-3.2-3b-instruct-config-test"

mkdir -p "$BUILD_MODEL_DIR"
"$FURIOSA_BIN/hf" download \
  meta-llama/Llama-3.2-3B-Instruct \
  config.json \
  --local-dir "$BUILD_MODEL_DIR"

BUILD_MODEL_DIR="$BUILD_MODEL_DIR" "$FURIOSA_BIN/python" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["BUILD_MODEL_DIR"]) / "config.json"
config = json.loads(path.read_text())
explicit = config.get("head_dim")
derived = config["hidden_size"] // config["num_attention_heads"]
if explicit is not None and explicit != derived:
    raise RuntimeError(
        f"head_dim is not redundant: explicit={explicit}, derived={derived}"
    )
config.pop("head_dim", None)
path.write_text(json.dumps(config, indent=2) + "\n")
print(f"head_dim override removed or already absent: previous={explicit}")
PY
```

Explain that SDK 2026.3.0 rejects the otherwise redundant override with `Invalid config override key: head_dim`.

- [ ] **Step 3: Add dry-run and actual FXB build commands**

```bash
FXB_OUT="$PWD/framework/models/llama-3.2-3b-real-tw1024-$(date +%s).fxb"

"$FURIOSA_BIN/fxb" build \
  "$BUILD_MODEL_DIR" \
  "$FXB_OUT" \
  -tp 8 \
  -O O0 \
  --max-model-len 4096 \
  --dry-run

time "$FURIOSA_BIN/fxb" build \
  "$BUILD_MODEL_DIR" \
  "$FXB_OUT" \
  -tp 8 \
  -O O0 \
  --max-model-len 4096

"$FURIOSA_BIN/fxb" show "$FXB_OUT"
printf 'FXB_OUT=%s\n' "$FXB_OUT"
```

Record the observed result as 9/9 kernels, approximately 15 minutes 21 seconds, and the server artifact example `framework/models/llama-3.2-3b-real-tw1024-1784794954.fxb`. State that a new run may produce a different timestamped name.

- [ ] **Step 4: Add the Llama 3.2 FXB reuse command**

```bash
cd ~/ML-HW-Benchmark-Framework/framework

../.venv-furiosa/bin/python src/main.py \
  --model llama-3.2-3b \
  --target furiosa-rngd \
  --model-path models/meta-llama_Llama-3.2-3B-Instruct-furiosa \
  --fxb models/llama-3.2-3b-real-tw1024-1784794954.fxb \
  --dataset datasets/squad2/val.json \
  --inference-mode e2e \
  --batch-size 1 \
  --warmup 2 \
  --max-new-tokens 64 \
  --max-steps 1
```

Warn that the Furiosa-specific weight directory must contain the materialized `lm_head.weight`; the earlier source lacking it failed with `param 'lm_head.weight' not in safetensors index`.

- [ ] **Step 5: Document BERT first-call JIT compilation**

Include the runtime contract:

```python
backend = furiosa.torch.backend.with_config(
    CompilerConfig(tactic_hint=TacticHintConfig.Default),
    eager_fallback=False,
)
compiled = torch.compile(
    model,
    backend=backend,
    fullgraph=True,
    dynamic=False,
)
```

State that compilation starts at the first warmup/inference call, not when `torch.compile()` returns. Record the fixed contracts: SST-2 uses two `(1, 128)` `int64` inputs; SQuAD uses three `(1, 384)` `int64` inputs. State that this framework path does not emit a reusable FXB and that a new process may compile again depending on the SDK cache.

- [ ] **Step 6: Document failed full-model compile attempts**

Record these representative outcomes without presenting them as supported commands:

```text
ResNet50: furiosa.UnsupportedOpError; compiler panics included
  align_up_required (true) != false (false)
  EinsumByDpe should be given only a single pass

YOLOv5m: CPU forward passed, strict full-model RNGD compile did not pass

PatchTST-FM-r1: CPU forward passed, strict full-model RNGD compile did not pass;
  isolated transpose -> clone -> view graph passed but does not prove model support
```

State that an isolated operator graph success is diagnostic evidence only and cannot change the model status to supported.

- [ ] **Step 7: Commit the runbook content**

```bash
git add \
  docs/furiosa-rngd-setup.md \
  docs/superpowers/plans/2026-07-29-furiosa-compilation-runbook.md
git commit -m "docs: Furiosa RNGD 컴파일 절차 기록"
```

### Task 2: Verify the runbook and update PR #43

**Files:**
- Verify: `docs/furiosa-rngd-setup.md`
- Verify: `docs/superpowers/specs/2026-07-29-furiosa-compilation-runbook-design.md`
- Verify: `docs/superpowers/plans/2026-07-29-furiosa-compilation-runbook.md`

**Interfaces:**
- Consumes: the completed runbook from Task 1
- Produces: a clean documentation diff pushed to `agent/furiosa-rngd-bert`, visible in PR #43

- [ ] **Step 1: Check required coverage**

Run:

```bash
rg -n \
  '컴파일과 artifact 재사용|fxb build|Artifact Build Completed|torch.compile|head_dim|lm_head.weight|ResNet50|YOLOv5m|PatchTST' \
  docs/furiosa-rngd-setup.md
```

Expected: every pattern appears in the new section.

- [ ] **Step 2: Validate Bash code-block syntax**

Run:

```bash
CHECK_DIR=$(mktemp -d /tmp/furiosa-runbook-bash.XXXXXX)
sed -n \
  '/^## 컴파일과 artifact 재사용$/,/^## BERT E2E 및 비동기 실행$/p' \
  docs/furiosa-rngd-setup.md | \
awk -v output_dir="$CHECK_DIR" '
  /^```bash$/ {
    in_block = 1
    count += 1
    output = sprintf("%s/block-%02d.sh", output_dir, count)
    next
  }
  in_block && /^```$/ {
    in_block = 0
    close(output)
    next
  }
  in_block { print >> output }
'

for script in "$CHECK_DIR"/*.sh; do
  bash -n "$script"
done
```

Expected: every extracted Bash block exits 0 under `bash -n`.

- [ ] **Step 3: Check Markdown diff integrity**

Run:

```bash
git diff --check HEAD^..HEAD
git status --short
```

Expected: `git diff --check` exits 0; the worktree has no uncommitted files.

- [ ] **Step 4: Push and verify PR metadata**

```bash
git push origin agent/furiosa-rngd-bert
gh pr view 43 \
  --repo hunterhhunter/ML-HW-Benchmark-Framework \
  --json number,title,url,isDraft,baseRefName,headRefName,state,mergeable
```

Expected: PR #43 is open, targets `main`, uses head `agent/furiosa-rngd-bert`, and reports `MERGEABLE` unless GitHub is still calculating.
