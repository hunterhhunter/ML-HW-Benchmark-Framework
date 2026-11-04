# Furiosa RNGD Troubleshooting Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create one evidence-backed Korean troubleshooting document that lets an operator recover Furiosa RNGD benchmark runs and lets a framework developer understand the responsible code boundaries and remaining SDK limitations.

**Architecture:** Add `docs/furiosa-rngd-troubleshooting.md` as a symptom-first runbook followed by a developer-analysis section. Keep normal setup and paper-serving procedures in their existing documents, link to them instead of duplicating them, and label every claim as verified, inferred, unresolved, or an SDK limitation according to the approved design.

**Tech Stack:** Markdown, Bash command examples, Furiosa SDK 2026.3.0 CLI/API, `ML-HW-Benchmark-Framework` target/runtime/async/monitor registries.

## Global Constraints

- Write the final document in Korean at `docs/furiosa-rngd-troubleshooting.md`.
- Preserve all pre-existing dirty and untracked user files; stage only files named by each task.
- Treat hardware logs and completed CSV runs as stronger evidence than unit tests.
- Do not claim YOLOv5m, BERT, PatchTST, or ResNet50 hardware support without a completed RNGD run.
- Mark existing power values as `미수집`; never substitute RNGD's 150 W TDP for measured power.
- Distinguish local Hugging Face weights, legacy artifact directories, and `.fxb` files.
- Distinguish async single-stream (`queue-capacity=1`, `worker-count=1`) from concurrency (`32`, `8`).
- State that `max-new-tokens` is an upper bound, so model comparisons must include tokens/s and TTFT/TPOT in addition to samples/s.
- Avoid destructive Git and cache commands. If stale cache isolation is required, move the exact entry to a timestamped backup path after resolving its concrete path.
- Do not include credentials, Hugging Face tokens, machine IDs, or user-specific home directory names.

---

## File Structure

- Create `docs/furiosa-rngd-troubleshooting.md`: operator runbook, model status, resolved incidents, unresolved incidents, known-good commands, metrics interpretation, and developer analysis.
- Modify `docs/furiosa-rngd-setup.md`: add a short link to the troubleshooting document near the introduction; leave its setup contract unchanged.
- Reference only, do not modify:
  - `docs/rngd-paper-benchmark.md`
  - `docs/async-inference-queue.md`
  - `framework/src/core/targets.py`
  - `framework/src/runtimes/furiosa_llm_rt.py`
  - `framework/src/core/runtime_executor.py`
  - `framework/src/core/async_inference/metrics.py`
  - `framework/src/monitors/base.py`
  - `framework/tests/test_furiosa_llm_runtime.py`
  - `framework/tests/test_furiosa_native_backend.py`

### Task 1: Create the runbook foundation and evidence boundary

**Files:**
- Create: `docs/furiosa-rngd-troubleshooting.md`
- Reference: `docs/furiosa-rngd-setup.md:1-123`
- Reference: `framework/src/core/targets.py:337-348`
- Reference: `framework/src/runtimes/furiosa_llm_rt.py:293-520`

**Interfaces:**
- Consumes: approved design at `docs/superpowers/specs/2026-07-23-furiosa-rngd-troubleshooting-design.md`.
- Produces: stable document headings and terminology used by Tasks 2-4.

- [ ] **Step 1: Create the document header and evidence labels**

Start the document with these headings and definitions:

```markdown
# Furiosa RNGD 트러블슈팅 Runbook과 개발자 분석

이 문서는 Furiosa RNGD 서버에서 ML-HW-Benchmark-Framework의 LLM·비전 추론을 준비하고 검증하면서 실제로 확인한 장애를 정리한다.

## 문서 사용법

- **검증 완료**: 서버 로그와 완료된 RNGD 벤치마크로 확인했다.
- **추정**: 로그로 가능성을 좁혔지만 통제된 대조 실험이 더 필요하다.
- **미해결**: 재현했지만 우회 또는 수정이 검증되지 않았다.
- **현재 SDK 한계**: 프레임워크 외부의 compiler/runtime에서 중단됐다.

정상 설치·실행 절차는 [Furiosa RNGD runtime](furiosa-rngd-setup.md), OpenAI-compatible serving 측정은 [RNGD 논문용 생성 지연 벤치마크 프로토콜](rngd-paper-benchmark.md)을 먼저 참고한다.
```

- [ ] **Step 2: Add the verified environment table**

Record the observed environment without presenting it as a universal requirement:

```markdown
## 검증 환경

| 항목 | 확인값 |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 6.8.0-124-generic |
| CPU | Intel Xeon Silver 4514Y, 64 logical CPUs |
| RNGD | `1ed2:0001`, `npu0`, 47.50 GiB |
| Furiosa driver | 2026.3.0 |
| Firmware | 1.11.0, `cfd5306` |
| Furiosa-LLM | 2026.3.0 |
| Furiosa Torch | 2026.3.0 |
| Python | Furiosa virtual environment: 3.12.13; host initially: 3.10.12 |
```

Add a note that Secure Boot was disabled and matching kernel headers were present on the verified host.

- [ ] **Step 3: Add the model status matrix**

Use explicit status language:

```markdown
## 모델 상태

| 모델 | 경로 | 상태 | 근거 |
|---|---|---|---|
| Llama 3.1 8B | Furiosa-LLM legacy artifact/repository ID | 검증 완료 | E2E 1,000건 및 native async 1,000건 완료 |
| Llama 3.2 3B | local HF weights + custom FXB | 조건부 검증 완료 | E2E/native async 완료, exact registry entry 없이 nearest preset fallback |
| ResNet50 | ONNX → PyTorch → Furiosa Torch | 현재 SDK 한계 | CPU parity 성공 후 Conv2d compiler panic |
| YOLOv5m | adapter/준비 코드 | 미검증 | RNGD full-graph 실행 증거 없음 |
| BERT SST-2/SQuAD | adapter/준비 코드 | 미검증 | RNGD full-graph 실행 증거 없음 |
| PatchTST | adapter/준비 코드 | 미검증 | RNGD full-graph 실행 증거 없음 |
```

- [ ] **Step 4: Add the five-minute diagnostic block**

Include read-only commands for OS, PCIe, driver, SMI, packages, branch, and files. Use variables instead of personal paths:

```bash
date -Is
cat /etc/os-release
uname -r
lspci -Dnnk | grep -A3 -iE 'FuriosaAI|1ed2:'
lsmod | grep -i furiosa
furiosa-smi info
furiosa-smi status
furiosa-smi ps
dkms status 2>/dev/null | grep -i furiosa || true

git branch --show-current
git rev-parse --short HEAD
git status --short
```

State the success evidence: PCI ID `1ed2:0001`, `Kernel driver in use: furiosa_rngd`, and an `npu0` row from `furiosa-smi info`.

- [ ] **Step 5: Add the quick error index**

Create a table mapping these exact search strings to document anchors:

- `unknown variant Primitive`
- `Invalid config override key: head_dim`
- `failed printing to stderr: Broken pipe`
- `param 'lm_head.weight' not in safetensors index`
- `No module named 'cv2'`
- `worker_count=8 exceeds runtime capability 1`
- `NativeAsyncBackpressureTimeout`
- `eager fallback is not allowed`
- `align_up_required (true) != false (false)`
- `EinsumByDpe should be given only a single pass`

- [ ] **Step 6: Verify the foundation**

Run:

```bash
test -f docs/furiosa-rngd-troubleshooting.md
rg -n '^#|^##|^###' docs/furiosa-rngd-troubleshooting.md
rg -n '검증 완료|추정|미해결|현재 SDK 한계' docs/furiosa-rngd-troubleshooting.md
git diff --check -- docs/furiosa-rngd-troubleshooting.md
```

Expected: the file exists, all four evidence labels appear, headings are listed, and `git diff --check` prints nothing.

- [ ] **Step 7: Commit the foundation**

```bash
git add docs/furiosa-rngd-troubleshooting.md
git commit -m "docs(furiosa): add RNGD troubleshooting foundation"
```

### Task 2: Document resolved driver, environment, artifact, and FXB incidents

**Files:**
- Modify: `docs/furiosa-rngd-troubleshooting.md`
- Reference: `framework/src/runtimes/furiosa_llm_rt.py:293-445`
- Reference: `framework/tests/test_furiosa_llm_runtime.py:1-230`
- Reference: `framework/tests/test_furiosa_native_backend.py:1-250`

**Interfaces:**
- Consumes: evidence labels and quick-index anchors from Task 1.
- Produces: symptom-first entries and known-good model preparation procedures used by Task 4.

- [ ] **Step 1: Add the driver and DKMS entry**

Document that a broken unrelated `rebellions-dkms` source entry can appear alongside a successfully installed `furiosa-driver-rngd`. The decisive checks are `lsmod`, `lspci -k`, and `furiosa-smi info`, not a generic DKMS error alone. Include reboot guidance: load the newly built module when safe, and schedule a reboot when the running kernel/module state cannot be reconciled; do not imply a reboot is always mandatory after package installation.

- [ ] **Step 2: Add Git/worktree and virtual-environment isolation**

Use this safe pattern:

```bash
SOURCE_REPO=/absolute/path/ML-HW-Benchmark-Framework
MAIN_WORKTREE=/absolute/path/ML-HW-Benchmark-Framework-main

FRAMEWORK="$MAIN_WORKTREE/framework"
PY="$SOURCE_REPO/.venv-furiosa/bin/python"
DATASET="$SOURCE_REPO/framework/datasets/squad2/val.json"
```

Explain that code can come from a clean main worktree while model, dataset, and virtual environment use absolute paths from the original dirty checkout. Record that executing the Mobilint verification branch caused `preprocessor/__init__.py` to eagerly import `mobilint_vision.py`, which required `cv2` before Furiosa selection; this was import-time dependency contamination, not Mobilint runtime selection.

- [ ] **Step 3: Add the model/artifact/FXB contract entry**

Define the three inputs:

- Hugging Face model directory: config, tokenizer, weights.
- Legacy Furiosa artifact: `artifact.json`, `binary_bundle.zip`, compiled parameter artifact.
- FXB: explicit `.fxb` containing compiled kernels and fingerprint metadata.

State that `fxb download` only succeeds for a repository that actually contains `.fxb` files. For a `furiosa-ai` legacy artifact repository, use a runtime-compatible repository revision or repository ID rather than passing an empty `--fxb` value.

- [ ] **Step 4: Add the Llama 3.1 `Primitive` incident**

Record:

- Symptom: deserialization rejects `Primitive` while listing older semantic variants.
- Cause: a newer legacy artifact snapshot was loaded by Furiosa-LLM 2026.3.0.
- Evidence: the traceback names the exact local artifact directory, proving current working directory was not the cause.
- Resolution on the observed legacy CLI path: use the working repository ID `furiosa-ai/Llama-3.1-8B-Instruct`, which lets the Furiosa runtime select its matching revision, or download the explicit compatible tag into a separate directory before testing.
- Contract boundary: the current source tree validates a local Hugging Face directory plus an explicit `.fxb`. Do not present the repository-ID success from the legacy verification branch as proof that the current main contract has an available Llama 3.1 FXB.
- Success: model/RNGD load proceeds and the framework writes final E2E/async metrics.

- [ ] **Step 5: Add Llama 3.2 build incidents in causal order**

Create separate subsections with the standard symptom template:

1. Exact Llama 3.2 3B registry entry absent; nearest Llama 3.1 8B preset selected. Mark the performance explanation as conditional, not official exact support.
2. `head_dim` is explicit and redundant (`3072 / 24 = 128`); back up `config.json` and remove only when equality is verified.
3. Final EDF stage requires `aarch64-linux-gnu-gcc`; confirm with `command -v` and install the distribution cross-compiler package through the approved system package workflow.
4. `Broken pipe` is a secondary surface error; inspect the first failing compiler/EDF process and its stderr before changing the model.
5. Isolate a concrete negative cache entry by moving that exact resolved entry to a timestamped backup directory; never show a broad recursive cache deletion.
6. Tied embeddings omit `lm_head.weight`; create a derived model directory that preserves original shards, writes one explicit `model-lm-head.safetensors`, updates a copied safetensors index, and keeps the original directory unchanged.

Record the verified FXB result: 9 kernels compiled, 0 failed, approximately 15 minutes 21 seconds on the observed host.

- [ ] **Step 6: Add the `attention_mask` input-contract incident**

Explain that pre-tokenized Furiosa generation accepts `BatchEncoding`, but the framework initially supplied only `input_ids`. The fixed boundary validates matching `input_ids` and `attention_mask` shapes, trims padding using the mask, and then creates the request payload. Point to `framework/src/runtimes/furiosa_llm_rt.py:431-445` and the Furiosa runtime tests rather than duplicating implementation code.

- [ ] **Step 7: Verify resolved-incident coverage**

Run:

```bash
DOC=docs/furiosa-rngd-troubleshooting.md
for pattern in \
  'unknown variant `Primitive`' \
  'head_dim' \
  'aarch64-linux-gnu-gcc' \
  'Broken pipe' \
  "lm_head.weight" \
  'attention_mask' \
  "No module named 'cv2'"
do
  rg -q "$pattern" "$DOC" || exit 1
done
git diff --check -- "$DOC"
```

Expected: exit status 0 and no whitespace errors.

- [ ] **Step 8: Commit the resolved incidents**

```bash
git add docs/furiosa-rngd-troubleshooting.md
git commit -m "docs(furiosa): record RNGD artifact and FXB incidents"
```

### Task 3: Document async diagnosis, monitoring, metrics, and unresolved vision support

**Files:**
- Modify: `docs/furiosa-rngd-troubleshooting.md`
- Reference: `docs/async-inference-queue.md:105-206`
- Reference: `framework/src/main.py:314-363`
- Reference: `framework/src/core/runtime_executor.py:357-542`
- Reference: `framework/src/core/async_inference/metrics.py:1040-1065`
- Reference: `framework/src/monitors/base.py:148-192`
- Reference: `framework/src/core/targets.py:337-348`

**Interfaces:**
- Consumes: model preparation and evidence labels from Tasks 1-2.
- Produces: mode-specific diagnosis, performance interpretation, power limitation, and unresolved-support analysis.

- [ ] **Step 1: Add async hang and progress diagnosis**

Explain the observable phases: model load, warmup, engine start, measurement, flush, finalization, unload. State that no per-request debug output does not prove the NPU is idle. Use `furiosa-smi status`, `furiosa-smi ps`, result trace growth, and `[AsyncDebug]` phase changes together. Distinguish an allocated 45+ GiB model with 0% instantaneous utilization from a confirmed deadlock.

- [ ] **Step 2: Add worker capability and backpressure entries**

Document both observed generations of behavior:

- Older path: `worker_count=8 exceeds runtime capability 1`; use `queue-capacity=1`, `worker-count=1` for a valid single-stream baseline.
- Timeout path: alternating `NativeAsyncBackpressureTimeout` and `flush_timeout`; inspect accepted/completed/failed/outstanding counts and do not report an `invalid` run as performance data.
- Current concurrency experiment: `queue=32`, `worker=8` completed 1,000 requests with zero failures on both Llama models, but it is a concurrency result and must not replace the single-stream baseline.

State that a worker is a framework submission lane, not an RNGD core or batch element.

- [ ] **Step 3: Add mode-specific metric interpretation**

Include one table with four rows: each Llama model in `e2e` and `async_queue`. Explain:

- E2E `Average Latency`/`P99 Latency` is a synchronous request-level metric.
- Async `async_e2e_latency_*` includes queue and service time.
- `async_completed_samples_per_sec` is system sample throughput.
- `async_completed_tokens_per_sec` is system token throughput.
- TTFT and TPOT are generation metrics; queue wait must not be added to a latency that already includes it.

Record the controlled single-stream result as the interpretation example:

| Model | samples/s | tokens/s | Avg TTFT | Avg TPOT | Generated tokens |
|---|---:|---:|---:|---:|---:|
| Llama 3.1 8B | 5.44 | 55.38 | 24.28 ms | 16.96 ms | 10,174 |
| Llama 3.2 3B | 4.80 | 61.55 | 69.86 ms | 11.33 ms | 12,834 |

Explain that Llama 3.2 produced about 26% more tokens and decoded faster per token in single-stream, so lower samples/s alone does not prove the 3B model is computationally slower.

- [ ] **Step 4: Add SMI and power limitations**

Record that `furiosa-smi info` exposes a point-in-time device power value, while existing benchmark CSVs did not collect power. In the current target registry, `furiosa-rngd` uses `monitor_names=("system",)`, so `--monitor` alone collects host metrics but does not create RNGD `hw_accel_power_w` samples. List the future metrics:

- idle/load average and maximum W
- total and idle-subtracted J
- samples/J and tokens/J
- average and maximum temperature

Point to `framework/src/monitors/base.py:183` to show that a future collector returning `hw_accel_power_w` already receives avg/max aggregation. State that energy integration still needs timestamps or a measured interval.

- [ ] **Step 5: Add the ResNet50 Furiosa Torch unresolved incident**

Record the established boundary:

- Kalray ResNet50 ONNX loads and ONNX Runtime versus converted PyTorch CPU parity passed after handling unsupported ONNX `Flatten` conversion.
- Furiosa Torch compilation failed for isolated Conv2d and full model variants across FP32/BF16, contiguous/channels-last, bias variants, and tactic choices.
- Preserve the exact panic signatures `align_up_required (true) != false (false)`, `EinsumByDpe should be given only a single pass`, and `Option::unwrap() on a None value`.
- Mark this as `현재 SDK 한계`; do not wrap, split, or fall back to CPU and then claim RNGD full-graph success.

- [ ] **Step 6: Add developer improvement tasks with completion criteria**

List these concrete follow-ups without implementing them:

1. Lazy/optional vendor preprocessor imports; success is importing the Furiosa CLI path without OpenCV installed.
2. Furiosa SMI collector registered in `furiosa-rngd`; success is CSV `hw_accel_power_w_avg/max` from a measurement-only interval.
3. Progress reporting with bounded frequency; success is visible completed/total counts without trace-volume or timing distortion.
4. Exact Llama 3.2 registry/FXB support; success is no nearest-preset warning plus successful hardware correctness/performance run.
5. Furiosa Torch vision compiler resolution; success is full-graph compile, CPU-reference parity, and non-zero SMI utilization.

- [ ] **Step 7: Verify async and unresolved coverage**

Run:

```bash
DOC=docs/furiosa-rngd-troubleshooting.md
for pattern in \
  'NativeAsyncBackpressureTimeout' \
  'worker_count=8 exceeds runtime capability 1' \
  'async_completed_tokens_per_sec' \
  'queue-capacity=1' \
  'hw_accel_power_w' \
  'align_up_required' \
  'EinsumByDpe'
do
  rg -q "$pattern" "$DOC" || exit 1
done
git diff --check -- "$DOC"
```

Expected: exit status 0 and no whitespace errors.

- [ ] **Step 8: Commit async and unresolved analysis**

```bash
git add docs/furiosa-rngd-troubleshooting.md
git commit -m "docs(furiosa): explain async metrics and SDK limits"
```

### Task 4: Add known-good commands, navigation links, and final verification

**Files:**
- Modify: `docs/furiosa-rngd-troubleshooting.md`
- Modify: `docs/furiosa-rngd-setup.md:1-5`
- Reference: `framework/src/main.py:180-370`
- Reference: `docs/rngd-paper-benchmark.md:23-136`

**Interfaces:**
- Consumes: all terminology, model statuses, and incidents from Tasks 1-3.
- Produces: final navigable documentation ready for operator use.

- [ ] **Step 1: Add contract-specific path variables and smoke tests**

Use neutral paths:

```bash
SOURCE_REPO=/absolute/path/ML-HW-Benchmark-Framework
FRAMEWORK="$SOURCE_REPO/framework"
PY="$SOURCE_REPO/.venv-furiosa/bin/python"
DATASET="$FRAMEWORK/datasets/squad2/val.json"
MODEL31_REPO=furiosa-ai/Llama-3.1-8B-Instruct
MODEL31_LOCAL=/absolute/path/to/Llama-3.1-8B
FXB31=/absolute/path/to/llama-3.1-8b.fxb
MODEL32="$FRAMEWORK/models/meta-llama_Llama-3.2-3B-Instruct-furiosa"
FXB32="$FRAMEWORK/models/llama-3.2-3b.fxb"
```

Add two clearly labelled Llama 3.1 smoke commands instead of merging contracts:

- **Observed legacy verification path:** use `MODEL31_REPO` without an empty `--fxb`; label it with the verification branch/commit recorded in the run rather than current main.
- **Current local-model/FXB contract:** use `MODEL31_LOCAL` with `FXB31`; label hardware status as unverified until an actual compatible Llama 3.1 FXB is shown and the run completes.

Llama 3.2 uses its derived local model path and explicit FXB. Never include `--fxb ""` in any command.

- [ ] **Step 2: Add 1,000-sample E2E and async command matrices**

Provide mode commands with common values `batch-size=1`, `warmup=2`, `max-new-tokens=32`, and SQuAD2 validation data. Put the observed legacy Llama 3.1 commands and current explicit-FXB command templates under separate subheadings:

- Legacy Llama 3.1 E2E and async: repository ID, with the observed run status and commit stated explicitly
- Current-contract Llama 3.1 E2E and async: local model plus `FXB31`, marked as a template until hardware evidence exists
- Llama 3.2 E2E: explicit model and FXB, `max-steps=1000`
- Llama 3.2 async single-stream: explicit model and FXB, `queue-capacity=1`, `worker-count=1`

For concurrency examples, show only the two changed options (`32`, `8`) and require a separate result file and label. Include `schedule-seed=0`, request trace, submit/flush/request timeouts, and unique result paths in full async commands.

- [ ] **Step 3: Add success gates**

For E2E require `num_samples=1000` and a saved result row. For async require:

```text
async_completed_samples=1000
async_failed_requests=0
async_timed_out_requests=0
async_outstanding_requests=0
async_run_status=valid
```

Also require model/RNGD load logs and normal scheduler shutdown. State that quality and speed are separate gates.

- [ ] **Step 4: Link the troubleshooting document from setup**

Add one paragraph after the introduction in `docs/furiosa-rngd-setup.md`:

```markdown
설치·빌드·실행 중 오류가 발생하면 [Furiosa RNGD 트러블슈팅 Runbook과 개발자 분석](furiosa-rngd-troubleshooting.md)에서 오류 문자열별 원인, 확인 명령, 해결 절차와 현재 SDK 한계를 확인하세요.
```

- [ ] **Step 5: Validate local Markdown links**

Run:

```bash
python3 - <<'PY'
import re
from pathlib import Path

for document in (
    Path("docs/furiosa-rngd-troubleshooting.md"),
    Path("docs/furiosa-rngd-setup.md"),
):
    text = document.read_text()
    for target in re.findall(r"\[[^]]+\]\(([^)]+\.md)(?:#[^)]+)?\)", text):
        path = (document.parent / target).resolve()
        if not path.is_file():
            raise SystemExit(f"broken local link: {document}: {target}")
print("local markdown links: OK")
PY
```

Expected: `local markdown links: OK`.

- [ ] **Step 6: Validate CLI option names and evidence strings**

Run:

```bash
for option in \
  --model-path --fxb --dataset --inference-mode --scenario \
  --batch-size --warmup --max-new-tokens --max-steps \
  --max-samples --min-samples --queue-capacity --worker-count \
  --submit-timeout-sec --flush-timeout-sec --request-timeout-ms \
  --schedule-seed --save-request-trace --results-path
do
  rg -q -- "${option}" framework/src/main.py || exit 1
done

! rg -n 'T[B]D|T[O]DO|PLACE[H]OLDER|/h[o]me/|etri_[e]cas|swl[a]b' \
  docs/furiosa-rngd-troubleshooting.md
git diff --check -- \
  docs/furiosa-rngd-troubleshooting.md \
  docs/furiosa-rngd-setup.md
```

Expected: exit status 0 and no output from the placeholder or whitespace checks.

- [ ] **Step 7: Review the final scoped diff**

Run:

```bash
git diff -- \
  docs/furiosa-rngd-troubleshooting.md \
  docs/furiosa-rngd-setup.md
git status --short
```

Expected: only the troubleshooting document and the single setup link belong to this implementation; unrelated user changes remain unstaged and unchanged.

- [ ] **Step 8: Commit final navigation and commands**

```bash
git add \
  docs/furiosa-rngd-troubleshooting.md \
  docs/furiosa-rngd-setup.md
git commit -m "docs(furiosa): finalize RNGD troubleshooting runbook"
```
