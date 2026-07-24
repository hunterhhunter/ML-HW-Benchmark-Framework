# RBLN Troubleshooting Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a model-first, evidence-backed troubleshooting guide for the validated RBLN-CA22 static and asynchronous benchmark integration.

**Architecture:** Keep normative installation and runtime contracts in `framework/docs/rbln-setup.md`. Add a separate model-first troubleshooting guide whose entries consistently record symptom, cause, resolution, verification, and status; link the two documents without changing runtime code.

**Tech Stack:** Markdown, Bash command examples, Python 3.10, Rebellions `rebel-compiler==0.11.0`, `rbln-smi`, Git.

## Global Constraints

- Work only in `/tmp/ml-hw-benchmark-rbln-runtime-monitor` on `feat/rbln-runtime-monitor`.
- Preserve the validated baseline: Ubuntu 22.04.5, Python 3.10.12, RBLN-CA22 device 0, KMD/FW 3.2.2, and `rebel-compiler==0.11.0`.
- Do not change runtime, model-profile, evaluator, dataset, or test code.
- Do not describe BERT SQuAD strict CPU/NPU logit parity or context masking as solved.
- Do not describe Llama 3.1 8B or 3.2 3B as supported by `rbln-static`.
- Distinguish observed facts, evidence-backed causes, mitigations, and open issues.
- Treat `hw_accel_energy_j` as whole-card energy including idle power.
- Require `rbln-smi -j` with `contexts: []` as the external lifecycle completion check.

---

### Task 1: Write the model-first troubleshooting guide

**Files:**
- Create: `framework/docs/rbln-troubleshooting.md`

**Interfaces:**
- Consumes: the contracts documented in `framework/docs/rbln-setup.md` and the validated server results.
- Produces: a standalone operator runbook organized by common checks and model name.

- [ ] **Step 1: Create the document header and baseline**

Create `framework/docs/rbln-troubleshooting.md` with this opening structure:

```markdown
# Rebellions RBLN-CA22 트러블슈팅

이 문서는 `rbln-static` 통합 과정에서 실제로 관찰한 실패와 검증된 복구
절차를 기록한다. 정상 설치·실행 계약은 [RBLN 운영 가이드](rbln-setup.md)를
먼저 확인한다.

## 1. 기준 환경과 빠른 판정
## 2. 공통 환경·브랜치·artifact 문제
## 3. ResNet50
## 4. YOLOv5m
## 5. BERT SST-2
## 6. PatchTST ETTh1
## 7. BERT SQuAD
## 8. 비동기 큐와 monitoring 해석
## 9. 최종 검증 상태
## 10. 재실행 체크리스트
```

Under section 1, record the exact validated baseline and include commands for:

```bash
python3 --version
python3 -m pip show rebel-compiler
rbln-smi -q
rbln-smi -j
```

Include a lifecycle precheck that exits non-zero when device 0 has a context:

```bash
rbln-smi -j | python3 -c 'import json,sys; payload=json.load(sys.stdin); contexts=[item for item in payload.get("contexts", []) if isinstance(item, dict) and str(item.get("npu")) == "0"]; print(json.dumps(contexts, indent=2)); raise SystemExit(1 if contexts else 0)'
```

- [ ] **Step 2: Document common environment and artifact failures**

Add symptom/cause/resolution/verification/status entries for all of the following:

1. Python 3.12 uv environment reports `No module named pip` or `No module named 'rebel'`, while `/usr/bin/python3.10` finds the SDK in `~/.local/lib/python3.10/site-packages`.
2. Keep `RBLN_BUILD_PY` for Model Zoo compilation and `RBLN_RUN_PY` for framework execution; verify both with `sys.executable` and `importlib.util.find_spec("rebel")`.
3. `origin/feat/rbln-runtime-monitor` is not initially available: fetch the exact branch and use the existing isolated worktree rather than modifying another accelerator experiment.
4. `cp: cannot stat 'resnet50.rbln'`: locate the artifact with `find "$RBLN_ZOO_ROOT" -type f -name '*.rbln'` and copy the resolved file.
5. `LOADING_FILE_NOT_FOUND` with an empty printed artifact path: export and verify `RBLN_SQUAD_ARTIFACT` using `test -s` before constructing `rebel.Runtime`.
6. SDK 0.11 rejects `timeout=60.0`: use an integer `runtime_timeout_sec=60`; keep the accepted range `[1, 2147483647]`.
7. SDK inspect may return a mapping or an attribute object: show a `field(value, name)` helper that supports both forms.

- [ ] **Step 3: Document ResNet50 and YOLOv5m evidence**

For ResNet50, record:

- The first short sync run reported zero average utilization because only two monitor polls were collected.
- The 3,000-sample sync run completed with Top-1 `80.7333`, Top-5 `95.0333`, average latency `0.7161 ms`, P99 `0.8135 ms`, and `1396.5449 samples/s`.
- The initial 3,000-request async attempt stopped making visible progress during measurement with an idle 66 MiB context.
- The verified rerun used `worker_count=1`, queue capacity 16, debug logging, request trace saving, and completed all 3,000 requests with status `valid`, zero logical/native failures, `500.1366 samples/s`, E2E P99 `39.5664 ms`, and queue-wait P99 `35.9588 ms`.

For YOLOv5m, record:

- The required Ultralytics submodule initialization command.
- The verified 128-request async run: mAP@0.5 `0.5903`, average inference latency `6.9404 ms`, async throughput `89.2293 samples/s`, E2E P99 `197.2488 ms`, queue-wait P99 `174.9483 ms`, status `valid`, and zero logical/native failures.
- Explain that the high offline E2E latency is dominated by bounded-queue backlog rather than the model service time alone.

- [ ] **Step 4: Document BERT SST-2 and PatchTST evidence**

For BERT SST-2, include:

- The Hugging Face URI failure for an unnamespaced dataset repository and the rule to use an explicit namespaced repository when the installed Hub/Datasets combination requires it.
- The fixed `(1,128)` int64 `input_ids` and `attention_mask` contract.
- The 872-sample sync result: accuracy `79.5872`, average latency `1.5951 ms`, P99 `1.5996 ms`, and `626.9312 samples/s`.
- The valid 872-request async result: `357.3785 samples/s`, E2E P99 `53.6269 ms`, queue-wait P99 `47.8658 ms`, and zero logical/native failures.

For PatchTST ETTh1, include:

- `NotImplementedError: ... ['aten::unfold']` and the static patch extraction workaround that produced 42 patches with length/stride 12 and passed CPU equivalence.
- `ValueError: Invalid integer data type 'b'` from Relay bool `clamp_min` and the wrapper-level bool-to-float mask conversion after exact CPU equivalence verification.
- Preserve the external fixed input contract while explaining the internal mask conversion.
- The valid 240-window async result: MAE `0.4242`, RMSE `0.6217`, average latency `1.4640 ms`, `337.0399 samples/s`, E2E P99 `55.2329 ms`, queue-wait P99 `50.4033 ms`, and zero logical/native failures.

- [ ] **Step 5: Document BERT SQuAD without overstating completion**

Record these resolved contracts:

- Inputs are `input_ids`, `attention_mask`, and `token_type_ids`, all int64 `(1,384)`.
- The deployed artifact SHA256 is `caada10a3e055df43b24ac388e8fccb5b71fc8fe4a1c08c51dca91922a600b33`.
- SDK 0.11 reports two unnamed float32 `(1,384)` outputs.
- CPU/NPU diagnostics identify output 0 as `start_logits` and output 1 as `end_logits`.
- Real-versus-zero token-type sensitivity proves that the NPU graph uses `token_type_ids`.
- CPU and NPU both selected span 11 through 15 and decoded `neural processing unit inference performance`.
- The SHA-bound `model.rbln.json` sidecar names the two positional outputs but does not certify numerical accuracy.

Record these open issues separately:

- Strict per-logit CPU/NPU equality failed: start context MAE `1.655135`, correlation `0.734280`; end context MAE `0.952200`, correlation `0.940383`.
- Padding logits showed low correlation and large maximum differences.
- `BertQAEvaluator` currently performs argmax without a persisted context mask, so final SQuAD benchmark validity requires context masking and task-level validation.
- SQuAD sync/full and asynchronous benchmark completion have not yet been demonstrated in the captured results.

- [ ] **Step 6: Add async and monitoring interpretation**

Add one comparison table with these exact rows:

| Model | Samples | Async samples/s | E2E P50/P99 ms | Queue P99 ms | NPU util avg/max | Memory MB | Power avg/max W | Energy J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv5m | 128 | 89.2293 | 189.1151 / 197.2488 | 174.9483 | 25.8 / 38.7 | 82 | 18.85 / 18.89 | 29.0225 |
| BERT SST-2 | 872 | 357.3785 | 49.6675 / 53.6269 | 47.8658 | 30.52 / 49.0 | 180 | 36.99 / 47.77 | 78.8359 |
| PatchTST ETTh1 | 240 | 337.0399 | 52.0644 / 55.2329 | 50.4033 | 2.35 / 4.7 | 16 | 18.93 / 18.93 | 15.6191 |
| ResNet50 | 3000 | 500.1366 | 35.1874 / 39.5664 | 35.9588 | 19.6 / 24.3 | 66 | 38.91 / 41.65 | 238.2348 |

Explain the three latency boundaries and require these invariants:

```text
accepted_requests == completed_requests == evaluator_samples
failed_requests == rejected_requests == timed_out_requests == 0
outstanding_requests == native_inflight == 0
native_duplicate_callbacks == native_late_callbacks == 0
native_submit_failures == native_timeouts == 0
monitor_coverage == 1.0
```

- [ ] **Step 7: Add final status and rerun checklist**

Use these final status labels:

| Model | Sync E2E | Async offline full | Remaining work |
|---|---|---|---|
| ResNet50 | Passed | Passed, 3,000 requests | Server-like and concurrency sweep |
| YOLOv5m | Passed | Passed, 128 requests | Server-like and concurrency sweep |
| BERT SST-2 | Passed | Passed, 872 requests | Server-like and concurrency sweep |
| PatchTST ETTh1 | Passed | Passed, 240 windows | Server-like and concurrency sweep |
| BERT SQuAD | Not yet accepted | Not run | Context-masked evaluation and task-level validation |
| Llama 3.1 8B / 3.2 3B | Out of static scope | Out of static scope | Future in-process `rbln-vllm` target |

Finish with a sequential checklist: inspect artifact, verify dataset contract, ensure zero starting contexts, run sync smoke, run sync full, run async offline at worker 1, verify exact accounting, save CSV/details/trace, and confirm `contexts: []` before starting the next model.

- [ ] **Step 8: Verify and commit the guide**

Run:

```bash
rg -n 'ResNet50|YOLOv5m|BERT SST-2|PatchTST ETTh1|BERT SQuAD|aten::unfold|clamp_min|runtime_timeout_sec|async_native_inflight|contexts: \[\]' framework/docs/rbln-troubleshooting.md
git diff --check
git diff -- framework/docs/rbln-troubleshooting.md
```

Expected: every keyword is present, `git diff --check` prints nothing, and only the new guide is shown.

Commit:

```bash
git add framework/docs/rbln-troubleshooting.md
git commit -m "docs: add RBLN troubleshooting guide"
```

### Task 2: Link the troubleshooting guide from the operations guide

**Files:**
- Modify: `framework/docs/rbln-setup.md:1-10`

**Interfaces:**
- Consumes: `framework/docs/rbln-troubleshooting.md` from Task 1.
- Produces: a discoverable relative link from the normative operations guide.

- [ ] **Step 1: Add the troubleshooting link**

After the opening paragraph in `framework/docs/rbln-setup.md`, add:

```markdown
실제 통합 과정에서 관찰한 오류 메시지, 원인, 복구 절차와 모델별 검증 결과는
[RBLN-CA22 트러블슈팅](rbln-troubleshooting.md)에 정리한다.
```

- [ ] **Step 2: Verify link and content consistency**

Run:

```bash
test -f framework/docs/rbln-troubleshooting.md
rg -n 'rbln-troubleshooting\.md' framework/docs/rbln-setup.md
rg -n 'Ubuntu 22\.04\.5|Python 3\.10\.12|rebel-compiler==0\.11\.0|RBLN-CA22|KMD/FW 3\.2\.2' framework/docs/rbln-troubleshooting.md
git diff --check
git status --short
```

Expected: the linked file exists, the setup guide has one relative link, all baseline values are present, there are no whitespace errors, and only `framework/docs/rbln-setup.md` is uncommitted after Task 1.

- [ ] **Step 3: Commit the navigation change**

```bash
git add framework/docs/rbln-setup.md
git commit -m "docs: link RBLN troubleshooting guide"
```

- [ ] **Step 4: Perform final documentation review**

Run:

```bash
git status --short
git log --oneline -3
git show --stat --oneline HEAD~1..HEAD
```

Expected: clean worktree and two implementation commits after the design/plan commits.
