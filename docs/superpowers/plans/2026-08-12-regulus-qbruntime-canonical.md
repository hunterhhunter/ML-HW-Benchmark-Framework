# Regulus qbruntime Canonical Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish `mobilint-regulus` as the canonical qbruntime board path with NPU-only binding evidence in every result.

**Architecture:** Keep the shared `MobilintRuntime` and enable strict bundle/core verification only through Regulus target options. Surface the proof through existing runtime diagnostics, result persistence, and canonical documentation; do not remove ARIES or legacy compatibility code.

**Tech Stack:** Python 3, qbruntime, mbltml, pytest, CSV/JSON result artifacts.

## Global Constraints

- Canonical target ID is exactly `mobilint-regulus`.
- The canonical runtime name is `mobilint`; it invokes the `qbruntime` SDK.
- Regulus NPU-only proof requires device 0, bundle 0, and Cluster0/Core0.
- `npu_only_verified` remains false unless every binding check succeeds.
- ARIES targets do not require Regulus-only bundle/core APIs.
- Native async is not presented as a validated Regulus path in this change.

---

### Task 1: Add strict Regulus binding evidence

**Files:**

- Modify: `framework/src/runtimes/mobilint_rt.py`
- Modify: `framework/src/core/targets.py`
- Test: `framework/tests/test_mobilint_runtime.py`

**Interfaces:** `MobilintRuntime.get_device_spec()` will expose
`runtime_version: str | None`, `npu_only_verified: bool`, and
`execution_binding: str | None`.

- [ ] Write tests for a verified `mobilint-regulus` configuration and for a
  false bundle setter, a nonzero bundle getter, and an incorrect launch core.
  Keep an ARIES test that loads when the fake SDK lacks Regulus-only APIs.
- [ ] Run the focused tests and observe RED.
- [ ] Add validated `npu_bundle_index` and `require_npu_only_binding` options;
  force bundle 0 before launch, validate Cluster0/Core0 after launch, and clear
  proof state during cleanup. Add the three options only to `mobilint-regulus`.
- [ ] Run `test_mobilint_runtime.py`, `test_mobilint_device.py`, and
  `test_mobilint_collector.py` and observe GREEN.
- [ ] Commit with `feat(regulus): verify qbruntime NPU-only binding`.

### Task 2: Persist binding evidence

**Files:**

- Modify: `framework/src/main.py`
- Modify: `framework/src/core/result_store.py`
- Test: `framework/tests/test_main_paths.py`
- Test: `framework/tests/test_result_store.py`

**Interfaces:** CSV rows will carry `runtime_version`, `npu_only_verified`, and
`execution_binding`; async detail JSON retains the same values under
`runtime_device_spec`.

- [ ] Write failing diagnostics allowlist and CSV persistence tests for the
  exact values `v1.2.0`, `True`, and
  `device=0,bundle=0,core=Cluster0/Core0`.
- [ ] Run the focused tests and observe RED.
- [ ] Allowlist only validated Mobilint evidence, add three optional
  `save_result()` parameters and schema fields, and pass them through existing
  sync/async save paths.
- [ ] Run focused main/result/async tests and observe GREEN.
- [ ] Commit with `feat(regulus): persist qbruntime binding evidence`.

### Task 3: Publish canonical documentation

**Files:**

- Create: `docs/mobilint-regulus-runtime.md`
- Modify: `docs/mobilint-accel-runtime.md`
- Modify: `docs/regulus-runtime.md`
- Modify: `README.md`
- Modify: `framework/README.md`

- [ ] Add one canonical sync command using `--target mobilint-regulus`,
  `--monitor`, a `.mxq` artifact, batch 1, and `Model.infer()` timing.
- [ ] Describe mbltml utilization/memory/temperature collection accurately.
- [ ] Replace current-board `regulus`/`regulus-maccel` commands with
  `mobilint-regulus`; retain only a short historical compatibility note in the
  old maccel document.
- [ ] Search documentation for stale current-board commands and confirm all
  remaining maccel mentions are historical context.
- [ ] Commit with `docs(regulus): publish qbruntime board workflow`.

### Task 4: Verify and prepare main integration

- [ ] Run all Mobilint runtime/device/collector, main-path, result-store, and
  async CLI tests.
- [ ] Over SSH, confirm qbruntime, mbltml Regulus-family detection, device 0,
  a real `Model.launch()`, and a real `Model.infer()`.
- [ ] Run one ResNet50 sync smoke and inspect CSV/details for verified binding.
- [ ] Run `git diff --check main...HEAD` and confirm the branch is clean before
  presenting the three commits for explicit main merge/push.
