# RBLN Troubleshooting Document Design

## Goal

Create one operator-facing troubleshooting document for the validated
Rebellions RBLN-CA22 integration work. The document must preserve the actual
failure evidence, root causes, fixes, verification commands, and current model
status so another engineer can reproduce the setup without reconstructing the
original terminal history.

## Location and navigation

- Add `framework/docs/rbln-troubleshooting.md`.
- Link it from `framework/docs/rbln-setup.md` near the beginning of the
  operations guide.
- Keep setup and normative runtime contracts in `rbln-setup.md`; keep observed
  failures and recovery procedures in the troubleshooting document.

## Organization

Use a model-first organization selected by the user:

1. Common environment and artifact checks.
2. ResNet50.
3. YOLOv5m.
4. BERT SST-2.
5. PatchTST ETTh1.
6. BERT SQuAD.
7. Cross-model asynchronous inference and monitoring interpretation.
8. Final status matrix and rerun checklist.

Each troubleshooting entry follows the same contract:

1. **Symptom**: exact error or observable behavior.
2. **Cause**: the evidence-backed failure boundary.
3. **Resolution**: the smallest verified recovery action.
4. **Verification**: commands and success criteria.
5. **Status**: resolved, mitigated, or open.

## Required evidence

The document records the validated server baseline:

- Ubuntu 22.04.5, Python 3.10.12.
- `rebel-compiler==0.11.0`.
- RBLN-CA22 device 0, KMD and firmware 3.2.2.
- The split build/runtime Python environments used with `uv`.

It covers these observed failures:

- A Python 3.12 uv environment without `pip` or `rebel`, while the SDK was
  installed in the Python 3.10 user site.
- Git remote/worktree setup errors encountered while isolating the RBLN
  branch.
- Artifact discovery/copy errors caused by using the wrong compile working
  directory.
- SDK 0.11 sync runtime construction rejecting a float timeout.
- Missing Hugging Face `datasets` dependencies and non-namespaced dataset URI
  failures.
- Short monitoring runs reporting zero utilization because only a few vendor
  samples were collected.
- PatchTST compilation failures for `aten::unfold` and bool `clamp_min`.
- BERT SQuAD three-input requirements, unnamed output binding, inspect result
  dict/object compatibility, missing artifact environment variables, and
  CPU/NPU numerical divergence.
- An initially incomplete ResNet50 asynchronous full run and the later valid
  3,000-sample rerun.

## Result reporting

Record the final asynchronous full-run evidence for ResNet50, YOLOv5m, BERT
SST-2, and PatchTST ETTh1. Include only the metrics needed to distinguish model
quality, service latency, end-to-end queue latency, throughput, device load,
power/energy, and lifecycle validity.

Explain that:

- Offline queue wait is expected to dominate end-to-end latency when the
  producer fills the bounded queue.
- Evaluator latency, async service time, and async end-to-end latency measure
  different boundaries.
- `hw_accel_energy_j` is whole-card energy and includes idle power.
- A valid result requires exact accounting, zero logical/native failures, and
  an external `rbln-smi -j` check showing no remaining contexts.

## Accuracy and open-issue policy

Do not present hypotheses as fixes. In particular:

- BERT SQuAD output position and single-sample semantic agreement are verified,
  but strict per-logit CPU/NPU equality is not.
- The current BERT QA evaluator applies argmax without a persisted context mask;
  record this as an open benchmark-validity issue rather than silently treating
  it as solved.
- Llama 3.1 8B and 3.2 3B remain outside the static adapter and belong to a
  future in-process `rbln-vllm` target.

## Verification

Documentation verification consists of:

- Searching both documents for every supported model and critical diagnostic
  keyword.
- Checking all shell examples for consistent paths and environment variables.
- Running `git diff --check`.
- Reviewing the final diff to ensure no unrelated files changed.
