# RBLN Llama 3.1 8B Single-NPU Experiment Design

## Goal

Allow an explicitly opted-in Llama 3.1 8B experiment on one RBLN-CA22 while
preserving every existing execution contract for Llama 3.2 3B, ResNet50,
YOLOv5m, PatchTST, BERT SST-2, and BERT SQuAD.

The change does not claim official support or guaranteed capacity. It creates
a bounded compile-and-load experiment that can distinguish compiler failure,
host-memory failure, NPU-memory failure, and successful inference on the
actual server.

## Non-goals

- Do not change `rbln-static`, its artifact contracts, or its async backend.
- Do not change any vision, time-series, classification, or QA model profile.
- Do not change the existing one-NPU Llama 3.2 3B limits or defaults.
- Do not change the official eight-NPU Llama 3.1 8B contract.
- Do not add quantization or infer the precision used by the Mobilint Aries
  experiment.
- Do not treat compile success as proof that the model fits at runtime.

## Existing behavior that must remain stable

| Path | Required behavior after this change |
|---|---|
| Llama 3.2 3B, one NPU | Still requires explicit opt-in, batch 1, and at most 1024 tokens |
| Llama 3.2 3B, eight NPUs | Still classified as `official` |
| Llama 3.1 8B, eight NPUs | Still classified as `official` with unchanged defaults |
| `rbln-static` models | No production code or model profile changes |
| Llama model identity | Manifest/config/CLI identity mismatches remain rejected |
| Artifact overwrite protection | Existing output directories remain rejected |

## Selected approach

Reuse the existing `allow_unsupported_single_npu` opt-in rather than adding a
second escape hatch. The flag already means that the operator accepts a
non-official single-NPU contract. Model-specific validation keeps the 3B and
8B experiments independent.

The rejected alternatives are:

1. A one-off Optimum script outside the framework. This could probe the
   compiler quickly but would omit the manifest and would not prove the
   framework runtime path.
2. A dedicated `allow_unsupported_single_npu_8b` flag. It is explicit, but it
   duplicates the existing safety mechanism without adding a distinct policy
   boundary.

## Compile contract

`framework/tools/prepare_rbln_vllm_model.py` accepts Llama 3.1 8B with one NPU
only when all of the following are true:

- `allow_unsupported_single_npu` is `True`.
- `num_devices == 1`.
- `batch_size == 1`.
- `max_seq_len` is explicitly supplied or resolves to 512.
- `max_seq_len <= 512`.
- `block_size` divides `max_seq_len` and does not exceed it.
- `decoder_batch_sizes == [1]` after normalization.

The resulting manifest records:

```json
{
  "model": "llama-3.1-8b",
  "num_devices": 1,
  "max_seq_len": 512,
  "block_size": 512,
  "batch_size": 1,
  "decoder_batch_sizes": [1],
  "support_classification": "unsupported_single_npu_experiment"
}
```

The default and official eight-NPU compile contracts remain unchanged.

## Runtime contract

`framework/src/runtimes/rbln_vllm_rt.py` accepts a prepared one-NPU Llama 3.1
8B directory only when:

- Artifact identity resolves to `llama-3.1-8b` without ambiguity.
- The manifest declares one device, batch 1, and decoder batch 1.
- `allow_unsupported_single_npu == True`.
- `max_num_seqs == 1`.
- Resolved `max_model_len <= 512`.
- Device inventory contains one healthy RBLN device.
- Device total memory is readable and at least 15 GiB.

The 15 GiB check is a coarse experiment prerequisite, not a fit guarantee.
Native engine creation remains the authoritative capacity test. If it fails,
the original SDK/vLLM exception is preserved so that an NPU allocation error
can be distinguished from a contract rejection.

Successful loads expose
`support_classification=unsupported_single_npu_experiment` through the normal
device-spec and result metadata path.

## Server execution flow

The server uses a new, non-overwriting output directory:

```text
~/rebelion/rbln-model-zoo/custom/framework-contracts/
  llama-3.1-8b-npu1-seq512/
```

Execution proceeds through strict gates:

1. Verify at least 30 GiB free disk space, no active RBLN context, and normal
   device status.
2. Compile in `tmux` with one device, sequence 512, block 512, batch 1, and
   decoder batch 1.
3. Inspect the manifest and generated `.rbln` files; record sizes and SHA256.
4. Run one synchronous sample with one generated token.
5. Confirm process exit and `rbln-smi -j` contexts are empty.
6. Only after the synchronous gate succeeds, run four asynchronous requests
   with one worker and queue capacity one.
7. Confirm exact request counts, zero failure counters, valid TTFT/TPOT, and
   empty contexts after shutdown.

Compile failure, host OOM, NPU OOM, and successful execution are all valid
experimental outcomes. A failed capacity experiment does not change the
status of the already-supported models.

## Error handling and recovery

- Existing output directories are never overwritten.
- A contract mismatch fails before engine creation.
- A native load failure is reported with its original cause.
- If the benchmark process exits but an RBLN context remains, no subsequent
  benchmark starts until the exact owning PID and context are inspected.
- No broad `killall`, recursive deletion, or unrelated process termination is
  part of the runbook.
- Partial compile output is retained for diagnosis and is moved or removed
  only after the operator confirms the exact path.

## Automated verification

Preparation tests must prove:

- Explicit opt-in accepts one-NPU Llama 3.1 8B at 512 tokens.
- Missing opt-in rejects the same contract.
- Batch greater than one is rejected.
- Sequence length greater than 512 is rejected.
- Existing Llama 3.2 3B and official eight-NPU expectations are unchanged.

Runtime tests must prove:

- Explicit opt-in accepts a matching one-NPU 8B manifest.
- Missing opt-in rejects it before importing vLLM.
- Sequence length and batch violations are rejected.
- The 15 GiB readable-memory prerequisite is enforced for 8B only.
- Existing 3B sync/async behavior, identity checks, lifecycle cleanup, and
  official eight-NPU behavior remain unchanged.

The focused suites are:

```bash
python -m pytest -q \
  framework/tests/test_prepare_rbln_vllm_model.py \
  framework/tests/test_rbln_vllm_runtime.py \
  framework/tests/test_main_paths.py
```

Before publishing, the complete framework test suite must also pass. Static
RBLN regression protection is provided by running its existing runtime,
native-async, collector, registry, and CLI tests without modifying those
production paths.

## Hardware acceptance criteria

A one-NPU Llama 3.1 8B run is accepted only if all of these are true:

- Compile exits successfully and produces a complete prepared directory.
- Manifest and runtime options match exactly.
- One-sample synchronous inference returns generated token output.
- The four-request async run reports `async_run_status=valid`.
- Submitted, accepted, completed, generation-observed, and evaluator counts
  all equal four.
- Failed, rejected, timed-out, outstanding, duplicate-callback,
  late-callback, submit-failure, native-timeout, and native-inflight counts
  are zero.
- Monitoring has at least one successful vendor sample.
- Shutdown completes and device-zero contexts are empty.

If compilation succeeds but runtime allocation fails, the outcome is recorded
as `compiled_but_single_npu_runtime_capacity_failed`, not as framework support
and not as a regression in the other models.
