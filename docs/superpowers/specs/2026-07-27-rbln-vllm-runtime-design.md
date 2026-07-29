# Rebellions vLLM Runtime Design

## Goal

Add an in-process `rbln-vllm` generation target that runs prepared RBLN model
directories through `vllm-rbln`, while preserving the framework's existing
E2E, native-async, evaluation, monitoring, and result-storage contracts.

The first model profiles are Llama 3.2 3B and Llama 3.1 8B. The runtime must
represent both the officially supported eight-NPU configuration and an
explicitly unsupported single-NPU experiment for Llama 3.2 3B. Llama 3.1 8B
must fail before engine initialization on a 15.7 GiB single NPU because its
unquantized weights alone cannot fit with KV cache and runtime allocations.

## Scope

This change includes:

- a separate `rbln-vllm` target and `rbln_vllm` runtime registry entry;
- synchronous in-process generation through `vllm.LLM`;
- native asynchronous streaming through `vllm.AsyncLLMEngine`;
- prompt trimming, output normalization, TTFT/TPOT observations, and clean
  shutdown through the existing runtime interfaces;
- explicit single- versus multi-NPU preflight validation;
- RBLN telemetry through the existing `rbln-smi` collector;
- a manual preparation utility and server runbook for downloading, compiling,
  and benchmarking both requested models.

Automatic compile-on-load and the OpenAI-compatible HTTP server remain out of
scope. The runtime consumes a local, already prepared Optimum RBLN directory.

## Target Contract

`rbln-vllm` is a generation-only local NPU target:

- runtime: `rbln_vllm`;
- artifact format: `rbln_llm_dir`;
- capabilities: `generation`, `native_async`, `streaming`, `token_events`,
  `monitor`, `npu`, `local`, and `continuous_batching`;
- monitoring: `rbln` plus `system`;
- default device group: one local NPU;
- compiler: none.

`--model-path` is the prepared RBLN directory and `--tokenizer-path` defaults
to that directory. The directory must contain `config.json` and at least one
`.rbln` file below it. It must also contain a locally loadable tokenizer
(`tokenizer_config.json` plus `tokenizer.json` or `tokenizer.model`), unless an
equivalent separate local tokenizer directory is supplied. A Hugging Face
repository ID is not accepted at this runtime boundary because the stable
vLLM RBLN workflow uses a precompiled Optimum RBLN directory.

## Engine Configuration

The runtime forwards the upstream vLLM arguments supported by vLLM RBLN:

- `block_size`: required and positive;
- `max_model_len`: positive when supplied; otherwise read from the framework
  preparation manifest;
- `max_num_seqs`: positive, default `1`;
- `tensor_parallel_size`: fixed at `1`;
- `dtype`, `seed`, and `trust_remote_code`;
- `additional_config.rbln_config.decoder_batch_sizes` when supplied.

When a framework preparation manifest is present, its `num_devices`,
`max_seq_len`, `block_size`, `batch_size`, and compiled decoder buckets are
validated against the runtime options. In particular, compiled `batch_size`
must equal runtime `max_num_seqs`; execution-only batch expansion is rejected.

RBLN device grouping is not vLLM tensor parallelism. The runtime sets
`VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK` to `num_devices` while constructing an
engine and restores the caller's previous environment afterward. `num_devices`
defaults to `1`. Explicit device placement remains owned by vLLM RBLN; the
framework records the requested count and primary monitor device.

For the requested model profiles:

- Llama 3.2 3B official mode: `num_devices=8`;
- Llama 3.2 3B experimental mode: `num_devices=1`, `max_num_seqs=1`, and a
  short static context such as 512 or 1024;
- Llama 3.1 8B official mode: `num_devices=8`;
- Llama 3.1 8B single-NPU mode: rejected before importing/constructing vLLM.

The unsupported 3B experiment requires
`allow_unsupported_single_npu=true`. This opt-in is recorded in device
metadata so an experimental result cannot be mistaken for an official
support claim.

## Runtime Lifecycle

`load()` validates the artifact and preflight contract but defers expensive
engine construction. The first execution path selects one engine mode:

- `generate()` constructs and owns one synchronous `LLM`;
- `create_native_backend()` constructs and owns one `AsyncLLMEngine` on a
  dedicated event-loop thread.

Mixing synchronous and native-async engines in the same loaded runtime is
rejected. This prevents two full model contexts from being created on the
same NPU. `unload()` shuts down the selected engine, clears references, and is
idempotent. A failed async shutdown prevents the model reference from being
discarded, so a live NPU context is never reported as successfully unloaded.

## Synchronous Generation

Inputs contain `input_ids` and an optional same-shaped `attention_mask`.
Prompt rows are normalized to two dimensions and selected by the boolean mask,
which handles both left and right padding. Empty prompts, rank mismatches, and
batch sizes above `max_num_seqs` fail before vLLM submission.

`SamplingParams` uses deterministic generation (`temperature=0.0`) and passes
`max_tokens` plus normalized stop-token IDs. Results are normalized to padded
`generated_ids`, `generated_lengths`, token count, wall-clock total time, and
vLLM request metrics when present.

## Native Async and Streaming

The framework's existing `async_queue` remains the only admission queue. It
owns request limits, arrival schedules, timeout policy, backpressure,
accounting, and result validity. The RBLN adapter must not add another Python
worker pool or batcher.

The native backend owns one event-loop thread and one `AsyncLLMEngine`. Each
framework request becomes a unique vLLM request ID and consumes the engine's
async generator. Cumulative token counts produce
`GenerationOutputEvent` entries. Final output produces exactly one
`NativeAsyncOutcome` containing:

- `generated_ids` and `generated_lengths`;
- `total_ms`, TTFT, and TPOT when observed;
- a cumulative `GenerationObservation` source named
  `rbln_vllm_async_python_stream`;
- a bounded error type and message on failure.

Abort and shutdown are idempotent. Shutdown first stops admission, aborts
active request IDs, waits for callbacks to retire, shuts down the async engine,
stops the event loop, and joins its thread. The common
`NativeAsyncRuntimeExecutor` continues to detect duplicate/late callbacks and
physical completion.

## Preflight and Monitoring

Preflight uses the local model `config.json` and optional `rbln-smi -j` device
inventory. It validates:

1. task/model family is Llama causal generation;
2. prepared RBLN files exist;
3. `block_size <= max_model_len` and divides `max_model_len`;
4. `tensor_parallel_size == 1`;
5. requested NPU count is available when inventory is readable;
6. the requested model's official NPU count or explicit experiment opt-in;
7. single-NPU estimated BF16 weight bytes plus a safety reserve fit device
   memory.

Inventory failure does not silently claim sufficient hardware. It is fatal
before engine initialization for both official and experimental modes; the
server must provide a valid `rbln-smi -j` inventory.

The initial target keeps the existing single-device `RblnCollector` on device
0. It records utilization, memory, temperature, power, energy, coverage, and
process context for the primary engine device. Multi-NPU fleet aggregation is
not fabricated; the run metadata records `num_devices`, and the server guide
requires before/during/after `rbln-smi -j` capture for all devices.

## Model Preparation

`framework/tools/prepare_rbln_vllm_model.py` is a manual preparation utility,
not a framework compiler plugin. It downloads through Optimum RBLN, compiles
with a static batch/context/device contract, saves the complete prepared model
and tokenizer directory, and writes a manifest containing model ID, SDK
versions, compile arguments, file hashes, and support classification. Compiled
batch size and decoder buckets are part of that immutable contract.

The utility defaults to the official eight-NPU configurations. A one-NPU
Llama 3.2 3B attempt requires the same explicit unsupported opt-in. A one-NPU
Llama 3.1 8B request is rejected before download/compile.

## Validation

SDK-free tests use fake `vllm` modules and fake `rbln-smi` output to prove:

- target/registry/CLI selection and local-directory validation;
- exact sync engine arguments and temporary environment scoping;
- prompt trimming, batching, output normalization, and timing extraction;
- single-NPU support guards and multi-NPU inventory checks;
- async streaming events, callback exactly-once behavior, failure/abort paths,
  and bounded shutdown;
- lazy imports so non-RBLN environments remain usable;
- preparation utility argument validation and manifest construction;
- no regression in the existing RBLN static and common async suites.

Real hardware acceptance separately verifies both E2E and `async_queue`,
throughput, TTFT/TPOT p50/p95/p99, utilization/power/energy/coverage, and an
empty `rbln-smi -j` context list after process exit.
