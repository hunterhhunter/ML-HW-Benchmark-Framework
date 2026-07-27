# Mobilint ARIES Transformer and LLM Runtime Design

## Goal

Extend the existing Mobilint integration beyond ResNet50 and YOLOv5m so the
framework can execute these existing benchmark profiles on ARIES:

- BERT SST-2 classification
- BERT SQuAD extractive QA
- PatchTST ETTh1 forecasting
- Llama 3.1 8B Instruct generation
- Llama 3.2 3B Instruct generation

Compiler integration remains out of scope. BERT and PatchTST therefore require
task-specific `.mxq` files supplied by the operator. Llama uses Mobilint's
official Model Zoo Hugging Face repositories, which already contain the NPU
artifacts and custom Transformers implementation.

## Runtime Boundaries

There are two execution lifecycles and they remain separate.

### Static tensor models

BERT and PatchTST use the existing `MobilintRuntime` and `qbruntime.Model`.
Their existing loaders, decoders, and evaluators remain unchanged. The runtime
orders multiple inputs using `ModelSpec`, converts inputs to contiguous NumPy
arrays, calls `Model.infer()`, and maps ordered SDK outputs back to the declared
output names.

The current vision-only artifact contract is generalized to a named tensor
contract. It records input names, input dtypes, unbatched input shapes, output
shapes, maximum requested batch, and whether SDK native async is valid. The
legacy vision profile fields remain supported and are normalized into the same
internal representation.

### Autoregressive generation

Llama uses `MobilintLlmRuntime` and Model Zoo's
`AutoModelForCausalLM.generate()`. It does not call `qbruntime.infer_async()`.
The official qb Runtime v1.3 contract limits native async to CNN models with
`N=1`; LLM, RNN/LSTM, and CPU-offload models are excluded.

Both E2E and framework `async_queue` therefore call the same blocking
`generate()` operation. In async mode one framework worker owns the Model Zoo
model instance. The framework request queue supplies backpressure and
completion accounting; no second vendor async queue is invented.

## Batch Semantics

Mobilint Llama repositories encode an artifact capacity in `config.json`:

- repository without a `Batch` suffix: `max_batch_size=1`
- `Batch16`: maximum grouped generation batch 16
- `Batch32`: maximum grouped generation batch 32

These values are capacities, not mandatory request sizes. A Batch16 model may
run any actual batch from 1 through 16. A Batch32 model may run 1 through 32.
The CLI `--batch-size` is the actual framework batch requested for a call and
must not exceed the loaded artifact capacity.

For capacity greater than one, the runtime accepts a rectangular padded token
batch, invokes one grouped `generate()` call, and returns a 2-D generated-token
array plus per-row lengths. The existing Llama evaluator already consumes this
format. Streamer callbacks for a grouped batch are aggregate producer events;
the framework must not present them as exact per-request token timestamps.

Mobilint Model Zoo grouped generation is not continuous batching. Requests are
grouped before `generate()` and the group completes together.

## Static Artifact Contract

For non-vision static models the contract is derived from the selected
`ModelSpec`, after the artifact and target are known:

- BERT SST-2: `input_ids`, `attention_mask`; `logits`
- BERT SQuAD: `input_ids`, `attention_mask`, `token_type_ids`;
  `start_logits`, `end_logits`
- PatchTST: `past_values`, `past_observed_mask`; `output`

SDK metadata has no portable tensor-name API, so ModelSpec order is
authoritative. SDK shapes and dtypes are checked against that ordered contract.
Leading singleton dimensions in SDK metadata are treated as representation
details, because qb Runtime artifacts may report the same unbatched tensor with
one or more leading `1` dimensions.

The contract is fail-fast: a compiled artifact whose input count, dtype, or
shape does not match the selected benchmark profile is rejected before dataset
measurement. This is important for BERT and PatchTST because a generic base
model or differently compiled sequence length is not interchangeable with the
task-specific artifact.

## Async Selection

The target advertises that Mobilint can provide native async, but the loaded
runtime instance decides whether the current artifact may use it.

- ResNet50 and YOLOv5m vision profiles: native async, maximum batch 1.
- BERT SST-2, BERT SQuAD, PatchTST: blocking executor inside the framework
  async queue.
- Llama: blocking `generate()` inside the framework async queue.

When native async is unavailable for the selected model,
`native_async_max_batch_size()` returns `None`. Executor selection treats this
as an intentional capability fallback, not as a broken target. Invalid
non-null capability values remain errors.

## Model Acquisition

A repository script downloads the complete selected official Llama repository
with `huggingface_hub.snapshot_download()` into a stable framework model path.
It supports standard, Batch16, and Batch32 variants for both requested Llama
models and writes no model payload into Git.

BERT SST-2, BERT SQuAD, and PatchTST are not downloaded from a misleading base
model repository. The operator supplies the matching compiled `.mxq`, then uses
the inspection utility to print SDK input/output metadata before a benchmark.

## Verification Boundary

Host tests use complete fake qbruntime and Model Zoo boundaries to prove
contract parsing, input ordering, native-async fallback, Llama capacity
validation, single/batched generation output mapping, and cleanup. Actual NPU
correctness and performance are verified later on the ARIES server with the
documented smoke and full-dataset commands.
