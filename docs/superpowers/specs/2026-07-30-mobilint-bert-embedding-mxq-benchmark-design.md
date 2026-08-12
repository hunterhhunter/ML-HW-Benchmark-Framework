# Mobilint BERT Embedding-MXQ Benchmark Design

## Goal

Run the already compiled BERT SST-2 and BERT SQuAD v1 MXQ artifacts through
the framework's normal `main.py` benchmark path on ARIES. Reuse the existing
BERT loaders and evaluators and the existing `MobilintRuntime`; do not add a
wrapper loader, a second Mobilint runtime, or a BERT-specific output decoder.

The measured latency uses the framework's existing runtime-call boundary:
`MobilintRuntime.run()`, which contains the ARIES inference call and bounded
output-contract normalization. Host-side BERT embedding construction belongs
to input preparation and must finish before the framework enters that timed
boundary.

## Verified Artifact Boundary

The two MXQ files produced with qbcompiler 1.2 are not full token-input BERT
artifacts. The compiler leaves the embedding layer on the host and exposes one
dynamic float input:

| Task | MXQ input | MXQ outputs |
|---|---|---|
| SST-2 | one `Float32` tensor shaped `[1, -1, 768]` | one tensor reported as `[1, 1, 2]` |
| SQuAD v1 | one `Float32` tensor shaped `[1, -1, 768]` | two tensors reported as `[1, -1, 1]` |

ARIES verification established that the SQuAD artifact returns the two
unnamed SDK outputs in `end_logits`, `start_logits` order. The framework must
bind that positional order explicitly before passing the named dictionary to
the existing evaluator.

The existing framework BERT profiles instead describe full-model token
inputs (`input_ids`, `attention_mask`, and for QA `token_type_ids`). Passing
those arrays directly to either MXQ is a contract error, not an equivalent
execution mode.

## Target and Task Classification

Both artifacts use the existing `mobilint-aries` target and
`MobilintRuntime`/qb Runtime path:

- SST-2 remains `Task.NLP_CLASSIFICATION`.
- SQuAD v1 remains `Task.QUESTION_ANSWERING`.
- Neither artifact uses the `mobilint-aries-llm` Model Zoo generation path.
- Neither artifact enters a vision loader, vision profile, or vision decoder.

`mobilint-aries-llm` remains reserved for autoregressive Model Zoo models such
as Llama. BERT is a static tensor MXQ even though its sequence dimension is
dynamic.

## Architecture

### Existing loader with an optional transform

Do not introduce `MobilintBertLoader(BertLoader)` or another loader factory
branch. Add one optional `input_transform` callable to the two existing BERT
loaders. The default is `None`, preserving the current ONNX, CPU, CUDA, and
other accelerator paths byte-for-byte at the payload boundary.

Each loader continues to own dataset access and labels. Immediately before it
returns a payload, it applies the transform when one was supplied:

```text
token NumPy arrays + labels
  -> optional input_transform(token arrays)
  -> existing payload {input, label}
  -> BenchmarkRunner
```

The same hook must be used by `load_single`, `load_batch`, and
`load_by_index`. This keeps sequential, warmup, and indexed request paths on
one input contract. The loader remains the only object responsible for
advancing its cursor and retrieving labels.

### Shared embedding transform

A single `MobilintBertEmbeddingTransform` serves classification and QA. It
loads `weight_dict.pth` once during construction, converts the five validated
embedding tensors to host tensors, and exposes a callable accepting the
loader's token dictionary.

For every batch-size-one input it:

1. validates `input_ids` and `attention_mask`, plus optional
   `token_type_ids`;
2. creates zero `token_type_ids` when SST-2 does not provide them;
3. trims right padding using the positive prefix selected by
   `attention_mask`;
4. computes word, token-type, and position embeddings;
5. applies the saved LayerNorm weights with epsilon `1e-12`;
6. returns one contiguous `float32` array named `embeddings`, shaped
   `[1, valid_sequence_length, 768]`.

The initial feature supports batch size exactly one. Variable-length samples
cannot be stacked safely without reintroducing padding that the MXQ no longer
has an attention-mask input to ignore. The CLI must reject larger batches
before model allocation.

### Mobilint BERT artifact profiles

Add small, explicit backend profiles for the two compiled boundaries. A
profile carries:

- framework model name and task;
- artifact-profile identifier;
- ordered input name, dtype, and dynamic logical shape;
- ordered SDK output names and logical output shapes;
- maximum batch size `1`;
- native async support `False`;
- expected embedding width `768`;
- positional output order (`logits` for SST-2 and
  `end_logits`, `start_logits` for SQuAD).

The profile selected for `mobilint-aries` produces the runtime-facing
`Model_Spec`. The task and model name remain unchanged so the existing loader
and evaluator factories still select `BertClassificationLoader`,
`BertQALoader`, `BertClassificationEvaluator`, and `BertQAEvaluator`.

The operator supplies the embedding weights explicitly through a
Mobilint-BERT-specific CLI path option. Missing, non-file, or unreadable
weights fail before loader construction. The `.mxq` continues to be supplied
through `--artifact`; neither file is copied into Git.

### Dynamic tensor contract and output normalization

Generalize only the existing Mobilint named-tensor contract:

- `-1` is allowed as a dynamic dimension in a declared Mobilint tensor
  profile and in SDK metadata.
- Other zero or negative values remain invalid.
- Concrete runtime arrays must still have positive dimensions.
- Contract matching treats `-1` as a single-dimension wildcard, not as an
  arbitrary-rank wildcard.
- Extra leading or trailing singleton dimensions reported by qb Runtime are
  ignored only when the declared logical rank is smaller.

After output validation, `MobilintRuntime` reshapes named tensor-contract
outputs to their declared logical batched shape. This is generic MXQ boundary
normalization, not a BERT decoder. It converts the SDK's singleton-heavy
representations to the existing evaluator contracts:

- SST-2 `logits` becomes `[1, 2]`.
- SQuAD `start_logits` and `end_logits` each become `[1, sequence_length]`.

The runtime maps positional SDK arrays to the ordered names declared by the
artifact profile. Therefore the SQuAD reversed SDK order is corrected without
a separate decoder object. Runtime diagnostics and saved result metadata must
include the artifact-profile identifier and declared output order.

## Data and Timing Flow

The synchronous framework flow is:

```text
BERT NumPy dataset
  -> existing BERT loader
  -> host embedding transform and padding trim
  -> BenchmarkRunner/InferencePipeline
  -> timer starts
  -> MobilintRuntime.run()
       -> qbruntime.Model.infer()
       -> generic MXQ output normalization/name binding
  -> timer stops
  -> existing BERT evaluator
  -> existing result store
```

`BlockingRuntimeExecutor` already starts its timer immediately before
`runtime.run()`. Because the transform executes while the loader creates the
payload, embedding time is excluded from `Average Latency`, `P99 Latency`, and
derived throughput metrics. As in the current framework, the measured runtime
call includes the small amount of host work required to validate and name SDK
outputs; qb Runtime does not expose a separate device-kernel timer here.
Command wall-clock time still naturally includes dataset loading and host
preparation; it is not presented as NPU latency.

## CLI and Failure Behavior

The first supported execution is synchronous `--inference-mode e2e` with:

- `--target mobilint-aries`;
- one of the two existing BERT model names;
- the matching `--artifact` MXQ;
- the existing prepared NumPy dataset;
- the matching embedding-weight file;
- `--batch-size 1`;
- `--no-compile`;
- an explicit ARIES core mode, initially `single` to match server
  verification.

With qb Runtime v1.3, the default `single` selection must pass an explicit
`CoreId(Cluster0, Core0)` list to `set_single_core_mode`; calling the setter
without `num_cores` or `core_ids` is invalid. This compatibility fix belongs
in the existing Mobilint core-mode configuration rather than in the BERT
transform.

Fail before launching the MXQ when:

- model, artifact profile, or weights do not match a supported Mobilint BERT
  task;
- batch size is not exactly one;
- weights are missing required keys or have incompatible shapes;
- MXQ metadata does not expose one float32 `[1, -1, 768]` input;
- output count or logical shapes differ from the selected artifact profile.

An invalid sample fails before `runtime.run()` when its attention mask has no
valid tokens, contains non-prefix padding, exceeds the saved position table,
or produces an embedding width other than `768`.

Native `infer_async()` is out of scope for this first connection. These
CPU-offload BERT artifacts remain on the blocking executor even if the caller
later selects the framework async queue. Hardware native async must be enabled
only after a separate ARIES safety and correctness verification.

## Compatibility

- Existing BERT loaders behave exactly as before when `input_transform` is
  absent.
- Existing ONNX and accelerator BERT commands keep the original token-input
  `Model_Spec`.
- Existing Mobilint vision profiles and their raw image preprocessing remain
  unchanged.
- Existing static PatchTST behavior is unchanged; dynamic wildcard support is
  generic but not required by a profile until its real MXQ is inspected.
- Existing Mobilint artifact fail-fast validation remains enabled.

## Verification

Host tests must prove:

1. the embedding transform trims padding, inserts missing token types,
   reproduces the saved embedding formula, emits contiguous float32, and
   rejects invalid masks/weights;
2. both existing loaders invoke the optional transform in single, batch, and
   indexed paths while their no-transform behavior remains unchanged;
3. Mobilint BERT profiles preserve the existing task/evaluator selection and
   declare the real ordered MXQ boundaries;
4. dynamic `-1` matches exactly one positive runtime dimension and does not
   weaken unrelated shape checks;
5. qb Runtime singleton-heavy outputs normalize to `[1, 2]` and
   `[1, sequence_length]` with SQuAD names bound in the verified reversed
   order;
6. CLI validation rejects missing weights and batch sizes greater than one
   before qbruntime model allocation;
7. existing Mobilint, BERT loader/evaluator, main-path, and vision tests remain
   green.

ARIES validation then runs both existing model names through `framework/src/main.py`
for a short smoke and a 64-sample benchmark. The run must show:

- target `mobilint-aries` and the correct artifact-profile identifier;
- SDK input `Float32 [1, -1, 768]` accepted before launch;
- SST-2 accuracy consistent with the already verified 59/64 result;
- SQuAD outputs bound to the correct start/end names and non-degenerate
  accuracy metrics;
- latency records derived only from the qb Runtime call;
- clean model disposal and no remaining ARIES process.

## Out of Scope

- recompiling either BERT artifact;
- including host embedding time in NPU latency;
- adding a wrapper BERT loader or BERT-specific runtime;
- native qb Runtime async for CPU-offload BERT;
- SQuAD sliding-window/text-normalized metric redesign;
- PatchTST compilation or benchmark connection in this branch;
- changing the `mobilint-aries-llm` generation path.
