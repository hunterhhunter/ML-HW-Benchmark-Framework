# TTM-R1 Cross-Vendor Compilation Design

## Goal

Add a reproducible, strict compilation and execution path for the pretrained
`ibm-granite/granite-timeseries-ttm-r1` checkpoint on Rebellions CA22,
Furiosa RNGD, and Mobilint ARIES. The path must retain CPU reference parity,
never enable device fallback, and leave a machine-readable result for both
success and failure.

## Scope

The benchmark fixes the public model boundary to one univariate sample:

```text
context:  float32 [1, 512, 1]
forecast: float32 [1, 96, 1]
```

TTM-R1's native context length and prediction length are used without slicing
or rolling forecasts. The checkpoint is acquired from Hugging Face once into a
caller-selected local model directory; runtime commands must not silently
download or change weights.

## Architecture

### Model-specific CPU host adapter

`ttm_r1` owns the reference loading, deterministic test contexts, NaN policy,
scaling, and conversion between the public context tensor and the core's
static tensors. The adapter must compare its reconstructed forecast against
the unmodified `TinyTimeMixerForPrediction` reference for a finite case and a
NaN-containing case before any device compiler is invoked.

The adapter will keep only operations whose CPU execution has exact reference
parity. If the native patcher contains a compiler-hostile operation such as
`unfold`, it remains on CPU and its output becomes an explicit core input.
Otherwise the core takes `past_values` directly. The selected ABI is recorded
in each result JSON rather than assumed in documentation.

### Static NPU core

The core contains only the pretrained TinyTimeMixer prediction network and
its prediction head. It has no data-dependent branches, dynamic sequence
lengths, Python generation loop, eager fallback, or vendor-specific numerical
relaxation. The wrapper exposes tensors, never a Hugging Face model-output
object.

### Vendor runners and evidence

Each vendor runner compiles the identical core ABI and executes it on the
physical NPU. It writes a result JSON containing checkpoint identity, input
and output ABI, artifact path/hash/size when one exists, NPU/compiler version,
reference parity, device parity, and an explicit status:

```text
passed | compile_failed | execution_failed | parity_failed
```

The runners are strict: Rebellions must create and execute an `.rbln` artifact;
Furiosa uses full-graph compilation with eager fallback disabled; Mobilint
exports ONNX, verifies its CPU reference result, then compiles and executes an
ARIES MBLT artifact. A compilation error is preserved as evidence, not hidden
by CPU execution.

## CLI and automation

`framework/tools/ttm_r1_compile.py` provides the following user-run commands:

```text
--vendor cpu
--vendor rbln
--vendor furiosa
--vendor mobilint
```

Every invocation takes `--model-path` and `--output-dir`; it uses no implicit
current-directory model path. A small acquisition command downloads the exact
checkpoint into the model path and prints its resolved revision and file hash.
Tests do not require any vendor SDK, accelerator, or model download.

## Testing and acceptance criteria

Unit tests use lightweight fake TTM components to prove the tensor contract,
CPU adapter semantics, static wrapper semantics, CLI dispatch, result writing,
and strict vendor configurations. The implementation starts with failing tests
for each behavior, then supplies the minimum implementation.

The real-device acceptance gate requires all of the following per vendor:

1. A compilation artifact is produced for the fixed core ABI.
2. The artifact is executed on the named physical device.
3. CPU reference and host-adapter parity are exact for finite and NaN cases.
4. Device parity meets the shared recorded tolerance without changing it per
   vendor.

TTM-R1 is only a cross-vendor benchmark success when all three vendors pass
these gates. A one-vendor pass is reported as vendor-specific evidence, not as
common support.

## Deliberate non-goals

- Refactoring the existing Chronos-Bolt implementation into a general TSFM
  framework.
- Claiming official vendor support before artifacts have compiled and run.
- Adding TTM-R2 or a larger checkpoint before the TTM-R1 gate is complete.
- Benchmarking multivariate, rolling, or fine-tuned TTM variants in this task.
