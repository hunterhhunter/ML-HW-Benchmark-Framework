# TimesFM 2.5 Cross-Vendor Design

## Goal

Evaluate the same pretrained `google/timesfm-2.5-200m-transformers` checkpoint on Rebellions CA22, Furiosa RNGD, and Mobilint ARIES.  The outcome for each vendor must distinguish reference success, strict compilation success, runtime success, numeric parity, and ETTh1 forecasting quality.

## Fixed Evaluation Contract

- Checkpoint: `google/timesfm-2.5-200m-transformers`, downloaded once into the framework model cache with a file manifest and SHA-256 evidence.
- Reference API: `TimesFm2_5ModelForPrediction` in FP32 eval mode, following the checkpoint's public forecasting path.
- Context: 1 univariate ETTh1 `OT` series with exactly 1,024 historical values.
- Horizon: the checkpoint's native 128-step point forecast.
- Reference result: `mean_predictions` with shape `[1, 128]`; quantile output is out of scope.
- Dataset split: ETTh1 chronological test split, with no calibration or target leakage into the reference test windows.

The checkpoint has a maximum context length of 16,384 and a native 128-step horizon; `forecast_context_len=1024` is the selected fixed benchmark context rather than a change to the checkpoint architecture.

## Architecture

The public prediction model is the behavioral authority.  The implementation first captures deterministic CPU reference output from its public inference call.  A static wrapper is then derived from that path, but only after a component-level comparison proves that it returns the same FP32 point forecast for the fixed `[1, 1024]` input.

The wrapper separates only operations that cannot form a stable common NPU graph:

- CPU host adapter: list-to-tensor packing, fixed-shape padding/masking construction, and any dynamic post-processing required by the public prediction API, including the configured flip-invariance combination if it cannot be traced statically.
- Device core: patch embedding, the 20-layer TimesFM decoder, and point-forecast projection.
- CPU restore: only an inverse normalization or output selection that is proved equivalent to the public API.

No normalization, patching, masking, or forecast combination is moved to the host merely to make a compiler succeed.  Each proposed split must be compared against the public reference before vendor compilation starts.

## Vendor Strategy

1. Create a dedicated reference environment with a Transformers release exposing `TimesFm2_5ModelForPrediction`; do not upgrade any vendor SDK environment globally.
2. Run import and CPU preflight separately in each vendor environment.  A missing model class is recorded as an environment incompatibility, not a model compile failure.
3. Compile the complete static device core first on each vendor.  The first run must forbid fallback where the vendor API supports that control.
4. Record artifact path, hash, compiler/runtime version, inspected ABI, device status before/after, CPU-vs-device tensor metrics, and failure traceback where applicable.
5. Only if full-core compilation fails, split the graph at a measured CPU boundary to localize the blocker.  Stage splitting is diagnostic, not the default deployment design.

ARIES compilation occurs locally with qbcompiler and only MXQ artifacts plus fixtures are transferred to the ARIES server for qbruntime execution.  MBLT artifacts are not treated as qbruntime-executable MXQ artifacts.

## Quality Protocol

For every vendor artifact that runs:

- Compare a deterministic finite fixture to CPU using max absolute error, mean absolute error, RMSE, mismatch count, and the explicit tolerance used.
- Measure ETTh1 point-forecast MAE and RMSE across a fixed chronological set of 128-step test windows.
- Report CPU task metric, NPU task metric, prediction delta, and percentage degradation separately from strict tensor parity.

`strict_parity_failed` does not imply that task quality failed; conversely, a task-quality result is invalid unless the CPU reference and checkpoint manifest match the compiled fixture.

## Failure Handling

- A public-reference/static-wrapper mismatch stops compilation work and is fixed at the wrapper boundary.
- Unsupported operators, compiler crashes, artifact-load failures, and NPU numerical mismatch are separate result states.
- Inputs that saturate during ARIES quantization are recorded explicitly.  Calibration or quantization settings are changed only one variable at a time and compared with the same fixture.

## Scope Boundaries

This work adds TimesFM 2.5 validation only.  It does not change existing Chronos-Bolt or TTM-R1 evidence, retrain or fine-tune TimesFM, change vendor SDK versions globally, or claim cross-vendor support before all three artifacts execute.
