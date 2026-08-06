# TTM-R2 Cross-Vendor Validation Design

## Goal

Validate the official `ibm-granite/granite-timeseries-ttm-r2` **main** revision
(`512-96-r2`) on Rebellions CA22, Furiosa RNGD, and Mobilint ARIES. Preserve the
existing ETTh1 comparison contract: one univariate context `[1,512,1]` produces
one 96-step point forecast `[1,96,1]`.

## Scope and selection

`main` is the 512-context, 96-horizon R2 checkpoint. It has the same
TinyTimeMixer input/output structure as the existing R1 benchmark, so it is the
only R2 variant included in this pass. Frequency-prefix variants, longer
contexts, and R2.1 are out of scope. The checkpoint is a distinct pretrained
weight set, not an architecture-size increase.

## Architecture

1. Generalize the current checkpoint loader and fixed core under a versioned
   TTM interface. The core keeps patchification and standard scaling on the CPU
   exactly as in R1, and exports only the tensor model core.
2. Download R2 main into a new local directory and write a file-hash manifest.
   The revision and checkpoint hash are recorded in every result.
3. Before every vendor attempt, compare the original public model output and
   the CPU-split core output for finite and NaN-containing fixed fixtures.
4. Invoke existing strict vendor paths without fallback:
   - CA22: try the full fixed core, then record stage-chain evidence if whole
     graph lowering fails.
   - RNGD: `torch.compile(fullgraph=True, dynamic=False, eager_fallback=False)`.
   - ARIES: export static ONNX, verify with CPU ONNX Runtime, calibrate into
     MXQ locally, transfer only MXQ plus fixture to the ARIES runtime host.
5. Where an artifact executes, measure 240 ETTh1 OT test windows alongside the
   same CPU reference. Strict tensor parity and task quality remain separate
   statuses.

## Error handling and evidence

Every attempt writes a non-overwriting JSON result. `compile_failed` means no
vendor artifact or device execution; `parity_failed` means an artifact ran but
missed the FP32 gate; `measured` means ETTh1 quality was calculated. Compiler
versions, target device, artifact hashes, tensor ABI, and calibration identity
are included.

## Acceptance criteria

- CPU split parity is exact within `rtol=1e-5, atol=1e-6` before a device call.
- Each vendor receives the same R2 checkpoint revision and fixed ABI.
- Each vendor outcome is classified from real artifact/runtime evidence, not
  operator assumptions.
- ETTh1 quality is produced only for artifacts that actually execute.
