# Regulus qbruntime Canonical Path Design

## Goal

Make `mobilint-regulus` the documented Regulus board target and make every
published Regulus result contain verifiable qbruntime NPU-only binding evidence.

## Current baseline

`main` already provides the correct implementation family:

- `mobilint-regulus` selects `MobilintRuntime`.
- `MobilintRuntime` calls `qbruntime.Accelerator`, `qbruntime.Model`,
  `Model.launch()`, and `Model.infer()`.
- `MobilintDeviceSession` validates a Regulus or USB-Regulus `mbltml` device.
- the `mobilint` collector records mbltml utilization, memory, and temperature.

The old `regulus-maccel` documentation does not describe this implementation.
The remaining gap is proof that the loaded model is bound to bundle 0 and
Cluster0/Core0, followed by persistence of that proof in result artifacts.

## Design

`mobilint-regulus` will pass three Regulus-only runtime options:

- `core_mode="single"`
- `npu_bundle_index=0`
- `require_npu_only_binding=True`

When this flag is set, `MobilintRuntime.load()` will do the following in order:

1. build `CoreId(Cluster0, Core0)` through the qbruntime API and require
   `set_single_core_mode()` to report success;
2. require `force_single_npu_bundle(0)` to report success and
   `get_forced_npu_bundle_index()` to return `0`;
3. launch the model on `Accelerator(0)` and require `get_target_cores()` to
   return only Cluster0/Core0;
4. set `npu_only_verified=True` only after all checks and model-contract
   validation succeed.

ARIES targets retain their existing configurable core-mode behaviour and do not
need these Regulus-only APIs.

`get_device_spec()` exposes `runtime_version`, `npu_only_verified`, and
`execution_binding`. Main's diagnostics allowlist copies these values into
sync and async result metadata. `result_store` persists them in CSV rows, and
async detail JSON retains them in `runtime_device_spec`.

## Documentation contract

The canonical user-facing target is `mobilint-regulus`, not `regulus` or
`regulus-maccel`. Its runtime is named `mobilint` in the framework and uses the
qbruntime SDK underneath. `--monitor` activates `mobilint`/mbltml plus system
telemetry on supported boards.

`docs/mobilint-accel-runtime.md` remains only as a short migration note so
existing links do not break; it must not tell users to use maccel. A new
`docs/mobilint-regulus-runtime.md` contains the canonical preflight and sync
E2E command. `docs/regulus-runtime.md`, root `README.md`, and
`framework/README.md` link to the canonical guide and use the correct target.

## Verification

Unit tests cover successful binding, each failed binding check, unchanged
ARIES compatibility, safe diagnostics filtering, and CSV persistence. The
Regulus board smoke test confirms qbruntime version, mbltml family detection,
device 0, and a real `Model.launch()`/`Model.infer()` call. Native async remains
non-canonical until the separate output-buffer lifetime change is integrated
and tested.
