# DEEPX DX-RT Native Async Main Forward-Port Design

## Goal

Implement DEEPX DX-RT 3.3 native asynchronous inference on top of the latest
`origin/main`, while preserving the existing synchronous E2E path and keeping
the change isolated from the earlier Hailo-based stacked branch.

The implementation must support the framework's `e2e` and `async_queue`
inference modes with precompiled DXNN artifacts. Device validation uses
ResNet50/ImageNet-1K and YOLOv5M/COCO128 on Jetson with DX-RT 3.3.2.

## Confirmed Device Failure

The first DeepX branch registered a DX-RT callback during runtime load. In
`async_queue` mode, the framework then performed warmup through the synchronous
`runtime.warmup() -> run()` path. DX-RT invoked the registered callback during
that synchronous warmup without a framework-owned async `user_arg` token. The
adapter recorded an unmatched callback and rejected every later `run_async()`
submission with:

```text
DeepX native async pipeline is recovering from an unmatched callback
```

The device evidence isolates warmup as the trigger:

- `warmup=0`: 8 submitted, 8 completed, 0 failed, status `valid`.
- `warmup=2`: 8 submitted, 0 completed, 8 failed, status `invalid`.

## Branch Strategy

The previous branch was stacked on `feat/hailo-native-async` because the common
native executor had not yet landed on main. It is now 156 commits behind
`origin/main` and contains Hailo-specific history. The forward-port therefore
starts from `origin/main` at `ad9be09` on a new branch:

```text
agent/deepx-dxrt-native-async-main
```

Only DeepX runtime, registry, tests, and operational documentation changes will
be applied. The old branch and the user's dirty original worktree remain
untouched.

## Architecture

The implementation follows the current main-branch native backend contract
already used by RBLN and Mobilint.

### `DeepXRuntime`

`DeepXRuntime` remains the owner of the DXNN model and SDK inference engine.
Its synchronous `run()` behavior remains unchanged for E2E measurements.

It adds:

- validated DX-RT async options (`buffer_count` and completion timeout);
- `native_async_max_batch_size()` returning `1`;
- `create_native_backend()` returning one runtime-owned
  `DeepXNativeBackend` instance;
- warmup routing based on whether the native backend has been created;
- unload coordination with the native backend.

E2E never asks for a native backend, so its warmup remains synchronous.

### `DeepXNativeBackend`

The native backend owns callback-specific state and registers the DX-RT
callback only when `async_queue` selects the DeepX native path. It implements
the main-branch `NativeAsyncRuntimeExecutor` backend contract:

```text
submit_async(inputs, callback) -> vendor job ID
shutdown(timeout) -> bool
```

It also provides `run_warmup_blocking(inputs, timeout)`. This method submits one
request through the same DX-RT callback path used for measured async requests,
waits for its completion, validates the outcome, and returns only after the
callback-owned output has been copied. Repeating this method implements
`--warmup N` without entering the synchronous DX-RT API.

### Mode and Warmup Flow

```text
e2e
  load runtime -> no native backend -> runtime.warmup(run) -> measured run

async_queue
  load runtime -> create native backend/register callback
  -> runtime.warmup(run_warmup_blocking/run_async)
  -> NativeAsyncRuntimeExecutor measured submissions
  -> executor shutdown -> runtime unload/unregister callback/dispose
```

## DX-RT Protocol

- Each asynchronous submission represents exactly one sample.
- Single-input models use `run_async`; named multi-input models use
  `run_async_multi_input` when available.
- Every job receives a unique exact-integer `user_arg` token.
- Callback completion is matched by token and may arrive inline or out of
  submission order.
- SDK-owned output arrays are deep-copied before returning from the callback.
- Input payloads remain owned until DX-RT has completed the corresponding job.
- Unmatched tokens, malformed outputs, duplicate callbacks, submission errors,
  and bounded completion timeouts produce explicit `NativeAsyncOutcome`
  failures without leaking pending jobs.

## Shutdown and Error Handling

The backend refuses unsafe shutdown while jobs or active callbacks remain.
Within the supplied timeout it waits for callback publication to finish,
unregisters the callback, proves the callback lane is quiescent, and releases
job-owned inputs. `DeepXRuntime.unload()` disposes the SDK engine only after the
native backend reports a clean shutdown.

Warmup failures are raised before measurement begins. A failed or timed-out
warmup must not leave unmatched tokens or in-flight work that could contaminate
the measured run.

## Registry and CLI Integration

The DeepX target gains the `native_async` capability. The latest main CLI then
constructs `DeepXNativeBackend` only for `--inference-mode async_queue` through
the existing `_build_async_runtime_executor()` path. No Hailo or generic async
runner behavior is changed.

Runtime options remain compatible with the device commands:

```text
device_ids=0
bound_option=NPU_ALL
buffer_count=6
async_completion_timeout_sec=30
```

`worker_count` must not exceed the DeepX in-flight capacity, and async batch
size remains one.

## Tests

Test-driven implementation will add a fake DX-RT engine that reproduces the
Jetson behavior: a synchronous call made while a callback is registered emits
an unmatched callback. The primary regression test must fail before the fix
and prove that:

1. E2E warmup uses synchronous `run()` without registering a native callback.
2. Async warmup uses `run_async()` through `DeepXNativeBackend`.
3. Warmup completion leaves no unmatched callbacks or jobs.
4. Measured async submissions succeed immediately after warmup.

Additional tests cover callback-owned output copying, single and multi-input
submission, out-of-order and inline callbacks, malformed output, timeout,
shutdown, callback unregister, option validation, target registration, and
blocking E2E compatibility.

Focused DeepX and common native executor suites must pass. The full main suite
will also run, with the pre-existing Furiosa native backend baseline recorded
separately: 11 Furiosa timeout failures were reproducible before any DeepX
change, while 1963 tests passed and 13 were skipped.

## Device Acceptance

On Jetson with DX-RT 3.3.2:

- ResNet50 and YOLOv5M E2E runs complete normally.
- Both models complete `async_queue` with `warmup=2` and `worker_count=4`.
- Submitted, accepted, and completed request counts match.
- Failed, rejected, timed-out, and outstanding counts are zero.
- `async_run_status` is `valid`.
