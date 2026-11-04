"""Mobilint qb Runtime adapter shared by ARIES and REGULUS MXQ targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
import threading
import time
from typing import Any, Callable, Dict

import numpy as np

from core.compiled_model import CompiledModel
from core.runtime_executor import NativeAsyncOutcome
from mobilint_device import MobilintDeviceSession
from .base import Runtime


_CORE_MODES = frozenset({"auto", "single", "multi", "global4", "global8"})
_BACKEND_NAMES = frozenset({"mobilint", "qbruntime", "mxq"})


@dataclass
class _MobilintAsyncJob:
    future: Any
    inputs: list[np.ndarray]
    thread: threading.Thread | None = None
    claim_lock: Any = field(default_factory=threading.Lock, repr=False)
    claimed: bool = False


class MobilintNativeBackend:
    """Bridge qb Runtime Futures to the framework native-async callback API."""

    def __init__(self, runtime: "MobilintRuntime"):
        if runtime._model is None:
            raise RuntimeError(
                "Mobilint MXQ model must be loaded before native async setup."
            )
        if not runtime.async_pipeline_enabled:
            raise RuntimeError(
                "Mobilint native async requires async_pipeline_enabled=True "
                "before load."
            )
        self.runtime = runtime
        self._condition = threading.Condition(threading.RLock())
        self._jobs: dict[str, _MobilintAsyncJob] = {}
        self._threads: set[threading.Thread] = set()
        self._slots = threading.BoundedSemaphore(
            runtime.max_concurrent_workers()
        )
        self._active_submissions = 0
        self._next_job_id = 1
        self._closing = False

    @staticmethod
    def _error_type(exc: BaseException) -> str:
        name = type(exc).__name__
        bounded = "".join(
            character
            for character in name
            if character.isalnum() or character == "_"
        )[:64]
        return bounded or "MobilintAsyncError"

    @staticmethod
    def _validate_single_batch(ordered: list[np.ndarray]) -> None:
        for value in ordered:
            if value.ndim == 0 or value.shape[0] != 1:
                raise ValueError(
                    "Mobilint native async supports batch dimension N=1 only."
                )

    @staticmethod
    def _thread_is_alive(thread: Any) -> bool:
        try:
            return bool(thread.is_alive())
        except BaseException:
            return False

    def submit_async(
        self,
        inputs: Dict[str, np.ndarray],
        callback: Callable[[NativeAsyncOutcome], None],
    ) -> str:
        ordered = self.runtime._ordered_inputs(inputs)
        self._validate_single_batch(ordered)
        with self._condition:
            if self._closing:
                raise RuntimeError(
                    "Mobilint native backend is shutting down."
                )
        if not self._slots.acquire(blocking=False):
            raise RuntimeError(
                "Mobilint native async waiter capacity is exhausted."
            )
        payload = ordered[0] if len(ordered) == 1 else ordered
        with self._condition:
            if self._closing:
                self._slots.release()
                raise RuntimeError(
                    "Mobilint native backend is shutting down."
                )
            self._active_submissions += 1
        try:
            future = self.runtime._model.infer_async(payload)
        except BaseException:
            with self._condition:
                self._active_submissions -= 1
                self._slots.release()
                self._condition.notify_all()
            raise

        use_fallback = False
        with self._condition:
            job_number = self._next_job_id
            self._next_job_id += 1
            job_id = f"mobilint-{job_number}"
            job = _MobilintAsyncJob(future=future, inputs=ordered)
            self._jobs[job_id] = job
            thread = None
            try:
                thread = threading.Thread(
                    target=self._wait_for_job,
                    args=(job_id, job, callback),
                    name=f"mobilint-future-{job_number}",
                    daemon=True,
                )
                job.thread = thread
                self._threads = {
                    item
                    for item in self._threads
                    if self._thread_is_alive(item)
                }
                self._threads.add(thread)
                thread.start()
            except BaseException:
                use_fallback = True
                if thread is not None and not self._thread_is_alive(thread):
                    self._threads.discard(thread)
                    job.thread = None
            finally:
                self._active_submissions -= 1
                self._condition.notify_all()
        if use_fallback:
            # infer_async already accepted the work, so waiter startup errors
            # cannot cross the submission boundary. The claim guard keeps a
            # start-then-raise implementation from consuming the Future twice.
            self._wait_for_job(job_id, job, callback)
        return job_id

    def _wait_for_job(
        self,
        job_id: str,
        job: _MobilintAsyncJob,
        callback: Callable[[NativeAsyncOutcome], None],
    ) -> None:
        with job.claim_lock:
            if job.claimed:
                return
            job.claimed = True
        started_ns = time.perf_counter_ns()
        try:
            outputs = self.runtime._normalize_outputs(job.future.get())
            outcome = NativeAsyncOutcome(
                outputs=outputs,
                timing_ms=(time.perf_counter_ns() - started_ns)
                / 1_000_000.0,
            )
        except BaseException as exc:
            outcome = NativeAsyncOutcome(
                error_type=self._error_type(exc),
                error_message="Mobilint asynchronous inference failed.",
            )
        try:
            callback(outcome)
        except BaseException:
            # Consumer failures must not strand accepted SDK work during unload.
            pass
        finally:
            with self._condition:
                self._jobs.pop(job_id, None)
                job.inputs = []
                self._slots.release()
                self._condition.notify_all()

    def shutdown(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            self._closing = True
            while self._jobs or self._active_submissions:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            threads = tuple(self._threads)
        for thread in threads:
            if not self._thread_is_alive(thread):
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if self._thread_is_alive(thread):
                    return False
                continue
            thread.join(timeout=remaining)
            if self._thread_is_alive(thread):
                return False
        with self._condition:
            self._threads = {
                thread
                for thread in self._threads
                if self._thread_is_alive(thread)
            }
        return True


class MobilintRuntime(Runtime):
    def __init__(self, **runtime_options):
        expected_family = runtime_options.get("expected_family")
        if expected_family is None:
            raise ValueError(
                "MobilintRuntime requires expected_family='aries' or 'regulus'."
            )
        self.device = str(runtime_options.get("device", "npu:0"))
        self.device_id = int(runtime_options.get("device_id", 0))
        self.expected_family = str(expected_family).strip().lower()
        self.async_pipeline_enabled = self._as_bool(
            runtime_options.get("async_pipeline_enabled", False),
            "async_pipeline_enabled",
        )
        activation_slots = runtime_options.get("activation_slots")
        self.activation_slots = (
            None if activation_slots is None else int(activation_slots)
        )
        if self.activation_slots is not None and self.activation_slots <= 0:
            raise ValueError("activation_slots must be a positive integer.")
        self.shutdown_timeout_sec = float(
            runtime_options.get("shutdown_timeout_sec", 5.0)
        )
        if self.shutdown_timeout_sec < 0:
            raise ValueError("shutdown_timeout_sec must be non-negative.")
        core_mode = runtime_options.get("core_mode")
        self.core_mode = (
            None if core_mode is None else str(core_mode).strip().lower()
        )
        if self.core_mode is not None and self.core_mode not in _CORE_MODES:
            raise ValueError(
                "core_mode must be one of auto, single, multi, global4, global8."
            )
        num_cores = runtime_options.get("num_cores")
        self.num_cores = None if num_cores is None else int(num_cores)
        if self.num_cores is not None and self.num_cores <= 0:
            raise ValueError("num_cores must be a positive integer.")
        if self.num_cores is not None and self.core_mode != "single":
            raise ValueError("num_cores is valid only with core_mode='single'.")

        self.compiled_model: CompiledModel | None = None
        self._model = None
        self._accelerator = None
        self._device_session: MobilintDeviceSession | None = None
        self._device_info = None
        self._input_names: tuple[str, ...] = ()
        self._output_names: tuple[str, ...] = ()
        self._sdk_version = None
        self._cleanup_pending = False
        self._native_backend: MobilintNativeBackend | None = None

    @staticmethod
    def _as_bool(value: Any, name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        raise ValueError(f"{name} must be a boolean.")

    @staticmethod
    def _load_qbruntime():
        try:
            return import_module("qbruntime")
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Mobilint MXQ inference requires the optional mobilint-qb-runtime "
                "package from Mobilint SDK v1.3."
            ) from exc

    def _configure_model(self, qbruntime):
        config = qbruntime.ModelConfig()
        if self.core_mode == "auto":
            result = config.set_auto_core_mode()
        elif self.core_mode == "single":
            result = (
                config.set_single_core_mode()
                if self.num_cores is None
                else config.set_single_core_mode(self.num_cores)
            )
        elif self.core_mode == "multi":
            result = config.set_multi_core_mode()
        elif self.core_mode == "global4":
            result = config.set_global4_core_mode()
        elif self.core_mode == "global8":
            result = config.set_global8_core_mode()
        else:
            result = True
        if result is False:
            raise RuntimeError(f"qbruntime rejected core_mode={self.core_mode}.")
        if self.activation_slots is not None:
            if config.set_activation_slots(self.activation_slots) is False:
                raise RuntimeError("qbruntime rejected activation_slots.")
        if self.async_pipeline_enabled:
            if config.set_async_pipeline_enabled(True) is False:
                raise RuntimeError(
                    "qbruntime rejected async pipeline configuration."
                )
        return config

    def _clear_model_state(self) -> None:
        self._model = None
        self._accelerator = None
        self.compiled_model = None
        self._input_names = ()
        self._output_names = ()
        self._sdk_version = None

    def _cleanup_resources(self) -> None:
        if self._model is not None:
            self._model.dispose()
        self._clear_model_state()

        if self._device_session is not None:
            self._device_session.release()
        self._device_session = None
        self._device_info = None
        self._cleanup_pending = False

    def load(self, compiled_model: CompiledModel) -> None:
        if self._cleanup_pending:
            raise RuntimeError(
                "Mobilint MXQ cleanup is incomplete; call unload() to retry."
            )
        if self._model is not None or self._device_session is not None:
            raise RuntimeError("Mobilint MXQ model is already loaded.")
        if not self.is_compatible(compiled_model):
            raise ValueError(
                "MobilintRuntime requires a .mxq artifact with backend 'mobilint'."
            )
        session = MobilintDeviceSession(self.device_id, self.expected_family)
        self._device_session = session
        self._cleanup_pending = True
        try:
            self._device_info = session.acquire()
            qbruntime = self._load_qbruntime()
            sdk_version = getattr(qbruntime, "__version__", None)
            config = self._configure_model(qbruntime)
            self._accelerator = qbruntime.Accelerator(self.device_id)
            self._model = qbruntime.Model(
                str(compiled_model.artifact_path), config
            )
            self._model.launch(self._accelerator)
        except BaseException as load_error:
            try:
                self._cleanup_resources()
            except BaseException as cleanup_error:
                raise RuntimeError(
                    "Mobilint MXQ load failed and rollback cleanup is "
                    f"incomplete ({type(cleanup_error).__name__}: "
                    f"{cleanup_error}); call unload() to retry cleanup."
                ) from load_error
            raise
        self.compiled_model = compiled_model
        self._sdk_version = sdk_version
        self._input_names = tuple(compiled_model.spec.input_shapes)
        self._output_names = tuple(compiled_model.spec.output_shapes)
        self._cleanup_pending = False

    def _ordered_inputs(self, inputs: Dict[str, np.ndarray]) -> list[np.ndarray]:
        missing = [name for name in self._input_names if name not in inputs]
        if missing:
            raise ValueError("missing required inputs: " + ", ".join(missing))
        return [
            np.ascontiguousarray(np.asarray(inputs[name]))
            for name in self._input_names
        ]

    def _normalize_outputs(self, outputs) -> Dict[str, np.ndarray]:
        if outputs is None:
            raise RuntimeError("qbruntime returned no outputs.")
        if not isinstance(outputs, (list, tuple)):
            raise RuntimeError("qbruntime outputs must be a list of arrays.")
        if len(outputs) != len(self._output_names):
            raise RuntimeError(
                f"qbruntime expected {len(self._output_names)} outputs, "
                f"received {len(outputs)}."
            )
        return {
            name: np.asarray(value)
            for name, value in zip(self._output_names, outputs)
        }

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self._cleanup_pending:
            raise RuntimeError(
                "Mobilint MXQ cleanup is incomplete; call unload() to retry."
            )
        if self._model is None:
            raise RuntimeError("Mobilint MXQ model is not loaded. Call load() first.")
        ordered = self._ordered_inputs(inputs)
        payload = ordered[0] if len(ordered) == 1 else ordered
        return self._normalize_outputs(self._model.infer(payload))

    def warmup(self, inputs: Dict[str, np.ndarray], num_runs: int = 1) -> None:
        for _ in range(max(0, int(num_runs))):
            self.run(inputs)

    def native_async_max_batch_size(self) -> int:
        return 1

    def create_native_backend(self) -> MobilintNativeBackend:
        if self._cleanup_pending:
            raise RuntimeError(
                "Mobilint MXQ cleanup is incomplete; call unload() to retry."
            )
        if self._model is None:
            raise RuntimeError(
                "Mobilint MXQ model is not loaded. Call load() first."
            )
        if not self.async_pipeline_enabled:
            raise RuntimeError(
                "Mobilint native async requires async_pipeline_enabled=True "
                "before load."
            )
        if self._native_backend is None:
            self._native_backend = MobilintNativeBackend(self)
        return self._native_backend

    def unload(self) -> None:
        if (
            self._native_backend is None
            and self._model is None
            and self._device_session is None
        ):
            self._cleanup_pending = False
            return
        self._cleanup_pending = True
        native_backend = self._native_backend
        if native_backend is not None:
            if not native_backend.shutdown(timeout=self.shutdown_timeout_sec):
                raise RuntimeError(
                    "Mobilint native async backend did not quiesce; "
                    "Model.dispose() was skipped."
                )
            self._native_backend = None
        self._cleanup_resources()

    def get_device_spec(self) -> Dict[str, Any]:
        return {
            "backend": "mobilint",
            "device": self.device,
            "device_id": self.device_id,
            "expected_family": self.expected_family,
            "detected_family": getattr(self._device_info, "family", None),
            "device_type": getattr(self._device_info, "device_type", None),
            "accelerator_vendor": "Mobilint",
            "accelerator_name": self.expected_family.upper(),
            "async_pipeline_enabled": self.async_pipeline_enabled,
            "activation_slots": self.activation_slots,
            "sdk_version": self._sdk_version,
        }

    def is_compatible(self, compiled_model: CompiledModel) -> bool:
        return (
            compiled_model.artifact_path.suffix.lower() == ".mxq"
            and compiled_model.backend_name.strip().lower() in _BACKEND_NAMES
        )

    def max_concurrent_workers(self) -> int:
        if not self.async_pipeline_enabled:
            return 1
        return self.activation_slots or 1
