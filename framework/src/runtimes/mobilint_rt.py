"""Mobilint qb Runtime adapter shared by ARIES and REGULUS MXQ targets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from importlib import import_module
from numbers import Integral
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
_VISION_CONTRACT_REQUIRED_KEYS = (
    "vision_profile_id",
    "expected_input_dtype",
    "expected_input_layout",
    "expected_unbatched_input_shape",
    "max_input_batch_size",
)
_VISION_CONTRACT_KEYS = frozenset(
    (*_VISION_CONTRACT_REQUIRED_KEYS, "expected_unbatched_output_shapes")
)
_TENSOR_CONTRACT_REQUIRED_KEYS = (
    "artifact_profile_id",
    "expected_input_names",
    "expected_input_dtypes",
    "expected_unbatched_input_shapes",
    "expected_output_names",
    "expected_unbatched_output_shapes",
    "max_input_batch_size",
    "native_async_supported",
)
_TENSOR_CONTRACT_KEYS = frozenset(_TENSOR_CONTRACT_REQUIRED_KEYS)


@dataclass
class _MobilintAsyncJob:
    future: Any
    inputs: list[np.ndarray]
    thread: threading.Thread | None = None
    claim_lock: Any = field(default_factory=threading.Lock, repr=False)
    claimed: bool = False
    slot_released: bool = False


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

    def _release_job_slot(self, job: _MobilintAsyncJob) -> bool:
        with self._condition:
            if job.slot_released:
                return False
            job.slot_released = True
            self._slots.release()
            self._condition.notify_all()
            return True

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
            try:
                outputs = self.runtime._normalize_outputs(
                    job.future.get(),
                    expected_batch_size=job.inputs[0].shape[0],
                )
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
            finally:
                self._release_job_slot(job)
            try:
                callback(outcome)
            except BaseException:
                # Consumer failures must not strand accepted SDK work during unload.
                pass
        finally:
            with self._condition:
                self._jobs.pop(job_id, None)
                job.inputs = []
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

        self._parse_artifact_contract(runtime_options)
        self.compiled_model: CompiledModel | None = None
        self._model = None
        self._accelerator = None
        self._device_session: MobilintDeviceSession | None = None
        self._device_info = None
        self._input_names: tuple[str, ...] = ()
        self._output_names: tuple[str, ...] = ()
        self._sdk_version = None
        self._actual_input_dtype: str | None = None
        self._actual_input_shape: tuple[int, ...] | None = None
        self._actual_input_dtypes: tuple[str, ...] = ()
        self._actual_input_shapes: tuple[tuple[int, ...], ...] = ()
        self._actual_output_shapes: tuple[tuple[int, ...], ...] = ()
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
    def _normalize_dtype(value: Any, name: str) -> str:
        candidate = getattr(value, "name", value)
        try:
            return np.dtype(candidate).name
        except (TypeError, ValueError):
            try:
                return np.dtype(str(candidate).lower()).name
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{name} must be a valid NumPy dtype, received {value!r}."
                ) from exc

    @staticmethod
    def _normalize_shape(
        value: Any,
        name: str,
        *,
        allow_dynamic: bool = False,
    ) -> tuple[int, ...]:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(
                f"{name} must be a non-empty list or tuple of positive integers."
            )
        if any(
            isinstance(dimension, bool)
            or not isinstance(dimension, Integral)
            or (
                dimension <= 0
                and not (allow_dynamic and dimension == -1)
            )
            for dimension in value
        ):
            suffix = " or -1" if allow_dynamic else ""
            raise ValueError(
                f"{name} must be a non-empty list or tuple of positive "
                f"integers{suffix}."
            )
        return tuple(int(dimension) for dimension in value)

    @staticmethod
    def _shape_matches(
        expected: tuple[int, ...],
        actual: tuple[int, ...],
    ) -> bool:
        return len(expected) == len(actual) and all(
            expected_dimension == actual_dimension
            or (
                expected_dimension == -1
                and (actual_dimension == -1 or actual_dimension > 0)
            )
            for expected_dimension, actual_dimension in zip(expected, actual)
        )

    def _initialize_artifact_contract(self) -> None:
        self.artifact_profile_id: str | None = None
        self.native_async_supported = False
        self._expected_input_names: tuple[str, ...] = ()
        self._expected_input_dtypes: tuple[str, ...] = ()
        self._expected_unbatched_input_shapes: tuple[tuple[int, ...], ...] = ()
        self._expected_output_names: tuple[str, ...] = ()
        self._tensor_contract = False
        self.vision_profile_id: str | None = None
        self.expected_input_dtype: str | None = None
        self.expected_input_layout: str | None = None
        self.expected_unbatched_input_shape: tuple[int, ...] | None = None
        self.max_input_batch_size: int | None = None
        self.expected_unbatched_output_shapes: tuple[
            tuple[int, ...], ...
        ] = ()

    def _parse_artifact_contract(self, runtime_options: dict[str, Any]) -> None:
        self._initialize_artifact_contract()
        tensor_specific = _TENSOR_CONTRACT_KEYS.difference(
            {"max_input_batch_size", "expected_unbatched_output_shapes"}
        )
        has_tensor_contract = bool(tensor_specific.intersection(runtime_options))
        has_vision_contract = bool(
            _VISION_CONTRACT_KEYS.intersection(runtime_options)
        )
        if has_tensor_contract and "vision_profile_id" in runtime_options:
            raise ValueError(
                "Mobilint tensor and vision artifact contracts are mutually exclusive."
            )
        if has_tensor_contract:
            self._parse_tensor_contract(runtime_options)
            return
        if has_vision_contract:
            self._parse_vision_contract(runtime_options)

    def _parse_vision_contract(self, runtime_options: dict[str, Any]) -> None:
        supplied = _VISION_CONTRACT_KEYS.intersection(runtime_options)
        if not supplied:
            return
        missing = [
            key
            for key in _VISION_CONTRACT_REQUIRED_KEYS
            if key not in runtime_options
        ]
        if missing:
            raise ValueError(
                "Mobilint vision contract options must all be provided together; "
                "missing " + ", ".join(missing) + "."
            )

        profile_id = runtime_options["vision_profile_id"]
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise ValueError("vision_profile_id must be a non-empty string.")
        self.vision_profile_id = profile_id.strip()
        self.artifact_profile_id = self.vision_profile_id
        self.expected_input_dtype = self._normalize_dtype(
            runtime_options["expected_input_dtype"],
            "expected_input_dtype",
        )
        layout = runtime_options["expected_input_layout"]
        if not isinstance(layout, str) or layout.strip().upper() not in {
            "NCHW",
            "NHWC",
        }:
            raise ValueError("expected_input_layout must be NCHW or NHWC.")
        self.expected_input_layout = layout.strip().upper()
        self.expected_unbatched_input_shape = self._normalize_shape(
            runtime_options["expected_unbatched_input_shape"],
            "expected_unbatched_input_shape",
        )
        max_batch_size = runtime_options["max_input_batch_size"]
        if (
            isinstance(max_batch_size, bool)
            or not isinstance(max_batch_size, Integral)
            or max_batch_size <= 0
        ):
            raise ValueError("max_input_batch_size must be a positive integer.")
        self.max_input_batch_size = int(max_batch_size)
        self.native_async_supported = True
        self._expected_input_dtypes = (self.expected_input_dtype,)
        self._expected_unbatched_input_shapes = (
            self.expected_unbatched_input_shape,
        )

        if "expected_unbatched_output_shapes" not in runtime_options:
            return
        output_shapes = runtime_options["expected_unbatched_output_shapes"]
        if not isinstance(output_shapes, (list, tuple)):
            raise ValueError(
                "expected_unbatched_output_shapes must be a list or tuple of shapes."
            )
        self.expected_unbatched_output_shapes = tuple(
            self._normalize_shape(shape, "expected_unbatched_output_shapes")
            for shape in output_shapes
        )

    @staticmethod
    def _normalize_name_list(value: Any, name: str) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(f"{name} must be a non-empty list of names.")
        normalized = tuple(
            item.strip() if isinstance(item, str) else "" for item in value
        )
        if any(not item for item in normalized) or len(set(normalized)) != len(
            normalized
        ):
            raise ValueError(f"{name} must contain unique non-empty strings.")
        return normalized

    def _parse_tensor_contract(self, runtime_options: dict[str, Any]) -> None:
        missing = [
            key
            for key in _TENSOR_CONTRACT_REQUIRED_KEYS
            if key not in runtime_options
        ]
        if missing:
            raise ValueError(
                "Mobilint tensor contract options must all be provided together; "
                "missing " + ", ".join(missing) + "."
            )
        profile_id = runtime_options["artifact_profile_id"]
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise ValueError("artifact_profile_id must be a non-empty string.")
        input_names = self._normalize_name_list(
            runtime_options["expected_input_names"],
            "expected_input_names",
        )
        input_dtypes = runtime_options["expected_input_dtypes"]
        input_shapes = runtime_options["expected_unbatched_input_shapes"]
        if not isinstance(input_dtypes, (list, tuple)) or len(input_dtypes) != len(
            input_names
        ):
            raise ValueError(
                "expected_input_dtypes must contain one dtype per input name."
            )
        if not isinstance(input_shapes, (list, tuple)) or len(input_shapes) != len(
            input_names
        ):
            raise ValueError(
                "expected_unbatched_input_shapes must contain one shape per input name."
            )
        output_names = self._normalize_name_list(
            runtime_options["expected_output_names"],
            "expected_output_names",
        )
        output_shapes = runtime_options["expected_unbatched_output_shapes"]
        if not isinstance(output_shapes, (list, tuple)) or len(output_shapes) != len(
            output_names
        ):
            raise ValueError(
                "expected_unbatched_output_shapes must contain one shape per output name."
            )
        max_batch_size = runtime_options["max_input_batch_size"]
        if (
            isinstance(max_batch_size, bool)
            or not isinstance(max_batch_size, Integral)
            or max_batch_size <= 0
        ):
            raise ValueError("max_input_batch_size must be a positive integer.")

        self.artifact_profile_id = profile_id.strip()
        self._expected_input_names = input_names
        self._expected_input_dtypes = tuple(
            self._normalize_dtype(value, "expected_input_dtypes")
            for value in input_dtypes
        )
        self._expected_unbatched_input_shapes = tuple(
            self._normalize_shape(
                shape,
                "expected_unbatched_input_shapes",
                allow_dynamic=True,
            )
            for shape in input_shapes
        )
        self._expected_output_names = output_names
        self.expected_unbatched_output_shapes = tuple(
            self._normalize_shape(
                shape,
                "expected_unbatched_output_shapes",
                allow_dynamic=True,
            )
            for shape in output_shapes
        )
        self.max_input_batch_size = int(max_batch_size)
        self.native_async_supported = self._as_bool(
            runtime_options["native_async_supported"],
            "native_async_supported",
        )
        self._tensor_contract = True

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
            if self.num_cores is None:
                core_id = qbruntime.CoreId(
                    qbruntime.Cluster.Cluster0,
                    qbruntime.Core.Core0,
                )
                result = config.set_single_core_mode(None, [core_id])
            else:
                result = config.set_single_core_mode(self.num_cores)
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

    def _model_contract_mismatch(
        self,
        compiled_model: CompiledModel,
        field: str,
        expected: Any,
        actual: Any,
    ) -> RuntimeError:
        return RuntimeError(
            f"Mobilint {field} mismatch for {self.artifact_profile_id} "
            f"artifact {compiled_model.artifact_path.name!r}: "
            f"expected {expected!r}, actual {actual!r}."
        )

    def _model_contract_value(
        self,
        compiled_model: CompiledModel,
        getter_name: str,
    ) -> Any:
        getter = getattr(self._model, getter_name, None)
        if not callable(getter):
            raise self._model_contract_mismatch(
                compiled_model,
                f"SDK metadata getter {getter_name}",
                "a qbruntime SDK v1.3-compatible callable",
                "missing",
            )
        try:
            return getter()
        except BaseException as exc:
            exception_type = "".join(
                character
                for character in type(exc).__name__
                if character.isalnum() or character == "_"
            )[:64] or "Exception"
            raise RuntimeError(
                f"Mobilint SDK metadata getter {getter_name} failed with "
                f"{exception_type}."
            ) from None

    def _model_contract_shape(
        self,
        compiled_model: CompiledModel,
        value: Any,
        field: str,
        expected: Any,
    ) -> tuple[int, ...]:
        try:
            return self._normalize_shape(
                value,
                field,
                allow_dynamic=self._tensor_contract,
            )
        except ValueError as exc:
            raise self._model_contract_mismatch(
                compiled_model, field, expected, value
            ) from exc

    @staticmethod
    def _canonical_contract_shape(
        actual: tuple[int, ...],
        expected: tuple[int, ...],
    ) -> tuple[int, ...]:
        while len(actual) > len(expected) and actual[0] == 1:
            actual = actual[1:]
        return actual

    @classmethod
    def _canonical_tensor_output_shape(
        cls,
        actual: tuple[int, ...],
        expected: tuple[int, ...],
    ) -> tuple[int, ...]:
        extra_dimensions = len(actual) - len(expected)
        if extra_dimensions < 0:
            return actual

        for leading_count in range(extra_dimensions + 1):
            trailing_count = extra_dimensions - leading_count
            leading = actual[:leading_count]
            trailing = (
                actual[len(actual) - trailing_count :]
                if trailing_count
                else ()
            )
            if any(dimension != 1 for dimension in (*leading, *trailing)):
                continue
            stop = len(actual) - trailing_count if trailing_count else None
            candidate = actual[leading_count:stop]
            if cls._shape_matches(expected, candidate):
                return candidate
        return actual

    def _validate_model_contract(
        self, compiled_model: CompiledModel
    ) -> None:
        if self.artifact_profile_id is None:
            return

        input_names = tuple(compiled_model.spec.input_shapes)
        expected_input_names = self._expected_input_names or input_names
        if input_names != expected_input_names:
            raise self._model_contract_mismatch(
                compiled_model,
                "input names/order",
                expected_input_names,
                input_names,
            )

        input_shapes = self._model_contract_value(
            compiled_model, "get_model_input_shape"
        )
        expected_input_shapes = self._expected_unbatched_input_shapes
        if (
            not isinstance(input_shapes, (list, tuple))
            or len(input_shapes) != len(expected_input_shapes)
        ):
            actual_count = (
                len(input_shapes)
                if isinstance(input_shapes, (list, tuple))
                else input_shapes
            )
            raise self._model_contract_mismatch(
                compiled_model,
                "SDK input count",
                len(expected_input_shapes),
                actual_count,
            )
        actual_input_shapes = tuple(
            self._canonical_contract_shape(
                self._model_contract_shape(
                    compiled_model,
                    shape,
                    "input shape",
                    expected,
                ),
                expected,
            )
            for shape, expected in zip(input_shapes, expected_input_shapes)
        )
        self._actual_input_shapes = actual_input_shapes
        self._actual_input_shape = (
            actual_input_shapes[0] if len(actual_input_shapes) == 1 else None
        )
        if not all(
            self._shape_matches(expected, actual)
            for expected, actual in zip(
                expected_input_shapes, actual_input_shapes
            )
        ):
            raise self._model_contract_mismatch(
                compiled_model,
                "input shapes",
                expected_input_shapes,
                actual_input_shapes,
            )

        input_dtypes = self._model_contract_value(
            compiled_model, "get_model_input_data_type"
        )
        if isinstance(input_dtypes, (list, tuple)):
            if len(input_dtypes) != len(expected_input_shapes):
                raise self._model_contract_mismatch(
                    compiled_model,
                    "SDK input dtype count",
                    len(expected_input_shapes),
                    len(input_dtypes),
                )
            raw_input_dtypes = tuple(input_dtypes)
        else:
            if len(set(self._expected_input_dtypes)) != 1:
                raise self._model_contract_mismatch(
                    compiled_model,
                    "SDK input dtypes",
                    self._expected_input_dtypes,
                    input_dtypes,
                )
            raw_input_dtypes = (input_dtypes,) * len(expected_input_shapes)
        try:
            actual_input_dtypes = tuple(
                self._normalize_dtype(value, "SDK input dtype")
                for value in raw_input_dtypes
            )
        except ValueError as exc:
            raise self._model_contract_mismatch(
                compiled_model,
                "input dtypes",
                self._expected_input_dtypes,
                raw_input_dtypes,
            ) from exc
        self._actual_input_dtypes = actual_input_dtypes
        self._actual_input_dtype = (
            actual_input_dtypes[0] if len(actual_input_dtypes) == 1 else None
        )
        if actual_input_dtypes != self._expected_input_dtypes:
            raise self._model_contract_mismatch(
                compiled_model,
                "input dtypes",
                self._expected_input_dtypes,
                actual_input_dtypes,
            )

        if not self.expected_unbatched_output_shapes:
            return
        output_names = tuple(compiled_model.spec.output_shapes)
        expected_output_names = self._expected_output_names or output_names
        if output_names != expected_output_names:
            raise self._model_contract_mismatch(
                compiled_model,
                "output names/order",
                expected_output_names,
                output_names,
            )
        output_shapes = self._model_contract_value(
            compiled_model, "get_model_output_shape"
        )
        if not isinstance(output_shapes, (list, tuple)):
            raise self._model_contract_mismatch(
                compiled_model,
                "output shapes",
                self.expected_unbatched_output_shapes,
                output_shapes,
            )
        expected_output_shapes = self.expected_unbatched_output_shapes
        if len(output_shapes) != len(expected_output_shapes):
            raise self._model_contract_mismatch(
                compiled_model,
                "output count",
                len(expected_output_shapes),
                len(output_shapes),
            )
        canonical_output_shape = (
            self._canonical_tensor_output_shape
            if self._tensor_contract
            else self._canonical_contract_shape
        )
        self._actual_output_shapes = tuple(
            canonical_output_shape(
                self._model_contract_shape(
                    compiled_model,
                    shape,
                    "output shape",
                    expected,
                ),
                expected,
            )
            for shape, expected in zip(output_shapes, expected_output_shapes)
        )
        if len(self._actual_output_shapes) != len(expected_output_shapes):
            raise self._model_contract_mismatch(
                compiled_model,
                "output count",
                len(expected_output_shapes),
                len(self._actual_output_shapes),
            )
        output_shapes_match = (
            all(
                self._shape_matches(expected, actual)
                for expected, actual in zip(
                    expected_output_shapes, self._actual_output_shapes
                )
            )
            if self._tensor_contract
            else Counter(self._actual_output_shapes)
            == Counter(expected_output_shapes)
        )
        if not output_shapes_match:
            raise self._model_contract_mismatch(
                compiled_model,
                "output shapes",
                expected_output_shapes,
                self._actual_output_shapes,
            )

    def _clear_model_state(self) -> None:
        self._model = None
        self._accelerator = None
        self.compiled_model = None
        self._input_names = ()
        self._output_names = ()
        self._sdk_version = None
        self._actual_input_dtype = None
        self._actual_input_shape = None
        self._actual_input_dtypes = ()
        self._actual_input_shapes = ()
        self._actual_output_shapes = ()

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
            self._validate_model_contract(compiled_model)
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
        unexpected = [name for name in inputs if name not in self._input_names]
        if unexpected:
            raise ValueError("unexpected inputs: " + ", ".join(unexpected))
        ordered = []
        for index, name in enumerate(self._input_names):
            array = np.ascontiguousarray(np.asarray(inputs[name]))
            if self.artifact_profile_id is not None:
                self._validate_runtime_input_array(name, array, index=index)
            ordered.append(array)
        if self._tensor_contract and len(ordered) > 1:
            batch_sizes = {array.shape[0] for array in ordered}
            if len(batch_sizes) != 1:
                raise ValueError(
                    "Mobilint input batch dimensions must match for "
                    f"{self.artifact_profile_id}: received "
                    f"{tuple(array.shape[0] for array in ordered)}."
                )
        return ordered

    def _validate_runtime_input_array(
        self,
        name: str,
        array: np.ndarray,
        *,
        index: int = 0,
    ) -> None:
        expected_dtype = self._expected_input_dtypes[index]
        expected_shape = self._expected_unbatched_input_shapes[index]
        if array.dtype.name != expected_dtype:
            raise ValueError(
                f"Mobilint input {name!r} dtype mismatch for "
                f"{self.artifact_profile_id}: expected {expected_dtype}, "
                f"received {array.dtype.name}."
            )
        if array.ndim != len(expected_shape) + 1:
            raise ValueError(
                f"Mobilint input {name!r} rank mismatch for "
                f"{self.artifact_profile_id}: expected batch plus "
                f"{expected_shape}, received {array.shape}."
            )
        if not 1 <= array.shape[0] <= self.max_input_batch_size:
            raise ValueError(
                f"Mobilint input {name!r} batch mismatch for "
                f"{self.artifact_profile_id}: expected 1 <= batch size <= "
                f"{self.max_input_batch_size}, "
                f"received {array.shape[0]}."
            )
        concrete_shape = tuple(array.shape[1:])
        if any(dimension <= 0 for dimension in concrete_shape) or not (
            self._shape_matches(expected_shape, concrete_shape)
        ):
            raise ValueError(
                f"Mobilint input {name!r} shape mismatch for "
                f"{self.artifact_profile_id}: expected "
                f"{expected_shape}, "
                f"received {concrete_shape}."
            )

    def _validate_runtime_output_arrays(
        self,
        arrays: list[np.ndarray],
        expected_batch_size: int | None = None,
    ) -> None:
        if not self.expected_unbatched_output_shapes:
            return
        received_shapes = tuple(tuple(array.shape) for array in arrays)
        if any(
            any(dimension <= 0 for dimension in shape)
            for shape in received_shapes
        ):
            raise RuntimeError(
                f"Mobilint output shape mismatch for {self.artifact_profile_id}: "
                f"received {received_shapes}."
            )
        if self._tensor_contract:
            unbatched_matches = all(
                self._shape_matches(expected, actual)
                for expected, actual in zip(
                    self.expected_unbatched_output_shapes,
                    received_shapes,
                )
            )
        else:
            unbatched_matches = Counter(received_shapes) == Counter(
                self.expected_unbatched_output_shapes
            )
        if (
            (not self._tensor_contract or expected_batch_size in {None, 1})
            and unbatched_matches
        ):
            return
        if expected_batch_size is None:
            batch_sizes = range(1, self.max_input_batch_size + 1)
        elif (
            type(expected_batch_size) is int
            and 1 <= expected_batch_size <= self.max_input_batch_size
        ):
            batch_sizes = (expected_batch_size,)
        else:
            batch_sizes = ()
        for batch_size in batch_sizes:
            batched_shapes = tuple(
                (batch_size, *shape)
                for shape in self.expected_unbatched_output_shapes
            )
            if self._tensor_contract:
                batched_match = all(
                    self._shape_matches(expected, actual)
                    for expected, actual in zip(
                        batched_shapes, received_shapes
                    )
                )
            else:
                batched_match = Counter(received_shapes) == Counter(
                    batched_shapes
                )
            if batched_match:
                return
        raise RuntimeError(
            f"Mobilint output shape mismatch for {self.artifact_profile_id}: "
            f"expected {self.expected_unbatched_output_shapes} all unbatched "
            "only for batch size 1, or all with the requested leading batch "
            "dimension, "
            f"received {received_shapes}."
        )

    def _normalize_outputs(
        self,
        outputs,
        *,
        expected_batch_size: int | None = None,
    ) -> Dict[str, np.ndarray]:
        if outputs is None:
            raise RuntimeError("qbruntime returned no outputs.")
        if not isinstance(outputs, (list, tuple)):
            raise RuntimeError("qbruntime outputs must be a list of arrays.")
        if len(outputs) != len(self._output_names):
            raise RuntimeError(
                f"qbruntime expected {len(self._output_names)} outputs, "
                f"received {len(outputs)}."
            )
        arrays = [np.asarray(value) for value in outputs]
        if self._tensor_contract:
            arrays = self._reshape_tensor_outputs(
                arrays,
                expected_batch_size=expected_batch_size,
            )
        self._validate_runtime_output_arrays(
            arrays,
            expected_batch_size=expected_batch_size,
        )
        return {
            name: value for name, value in zip(self._output_names, arrays)
        }

    def _reshape_tensor_outputs(
        self,
        arrays: list[np.ndarray],
        *,
        expected_batch_size: int | None,
    ) -> list[np.ndarray]:
        if (
            type(expected_batch_size) is not int
            or expected_batch_size < 1
            or expected_batch_size > self.max_input_batch_size
        ):
            raise RuntimeError(
                "Mobilint tensor output requested batch size must be an "
                f"integer in [1, {self.max_input_batch_size}], received "
                f"{expected_batch_size!r}."
            )

        normalized = []
        for name, array, unbatched_shape in zip(
            self._output_names,
            arrays,
            self.expected_unbatched_output_shapes,
        ):
            dynamic_dimensions = tuple(
                index
                for index, dimension in enumerate(unbatched_shape)
                if dimension == -1
            )
            if len(dynamic_dimensions) > 1:
                raise RuntimeError(
                    f"Mobilint output shape mismatch for {name!r}: multiple "
                    "dynamic dimensions cannot be resolved from one output."
                )

            raw_shape = tuple(array.shape)
            expected_batched_shape = (
                expected_batch_size,
                *unbatched_shape,
            )
            canonical_shape = self._canonical_tensor_output_shape(
                raw_shape,
                expected_batched_shape,
            )
            concrete_unbatched_shape = None
            if self._shape_matches(
                expected_batched_shape,
                canonical_shape,
            ):
                concrete_unbatched_shape = canonical_shape[1:]
            elif expected_batch_size == 1:
                canonical_unbatched_shape = (
                    self._canonical_tensor_output_shape(
                        raw_shape,
                        unbatched_shape,
                    )
                )
                if self._shape_matches(
                    unbatched_shape,
                    canonical_unbatched_shape,
                ):
                    concrete_unbatched_shape = canonical_unbatched_shape

            if concrete_unbatched_shape is None:
                raise RuntimeError(
                    f"Mobilint output shape mismatch for {name!r}: raw shape "
                    f"{raw_shape} has element count {array.size}, but only "
                    "extra leading or trailing singleton axes may surround "
                    f"{expected_batched_shape}."
                )

            resolved_shape = list(unbatched_shape)
            if dynamic_dimensions:
                dynamic_index = dynamic_dimensions[0]
                resolved_shape[dynamic_index] = concrete_unbatched_shape[
                    dynamic_index
                ]
            logical_shape = (expected_batch_size, *resolved_shape)
            normalized.append(array.reshape(logical_shape))
        return normalized

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self._cleanup_pending:
            raise RuntimeError(
                "Mobilint MXQ cleanup is incomplete; call unload() to retry."
            )
        if self._model is None:
            raise RuntimeError("Mobilint MXQ model is not loaded. Call load() first.")
        ordered = self._ordered_inputs(inputs)
        payload = ordered[0] if len(ordered) == 1 else ordered
        return self._normalize_outputs(
            self._model.infer(payload),
            expected_batch_size=(
                ordered[0].shape[0]
                if self.artifact_profile_id is not None
                else None
            ),
        )

    def warmup(self, inputs: Dict[str, np.ndarray], num_runs: int = 1) -> None:
        for _ in range(max(0, int(num_runs))):
            self.run(inputs)

    def native_async_max_batch_size(self) -> int | None:
        return 1 if self.native_async_supported else None

    def supports_dynamic_batching(self) -> bool:
        return bool(
            self.artifact_profile_id is not None
            and self.max_input_batch_size is not None
            and self.max_input_batch_size > 1
        )

    def max_dynamic_batch_size(self) -> int:
        return self.max_input_batch_size or 1

    def create_native_backend(self) -> MobilintNativeBackend:
        if self._cleanup_pending:
            raise RuntimeError(
                "Mobilint MXQ cleanup is incomplete; call unload() to retry."
            )
        if self._model is None:
            raise RuntimeError(
                "Mobilint MXQ model is not loaded. Call load() first."
            )
        if not self.native_async_supported:
            raise RuntimeError(
                f"Mobilint artifact {self.artifact_profile_id!r} does not support "
                "SDK native async; use the framework blocking async queue."
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
            "artifact_profile_id": self.artifact_profile_id,
            "native_async_supported": self.native_async_supported,
            "expected_input_names": (
                self._expected_input_names or self._input_names
            ),
            "expected_output_names": (
                self._expected_output_names or self._output_names
            ),
            "expected_input_dtypes": self._expected_input_dtypes,
            "actual_input_dtypes": self._actual_input_dtypes,
            "expected_unbatched_input_shapes": (
                self._expected_unbatched_input_shapes
            ),
            "actual_input_shapes": self._actual_input_shapes,
            "vision_profile_id": self.vision_profile_id,
            "expected_input_dtype": self.expected_input_dtype,
            "actual_input_dtype": self._actual_input_dtype,
            "expected_input_layout": self.expected_input_layout,
            "expected_unbatched_input_shape": (
                self.expected_unbatched_input_shape
            ),
            "max_input_batch_size": self.max_input_batch_size,
            "actual_input_shape": self._actual_input_shape,
            "expected_unbatched_output_shapes": (
                self.expected_unbatched_output_shapes
            ),
            "actual_output_shapes": self._actual_output_shapes,
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
