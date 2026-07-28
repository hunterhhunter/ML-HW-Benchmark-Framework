from collections.abc import Iterable
from importlib import import_module
import math
from numbers import Integral, Real
from pathlib import Path
import threading
import time
from typing import Any, Dict

import numpy as np

from core.compiled_model import CompiledModel
from core.runtime_executor import NativeAsyncOutcome
from .base import Runtime


class DeepXNativeBackend:
    """DX-RT callback adapter used only by async_queue execution."""

    def __init__(self, runtime: "DeepXRuntime"):
        self.runtime = runtime
        self._engine = runtime._engine
        register_callback = getattr(self._engine, "register_callback", None)
        run_async = getattr(self._engine, "run_async", None)
        if not callable(register_callback) or not callable(run_async):
            raise NotImplementedError(
                "DeepX native async requires DX-RT run_async() and "
                "register_callback() support."
            )
        self._condition = threading.Condition(threading.RLock())
        self._jobs = {}
        self._next_token = 1
        self._closing = False
        self._active_submissions = 0
        self._active_callbacks = 0
        self._callback_threads = {}
        self._callback_registered = False
        register_callback(self._handle_completion)
        self._callback_registered = True

    def submit_async(self, inputs, callback):
        if not callable(callback):
            raise ValueError("DeepX async callback must be callable.")
        ordered_inputs = self.runtime._prepare_ordered_inputs(inputs)
        if self.runtime._infer_batch_size(ordered_inputs) != 1:
            raise ValueError(
                "DX-RT native async inference accepts one sample per job."
            )
        if len(ordered_inputs) == 1:
            input_array = self.runtime._sdk_input_array(ordered_inputs[0][1])
            sdk_payload = (
                input_array
                if self.runtime.single_input_run_style == "array"
                else [input_array]
            )
            submit = self._engine.run_async
        else:
            sdk_payload = {
                name: self.runtime._sdk_input_array(value)
                for name, value in ordered_inputs
            }
            submit = getattr(self._engine, "run_async_multi_input", None)
            if not callable(submit):
                sdk_payload = [sdk_payload[name] for name, _ in ordered_inputs]
                submit = self._engine.run_async

        with self._condition:
            if self._closing:
                raise RuntimeError("DeepX native backend is shutting down.")
            token = self._next_token
            self._next_token += 1
            job = {
                "callback": callback,
                "input_payload": sdk_payload,
                "started_ns": time.perf_counter_ns(),
                "completion_started": False,
                "completion_finished": False,
                "submission_finished": False,
            }
            # Publish before run_async(): DX-RT may invoke the callback inline.
            self._jobs[token] = job
            self._active_submissions += 1
        try:
            vendor_job_id = submit(sdk_payload, user_arg=token)
        except BaseException:
            with self._condition:
                job["submission_finished"] = True
                self._active_submissions -= 1
                if not job["completion_started"] or job["completion_finished"]:
                    self._retire_job_locked(token, job)
                self._condition.notify_all()
            raise
        with self._condition:
            job["submission_finished"] = True
            self._active_submissions -= 1
            if job["completion_finished"]:
                self._retire_job_locked(token, job)
            self._condition.notify_all()
        return vendor_job_id

    def run_warmup_blocking(self, inputs, timeout):
        completed = []
        done = threading.Event()

        def collect(outcome):
            completed.append(outcome)
            done.set()

        self.submit_async(inputs, collect)
        if not done.wait(timeout=max(0.0, float(timeout))):
            raise TimeoutError("DeepX native async warmup timed out.")
        outcome = completed[0]
        if outcome.error_type is not None:
            raise RuntimeError(
                outcome.error_message or "DeepX native async warmup failed."
            )
        return outcome.outputs

    def shutdown(self, timeout):
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            self._closing = True
            if threading.get_ident() in self._callback_threads:
                raise RuntimeError(
                    "Cannot shut down DeepX native backend from its callback."
                )
            while (
                self._jobs
                or self._active_submissions
                or self._active_callbacks
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            unregister = self._callback_registered

        if unregister:
            self._engine.register_callback(None)
            with self._condition:
                self._callback_registered = False
                while self._active_callbacks:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._condition.wait(timeout=remaining)
        return True

    def _handle_completion(self, outputs, user_arg):
        callback_thread_id = threading.get_ident()
        with self._condition:
            self._active_callbacks += 1
            self._callback_threads[callback_thread_id] = (
                self._callback_threads.get(callback_thread_id, 0) + 1
            )
        try:
            completions = []
            with self._condition:
                job = self._jobs.get(user_arg) if type(user_arg) is int else None
                if job is not None and not job["completion_started"]:
                    job["completion_started"] = True
                    completions.append(
                        (user_arg, job, self._completion_outcome(outputs, job))
                    )
                elif job is None and not self._is_retired_token(user_arg):
                    for token, pending in self._jobs.items():
                        if pending["completion_started"]:
                            continue
                        pending["completion_started"] = True
                        completions.append(
                            (
                                token,
                                pending,
                                NativeAsyncOutcome(
                                    timing_ms=self._elapsed_ms(pending),
                                    error_type="DeepXAsyncProtocolError",
                                    error_message=(
                                        "DX-RT callback returned an unmatched "
                                        "user_arg token."
                                    ),
                                ),
                            )
                        )

            for token, pending, outcome in completions:
                try:
                    pending["callback"](outcome)
                except BaseException:
                    # Never propagate Python callback failures through DX-RT.
                    pass
                finally:
                    with self._condition:
                        pending["completion_finished"] = True
                        if pending["submission_finished"]:
                            self._retire_job_locked(token, pending)
                        self._condition.notify_all()
            return 0
        finally:
            with self._condition:
                self._active_callbacks -= 1
                depth = self._callback_threads[callback_thread_id] - 1
                if depth:
                    self._callback_threads[callback_thread_id] = depth
                else:
                    self._callback_threads.pop(callback_thread_id, None)
                self._condition.notify_all()

    def _completion_outcome(self, outputs, job):
        timing_ms = self._elapsed_ms(job)
        try:
            if outputs is None:
                raise ValueError("DX-RT callback returned no outputs.")
            if isinstance(outputs, (dict, list, tuple)) and not outputs:
                raise ValueError("DX-RT callback returned no outputs.")
            if self.runtime._looks_like_batch_outputs(outputs):
                raise ValueError("DX-RT callback returned batched outputs.")
            normalized = self.runtime._normalize_outputs(outputs)
            if not normalized:
                raise ValueError("DX-RT callback returned no outputs.")
            expected_names = self.runtime._output_names
            if expected_names and (
                len(normalized) != len(expected_names)
                or any(name not in normalized for name in expected_names)
            ):
                raise ValueError(
                    "DX-RT callback returned unexpected output names or count."
                )
            return NativeAsyncOutcome(
                outputs={
                    name: self._copy_output(value)
                    for name, value in normalized.items()
                },
                timing_ms=timing_ms,
            )
        except BaseException:
            return NativeAsyncOutcome(
                timing_ms=timing_ms,
                error_type="DeepXAsyncCompletionError",
                error_message="DX-RT asynchronous completion failed.",
            )

    def _elapsed_ms(self, job):
        return max(
            0.0,
            (time.perf_counter_ns() - job["started_ns"]) / 1_000_000.0,
        )

    def _is_retired_token(self, token):
        return type(token) is int and 0 < token < self._next_token

    def _retire_job_locked(self, token, job):
        if self._jobs.get(token) is job:
            self._jobs.pop(token, None)

    def _copy_output(self, value):
        if isinstance(value, np.ndarray):
            if value.dtype != object:
                return np.array(value, copy=True)
            copied = np.empty(value.shape, dtype=object)
            for index in np.ndindex(value.shape):
                copied[index] = self._copy_output(value[index])
            return copied
        if isinstance(value, list):
            return [self._copy_output(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._copy_output(item) for item in value)
        return value


class DeepXRuntime(Runtime):
    """
    DEEPX DX-RT Python runtime adapter.

    DX-RT exposes its Python API through the dx_engine package. This adapter
    intentionally stays on the blocking run()/run_multi_input() path so the
    benchmark runner measures one synchronous inference call at a time.
    """

    def __init__(self, **runtime_options):
        self.device = str(runtime_options.get("device", "npu0"))
        self.runtime_options = runtime_options
        self.sdk_module = str(runtime_options.get("sdk_module", "dx_engine"))
        self.engine_class = str(runtime_options.get("engine_class", "InferenceEngine"))
        self.input_layout = str(runtime_options.get("input_layout", "auto")).upper()
        self.input_dtype = str(runtime_options.get("input_dtype", "auto")).lower()
        self.batch_mode = str(runtime_options.get("batch_mode", "sdk_batch")).lower()
        self.input_batch_axis = str(runtime_options.get("input_batch_axis", "keep")).lower()
        self.single_input_run_style = str(runtime_options.get("single_input_run_style", "list")).lower()
        self.debug_tensors = self._coerce_bool(runtime_options.get("debug_tensors", False))
        self.bound_option = str(runtime_options.get("bound_option", "NPU_ALL")).upper()
        self._buffer_count_option = runtime_options.get("buffer_count", 6)
        self._async_completion_timeout_option = runtime_options.get(
            "async_completion_timeout_sec",
            30.0,
        )
        self.buffer_count = 6
        self.async_completion_timeout_sec = 30.0
        self.compatible_suffixes = tuple(
            str(item).lower()
            for item in runtime_options.get("compatible_suffixes", (".dxnn",))
        )
        self.device_ids = self._parse_device_ids(runtime_options.get("device_ids"), self.device)

        if self.batch_mode not in ("sdk_batch", "microbatch"):
            raise ValueError("DeepX batch_mode must be 'sdk_batch' or 'microbatch'.")
        if self.input_layout not in ("AUTO", "NCHW", "NHWC"):
            raise ValueError("DeepX input_layout must be 'auto', 'NCHW', or 'NHWC'.")
        if self.input_dtype not in ("auto", "float32", "uint8"):
            raise ValueError("DeepX input_dtype must be 'auto', 'float32', or 'uint8'.")
        if self.input_batch_axis not in ("keep", "squeeze"):
            raise ValueError("DeepX input_batch_axis must be 'keep' or 'squeeze'.")
        if self.single_input_run_style not in ("list", "array"):
            raise ValueError("DeepX single_input_run_style must be 'list' or 'array'.")

        self.compiled_model: CompiledModel | None = None
        self._sdk = None
        self._engine = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._input_infos: list[dict[str, Any]] = []
        self._output_infos: list[dict[str, Any]] = []
        self._execution_mode = None
        self._native_backend: DeepXNativeBackend | None = None

    def load(self, compiled_model: CompiledModel) -> None:
        if not self.is_compatible(compiled_model):
            raise ValueError(f"Incompatible DEEPX artifact: {compiled_model.artifact_path}")

        self.buffer_count = self._parse_buffer_count(
            self._buffer_count_option
        )
        self.async_completion_timeout_sec = self._parse_positive_timeout(
            self._async_completion_timeout_option,
            "async_completion_timeout_sec",
        )
        self.compiled_model = compiled_model
        self._input_names = list(compiled_model.spec.input_shapes.keys())
        self._output_names = list(compiled_model.spec.output_shapes.keys())

        self._sdk = self._import_deepx_sdk()
        engine_cls = self._resolve_attr(self._sdk, self.engine_class)
        option = self._create_inference_option(self._sdk)
        artifact_path = str(Path(compiled_model.artifact_path))

        try:
            self._engine = engine_cls(artifact_path, option)
        except TypeError:
            if option is None:
                self._engine = engine_cls(artifact_path)
            else:
                raise

        self._load_engine_tensor_metadata(compiled_model)

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self._engine is None:
            raise RuntimeError("DeepXRuntime is not loaded. Call load() first.")
        if self._execution_mode == "native_async":
            raise RuntimeError(
                "DeepX synchronous execution is unavailable in native async mode."
            )
        self._execution_mode = "sync"

        ordered_inputs = self._prepare_ordered_inputs(inputs)
        batch_size = self._infer_batch_size(ordered_inputs)

        if self.batch_mode == "microbatch" and batch_size > 1:
            sample_outputs = [
                self._single_output_list(self._run_single_sample(ordered_inputs, sample_idx, batch_size))
                for sample_idx in range(batch_size)
            ]
            return self._merge_batch_outputs(sample_outputs)

        raw_outputs = self._run_sdk(ordered_inputs, batch_size)
        return self._normalize_outputs(raw_outputs)

    def warmup(self, inputs: Dict[str, np.ndarray], num_runs: int = 1) -> None:
        if self._native_backend is not None:
            for _ in range(num_runs):
                self._native_backend.run_warmup_blocking(
                    inputs,
                    timeout=self.async_completion_timeout_sec,
                )
            return
        for _ in range(num_runs):
            self.run(inputs)

    def native_async_max_batch_size(self) -> int:
        return 1

    def create_native_backend(self) -> DeepXNativeBackend:
        if self._engine is None:
            raise RuntimeError("DeepXRuntime is not loaded. Call load() first.")
        if self._execution_mode == "sync":
            raise RuntimeError("DeepX native async is unavailable in sync mode.")
        if self._native_backend is None:
            self._native_backend = DeepXNativeBackend(self)
        self._execution_mode = "native_async"
        return self._native_backend

    def unload(self) -> None:
        if self._native_backend is not None:
            if not self._native_backend.shutdown(
                self.async_completion_timeout_sec
            ):
                raise RuntimeError(
                    "DeepX native backend did not drain before unload."
                )
        if self._engine is not None:
            dispose = getattr(self._engine, "dispose", None)
            if callable(dispose):
                dispose()
        self._engine = None
        self._sdk = None
        self.compiled_model = None
        self._input_names = []
        self._output_names = []
        self._input_infos = []
        self._output_infos = []
        self._execution_mode = None
        self._native_backend = None

    def get_device_spec(self) -> Dict[str, Any]:
        return {
            "backend": "deepx",
            "device": self.device,
            "device_ids": self.device_ids,
            "bound_option": self.bound_option,
            "accelerator_vendor": "DEEPX",
            "accelerator_name": self.runtime_options.get("accelerator_name", "DEEPX NPU"),
            "runtime_version": getattr(self._sdk, "__version__", None) if self._sdk is not None else None,
            "input_names": list(self._input_names),
            "output_names": list(self._output_names),
        }

    def is_compatible(self, compiled_model: CompiledModel) -> bool:
        backend_match = "deepx" in compiled_model.backend_name.lower()
        suffix = str(compiled_model.artifact_path).lower()
        suffix_match = any(suffix.endswith(item) for item in self.compatible_suffixes)
        return backend_match or suffix_match

    def _import_deepx_sdk(self):
        try:
            return import_module(self.sdk_module)
        except ImportError as exc:
            raise ImportError(
                "DEEPX DX-RT Python package is not installed or not importable. "
                f"Tried module '{self.sdk_module}'. Install the dx_engine wheel from "
                "the DEEPX DX-RT SDK, then retry."
            ) from exc

    def _resolve_attr(self, module, dotted_name: str):
        value = module
        for part in dotted_name.split("."):
            value = getattr(value, part)
        return value

    def _create_inference_option(self, sdk_module):
        option_cls = getattr(sdk_module, "InferenceOption", None)
        if option_cls is None:
            return None

        option = option_cls()
        self._set_option_value(option, "devices", "set_devices", self.device_ids)
        self._set_option_value(
            option,
            "bound_option",
            "set_bound_option",
            self._resolve_bound_option(option_cls),
        )

        if "use_ort" in self.runtime_options:
            self._set_option_value(
                option,
                "use_ort",
                "set_use_ort",
                self._coerce_bool(self.runtime_options["use_ort"]),
            )
        if self.buffer_count < 1 or self.buffer_count > 100:
            raise ValueError("DeepX buffer_count must be in the range 1..100.")
        self._set_option_value(
            option,
            "buffer_count",
            "set_buffer_count",
            self.buffer_count,
        )

        return option

    def _resolve_bound_option(self, option_cls):
        enum_scope = getattr(option_cls, "BOUND_OPTION", option_cls)
        if hasattr(enum_scope, self.bound_option):
            return getattr(enum_scope, self.bound_option)

        available = [
            name
            for name in dir(enum_scope)
            if name.startswith("NPU_")
        ]
        raise ValueError(
            f"Unsupported DeepX bound_option: {self.bound_option}. "
            f"Available options: {sorted(available)}"
        )

    def _set_option_value(self, option, attr_name: str, setter_name: str, value) -> None:
        setter = getattr(option, setter_name, None)
        if callable(setter):
            setter(value)
        else:
            setattr(option, attr_name, value)

    def _load_engine_tensor_metadata(self, compiled_model: CompiledModel) -> None:
        input_names = self._call_optional_engine_method("get_input_tensor_names")
        output_names = self._call_optional_engine_method("get_output_tensor_names")
        input_infos = self._call_optional_engine_method("get_input_tensors_info")
        output_infos = self._call_optional_engine_method("get_output_tensors_info")

        if input_names:
            self._input_names = [str(name) for name in input_names]
        if output_names:
            self._output_names = [str(name) for name in output_names]
        if input_infos:
            self._input_infos = list(input_infos)
        if output_infos:
            self._output_infos = list(output_infos)

        if not self._input_names:
            self._input_names = list(compiled_model.spec.input_shapes.keys())
        if not self._output_names:
            self._output_names = list(compiled_model.spec.output_shapes.keys())

        if self.debug_tensors:
            print(f"[DeepXRuntime][debug] input_names={self._input_names}")
            print(f"[DeepXRuntime][debug] output_names={self._output_names}")
            print(f"[DeepXRuntime][debug] input_infos={self._input_infos}")
            print(f"[DeepXRuntime][debug] output_infos={self._output_infos}")

    def _call_optional_engine_method(self, method_name: str):
        method = getattr(self._engine, method_name, None)
        if not callable(method):
            return None
        try:
            return method()
        except Exception:
            return None

    def _prepare_ordered_inputs(self, inputs: Dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
        if not inputs:
            raise ValueError("DeepX runtime received no inputs.")

        if len(self._input_names) == 1:
            input_name = self._input_names[0]
            value = inputs[input_name] if input_name in inputs else next(iter(inputs.values()))
            return [(input_name, self._prepare_array(value))]

        missing = [name for name in self._input_names if name not in inputs]
        if missing:
            raise ValueError(f"Missing required model inputs: {missing}. Provided keys: {list(inputs.keys())}")

        return [
            (name, self._prepare_array(inputs[name]))
            for name in self._input_names
        ]

    def _prepare_array(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value)
        if self.input_layout == "NHWC" and array.ndim == 4 and array.shape[1] in (1, 3):
            array = np.transpose(array, (0, 2, 3, 1))
        array = self._cast_input_array(array)
        return np.ascontiguousarray(array)

    def _cast_input_array(self, array: np.ndarray) -> np.ndarray:
        if self.input_dtype == "auto":
            return array
        if self.input_dtype == "float32":
            return array.astype(np.float32, copy=False)
        if self.input_dtype == "uint8":
            if np.issubdtype(array.dtype, np.integer):
                return np.clip(array, 0, 255).astype(np.uint8, copy=False)
            return np.clip(np.rint(array), 0, 255).astype(np.uint8)
        return array

    def _sdk_input_array(self, array: np.ndarray) -> np.ndarray:
        """Shape the array exactly as it will be handed to DX-RT."""
        if self.input_batch_axis == "squeeze" and array.ndim > 0 and array.shape[0] == 1:
            return np.ascontiguousarray(array[0])
        return array

    def _infer_batch_size(self, ordered_inputs: list[tuple[str, np.ndarray]]) -> int:
        batch_dims = [
            int(array.shape[0])
            for _, array in ordered_inputs
            if array.ndim > 0
        ]
        if not batch_dims:
            return 1
        if len(set(batch_dims)) != 1:
            raise ValueError(
                f"DeepX runtime requires matching leading batch dimensions. "
                f"Got batch dimensions: {batch_dims}"
            )
        return batch_dims[0]

    def _run_sdk(self, ordered_inputs: list[tuple[str, np.ndarray]], batch_size: int):
        if len(ordered_inputs) == 1:
            if batch_size == 1:
                single_input = self._sdk_input_array(ordered_inputs[0][1])
                if self.debug_tensors:
                    self._print_tensor_debug("input", ordered_inputs[0][0], single_input)
                if self.single_input_run_style == "array":
                    return self._engine.run(single_input)
                return self._engine.run([single_input])
            return self._engine.run([
                self._sdk_input_array(
                    self._slice_sample(ordered_inputs[0][1], sample_idx, batch_size)
                )
                for sample_idx in range(batch_size)
            ])

        if batch_size == 1:
            input_dict = {name: self._sdk_input_array(array) for name, array in ordered_inputs}
            if self.debug_tensors:
                for name, array in input_dict.items():
                    self._print_tensor_debug("input", name, array)
            run_multi_input = getattr(self._engine, "run_multi_input", None)
            if callable(run_multi_input):
                return run_multi_input(input_dict)
            return self._engine.run([array for array in input_dict.values()])

        return self._engine.run([
            [
                self._sdk_input_array(self._slice_sample(array, sample_idx, batch_size))
                for _, array in ordered_inputs
            ]
            for sample_idx in range(batch_size)
        ])

    def _run_single_sample(self, ordered_inputs: list[tuple[str, np.ndarray]], sample_idx: int, batch_size: int):
        if len(ordered_inputs) == 1:
            single_input = self._sdk_input_array(
                self._slice_sample(ordered_inputs[0][1], sample_idx, batch_size)
            )
            if self.single_input_run_style == "array":
                return self._engine.run(single_input)
            return self._engine.run([single_input])

        input_dict = {
            name: self._sdk_input_array(self._slice_sample(array, sample_idx, batch_size))
            for name, array in ordered_inputs
        }
        run_multi_input = getattr(self._engine, "run_multi_input", None)
        if callable(run_multi_input):
            return run_multi_input(input_dict)
        return self._engine.run([input_dict[name] for name, _ in ordered_inputs])

    def _slice_sample(self, array: np.ndarray, sample_idx: int, batch_size: int) -> np.ndarray:
        if array.ndim > 0 and array.shape[0] == batch_size:
            return np.ascontiguousarray(array[sample_idx:sample_idx + 1])
        return array

    def _normalize_outputs(self, outputs: Any) -> Dict[str, np.ndarray]:
        if isinstance(outputs, dict):
            return {str(name): np.asarray(value) for name, value in outputs.items()}

        if self._looks_like_batch_outputs(outputs):
            return self._merge_batch_outputs([
                self._single_output_list(sample_outputs)
                for sample_outputs in outputs
            ])

        return self._single_outputs_to_dict(self._single_output_list(outputs))

    def _looks_like_batch_outputs(self, outputs: Any) -> bool:
        if not isinstance(outputs, (list, tuple)) or not outputs:
            return False
        first = outputs[0]
        return isinstance(first, (list, tuple)) and not isinstance(first, np.ndarray)

    def _single_output_list(self, outputs: Any) -> list[np.ndarray]:
        if isinstance(outputs, dict):
            if self._output_names and all(name in outputs for name in self._output_names):
                return [np.asarray(outputs[name]) for name in self._output_names]
            return [np.asarray(value) for value in outputs.values()]

        if isinstance(outputs, np.ndarray):
            return [np.asarray(outputs)]

        if isinstance(outputs, Iterable) and not isinstance(outputs, (str, bytes)):
            return [np.asarray(value) for value in outputs]

        return [np.asarray(outputs)]

    def _single_outputs_to_dict(self, outputs: list[np.ndarray]) -> Dict[str, np.ndarray]:
        mapped = {
            self._output_name(idx): np.asarray(value)
            for idx, value in enumerate(outputs)
        }
        if self.debug_tensors:
            for name, value in mapped.items():
                self._print_tensor_debug("output", name, value)
        return mapped

    def _merge_batch_outputs(self, batch_outputs: list[list[np.ndarray]]) -> Dict[str, np.ndarray]:
        if not batch_outputs:
            return {}

        output_count = max(len(sample_outputs) for sample_outputs in batch_outputs)
        merged: Dict[str, np.ndarray] = {}
        for output_idx in range(output_count):
            chunks = [
                np.asarray(sample_outputs[output_idx])
                for sample_outputs in batch_outputs
                if output_idx < len(sample_outputs)
            ]
            merged[self._output_name(output_idx)] = self._merge_output_chunks(
                chunks,
                self._output_name(output_idx),
            )
        return merged

    def _merge_output_chunks(self, chunks: list[np.ndarray], output_name: str) -> np.ndarray:
        if not chunks:
            return np.array([])

        first = chunks[0]
        if first.ndim == 0:
            return np.stack(chunks, axis=0)

        expected_shape = self._expected_output_shape(output_name)
        has_leading_batch = (
            first.shape[0] == 1
            and expected_shape is not None
            and len(expected_shape) == first.ndim
            and expected_shape[0] == 1
        )
        if has_leading_batch or all(chunk.ndim == first.ndim and chunk.shape[:1] == (1,) for chunk in chunks):
            try:
                return np.concatenate(chunks, axis=0)
            except ValueError:
                pass
        return np.stack(chunks, axis=0)

    def _expected_output_shape(self, output_name: str) -> tuple[int, ...] | None:
        if self.compiled_model is None:
            return None
        shape = self.compiled_model.spec.output_shapes.get(output_name)
        if shape is None:
            return None
        return tuple(shape)

    def _output_name(self, idx: int) -> str:
        if idx < len(self._output_names):
            return self._output_names[idx]
        if self.compiled_model is not None:
            spec_names = list(self.compiled_model.spec.output_shapes.keys())
            if idx < len(spec_names):
                return spec_names[idx]
        return f"output_{idx}"

    def _parse_device_ids(self, value, device: str) -> list[int]:
        if value is None:
            digits = "".join(ch for ch in device if ch.isdigit())
            return [int(digits)] if digits else []
        if isinstance(value, int):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return [int(item) for item in value]
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            return [int(item.strip()) for item in stripped.split(",") if item.strip()]
        raise ValueError(f"Unsupported DeepX device_ids value: {value!r}")

    def _parse_buffer_count(self, value) -> int:
        error_message = (
            "DeepX buffer_count must be an integer in the range 1..100."
        )
        if isinstance(value, bool):
            raise ValueError(error_message)
        if isinstance(value, Integral):
            buffer_count = int(value)
        elif isinstance(value, Real):
            numeric_value = float(value)
            if not math.isfinite(numeric_value) or not numeric_value.is_integer():
                raise ValueError(error_message)
            buffer_count = int(numeric_value)
        elif isinstance(value, str):
            try:
                buffer_count = int(value)
            except ValueError as exc:
                raise ValueError(error_message) from exc
        else:
            raise ValueError(error_message)
        if buffer_count < 1 or buffer_count > 100:
            raise ValueError(error_message)
        return buffer_count

    def _parse_positive_timeout(self, value, name: str) -> float:
        error_message = f"DeepX {name} must be a finite positive number."
        if isinstance(value, bool):
            raise ValueError(error_message)
        try:
            timeout = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(error_message) from exc
        if (
            not math.isfinite(timeout)
            or timeout <= 0
            or timeout > threading.TIMEOUT_MAX
        ):
            raise ValueError(error_message)
        return timeout

    def _coerce_bool(self, value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "1", "yes", "on"):
                return True
            if lowered in ("false", "0", "no", "off"):
                return False
        return bool(value)

    def _print_tensor_debug(self, kind: str, name: str, value: np.ndarray) -> None:
        array = np.asarray(value)
        if array.size == 0:
            print(f"[DeepXRuntime][debug] {kind} {name}: shape={array.shape} dtype={array.dtype} empty")
            return
        print(
            f"[DeepXRuntime][debug] {kind} {name}: "
            f"shape={array.shape} dtype={array.dtype} "
            f"min={float(np.min(array)):.6g} max={float(np.max(array)):.6g} "
            f"mean={float(np.mean(array)):.6g}"
        )
