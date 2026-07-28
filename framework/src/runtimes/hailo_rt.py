from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
from typing import Any, Dict

import numpy as np

from core.compiled_model import CompiledModel
from core.runtime_executor import NativeAsyncOutcome
from .base import Runtime


class HailoRuntime(Runtime):
    """
    HailoRT adapter for executing precompiled HEF artifacts on Hailo devices.

    The Hailo Python binding is imported lazily in load(), so the rest of the
    benchmark framework remains usable on machines without HailoRT installed.
    """

    def __init__(self, **runtime_options):
        self.device = str(runtime_options.get("device", "device0"))
        self.runtime_options = runtime_options
        self.accelerator_name = str(runtime_options.get("accelerator_name", "Hailo"))
        self.interface = str(runtime_options.get("interface", "pcie")).lower()
        self.input_format_type = str(runtime_options.get("input_format_type", "float32")).lower()
        self.output_format_type = str(runtime_options.get("output_format_type", "float32")).lower()
        self.input_layout = str(runtime_options.get("input_layout", "auto")).upper()
        self.batch_size = runtime_options.get("batch_size")
        self.async_ready_timeout_ms = self._positive_timeout_ms(
            runtime_options.get(
                "async_ready_timeout_ms",
                runtime_options.get("async_timeout_ms", 10_000),
            ),
            "async_ready_timeout_ms",
        )
        self.async_completion_timeout_ms = self._positive_timeout_ms(
            runtime_options.get(
                "async_completion_timeout_ms",
                runtime_options.get("async_timeout_ms", 10_000),
            ),
            "async_completion_timeout_ms",
        )
        self.tf_nms_format = self._as_bool(
            runtime_options.get("tf_nms_format", runtime_options.get("hailo_tf_nms_format", False))
        )
        self.device_ids = self._parse_device_ids(runtime_options.get("device_ids"), self.device)

        self.compiled_model: CompiledModel | None = None
        self._hailo = None
        self._hef = None
        self._vdevice_ctx = None
        self._vdevice = None
        self._network_group = None
        self._network_group_params = None
        self._activation_ctx = None
        self._infer_ctx = None
        self._infer_pipeline = None
        self._infer_model = None
        self._configured_infer_model_ctx = None
        self._configured_infer_model = None
        self._input_infos = []
        self._output_infos = []
        self._async_queue_size = 1
        self._async_submission_executor = None
        self._async_completion_executor = None
        self._async_submit_lock = threading.Lock()
        self._async_jobs_lock = threading.Lock()
        self._async_jobs: dict[int, dict[str, Any]] = {}
        self._next_async_job_id = 1
        self._async_pipeline_failed = False
        self._unloading = False

    def load(self, compiled_model: CompiledModel) -> None:
        if not self.is_compatible(compiled_model):
            raise ValueError(f"Incompatible Hailo artifact: {compiled_model.artifact_path}")

        with self._async_jobs_lock:
            if self._async_jobs:
                raise RuntimeError(
                    "Cannot load HailoRuntime while native async jobs are in flight"
                )
            self._unloading = False
            self._async_pipeline_failed = False
            self._next_async_job_id = 1

        self._hailo = self._import_hailo_platform()
        hef_path = str(Path(compiled_model.artifact_path))
        self._hef = self._hailo.HEF(hef_path)

        params = self._hailo.VDevice.create_params()
        self._apply_vdevice_params(params)

        self._vdevice_ctx = self._create_vdevice(params)
        if hasattr(self._vdevice_ctx, "__enter__"):
            self._vdevice = self._vdevice_ctx.__enter__()
        else:
            self._vdevice = self._vdevice_ctx
        self._input_infos = list(self._hef.get_input_vstream_infos())
        self._output_infos = list(self._hef.get_output_vstream_infos())

        if hasattr(self._vdevice, "create_infer_model"):
            self._load_with_infer_model_api(hef_path)
        else:
            self._load_with_vstreams_api()

        self.compiled_model = compiled_model

        input_desc = ", ".join(f"{info.name}:{tuple(info.shape)}" for info in self._input_infos)
        output_desc = ", ".join(f"{info.name}:{tuple(info.shape)}" for info in self._output_infos)
        print(f"[HailoRT] Loaded HEF: {hef_path}")
        print(f"[HailoRT] API: {'InferModel' if self._configured_infer_model is not None else 'InferVStreams'}")
        if self.supports_native_async():
            print(f"[HailoRT] Native async queue size: {self._async_queue_size}")
        print(f"[HailoRT] Inputs: {input_desc}")
        print(f"[HailoRT] Outputs: {output_desc}")

    def _load_with_infer_model_api(self, hef_path: str) -> None:
        self._infer_model = self._vdevice.create_infer_model(hef_path)
        if self.batch_size is not None and hasattr(self._infer_model, "set_batch_size"):
            self._infer_model.set_batch_size(int(self.batch_size))

        self._apply_infer_model_formats()
        self._apply_nms_runtime_options(self._infer_model)

        self._configured_infer_model_ctx = self._infer_model.configure()
        self._configured_infer_model = self._configured_infer_model_ctx.__enter__()
        self._apply_nms_runtime_options(self._configured_infer_model)
        self._async_queue_size = self._read_async_queue_size()
        if self.supports_native_async():
            self._async_submission_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="hailort-submit",
            )
            self._async_completion_executor = ThreadPoolExecutor(
                max_workers=self._async_queue_size,
                thread_name_prefix="hailort-complete",
            )

    def _load_with_vstreams_api(self) -> None:
        configure_params = self._hailo.ConfigureParams.create_from_hef(
            hef=self._hef,
            interface=self._get_stream_interface(),
        )

        if self.batch_size is not None:
            self._set_configured_batch_size(configure_params, int(self.batch_size))

        network_groups = self._vdevice.configure(self._hef, configure_params)
        self._network_group = network_groups[0]
        self._network_group_params = self._network_group.create_params()
        self._input_infos = list(self._hef.get_input_vstream_infos())
        self._output_infos = list(self._hef.get_output_vstream_infos())

        input_params = self._make_vstream_params(
            self._hailo.InputVStreamParams,
            self._network_group,
            self.input_format_type,
        )
        output_params = self._make_vstream_params(
            self._hailo.OutputVStreamParams,
            self._network_group,
            self.output_format_type,
        )

        self._activation_ctx = self._network_group.activate(self._network_group_params)
        if self._activation_ctx is not None and hasattr(self._activation_ctx, "__enter__"):
            self._activation_ctx.__enter__()
        self._infer_ctx = self._create_infer_vstreams(input_params, output_params)
        self._apply_nms_runtime_options(self._infer_ctx)
        self._infer_pipeline = self._infer_ctx.__enter__()

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self._configured_infer_model is None and self._infer_pipeline is None:
            raise RuntimeError("HailoRuntime is not loaded. Call load() first.")

        input_data = self._prepare_inputs(inputs)
        if self._configured_infer_model is not None:
            return self._run_infer_model(input_data)

        results = self._infer_pipeline.infer(input_data)
        return self._normalize_outputs(results)

    def warmup(self, inputs: Dict[str, np.ndarray], num_runs: int = 1) -> None:
        for _ in range(num_runs):
            self.run(inputs)

    def supports_native_async(self) -> bool:
        configured = self._configured_infer_model
        return bool(
            configured is not None
            and callable(getattr(configured, "wait_for_async_ready", None))
            and callable(getattr(configured, "run_async", None))
        )

    def native_async_max_batch_size(self) -> int:
        return self._configured_batch_size()

    def max_concurrent_workers(self) -> int:
        if not self.supports_native_async():
            return 1
        return self._async_queue_size

    def native_async_max_inflight(self) -> int:
        if not self.supports_native_async():
            return 1
        return self._async_queue_size

    def native_async_completion_timeout_sec(self) -> float:
        total_timeout_ms = (
            self._async_queue_size * self.async_ready_timeout_ms
            + self.async_completion_timeout_ms
        )
        return total_timeout_ms / 1000.0

    def create_native_backend(self):
        if not self.supports_native_async():
            raise RuntimeError(
                "Hailo native async inference requires the InferModel API"
            )
        return self

    def supports_dynamic_batching(self) -> bool:
        return self.supports_native_async() and self._configured_batch_size() > 1

    def max_dynamic_batch_size(self):
        return self._configured_batch_size()

    def submit_async(self, inputs, callback):
        """Submit one Hailo InferModel job using the framework native contract.

        The job-local closure and registry keep prepared input, bindings and
        output buffers alive until Hailo invokes its completion callback.
        ``NativeAsyncRuntimeExecutor`` then retains the copied outcome through
        the framework terminal ACK boundary.
        """
        if not self.supports_native_async():
            raise NotImplementedError(
                "Hailo native async inference requires the InferModel API"
            )
        if not callable(callback):
            raise ValueError("Hailo async callback must be callable")

        started_ns = time.perf_counter_ns()
        input_data = self._prepare_inputs(inputs)
        batch_size = self._infer_batch_size(input_data)

        with self._async_jobs_lock:
            if self._unloading:
                raise RuntimeError("HailoRuntime is unloading")
            if self._async_pipeline_failed:
                raise RuntimeError(
                    "HailoRT async pipeline is unavailable after a failed completion"
                )
            if len(self._async_jobs) >= self._async_queue_size:
                raise RuntimeError(
                    "HailoRT native async queue capacity is exhausted"
                )
            submission_executor = self._async_submission_executor
            if submission_executor is None:
                raise RuntimeError("HailoRT async submission worker is unavailable")
            vendor_job_id = self._next_async_job_id
            self._next_async_job_id += 1
            job_record = {
                "bindings": None,
                "input_data": input_data,
                "job": None,
                "submission_future": None,
            }
            self._async_jobs[vendor_job_id] = job_record

        try:
            submission_future = submission_executor.submit(
                self._submit_infer_model_async,
                vendor_job_id,
                callback,
                started_ns,
                batch_size,
            )
        except BaseException:
            with self._async_jobs_lock:
                self._async_jobs.pop(vendor_job_id, None)
            raise
        with self._async_jobs_lock:
            active_record = self._async_jobs.get(vendor_job_id)
            if active_record is job_record:
                active_record["submission_future"] = submission_future
        return vendor_job_id

    def _submit_infer_model_async(
        self,
        vendor_job_id: int,
        callback,
        started_ns: int,
        batch_size: int,
    ) -> None:
        failure_outcome = None
        with self._async_submit_lock:
            with self._async_jobs_lock:
                job_record = self._async_jobs.get(vendor_job_id)
                if job_record is None:
                    return
                if self._unloading or self._async_pipeline_failed:
                    failure_outcome = NativeAsyncOutcome(
                        timing_ms=self._async_elapsed_ms(started_ns),
                        error_type="HailoRTAsyncPipelineClosed",
                        error_message="HailoRT async pipeline is unavailable",
                    )
                input_data = job_record["input_data"]

            if failure_outcome is None:
                try:
                    bindings = [
                        self._create_infer_binding(
                            input_data,
                            sample_idx,
                            batch_size,
                        )
                        for sample_idx in range(batch_size)
                    ]
                    with self._async_jobs_lock:
                        active_record = self._async_jobs.get(vendor_job_id)
                        if active_record is not job_record:
                            return
                        active_record["bindings"] = bindings
                except BaseException as exc:
                    with self._async_jobs_lock:
                        self._async_pipeline_failed = True
                    failure_outcome = NativeAsyncOutcome(
                        timing_ms=self._async_elapsed_ms(started_ns),
                        error_type="HailoRTAsyncBindingError",
                        error_message=self._async_error_message(exc),
                    )

            if failure_outcome is None:
                try:
                    self._configured_infer_model.wait_for_async_ready(
                        timeout_ms=self.async_ready_timeout_ms,
                        frames_count=batch_size,
                    )
                except BaseException as exc:
                    failure_outcome = NativeAsyncOutcome(
                        timing_ms=self._async_elapsed_ms(started_ns),
                        error_type="HailoRTAsyncReadyError",
                        error_message=self._async_error_message(exc),
                    )

            if failure_outcome is None:
                with self._async_jobs_lock:
                    if self._async_pipeline_failed:
                        failure_outcome = NativeAsyncOutcome(
                            timing_ms=self._async_elapsed_ms(started_ns),
                            error_type="HailoRTAsyncPipelineClosed",
                            error_message="HailoRT async pipeline is unavailable",
                        )

            if failure_outcome is None:
                def hailo_completion_callback(completion_info=None, **_kwargs):
                    completion_exception, protocol_error = (
                        self._snapshot_async_completion(completion_info)
                    )
                    self._queue_async_completion(
                        vendor_job_id,
                        callback,
                        completion_exception,
                        protocol_error,
                        bindings,
                        started_ns,
                    )

                try:
                    job = self._configured_infer_model.run_async(
                        bindings,
                        hailo_completion_callback,
                    )
                except BaseException as exc:
                    with self._async_jobs_lock:
                        self._async_pipeline_failed = True
                    failure_outcome = NativeAsyncOutcome(
                        timing_ms=self._async_elapsed_ms(started_ns),
                        error_type="HailoRTAsyncSubmitError",
                        error_message=self._async_error_message(exc),
                    )
                else:
                    with self._async_jobs_lock:
                        active_record = self._async_jobs.get(vendor_job_id)
                        if active_record is job_record:
                            active_record["job"] = job

        if failure_outcome is not None:
            self._finish_async_job(vendor_job_id, callback, failure_outcome)

    def _queue_async_completion(
        self,
        vendor_job_id: int,
        callback,
        completion_exception,
        protocol_error: str | None,
        bindings: list[Any],
        started_ns: int,
    ) -> None:
        with self._async_jobs_lock:
            if vendor_job_id not in self._async_jobs:
                return
            completion_executor = self._async_completion_executor

        if completion_executor is None:
            with self._async_jobs_lock:
                self._async_pipeline_failed = True
            self._finish_async_job(
                vendor_job_id,
                callback,
                NativeAsyncOutcome(
                    timing_ms=self._async_elapsed_ms(started_ns),
                    error_type="HailoRTAsyncCompletionError",
                    error_message="HailoRT async completion worker is unavailable",
                ),
            )
            return

        try:
            completion_future = completion_executor.submit(
                self._process_async_completion,
                vendor_job_id,
                callback,
                completion_exception,
                protocol_error,
                bindings,
                started_ns,
            )
        except BaseException as exc:
            with self._async_jobs_lock:
                self._async_pipeline_failed = True
            self._finish_async_job(
                vendor_job_id,
                callback,
                NativeAsyncOutcome(
                    timing_ms=self._async_elapsed_ms(started_ns),
                    error_type="HailoRTAsyncCompletionError",
                    error_message=self._async_error_message(exc),
                ),
            )
            return

        with self._async_jobs_lock:
            job_record = self._async_jobs.get(vendor_job_id)
            if job_record is not None:
                job_record["completion_future"] = completion_future

    def _process_async_completion(
        self,
        vendor_job_id: int,
        callback,
        completion_exception,
        protocol_error: str | None,
        bindings: list[Any],
        started_ns: int,
    ) -> None:
        outcome = self._async_completion_outcome(
            completion_exception,
            protocol_error,
            bindings,
            started_ns,
        )
        self._finish_async_job(vendor_job_id, callback, outcome)

    def _finish_async_job(
        self,
        vendor_job_id: int,
        callback,
        outcome: NativeAsyncOutcome,
    ) -> None:
        with self._async_jobs_lock:
            job_record = self._async_jobs.pop(vendor_job_id, None)
        if job_record is None:
            return
        callback(outcome)

    def unload(self) -> None:
        with self._async_submit_lock:
            with self._async_jobs_lock:
                async_jobs_inflight = len(self._async_jobs)
                if async_jobs_inflight:
                    raise RuntimeError(
                        "Cannot unload HailoRuntime while native async jobs are in flight"
                    )
                self._unloading = True
            if self._async_submission_executor is not None:
                self._async_submission_executor.shutdown(wait=True)
                self._async_submission_executor = None
            if self._async_completion_executor is not None:
                self._async_completion_executor.shutdown(wait=True)
                self._async_completion_executor = None
            if self._configured_infer_model_ctx is not None:
                self._configured_infer_model_ctx.__exit__(None, None, None)
                self._configured_infer_model_ctx = None
            self._configured_infer_model = None
            self._infer_model = None
            if self._infer_ctx is not None:
                self._infer_ctx.__exit__(None, None, None)
                self._infer_ctx = None
                self._infer_pipeline = None
            if self._activation_ctx is not None and hasattr(self._activation_ctx, "__exit__"):
                self._activation_ctx.__exit__(None, None, None)
            self._activation_ctx = None
            if self._vdevice_ctx is not None and hasattr(self._vdevice_ctx, "__exit__"):
                self._vdevice_ctx.__exit__(None, None, None)
            self._vdevice_ctx = None
            self._vdevice = None
            self._network_group = None
            self._network_group_params = None
            self._hef = None
            self.compiled_model = None
            self._async_queue_size = 1
            self._async_pipeline_failed = False

    def get_device_spec(self) -> Dict[str, Any]:
        return {
            "backend": "hailort",
            "device": self.device,
            "device_ids": self.device_ids,
            "accelerator_vendor": "Hailo",
            "accelerator_name": self.accelerator_name,
            "native_async": self.supports_native_async(),
            "native_async_queue_size": (
                self._async_queue_size if self.supports_native_async() else 0
            ),
            "async_ready_timeout_ms": self.async_ready_timeout_ms,
            "async_completion_timeout_ms": self.async_completion_timeout_ms,
            "runtime_options": self.runtime_options,
        }

    def is_compatible(self, compiled_model: CompiledModel) -> bool:
        return str(compiled_model.artifact_path).lower().endswith(".hef")

    def _import_hailo_platform(self):
        try:
            import hailo_platform
        except ImportError as exc:
            raise ImportError(
                "HailoRT Python package is not installed. Install the matching "
                "hailort wheel for this Jetson Python environment, then retry."
            ) from exc
        return hailo_platform

    def _get_stream_interface(self):
        interfaces = self._hailo.HailoStreamInterface
        if self.interface in ("pcie", "pci", "m2", "m.2"):
            return interfaces.PCIe
        if self.interface == "ethernet" and hasattr(interfaces, "ETH"):
            return interfaces.ETH
        if self.interface == "integrated" and hasattr(interfaces, "INTEGRATED"):
            return interfaces.INTEGRATED
        return interfaces.PCIe

    def _apply_vdevice_params(self, params) -> None:
        if hasattr(params, "group_id") and self.runtime_options.get("group_id"):
            params.group_id = self.runtime_options["group_id"]
        if hasattr(params, "multi_process_service") and self.runtime_options.get("multi_process_service") is not None:
            params.multi_process_service = bool(self.runtime_options["multi_process_service"])
            if params.multi_process_service and hasattr(params, "group_id") and not getattr(params, "group_id", None):
                params.group_id = "SHARED"

        scheduling = self.runtime_options.get("scheduling_algorithm", "ROUND_ROBIN")
        algorithms = getattr(self._hailo, "HailoSchedulingAlgorithm", None)
        if hasattr(params, "scheduling_algorithm") and scheduling and algorithms is not None:
            try:
                params.scheduling_algorithm = getattr(algorithms, str(scheduling).upper())
            except AttributeError:
                print(f"[HailoRT] Unknown scheduling_algorithm ignored: {scheduling}")

    def _create_vdevice(self, params):
        uses_custom_params = (
            self.runtime_options.get("group_id") is not None
            or self.runtime_options.get("multi_process_service") is not None
        )
        if self.device_ids:
            if uses_custom_params:
                raise ValueError(
                    "HailoRT device_ids cannot be combined with group_id or "
                    "multi_process_service because VDevice accepts either params "
                    "or device_ids."
                )
            return self._hailo.VDevice(device_ids=self.device_ids)
        return self._hailo.VDevice(params)

    def _apply_infer_model_formats(self) -> None:
        input_format = self._get_format_type(self.input_format_type)
        if input_format is not None:
            for info in self._input_infos:
                self._set_infer_tensor_format(self._infer_model.input, info.name, input_format)

        output_format = self._get_format_type(self.output_format_type)
        if output_format is not None:
            for info in self._output_infos:
                self._set_infer_tensor_format(self._infer_model.output, info.name, output_format)

    def _set_infer_tensor_format(self, accessor, name: str, format_type) -> None:
        try:
            tensor = accessor(name) if name else accessor()
        except TypeError:
            tensor = accessor()

        if self._is_byte_mask_nms_tensor(tensor):
            return

        try:
            tensor.set_format_type(format_type)
        except Exception as exc:
            raise RuntimeError(f"Could not set Hailo tensor format for '{name}': {exc}") from exc

    def _is_byte_mask_nms_tensor(self, tensor) -> bool:
        order = getattr(getattr(tensor, "format", None), "order", None)
        return "HAILO_NMS_WITH_BYTE_MASK" in str(order)

    def _create_infer_vstreams(self, input_params, output_params):
        try:
            return self._hailo.InferVStreams(
                self._network_group,
                input_params,
                output_params,
                tf_nms_format=self.tf_nms_format,
            )
        except TypeError:
            if self.tf_nms_format:
                raise
            return self._hailo.InferVStreams(self._network_group, input_params, output_params)

    def _apply_nms_runtime_options(self, infer_vstreams) -> None:
        option_specs = [
            (
                "hailo_nms_score_threshold",
                "set_nms_score_threshold",
                float,
                "hailo_nms_conf_threshold",
            ),
            ("hailo_nms_iou_threshold", "set_nms_iou_threshold", float, None),
            (
                "hailo_nms_max_proposals_per_class",
                "set_nms_max_proposals_per_class",
                int,
                None,
            ),
            (
                "hailo_nms_max_accumulated_mask_size",
                "set_nms_max_accumulated_mask_size",
                int,
                None,
            ),
        ]
        for key, method_name, coerce, fallback_key in option_specs:
            value = self.runtime_options.get(key)
            if value is None and fallback_key:
                value = self.runtime_options.get(fallback_key)
            if value is None:
                continue

            method = getattr(infer_vstreams, method_name, None)
            if method is None:
                print(f"[HailoRT] NMS runtime option ignored: {method_name} is unavailable")
                continue
            try:
                method(coerce(value))
            except Exception as exc:
                print(f"[HailoRT] NMS runtime option {key} could not be applied: {exc}")

    def _run_infer_model(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        batch_size = self._infer_batch_size(input_data)
        bindings = [
            self._create_infer_binding(input_data, sample_idx, batch_size)
            for sample_idx in range(batch_size)
        ]
        timeout_ms = int(self.runtime_options.get("timeout_ms", 1000))
        self._configured_infer_model.run(bindings, timeout_ms)
        return self._collect_binding_outputs(bindings)

    def _read_async_queue_size(self) -> int:
        get_queue_size = getattr(
            self._configured_infer_model,
            "get_async_queue_size",
            None,
        )
        if not callable(get_queue_size):
            return 1
        try:
            queue_size = int(get_queue_size())
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                "HailoRT returned an invalid async queue size"
            ) from exc
        if queue_size <= 0:
            raise RuntimeError("HailoRT async queue size must be positive")
        return queue_size

    def _configured_batch_size(self) -> int:
        if self.batch_size is None:
            return 1
        try:
            batch_size = int(self.batch_size)
        except (TypeError, ValueError, OverflowError):
            return 1
        return max(1, batch_size)

    def _async_completion_outcome(
        self,
        completion_exception,
        protocol_error: str | None,
        bindings: list[Any],
        started_ns: int,
    ) -> NativeAsyncOutcome:
        elapsed_ms = self._async_elapsed_ms(started_ns)
        if protocol_error is not None:
            with self._async_jobs_lock:
                self._async_pipeline_failed = True
            return NativeAsyncOutcome(
                timing_ms=elapsed_ms,
                error_type="HailoRTAsyncProtocolError",
                error_message=protocol_error,
            )

        if completion_exception is not None:
            with self._async_jobs_lock:
                self._async_pipeline_failed = True
            return NativeAsyncOutcome(
                timing_ms=elapsed_ms,
                error_type="HailoRTAsyncError",
                error_message=self._async_error_message(completion_exception),
            )

        try:
            outputs = {
                name: self._copy_async_output(value)
                for name, value in self._collect_binding_outputs(bindings).items()
            }
        except BaseException as exc:
            with self._async_jobs_lock:
                self._async_pipeline_failed = True
            return NativeAsyncOutcome(
                timing_ms=elapsed_ms,
                error_type="HailoRTAsyncCompletionError",
                error_message=self._async_error_message(exc),
            )
        return NativeAsyncOutcome(outputs=outputs, timing_ms=elapsed_ms)

    def _snapshot_async_completion(self, completion_info):
        if completion_info is None:
            return (
                None,
                "HailoRT callback did not provide completion_info",
            )
        try:
            return completion_info.exception, None
        except BaseException:
            return (
                None,
                "HailoRT completion_info.exception is unavailable",
            )

    def _async_elapsed_ms(self, started_ns: int) -> float:
        return max(0.0, (time.perf_counter_ns() - started_ns) / 1_000_000.0)

    def _copy_async_output(self, value):
        if isinstance(value, np.ndarray):
            if value.dtype != object:
                return np.array(value, copy=True)
            copied = np.empty(value.shape, dtype=object)
            for index in np.ndindex(value.shape):
                copied[index] = self._copy_async_output(value[index])
            return copied
        if isinstance(value, list):
            return [self._copy_async_output(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._copy_async_output(item) for item in value)
        return value

    def _async_error_message(self, exception) -> str:
        try:
            error_type = type(exception).__name__
        except BaseException:
            error_type = "HailoRTException"
        try:
            message = str(exception)
        except BaseException:
            message = "HailoRT asynchronous inference failed"
        return " ".join(f"{error_type}: {message}".split())[:512]

    def _create_infer_binding(
        self,
        input_data: Dict[str, np.ndarray],
        sample_idx: int,
        batch_size: int,
    ):
        output_buffers = {
            info.name: np.empty(
                self._infer_output_shape(info.name, tuple(info.shape)),
                dtype=self._infer_output_dtype(info.name, self.output_format_type),
            )
            for info in self._output_infos
        }
        binding = self._configured_infer_model.create_bindings(output_buffers=output_buffers)

        for info in self._input_infos:
            sample = self._slice_batch_value(input_data[info.name], sample_idx, batch_size)
            self._binding_input(binding, info.name).set_buffer(np.ascontiguousarray(sample))
        return binding

    def _collect_binding_outputs(self, bindings: list[Any]) -> Dict[str, np.ndarray]:
        collected: Dict[str, list[Any]] = {info.name: [] for info in self._output_infos}
        for binding in bindings:
            for info in self._output_infos:
                collected[info.name].append(self._binding_output(binding, info.name).get_buffer())

        return {
            name: self._stack_or_object_array(values)
            for name, values in collected.items()
        }

    def _binding_input(self, binding, name: str):
        if len(self._input_infos) == 1:
            try:
                return binding.input()
            except TypeError:
                pass
        return binding.input(name)

    def _binding_output(self, binding, name: str):
        if len(self._output_infos) == 1:
            try:
                return binding.output()
            except TypeError:
                pass
        return binding.output(name)

    def _infer_batch_size(self, input_data: Dict[str, np.ndarray]) -> int:
        first = next(iter(input_data.values()))
        return int(first.shape[0]) if getattr(first, "ndim", 0) > 0 else 1

    def _slice_batch_value(self, value: np.ndarray, sample_idx: int, batch_size: int) -> np.ndarray:
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[sample_idx]
        return value

    def _infer_output_shape(self, name: str, fallback_shape: tuple[int, ...]) -> tuple[int, ...]:
        try:
            shape = self._infer_model.output(name).shape
            return tuple(shape)
        except Exception:
            return fallback_shape

    def _infer_output_dtype(self, name: str, format_type_name: str) -> np.dtype:
        normalized = str(format_type_name or "auto").lower()
        if normalized == "auto":
            normalized = self._infer_output_format_from_metadata(name)
        if normalized == "int8":
            normalized = "uint8"
        dtype_map = {
            "float32": np.float32,
            "uint8": np.uint8,
            "uint16": np.uint16,
        }
        try:
            return np.dtype(dtype_map[normalized])
        except KeyError as exc:
            raise ValueError(f"Unsupported Hailo output format_type: {format_type_name}") from exc

    def _infer_output_format_from_metadata(self, name: str) -> str:
        try:
            tensor = self._infer_model.output(name)
            format_type = getattr(getattr(tensor, "format", None), "type", None)
            if format_type is not None:
                return str(format_type).split(".")[-1].lower()
        except Exception:
            pass

        for info in self._output_infos:
            if info.name == name:
                format_type = getattr(getattr(info, "format", None), "type", None)
                if format_type is not None:
                    return str(format_type).split(".")[-1].lower()
        return "float32"

    def _stack_or_object_array(self, values: list[Any]) -> np.ndarray:
        arrays = [self._as_numpy(value) for value in values]
        try:
            return np.stack(arrays, axis=0)
        except ValueError:
            return np.asarray(arrays, dtype=object)

    def _get_format_type(self, name: str):
        fmt = name.upper()
        if fmt == "INT8":
            fmt = "UINT8"
        if fmt == "AUTO":
            return None
        try:
            return getattr(self._hailo.FormatType, fmt)
        except AttributeError as exc:
            raise ValueError(f"Unsupported HailoRT format_type: {name}") from exc

    def _as_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _positive_timeout_ms(self, value: Any, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive integer")
        try:
            timeout_ms = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if timeout_ms <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return timeout_ms

    def _make_vstream_params(self, params_cls, network_group, format_type_name: str):
        format_type = self._get_format_type(format_type_name)
        factories = []
        if hasattr(params_cls, "make"):
            factories.append(params_cls.make)
        if hasattr(params_cls, "make_from_network_group"):
            factories.append(params_cls.make_from_network_group)

        last_error = None
        for factory in factories:
            kwargs = {"quantized": False}
            if format_type is not None:
                kwargs["format_type"] = format_type
            try:
                return factory(network_group, **kwargs)
            except TypeError as exc:
                last_error = exc
                kwargs.pop("quantized", None)
                try:
                    return factory(network_group, **kwargs)
                except TypeError as retry_exc:
                    last_error = retry_exc
                    try:
                        return factory(network_group)
                    except TypeError as final_exc:
                        last_error = final_exc
        raise RuntimeError(f"Could not create Hailo vstream params: {last_error}")

    def _set_configured_batch_size(self, configure_params, batch_size: int) -> None:
        if hasattr(self._hef, "get_network_group_names"):
            for name in self._hef.get_network_group_names():
                try:
                    configure_params[name].batch_size = batch_size
                except Exception:
                    pass

    def _parse_device_ids(self, value, device: str) -> list[str]:
        default_aliases = {"", "auto", "cpu", "default", "device0"}
        if value is None:
            selected = str(device).strip()
            return [] if selected.lower() in default_aliases else [selected]
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.lower() in default_aliases:
                return []
            return [item.strip() for item in stripped.split(",") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        raise ValueError(f"Unsupported HailoRT device_ids value: {value!r}")

    def _prepare_inputs(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self._input_infos:
            raise RuntimeError("HEF does not expose input vstream information.")

        if len(self._input_infos) == 1:
            value = self._select_single_input(inputs)
            info = self._input_infos[0]
            return {info.name: self._prepare_input_array(value, tuple(info.shape))}

        prepared = {}
        remaining_values = iter(inputs.values())
        for info in self._input_infos:
            value = inputs.get(info.name)
            if value is None:
                value = next(remaining_values)
            prepared[info.name] = self._prepare_input_array(value, tuple(info.shape))
        return prepared

    def _select_single_input(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        if len(inputs) != 1 and self._input_infos[0].name in inputs:
            return inputs[self._input_infos[0].name]
        return next(iter(inputs.values()))

    def _prepare_input_array(self, value: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim == len(expected_shape):
            array = np.expand_dims(array, axis=0)

        if len(expected_shape) == 3:
            array = self._ensure_nhwc(array, expected_shape)

        array = self._cast_input(array)
        return np.ascontiguousarray(array)

    def _ensure_nhwc(self, array: np.ndarray, expected_hwc: tuple[int, int, int]) -> np.ndarray:
        if array.ndim != 4:
            return array
        if tuple(array.shape[1:]) == expected_hwc:
            return array

        expected_chw = (expected_hwc[2], expected_hwc[0], expected_hwc[1])
        if tuple(array.shape[1:]) == expected_chw:
            return np.transpose(array, (0, 2, 3, 1))

        if self.input_layout == "NCHW" and array.shape[1] in (1, 3):
            return np.transpose(array, (0, 2, 3, 1))
        return array

    def _cast_input(self, array: np.ndarray) -> np.ndarray:
        if self.input_format_type == "uint8":
            return np.clip(array, 0, 255).astype(np.uint8)
        if self.input_format_type == "uint16":
            return np.clip(array, 0, 65535).astype(np.uint16)
        return array.astype(np.float32, copy=False)

    def _normalize_outputs(self, results: Dict[str, Any]) -> Dict[str, np.ndarray]:
        normalized: Dict[str, np.ndarray] = {}
        for name, value in results.items():
            normalized[name] = self._as_numpy(value)
        return normalized

    def _as_numpy(self, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
            try:
                return np.asarray(value)
            except ValueError:
                return np.asarray(list(value), dtype=object)
        return np.asarray(value)
