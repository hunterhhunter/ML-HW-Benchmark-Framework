"""Rebellions runtime for precompiled static RBLN artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from importlib import metadata as importlib_metadata
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np

from core.compiled_model import CompiledModel
from .base import Runtime


_BACKEND_NAMES = frozenset({"rbln", "rebel", "rbln-static"})
_EXPECTED_NPU = "RBLN-CA22"
_MISSING = object()


def _require_builtin_int(value: Any, name: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a built-in integer >= {minimum}.")
    return value


def _require_positive_finite_number(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a positive finite number.")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a positive finite number.") from exc
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return normalized


def _metadata_field(value: Any, name: str) -> Any:
    try:
        if isinstance(value, Mapping):
            return value.get(name, _MISSING)
        return getattr(value, name, _MISSING)
    except Exception as exc:
        raise ValueError(
            f"RBLN artifact metadata field '{name}' could not be read."
        ) from exc


def _bounded_string(value: Any, name: str, *, maximum: int = 256) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        raise ValueError(
            f"RBLN artifact metadata field '{name}' must be a bounded string."
        )
    return value


@dataclass(frozen=True)
class _TensorDescriptor:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype


class RblnRuntime(Runtime):
    """Inspect and execute one precompiled RBLN-CA22 artifact."""

    def __init__(self, **runtime_options):
        self.device = str(runtime_options.get("device", "0"))
        self.device_id = _require_builtin_int(
            runtime_options.get("device_id", 0), "device_id", minimum=0
        )
        self.async_parallel = _require_builtin_int(
            runtime_options.get("async_parallel", 1),
            "async_parallel",
            minimum=1,
        )
        self.runtime_timeout_sec = _require_positive_finite_number(
            runtime_options.get("runtime_timeout_sec", 60),
            "runtime_timeout_sec",
        )
        self.shutdown_timeout_sec = _require_positive_finite_number(
            runtime_options.get("shutdown_timeout_sec", 300.0),
            "shutdown_timeout_sec",
        )
        self.max_async_inflight = _require_builtin_int(
            runtime_options.get("max_async_inflight", 1),
            "max_async_inflight",
            minimum=1,
        )
        if self.device_id != 0:
            raise ValueError("device_id must be exactly 0 for RBLN-CA22.")
        if self.async_parallel not in (1, 2):
            raise ValueError("async_parallel must be exactly 1 or 2.")

        self.compiled_model: CompiledModel | None = None
        self._rebel = None
        self._sync_runtime = None
        self._native_backend = None
        self._execution_mode: str | None = None
        self._cleanup_pending = False
        self._detected_npu: str | None = None
        self._sdk_version: str | None = None
        self._artifact_metadata: dict[str, Any] = {}
        self._input_descriptors: tuple[_TensorDescriptor, ...] = ()
        self._input_bindings: tuple[str, ...] = ()
        self._output_descriptors: tuple[_TensorDescriptor, ...] = ()
        self._inspection_context: tuple[CompiledModel, str] | None = None
        self._pending_contract: dict[str, Any] | None = None

    @staticmethod
    def _load_rebel():
        try:
            return import_module("rebel")
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "RBLN static inference requires the optional "
                "rebel-compiler package. Install the vendor SDK and retry."
            ) from exc

    @staticmethod
    def _inspect_compiled_model(rebel, path):
        try:
            compiled_model_api = rebel.RBLNCompiledModel
            inspect = compiled_model_api.inspect
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                "The installed rebel-compiler package does not expose "
                "RBLNCompiledModel.inspect()."
            ) from exc
        try:
            return inspect(str(path))
        except Exception as exc:
            raise RuntimeError("Could not inspect RBLN artifact metadata.") from exc

    @staticmethod
    def _normalize_shape(raw_shape) -> tuple[int, ...]:
        if isinstance(raw_shape, (str, bytes)):
            raise ValueError(
                "RBLN tensor shapes require static positive integer dimensions."
            )
        try:
            dimensions = tuple(raw_shape)
        except Exception as exc:
            raise ValueError(
                "RBLN tensor shapes require static positive integer dimensions."
            ) from exc
        normalized = []
        for dimension in dimensions:
            if isinstance(dimension, (bool, np.bool_)) or not isinstance(
                dimension, (int, np.integer)
            ):
                raise ValueError(
                    "RBLN tensor shapes require static positive integer dimensions."
                )
            integer = int(dimension)
            if integer < 1:
                raise ValueError(
                    "RBLN tensor shapes require static positive integer dimensions."
                )
            normalized.append(integer)
        return tuple(normalized)

    @staticmethod
    def _normalize_dtype(raw_dtype) -> np.dtype:
        if isinstance(raw_dtype, np.dtype):
            return raw_dtype
        if type(raw_dtype) is not str or not raw_dtype or len(raw_dtype) > 64:
            raise ValueError("RBLN tensor dtype must be a valid bounded NumPy dtype.")
        try:
            return np.dtype(raw_dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "RBLN tensor dtype must be a valid bounded NumPy dtype."
            ) from exc

    @classmethod
    def _normalize_descriptors(
        cls, raw_descriptors: Any, kind: str
    ) -> tuple[_TensorDescriptor, ...]:
        if raw_descriptors is _MISSING or raw_descriptors is None:
            raise ValueError(
                f"RBLN artifact is missing required {kind} descriptors."
            )
        if isinstance(raw_descriptors, (str, bytes, Mapping)):
            raise ValueError(
                f"RBLN artifact {kind} descriptors must be a sequence."
            )
        try:
            raw_items = tuple(raw_descriptors)
        except Exception as exc:
            raise ValueError(
                f"RBLN artifact {kind} descriptors must be a sequence."
            ) from exc
        if not raw_items:
            raise ValueError(
                f"RBLN artifact must declare at least one {kind} descriptor."
            )

        descriptors = []
        names = set()
        for raw_descriptor in raw_items:
            raw_name = _metadata_field(raw_descriptor, "name")
            if raw_name is _MISSING or raw_name is None or raw_name == "":
                raise ValueError(
                    f"RBLN artifact has a missing {kind} descriptor name."
                )
            name = _bounded_string(raw_name, f"{kind} descriptor name")
            if name in names:
                raise ValueError(
                    f"RBLN artifact has a duplicate {kind} descriptor name."
                )
            names.add(name)
            shape = cls._normalize_shape(_metadata_field(raw_descriptor, "shape"))
            dtype = cls._normalize_dtype(_metadata_field(raw_descriptor, "dtype"))
            descriptors.append(_TensorDescriptor(name, shape, dtype))
        return tuple(descriptors)

    @staticmethod
    def _optional_metadata_string(
        inspected: Any, field: str
    ) -> str | None:
        value = _metadata_field(inspected, field)
        if value is _MISSING or value is None:
            return None
        return _bounded_string(value, field)

    @staticmethod
    def _optional_allocation(inspected: Any) -> list[int] | None:
        value = _metadata_field(inspected, "alloc_per_node")
        if value is _MISSING or value is None:
            return None
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                "RBLN artifact alloc_per_node must be a list of integers."
            )
        normalized = []
        for allocation in value:
            if type(allocation) is not int or allocation < 0:
                raise ValueError(
                    "RBLN artifact alloc_per_node must be a list of integers."
                )
            normalized.append(allocation)
        return normalized

    def _inspect_contract(self, inspected) -> None:
        if self._inspection_context is None:
            raise RuntimeError("RBLN inspection context is unavailable.")
        compiled_model, detected_npu = self._inspection_context

        artifact_npu = _metadata_field(inspected, "npu")
        if artifact_npu is _MISSING or artifact_npu is None:
            raise ValueError("RBLN artifact target NPU metadata is required.")
        artifact_npu = _bounded_string(artifact_npu, "npu")
        if artifact_npu != detected_npu:
            raise ValueError(
                "RBLN artifact target NPU does not match the detected NPU."
            )

        tensor_parallel_size = _metadata_field(
            inspected, "tensor_parallel_size"
        )
        if type(tensor_parallel_size) is not int or tensor_parallel_size != 1:
            raise ValueError(
                "RBLN artifact tensor_parallel_size must be exactly 1."
            )

        input_descriptors = self._normalize_descriptors(
            _metadata_field(inspected, "inputs"), "input"
        )
        output_descriptors = self._normalize_descriptors(
            _metadata_field(inspected, "outputs"), "output"
        )

        spec = compiled_model.spec
        spec_input_names = tuple(spec.input_shapes)
        artifact_input_names = tuple(item.name for item in input_descriptors)
        if len(input_descriptors) != len(spec_input_names):
            raise ValueError(
                "RBLN artifact input descriptor count does not match Model_Spec."
            )
        if len(input_descriptors) == 1:
            input_bindings = spec_input_names
        else:
            if set(artifact_input_names) != set(spec_input_names):
                raise ValueError(
                    "RBLN artifact input descriptor names do not match "
                    "Model_Spec; positional guessing is disabled."
                )
            input_bindings = artifact_input_names

        for descriptor, binding in zip(input_descriptors, input_bindings):
            spec_shape = self._normalize_shape(spec.input_shapes[binding])
            if descriptor.shape != spec_shape:
                raise ValueError(
                    f"RBLN artifact input shape for '{binding}' does not match "
                    "Model_Spec."
                )
            if binding not in spec.input_dtype:
                raise ValueError(
                    f"Model_Spec is missing input dtype for '{binding}'."
                )
            spec_dtype = self._normalize_dtype(spec.input_dtype[binding])
            if descriptor.dtype != spec_dtype:
                raise ValueError(
                    f"RBLN artifact input dtype for '{binding}' does not match "
                    "Model_Spec."
                )

        spec_output_names = tuple(spec.output_shapes)
        artifact_output_names = tuple(item.name for item in output_descriptors)
        if set(artifact_output_names) != set(spec_output_names):
            raise ValueError(
                "RBLN artifact output descriptor names do not match Model_Spec."
            )
        for descriptor in output_descriptors:
            spec_shape = self._normalize_shape(spec.output_shapes[descriptor.name])
            if descriptor.shape != spec_shape:
                raise ValueError(
                    f"RBLN artifact output shape for '{descriptor.name}' does "
                    "not match Model_Spec."
                )

        self._pending_contract = {
            "artifact_metadata": {
                "compiler_version": self._optional_metadata_string(
                    inspected, "compiler_version"
                ),
                "npu": artifact_npu,
                "tensor_parallel_size": tensor_parallel_size,
                "uuid": self._optional_metadata_string(inspected, "uuid"),
                "alloc_per_node": self._optional_allocation(inspected),
            },
            "input_descriptors": input_descriptors,
            "input_bindings": input_bindings,
            "output_descriptors": output_descriptors,
        }

    @staticmethod
    def _sdk_package_version(rebel) -> str | None:
        try:
            version = importlib_metadata.version("rebel-compiler")
        except importlib_metadata.PackageNotFoundError:
            try:
                version = getattr(rebel, "__version__", None)
            except Exception:
                return None
        except Exception:
            return None
        if type(version) is not str or not version or len(version) > 128:
            return None
        return version

    def load(self, compiled_model: CompiledModel) -> None:
        if self._cleanup_pending:
            raise RuntimeError(
                "RBLN cleanup is incomplete; call unload() to retry."
            )
        if self.compiled_model is not None:
            raise RuntimeError("An RBLN artifact is already loaded.")
        if not self.is_compatible(compiled_model):
            raise ValueError(
                "RblnRuntime requires a .rbln artifact with backend "
                "'rbln', 'rebel', or 'rbln-static'."
            )
        artifact_path = Path(compiled_model.artifact_path)
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"RBLN artifact file does not exist: {artifact_path}"
            )

        rebel = self._load_rebel()
        try:
            available = rebel.npu_is_available(self.device_id)
        except Exception as exc:
            raise RuntimeError("RBLN device availability check failed.") from exc
        if available is not True:
            raise RuntimeError("RBLN device 0 is not available.")

        try:
            detected_npu = rebel.get_npu_name(self.device_id)
        except Exception as exc:
            raise RuntimeError("RBLN device name query failed.") from exc
        if detected_npu != _EXPECTED_NPU:
            raise RuntimeError(
                "RBLN static runtime requires detected NPU RBLN-CA22."
            )

        inspected = self._inspect_compiled_model(rebel, artifact_path)
        self._inspection_context = (compiled_model, detected_npu)
        self._pending_contract = None
        try:
            self._inspect_contract(inspected)
            contract = self._pending_contract
            if contract is None:
                raise RuntimeError("RBLN artifact inspection produced no contract.")
        finally:
            self._inspection_context = None
            self._pending_contract = None

        sdk_version = self._sdk_package_version(rebel)
        self.compiled_model = compiled_model
        self._rebel = rebel
        self._detected_npu = detected_npu
        self._sdk_version = sdk_version
        self._artifact_metadata = contract["artifact_metadata"]
        self._input_descriptors = contract["input_descriptors"]
        self._input_bindings = contract["input_bindings"]
        self._output_descriptors = contract["output_descriptors"]
        self._execution_mode = None

    def _require_loaded(self) -> None:
        if self._cleanup_pending:
            raise RuntimeError(
                "RBLN cleanup is incomplete; call unload() to retry."
            )
        if self.compiled_model is None or self._rebel is None:
            raise RuntimeError("RBLN artifact is not loaded. Call load() first.")

    def _reject_async_ownership(self) -> None:
        if self._execution_mode == "native_async" or self._native_backend is not None:
            raise RuntimeError(
                "RBLN synchronous execution is unavailable in native async mode."
            )

    def _ordered_inputs(
        self, inputs: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        self._require_loaded()
        self._reject_async_ownership()
        if not isinstance(inputs, dict):
            raise TypeError("RBLN inputs must be a dictionary of NumPy arrays.")

        expected_names = set(self._input_bindings)
        provided_names = set(inputs)
        missing = [
            name for name in self._input_bindings if name not in provided_names
        ]
        if missing:
            raise ValueError(
                "RBLN missing required inputs: " + ", ".join(missing)
            )
        if provided_names != expected_names:
            raise ValueError("RBLN received unexpected inputs.")

        ordered = []
        for descriptor, binding in zip(
            self._input_descriptors, self._input_bindings
        ):
            value = inputs[binding]
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"RBLN input '{binding}' must be a NumPy array."
                )
            if value.ndim == 0:
                raise ValueError(
                    f"RBLN scalar input '{binding}' is unsupported."
                )
            if value.shape[0] != 1:
                raise ValueError(
                    "RBLN synchronous execution requires batch dimension N=1."
                )
            if value.dtype != descriptor.dtype:
                raise ValueError(
                    f"RBLN input dtype for '{binding}' does not match the "
                    "artifact contract."
                )
            if value.shape != descriptor.shape:
                raise ValueError(
                    f"RBLN input shape for '{binding}' does not match the "
                    "artifact contract."
                )
            ordered.append(np.ascontiguousarray(value))
        return ordered

    def _normalize_outputs(
        self, raw_outputs
    ) -> dict[str, np.ndarray]:
        self._require_loaded()
        expected_names = tuple(
            descriptor.name for descriptor in self._output_descriptors
        )
        if isinstance(raw_outputs, np.ndarray):
            values = (raw_outputs,)
        elif isinstance(raw_outputs, dict):
            if set(raw_outputs) != set(expected_names):
                raise RuntimeError(
                    "RBLN SDK output names do not match the artifact contract."
                )
            values = tuple(raw_outputs[name] for name in expected_names)
        elif isinstance(raw_outputs, (list, tuple)):
            values = tuple(raw_outputs)
        else:
            raise RuntimeError(
                "RBLN SDK outputs must be NumPy arrays or an array sequence."
            )

        if len(values) != len(self._output_descriptors):
            raise RuntimeError(
                "RBLN SDK output count does not match the artifact contract."
            )

        outputs = {}
        for descriptor, value in zip(self._output_descriptors, values):
            if not isinstance(value, np.ndarray):
                raise RuntimeError("RBLN SDK outputs must be NumPy arrays.")
            if value.dtype != descriptor.dtype:
                raise RuntimeError(
                    f"RBLN SDK output dtype for '{descriptor.name}' does not "
                    "match the artifact contract."
                )
            if value.shape != descriptor.shape:
                raise RuntimeError(
                    f"RBLN SDK output shape for '{descriptor.name}' does not "
                    "match the artifact contract."
                )
            outputs[descriptor.name] = value
        return outputs

    def _ensure_sync_runtime(self):
        self._require_loaded()
        self._reject_async_ownership()
        if self._sync_runtime is not None:
            return self._sync_runtime
        try:
            runtime = self._rebel.Runtime(
                str(self.compiled_model.artifact_path),
                device=self.device_id,
                tensor_type="np",
                timeout=self.runtime_timeout_sec,
            )
        except Exception as exc:
            raise RuntimeError("RBLN could not create sync runtime.") from exc
        self._sync_runtime = runtime
        self._execution_mode = "sync"
        return runtime

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        ordered_inputs = self._ordered_inputs(inputs)
        runtime = self._ensure_sync_runtime()
        try:
            raw_outputs = runtime(*ordered_inputs)
        except Exception as exc:
            raise RuntimeError("RBLN synchronous inference failed.") from exc
        return self._normalize_outputs(raw_outputs)

    def warmup(self, inputs: Dict[str, np.ndarray], num_runs: int = 1) -> None:
        self._require_loaded()
        self._reject_async_ownership()
        for _ in range(num_runs):
            self.run(inputs)

    def native_async_max_batch_size(self) -> int:
        return 1

    def max_concurrent_workers(self) -> int:
        if self._execution_mode == "native_async" or self._native_backend is not None:
            return self.max_async_inflight
        return 1

    def unload(self) -> None:
        native_backend = self._native_backend
        if native_backend is not None:
            self._cleanup_pending = True
            try:
                quiesced = native_backend.shutdown(
                    timeout=self.shutdown_timeout_sec
                )
            except Exception as exc:
                raise RuntimeError(
                    "RBLN native async shutdown could not prove completion."
                ) from exc
            if quiesced is not True:
                raise RuntimeError(
                    "RBLN native async backend did not quiesce; loaded "
                    "artifact state was retained."
                )
            self._native_backend = None
        self._sync_runtime = None
        self.compiled_model = None
        self._rebel = None
        self._execution_mode = None
        self._cleanup_pending = False
        self._detected_npu = None
        self._sdk_version = None
        self._artifact_metadata = {}
        self._input_descriptors = ()
        self._input_bindings = ()
        self._output_descriptors = ()

    def get_device_spec(self) -> Dict[str, Any]:
        device_spec: dict[str, Any] = {
            "backend": "rbln",
            "device": self.device,
            "device_id": self.device_id,
            "accelerator_vendor": "Rebellions",
            "accelerator_name": self._detected_npu or _EXPECTED_NPU,
            "async_parallel": self.async_parallel,
            "max_async_inflight": self.max_async_inflight,
        }
        if self.compiled_model is not None:
            device_spec["execution_mode"] = self._execution_mode or "loaded"
        if self._detected_npu is not None:
            device_spec["detected_npu"] = self._detected_npu
        if self._sdk_version is not None:
            device_spec["sdk_version"] = self._sdk_version
        if self._artifact_metadata:
            metadata = self._artifact_metadata
            if metadata.get("compiler_version") is not None:
                device_spec["artifact_compiler_version"] = metadata[
                    "compiler_version"
                ]
            device_spec["artifact_npu"] = metadata["npu"]
            device_spec["tensor_parallel_size"] = metadata[
                "tensor_parallel_size"
            ]
            if metadata.get("uuid") is not None:
                device_spec["artifact_uuid"] = metadata["uuid"]
            if metadata.get("alloc_per_node") is not None:
                device_spec["artifact_alloc_per_node"] = list(
                    metadata["alloc_per_node"]
                )
        if self._input_descriptors:
            device_spec["input_names"] = [
                item.name for item in self._input_descriptors
            ]
            device_spec["input_shapes"] = [
                list(item.shape) for item in self._input_descriptors
            ]
            device_spec["input_dtypes"] = [
                item.dtype.name for item in self._input_descriptors
            ]
        if self._output_descriptors:
            device_spec["output_names"] = [
                item.name for item in self._output_descriptors
            ]
            device_spec["output_shapes"] = [
                list(item.shape) for item in self._output_descriptors
            ]
            device_spec["output_dtypes"] = [
                item.dtype.name for item in self._output_descriptors
            ]
        return device_spec

    def is_compatible(self, compiled_model: CompiledModel) -> bool:
        if not isinstance(compiled_model, CompiledModel):
            return False
        backend_name = compiled_model.backend_name
        if type(backend_name) is not str:
            return False
        return (
            backend_name.strip().lower() in _BACKEND_NAMES
            and Path(compiled_model.artifact_path).suffix.lower() == ".rbln"
        )
