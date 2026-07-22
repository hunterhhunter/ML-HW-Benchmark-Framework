"""Mobilint qb Runtime adapter shared by ARIES and REGULUS MXQ targets."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict

import numpy as np

from core.compiled_model import CompiledModel
from mobilint_device import MobilintDeviceSession
from .base import Runtime


_CORE_MODES = frozenset({"auto", "single", "multi", "global4", "global8"})
_BACKEND_NAMES = frozenset({"mobilint", "qbruntime", "mxq"})


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
        self._device_info = session.acquire()
        self._device_session = session
        self._cleanup_pending = True
        try:
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

    def unload(self) -> None:
        if self._model is None and self._device_session is None:
            self._cleanup_pending = False
            return
        self._cleanup_pending = True
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
