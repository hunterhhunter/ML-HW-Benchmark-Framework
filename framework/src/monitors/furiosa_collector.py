"""Furiosa RNGD raw device power through the official Python SMI binding."""

from __future__ import annotations

from importlib import import_module
import math
from typing import Any, Dict, Optional

from core.power_trace import PowerReading, PowerTraceSource

from .base import Collector


class FuriosaCollector(Collector):
    """Read one explicitly selected RNGD device without a CLI fallback."""

    def __init__(
        self,
        device_name: str = "npu0",
        power_sample_interval_sec: float = 0.2,
    ):
        if not isinstance(device_name, str) or not device_name.strip():
            raise ValueError("device_name must be a non-empty string")
        if (
            type(power_sample_interval_sec) not in (int, float)
            or isinstance(power_sample_interval_sec, bool)
            or not math.isfinite(float(power_sample_interval_sec))
            or power_sample_interval_sec <= 0
        ):
            raise ValueError(
                "power_sample_interval_sec must be a positive finite number"
            )
        self.device_name = device_name.strip()
        self.power_sample_interval_sec = float(power_sample_interval_sec)
        self._device = None
        self._started = False
        self._unavailable_reason = "collector_not_started"

    def is_available(self) -> bool:
        try:
            module = import_module("furiosa_smi_py")
        except Exception:
            return False
        return callable(getattr(module, "init", None)) and callable(
            getattr(module, "list_devices", None)
        )

    def power_trace_source(self) -> PowerTraceSource:
        return PowerTraceSource(
            collector="furiosa",
            monitor_source="furiosa-smi-py",
            device_id=self.device_name,
            power_scope="device",
            sample_interval_sec=self.power_sample_interval_sec,
        )

    def start(self) -> None:
        if self._started:
            raise RuntimeError("FuriosaCollector is already started")
        self._device = None
        self._unavailable_reason = "smi_unavailable"
        try:
            module = import_module("furiosa_smi_py")
            init = getattr(module, "init")
            list_devices = getattr(module, "list_devices")
            if not callable(init) or not callable(list_devices):
                raise AttributeError("Furiosa SMI entry points are unavailable")
            init()
            devices = list(list_devices())
            selected = []
            for device in devices:
                device_info = getattr(device, "device_info", None)
                if not callable(device_info):
                    continue
                info = device_info()
                name = getattr(info, "name", None)
                if not callable(name):
                    continue
                if name() == self.device_name:
                    selected.append(device)
            if len(selected) != 1:
                raise RuntimeError(
                    "Furiosa SMI must expose exactly one matching device"
                )
            power = getattr(selected[0], "power_consumption", None)
            if not callable(power):
                raise AttributeError("Furiosa power API is unavailable")
        except Exception as exc:
            self._unavailable_reason = f"smi:{self._safe_exception_type(exc)}"
            self._started = True
            return

        self._device = selected[0]
        self._unavailable_reason = ""
        self._started = True

    def collect(self) -> Dict[str, Optional[float]]:
        return {}

    def collect_power(self) -> PowerReading:
        device = self._device
        if not self._started or device is None:
            return PowerReading(
                status="unavailable",
                error_code=self._unavailable_reason or "device_unavailable",
            )
        try:
            power_w = float(device.power_consumption())
            if not math.isfinite(power_w) or power_w < 0:
                raise ValueError("power must be finite and non-negative")
        except Exception as exc:
            return PowerReading(
                status="read_error",
                error_code=f"smi:{self._safe_exception_type(exc)}",
            )
        return PowerReading(status="ok", power_w=power_w)

    def stop(self) -> None:
        self._device = None
        self._started = False
        self._unavailable_reason = "collector_not_started"

    def get_static_info(self) -> Dict[str, Any]:
        if self._device is None:
            return {}
        return {
            "hw_accel_vendor": "FuriosaAI",
            "hw_accel_name": "RNGD",
            "hw_accel_device_id": self.device_name,
            "hw_accel_monitor_source": "furiosa-smi-py",
        }

    @staticmethod
    def _safe_exception_type(exc: Exception) -> str:
        exception_type = "".join(
            character
            for character in type(exc).__name__
            if character.isascii()
            and (character.isalnum() or character == "_")
        )[:64]
        return exception_type or "Exception"

