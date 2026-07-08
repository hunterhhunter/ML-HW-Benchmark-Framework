from importlib import import_module
from typing import Any, Dict, Optional

from .base import Collector


class DeepXCollector(Collector):
    """DEEPX DX-RT telemetry collector using the dx_engine DeviceStatus API."""

    def __init__(
        self,
        device_id: str | int = "all",
        core_count: int | None = None,
        max_core_probe: int = 8,
    ):
        self.device_id = device_id
        self.core_count = core_count
        self.max_core_probe = max_core_probe
        self._device_status_cls = None
        self._device_ids: list[int] = []
        self._core_channels: list[int] = []
        self._device_count: int | None = None
        self._started = False
        self._last_error: str | None = None

    def is_available(self) -> bool:
        try:
            device_status_cls = self._import_device_status()
            return int(device_status_cls.get_device_count()) > 0
        except Exception:
            return False

    def start(self) -> None:
        self._device_status_cls = self._import_device_status()
        self._device_count = int(self._device_status_cls.get_device_count())
        if self._device_count <= 0:
            raise RuntimeError("No DEEPX devices found by dx_engine DeviceStatus.")

        self._device_ids = self._resolve_device_ids(self._device_count)
        self._core_channels = self._resolve_core_channels()
        self._started = True
        self._last_error = None

    def collect(self) -> Dict[str, Optional[float]]:
        if not self._started or self._device_status_cls is None:
            return {}

        temperatures: list[float] = []
        voltages: list[float] = []
        clocks: list[float] = []

        for device_id in self._device_ids:
            try:
                status = self._device_status_cls.get_current_status(device_id)
            except Exception as exc:
                self._last_error = f"DeepX device status unavailable for device {device_id}: {exc}"
                continue

            for channel in self._core_channels:
                temp = self._read_metric(status, "get_temperature", channel)
                voltage = self._read_metric(status, "get_npu_voltage", channel)
                clock = self._read_metric(status, "get_npu_clock", channel)
                if temp is not None:
                    temperatures.append(temp)
                if voltage is not None:
                    voltages.append(voltage)
                if clock is not None:
                    clocks.append(clock)

        metrics: Dict[str, Optional[float]] = {}
        if temperatures:
            metrics["hw_accel_temp_c"] = round(max(temperatures), 2)
        if voltages:
            metrics["hw_accel_voltage_mv"] = round(sum(voltages) / len(voltages), 2)
        if clocks:
            metrics["hw_accel_clock_mhz"] = round(sum(clocks) / len(clocks), 2)
        if not metrics and self._last_error is None:
            self._last_error = "DeepX DeviceStatus returned no readable telemetry metrics."

        return metrics

    def stop(self) -> None:
        self._started = False
        self._device_status_cls = None

    def get_static_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "hw_accel_vendor": "DEEPX",
            "hw_accel_name": "DEEPX NPU",
            "hw_accel_device_id": self._format_device_id(),
            "hw_accel_monitor_source": "dx_engine.dev_status.DeviceStatus",
        }
        if self._device_count is not None:
            info["hw_accel_device_count"] = self._device_count
        if self._core_channels:
            info["hw_accel_core_channels"] = ",".join(str(ch) for ch in self._core_channels)
        if self._last_error:
            info["hw_accel_monitor_note"] = self._last_error
        return info

    def _import_device_status(self):
        try:
            module = import_module("dx_engine.dev_status")
            return getattr(module, "DeviceStatus")
        except ImportError:
            try:
                module = import_module("dx_engine")
                return getattr(module, "DeviceStatus")
            except (ImportError, AttributeError) as exc:
                raise ImportError(
                    "DEEPX DX-RT DeviceStatus API is not importable. "
                    "Install the dx_engine package from DX-RT v3.3+ to enable DeepX telemetry."
                ) from exc
        except AttributeError as exc:
            raise ImportError(
                "dx_engine.dev_status is present but DeviceStatus is unavailable."
            ) from exc

    def _resolve_device_ids(self, device_count: int) -> list[int]:
        if isinstance(self.device_id, int):
            requested = [self.device_id]
        else:
            raw = str(self.device_id).strip().lower()
            if raw in ("", "all", "auto"):
                requested = list(range(device_count))
            else:
                requested = [self._parse_device_id(part) for part in raw.split(",") if part.strip()]

        valid = [device_id for device_id in requested if 0 <= device_id < device_count]
        if not valid:
            raise ValueError(
                f"No valid DeepX device IDs selected. requested={requested}, device_count={device_count}"
            )
        return valid

    def _parse_device_id(self, value: str) -> int:
        stripped = value.strip().lower()
        if stripped.startswith("npu"):
            stripped = stripped[3:]
        if stripped.startswith("device"):
            stripped = stripped[6:]
        return int(stripped)

    def _resolve_core_channels(self) -> list[int]:
        if self.core_count is not None:
            if self.core_count <= 0:
                raise ValueError("DeepX core_count must be positive when provided.")
            return list(range(self.core_count))

        try:
            status = self._device_status_cls.get_current_status(self._device_ids[0])
        except Exception:
            return [0]

        channels: list[int] = []
        for channel in range(max(1, self.max_core_probe)):
            readable = any(
                self._read_metric(status, method_name, channel) is not None
                for method_name in ("get_temperature", "get_npu_voltage", "get_npu_clock")
            )
            if readable:
                channels.append(channel)
                continue
            if channels:
                break

        return channels or [0]

    def _read_metric(self, status, method_name: str, channel: int) -> float | None:
        method = getattr(status, method_name, None)
        if not callable(method):
            return None
        try:
            value = method(channel)
        except Exception:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _format_device_id(self) -> str:
        if self._device_ids:
            return ",".join(str(device_id) for device_id in self._device_ids)
        return str(self.device_id)
