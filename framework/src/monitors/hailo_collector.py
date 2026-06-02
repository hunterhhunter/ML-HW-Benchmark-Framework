import contextlib
import os
from typing import Any, Dict, Optional

from .base import Collector


class HailoCollector(Collector):
    """Hailo-8/8L telemetry collector using the HailoRT Python control API."""

    def __init__(
        self,
        device_id: str = "device0",
        enable_power: bool = True,
        power_mode: str = "auto",
        suppress_power_errors: bool = True,
    ):
        self.device_id = device_id
        self.enable_power = enable_power
        self.power_mode = power_mode.lower()
        self.suppress_power_errors = suppress_power_errors
        self._hailo = None
        self._device = None
        self._started = False
        self._power_available = False
        self._power_unsupported = False
        self._last_error: str | None = None

    def is_available(self) -> bool:
        try:
            hailo = self._import_hailo_platform()
            device_infos = hailo.Device.scan()
            return bool(device_infos)
        except Exception:
            return False

    def start(self) -> None:
        self._hailo = self._import_hailo_platform()
        self._device = self._open_device()
        self._started = True

        if self.enable_power and self.power_mode != "off":
            self._start_power_measurement()

    def collect(self) -> Dict[str, Optional[float]]:
        if not self._started or self._device is None:
            return {}

        metrics: Dict[str, Optional[float]] = {}
        try:
            temp = self._device.control.get_chip_temperature()
            metrics["hw_accel_temp_c"] = float(getattr(temp, "ts0_temperature"))
        except Exception as exc:
            self._last_error = str(exc)
            metrics["hw_accel_temp_c"] = None

        if self._power_available:
            try:
                power = self._device.control.get_power_measurement()
                metrics["hw_accel_power_w"] = float(getattr(power, "average_value"))
            except Exception as exc:
                self._power_available = False
                self._last_error = self._format_power_error(exc)
                metrics["hw_accel_power_w"] = None

        return metrics

    def stop(self) -> None:
        if self._device is not None and self._power_available:
            self._stop_power_measurement(ignore_errors=True)
        self._started = False
        self._device = None
        self._power_available = False

    def get_static_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "hw_accel_vendor": "Hailo",
            "hw_accel_name": "Hailo-8/8L",
            "hw_accel_device_id": self.device_id,
        }
        if self._last_error:
            info["hw_accel_monitor_note"] = self._last_error
        return info

    def _start_power_measurement(self) -> None:
        try:
            with self._maybe_suppress_native_power_logs():
                self._stop_power_measurement(ignore_errors=True)
                self._device.control.set_power_measurement()
                self._device.control.start_power_measurement()
            self._power_available = True
            self._last_error = None
        except Exception as exc:
            self._power_available = False
            self._power_unsupported = self._is_unsupported_power_error(exc)
            self._last_error = self._format_power_error(exc)

    def _stop_power_measurement(self, ignore_errors: bool = False) -> None:
        try:
            with self._maybe_suppress_native_power_logs():
                self._device.control.stop_power_measurement()
        except Exception:
            if not ignore_errors:
                raise

    def _format_power_error(self, exc: Exception) -> str:
        if self._is_unsupported_power_error(exc):
            return (
                "Hailo power measurement is unsupported on this board; "
                "temperature-only monitoring is active."
            )
        return f"Hailo power measurement unavailable: {exc}"

    def _is_unsupported_power_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return "unsupported" in text or "not supported" in text

    @contextlib.contextmanager
    def _maybe_suppress_native_power_logs(self):
        if not self.suppress_power_errors:
            yield
            return

        # HailoRT emits unsupported-power messages from native code directly to
        # stderr before Python raises. Hide only this optional probe path.
        stderr_fd = 2
        saved_fd = os.dup(stderr_fd)
        try:
            with open(os.devnull, "w") as devnull:
                os.dup2(devnull.fileno(), stderr_fd)
                yield
        finally:
            os.dup2(saved_fd, stderr_fd)
            os.close(saved_fd)

    def _import_hailo_platform(self):
        try:
            import hailo_platform
        except ImportError as exc:
            raise ImportError(
                "HailoRT Python package is not installed; Hailo telemetry is unavailable."
            ) from exc
        return hailo_platform

    def _open_device(self):
        device_infos = self._hailo.Device.scan()
        if device_infos:
            selected = self._select_device_info(device_infos)
            return self._hailo.Device(selected)
        return self._hailo.Device()

    def _select_device_info(self, device_infos):
        if self.device_id in ("", "device0", "default"):
            return device_infos[0]
        for device_info in device_infos:
            if self.device_id in str(device_info):
                return device_info
        return device_infos[0]
