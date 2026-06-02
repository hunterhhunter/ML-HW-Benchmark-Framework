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
        power_buffer_index: str = "MEASUREMENT_BUFFER_INDEX_0",
        power_should_clear: bool = True,
    ):
        self.device_id = device_id
        self.enable_power = enable_power
        self.power_mode = power_mode.lower()
        self.suppress_power_errors = suppress_power_errors
        self.power_buffer_index = power_buffer_index
        self.power_should_clear = power_should_clear
        self._hailo = None
        self._device = None
        self._started = False
        self._power_available = False
        self._power_unsupported = False
        self._power_api: str | None = None
        self._power_buffer = None
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
                power = self._get_power_measurement()
                metrics["hw_accel_power_w"] = float(getattr(power, "average_value"))
                if hasattr(power, "min_value"):
                    metrics["hw_accel_power_min_w"] = float(getattr(power, "min_value"))
                if hasattr(power, "max_value"):
                    metrics["hw_accel_power_max_w"] = float(getattr(power, "max_value"))
                if hasattr(power, "average_time_value_milliseconds"):
                    metrics["hw_accel_power_sample_period_ms"] = float(
                        getattr(power, "average_time_value_milliseconds")
                    )
            except Exception as exc:
                self._power_available = False
                self._last_error = self._format_power_error(exc)
                metrics["hw_accel_power_w"] = None

        return metrics

    def stop(self) -> None:
        if self._device is not None and self._power_available:
            self._stop_power_measurement(ignore_errors=True)
        if self._device is not None and hasattr(self._device, "release"):
            try:
                self._device.release()
            except Exception:
                pass
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
        if self._power_api:
            info["hw_accel_power_api"] = self._power_api
        return info

    def _start_power_measurement(self) -> None:
        errors = []

        if self.power_mode in ("auto", "buffer"):
            try:
                self._start_buffered_power_measurement()
                return
            except Exception as exc:
                errors.append(exc)
                if self.power_mode == "buffer":
                    self._mark_power_unavailable(exc)
                    return

        if self.power_mode in ("auto", "legacy"):
            try:
                self._start_legacy_power_measurement()
                return
            except Exception as exc:
                errors.append(exc)

        exc = errors[-1] if errors else RuntimeError("power measurement disabled")
        self._mark_power_unavailable(exc)

    def _start_buffered_power_measurement(self) -> None:
        try:
            with self._maybe_suppress_native_power_logs():
                self._stop_power_measurement(ignore_errors=True)
                buffer_index = self._resolve_power_buffer_index()
                self._device.control.set_power_measurement(buffer_index=buffer_index)
                self._device.control.start_power_measurement()
            self._power_available = True
            self._power_api = "buffer"
            self._power_buffer = buffer_index
            self._last_error = None
        except Exception:
            self._power_api = None
            self._power_buffer = None
            raise

    def _start_legacy_power_measurement(self) -> None:
        with self._maybe_suppress_native_power_logs():
            self._stop_power_measurement(ignore_errors=True)
            self._device.control.set_power_measurement()
            self._device.control.start_power_measurement()
        self._power_available = True
        self._power_api = "legacy"
        self._power_buffer = None
        self._last_error = None

    def _get_power_measurement(self):
        if self._power_api == "buffer":
            return self._device.control.get_power_measurement(
                buffer_index=self._power_buffer,
                should_clear=self.power_should_clear,
            )
        return self._device.control.get_power_measurement()

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
                "Hailo power measurement is unsupported or unavailable on this board/API; "
                "temperature-only monitoring is active."
            )
        return f"Hailo power measurement unavailable: {exc}"

    def _is_unsupported_power_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return "unsupported" in text or "not supported" in text

    def _mark_power_unavailable(self, exc: Exception) -> None:
        self._power_available = False
        self._power_unsupported = self._is_unsupported_power_error(exc)
        self._power_api = None
        self._power_buffer = None
        self._last_error = self._format_power_error(exc)

    def _resolve_power_buffer_index(self):
        buffer_enum = getattr(self._hailo, "MeasurementBufferIndex", None)
        if buffer_enum is None:
            raise RuntimeError("MeasurementBufferIndex is not available in hailo_platform")
        if isinstance(self.power_buffer_index, str):
            return getattr(buffer_enum, self.power_buffer_index)
        return self.power_buffer_index

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
