"""Mobilint ARIES/REGULUS telemetry via the direct mbltml Python API."""

from __future__ import annotations

from importlib import import_module
import time
from typing import Any, Callable, Dict, Optional

from mobilint_device import MobilintDeviceSession

from .base import Collector


_MIB = 1024 ** 2


class MobilintCollector(Collector):
    """Collect one explicitly selected Mobilint device through mbltml."""

    def __init__(
        self,
        device_id: int = 0,
        expected_family: str = "aries",
        accelerator_name: str | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ):
        self.device_id = device_id
        self.expected_family = str(expected_family).strip().lower()
        self.accelerator_name = accelerator_name or self.expected_family.upper()
        self._clock_ns = clock_ns

        self._session: MobilintDeviceSession | None = None
        self._started = False
        self._cleanup_pending = False
        self._stop_boundary_attempted = False

        self._device_type: int | None = None
        self._node_name: str | None = None
        self._firmware_version: str | None = None
        self._memory_total_mb: float | None = None
        self._last_error: str | None = None

        self._energy_j = 0.0
        self._power_attempts = 0
        self._power_successes = 0
        self._last_power_w: float | None = None
        self._last_power_ns: int | None = None

    def is_available(self) -> bool:
        try:
            import_module("mbltml")
        except Exception:
            return False
        return True

    def start(self) -> None:
        if self._cleanup_pending:
            raise RuntimeError(
                "MobilintCollector cleanup is incomplete; call stop() to "
                "retry cleanup before starting again."
            )
        if self._session is not None:
            raise RuntimeError("MobilintCollector is already started")

        self._reset_measurements()
        self._reset_static_info()

        session = MobilintDeviceSession(self.device_id, self.expected_family)
        # Keep cleanup ownership reachable even if acquire() itself fails while
        # rolling back the process-global mbltml lease.
        self._session = session
        try:
            info = session.acquire()
        except BaseException as acquire_error:
            if session.module is not None:
                self._cleanup_pending = True
                raise RuntimeError(
                    "MobilintCollector start failed and rollback cleanup is "
                    f"incomplete ({type(acquire_error).__name__}: "
                    f"{acquire_error}); call stop() to retry cleanup."
                ) from acquire_error
            self._session = None
            raise

        try:
            module = session.module
            if module is None:
                raise RuntimeError(
                    "mbltml session acquired without an active module"
                )
            self._device_type = info.device_type
            self._node_name = self._safe_static(module, "mbltmlGetNodeName")
            self._firmware_version = self._safe_static(
                module,
                "mbltmlGetFirmwareVersion",
            )
            total_bytes = self._safe_static(module, "mbltmlGetMemoryTotal")
            self._memory_total_mb = (
                float(total_bytes) / _MIB if total_bytes is not None else None
            )
        except BaseException as start_error:
            try:
                session.release()
            except BaseException as cleanup_error:
                self._cleanup_pending = True
                raise RuntimeError(
                    "MobilintCollector start failed and rollback cleanup is "
                    f"incomplete: {cleanup_error}; call stop() to retry cleanup."
                ) from start_error
            self._session = None
            raise

        self._started = True

    def collect(self) -> Dict[str, Optional[float]]:
        if self._cleanup_pending:
            raise RuntimeError(
                "MobilintCollector cleanup is incomplete; call stop() to "
                "retry cleanup before collecting."
            )
        if not self._started or self._session is None:
            return {}

        metrics: Dict[str, Optional[float]] = {}
        util = self._safe_metric(
            "mbltmlGetTotalUtilization",
            self._scale_utilization,
        )
        memory = self._safe_metric(
            "mbltmlGetMemoryUsage",
            lambda value: value / _MIB,
        )
        temperature = self._safe_metric("mbltmlGetTemperature", float)

        if util is not None:
            metrics["hw_accel_util"] = util
        if memory is not None:
            metrics["hw_accel_mem_used_mb"] = memory
        if temperature is not None:
            metrics["hw_accel_temp_c"] = temperature

        if self.expected_family == "aries":
            power = self._read_power()
            current = self._safe_metric("mbltmlGetTotalCurrent", float)
            voltage = self._safe_metric(
                "mbltmlGetTotalVoltage",
                lambda value: value * 1000.0,
            )
            if power is not None:
                metrics["hw_accel_power_w"] = power
            if current is not None:
                metrics["hw_accel_current_a"] = current
            if voltage is not None:
                metrics["hw_accel_voltage_mv"] = voltage

        return metrics

    def stop(self) -> None:
        session = self._session
        if session is None:
            return

        if (
            self._started
            and self.expected_family == "aries"
            and not self._stop_boundary_attempted
        ):
            # The boundary reading belongs to the energy summary only. Mark it
            # before attempting release so a retry cannot count it twice.
            self._stop_boundary_attempted = True
            try:
                self._read_power()
            except BaseException as exc:
                boundary_error = exc
            else:
                boundary_error = None
        else:
            boundary_error = None

        try:
            session.release()
        except BaseException as cleanup_error:
            self._cleanup_pending = True
            if boundary_error is not None:
                raise RuntimeError(
                    "MobilintCollector stop boundary sampling failed and "
                    "cleanup is incomplete "
                    f"({type(cleanup_error).__name__}: {cleanup_error}); "
                    "call stop() to retry cleanup."
                ) from boundary_error
            raise

        self._session = None
        self._started = False
        self._cleanup_pending = False
        if boundary_error is not None:
            raise boundary_error

    def get_static_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "hw_accel_vendor": "Mobilint",
            "hw_accel_name": self.accelerator_name,
            "hw_accel_device_id": str(self.device_id),
            "hw_accel_family": self.expected_family,
            "hw_accel_monitor_source": "mbltml",
        }
        if self._device_type is not None:
            info["hw_accel_device_type"] = self._device_type
        if self._node_name is not None:
            info["hw_accel_node_name"] = self._node_name
        if self._firmware_version is not None:
            info["hw_accel_firmware_version"] = self._firmware_version
        if self._memory_total_mb is not None:
            info["hw_accel_mem_total_mb"] = round(self._memory_total_mb, 2)
        if self._last_error is not None:
            info["hw_accel_monitor_note"] = self._last_error
        return info

    def get_summary_metrics(self) -> Dict[str, Any]:
        if self.expected_family != "aries":
            return {}
        coverage = (
            self._power_successes / self._power_attempts
            if self._power_attempts
            else 0.0
        )
        return {
            "hw_accel_energy_j": round(self._energy_j, 6),
            "hw_accel_power_samples": self._power_successes,
            "hw_accel_power_sample_coverage": round(coverage, 6),
        }

    def _reset_measurements(self) -> None:
        self._energy_j = 0.0
        self._power_attempts = 0
        self._power_successes = 0
        self._last_power_w = None
        self._last_power_ns = None
        self._last_error = None
        self._stop_boundary_attempted = False

    def _reset_static_info(self) -> None:
        self._device_type = None
        self._node_name = None
        self._firmware_version = None
        self._memory_total_mb = None

    def _module(self):
        if self._session is None or self._session.module is None:
            raise RuntimeError("MobilintCollector has no active mbltml session")
        return self._session.module

    def _safe_static(self, module, method_name: str):
        try:
            return getattr(module, method_name)(self.device_id)
        except Exception as exc:
            self._last_error = f"Mobilint {method_name} failed: {exc}"
            return None

    def _safe_metric(
        self,
        method_name: str,
        transform: Callable[[float], float],
    ) -> float | None:
        try:
            raw = float(getattr(self._module(), method_name)(self.device_id))
            return float(transform(raw))
        except Exception as exc:
            self._last_error = f"Mobilint {method_name} failed: {exc}"
            return None

    def _read_power(self) -> float | None:
        self._power_attempts += 1
        try:
            power_w = float(
                self._module().mbltmlGetTotalPower(self.device_id)
            )
        except Exception as exc:
            self._last_error = f"Mobilint power sampling failed: {exc}"
            self._last_power_w = None
            self._last_power_ns = None
            return None

        self._power_successes += 1
        try:
            observed_ns = int(self._clock_ns())
        except Exception as exc:
            self._last_error = f"Mobilint power timestamp failed: {exc}"
            self._last_power_w = None
            self._last_power_ns = None
            return power_w

        if self._last_power_w is not None and self._last_power_ns is not None:
            elapsed_sec = (
                max(0, observed_ns - self._last_power_ns) / 1_000_000_000
            )
            self._energy_j += (
                (self._last_power_w + power_w) / 2.0 * elapsed_sec
            )
        self._last_power_w = power_w
        self._last_power_ns = observed_ns
        return power_w

    @staticmethod
    def _scale_utilization(value: float) -> float:
        return value * 100.0 if 0.0 <= value <= 1.0 else value
