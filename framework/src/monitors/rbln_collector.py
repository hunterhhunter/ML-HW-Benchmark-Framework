"""Rebellions device telemetry collected from ``rbln-smi`` JSON."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, Optional

from .base import Collector


_MIB = 1024**2
_NUMBER_PATTERN = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*?)$"
)
_NONFINITE_PATTERN = re.compile(
    r"^[+-]?(?:nan|inf(?:inity)?)\s*(?:[%a-zA-Zµμ/]+)?$",
    re.IGNORECASE,
)


def _number(value: object, suffix: str | None = None) -> float | None:
    """Return one finite numeric value with an optional unit suffix."""
    if type(value) in (int, float):
        result = float(value)
    elif type(value) is str:
        match = _NUMBER_PATTERN.fullmatch(value.strip())
        if match is None:
            return None
        unit = match.group(2)
        if suffix is not None and unit.casefold() != suffix.casefold():
            return None
        if suffix is None and unit:
            return None
        try:
            result = float(match.group(1))
        except ValueError:
            return None
    else:
        return None
    return result if math.isfinite(result) else None


def _bytes(value: object) -> float | None:
    """Normalize a byte count without coercing arbitrary objects."""
    for suffix, multiplier in (
        (None, 1.0),
        ("B", 1.0),
        ("KiB", 1024.0),
        ("MiB", float(_MIB)),
        ("GiB", float(1024**3)),
    ):
        parsed = _number(value, suffix=suffix)
        if parsed is not None:
            result = parsed * multiplier
            return result if result >= 0.0 else None
    return None


def _power_w(value: object) -> float | None:
    """Normalize numeric watts or W/mW/uW suffixed strings to watts."""
    if type(value) in (int, float):
        parsed = _number(value)
        return parsed if parsed is not None and parsed >= 0.0 else None
    for suffix, divisor in (
        ("W", 1.0),
        ("mW", 1_000.0),
        ("uW", 1_000_000.0),
        ("µW", 1_000_000.0),
        ("μW", 1_000_000.0),
    ):
        parsed = _number(value, suffix=suffix)
        if parsed is not None:
            return parsed / divisor if parsed >= 0.0 else None
    return None


def _bounded_text(value: object, limit: int = 128) -> str | None:
    """Copy bounded primitive text without invoking arbitrary ``str`` methods."""
    if type(value) is not str:
        return None
    result = " ".join(value.split())
    return result[:limit] if result else None


def _integer(value: object) -> int | None:
    parsed = _number(value)
    if parsed is None or not parsed.is_integer():
        return None
    return int(parsed)


def _require_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative built-in int")
    return value


def _require_positive_number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


class RblnCollector(Collector):
    """Collect one explicitly selected RBLN device through ``rbln-smi``."""

    def __init__(
        self,
        device_id: int = 0,
        sample_interval_sec: float = 1.0,
        command_timeout_sec: float = 2.0,
        runner: Callable[..., object] = subprocess.run,
        clock: Callable[[], float] = time.monotonic,
        executable_resolver: Callable[[str], str | None] = shutil.which,
        process_id: int | None = None,
    ):
        self.device_id = _require_nonnegative_int(device_id, "device_id")
        self.sample_interval_sec = max(
            1.0,
            _require_positive_number(
                sample_interval_sec,
                "sample_interval_sec",
            ),
        )
        self.command_timeout_sec = _require_positive_number(
            command_timeout_sec,
            "command_timeout_sec",
        )
        self._runner = runner
        self._clock = clock
        self._executable_resolver = executable_resolver
        self._process_id = os.getpid() if process_id is None else process_id

        self._started = False
        self._stopped = False
        self._last_poll_at: float | None = None
        self._last_snapshot: Dict[str, Optional[float]] = {}
        self._poll_attempts = 0
        self._poll_successes = 0
        self._power_samples = 0
        self._energy_joules = 0.0
        self._last_power_w: float | None = None
        self._last_power_at: float | None = None
        self._last_error_type: str | None = None
        self._static_device_info: Dict[str, Any] = {}

    def is_available(self) -> bool:
        return True

    def start(self) -> None:
        if self._started:
            raise RuntimeError("RblnCollector is already started")
        resolved_executable = self._executable_resolver("rbln-smi")
        if (
            type(resolved_executable) is not str
            or not resolved_executable.strip()
        ):
            raise RuntimeError("rbln-smi executable was not found")

        self._reset_state()
        observed_at = float(self._clock())
        self._last_poll_at = observed_at
        self._poll_attempts += 1
        current, static = self._snapshot()
        self._record_success(current, observed_at)
        self._last_snapshot = current
        self._static_device_info = static
        self._started = True
        self._stopped = False

    def collect(self, force: bool = False) -> Dict[str, Optional[float]]:
        if not self._started or self._stopped:
            return {}
        observed_at = float(self._clock())
        if (
            not force
            and self._last_poll_at is not None
            and observed_at - self._last_poll_at < self.sample_interval_sec
        ):
            return {}
        self._last_poll_at = observed_at
        self._poll_attempts += 1
        try:
            current, static = self._snapshot()
            self._record_success(current, observed_at)
        except Exception as exc:
            self._record_failure(exc)
            return {}
        self._last_snapshot = current
        self._static_device_info.update(static)
        return dict(current)

    def stop(self) -> None:
        if not self._started or self._stopped:
            return
        self.collect(force=True)
        self._stopped = True
        self._started = False

    def get_static_info(self) -> Dict[str, Any]:
        return dict(self._static_device_info)

    def get_summary_metrics(self) -> Dict[str, Any]:
        coverage = (
            self._poll_successes / self._poll_attempts
            if self._poll_attempts
            else 0.0
        )
        summary: Dict[str, Any] = {
            "hw_accel_power_samples": self._power_samples,
            "hw_accel_monitor_attempts": self._poll_attempts,
            "hw_accel_monitor_successes": self._poll_successes,
            "hw_accel_monitor_coverage": round(coverage, 6),
        }
        if self._power_samples >= 2:
            summary["hw_accel_energy_j"] = round(self._energy_joules, 6)
        if self._last_error_type is not None:
            summary["hw_accel_monitor_note"] = (
                f"RBLN snapshot failed: {self._last_error_type}"
            )[:128]
        return summary

    def _reset_state(self) -> None:
        self._stopped = False
        self._last_poll_at = None
        self._last_snapshot = {}
        self._poll_attempts = 0
        self._poll_successes = 0
        self._power_samples = 0
        self._energy_joules = 0.0
        self._last_power_w = None
        self._last_power_at = None
        self._last_error_type = None
        self._static_device_info = {}

    def _record_success(
        self,
        current: Dict[str, Optional[float]],
        observed_at: float,
    ) -> None:
        power_w = current.get("hw_accel_power_w")
        if power_w is None:
            self._poll_successes += 1
            self._last_power_w = None
            self._last_power_at = None
            return

        next_energy = self._energy_joules
        if self._last_power_w is not None and self._last_power_at is not None:
            elapsed = max(0.0, observed_at - self._last_power_at)
            interval_energy = (
                (self._last_power_w + power_w) / 2.0 * elapsed
            )
            next_energy += interval_energy
            if not math.isfinite(interval_energy) or not math.isfinite(
                next_energy
            ):
                raise ValueError("RBLN energy integration must remain finite")
        self._poll_successes += 1
        self._power_samples += 1
        self._energy_joules = next_energy
        self._last_power_w = power_w
        self._last_power_at = observed_at

    def _record_failure(self, exc: Exception) -> None:
        raw_type = type(exc).__name__
        safe_type = "".join(
            character
            for character in raw_type
            if character.isascii()
            and (character.isalnum() or character == "_")
        )[:64]
        self._last_error_type = safe_type or "Exception"
        self._last_power_w = None
        self._last_power_at = None

    def _snapshot(self) -> tuple[Dict[str, Optional[float]], Dict[str, Any]]:
        completed = self._runner(
            ["rbln-smi", "-b", "-j", "-d", str(self.device_id)],
            capture_output=True,
            text=True,
            check=True,
            timeout=self.command_timeout_sec,
            shell=False,
        )
        payload = json.loads(completed.stdout)
        if type(payload) is not dict:
            raise RuntimeError("rbln-smi JSON root must be an object")

        devices = payload.get("devices")
        if type(devices) is not list:
            raise RuntimeError("rbln-smi JSON is missing devices")
        selected = [
            device
            for device in devices
            if type(device) is dict
            and _integer(device.get("npu")) == self.device_id
        ]
        if len(selected) != 1:
            raise RuntimeError(
                f"rbln-smi JSON must contain exactly one device {self.device_id}"
            )

        device = selected[0]
        status = _bounded_text(device.get("status"))
        if status is None or status.casefold() != "normal":
            raise RuntimeError(
                f"RBLN device {self.device_id} status is not normal"
            )

        self._reject_nonfinite_fields(payload, device)
        current = self._current_metrics(payload, device)
        static = self._static_info(payload, device, status)
        self._reject_nonfinite_normalized(current, static)
        return current, static

    @staticmethod
    def _reject_nonfinite_normalized(
        current: Dict[str, Optional[float]],
        static: Dict[str, Any],
    ) -> None:
        for key, value in (*current.items(), *static.items()):
            if type(value) is float and not math.isfinite(value):
                raise ValueError(
                    f"normalized RBLN field {key} must be finite"
                )

    @staticmethod
    def _reject_nonfinite_fields(
        payload: dict[str, object],
        device: dict[str, object],
    ) -> None:
        values: list[tuple[str, object]] = [
            ("util", device.get("util")),
            ("temperature", device.get("temperature")),
            ("card_power", device.get("card_power")),
        ]
        memory = device.get("memory")
        if type(memory) is dict:
            values.extend(
                (
                    ("memory.used", memory.get("used")),
                    ("memory.total", memory.get("total")),
                )
            )
        pci = device.get("pci")
        if type(pci) is dict:
            values.extend(
                (
                    ("pci.numa_node", pci.get("numa_node")),
                    ("pci.link_speed", pci.get("link_speed")),
                    ("pci.link_width", pci.get("link_width")),
                )
            )
        contexts = payload.get("contexts")
        if type(contexts) is list:
            for index, context in enumerate(contexts):
                if type(context) is not dict:
                    continue
                values.extend(
                    (
                        (f"contexts[{index}].npu", context.get("npu")),
                        (f"contexts[{index}].pid", context.get("pid")),
                        (
                            f"contexts[{index}].memalloc",
                            context.get("memalloc"),
                        ),
                    )
                )

        for field, value in values:
            if (
                type(value) is float
                and not math.isfinite(value)
            ) or (
                type(value) is str
                and _NONFINITE_PATTERN.fullmatch(value.strip()) is not None
            ):
                raise ValueError(
                    f"rbln-smi JSON field {field} must be finite"
                )

    def _current_metrics(
        self,
        payload: dict[str, object],
        device: dict[str, object],
    ) -> Dict[str, Optional[float]]:
        metrics: Dict[str, Optional[float]] = {}
        utilization = _number(device.get("util"))
        if utilization is None:
            utilization = _number(device.get("util"), suffix="%")
        temperature = _number(device.get("temperature"), suffix="C")
        if temperature is None:
            temperature = _number(device.get("temperature"))
        power_w = _power_w(device.get("card_power"))

        memory = device.get("memory")
        memory_used = (
            _bytes(memory.get("used")) if type(memory) is dict else None
        )
        process_memory = self._process_memory_bytes(payload.get("contexts"))

        if utilization is not None:
            metrics["hw_accel_util"] = utilization
        if memory_used is not None:
            metrics["hw_accel_mem_used_mb"] = memory_used / _MIB
        if process_memory is not None:
            metrics["hw_accel_mem_proc_mb"] = process_memory / _MIB
        if temperature is not None:
            metrics["hw_accel_temp_c"] = temperature
        if power_w is not None:
            metrics["hw_accel_power_w"] = power_w
        return metrics

    def _process_memory_bytes(self, contexts: object) -> float | None:
        if type(contexts) is not list:
            return None
        total = 0.0
        found = False
        for context in contexts:
            if type(context) is not dict:
                continue
            if _integer(context.get("npu")) != self.device_id:
                continue
            if _integer(context.get("pid")) != self._process_id:
                continue
            allocated = _bytes(context.get("memalloc"))
            if allocated is None:
                continue
            total += allocated
            found = True
        return total if found else None

    def _static_info(
        self,
        payload: dict[str, object],
        device: dict[str, object],
        status: str,
    ) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "hw_accel_vendor": "Rebellions",
            "hw_accel_device_id": self.device_id,
            "hw_accel_status": status,
            "hw_accel_monitor_source": "rbln-smi-json",
        }
        text_fields = {
            "name": "hw_accel_name",
            "device": "hw_accel_device_node",
            "uuid": "hw_accel_uuid",
            "sid": "hw_accel_serial_id",
            "pstate": "hw_accel_pstate",
            "fw_ver": "hw_accel_firmware_version",
        }
        for source, target in text_fields.items():
            value = _bounded_text(device.get(source))
            if value is not None:
                info[target] = value

        kmd = _bounded_text(payload.get("KMD_version"))
        if kmd is None:
            kmd = _bounded_text(payload.get("driver_version"))
        if kmd is not None:
            info["hw_accel_kmd_version"] = kmd

        memory = device.get("memory")
        total_bytes = (
            _bytes(memory.get("total")) if type(memory) is dict else None
        )
        if total_bytes is not None:
            info["hw_accel_mem_total_mb"] = total_bytes / _MIB

        pci = device.get("pci")
        if type(pci) is dict:
            pci_text_fields = {
                "bus_id": "hw_accel_pci_bus_id",
                "link_speed": "hw_accel_pci_link_speed",
            }
            for source, target in pci_text_fields.items():
                value = _bounded_text(pci.get(source))
                if value is not None:
                    info[target] = value
            pci_integer_fields = {
                "numa_node": "hw_accel_pci_numa_node",
                "link_width": "hw_accel_pci_link_width",
            }
            for source, target in pci_integer_fields.items():
                value = _integer(pci.get(source))
                if value is not None:
                    info[target] = value
        return info
