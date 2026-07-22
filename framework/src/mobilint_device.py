"""Shared, lazy mbltml lifecycle and Mobilint device-family validation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import threading
from types import ModuleType


_FAMILIES = frozenset({"aries", "regulus"})


@dataclass(frozen=True)
class MobilintDeviceInfo:
    device_id: int
    device_type: int
    family: str


@dataclass
class _MbltmlState:
    module: ModuleType | None = None
    ref_count: int = 0
    family: str | None = None
    cleanup_pending: bool = False


_LOCK = threading.RLock()
_STATE = _MbltmlState()


def _load_mbltml() -> ModuleType:
    try:
        return import_module("mbltml")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "Mobilint device validation requires the optional 'mbltml' package. "
            "Install mbltml from the Mobilint SDK v1.3 distribution."
        ) from exc


def _device_constants(module: ModuleType) -> tuple[int, int, int]:
    return (
        int(getattr(module, "MBLTML_DEVICE_ARIES", 1)),
        int(getattr(module, "MBLTML_DEVICE_REGULUS", 2)),
        int(getattr(module, "MBLTML_DEVICE_REGULUS_USB", 4)),
    )


def _family_for_type(module: ModuleType, device_type: int) -> str:
    aries, regulus, regulus_usb = _device_constants(module)
    if device_type == aries:
        return "aries"
    if device_type in {regulus, regulus_usb}:
        return "regulus"
    return "unknown"


def _initialize(module: ModuleType, family: str) -> None:
    init_devices = getattr(module, "mbltmlInitDevices", None)
    if callable(init_devices):
        aries, regulus, regulus_usb = _device_constants(module)
        device_types = {aries} if family == "aries" else {regulus, regulus_usb}
        init_devices(device_types)
        return
    module.mbltmlInit()


class MobilintDeviceSession:
    def __init__(self, device_id: int, expected_family: str):
        if type(device_id) is not int or device_id < 0:
            raise ValueError("Mobilint device_id must be a non-negative integer.")
        normalized_family = str(expected_family).strip().lower()
        if normalized_family not in _FAMILIES:
            raise ValueError("expected_family must be 'aries' or 'regulus'.")
        self.device_id = device_id
        self.expected_family = normalized_family
        self._acquired = False
        self._info: MobilintDeviceInfo | None = None

    @property
    def module(self) -> ModuleType | None:
        with _LOCK:
            return _STATE.module if self._acquired else None

    @property
    def info(self) -> MobilintDeviceInfo | None:
        return self._info

    def acquire(self) -> MobilintDeviceInfo:
        with _LOCK:
            if self._acquired:
                return self._info
            if _STATE.cleanup_pending:
                raise RuntimeError(
                    "mbltml shutdown cleanup is incomplete; retry release() "
                    "on the owning Mobilint device session."
                )
            if _STATE.ref_count == 0:
                module = _load_mbltml()
                _initialize(module, self.expected_family)
                _STATE.module = module
                _STATE.family = self.expected_family
            elif _STATE.family != self.expected_family:
                raise RuntimeError(
                    "mbltml is already initialized for "
                    f"{str(_STATE.family).upper()}; cannot acquire "
                    f"{self.expected_family.upper()} concurrently."
                )
            module = _STATE.module
            _STATE.ref_count += 1
            self._acquired = True

        try:
            device_count = int(module.mbltmlGetDeviceCount())
            if self.device_id >= device_count:
                raise ValueError(
                    f"Mobilint device_id={self.device_id} is out of range "
                    f"for device_count={device_count}."
                )
            device_type = int(module.mbltmlGetDeviceType(self.device_id))
            actual_family = _family_for_type(module, device_type)
            if actual_family != self.expected_family:
                raise RuntimeError(
                    f"Mobilint target expected {self.expected_family.upper()} "
                    f"at device_id={self.device_id}, but detected "
                    f"{actual_family.upper()} (device_type={device_type})."
                )
            self._info = MobilintDeviceInfo(
                device_id=self.device_id,
                device_type=device_type,
                family=actual_family,
            )
            return self._info
        except BaseException:
            self.release()
            raise

    def release(self) -> None:
        with _LOCK:
            if not self._acquired:
                return
            if _STATE.ref_count > 1:
                self._acquired = False
                self._info = None
                _STATE.ref_count -= 1
                return

            module = _STATE.module
            _STATE.cleanup_pending = True
            if module is not None:
                module.mbltmlShutdown()

            self._acquired = False
            self._info = None
            _STATE.ref_count = 0
            _STATE.module = None
            _STATE.family = None
            _STATE.cleanup_pending = False
