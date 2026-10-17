"""Isolated INA3221 I2C reader for Jetson DC input power.

The parent sends one ``sample`` line per reading. Each reply is one JSON line.
Only this child imports smbus2 and opens the I2C device.
"""

from __future__ import annotations

import argparse
import json
import sys
import time


_CHANNEL_ONE_CONTINUOUS_CONFIG = 0x4127


def _register(bus, address: int, register: int) -> int:
    word = bus.read_word_data(address, register)
    return ((word & 0xFF) << 8) | (word >> 8)


def verify_identity(bus, address: int) -> None:
    if (_register(bus, address, 0xFE), _register(bus, address, 0xFF)) != (
        0x5449,
        0x3220,
    ):
        raise ValueError("sensor_id_mismatch")


def configure_sensor(bus, address: int) -> None:
    """Use channel 1, 1.1-ms bus/shunt conversions, no averaging."""
    value = _CHANNEL_ONE_CONTINUOUS_CONFIG
    bus.write_word_data(address, 0x00, ((value & 0xFF) << 8) | (value >> 8))
    if _register(bus, address, 0x00) != value:
        raise ValueError("sensor_config_failed")


def read_sample(
    bus, address: int, *, shunt_ohms: float, max_current_a: float
) -> dict[str, float]:
    shunt_bits = _register(bus, address, 0x01) >> 3
    if shunt_bits & 0x1000:
        shunt_bits -= 0x2000
    current_a = (shunt_bits * 40e-6) / shunt_ohms
    if current_a < 0:
        raise ValueError("negative_current")
    if current_a > max_current_a:
        raise ValueError("overcurrent")

    bus_v = (_register(bus, address, 0x02) >> 3) * 0.008
    if bus_v <= 0 or bus_v > 26:
        raise ValueError("invalid_bus_voltage")
    return {
        "bus_v": bus_v,
        "current_a": current_a,
        "power_w": bus_v * current_a,
    }


def _reply(**fields) -> None:
    print(json.dumps(fields, separators=(",", ":")), flush=True)


def _error_code(exc: Exception) -> str:
    if isinstance(exc, ValueError) and str(exc) in {
        "sensor_id_mismatch",
        "sensor_config_failed",
        "negative_current",
        "overcurrent",
        "invalid_bus_voltage",
    }:
        return str(exc)
    return type(exc).__name__


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bus", type=int, required=True)
    parser.add_argument("--address", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--shunt-ohms", type=float, required=True)
    parser.add_argument("--max-current-a", type=float, required=True)
    args = parser.parse_args()

    if args.bus < 0 or not 0x03 <= args.address <= 0x77:
        parser.error("invalid I2C bus or address")
    if args.shunt_ohms <= 0 or args.max_current_a <= 0:
        parser.error("shunt and current limit must be positive")

    bus = None
    try:
        from smbus2 import SMBus

        bus = SMBus(args.bus)
        verify_identity(bus, args.address)
        configure_sensor(bus, args.address)
        time.sleep(0.01)
    except Exception as exc:
        _reply(status="unavailable", error_code=_error_code(exc))
        if bus is not None:
            bus.close()
        return 1

    _reply(status="ready")
    try:
        for command in sys.stdin:
            if command.strip() == "stop":
                break
            if command.strip() != "sample":
                _reply(status="read_error", error_code="invalid_command")
                continue
            try:
                sample = read_sample(
                    bus,
                    args.address,
                    shunt_ohms=args.shunt_ohms,
                    max_current_a=args.max_current_a,
                )
            except Exception as exc:
                code = _error_code(exc)
                _reply(status="read_error", error_code=code)
                if code in {"negative_current", "overcurrent"}:
                    break
            else:
                _reply(status="ok", **sample)
    finally:
        bus.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
