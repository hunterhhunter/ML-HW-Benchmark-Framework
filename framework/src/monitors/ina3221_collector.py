"""Jetson DC input power from an INA3221 reader subprocess."""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import time
from typing import Optional

from core.power_trace import PowerReading, PowerTraceSource

from .base import Collector


class Ina3221Collector(Collector):
    """Poll a dedicated child that owns the I2C handle, never the benchmark."""

    def __init__(
        self,
        bus: int = 7,
        address: int = 0x40,
        shunt_ohms: float = 0.05,
        max_current_a: float = 3.0,
        power_sample_interval_sec: float = 0.1,
        response_timeout_sec: float = 1.0,
    ):
        if type(bus) is not int or bus < 0:
            raise ValueError("bus must be a non-negative integer")
        if type(address) is not int or not 0x03 <= address <= 0x77:
            raise ValueError("address must be a valid 7-bit I2C address")
        for name, value in (
            ("shunt_ohms", shunt_ohms),
            ("max_current_a", max_current_a),
            ("power_sample_interval_sec", power_sample_interval_sec),
            ("response_timeout_sec", response_timeout_sec),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        self.bus = bus
        self.address = address
        self.shunt_ohms = float(shunt_ohms)
        self.max_current_a = float(max_current_a)
        self.power_sample_interval_sec = float(power_sample_interval_sec)
        self.response_timeout_sec = float(response_timeout_sec)
        self._process: subprocess.Popen | None = None
        self._buffer = b""
        self._unavailable_error = "not_started"

    def is_available(self) -> bool:
        return (
            Path(f"/dev/i2c-{self.bus}").exists()
            and importlib.util.find_spec("smbus2") is not None
        )

    def power_trace_source(self) -> PowerTraceSource | None:
        if self._process is None or self._process.poll() is not None:
            return None
        return PowerTraceSource(
            collector="ina3221",
            monitor_source="ina3221-smbus2-subprocess",
            device_id=(
                f"i2c-{self.bus}@0x{self.address:02x}/ch1/"
                f"shunt{self.shunt_ohms:g}ohm"
            ),
            power_scope="jetson_dc_input",
            sample_interval_sec=self.power_sample_interval_sec,
        )

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("INA3221 reader is already started")
        self._buffer = b""
        self._unavailable_error = "worker_start_failed"
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).with_name("ina3221_worker.py")),
            "--bus",
            str(self.bus),
            "--address",
            hex(self.address),
            "--shunt-ohms",
            str(self.shunt_ohms),
            "--max-current-a",
            str(self.max_current_a),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            reply = self._receive()
            if reply.get("status") == "ready":
                self._unavailable_error = ""
                return
            self._unavailable_error = self._safe_error_code(
                reply.get("error_code"), "worker_start_failed"
            )
        except (OSError, ValueError, TimeoutError, EOFError):
            self._unavailable_error = "worker_start_failed"
        self._close_process()
        print(f"[INA3221] reader unavailable: {self._unavailable_error}", file=sys.stderr)

    def collect(self) -> dict[str, Optional[float]]:
        # The raw-power poller is the only reader of the subprocess pipe.
        return {}

    def collect_power(self) -> PowerReading:
        process = self._process
        if process is None or process.poll() is not None:
            return PowerReading(
                status="unavailable",
                error_code=self._unavailable_error or "worker_exited",
            )
        try:
            assert process.stdin is not None
            os.write(process.stdin.fileno(), b"sample\n")
            reply = self._receive()
            status = reply.get("status")
            if status == "ok":
                return PowerReading(status="ok", power_w=reply["power_w"])
            error_code = self._safe_error_code(
                reply.get("error_code"), "worker_read_failed"
            )
            if error_code in {"negative_current", "overcurrent"}:
                self._unavailable_error = error_code
            return PowerReading(status="read_error", error_code=error_code)
        except (OSError, ValueError, KeyError, TimeoutError, EOFError):
            self._unavailable_error = "worker_io_failed"
            self._close_process()
            return PowerReading(status="read_error", error_code="worker_io_failed")

    def stop(self) -> None:
        self._close_process()

    def _receive(self) -> dict:
        process = self._process
        if process is None or process.stdout is None:
            raise EOFError("worker stdout is unavailable")
        deadline = time.monotonic() + self.response_timeout_sec
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("INA3221 worker response timed out")
            readable, _, _ = select.select([process.stdout], [], [], remaining)
            if not readable:
                raise TimeoutError("INA3221 worker response timed out")
            chunk = os.read(process.stdout.fileno(), 4096)
            if not chunk:
                raise EOFError("INA3221 worker exited")
            self._buffer += chunk
            if len(self._buffer) > 65536:
                raise ValueError("INA3221 worker reply too large")
        line, self._buffer = self._buffer.split(b"\n", 1)
        reply = json.loads(line)
        if not isinstance(reply, dict):
            raise ValueError("INA3221 worker reply is not an object")
        return reply

    def _close_process(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            try:
                assert process.stdin is not None
                os.write(process.stdin.fileno(), b"stop\n")
                process.wait(timeout=0.5)
            except (OSError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=0.5)
        else:
            process.wait()
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        self._buffer = b""

    @staticmethod
    def _safe_error_code(value, fallback: str) -> str:
        if isinstance(value, str) and value and len(value) <= 128 and all(
            character.isascii() and (character.isalnum() or character in "_:-")
            for character in value
        ):
            return value
        return fallback
