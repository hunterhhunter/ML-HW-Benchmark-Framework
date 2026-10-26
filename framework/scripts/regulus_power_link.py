"""Script-only INA3221 power link between a Jetson logger and REGULUS.

The Jetson owns the I2C device and timestamps both power samples and phase
events. REGULUS only sends coarse control messages around the existing
framework run; no framework source file needs to change.
"""

from __future__ import annotations

import base64
import bisect
import csv
import hashlib
import json
import math
import re
import socket
import subprocess
import sys
import threading
import time
import zlib
from pathlib import Path
from statistics import median


_ID = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")
_MAX_MESSAGE = 4_000_000
_RUN_TIMEOUT_SEC = 14_400  # REGULUS System E2E can spend hours in CPU preprocessing.


def _send(stream, message: dict) -> None:
    stream.write((json.dumps(message, separators=(",", ":")) + "\n").encode())
    stream.flush()


def _receive(stream) -> dict:
    line = stream.readline(_MAX_MESSAGE + 1)
    if not line or len(line) > _MAX_MESSAGE:
        raise ConnectionError("power protocol message missing or too large")
    message = json.loads(line)
    if not isinstance(message, dict):
        raise ValueError("power protocol message must be an object")
    return message


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class InaWorkerSampler:
    """Reuse the existing INA3221 subprocess without importing its internals."""

    def __init__(self, worker_path: Path, *, bus=7, address=0x40,
                 shunt_ohms=0.05, max_current_a=3.0, timeout_sec=2.0):
        self.worker_path = Path(worker_path)
        self.bus = bus
        self.address = address
        self.shunt_ohms = shunt_ohms
        self.max_current_a = max_current_a
        self.timeout_sec = timeout_sec
        self._process = None

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("INA3221 worker is already running")
        self._process = subprocess.Popen(
            [sys.executable, "-u", str(self.worker_path),
             "--bus", str(self.bus), "--address", hex(self.address),
             "--shunt-ohms", str(self.shunt_ohms),
             "--max-current-a", str(self.max_current_a)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=0,
        )
        reply = self._reply()
        if reply.get("status") != "ready":
            self.close()
            raise RuntimeError(f"INA3221 unavailable: {reply.get('error_code')}")

    def _reply(self) -> dict:
        import select
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("INA3221 worker is not running")
        ready, _, _ = select.select([self._process.stdout], [], [], self.timeout_sec)
        if not ready:
            raise TimeoutError("INA3221 worker response timed out")
        return _receive(self._process.stdout)

    def __call__(self) -> tuple[float, float, float]:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("INA3221 worker is not running")
        self._process.stdin.write(b"sample\n")
        self._process.stdin.flush()
        reply = self._reply()
        if reply.get("status") != "ok":
            raise RuntimeError(f"INA3221 read failed: {reply.get('error_code')}")
        return float(reply["bus_v"]), float(reply["current_a"]), float(reply["power_w"])

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            try:
                assert process.stdin is not None
                process.stdin.write(b"stop\n")
                process.stdin.flush()
                process.wait(timeout=1)
            except (OSError, BrokenPipeError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()


class PowerServer:
    """Single-client TCP server; one independent INA sample loop per run."""

    def __init__(self, *, host: str, port: int, output_root: Path,
                 sample_interval_sec: float = 0.1, sample_fn=None,
                 max_current_a: float = 3.0,
                 allowed_client_ip: str | None = None,
                 min_bus_v: float = 10.0, max_bus_v: float = 14.0):
        if not 0 < sample_interval_sec <= 1:
            raise ValueError("sample interval must be in (0, 1] seconds")
        if not 0 < min_bus_v < max_bus_v <= 26:
            raise ValueError("invalid REGULUS bus-voltage limits")
        self.host = host
        self.port = port
        self.output_root = Path(output_root)
        self.sample_interval_sec = sample_interval_sec
        self.sample_fn = sample_fn
        self.max_current_a = max_current_a
        self.allowed_client_ip = allowed_client_ip
        self.min_bus_v = min_bus_v
        self.max_bus_v = max_bus_v
        self._listener = None
        self._thread = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("power server already started")
        self.output_root.mkdir(parents=True, exist_ok=True)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        listener.listen(1)
        listener.settimeout(0.2)
        self.port = listener.getsockname()[1]
        self._listener = listener
        self._stop.clear()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("power server did not stop")
            self._thread = None

    def serve_forever(self) -> None:
        self.start()
        try:
            while self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                assert self._listener is not None
                connection, peer = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                # Model/data setup and System E2E may both run for a long time.
                connection.settimeout(_RUN_TIMEOUT_SEC)
                if self.allowed_client_ip and peer[0] != self.allowed_client_ip:
                    with connection.makefile("wb") as writer:
                        _send(writer, {"type": "ERROR", "reason": "client address not allowed"})
                    continue
                self._handle(connection)

    def _handle(self, connection) -> None:
        reader = connection.makefile("rb")
        writer = connection.makefile("wb")
        sample_stop = threading.Event()
        sample_errors: list[str] = []
        latest_sample_ns = [0]
        sampler_thread = None
        sampler = self.sample_fn
        status = "failed"
        run_dir = None
        try:
            ready = _receive(reader)
            measurement_id = ready.get("measurement_id")
            baseline_sec = ready.get("baseline_sec")
            expected = ready.get("expected_samples")
            if (ready.get("type") != "READY" or not isinstance(measurement_id, str)
                    or not _ID.fullmatch(measurement_id)
                    or not isinstance(baseline_sec, (int, float))
                    or not 0 <= baseline_sec <= 3600
                    or type(expected) is not int or expected <= 0):
                raise ValueError("invalid READY request")
            run_dir = self.output_root / measurement_id
            run_dir.mkdir(exist_ok=False)
            sensor_path = run_dir / "sensor.csv"
            events_path = run_dir / "events.jsonl"
            if sampler is None:
                raise RuntimeError("sensor sampler not configured")
            if hasattr(sampler, "start"):
                sampler.start()
            sensor_file = sensor_path.open("w", newline="", encoding="utf-8")
            events_file = events_path.open("w", encoding="utf-8")
            csv_writer = csv.writer(sensor_file)
            csv_writer.writerow(["monotonic_ns", "bus_V", "current_A", "power_W"])
            sensor_file.flush()

            def record_event(name: str, timestamp_ns: int) -> None:
                events_file.write(json.dumps({"event": name, "monotonic_ns": timestamp_ns}) + "\n")
                events_file.flush()

            def sample_loop() -> None:
                next_due = time.monotonic()
                while not sample_stop.is_set():
                    try:
                        before = time.monotonic_ns()
                        bus_v, current_a, power_w = sampler()
                        after = time.monotonic_ns()
                        if (not all(math.isfinite(value) for value in (bus_v, current_a, power_w))
                                or not self.min_bus_v <= bus_v <= self.max_bus_v
                                or not 0 <= current_a <= self.max_current_a
                                or power_w < 0):
                            raise ValueError("sensor reading outside configured range")
                        csv_writer.writerow([(before + after) // 2, bus_v, current_a, power_w])
                        sensor_file.flush()
                        latest_sample_ns[0] = (before + after) // 2
                    except Exception as exc:
                        sample_errors.append(f"{type(exc).__name__}: {exc}")
                        sample_stop.set()
                        return
                    next_due += self.sample_interval_sec
                    sample_stop.wait(max(0, next_due - time.monotonic()))

            baseline_start = time.monotonic_ns()
            record_event("baseline_start", baseline_start)
            sampler_thread = threading.Thread(target=sample_loop, daemon=True)
            sampler_thread.start()
            sample_stop.wait(float(baseline_sec))
            if sample_errors:
                raise RuntimeError(sample_errors[0])
            boundary_ns = time.monotonic_ns()
            record_event("baseline_end", boundary_ns)
            _send(writer, {"type": "BASELINE_DONE", "measurement_id": measurement_id})

            begin = _receive(reader)
            if begin.get("type") == "ABORT" and begin.get("measurement_id") == measurement_id:
                (run_dir / "failure.txt").write_text("aborted before inference\n")
                _send(writer, {"type": "ABORT_ACK", "measurement_id": measurement_id})
                return
            if begin.get("type") != "START" or begin.get("measurement_id") != measurement_id:
                raise ValueError("invalid START request")
            if sample_errors:
                raise RuntimeError(sample_errors[0])
            record_event("inference_start", time.monotonic_ns())
            _send(writer, {"type": "GO", "measurement_id": measurement_id})

            done = _receive(reader)
            inference_end = time.monotonic_ns()
            if done.get("type") != "DONE" or done.get("measurement_id") != measurement_id:
                raise ValueError("invalid DONE request")
            record_event("inference_end", inference_end)
            _send(writer, {"type": "DONE_ACK", "measurement_id": measurement_id})

            result = _receive(reader)
            completed = result.get("completed_samples")
            if result.get("type") != "RESULT" or result.get("measurement_id") != measurement_id:
                raise ValueError("invalid RESULT request")
            deadline = time.monotonic() + max(1.0, 3 * self.sample_interval_sec)
            while latest_sample_ns[0] <= inference_end and not sample_errors and time.monotonic() < deadline:
                time.sleep(min(0.01, self.sample_interval_sec / 2))
            sample_stop.set()
            sampler_thread.join(timeout=5)
            if sampler_thread.is_alive():
                raise RuntimeError("sensor sampler did not stop")
            sensor_file.close()
            events_file.close()
            status = (
                "complete" if not sample_errors and latest_sample_ns[0] > inference_end
                and result.get("run_status") == "valid"
                and type(completed) is int and completed == expected else "partial"
            )
            payloads = {}
            hashes = {}
            for name, path in (("sensor", sensor_path), ("events", events_path)):
                data = path.read_bytes()
                payloads[name] = base64.b64encode(zlib.compress(data)).decode("ascii")
                hashes[name] = _sha256(data)
            manifest = {
                "schema_version": 1,
                "measurement_id": measurement_id,
                "model": ready.get("model"),
                "mode": ready.get("mode"),
                "expected_samples": expected,
                "completed_samples": completed,
                "power_scope": "regulus_dc_input",
                "sample_interval_sec": self.sample_interval_sec,
                "baseline_sec": baseline_sec,
                "status": status,
                "sample_errors": sample_errors,
                "sha256": hashes,
            }
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            _send(writer, {"type": "ARCHIVE", "manifest": manifest, "payloads": payloads})
        except Exception as exc:
            try:
                _send(writer, {"type": "ERROR", "reason": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass
            if run_dir is not None:
                (run_dir / "failure.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        finally:
            sample_stop.set()
            if sampler_thread is not None:
                sampler_thread.join(timeout=5)
            if sampler is not None and hasattr(sampler, "close"):
                sampler.close()
            for handle in (locals().get("sensor_file"), locals().get("events_file")):
                if handle is not None and not handle.closed:
                    handle.close()
            reader.close()
            writer.close()


class RemotePowerSession:
    """Monitor-shaped client that brackets the unmodified framework run."""

    def __init__(self, *, host: str, port: int, measurement_id: str,
                 baseline_sec: float, output_dir: Path):
        if not _ID.fullmatch(measurement_id):
            raise ValueError("invalid measurement ID")
        self.host = host
        self.port = port
        self.measurement_id = measurement_id
        self.baseline_sec = baseline_sec
        self.output_dir = Path(output_dir)
        self._socket = None
        self._reader = None
        self._writer = None
        self._prepared = False
        self._started = False
        self._stopped = False
        self.local_start_ns = None
        self.local_end_ns = None

    def startup_timeout_hint_sec(self) -> float:
        return 0.0

    def record_after_load_vram(self) -> None:
        pass

    def summary(self) -> dict:
        return {}

    def prepare(self, *, model: str, mode: str, expected_samples: int) -> None:
        if self._prepared or self._socket is not None:
            raise RuntimeError("power session already started")
        sock = socket.create_connection((self.host, self.port), timeout=10)
        sock.settimeout(max(30, self.baseline_sec + 30))
        self._socket = sock
        self._reader = sock.makefile("rb")
        self._writer = sock.makefile("wb")
        try:
            _send(self._writer, {
                "type": "READY", "measurement_id": self.measurement_id,
                "baseline_sec": self.baseline_sec, "model": model,
                "mode": mode, "expected_samples": expected_samples,
            })
            reply = _receive(self._reader)
            if reply.get("type") != "BASELINE_DONE" or reply.get("measurement_id") != self.measurement_id:
                raise RuntimeError(f"Jetson power server rejected READY: {reply}")
        except BaseException:
            self.close()
            raise
        sock.settimeout(_RUN_TIMEOUT_SEC)
        self._prepared = True

    def start(self, *, model: str = "unknown", mode: str = "e2e",
              expected_samples: int = 5000) -> None:
        if not self._prepared:
            self.prepare(model=model, mode=mode, expected_samples=expected_samples)
        if self._started:
            raise RuntimeError("power session already started")
        _send(self._writer, {"type": "START", "measurement_id": self.measurement_id})
        reply = _receive(self._reader)
        if reply.get("type") != "GO" or reply.get("measurement_id") != self.measurement_id:
            raise RuntimeError(f"Jetson power server rejected START: {reply}")
        self.local_start_ns = time.monotonic_ns()
        self._started = True

    def abort(self) -> None:
        try:
            if self._prepared and not self._started and self._socket is not None:
                _send(self._writer, {"type": "ABORT", "measurement_id": self.measurement_id})
                reply = _receive(self._reader)
                if reply.get("type") != "ABORT_ACK":
                    raise RuntimeError(f"Jetson power server rejected ABORT: {reply}")
        finally:
            self.close()

    def stop(self) -> None:
        if not self._started or self._stopped:
            return
        self.local_end_ns = time.monotonic_ns()
        _send(self._writer, {"type": "DONE", "measurement_id": self.measurement_id})
        reply = _receive(self._reader)
        if reply.get("type") != "DONE_ACK" or reply.get("measurement_id") != self.measurement_id:
            raise RuntimeError(f"Jetson power server rejected DONE: {reply}")
        self._stopped = True

    def finalize(self, *, completed_samples: int, run_status: str) -> dict:
        if not self._stopped:
            raise RuntimeError("cannot finalize an unclosed power interval")
        try:
            _send(self._writer, {
                "type": "RESULT", "measurement_id": self.measurement_id,
                "completed_samples": completed_samples, "run_status": run_status,
            })
            reply = _receive(self._reader)
            if reply.get("type") != "ARCHIVE":
                raise RuntimeError(f"Jetson power archive failed: {reply}")
            manifest = reply["manifest"]
            if manifest.get("measurement_id") != self.measurement_id:
                raise ValueError("power archive belongs to another run")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            paths = {}
            for name, filename in (("sensor", "sensor.csv"), ("events", "events.jsonl")):
                data = zlib.decompress(base64.b64decode(reply["payloads"][name], validate=True))
                if _sha256(data) != manifest["sha256"][name]:
                    raise ValueError(f"power archive checksum mismatch: {name}")
                path = self.output_dir / filename
                path.write_bytes(data)
                paths[f"{name}_path"] = str(path)
            local_duration_s = (self.local_end_ns - self.local_start_ns) / 1e9
            manifest["regulus_local_duration_s"] = local_duration_s
            manifest.update(paths)
            if manifest["status"] == "complete":
                try:
                    manifest["analysis"] = analyze_power_run(
                        paths["sensor_path"], paths["events_path"],
                        completed_samples=completed_samples,
                        max_sample_gap_sec=max(0.4, 4 * manifest["sample_interval_sec"]),
                    )
                except ValueError as exc:
                    manifest["status"] = "partial"
                    manifest["analysis_error"] = str(exc)
                else:
                    manifest["control_boundary_overhead_s"] = (
                        manifest["analysis"]["inference_duration_s"] - local_duration_s
                    )
            (self.output_dir / "power-summary.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            return manifest
        finally:
            self.close()

    def close(self) -> None:
        for handle in (self._reader, self._writer):
            if handle is not None:
                handle.close()
        if self._socket is not None:
            self._socket.close()
        self._socket = self._reader = self._writer = None
        self._prepared = False


def analyze_power_run(sensor_path, events_path, *, completed_samples: int,
                      max_sample_gap_sec: float = 0.4) -> dict:
    """Integrate system input power on the Jetson's monotonic time axis."""
    if type(completed_samples) is not int or completed_samples <= 0:
        raise ValueError("completed sample count must be positive")
    with Path(sensor_path).open(newline="", encoding="utf-8") as handle:
        samples = [(int(row["monotonic_ns"]), float(row["power_W"]))
                   for row in csv.DictReader(handle)]
    events = [json.loads(line) for line in Path(events_path).read_text().splitlines()]
    names = [event.get("event") for event in events]
    if names != ["baseline_start", "baseline_end", "inference_start", "inference_end"]:
        raise ValueError("incomplete or unordered power phase markers")
    boundaries = [int(event["monotonic_ns"]) for event in events]
    if any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("power phase markers are not increasing")
    if len(samples) < 4 or any(a[0] >= b[0] for a, b in zip(samples, samples[1:])):
        raise ValueError("power samples are missing or unordered")
    if any(not math.isfinite(power) or power < 0 for _, power in samples):
        raise ValueError("power sample is invalid")
    idle = [power for timestamp, power in samples if boundaries[0] <= timestamp <= boundaries[1]]
    if len(idle) < 2:
        raise ValueError("idle baseline has too few samples")
    timestamps = [timestamp for timestamp, _ in samples]
    max_gap_ns = int(max_sample_gap_sec * 1e9)
    start, end = boundaries[2], boundaries[3]
    if timestamps[0] > start or timestamps[-1] < end:
        raise ValueError("power samples do not bracket inference interval")

    def power_at(timestamp):
        index = bisect.bisect_left(timestamps, timestamp)
        if index < len(samples) and timestamps[index] == timestamp:
            return samples[index][1]
        left_time, left_power = samples[index - 1]
        right_time, right_power = samples[index]
        if right_time - left_time > max_gap_ns:
            raise ValueError("power sample gap exceeds limit")
        ratio = (timestamp - left_time) / (right_time - left_time)
        return left_power + (right_power - left_power) * ratio

    points = [(start, power_at(start))]
    points.extend((timestamp, power) for timestamp, power in samples if start < timestamp < end)
    points.append((end, power_at(end)))
    if any(right[0] - left[0] > max_gap_ns for left, right in zip(points, points[1:])):
        raise ValueError("power sample gap exceeds limit")
    duration_s = (end - start) / 1e9
    energy_j = sum((right_t - left_t) / 1e9 * (left_w + right_w) / 2
                   for (left_t, left_w), (right_t, right_w) in zip(points, points[1:]))
    idle_w = median(idle)
    incremental_j = energy_j - idle_w * duration_s
    return {
        "power_scope": "regulus_dc_input",
        "completed_samples": completed_samples,
        "baseline_duration_s": (boundaries[1] - boundaries[0]) / 1e9,
        "inference_duration_s": duration_s,
        "idle_median_w": idle_w,
        "inference_mean_w": energy_j / duration_s,
        "incremental_mean_w": incremental_j / duration_s,
        "system_energy_j": energy_j,
        "incremental_energy_j": incremental_j,
        "system_j_per_sample": energy_j / completed_samples,
        "incremental_j_per_sample": incremental_j / completed_samples,
    }
