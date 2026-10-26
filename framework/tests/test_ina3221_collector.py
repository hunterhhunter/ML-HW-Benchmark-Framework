"""The INA3221 reader must run outside the benchmark process."""

import os
from pathlib import Path
import time

import pytest


@pytest.fixture
def fake_i2c(tmp_path, monkeypatch):
    (tmp_path / "smbus2.py").write_text(
        """
import os
from pathlib import Path
import time

class SMBus:
    def __init__(self, bus):
        assert bus == 7
        self.config = 0x2771
        Path(os.environ["INA_TEST_PID_FILE"]).write_text(str(os.getpid()))

    def read_word_data(self, address, register):
        assert address == 0x40
        if register == 0x01:
            time.sleep(float(os.getenv("INA_TEST_SHUNT_DELAY_SEC", "0")))
        words = {
            0x00: self.config,
            0xFE: 0 if os.getenv("INA_TEST_BAD_ID") else 0x4954,
            0xFF: 0x2032,
            0x01: int(os.getenv("INA_TEST_SHUNT_WORD", "0x400B"), 0),
            0x02: 0x004B,
        }
        return words[register]

    def write_word_data(self, address, register, value):
        assert address == 0x40 and register == 0x00
        assert value == 0x2741
        self.config = value

    def close(self):
        pass
""",
        encoding="utf-8",
    )
    pid_file = tmp_path / "worker.pid"
    monkeypatch.setenv("INA_TEST_PID_FILE", str(pid_file))
    monkeypatch.setenv(
        "PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    return pid_file


def test_collector_samples_in_child_process_and_labels_jetson_dc_input(fake_i2c):
    from monitors.ina3221_collector import Ina3221Collector

    collector = Ina3221Collector()
    collector.start()
    try:
        reading = collector.collect_power()
        assert reading.status == "ok"
        assert reading.power_w == pytest.approx(5.5296)
        assert int(Path(fake_i2c).read_text()) != os.getpid()
        assert collector.power_trace_source().power_scope == "jetson_dc_input"
        assert collector.power_trace_source().device_id == "i2c-7@0x40/ch1/shunt0.05ohm"
    finally:
        collector.stop()


def test_bad_sensor_id_is_unavailable_not_zero_power(fake_i2c, monkeypatch):
    from monitors.ina3221_collector import Ina3221Collector

    monkeypatch.setenv("INA_TEST_BAD_ID", "1")
    collector = Ina3221Collector()
    collector.start()
    try:
        reading = collector.collect_power()
        assert reading.status == "unavailable"
        assert reading.power_w is None
        assert reading.error_code == "sensor_id_mismatch"
        assert collector.power_trace_source() is None
    finally:
        collector.stop()


def test_overcurrent_reports_failed_reading_and_stops_child(fake_i2c, monkeypatch):
    from monitors.ina3221_collector import Ina3221Collector

    monkeypatch.setenv("INA_TEST_SHUNT_WORD", "0x1078")
    collector = Ina3221Collector()
    collector.start()
    try:
        reading = collector.collect_power()
        assert reading.status == "read_error"
        assert reading.error_code == "overcurrent"
        assert reading.power_w is None
    finally:
        collector.stop()


def test_unresponsive_i2c_read_reaps_child_and_reports_error(fake_i2c, monkeypatch):
    from monitors.ina3221_collector import Ina3221Collector

    monkeypatch.setenv("INA_TEST_SHUNT_DELAY_SEC", "3")
    collector = Ina3221Collector(response_timeout_sec=0.2)
    collector.start()
    child = collector._process
    assert child is not None

    try:
        reading = collector.collect_power()
        assert reading.status == "read_error"
        assert reading.error_code == "worker_io_failed"
        assert reading.power_w is None
        assert child.poll() is not None
    finally:
        collector.stop()


def test_hailo_trace_uses_child_ina_readings_and_keeps_run_link(fake_i2c, monkeypatch, tmp_path):
    from core.result_store import reserve_run_artifacts
    from core.targets import get_target
    from monitors import create_hw_monitor
    from monitors.ina3221_collector import Ina3221Collector

    monkeypatch.setattr(Ina3221Collector, "is_available", lambda self: True)
    target = get_target("hailo10h")
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv", run_id="jetson-ina-smoke"
    )
    monitor = create_hw_monitor(
        collector_names=list(target.monitor_names),
        collector_options=target.monitor_options,
        summary_enabled=False,
        power_trace_enabled=True,
        power_trace_baseline_sec=0.02,
    )
    monitor.configure_power_trace(reservation=reservation, target_id=target.target_id)
    monitor.start()
    try:
        time.sleep(0.03)
    finally:
        monitor.stop()

    metadata = monitor.power_trace_metadata()
    assert metadata["power_trace_status"] == "complete"
    assert metadata["power_scope"] == "jetson_dc_input"
    trace = tmp_path / metadata["power_trace_path"]
    assert trace.exists()
    assert "jetson_dc_input" in trace.read_text()
    assert "jetson-ina-smoke" in trace.read_text()


def test_missing_sensor_skips_idle_wait_and_marks_trace_unavailable(
    fake_i2c, monkeypatch, tmp_path
):
    from core.result_store import reserve_run_artifacts
    from monitors import create_hw_monitor
    from monitors.ina3221_collector import Ina3221Collector

    monkeypatch.setenv("INA_TEST_BAD_ID", "1")
    monkeypatch.setattr(Ina3221Collector, "is_available", lambda self: True)
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv", run_id="missing-sensor"
    )
    monitor = create_hw_monitor(
        collector_names=["ina3221"],
        summary_enabled=False,
        power_trace_enabled=True,
        power_trace_baseline_sec=10.0,
    )
    monitor.configure_power_trace(reservation=reservation, target_id="hailo10h")
    started = time.monotonic()
    monitor.start()
    monitor.stop()

    assert time.monotonic() - started < 1.0
    assert monitor.power_trace_metadata()["power_trace_status"] == "unavailable"


def test_missing_sensor_skips_preload_idle_wait(fake_i2c, monkeypatch, tmp_path):
    from core.result_store import reserve_run_artifacts
    from monitors import create_hw_monitor
    from monitors.ina3221_collector import Ina3221Collector

    monkeypatch.setenv("INA_TEST_BAD_ID", "1")
    monkeypatch.setattr(Ina3221Collector, "is_available", lambda self: True)
    reservation = reserve_run_artifacts(results_path=tmp_path / "results.csv")
    monitor = create_hw_monitor(
        collector_names=["ina3221"],
        summary_enabled=False,
        power_trace_enabled=True,
        power_trace_baseline_sec=60.0,
    )
    monitor.configure_power_trace(reservation=reservation, target_id="hailo8")
    started = time.monotonic()
    monitor.start_preload_baseline()
    monitor.begin_inference()
    monitor.stop()

    assert time.monotonic() - started < 1.0
    assert monitor.power_trace_metadata()["power_trace_status"] == "unavailable"
    assert monitor.power_trace_metadata()["power_baseline_state"] == ""
