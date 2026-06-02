import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from monitors.hailo_collector import HailoCollector


class _Temperature:
    ts0_temperature = 42.5


class _Power:
    average_value = 3.25


class _FakeControl:
    def __init__(self, power_supported=True):
        self.power_supported = power_supported
        self.started = False
        self.stopped = False

    def stop_power_measurement(self):
        self.stopped = True

    def set_power_measurement(self):
        if not self.power_supported:
            raise RuntimeError("CONTROL_PROTOCOL_STATUS_UNSUPPORTED_DEVICE")

    def start_power_measurement(self):
        if not self.power_supported:
            raise RuntimeError("power not supported")
        self.started = True

    def get_power_measurement(self):
        return _Power()

    def get_chip_temperature(self):
        return _Temperature()


class _FakeDevice:
    control = None

    @staticmethod
    def scan():
        return ["device0"]

    def __init__(self, _device_info=None):
        self.control = type(self).control


def _fake_hailo(power_supported=True):
    _FakeDevice.control = _FakeControl(power_supported=power_supported)
    return SimpleNamespace(Device=_FakeDevice)


def test_hailo_collector_reports_power_and_temperature(monkeypatch):
    collector = HailoCollector()
    monkeypatch.setattr(collector, "_import_hailo_platform", lambda: _fake_hailo(power_supported=True))

    collector.start()
    metrics = collector.collect()
    collector.stop()

    assert metrics["hw_accel_temp_c"] == 42.5
    assert metrics["hw_accel_power_w"] == 3.25
    assert collector.get_static_info()["hw_accel_vendor"] == "Hailo"


def test_hailo_collector_falls_back_to_temperature_only(monkeypatch):
    collector = HailoCollector()
    monkeypatch.setattr(collector, "_import_hailo_platform", lambda: _fake_hailo(power_supported=False))

    collector.start()
    metrics = collector.collect()
    static_info = collector.get_static_info()
    collector.stop()

    assert metrics["hw_accel_temp_c"] == 42.5
    assert "hw_accel_power_w" not in metrics
    assert "temperature-only" in static_info["hw_accel_monitor_note"]


def test_hailo_collector_can_disable_power_probe(monkeypatch):
    collector = HailoCollector(enable_power=False)
    monkeypatch.setattr(collector, "_import_hailo_platform", lambda: _fake_hailo(power_supported=False))

    collector.start()
    metrics = collector.collect()
    collector.stop()

    assert metrics["hw_accel_temp_c"] == 42.5
    assert "hw_accel_power_w" not in metrics
