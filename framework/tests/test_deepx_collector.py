import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from monitors.deepx_collector import DeepXCollector


class _FakeStatus:
    def __init__(self, device_id):
        self.device_id = device_id

    def get_temperature(self, channel):
        if channel > 2:
            raise RuntimeError("invalid core")
        return 40 + self.device_id * 10 + channel

    def get_npu_voltage(self, channel):
        if channel > 2:
            raise RuntimeError("invalid core")
        return 750 + channel * 5

    def get_npu_clock(self, channel):
        if channel > 2:
            raise RuntimeError("invalid core")
        return 1000 + self.device_id * 100 + channel * 10


class _FakeDeviceStatus:
    @classmethod
    def get_device_count(cls):
        return 2

    @classmethod
    def get_current_status(cls, device_id):
        return _FakeStatus(device_id)


def _install_fake_dx_engine(monkeypatch, device_status_cls=_FakeDeviceStatus):
    dx_engine = types.ModuleType("dx_engine")
    dx_engine.__path__ = []
    dev_status = types.ModuleType("dx_engine.dev_status")
    dev_status.DeviceStatus = device_status_cls
    monkeypatch.setitem(sys.modules, "dx_engine", dx_engine)
    monkeypatch.setitem(sys.modules, "dx_engine.dev_status", dev_status)


def test_deepx_collector_reports_dev_status_metrics(monkeypatch):
    _install_fake_dx_engine(monkeypatch)
    collector = DeepXCollector(device_id="all")

    assert collector.is_available()

    collector.start()
    metrics = collector.collect()
    static_info = collector.get_static_info()
    collector.stop()

    assert metrics["hw_accel_temp_c"] == 52.0
    assert metrics["hw_accel_voltage_mv"] == 755.0
    assert metrics["hw_accel_clock_mhz"] == 1060.0
    assert static_info["hw_accel_vendor"] == "DEEPX"
    assert static_info["hw_accel_device_id"] == "0,1"
    assert static_info["hw_accel_device_count"] == 2
    assert static_info["hw_accel_core_channels"] == "0,1,2"


def test_deepx_collector_can_select_npu_style_device_id(monkeypatch):
    _install_fake_dx_engine(monkeypatch)
    collector = DeepXCollector(device_id="npu1", core_count=1)

    collector.start()
    metrics = collector.collect()
    static_info = collector.get_static_info()
    collector.stop()

    assert metrics["hw_accel_temp_c"] == 50.0
    assert metrics["hw_accel_clock_mhz"] == 1100.0
    assert static_info["hw_accel_device_id"] == "1"
