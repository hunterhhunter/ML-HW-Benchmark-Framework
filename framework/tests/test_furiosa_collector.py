"""Furiosa RNGD raw power collector contract tests."""

import math
import threading
from queue import Queue
from types import SimpleNamespace

import pytest

import monitors.furiosa_collector as furiosa_collector_module
from monitors.furiosa_collector import FuriosaCollector


class FakeDeviceInfo:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


class FakeDevice:
    def __init__(self, name, power_actions=(39.36,)):
        self._info = FakeDeviceInfo(name)
        self._power_actions = list(power_actions)
        self.power_calls = 0

    def device_info(self):
        return self._info

    def power_consumption(self):
        self.power_calls += 1
        action = self._power_actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        return action


class FakeSmi:
    def __init__(self, devices):
        self.devices = list(devices)
        self.init_calls = 0
        self.list_calls = 0

    def init(self):
        self.init_calls += 1

    def list_devices(self):
        self.list_calls += 1
        return list(self.devices)


class ThreadLocalFakeSmi:
    def __init__(self, power_w=39.36):
        self.power_w = power_w
        self.init_calls = 0
        self.list_calls = 0

    def init(self):
        self.init_calls += 1

    def list_devices(self):
        self.list_calls += 1
        owner_thread_id = threading.get_ident()

        class ThreadBoundDevice(FakeDevice):
            def power_consumption(inner_self):
                if threading.get_ident() != owner_thread_id:
                    raise RuntimeError("device crossed a thread boundary")
                return self.power_w

        return [ThreadBoundDevice("npu0")]


def install_fake(monkeypatch, module):
    monkeypatch.setattr(
        furiosa_collector_module,
        "import_module",
        lambda name: module,
    )


def test_furiosa_selects_exact_npu_name_after_init_and_list_devices(monkeypatch):
    other = FakeDevice("npu1")
    selected = FakeDevice("npu0")
    smi = FakeSmi([other, selected])
    install_fake(monkeypatch, smi)
    collector = FuriosaCollector(device_name="npu0")

    collector.start()
    reading = collector.collect_power()

    assert smi.init_calls == 1
    assert selected.power_calls == 1
    assert other.power_calls == 0
    assert reading.power_w == 39.36


def test_furiosa_collect_power_calls_device_power_consumption(monkeypatch):
    device = FakeDevice("npu0", power_actions=(40.32, 41.28))
    install_fake(monkeypatch, FakeSmi([device]))
    collector = FuriosaCollector()
    collector.start()

    assert collector.collect_power().power_w == 40.32
    assert collector.collect_power().power_w == 41.28
    assert device.power_calls == 2


def test_furiosa_resolves_device_in_the_power_calling_thread(monkeypatch):
    smi = ThreadLocalFakeSmi(power_w=39.36)
    install_fake(monkeypatch, smi)
    collector = FuriosaCollector()
    collector.start()
    results = Queue()

    worker = threading.Thread(
        target=lambda: results.put(collector.collect_power())
    )
    worker.start()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    reading = results.get_nowait()
    assert reading.status == "ok"
    assert reading.power_w == 39.36


@pytest.mark.parametrize("value", [-0.1, math.nan, math.inf, -math.inf])
def test_furiosa_rejects_non_finite_or_negative_power(monkeypatch, value):
    install_fake(monkeypatch, FakeSmi([FakeDevice("npu0", (value,))]))
    collector = FuriosaCollector()
    collector.start()

    reading = collector.collect_power()

    assert reading.status == "read_error"
    assert reading.power_w is None


@pytest.mark.parametrize(
    "module",
    [
        None,
        SimpleNamespace(init=lambda: None),
        SimpleNamespace(init=lambda: None, list_devices=lambda: [object()]),
    ],
)
def test_furiosa_missing_package_or_method_is_unavailable(monkeypatch, module):
    if module is None:
        def missing(name):
            raise ModuleNotFoundError(name)

        monkeypatch.setattr(furiosa_collector_module, "import_module", missing)
    else:
        install_fake(monkeypatch, module)
    collector = FuriosaCollector()

    collector.start()
    reading = collector.collect_power()

    assert reading.status == "unavailable"
    assert reading.power_w is None


def test_furiosa_source_is_device_at_two_hundred_ms():
    source = FuriosaCollector(device_name="npu0").power_trace_source()

    assert source.collector == "furiosa"
    assert source.monitor_source == "furiosa-smi-py"
    assert source.device_id == "npu0"
    assert source.power_scope == "device"
    assert source.sample_interval_sec == 0.2


def test_furiosa_does_not_spawn_or_parse_cli_json_or_table(
    monkeypatch,
):
    device = FakeDevice("npu0")
    install_fake(monkeypatch, FakeSmi([device]))

    def forbidden_subprocess(*args, **kwargs):
        raise AssertionError("Furiosa collector must not spawn furiosa-smi")

    monkeypatch.setattr("subprocess.run", forbidden_subprocess)
    collector = FuriosaCollector()
    collector.start()

    assert collector.collect_power().status == "ok"
