import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mobilint_device
from mobilint_device import MobilintDeviceSession
from monitors.base import HWMonitor
from monitors.mobilint_collector import MobilintCollector


MIB = 1024 ** 2


class FakeMbltml:
    MBLTML_DEVICE_ARIES = 1
    MBLTML_DEVICE_REGULUS = 2
    MBLTML_DEVICE_REGULUS_USB = 4

    def __init__(
        self,
        *,
        device_types=(1,),
        utilization=0.5,
        memory_usage=2 * MIB,
        memory_total=8 * MIB,
        temperature=55,
        current=2.5,
        voltage=12.0,
        power_actions=(10.0, 14.0, 18.0),
        failing_metrics=(),
    ):
        self.device_types = list(device_types)
        self.utilization = utilization
        self.memory_usage = memory_usage
        self.memory_total = memory_total
        self.temperature = temperature
        self.current = current
        self.voltage = voltage
        self.power_actions = list(power_actions)
        self.failing_metrics = set(failing_metrics)
        self.init_devices_calls = []
        self.shutdown_calls = 0
        self.shutdown_error = None
        self.power_reads = 0
        self.current_reads = 0
        self.voltage_reads = 0
        self.clock_reads = 0

    def _fail_if(self, name):
        if name in self.failing_metrics:
            raise RuntimeError(f"{name} unavailable")

    def mbltmlInitDevices(self, selected):
        self.init_devices_calls.append(set(selected))

    def mbltmlShutdown(self):
        self.shutdown_calls += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error

    def mbltmlGetDeviceCount(self):
        return len(self.device_types)

    def mbltmlGetDeviceType(self, device_id):
        return self.device_types[device_id]

    def mbltmlGetNodeName(self, device_id):
        return f"mobilint{device_id}"

    def mbltmlGetFirmwareVersion(self, device_id):
        return f"fw-{device_id}"

    def mbltmlGetMemoryTotal(self, device_id):
        self._fail_if("memory_total")
        return self.memory_total

    def mbltmlGetTotalUtilization(self, device_id):
        self._fail_if("utilization")
        return self.utilization

    def mbltmlGetMemoryUsage(self, device_id):
        self._fail_if("memory_usage")
        return self.memory_usage

    def mbltmlGetTemperature(self, device_id):
        self._fail_if("temperature")
        return self.temperature

    def mbltmlGetTotalPower(self, device_id):
        self.power_reads += 1
        action = self.power_actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        return action

    def mbltmlGetTotalCurrent(self, device_id):
        self.current_reads += 1
        self._fail_if("current")
        return self.current

    def mbltmlGetTotalVoltage(self, device_id):
        self.voltage_reads += 1
        self._fail_if("voltage")
        return self.voltage

    def mbltmlGetNPUClock(self, device_id):
        self.clock_reads += 1
        return 999


@pytest.fixture(autouse=True)
def isolated_mbltml_state(monkeypatch):
    monkeypatch.setattr(mobilint_device, "_STATE", mobilint_device._MbltmlState())


def install_fake(monkeypatch, fake):
    def fake_import(name):
        assert name == "mbltml"
        return fake

    monkeypatch.setattr(mobilint_device, "import_module", fake_import)


def test_aries_metrics_units_stop_boundary_and_trapezoidal_energy(monkeypatch):
    fake = FakeMbltml(
        utilization=0.5,
        memory_usage=2 * MIB,
        current=2.5,
        voltage=12.0,
        power_actions=(10.0, 14.0, 18.0),
    )
    install_fake(monkeypatch, fake)
    ticks = iter((0, 1_000_000_000, 2_000_000_000))
    collector = MobilintCollector(
        device_id=0,
        expected_family="aries",
        accelerator_name="ARIES",
        clock_ns=lambda: next(ticks),
    )

    collector.start()
    first = collector.collect()
    second = collector.collect()
    collector.stop()
    summary = collector.get_summary_metrics()
    static = collector.get_static_info()

    assert first == {
        "hw_accel_util": 50.0,
        "hw_accel_mem_used_mb": 2.0,
        "hw_accel_temp_c": 55.0,
        "hw_accel_power_w": 10.0,
        "hw_accel_current_a": 2.5,
        "hw_accel_voltage_mv": 12000.0,
    }
    assert second["hw_accel_power_w"] == 14.0
    assert summary == {
        "hw_accel_energy_j": 28.0,
        "hw_accel_power_samples": 3,
        "hw_accel_power_sample_coverage": 1.0,
    }
    assert static["hw_accel_vendor"] == "Mobilint"
    assert static["hw_accel_name"] == "ARIES"
    assert static["hw_accel_family"] == "aries"
    assert static["hw_accel_mem_total_mb"] == 8.0
    assert fake.power_reads == 3
    assert fake.clock_reads == 0
    assert fake.shutdown_calls == 1


def test_failed_power_read_breaks_energy_chain_and_reduces_coverage(monkeypatch):
    fake = FakeMbltml(
        power_actions=(10.0, RuntimeError("power gap"), 14.0, 18.0),
    )
    install_fake(monkeypatch, fake)
    ticks = iter((0, 2_000_000_000, 3_000_000_000))
    collector = MobilintCollector(
        expected_family="aries",
        clock_ns=lambda: next(ticks),
    )

    collector.start()
    assert collector.collect()["hw_accel_power_w"] == 10.0
    assert "hw_accel_power_w" not in collector.collect()
    assert collector.collect()["hw_accel_power_w"] == 14.0
    collector.stop()

    assert collector.get_summary_metrics() == {
        "hw_accel_energy_j": 16.0,
        "hw_accel_power_samples": 3,
        "hw_accel_power_sample_coverage": 0.75,
    }


def test_regulus_usb_reports_only_common_metrics(monkeypatch):
    fake = FakeMbltml(
        device_types=(4,),
        utilization=35.0,
        power_actions=(99.0,),
    )
    install_fake(monkeypatch, fake)
    collector = MobilintCollector(
        expected_family="regulus",
        accelerator_name="REGULUS",
    )

    collector.start()
    metrics = collector.collect()
    collector.stop()

    assert metrics == {
        "hw_accel_util": 35.0,
        "hw_accel_mem_used_mb": 2.0,
        "hw_accel_temp_c": 55.0,
    }
    assert collector.get_summary_metrics() == {}
    assert collector.get_static_info()["hw_accel_family"] == "regulus"
    assert fake.power_reads == 0
    assert fake.current_reads == 0
    assert fake.voltage_reads == 0


def test_metric_failures_are_isolated_and_stop_is_idempotent(monkeypatch):
    fake = FakeMbltml(
        failing_metrics=("utilization", "temperature", "current"),
        power_actions=(RuntimeError("power unavailable"), 8.0),
    )
    install_fake(monkeypatch, fake)
    collector = MobilintCollector(expected_family="aries", clock_ns=lambda: 1)

    collector.start()
    metrics = collector.collect()
    collector.stop()
    collector.stop()

    assert metrics == {
        "hw_accel_mem_used_mb": 2.0,
        "hw_accel_voltage_mv": 12000.0,
    }
    assert collector.get_summary_metrics() == {
        "hw_accel_energy_j": 0.0,
        "hw_accel_power_samples": 1,
        "hw_accel_power_sample_coverage": 0.5,
    }
    assert fake.shutdown_calls == 1


def test_collector_shares_session_with_runtime_owner(monkeypatch):
    fake = FakeMbltml(power_actions=(10.0,))
    install_fake(monkeypatch, fake)
    runtime_session = MobilintDeviceSession(0, "aries")
    runtime_session.acquire()
    collector = MobilintCollector(expected_family="aries", clock_ns=lambda: 0)

    collector.start()
    collector.stop()

    assert len(fake.init_devices_calls) == 1
    assert fake.shutdown_calls == 0
    runtime_session.release()
    assert fake.shutdown_calls == 1


@pytest.mark.parametrize(
    ("device_id", "expected_family", "message"),
    [
        (1, "aries", "out of range"),
        (0, "regulus", "expected REGULUS"),
    ],
)
def test_start_enforces_explicit_device_and_family(
    monkeypatch, device_id, expected_family, message
):
    fake = FakeMbltml(device_types=(1,))
    install_fake(monkeypatch, fake)
    collector = MobilintCollector(
        device_id=device_id,
        expected_family=expected_family,
    )

    with pytest.raises((ValueError, RuntimeError), match=message):
        collector.start()

    assert fake.shutdown_calls == 1


def test_failed_stop_release_retains_owner_and_retries_without_boundary_reread(
    monkeypatch,
):
    fake = FakeMbltml(power_actions=(10.0, 14.0))
    install_fake(monkeypatch, fake)
    collector = MobilintCollector(expected_family="aries", clock_ns=lambda: 0)
    collector.start()
    collector.collect()
    fake.shutdown_error = RuntimeError("shutdown failed")

    with pytest.raises(RuntimeError, match="shutdown failed"):
        collector.stop()

    assert fake.power_reads == 2
    assert fake.shutdown_calls == 1
    assert mobilint_device._STATE.cleanup_pending is True
    assert collector.get_summary_metrics() == {
        "hw_accel_energy_j": 0.0,
        "hw_accel_power_samples": 2,
        "hw_accel_power_sample_coverage": 1.0,
    }
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        collector.start()
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        collector.collect()

    fake.shutdown_error = None
    collector.stop()
    collector.stop()

    assert fake.power_reads == 2
    assert fake.shutdown_calls == 2
    assert mobilint_device._STATE.cleanup_pending is False
    assert collector.collect() == {}


def test_hw_monitor_retains_mobilint_owner_after_normal_stop_failure(
    monkeypatch,
):
    fake = FakeMbltml(device_types=(2,))
    install_fake(monkeypatch, fake)
    collector = MobilintCollector(expected_family="regulus")
    monitor = HWMonitor(interval=60.0)
    monitor.add_collector(collector)
    monitor.start()
    fake.shutdown_error = RuntimeError("shutdown failed")

    with pytest.raises(RuntimeError, match="shutdown failed"):
        monitor.stop()

    assert monitor._started_collectors == [collector]
    assert mobilint_device._STATE.cleanup_pending is True
    assert mobilint_device._STATE.ref_count == 1
    with pytest.raises(RuntimeError, match="already started"):
        monitor.start()

    fake.shutdown_error = None
    monitor.stop()

    assert monitor._started_collectors == []
    assert mobilint_device._STATE.cleanup_pending is False
    assert mobilint_device._STATE.ref_count == 0


def test_hw_monitor_retains_failed_mobilint_starter_for_later_cleanup(
    monkeypatch,
):
    fake = FakeMbltml(device_types=(1,))
    shutdown_fails = [True]

    def shutdown():
        fake.shutdown_calls += 1
        if shutdown_fails[0]:
            raise RuntimeError("shutdown failed")

    fake.mbltmlShutdown = shutdown
    install_fake(monkeypatch, fake)
    collector = MobilintCollector(expected_family="regulus")
    monitor = HWMonitor()
    monitor.add_collector(collector)

    with pytest.raises(
        RuntimeError,
        match="start failed and rollback cleanup is incomplete",
    ) as caught:
        monitor.start()

    assert "shutdown failed" in str(caught.value)
    assert "expected REGULUS" in str(caught.value.__cause__.__context__)
    assert fake.shutdown_calls == 2
    assert monitor._started_collectors == [collector]
    assert mobilint_device._STATE.cleanup_pending is True
    assert mobilint_device._STATE.ref_count == 1

    shutdown_fails[0] = False
    monitor.stop()

    assert fake.shutdown_calls == 3
    assert monitor._started_collectors == []
    assert mobilint_device._STATE.cleanup_pending is False
    assert mobilint_device._STATE.ref_count == 0


def test_startup_failure_and_rollback_release_failure_retains_retry_owner(
    monkeypatch,
):
    class CleanupFailingSession:
        instances = []

        def __init__(self, device_id, expected_family):
            self.device_id = device_id
            self.expected_family = expected_family
            self.info = None
            self.module = None
            self.release_calls = 0
            self.release_error = RuntimeError("release failed")
            type(self).instances.append(self)

        def acquire(self):
            self.info = SimpleNamespace(
                device_id=self.device_id,
                device_type=1,
                family=self.expected_family,
            )
            return self.info

        def release(self):
            self.release_calls += 1
            if self.release_error is not None:
                raise self.release_error
            self.info = None

    monkeypatch.setattr(
        "monitors.mobilint_collector.MobilintDeviceSession",
        CleanupFailingSession,
    )
    collector = MobilintCollector(expected_family="aries")

    with pytest.raises(
        RuntimeError,
        match=(
            "start failed and rollback cleanup is incomplete.*release failed"
            r".*call stop\(\) to retry cleanup"
        ),
    ) as caught:
        collector.start()

    session = CleanupFailingSession.instances[0]
    assert "active module" in str(caught.value.__cause__)
    assert session.release_calls == 1
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        collector.start()
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        collector.collect()

    session.release_error = None
    collector.stop()
    collector.stop()
    assert session.release_calls == 2
    assert collector.collect() == {}


def test_acquire_rollback_failure_retains_owner_without_second_release(
    monkeypatch,
):
    fake = FakeMbltml(device_types=(1,))
    fake.shutdown_error = RuntimeError("shutdown failed")
    install_fake(monkeypatch, fake)
    collector = MobilintCollector(expected_family="regulus")

    with pytest.raises(
        RuntimeError,
        match=(
            "start failed and rollback cleanup is incomplete.*shutdown failed"
            r".*call stop\(\) to retry cleanup"
        ),
    ) as caught:
        collector.start()

    assert "shutdown failed" in str(caught.value.__cause__)
    assert "expected REGULUS" in str(caught.value.__cause__.__context__)
    assert fake.shutdown_calls == 1
    assert mobilint_device._STATE.cleanup_pending is True
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        collector.start()
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        collector.collect()

    fake.shutdown_error = None
    collector.stop()

    assert fake.shutdown_calls == 2
    assert mobilint_device._STATE.cleanup_pending is False
    assert collector.collect() == {}


class BoundarySamplingError(BaseException):
    pass


def test_stop_releases_session_when_boundary_sampling_raises_baseexception(
    monkeypatch,
):
    boundary_error = BoundarySamplingError("boundary interrupted")
    fake = FakeMbltml(power_actions=(boundary_error,))
    install_fake(monkeypatch, fake)
    collector = MobilintCollector(expected_family="aries")
    collector.start()

    with pytest.raises(BoundarySamplingError) as caught:
        collector.stop()

    assert caught.value is boundary_error
    assert fake.power_reads == 1
    assert fake.shutdown_calls == 1
    assert mobilint_device._STATE.ref_count == 0
    collector.stop()
    assert fake.power_reads == 1
    assert fake.shutdown_calls == 1


def test_boundary_and_release_failures_retain_owner_without_boundary_reread(
    monkeypatch,
):
    boundary_error = BoundarySamplingError("boundary interrupted")
    shutdown_error = RuntimeError("shutdown failed")
    fake = FakeMbltml(power_actions=(boundary_error,))
    fake.shutdown_error = shutdown_error
    install_fake(monkeypatch, fake)
    collector = MobilintCollector(expected_family="aries")
    collector.start()

    with pytest.raises(
        RuntimeError,
        match=(
            "stop boundary sampling failed and cleanup is incomplete"
            ".*shutdown failed"
            r".*call stop\(\) to retry cleanup"
        ),
    ) as caught:
        collector.stop()

    assert caught.value.__cause__ is boundary_error
    assert caught.value.__context__ is shutdown_error
    assert fake.power_reads == 1
    assert fake.shutdown_calls == 1
    assert mobilint_device._STATE.cleanup_pending is True
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        collector.collect()

    fake.shutdown_error = None
    collector.stop()
    collector.stop()

    assert fake.power_reads == 1
    assert fake.shutdown_calls == 2
    assert mobilint_device._STATE.cleanup_pending is False


def test_clock_failure_keeps_power_sample_but_breaks_energy_chain(monkeypatch):
    fake = FakeMbltml(power_actions=(10.0, 14.0, 18.0))
    install_fake(monkeypatch, fake)
    clock_actions = iter((0, RuntimeError("clock failed"), 2_000_000_000))

    def clock_ns():
        action = next(clock_actions)
        if isinstance(action, BaseException):
            raise action
        return action

    collector = MobilintCollector(expected_family="aries", clock_ns=clock_ns)
    collector.start()

    assert collector.collect()["hw_accel_power_w"] == 10.0
    assert collector.collect()["hw_accel_power_w"] == 14.0
    collector.stop()

    assert collector.get_summary_metrics() == {
        "hw_accel_energy_j": 0.0,
        "hw_accel_power_samples": 3,
        "hw_accel_power_sample_coverage": 1.0,
    }
    assert collector.get_static_info()["hw_accel_monitor_note"] == (
        "Mobilint power timestamp failed: RuntimeError"
    )


class VendorDiagnosticError(RuntimeError):
    pass


def test_static_diagnostic_note_excludes_vendor_exception_text(monkeypatch):
    fake = FakeMbltml()

    def fail_memory_total(device_id):
        raise VendorDiagnosticError("secret /dev/mblt0 payload=" + "x" * 1000)

    fake.mbltmlGetMemoryTotal = fail_memory_total
    install_fake(monkeypatch, fake)
    collector = MobilintCollector(expected_family="aries")

    collector.start()

    assert collector.get_static_info()["hw_accel_monitor_note"] == (
        "Mobilint mbltmlGetMemoryTotal failed: VendorDiagnosticError"
    )
    collector.stop()


def test_metric_diagnostic_note_excludes_vendor_exception_text(monkeypatch):
    fake = FakeMbltml()

    def fail_temperature(device_id):
        raise VendorDiagnosticError("secret /dev/mblt0 metric payload")

    fake.mbltmlGetTemperature = fail_temperature
    install_fake(monkeypatch, fake)
    collector = MobilintCollector(expected_family="aries")
    collector.start()

    collector.collect()

    assert collector.get_static_info()["hw_accel_monitor_note"] == (
        "Mobilint mbltmlGetTemperature failed: VendorDiagnosticError"
    )
    collector.stop()


def test_power_diagnostic_note_excludes_vendor_exception_text(monkeypatch):
    fake = FakeMbltml(
        power_actions=(
            VendorDiagnosticError("secret /dev/mblt0 power payload"),
            8.0,
        )
    )
    install_fake(monkeypatch, fake)
    collector = MobilintCollector(expected_family="aries", clock_ns=lambda: 0)
    collector.start()

    collector.collect()

    assert collector.get_static_info()["hw_accel_monitor_note"] == (
        "Mobilint mbltmlGetTotalPower failed: VendorDiagnosticError"
    )
    collector.stop()


def test_clock_diagnostic_note_excludes_vendor_exception_text(monkeypatch):
    fake = FakeMbltml(power_actions=(10.0, 14.0))
    install_fake(monkeypatch, fake)

    def fail_clock():
        raise VendorDiagnosticError("secret clock payload=" + "x" * 1000)

    collector = MobilintCollector(expected_family="aries", clock_ns=fail_clock)
    collector.start()

    collector.collect()

    assert collector.get_static_info()["hw_accel_monitor_note"] == (
        "Mobilint power timestamp failed: VendorDiagnosticError"
    )
    collector.stop()
