import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mobilint_device
from mobilint_device import MobilintDeviceSession


class FakeMbltml:
    MBLTML_DEVICE_ARIES = 1
    MBLTML_DEVICE_REGULUS = 2
    MBLTML_DEVICE_REGULUS_USB = 4

    def __init__(self, device_types=(1, 4)):
        self.device_types = list(device_types)
        self.init_devices_calls = []
        self.init_calls = 0
        self.shutdown_calls = 0
        self.shutdown_error = None

    def mbltmlInitDevices(self, device_types):
        self.init_devices_calls.append(set(device_types))

    def mbltmlInit(self):
        self.init_calls += 1

    def mbltmlGetDeviceCount(self):
        return len(self.device_types)

    def mbltmlGetDeviceType(self, device_id):
        return self.device_types[device_id]

    def mbltmlShutdown(self):
        self.shutdown_calls += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error


@pytest.fixture(autouse=True)
def isolated_mbltml_state(monkeypatch):
    monkeypatch.setattr(
        mobilint_device,
        "_STATE",
        mobilint_device._MbltmlState(),
    )


def _install(monkeypatch, fake):
    def fake_import(name):
        assert name == "mbltml"
        return fake

    monkeypatch.setattr(mobilint_device, "import_module", fake_import)


def test_sessions_share_init_and_only_last_release_shuts_down(monkeypatch):
    fake = FakeMbltml(device_types=(1,))
    _install(monkeypatch, fake)
    first = MobilintDeviceSession(device_id=0, expected_family="aries")
    second = MobilintDeviceSession(device_id=0, expected_family="aries")

    assert first.acquire().family == "aries"
    assert second.acquire().family == "aries"
    first.release()
    assert fake.shutdown_calls == 0
    second.release()
    second.release()

    assert fake.init_devices_calls == [{1}]
    assert fake.init_calls == 0
    assert fake.shutdown_calls == 1


def test_failed_final_shutdown_retains_lease_for_retry_and_blocks_acquire(
    monkeypatch,
):
    fake = FakeMbltml(device_types=(1,))
    _install(monkeypatch, fake)
    session = MobilintDeviceSession(device_id=0, expected_family="aries")
    session.acquire()
    fake.shutdown_error = RuntimeError("shutdown failed")

    with pytest.raises(RuntimeError, match="shutdown failed"):
        session.release()

    assert session.info.family == "aries"
    assert mobilint_device._STATE.module is fake
    assert mobilint_device._STATE.family == "aries"
    assert mobilint_device._STATE.ref_count == 1
    assert mobilint_device._STATE.cleanup_pending is True
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        MobilintDeviceSession(0, "aries").acquire()

    fake.shutdown_error = None
    session.release()
    assert session.info is None
    assert mobilint_device._STATE.module is None
    assert mobilint_device._STATE.family is None
    assert mobilint_device._STATE.ref_count == 0
    assert mobilint_device._STATE.cleanup_pending is False
    assert fake.shutdown_calls == 2

    replacement = MobilintDeviceSession(0, "aries")
    replacement.acquire()
    replacement.release()
    assert fake.init_devices_calls == [{1}, {1}]
    assert fake.shutdown_calls == 3


def test_regulus_pcie_and_usb_types_belong_to_same_family(monkeypatch):
    fake = FakeMbltml(device_types=(2, 4))
    _install(monkeypatch, fake)

    first = MobilintDeviceSession(0, "regulus")
    second = MobilintDeviceSession(1, "regulus")
    assert first.acquire().family == "regulus"
    assert second.acquire().family == "regulus"
    first.release()
    second.release()

    assert fake.init_devices_calls == [{2, 4}]


def test_active_family_rejects_a_different_family(monkeypatch):
    fake = FakeMbltml(device_types=(1,))
    _install(monkeypatch, fake)
    active = MobilintDeviceSession(0, "aries")
    active.acquire()

    with pytest.raises(RuntimeError, match="already initialized for ARIES"):
        MobilintDeviceSession(0, "regulus").acquire()

    active.release()
    assert fake.shutdown_calls == 1


def test_family_mismatch_releases_failed_acquire(monkeypatch):
    fake = FakeMbltml(device_types=(1,))
    _install(monkeypatch, fake)
    session = MobilintDeviceSession(0, "regulus")

    with pytest.raises(RuntimeError, match="expected REGULUS.*detected ARIES"):
        session.acquire()

    assert fake.shutdown_calls == 1
    assert session.info is None


def test_invalid_device_id_releases_failed_acquire(monkeypatch):
    fake = FakeMbltml(device_types=(1,))
    _install(monkeypatch, fake)

    with pytest.raises(ValueError, match="device_id=3.*device_count=1"):
        MobilintDeviceSession(3, "aries").acquire()

    assert fake.shutdown_calls == 1


def test_missing_mbltml_reports_optional_dependency(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(mobilint_device, "import_module", missing)

    with pytest.raises(ImportError, match="mbltml"):
        MobilintDeviceSession(0, "aries").acquire()


def test_legacy_module_falls_back_to_mbltml_init(monkeypatch):
    fake = FakeMbltml(device_types=(1,))
    legacy = SimpleNamespace(
        MBLTML_DEVICE_ARIES=1,
        MBLTML_DEVICE_REGULUS=2,
        MBLTML_DEVICE_REGULUS_USB=4,
        mbltmlInit=fake.mbltmlInit,
        mbltmlGetDeviceCount=fake.mbltmlGetDeviceCount,
        mbltmlGetDeviceType=fake.mbltmlGetDeviceType,
        mbltmlShutdown=fake.mbltmlShutdown,
    )
    _install(monkeypatch, legacy)

    session = MobilintDeviceSession(0, "aries")
    session.acquire()
    session.release()

    assert fake.init_calls == 1
    assert fake.shutdown_calls == 1
