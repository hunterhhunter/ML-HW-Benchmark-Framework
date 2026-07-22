from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import monitors.rbln_collector as rbln_collector
from monitors.rbln_collector import RblnCollector, _number


MIB = 1024**2


class FakeRunner:
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        if isinstance(action, SimpleNamespace):
            return action
        return SimpleNamespace(stdout=json.dumps(action))


class FakeClock:
    def __init__(self, initial=100.0):
        self.current = float(initial)

    def __call__(self):
        return self.current

    def advance(self, seconds):
        self.current += seconds


@pytest.fixture
def user_payload():
    return {
        "KMD_version": "3.2.2",
        "devices": [
            {
                "npu": 0,
                "name": "RBLN-CA22",
                "sid": "0000000022509017",
                "uuid": "adf1ae64-e53d-4f66-adf0-d561a18b202e",
                "device": "rbln0",
                "status": "normal",
                "fw_ver": "3.2.2",
                "pci": {
                    "dev": "0x1220",
                    "bus_id": "0000:ab:00.0",
                    "numa_node": "1",
                    "link_speed": "32.0GT/s",
                    "link_width": "8",
                },
                "temperature": "38C",
                "card_power": "18810987uW",
                "pstate": "P14",
                "memory": {"used": "0", "total": "16877879296"},
                "util": "0.0",
                "board_info": "0005000c",
                "location": 5,
            }
        ],
        "contexts": [],
    }


@pytest.fixture
def numeric_payload(user_payload):
    payload = deepcopy(user_payload)
    payload.pop("KMD_version")
    payload["driver_version"] = "3.3.0"
    device = payload["devices"][0]
    device["util"] = 37.5
    device["memory"] = {"used": 2 * MIB, "total": 4 * MIB}
    device["temperature"] = 42
    device["card_power"] = 12.5
    device["pci"]["numa_node"] = 1
    device["pci"]["link_width"] = 8
    return payload


@pytest.fixture
def missing_fields_payload(user_payload):
    payload = deepcopy(user_payload)
    device = payload["devices"][0]
    device.pop("temperature")
    device.pop("card_power")
    device.pop("memory")
    payload.pop("contexts")
    return payload


def make_collector(runner, **options):
    return RblnCollector(
        runner=runner,
        executable_resolver=lambda name: "/usr/bin/rbln-smi",
        **options,
    )


def with_power(payload, watts):
    result = deepcopy(payload)
    result["devices"][0]["card_power"] = f"{watts}W"
    return result


def test_module_import_and_is_available_do_not_import_rebel():
    script = """
import builtins
import sys

sys.path.insert(0, sys.argv[1])
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] == "rebel":
        raise AssertionError("RblnCollector must not import the Python SDK")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from monitors.rbln_collector import RblnCollector
assert RblnCollector().is_available() is True
"""

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(__file__).resolve().parent.parent / "src"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("device_id", [True, -1, 0.0, "0"])
def test_init_rejects_invalid_device_id(device_id):
    with pytest.raises(ValueError, match="device_id"):
        RblnCollector(device_id=device_id)


@pytest.mark.parametrize(
    "command_timeout_sec",
    [True, 0, -1.0, float("nan"), float("inf"), "2.0"],
)
def test_init_rejects_invalid_command_timeout(command_timeout_sec):
    with pytest.raises(ValueError, match="command_timeout_sec"):
        RblnCollector(command_timeout_sec=command_timeout_sec)


@pytest.mark.parametrize(
    "sample_interval_sec",
    [True, 0, -1.0, float("nan"), float("inf"), "1.0"],
)
def test_init_rejects_invalid_sample_interval(sample_interval_sec):
    with pytest.raises(ValueError, match="sample_interval_sec"):
        RblnCollector(sample_interval_sec=sample_interval_sec)


def test_init_accepts_nonnegative_device_and_clamps_short_sample_interval():
    collector = RblnCollector(device_id=1, sample_interval_sec=0.25)

    assert collector.device_id == 1
    assert collector.sample_interval_sec == 1.0
    assert collector.command_timeout_sec == 2.0


def test_start_uses_safe_device_scoped_command_and_parses_user_payload(
    user_payload,
):
    runner = FakeRunner([user_payload, user_payload])
    collector = make_collector(runner, device_id=0)

    collector.start()
    metrics = collector.collect(force=True)

    args, kwargs = runner.calls[0]
    assert args == ["rbln-smi", "-b", "-j", "-d", "0"]
    assert kwargs == {
        "capture_output": True,
        "text": True,
        "check": True,
        "timeout": 2.0,
        "shell": False,
    }
    assert metrics == {
        "hw_accel_util": 0.0,
        "hw_accel_mem_used_mb": 0.0,
        "hw_accel_temp_c": 38.0,
        "hw_accel_power_w": pytest.approx(18.810987),
    }
    assert collector.get_static_info() == {
        "hw_accel_vendor": "Rebellions",
        "hw_accel_name": "RBLN-CA22",
        "hw_accel_device_id": 0,
        "hw_accel_device_node": "rbln0",
        "hw_accel_uuid": "adf1ae64-e53d-4f66-adf0-d561a18b202e",
        "hw_accel_serial_id": "0000000022509017",
        "hw_accel_status": "normal",
        "hw_accel_pstate": "P14",
        "hw_accel_monitor_source": "rbln-smi-json",
        "hw_accel_kmd_version": "3.2.2",
        "hw_accel_firmware_version": "3.2.2",
        "hw_accel_pci_bus_id": "0000:ab:00.0",
        "hw_accel_pci_numa_node": 1,
        "hw_accel_pci_link_speed": "32.0GT/s",
        "hw_accel_pci_link_width": 8,
        "hw_accel_mem_total_mb": pytest.approx(16096.0),
    }


def test_parse_accepts_numeric_values_and_driver_version(numeric_payload):
    runner = FakeRunner([numeric_payload, numeric_payload])
    collector = make_collector(runner)

    collector.start()
    metrics = collector.collect(force=True)

    assert metrics == {
        "hw_accel_util": 37.5,
        "hw_accel_mem_used_mb": 2.0,
        "hw_accel_temp_c": 42.0,
        "hw_accel_power_w": 12.5,
    }
    static = collector.get_static_info()
    assert static["hw_accel_kmd_version"] == "3.3.0"
    assert static["hw_accel_mem_total_mb"] == 4.0
    assert static["hw_accel_pci_numa_node"] == 1
    assert static["hw_accel_pci_link_width"] == 8


def test_context_memory_sums_only_matching_device_and_process(user_payload):
    payload = deepcopy(user_payload)
    current_pid = os.getpid()
    payload["contexts"] = [
        {"npu": 0, "pid": current_pid, "memalloc": str(2 * MIB)},
        {"npu": 0, "pid": str(current_pid), "memalloc": 3 * MIB},
        {"npu": 0, "pid": current_pid + 1, "memalloc": 100 * MIB},
        {"npu": 1, "pid": current_pid, "memalloc": 200 * MIB},
        {"npu": 0, "pid": current_pid, "memalloc": "invalid"},
    ]
    runner = FakeRunner([payload, payload])
    collector = make_collector(runner)

    collector.start()
    metrics = collector.collect(force=True)

    assert metrics["hw_accel_mem_proc_mb"] == 5.0


def test_context_nonfinite_is_ignored_for_unrelated_or_malformed_rows(
    user_payload,
):
    payload = deepcopy(user_payload)
    current_pid = os.getpid()
    payload["contexts"] = [
        {"npu": 1, "pid": current_pid, "memalloc": "NaN MiB"},
        {"npu": 0, "pid": current_pid + 1, "memalloc": float("inf")},
        {"npu": "invalid", "pid": current_pid, "memalloc": "NaN"},
        {"npu": 0, "pid": "invalid", "memalloc": "Infinity"},
        {"npu": 0, "pid": current_pid, "memalloc": 2 * MIB},
    ]
    collector = make_collector(FakeRunner([payload, payload]))

    collector.start()
    metrics = collector.collect(force=True)

    assert metrics["hw_accel_mem_proc_mb"] == 2.0
    assert collector.get_summary_metrics()["hw_accel_monitor_successes"] == 2


def test_context_nonfinite_invalidates_matching_process_row(user_payload):
    payload = deepcopy(user_payload)
    payload["contexts"] = [
        {"npu": 0, "pid": os.getpid(), "memalloc": "NaN MiB"}
    ]
    collector = make_collector(FakeRunner([user_payload, payload]))
    collector.start()

    assert collector.collect(force=True) == {}

    summary = collector.get_summary_metrics()
    assert summary["hw_accel_monitor_successes"] == 1
    assert "ValueError" in summary["hw_accel_monitor_note"]


def test_context_ignores_huge_unrelated_selectors_and_their_values(
    user_payload,
):
    payload = deepcopy(user_payload)
    current_pid = os.getpid()
    payload["contexts"] = [
        {"npu": 10**400, "pid": current_pid, "memalloc": "NaN MiB"},
        {"npu": 0, "pid": 10**400, "memalloc": "1e309 MiB"},
        {"npu": 0, "pid": current_pid, "memalloc": 2 * MIB},
    ]
    collector = make_collector(FakeRunner([payload, payload]))

    collector.start()
    metrics = collector.collect(force=True)

    assert metrics["hw_accel_mem_proc_mb"] == 2.0
    assert collector.get_summary_metrics()["hw_accel_monitor_successes"] == 2


@pytest.mark.parametrize("resolved", [None, "", 123])
def test_start_fails_when_executable_is_missing_or_invalid(resolved):
    runner = FakeRunner([])
    collector = RblnCollector(
        runner=runner,
        executable_resolver=lambda name: resolved,
    )

    with pytest.raises(RuntimeError, match="rbln-smi executable"):
        collector.start()

    assert runner.calls == []


def test_start_propagates_initial_command_failure():
    error = subprocess.CalledProcessError(
        1,
        ["rbln-smi"],
        output="sensitive stdout",
        stderr="sensitive stderr",
    )
    collector = make_collector(FakeRunner([error]))

    with pytest.raises(subprocess.CalledProcessError) as caught:
        collector.start()

    assert caught.value is error


def test_start_fails_for_malformed_initial_json():
    runner = FakeRunner([SimpleNamespace(stdout="{not-json")])
    collector = make_collector(runner)

    with pytest.raises(json.JSONDecodeError):
        collector.start()


@pytest.mark.parametrize("devices", [[], [{"npu": 1, "status": "normal"}]])
def test_start_fails_when_selected_device_is_missing(user_payload, devices):
    payload = deepcopy(user_payload)
    payload["devices"] = devices
    collector = make_collector(FakeRunner([payload]))

    with pytest.raises(RuntimeError, match="device 0"):
        collector.start()


def test_start_fails_when_selected_device_is_duplicated(user_payload):
    payload = deepcopy(user_payload)
    payload["devices"].append(deepcopy(payload["devices"][0]))
    collector = make_collector(FakeRunner([payload]))

    with pytest.raises(RuntimeError, match="device 0"):
        collector.start()


def test_start_fails_when_selected_device_status_is_not_normal(user_payload):
    payload = deepcopy(user_payload)
    payload["devices"][0]["status"] = "error"
    collector = make_collector(FakeRunner([payload]))

    with pytest.raises(RuntimeError, match="status"):
        collector.start()


def test_snapshot_selects_only_requested_device_when_others_are_present(
    user_payload,
):
    payload = deepcopy(user_payload)
    unrelated = deepcopy(payload["devices"][0])
    unrelated.update(
        {"npu": 1, "name": "UNRELATED-NPU", "status": "error", "util": 99.0}
    )
    payload["devices"].insert(0, unrelated)
    collector = make_collector(FakeRunner([payload, payload]))

    collector.start()
    metrics = collector.collect(force=True)

    assert metrics["hw_accel_util"] == 0.0
    assert collector.get_static_info()["hw_accel_name"] == "RBLN-CA22"


def test_snapshot_ignores_unrelated_device_with_huge_integer_selector(
    user_payload,
):
    payload = deepcopy(user_payload)
    unrelated = deepcopy(payload["devices"][0])
    unrelated.update(
        {
            "npu": 10**400,
            "name": "UNRELATED-NPU",
            "status": "error",
        }
    )
    payload["devices"].insert(0, unrelated)
    collector = make_collector(FakeRunner([payload, payload]))

    collector.start()
    metrics = collector.collect(force=True)

    assert metrics["hw_accel_util"] == 0.0
    assert collector.get_static_info()["hw_accel_name"] == "RBLN-CA22"


def test_throttle_clamps_interval_and_skips_subprocess_calls(user_payload):
    runner = FakeRunner([user_payload, user_payload])
    clock = FakeClock()
    collector = make_collector(
        runner,
        clock=clock,
        sample_interval_sec=0.01,
    )

    collector.start()
    assert collector.collect() == {}
    clock.advance(0.999)
    assert collector.collect() == {}
    assert len(runner.calls) == 1

    clock.advance(0.001)
    assert collector.collect()["hw_accel_power_w"] == pytest.approx(18.810987)
    assert len(runner.calls) == 2


def test_stop_forces_one_final_snapshot_despite_throttle(user_payload):
    runner = FakeRunner([user_payload, user_payload])
    collector = make_collector(runner, clock=FakeClock())

    collector.start()
    assert collector.collect() == {}
    collector.stop()
    collector.stop()

    assert len(runner.calls) == 2
    assert collector.collect(force=True) == {}


def test_stop_final_power_sample_updates_energy_and_counters(user_payload):
    runner = FakeRunner(
        [with_power(user_payload, 10.0), with_power(user_payload, 14.0)]
    )
    clock = FakeClock(initial=0.0)
    collector = make_collector(runner, clock=clock)
    collector.start()
    clock.advance(2.0)

    collector.stop()

    assert collector.get_summary_metrics() == {
        "hw_accel_energy_j": 24.0,
        "hw_accel_power_samples": 2,
        "hw_accel_monitor_attempts": 2,
        "hw_accel_monitor_successes": 2,
        "hw_accel_monitor_coverage": 1.0,
    }


def test_energy_is_absent_until_two_power_samples_exist(user_payload):
    collector = make_collector(FakeRunner([with_power(user_payload, 10.0)]))

    collector.start()

    assert collector.get_summary_metrics() == {
        "hw_accel_power_samples": 1,
        "hw_accel_monitor_attempts": 1,
        "hw_accel_monitor_successes": 1,
        "hw_accel_monitor_coverage": 1.0,
    }


def test_energy_integrates_successful_power_samples_with_trapezoids(
    user_payload,
):
    runner = FakeRunner(
        [with_power(user_payload, 10.0), with_power(user_payload, 14.0)]
    )
    clock = FakeClock(initial=0.0)
    collector = make_collector(runner, clock=clock)
    collector.start()
    clock.advance(2.0)

    metrics = collector.collect(force=True)

    assert metrics["hw_accel_power_w"] == 14.0
    assert collector.get_summary_metrics() == {
        "hw_accel_energy_j": 24.0,
        "hw_accel_power_samples": 2,
        "hw_accel_monitor_attempts": 2,
        "hw_accel_monitor_successes": 2,
        "hw_accel_monitor_coverage": 1.0,
    }


def test_coverage_does_not_count_polls_skipped_by_throttle(user_payload):
    runner = FakeRunner([user_payload, user_payload])
    clock = FakeClock(initial=0.0)
    collector = make_collector(runner, clock=clock)
    collector.start()

    assert collector.collect() == {}
    clock.advance(1.0)
    collector.collect()

    summary = collector.get_summary_metrics()
    assert summary["hw_accel_monitor_attempts"] == 2
    assert summary["hw_accel_monitor_successes"] == 2
    assert summary["hw_accel_monitor_coverage"] == 1.0


def transient_action(kind, user_payload):
    if kind == "timeout":
        return subprocess.TimeoutExpired(
            ["rbln-smi"],
            2.0,
            output="private timeout stdout",
            stderr="private timeout stderr",
        )
    if kind == "nonzero":
        return subprocess.CalledProcessError(
            7,
            ["rbln-smi"],
            output="private failure stdout",
            stderr="private failure stderr",
        )
    if kind == "malformed":
        return SimpleNamespace(stdout="{private malformed output")

    payload = deepcopy(user_payload)
    if kind == "missing_device":
        payload["devices"] = []
    elif kind == "non_normal":
        payload["devices"][0]["status"] = "error"
    elif kind == "nonfinite":
        payload["devices"][0]["util"] = float("nan")
    elif kind == "nonfinite_percent":
        payload["devices"][0]["util"] = "NaN %"
    else:
        raise AssertionError(f"unknown transient kind: {kind}")
    return payload


@pytest.mark.parametrize(
    ("kind", "error_type"),
    [
        ("timeout", "TimeoutExpired"),
        ("nonzero", "CalledProcessError"),
        ("malformed", "JSONDecodeError"),
        ("missing_device", "RuntimeError"),
        ("non_normal", "RuntimeError"),
        ("nonfinite", "ValueError"),
        ("nonfinite_percent", "ValueError"),
    ],
)
def test_transient_sample_failure_is_omitted_and_safely_summarized(
    user_payload,
    kind,
    error_type,
):
    runner = FakeRunner([user_payload, transient_action(kind, user_payload)])
    collector = make_collector(runner, clock=FakeClock())
    collector.start()

    assert collector.collect(force=True) == {}

    summary = collector.get_summary_metrics()
    assert summary["hw_accel_monitor_attempts"] == 2
    assert summary["hw_accel_monitor_successes"] == 1
    assert summary["hw_accel_monitor_coverage"] == 0.5
    assert error_type in summary["hw_accel_monitor_note"]
    assert len(summary["hw_accel_monitor_note"]) <= 128
    serialized = json.dumps(
        {"summary": summary, "static": collector.get_static_info()}
    )
    assert "private" not in serialized
    assert "stdout" not in serialized
    assert "stderr" not in serialized


def test_transient_failure_breaks_energy_chain(user_payload):
    runner = FakeRunner(
        [
            with_power(user_payload, 10.0),
            subprocess.TimeoutExpired(["rbln-smi"], 2.0),
            with_power(user_payload, 14.0),
            with_power(user_payload, 18.0),
        ]
    )
    clock = FakeClock(initial=0.0)
    collector = make_collector(runner, clock=clock)
    collector.start()
    clock.advance(1.0)
    assert collector.collect(force=True) == {}
    clock.advance(1.0)
    collector.collect(force=True)
    clock.advance(1.0)
    collector.collect(force=True)

    summary = collector.get_summary_metrics()
    assert summary["hw_accel_energy_j"] == 16.0
    assert summary["hw_accel_power_samples"] == 3
    assert summary["hw_accel_monitor_attempts"] == 4
    assert summary["hw_accel_monitor_successes"] == 3
    assert summary["hw_accel_monitor_coverage"] == 0.75


def test_unit_conversion_overflow_invalidates_the_whole_sample(user_payload):
    overflow = deepcopy(user_payload)
    overflow["devices"][0]["memory"]["used"] = "1e308 GiB"
    collector = make_collector(FakeRunner([user_payload, overflow]))
    collector.start()

    assert collector.collect(force=True) == {}

    summary = collector.get_summary_metrics()
    assert summary["hw_accel_monitor_attempts"] == 2
    assert summary["hw_accel_monitor_successes"] == 1
    assert "ValueError" in summary["hw_accel_monitor_note"]


@pytest.mark.parametrize("field", ["power", "util", "memory"])
def test_numeric_token_overflow_invalidates_the_whole_sample(
    user_payload,
    field,
):
    overflow = deepcopy(user_payload)
    device = overflow["devices"][0]
    if field == "power":
        device["card_power"] = "1e309 W"
    elif field == "util":
        device["util"] = "1e309"
    else:
        device["memory"]["used"] = "1e309 MiB"
    collector = make_collector(FakeRunner([user_payload, overflow]))
    collector.start()

    assert collector.collect(force=True) == {}

    summary = collector.get_summary_metrics()
    assert summary["hw_accel_monitor_successes"] == 1
    assert "ValueError" in summary["hw_accel_monitor_note"]


def test_ordinary_invalid_numeric_fields_are_omitted_without_failure(
    user_payload,
):
    invalid = deepcopy(user_payload)
    device = invalid["devices"][0]
    device["card_power"] = "unavailable"
    device["util"] = "not-a-number"
    device["memory"]["used"] = "unknown"
    collector = make_collector(FakeRunner([user_payload, invalid]))
    collector.start()

    assert collector.collect(force=True) == {"hw_accel_temp_c": 38.0}

    summary = collector.get_summary_metrics()
    assert summary["hw_accel_monitor_successes"] == 2
    assert "hw_accel_monitor_note" not in summary


def test_number_does_not_coerce_arbitrary_objects_to_text_or_float():
    class HostileNumber:
        def __str__(self):
            raise AssertionError("must not stringify arbitrary telemetry")

        def __float__(self):
            raise AssertionError("must not coerce arbitrary telemetry")

    assert _number(HostileNumber()) is None


def test_default_process_id_is_resolved_when_collector_is_constructed(
    user_payload,
    monkeypatch,
):
    payload = deepcopy(user_payload)
    payload["contexts"] = [
        {"npu": 0, "pid": 4242, "memalloc": 3 * MIB},
        {"npu": 0, "pid": 9999, "memalloc": 100 * MIB},
    ]
    monkeypatch.setattr(rbln_collector.os, "getpid", lambda: 4242)
    collector = make_collector(FakeRunner([payload, payload]))
    monkeypatch.setattr(rbln_collector.os, "getpid", lambda: 9999)

    collector.start()
    metrics = collector.collect(force=True)

    assert metrics["hw_accel_mem_proc_mb"] == 3.0


def test_energy_overflow_rejects_sample_without_partial_state(user_payload):
    runner = FakeRunner(
        [with_power(user_payload, 1e308), with_power(user_payload, 1e308)]
    )
    clock = FakeClock(initial=0.0)
    collector = make_collector(runner, clock=clock)
    collector.start()
    clock.advance(10.0)

    assert collector.collect(force=True) == {}

    assert collector.get_summary_metrics() == {
        "hw_accel_power_samples": 1,
        "hw_accel_monitor_attempts": 2,
        "hw_accel_monitor_successes": 1,
        "hw_accel_monitor_coverage": 0.5,
        "hw_accel_monitor_note": "RBLN snapshot failed: ValueError",
    }


def test_missing_sensor_fields_are_omitted_without_erasing_static_cache(
    user_payload,
    missing_fields_payload,
):
    runner = FakeRunner([user_payload, missing_fields_payload])
    collector = make_collector(runner, clock=FakeClock())
    collector.start()

    metrics = collector.collect(force=True)

    assert metrics == {"hw_accel_util": 0.0}
    assert collector.get_static_info()["hw_accel_mem_total_mb"] == pytest.approx(
        16096.0
    )
    summary = collector.get_summary_metrics()
    assert summary["hw_accel_monitor_successes"] == 2
    assert summary["hw_accel_power_samples"] == 1
    assert "hw_accel_energy_j" not in summary
    assert "hw_accel_monitor_note" not in summary


def test_missing_power_breaks_energy_chain(user_payload, missing_fields_payload):
    missing = deepcopy(missing_fields_payload)
    missing["devices"][0]["memory"] = deepcopy(
        user_payload["devices"][0]["memory"]
    )
    runner = FakeRunner(
        [
            with_power(user_payload, 10.0),
            missing,
            with_power(user_payload, 14.0),
        ]
    )
    clock = FakeClock(initial=0.0)
    collector = make_collector(runner, clock=clock)
    collector.start()
    clock.advance(1.0)
    collector.collect(force=True)
    clock.advance(1.0)
    collector.collect(force=True)

    summary = collector.get_summary_metrics()
    assert summary["hw_accel_power_samples"] == 2
    assert summary["hw_accel_energy_j"] == 0.0


def test_transient_failure_is_throttled_like_any_other_attempt(user_payload):
    runner = FakeRunner(
        [
            user_payload,
            subprocess.TimeoutExpired(["rbln-smi"], 2.0),
            user_payload,
        ]
    )
    clock = FakeClock(initial=0.0)
    collector = make_collector(runner, clock=clock)
    collector.start()
    clock.advance(1.0)
    assert collector.collect() == {}

    assert collector.collect() == {}
    assert len(runner.calls) == 2
    clock.advance(1.0)
    assert collector.collect()["hw_accel_util"] == 0.0
    assert len(runner.calls) == 3


def test_stop_suppresses_final_transient_failure_and_is_idempotent(user_payload):
    runner = FakeRunner(
        [user_payload, subprocess.TimeoutExpired(["rbln-smi"], 2.0)]
    )
    collector = make_collector(runner, clock=FakeClock())
    collector.start()

    collector.stop()
    collector.stop()

    assert len(runner.calls) == 2
    summary = collector.get_summary_metrics()
    assert summary["hw_accel_monitor_attempts"] == 2
    assert summary["hw_accel_monitor_successes"] == 1
    assert "TimeoutExpired" in summary["hw_accel_monitor_note"]


def test_restart_resets_measurements_static_cache_and_error_state(user_payload):
    changed = deepcopy(user_payload)
    changed["devices"][0]["name"] = "RBLN-CA22-RESTARTED"
    runner = FakeRunner(
        [
            user_payload,
            subprocess.TimeoutExpired(["rbln-smi"], 2.0),
            user_payload,
            changed,
        ]
    )
    collector = make_collector(runner, clock=FakeClock())
    collector.start()
    assert collector.collect(force=True) == {}
    collector.stop()
    assert "hw_accel_monitor_note" in collector.get_summary_metrics()

    collector.start()

    assert collector.get_summary_metrics() == {
        "hw_accel_power_samples": 1,
        "hw_accel_monitor_attempts": 1,
        "hw_accel_monitor_successes": 1,
        "hw_accel_monitor_coverage": 1.0,
    }
    assert (
        collector.get_static_info()["hw_accel_name"]
        == "RBLN-CA22-RESTARTED"
    )


@pytest.mark.parametrize(
    ("resolver_failure", "message"),
    [
        (None, "rbln-smi executable"),
        (RuntimeError("resolver unavailable"), "resolver unavailable"),
    ],
)
def test_restart_resolver_failure_clears_stale_public_state(
    user_payload,
    resolver_failure,
    message,
):
    resolutions = iter(("/usr/bin/rbln-smi", resolver_failure))

    def resolver(name):
        assert name == "rbln-smi"
        action = next(resolutions)
        if isinstance(action, BaseException):
            raise action
        return action

    runner = FakeRunner(
        [
            user_payload,
            user_payload,
            subprocess.TimeoutExpired(["rbln-smi"], 2.0),
        ]
    )
    collector = RblnCollector(
        runner=runner,
        clock=FakeClock(),
        executable_resolver=resolver,
    )
    collector.start()
    collector.collect(force=True)
    collector.stop()
    assert collector.get_static_info()
    assert "hw_accel_monitor_note" in collector.get_summary_metrics()

    with pytest.raises(RuntimeError, match=message):
        collector.start()

    assert collector.get_static_info() == {}
    assert collector.get_summary_metrics() == {
        "hw_accel_power_samples": 0,
        "hw_accel_monitor_attempts": 0,
        "hw_accel_monitor_successes": 0,
        "hw_accel_monitor_coverage": 0.0,
    }
    assert collector.collect(force=True) == {}


def test_already_started_guard_preserves_active_measurements(user_payload):
    collector = make_collector(FakeRunner([user_payload]))
    collector.start()

    with pytest.raises(RuntimeError, match="already started"):
        collector.start()

    assert collector.get_static_info()["hw_accel_name"] == "RBLN-CA22"
    assert collector.get_summary_metrics()["hw_accel_monitor_attempts"] == 1


def test_string_unit_variants_are_parsed_without_unsafe_coercion(user_payload):
    payload = deepcopy(user_payload)
    device = payload["devices"][0]
    device["util"] = "37.5 %"
    device["temperature"] = "42 C"
    device["card_power"] = "12500 mW"
    device["memory"] = {"used": "2 MiB", "total": "4 GiB"}
    payload["contexts"] = [
        {"npu": 0, "pid": os.getpid(), "memalloc": "1 MiB"}
    ]
    collector = make_collector(FakeRunner([payload, payload]))

    collector.start()
    metrics = collector.collect(force=True)

    assert metrics == {
        "hw_accel_util": 37.5,
        "hw_accel_mem_used_mb": 2.0,
        "hw_accel_mem_proc_mb": 1.0,
        "hw_accel_temp_c": 42.0,
        "hw_accel_power_w": 12.5,
    }
    assert collector.get_static_info()["hw_accel_mem_total_mb"] == 4096.0
