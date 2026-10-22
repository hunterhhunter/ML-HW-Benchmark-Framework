"""Furiosa raw-power boundary for the server trace."""

import importlib

from monitors import create_hw_monitor, get_collector_entry


def test_furiosa_power_source_preserves_vendor_scope():
    collector_module = importlib.import_module("monitors.furiosa_collector")
    source = collector_module.FuriosaCollector("npu0").power_trace_source()
    assert source.monitor_source == "furiosa-smi-py"
    assert source.power_scope == "device"
    assert source.device_id == "npu0"


def test_power_only_monitor_factory_keeps_summary_disabled():
    monitor = create_hw_monitor(
        collector_names=[],
        summary_enabled=False,
        power_trace_enabled=True,
    )
    assert monitor.startup_timeout_hint_sec() > 0
    assert get_collector_entry("furiosa").class_name == "FuriosaCollector"
