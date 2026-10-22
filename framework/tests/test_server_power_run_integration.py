"""SDK-free run integration for reserved CSV, raw power and async sidecar."""

import csv
import hashlib
import json
from types import SimpleNamespace

import pytest

import main as benchmark_main
import core.result_store as result_store
from core.async_inference import AsyncBenchmarkResult, RunStatus
from core.power_trace import PowerReading, PowerTraceSource
from monitors.base import Collector, HWMonitor


class _Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += int(seconds * 1_000_000_000)


class _PowerCollector(Collector):
    def start(self):
        return None

    def stop(self):
        return None

    def collect(self):
        return {}

    def power_trace_source(self):
        return PowerTraceSource(
            collector="mock",
            monitor_source="mock-power-api",
            device_id="0",
            power_scope="device",
            sample_interval_sec=0.1,
        )

    def collect_power(self):
        return PowerReading(status="ok", power_w=12.5)


@pytest.fixture
def monitor(monkeypatch):
    class _DormantThread:
        def __init__(self, *, target, daemon):
            pass

        def start(self):
            pass

        def join(self, timeout):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr("monitors.base.threading.Thread", _DormantThread)
    clock = _Clock()
    hw_monitor = HWMonitor(
        summary_enabled=False,
        power_trace_enabled=True,
        power_trace_baseline_sec=0.01,
        clock_ns=clock,
        wait_fn=clock.advance,
    )
    hw_monitor.add_collector(_PowerCollector())
    return hw_monitor


def _args(results_path, mode):
    values = [
        "--model", "resnet50", "--target", "cpu",
        "--results-path", str(results_path), "--power-trace",
        "--warmup", "0", "--inference-mode", mode,
    ]
    if mode == "async_queue":
        values += [
            "--async-pass", "runtime-only", "--scenario", "offline",
            "--min-samples", "1", "--max-samples", "1",
        ]
    args = benchmark_main.build_parser().parse_args(values)
    args.backend = "mock"
    args.device = "0"
    return args


def _target_meta():
    return {
        "target_id": "mock-npu",
        "accelerator_vendor": "Mock",
        "accelerator_name": "Mock NPU",
        "runtime_name": "mock",
        "compiler_name": "",
        "artifact_format": "mockbin",
    }


def _assert_linked_artifacts(results_path, *, async_mode):
    row = list(csv.DictReader(results_path.open(newline="", encoding="utf-8")))[-1]
    trace_path = results_path.parent / row["power_trace_path"]
    trace = trace_path.read_bytes()
    samples = list(csv.DictReader(trace_path.open(newline="", encoding="utf-8")))
    assert row["power_trace_status"] == "complete"
    assert row["power_trace_sha256"] == hashlib.sha256(trace).hexdigest()
    assert int(row["power_trace_sample_count"]) == len(samples)
    assert {sample["run_id"] for sample in samples} == {row["run_id"]}
    assert {sample["phase"] for sample in samples} == {"baseline", "inference"}
    assert {sample["sample_status"] for sample in samples} == {"ok"}
    if async_mode:
        details = json.loads(
            (results_path.parent.parent / row["details_path"]).read_text()
        )
        assert details["power_trace"]["power_trace_path"] == row["power_trace_path"]
        assert details["power_trace"]["power_trace_sha256"] == row["power_trace_sha256"]
        assert row["quality_status"] == "skipped"
        assert row["async_run_status"] == "valid"
    return row


def test_e2e_links_real_result_store_and_raw_power(monkeypatch, tmp_path, monitor):
    results_path = tmp_path / "e2e.csv"

    class _Runner:
        def __init__(self, **kwargs):
            self.monitor = kwargs["monitor"]

        def run(self, **kwargs):
            self.monitor.start()
            self.monitor.stop()
            return {"accuracy": 1.0}

    monkeypatch.setattr(benchmark_main, "BenchmarkRunner", _Runner)
    runtime = SimpleNamespace(unload=lambda: None, get_device_spec=lambda: {})
    assert benchmark_main.execute_benchmark(
        _args(results_path, "e2e"),
        target=SimpleNamespace(target_id="mock-npu", capabilities=("sync",)),
        loader=object(), runtime=runtime, evaluator=object(), decoder=None,
        hw_monitor=monitor, task_name="IMAGE_CLASSIFICATION",
        target_meta=_target_meta(), results_path=results_path,
    ) == 0
    assert _assert_linked_artifacts(results_path, async_mode=False)["inference_mode"] == "e2e"


def test_failed_e2e_links_published_trace_to_failed_result(
    monkeypatch, tmp_path, monitor
):
    results_path = tmp_path / "failed-e2e.csv"

    class _Runner:
        def __init__(self, **kwargs):
            self.monitor = kwargs["monitor"]

        def run(self, **kwargs):
            self.monitor.start()
            self.monitor.stop()
            raise RuntimeError("inference failed")

    monkeypatch.setattr(benchmark_main, "BenchmarkRunner", _Runner)
    runtime = SimpleNamespace(unload=lambda: None, get_device_spec=lambda: {})
    with pytest.raises(RuntimeError, match="inference failed") as raised:
        benchmark_main.execute_benchmark(
            _args(results_path, "e2e"),
            target=SimpleNamespace(target_id="mock-npu", capabilities=("sync",)),
            loader=object(), runtime=runtime, evaluator=object(), decoder=None,
            hw_monitor=monitor, task_name="IMAGE_CLASSIFICATION",
            target_meta=_target_meta(), results_path=results_path,
        )

    assert not getattr(raised.value, "cleanup_secondary_errors", [])
    row = _assert_linked_artifacts(results_path, async_mode=False)
    assert row["quality_status"] == "failed"
    assert row["benchmark_run_status"] == "failed"
    assert row["failure_details_path"] == ""
    assert not list((tmp_path / "details").glob("*.failure.json"))


def test_e2e_csv_save_failure_recovers_trace_link(monkeypatch, tmp_path, monitor):
    results_path = tmp_path / "save-failed-e2e.csv"

    class _Runner:
        def __init__(self, **kwargs):
            self.monitor = kwargs["monitor"]

        def run(self, **kwargs):
            self.monitor.start()
            self.monitor.stop()
            return {"accuracy": 1.0}

    real_save = benchmark_main.save_result
    attempts = 0

    def fail_once(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("initial CSV save failed")
        return real_save(**kwargs)

    monkeypatch.setattr(benchmark_main, "BenchmarkRunner", _Runner)
    monkeypatch.setattr(benchmark_main, "save_result", fail_once)
    runtime = SimpleNamespace(unload=lambda: None, get_device_spec=lambda: {})
    with pytest.raises(OSError, match="initial CSV save failed"):
        benchmark_main.execute_benchmark(
            _args(results_path, "e2e"),
            target=SimpleNamespace(target_id="mock-npu", capabilities=("sync",)),
            loader=object(), runtime=runtime, evaluator=object(), decoder=None,
            hw_monitor=monitor, task_name="IMAGE_CLASSIFICATION",
            target_meta=_target_meta(), results_path=results_path,
        )
    assert attempts == 2
    row = _assert_linked_artifacts(results_path, async_mode=False)
    assert row["benchmark_run_status"] == "failed"


def test_e2e_pending_csv_retry_uses_original_row(monkeypatch, tmp_path, monitor):
    results_path = tmp_path / "pending-e2e.csv"

    class _Runner:
        def __init__(self, **kwargs):
            self.monitor = kwargs["monitor"]

        def run(self, **kwargs):
            self.monitor.start()
            self.monitor.stop()
            return {"accuracy": 1.0}

    real_atomic_write = result_store._atomic_write_csv_at
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("CSV publication interrupted after pending")
        return real_atomic_write(*args, **kwargs)

    monkeypatch.setattr(benchmark_main, "BenchmarkRunner", _Runner)
    monkeypatch.setattr(result_store, "_atomic_write_csv_at", fail_once)
    runtime = SimpleNamespace(unload=lambda: None, get_device_spec=lambda: {})
    with pytest.raises(OSError, match="after pending") as raised:
        benchmark_main.execute_benchmark(
            _args(results_path, "e2e"),
            target=SimpleNamespace(target_id="mock-npu", capabilities=("sync",)),
            loader=object(), runtime=runtime, evaluator=object(), decoder=None,
            hw_monitor=monitor, task_name="IMAGE_CLASSIFICATION",
            target_meta=_target_meta(), results_path=results_path,
        )

    assert attempts == 2
    assert not getattr(raised.value, "cleanup_secondary_errors", [])
    row = _assert_linked_artifacts(results_path, async_mode=False)
    assert row.get("benchmark_run_status") != "failed"
    assert row["failure_details_path"] == ""
    assert not list((tmp_path / "details").glob("*.failure.json"))


def test_e2e_consumed_csv_does_not_publish_orphan_failure_sidecar(
    monkeypatch, tmp_path, monitor
):
    results_path = tmp_path / "consumed-e2e.csv"

    class _Runner:
        def __init__(self, **kwargs):
            self.monitor = kwargs["monitor"]

        def run(self, **kwargs):
            self.monitor.start()
            self.monitor.stop()
            return {"accuracy": 1.0}

    real_save = benchmark_main.save_result
    attempts = 0

    def save_then_raise(**kwargs):
        nonlocal attempts
        attempts += 1
        real_save(**kwargs)
        raise OSError("post-commit acknowledgement failed")

    monkeypatch.setattr(benchmark_main, "BenchmarkRunner", _Runner)
    monkeypatch.setattr(benchmark_main, "save_result", save_then_raise)
    runtime = SimpleNamespace(unload=lambda: None, get_device_spec=lambda: {})
    with pytest.raises(OSError, match="post-commit acknowledgement failed"):
        benchmark_main.execute_benchmark(
            _args(results_path, "e2e"),
            target=SimpleNamespace(target_id="mock-npu", capabilities=("sync",)),
            loader=object(), runtime=runtime, evaluator=object(), decoder=None,
            hw_monitor=monitor, task_name="IMAGE_CLASSIFICATION",
            target_meta=_target_meta(), results_path=results_path,
        )

    assert attempts == 1
    row = _assert_linked_artifacts(results_path, async_mode=False)
    assert row["failure_details_path"] == ""
    assert not list((tmp_path / "details").glob("*.failure.json"))


def test_async_runtime_only_links_result_sidecar_and_raw_power(
    monkeypatch, tmp_path, monitor
):
    results_path = tmp_path / "async.csv"

    class _Engine:
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            pass

        def run_async(self, config, **kwargs):
            hw_monitor = kwargs["monitor"]
            hw_monitor.start()
            hw_monitor.stop()
            return AsyncBenchmarkResult(
                metrics={
                    "async_submitted_requests": 1,
                    "async_accepted_requests": 1,
                    "async_completed_requests": 1,
                    "async_completed_samples": 1,
                    "async_runtime_completed_samples": 1,
                    "async_outstanding_requests": 0,
                    "async_failed_requests": 0,
                },
                details={}, status=RunStatus.VALID,
            )

    monkeypatch.setattr(benchmark_main, "InferenceEngine", _Engine)
    monkeypatch.setattr(
        benchmark_main, "_build_async_runtime_executor", lambda *args, **kwargs: object()
    )
    runtime = SimpleNamespace(unload=lambda: None, get_device_spec=lambda: {})
    assert benchmark_main.execute_benchmark(
        _args(results_path, "async_queue"),
        target=SimpleNamespace(target_id="mock-npu", capabilities=("native_async",)),
        loader=object(), runtime=runtime, evaluator=None, decoder=None,
        hw_monitor=monitor, task_name="IMAGE_CLASSIFICATION",
        target_meta=_target_meta(), results_path=results_path,
    ) == 0
    row = _assert_linked_artifacts(results_path, async_mode=True)
    assert int(row["async_completed_samples"]) == 1
