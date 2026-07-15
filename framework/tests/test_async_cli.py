import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import main as benchmark_main
import core.async_inference.runner as async_runner_module
from core.async_inference import AsyncBenchmarkResult, RunStatus


def parse(extra):
    return benchmark_main.build_parser().parse_args(
        ["--model", "resnet50", *extra]
    )


def test_default_inference_mode_is_e2e():
    assert parse([]).inference_mode == "e2e"


@pytest.mark.parametrize(
    "extra",
    [
        ["--scenario", "offline"],
        ["--queue-capacity", "16"],
        ["--save-request-trace"],
    ],
)
def test_e2e_rejects_async_only_options(extra):
    with pytest.raises(ValueError, match="async_queue"):
        benchmark_main.validate_async_args(parse(extra))


def test_server_like_requires_target_qps():
    args = parse(
        ["--inference-mode", "async_queue", "--scenario", "server_like"]
    )

    with pytest.raises(ValueError, match="target-qps"):
        benchmark_main.validate_async_args(args)


def test_async_rejects_max_steps_and_points_to_max_samples():
    args = parse(
        ["--inference-mode", "async_queue", "--max-steps", "2"]
    )

    with pytest.raises(ValueError, match="max-samples"):
        benchmark_main.validate_async_args(args)


def _async_args(*extra):
    args = parse(
        [
            "--inference-mode",
            "async_queue",
            "--min-samples",
            "1",
            "--max-samples",
            "1",
            *extra,
        ]
    )
    benchmark_main.validate_async_args(args)
    return args


def _reservation(tmp_path):
    root = tmp_path / "results"
    return SimpleNamespace(
        run_id="async001",
        results_root=root,
        results_path=root / "benchmark_results.csv",
        details_path=root / "details" / "async001.json",
        trace_path=root / "traces" / "async001.jsonl",
    )


def _result(*, status=RunStatus.VALID, outstanding=0):
    reasons = () if status is RunStatus.VALID else ("synthetic_invalid",)
    return AsyncBenchmarkResult(
        metrics={
            "accuracy": 1.0,
            "async_outstanding_requests": outstanding,
            "async_run_status": status.value,
            "async_invalid_reasons": ",".join(reasons),
        },
        details={
            "invalid_reasons": list(reasons),
            "warnings": [],
            "status": status.value,
        },
        status=status,
        invalid_reasons=reasons,
        warnings=(),
    )


def _execute(
    args,
    tmp_path,
    *,
    monkeypatch,
    result=None,
    events=None,
    detail_error=None,
    csv_error=None,
):
    events = [] if events is None else events
    reservation = _reservation(tmp_path)
    runtime = SimpleNamespace(unload=lambda: events.append("unload"))
    saved = {}

    def reserve(*, results_path, run_id=None):
        events.append(("reserve", Path(results_path), run_id))
        return reservation

    class Runner:
        def __init__(self, **kwargs):
            events.append(("async_init", kwargs))

        def run(self, config, warmup_runs):
            events.append(("async_run", config, warmup_runs))
            return result or _result()

    def save_details(run_id, details, *, results_dir, reservation):
        events.append(("details", run_id, reservation, results_dir))
        saved["details"] = details
        if detail_error is not None:
            raise detail_error
        return reservation.details_path

    def save_csv(**kwargs):
        events.append(("csv", kwargs["run_id"], kwargs["reservation"]))
        saved["csv"] = kwargs
        if csv_error is not None:
            raise csv_error
        return kwargs["run_id"]

    monkeypatch.setattr(benchmark_main, "reserve_run_artifacts", reserve)
    monkeypatch.setattr(benchmark_main, "AsyncBenchmarkRunner", Runner)
    monkeypatch.setattr(benchmark_main, "save_async_details", save_details)
    monkeypatch.setattr(benchmark_main, "save_result", save_csv)

    exit_code = benchmark_main.execute_benchmark(
        args,
        loader=object(),
        runtime=runtime,
        evaluator=object(),
        decoder=object(),
        hw_monitor=None,
        task_name="IMAGE_CLASSIFICATION",
        target_meta={
            "target_id": "cpu",
            "accelerator_vendor": "",
            "accelerator_name": "CPU",
            "runtime_name": "onnxruntime",
            "compiler_name": "",
            "artifact_format": "onnx",
        },
        results_path=reservation.results_path,
    )
    return exit_code, events, saved, reservation


def test_async_branch_reserves_before_measurement_and_propagates_token(
    monkeypatch, tmp_path, capsys
):
    exit_code, events, saved, reservation = _execute(
        _async_args(), tmp_path, monkeypatch=monkeypatch
    )

    assert exit_code == 0
    names = [event[0] if isinstance(event, tuple) else event for event in events]
    assert names.index("reserve") < names.index("async_run")
    assert names.index("async_run") < names.index("details")
    assert names.index("details") < names.index("csv") < names.index("unload")
    assert saved["csv"]["reservation"] is reservation
    assert saved["csv"]["run_id"] == reservation.run_id
    assert saved["csv"]["results_path"] == reservation.results_path
    assert saved["csv"]["details_path"] == "results/details/async001.json"
    assert saved["csv"]["inference_mode"] == "async_queue"
    assert capsys.readouterr().out.rstrip().endswith("RUN_ID=async001")


def test_trace_uses_reserved_path_and_same_token(monkeypatch, tmp_path):
    args = _async_args("--save-request-trace")
    events = []
    writer_instances = []

    class TraceWriter:
        def __init__(self, path, reservation):
            self.path = path
            self.reservation = reservation
            self.dropped = 0
            self.error = None
            writer_instances.append(self)

        def start(self):
            events.append("trace_start")

        def write(self, value):
            pass

        def close(self, timeout):
            events.append(("trace_close", timeout))
            return True

    monkeypatch.setattr(benchmark_main, "RequestTraceWriter", TraceWriter)
    exit_code, events, saved, reservation = _execute(
        args,
        tmp_path,
        monkeypatch=monkeypatch,
        events=events,
    )

    assert exit_code == 0
    assert writer_instances[0].path == reservation.trace_path
    assert writer_instances[0].reservation is reservation
    assert saved["csv"]["request_trace_path"] == (
        "results/traces/async001.jsonl"
    )
    assert events.index("trace_start") < next(
        index for index, event in enumerate(events) if event[0] == "async_run"
    )
    assert next(
        index for index, event in enumerate(events) if event[0] == "trace_close"
    ) < next(index for index, event in enumerate(events) if event[0] == "details")


def test_e2e_keeps_legacy_runner_and_save_contract(
    monkeypatch, tmp_path, capsys
):
    args = parse(["--max-steps", "3"])
    events = []
    saved = {}
    runtime = SimpleNamespace(unload=lambda: events.append("unload"))

    class Runner:
        def __init__(self, **kwargs):
            events.append(("e2e_init", kwargs))

        def run(self, **kwargs):
            events.append(("e2e_run", kwargs))
            return {"accuracy": 1.0}

    def reject_reservation(**kwargs):
        pytest.fail("e2e must not reserve async artifacts")

    def save_csv(**kwargs):
        saved.update(kwargs)
        events.append("csv")
        return "e2e0001"

    monkeypatch.setattr(benchmark_main, "BenchmarkRunner", Runner)
    monkeypatch.setattr(
        benchmark_main, "reserve_run_artifacts", reject_reservation
    )
    monkeypatch.setattr(benchmark_main, "save_result", save_csv)

    exit_code = benchmark_main.execute_benchmark(
        args,
        loader=object(),
        runtime=runtime,
        evaluator=object(),
        decoder=object(),
        hw_monitor=None,
        task_name="IMAGE_CLASSIFICATION",
        target_meta={
            "target_id": "cpu",
            "accelerator_vendor": "",
            "accelerator_name": "CPU",
            "runtime_name": "onnxruntime",
            "compiler_name": "",
            "artifact_format": "onnx",
        },
    )

    assert exit_code == 0
    assert events[-2:] == ["csv", "unload"]
    assert "reservation" not in saved
    assert "inference_mode" not in saved
    assert saved["max_steps"] == 3
    assert capsys.readouterr().out.rstrip().endswith("RUN_ID=e2e0001")


@pytest.mark.parametrize(
    ("result", "expected_unload"),
    [
        (_result(status=RunStatus.INVALID), True),
        (_result(status=RunStatus.INVALID, outstanding=2), False),
    ],
)
def test_invalid_and_outstanding_runs_persist_before_nonzero_exit(
    monkeypatch, tmp_path, result, expected_unload
):
    exit_code, events, _, _ = _execute(
        _async_args(),
        tmp_path,
        monkeypatch=monkeypatch,
        result=result,
    )
    names = [event[0] if isinstance(event, tuple) else event for event in events]

    assert exit_code == 1
    assert names.index("details") < names.index("csv")
    assert ("unload" in names) is expected_unload
    if expected_unload:
        assert names.index("csv") < names.index("unload")


def test_trace_close_failure_is_persisted_and_returns_nonzero(
    monkeypatch, tmp_path
):
    args = _async_args("--save-request-trace")

    class TraceWriter:
        dropped = 0
        error = {"phase": "close", "error_type": "TimeoutError"}

        def __init__(self, path, reservation):
            pass

        def start(self):
            pass

        def write(self, value):
            pass

        def close(self, timeout):
            return False

    monkeypatch.setattr(benchmark_main, "RequestTraceWriter", TraceWriter)
    exit_code, _, saved, _ = _execute(
        args, tmp_path, monkeypatch=monkeypatch
    )

    assert exit_code == 1
    assert saved["csv"]["request_trace_path"] == ""
    assert "request_trace_persistence_failed" in saved["csv"][
        "async_invalid_reasons"
    ]
    assert saved["details"]["persistence_errors"][0]["phase"] == "close"


def test_false_trace_close_with_certain_commit_keeps_trace_link_but_fails(
    monkeypatch, tmp_path
):
    args = _async_args("--save-request-trace")

    class TraceWriter:
        dropped = 0
        error = {
            "phase": "close",
            "error_type": "CloseDiagnostic",
            "error_message": "directory descriptor close failed",
            "final_file_committed": True,
            "publication_state_uncertain": False,
        }

        def __init__(self, path, reservation):
            pass

        def start(self):
            pass

        def write(self, value):
            pass

        def close(self, timeout):
            return False

    monkeypatch.setattr(benchmark_main, "RequestTraceWriter", TraceWriter)

    exit_code, _, saved, _ = _execute(
        args, tmp_path, monkeypatch=monkeypatch
    )

    assert exit_code == 1
    assert saved["csv"]["request_trace_path"] == (
        "results/traces/async001.jsonl"
    )
    assert "request_trace_persistence_failed" in saved["csv"][
        "async_invalid_reasons"
    ]
    assert saved["details"]["persistence_errors"][0][
        "final_file_committed"
    ] is True


def test_runner_exception_closes_trace_without_masking_original(
    monkeypatch, tmp_path
):
    args = _async_args("--save-request-trace")
    reservation = _reservation(tmp_path)
    closed = []

    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )

    class TraceWriter:
        def __init__(self, path, reservation):
            pass

        def start(self):
            pass

        def write(self, value):
            pass

        def close(self, timeout):
            closed.append(timeout)
            raise RuntimeError("secondary trace error")

    class Runner:
        def __init__(self, **kwargs):
            pass

        def run(self, config, warmup_runs):
            raise LookupError("primary runner error")

    monkeypatch.setattr(benchmark_main, "RequestTraceWriter", TraceWriter)
    monkeypatch.setattr(benchmark_main, "AsyncBenchmarkRunner", Runner)

    with pytest.raises(LookupError, match="primary runner error") as raised:
        benchmark_main.execute_benchmark(
            args,
            loader=object(),
            runtime=SimpleNamespace(unload=lambda: None),
            evaluator=object(),
            decoder=object(),
            hw_monitor=None,
            task_name="IMAGE_CLASSIFICATION",
            target_meta={
                "target_id": "cpu",
                "accelerator_vendor": "",
                "accelerator_name": "CPU",
                "runtime_name": "onnxruntime",
                "compiler_name": "",
                "artifact_format": "onnx",
            },
            results_path=reservation.results_path,
        )

    assert closed
    assert "secondary trace error" in "\n".join(raised.value.__notes__)


def _execute_with_runtime(args, tmp_path, runtime):
    return benchmark_main.execute_benchmark(
        args,
        loader=object(),
        runtime=runtime,
        evaluator=object(),
        decoder=object(),
        hw_monitor=None,
        task_name="IMAGE_CLASSIFICATION",
        target_meta={
            "target_id": "cpu",
            "accelerator_vendor": "",
            "accelerator_name": "CPU",
            "runtime_name": "onnxruntime",
            "compiler_name": "",
            "artifact_format": "onnx",
        },
        results_path=tmp_path / "results" / "benchmark_results.csv",
    )


def test_reservation_setup_failure_unloads_runtime_and_preserves_primary(
    monkeypatch, tmp_path
):
    primary = LookupError("reservation failed")
    unloads = []
    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: (_ for _ in ()).throw(primary),
    )

    with pytest.raises(LookupError) as raised:
        _execute_with_runtime(
            _async_args(),
            tmp_path,
            SimpleNamespace(unload=lambda: unloads.append(True)),
        )

    assert raised.value is primary
    assert unloads == [True]


def test_trace_constructor_failure_unloads_runtime_and_preserves_primary(
    monkeypatch, tmp_path
):
    primary = RuntimeError("trace constructor failed")
    unloads = []
    reservation = _reservation(tmp_path)
    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )

    class TraceWriter:
        def __init__(self, path, reservation):
            raise primary

    monkeypatch.setattr(benchmark_main, "RequestTraceWriter", TraceWriter)

    with pytest.raises(RuntimeError) as raised:
        _execute_with_runtime(
            _async_args("--save-request-trace"),
            tmp_path,
            SimpleNamespace(unload=lambda: unloads.append(True)),
        )

    assert raised.value is primary
    assert unloads == [True]


def test_trace_start_failure_attempts_bounded_close_and_runtime_unload(
    monkeypatch, tmp_path
):
    primary = RuntimeError("trace start failed")
    events = []
    reservation = _reservation(tmp_path)
    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )

    class TraceWriter:
        error = {"phase": "start", "error_type": "RuntimeError"}

        def __init__(self, path, reservation):
            pass

        def start(self):
            events.append("start")
            raise primary

        def close(self, timeout):
            events.append(("close", timeout))
            return False

    monkeypatch.setattr(benchmark_main, "RequestTraceWriter", TraceWriter)

    with pytest.raises(RuntimeError) as raised:
        _execute_with_runtime(
            _async_args("--save-request-trace"),
            tmp_path,
            SimpleNamespace(unload=lambda: events.append("unload")),
        )

    assert raised.value is primary
    assert events == ["start", ("close", 300.0), "unload"]
    assert "request_trace_cleanup" in "\n".join(primary.__notes__)


def test_setup_cleanup_failure_is_secondary_to_original_error(
    monkeypatch, tmp_path
):
    primary = LookupError("reservation failed")
    secondary = OSError("unload failed")
    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: (_ for _ in ()).throw(primary),
    )

    with pytest.raises(LookupError) as raised:
        _execute_with_runtime(
            _async_args(),
            tmp_path,
            SimpleNamespace(unload=lambda: (_ for _ in ()).throw(secondary)),
        )

    assert raised.value is primary
    assert primary.cleanup_secondary_errors[0]["phase"] == "runtime_unload"
    assert primary.cleanup_secondary_errors[0]["error_type"] == "OSError"


def test_real_runner_warmup_failure_unloads_and_preserves_primary(
    monkeypatch,
    tmp_path,
):
    primary = RuntimeError("warmup failed")
    unloads = []

    class Loader:
        current_idx = 0

        def get_metadata(self):
            return {"total_samples": 1, "is_static_batched": False}

        def load_batch(self, batch_size):
            del batch_size
            self.current_idx = 1
            return [{"input": np.array([1.0]), "label": 1}]

    class Runtime:
        compiled_model = None

        def supports_generate(self):
            return False

        def max_concurrent_workers(self):
            return 1

        def max_dynamic_batch_size(self):
            return 1

        def supports_dynamic_batching(self):
            return False

        def supports_batch_generation(self):
            return False

        def warmup(self, inputs, num_runs):
            del inputs, num_runs
            raise primary

        def unload(self):
            unloads.append("unload")

    reservation = _reservation(tmp_path)
    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    loader = Loader()

    with pytest.raises(RuntimeError) as raised:
        benchmark_main.execute_benchmark(
            _async_args(),
            loader=loader,
            runtime=Runtime(),
            evaluator=object(),
            decoder=object(),
            hw_monitor=None,
            task_name="IMAGE_CLASSIFICATION",
            target_meta={},
            results_path=reservation.results_path,
        )

    assert raised.value is primary
    assert unloads == ["unload"]
    assert loader.current_idx == 0


def test_real_runner_active_failure_skips_unload_without_cleanup_proof(
    monkeypatch,
    tmp_path,
):
    class FatalRun(BaseException):
        pass

    primary = FatalRun("worker start failed")
    unloads = []

    class Loader:
        def get_metadata(self):
            return {"total_samples": 1, "is_static_batched": False}

    class Runtime:
        compiled_model = None

        def supports_generate(self):
            return False

        def unload(self):
            unloads.append("unload")

    class UnsafeEngine:
        def __init__(self, runtime, pipeline, config, coordinator, metrics):
            del runtime, pipeline, config, coordinator, metrics

        def start(self):
            raise primary

        def close_submission(self):
            pass

        def flush(self):
            return True

        def shutdown(self):
            return False

        def outstanding_request_ids(self):
            return (7,)

    reservation = _reservation(tmp_path)
    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    monkeypatch.setattr(
        async_runner_module,
        "AsyncInferenceEngine",
        UnsafeEngine,
    )

    with pytest.raises(FatalRun) as raised:
        benchmark_main.execute_benchmark(
            _async_args("--warmup", "0"),
            loader=Loader(),
            runtime=Runtime(),
            evaluator=object(),
            decoder=object(),
            hw_monitor=None,
            task_name="IMAGE_CLASSIFICATION",
            target_meta={},
            results_path=reservation.results_path,
        )

    assert raised.value is primary
    assert unloads == []


def test_safe_runner_failure_keeps_unload_error_secondary(
    monkeypatch,
    tmp_path,
):
    primary = LookupError("warmup failed")
    secondary = OSError("unload failed")

    class Runner:
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            del kwargs

        def run(self, config, warmup_runs):
            del config, warmup_runs
            raise primary

    monkeypatch.setattr(benchmark_main, "AsyncBenchmarkRunner", Runner)

    with pytest.raises(LookupError) as raised:
        _execute_with_runtime(
            _async_args(),
            tmp_path,
            SimpleNamespace(
                unload=lambda: (_ for _ in ()).throw(secondary)
            ),
        )

    assert raised.value is primary
    assert primary.cleanup_secondary_errors[-1]["phase"] == "runtime_unload"
    assert primary.cleanup_secondary_errors[-1]["error_type"] == "OSError"


def test_invalid_async_config_fails_before_artifact_reservation(
    monkeypatch, tmp_path
):
    args = _async_args("--queue-capacity", "0")
    called = False

    def reserve(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(benchmark_main, "reserve_run_artifacts", reserve)

    with pytest.raises(ValueError, match="queue_capacity"):
        benchmark_main.execute_benchmark(
            args,
            loader=object(),
            runtime=SimpleNamespace(unload=lambda: None),
            evaluator=object(),
            decoder=object(),
            hw_monitor=None,
            task_name="IMAGE_CLASSIFICATION",
            target_meta={},
            results_path=tmp_path / "results" / "benchmark_results.csv",
        )

    assert called is False


def test_main_reports_invalid_async_config_before_target_or_runtime_setup(
    monkeypatch,
):
    target_called = False

    def resolve(*args, **kwargs):
        nonlocal target_called
        target_called = True

    monkeypatch.setattr(benchmark_main, "resolve_target", resolve)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--model",
            "resnet50",
            "--inference-mode",
            "async_queue",
            "--queue-capacity",
            "0",
        ],
    )

    with pytest.raises(SystemExit) as raised:
        benchmark_main.main()

    assert raised.value.code == 2
    assert target_called is False


def test_committed_sidecar_close_failure_is_linked_but_never_success(
    monkeypatch, tmp_path
):
    error = OSError("details directory close failed")
    error.final_file_committed = True
    error.publication_state_uncertain = False

    exit_code, events, saved, _ = _execute(
        _async_args(),
        tmp_path,
        monkeypatch=monkeypatch,
        detail_error=error,
    )

    assert exit_code == 1
    assert saved["csv"]["details_path"] == "results/details/async001.json"
    assert "async_details_persistence_failed" in saved["csv"][
        "async_invalid_reasons"
    ]
    assert events[-1] == "unload"


def test_hostile_sidecar_exception_is_safely_diagnosed_and_csv_still_runs(
    monkeypatch, tmp_path, capsys
):
    class HostilePersistenceError(Exception):
        def __str__(self):
            raise AssertionError("must not call hostile __str__")

        def __getattribute__(self, name):
            if name in {"final_file_committed", "publication_state_uncertain"}:
                raise AssertionError("must not use dynamic attribute lookup")
            return super().__getattribute__(name)

    error = HostilePersistenceError("safe detail failure")

    exit_code, events, saved, _ = _execute(
        _async_args(),
        tmp_path,
        monkeypatch=monkeypatch,
        detail_error=error,
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert saved["csv"]["details_path"] == ""
    assert saved["details"]["persistence_errors"][0] == {
        "phase": "save_async_details",
        "error_type": "HostilePersistenceError",
        "error_message": "safe detail failure",
        "final_file_committed": False,
        "publication_state_uncertain": False,
    }
    names = [event[0] if isinstance(event, tuple) else event for event in events]
    assert names.index("details") < names.index("csv") < names.index("unload")
    assert "save_async_details" in captured.err


def test_uncertain_csv_commit_returns_nonzero_after_sidecar_and_keeps_run_id(
    monkeypatch, tmp_path, capsys
):
    error = OSError("results directory fsync failed")
    error.publication_state_uncertain = True

    exit_code, events, _, reservation = _execute(
        _async_args(),
        tmp_path,
        monkeypatch=monkeypatch,
        csv_error=error,
    )
    captured = capsys.readouterr()
    names = [event[0] if isinstance(event, tuple) else event for event in events]

    assert exit_code == 1
    assert names.index("details") < names.index("csv") < names.index("unload")
    assert f"RUN_ID={reservation.run_id}" in captured.out
    assert "결과 CSV 저장 실패" in captured.err
