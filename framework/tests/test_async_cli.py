import json
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


def test_results_path_is_common_and_defaults_to_none(tmp_path):
    assert parse([]).results_path is None
    chosen = tmp_path / "isolated" / "results.csv"
    assert parse(["--results-path", str(chosen)]).results_path == str(chosen)


def test_results_path_is_not_rejected_in_e2e_mode(tmp_path):
    args = parse(["--results-path", str(tmp_path / "results.csv")])
    benchmark_main.validate_async_args(args)


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
    csv_run_id=None,
):
    events = [] if events is None else events
    reservation = _reservation(tmp_path)
    runtime = SimpleNamespace(
        unload=lambda: events.append("unload"),
        get_device_spec=lambda: {
            "backend": "onnxruntime",
            "active_providers": ["CPUExecutionProvider"],
        },
    )
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
        return kwargs["run_id"] if csv_run_id is None else csv_run_id

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
    args = _async_args()
    exit_code, events, saved, reservation = _execute(
        args, tmp_path, monkeypatch=monkeypatch
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
    assert saved["details"]["run"] == {
        "model_name": "resnet50",
        "task": "IMAGE_CLASSIFICATION",
        "backend": args.backend,
        "device": args.device,
        "batch_size": args.batch_size,
        "warmup_runs": args.warmup,
        "target_id": "cpu",
        "dataset_path": str(args.dataset or ""),
        "model_artifact_path": str(
            args.onnx
            or args.hef
            or args.artifact
            or args.model_path
            or ""
        ),
        "runtime_device_spec": {
            "backend": "onnxruntime",
            "active_providers": ["CPUExecutionProvider"],
        },
    }
    lines = capsys.readouterr().out.splitlines()
    assert lines.count("RUN_ID_RESERVED=async001") == 1
    assert lines.count("RUN_ID=async001") == 1
    assert lines.index("RUN_ID_RESERVED=async001") < lines.index(
        "RUN_ID=async001"
    )


def test_debug_paths_report_reserved_artifacts_without_request_ids(
    monkeypatch, tmp_path, capsys
):
    _, _, _, reservation = _execute(
        _async_args("--debug"),
        tmp_path,
        monkeypatch=monkeypatch,
    )

    captured = capsys.readouterr()
    assert "phase=reservation" in captured.err
    assert str(reservation.results_path) in captured.err
    assert str(reservation.details_path) in captured.err
    assert str(reservation.trace_path) in captured.err
    assert "request_id" not in captured.err


def test_nonmatching_reserved_csv_return_emits_no_terminal_identity(
    monkeypatch, tmp_path, capsys
):
    exit_code, _, _, reservation = _execute(
        _async_args(),
        tmp_path,
        monkeypatch=monkeypatch,
        csv_run_id="different-run",
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out.splitlines().count(
        f"RUN_ID_RESERVED={reservation.run_id}"
    ) == 1
    assert captured.out.splitlines().count(f"RUN_ID={reservation.run_id}") == 0


def test_runtime_diagnostics_are_safe_exact_builtins():
    primary = RuntimeError("diagnostics failed")

    class FailingRuntime:
        def get_device_spec(self):
            raise primary

    class InvalidRuntime:
        def get_device_spec(self):
            return []

    assert benchmark_main._safe_runtime_diagnostics(FailingRuntime()) == {}
    assert benchmark_main._safe_runtime_diagnostics(InvalidRuntime()) == {}


def test_runtime_diagnostics_allowlist_omits_payload_bearing_fields():
    class Runtime:
        def get_device_spec(self):
            return {
                "backend": "onnxruntime",
                "device": "cpu",
                "active_providers": [
                    "CPUExecutionProvider",
                    "SECRET PROMPT PROVIDER",
                ],
                "prompt": "SECRET PROMPT TEXT",
                "input": np.array([123.0]),
                "output_tensor": {"secret": "SECRET OUTPUT"},
                "nested": {"input_tensor": [456]},
            }

    diagnostics = benchmark_main._safe_runtime_diagnostics(Runtime())

    assert diagnostics == {
        "backend": "onnxruntime",
        "device": "cpu",
        "active_providers": [
            "CPUExecutionProvider",
            "<redacted>",
        ],
    }
    serialized = json.dumps(diagnostics, sort_keys=True)
    assert "SECRET" not in serialized
    assert "prompt" not in serialized.lower()
    assert "input" not in serialized.lower()
    assert "output_tensor" not in serialized.lower()


def test_runtime_diagnostics_snapshot_does_not_alias_live_runtime_state():
    providers = ["CPUExecutionProvider"]
    live = {
        "backend": "onnxruntime",
        "device": "cpu",
        "active_providers": providers,
    }

    class Runtime:
        def get_device_spec(self):
            return live

    snapshot = benchmark_main._safe_runtime_diagnostics(Runtime())
    live["backend"] = "vllm"
    live["device"] = "cuda"
    providers.append("CUDAExecutionProvider")

    assert snapshot == {
        "backend": "onnxruntime",
        "device": "cpu",
        "active_providers": ["CPUExecutionProvider"],
    }


def test_failure_diagnostic_redacts_non_framework_phase_identifiers():
    diagnostic = benchmark_main._failure_diagnostic(
        RuntimeError("SECRET PROMPT"),
        "SECRET_PROMPT",
    )

    assert diagnostic == {
        "phase": "<redacted>",
        "error_type": "RuntimeError",
        "error_message": (
            "benchmark failed during <redacted> (RuntimeError)"
        ),
    }


def test_async_runner_lifecycle_callback_is_debug_only(
    monkeypatch, tmp_path
):
    _, normal_events, _, _ = _execute(
        _async_args(), tmp_path, monkeypatch=monkeypatch
    )
    normal_kwargs = next(
        event[1] for event in normal_events if event[0] == "async_init"
    )
    assert normal_kwargs.get("lifecycle_callback") is None

    _, debug_events, _, _ = _execute(
        _async_args("--debug"), tmp_path, monkeypatch=monkeypatch
    )
    debug_kwargs = next(
        event[1] for event in debug_events if event[0] == "async_init"
    )
    assert callable(debug_kwargs.get("lifecycle_callback"))


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


def test_post_run_trace_close_baseexception_uses_failure_artifact_path(
    monkeypatch, tmp_path, capsys
):
    class FatalTraceClose(BaseException):
        pass

    primary = FatalTraceClose("SECRET trace payload")
    reservation = _reservation(tmp_path)
    close_calls = []
    unloads = []
    saved_details = {}

    class TraceWriter:
        dropped = 0
        error = None

        def __init__(self, path, reservation):
            del path, reservation

        def start(self):
            pass

        def write(self, value):
            del value

        def close(self, timeout):
            close_calls.append(timeout)
            if len(close_calls) == 1:
                raise primary
            return True

    class Runner:
        failure_phase = "complete"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            del kwargs

        def run(self, config, warmup_runs):
            del config, warmup_runs
            return _result()

    def save_details(run_id, details, *, results_dir, reservation):
        del run_id, results_dir
        saved_details.update(details)
        return reservation.details_path

    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    monkeypatch.setattr(benchmark_main, "RequestTraceWriter", TraceWriter)
    monkeypatch.setattr(benchmark_main, "AsyncBenchmarkRunner", Runner)
    monkeypatch.setattr(benchmark_main, "save_async_details", save_details)
    monkeypatch.setattr(
        benchmark_main,
        "save_result",
        lambda **kwargs: kwargs["run_id"],
    )
    runtime = SimpleNamespace(
        get_device_spec=lambda: {
            "backend": "onnxruntime",
            "device": "cpu",
            "active_providers": ["CPUExecutionProvider"],
        },
        unload=lambda: unloads.append("unload"),
    )

    with pytest.raises(FatalTraceClose) as raised:
        benchmark_main.execute_benchmark(
            _async_args("--save-request-trace"),
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
    captured = capsys.readouterr()

    assert raised.value is primary
    assert len(close_calls) == 2
    assert unloads == ["unload"]
    assert saved_details["status"] == "invalid"
    assert saved_details["failure"] == {
        "phase": "trace_close",
        "error_type": "FatalTraceClose",
        "error_message": (
            "benchmark failed during trace_close (FatalTraceClose)"
        ),
    }
    assert saved_details["run"]["measurement_started"] is True
    assert saved_details["counts"] is None
    assert saved_details["counts_available"] is False
    assert "SECRET trace payload" not in json.dumps(saved_details)
    assert captured.out.splitlines().count("RUN_ID_RESERVED=async001") == 1
    assert captured.out.splitlines().count("RUN_ID=async001") == 1


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
    monkeypatch, tmp_path, capsys
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
    captured = capsys.readouterr()

    assert raised.value is primary
    assert unloads == [True]
    assert not any(
        line.startswith("RUN_ID_RESERVED=")
        for line in captured.out.splitlines()
    )
    assert not any(
        line.startswith("RUN_ID=") for line in captured.out.splitlines()
    )


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
    capsys,
):
    primary = RuntimeError("warmup failed")
    unloads = []
    saved_details = {}
    saved_csv = {}

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

        def __init__(self):
            self.loaded = True
            self.providers = ["CPUExecutionProvider"]

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
            self.loaded = False
            self.providers.append("MUTATEDAfterUnloadProvider")
            unloads.append("unload")

        def get_device_spec(self):
            return {
                "backend": "onnxruntime",
                "device": "cpu",
                "active_providers": self.providers,
                "loaded": self.loaded,
            }

    reservation = _reservation(tmp_path)
    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )

    def save_details(run_id, details, *, results_dir, reservation):
        del run_id, results_dir, reservation
        saved_details.update(details)
        return _reservation(tmp_path).details_path

    def save_csv(**kwargs):
        saved_csv.update(kwargs)
        return kwargs["run_id"]

    monkeypatch.setattr(benchmark_main, "save_async_details", save_details)
    monkeypatch.setattr(benchmark_main, "save_result", save_csv)
    loader = Loader()
    runtime = Runtime()

    with pytest.raises(RuntimeError) as raised:
        benchmark_main.execute_benchmark(
            _async_args(),
            loader=loader,
            runtime=runtime,
            evaluator=object(),
            decoder=object(),
            hw_monitor=None,
            task_name="IMAGE_CLASSIFICATION",
            target_meta={},
            results_path=reservation.results_path,
        )
    captured = capsys.readouterr()

    assert raised.value is primary
    assert unloads == ["unload"]
    assert loader.current_idx == 0
    assert saved_details["status"] == "invalid"
    assert saved_details["run"]["measurement_started"] is False
    assert saved_details["run"]["runtime_device_spec"] == {
        "backend": "onnxruntime",
        "device": "cpu",
        "active_providers": ["CPUExecutionProvider"],
    }
    assert saved_details["failure"] == {
        "phase": "warmup",
        "error_type": "RuntimeError",
        "error_message": "benchmark failed during warmup (RuntimeError)",
    }
    assert saved_details["counts"] == {
        "submitted": 0,
        "accepted": 0,
        "completed": 0,
        "failed": 0,
        "rejected": 0,
        "outstanding": 0,
    }
    assert saved_details["counts_available"] is True
    assert saved_csv["async_run_status"] == "invalid"
    assert saved_csv["async_invalid_reasons"] == "benchmark_exception"
    assert captured.out.splitlines().count("RUN_ID=async001") == 1


def test_failure_details_redact_exception_and_runtime_payloads(
    monkeypatch, tmp_path
):
    primary = RuntimeError(
        "SECRET PROMPT TEXT input_tensor output_tensor [123, 456]"
    )
    reservation = _reservation(tmp_path)
    serialized_details = {}

    class Runner:
        failure_phase = "warmup"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            del kwargs

        def run(self, config, warmup_runs):
            del config, warmup_runs
            raise primary

    class Runtime:
        def get_device_spec(self):
            return {
                "backend": "onnxruntime",
                "device": "cpu",
                "active_providers": ["CPUExecutionProvider"],
                "prompt": "SECRET PROMPT TEXT",
                "input_tensor": np.array([123, 456]),
                "output_tensor": {"secret": "SECRET OUTPUT"},
            }

        def unload(self):
            pass

    def save_details(run_id, details, *, results_dir, reservation):
        del run_id, results_dir
        serialized_details["text"] = json.dumps(
            details,
            default=repr,
            sort_keys=True,
        )
        serialized_details["value"] = details
        return reservation.details_path

    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    monkeypatch.setattr(benchmark_main, "AsyncBenchmarkRunner", Runner)
    monkeypatch.setattr(benchmark_main, "save_async_details", save_details)
    monkeypatch.setattr(
        benchmark_main,
        "save_result",
        lambda **kwargs: kwargs["run_id"],
    )

    with pytest.raises(RuntimeError) as raised:
        _execute_with_runtime(_async_args(), tmp_path, Runtime())

    assert raised.value is primary
    assert serialized_details["value"]["failure"] == {
        "phase": "warmup",
        "error_type": "RuntimeError",
        "error_message": "benchmark failed during warmup (RuntimeError)",
    }
    assert serialized_details["value"]["run"]["runtime_device_spec"] == {
        "backend": "onnxruntime",
        "device": "cpu",
        "active_providers": ["CPUExecutionProvider"],
    }
    for forbidden in (
        "SECRET PROMPT TEXT",
        "SECRET OUTPUT",
        "input_tensor",
        "output_tensor",
        "[123, 456]",
    ):
        assert forbidden not in serialized_details["text"]


def test_warmup_failure_persistence_sidecar_error_is_secondary(
    monkeypatch, tmp_path, capsys
):
    primary = RuntimeError("warmup failed")
    secondary = OSError("sidecar failed")
    reservation = _reservation(tmp_path)

    class Runner:
        failure_phase = "warmup"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            del kwargs

        def run(self, config, warmup_runs):
            del config, warmup_runs
            raise primary

    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    monkeypatch.setattr(benchmark_main, "AsyncBenchmarkRunner", Runner)
    monkeypatch.setattr(
        benchmark_main,
        "save_async_details",
        lambda *args, **kwargs: (_ for _ in ()).throw(secondary),
    )
    monkeypatch.setattr(
        benchmark_main,
        "save_result",
        lambda **kwargs: kwargs["run_id"],
    )

    runtime = SimpleNamespace(
        get_device_spec=lambda: {"loaded": True},
        unload=lambda: None,
    )
    with pytest.raises(RuntimeError) as raised:
        _execute_with_runtime(_async_args(), tmp_path, runtime)
    captured = capsys.readouterr()

    assert raised.value is primary
    assert primary.cleanup_secondary_errors[-1]["phase"] == "failure_sidecar"
    assert primary.cleanup_secondary_errors[-1]["error_type"] == "OSError"
    assert "phase=failure_sidecar" in captured.err
    assert captured.out.splitlines().count("RUN_ID=async001") == 1


def test_warmup_failure_persistence_links_certain_sidecar_commit(
    monkeypatch, tmp_path
):
    primary = RuntimeError("warmup failed")
    secondary = OSError("sidecar directory close failed")
    secondary.final_file_committed = True
    secondary.publication_state_uncertain = False
    reservation = _reservation(tmp_path)
    saved_csv = {}

    class Runner:
        failure_phase = "warmup"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            del kwargs

        def run(self, config, warmup_runs):
            del config, warmup_runs
            raise primary

    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    monkeypatch.setattr(benchmark_main, "AsyncBenchmarkRunner", Runner)
    monkeypatch.setattr(
        benchmark_main,
        "save_async_details",
        lambda *args, **kwargs: (_ for _ in ()).throw(secondary),
    )

    def save_csv(**kwargs):
        saved_csv.update(kwargs)
        return kwargs["run_id"]

    monkeypatch.setattr(benchmark_main, "save_result", save_csv)
    runtime = SimpleNamespace(
        get_device_spec=lambda: {"loaded": True},
        unload=lambda: None,
    )

    with pytest.raises(RuntimeError) as raised:
        _execute_with_runtime(_async_args(), tmp_path, runtime)

    assert raised.value is primary
    assert saved_csv["details_path"] == "results/details/async001.json"
    assert primary.cleanup_secondary_errors[-1]["phase"] == "failure_sidecar"


def test_warmup_failure_persistence_csv_error_has_only_reserved_identity(
    monkeypatch, tmp_path, capsys
):
    primary = RuntimeError("warmup failed")
    secondary = OSError("csv failed")
    reservation = _reservation(tmp_path)

    class Runner:
        failure_phase = "warmup"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            del kwargs

        def run(self, config, warmup_runs):
            del config, warmup_runs
            raise primary

    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    monkeypatch.setattr(benchmark_main, "AsyncBenchmarkRunner", Runner)
    monkeypatch.setattr(
        benchmark_main,
        "save_async_details",
        lambda *args, **kwargs: reservation.details_path,
    )
    monkeypatch.setattr(
        benchmark_main,
        "save_result",
        lambda **kwargs: (_ for _ in ()).throw(secondary),
    )

    runtime = SimpleNamespace(
        get_device_spec=lambda: {"loaded": True},
        unload=lambda: None,
    )
    with pytest.raises(RuntimeError) as raised:
        _execute_with_runtime(_async_args(), tmp_path, runtime)
    captured = capsys.readouterr()

    assert raised.value is primary
    assert primary.cleanup_secondary_errors[-1]["phase"] == "failure_csv"
    assert primary.cleanup_secondary_errors[-1]["error_type"] == "OSError"
    assert "phase=failure_csv" in captured.err
    assert captured.out.splitlines().count("RUN_ID_RESERVED=async001") == 1
    assert captured.out.splitlines().count("RUN_ID=async001") == 0


def test_measurement_failure_persistence_marks_counts_unavailable(
    monkeypatch, tmp_path
):
    primary = RuntimeError("measurement failed")
    reservation = _reservation(tmp_path)
    saved_details = {}

    class Runner:
        failure_phase = "measurement"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            del kwargs

        def run(self, config, warmup_runs):
            del config, warmup_runs
            raise primary

    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    monkeypatch.setattr(benchmark_main, "AsyncBenchmarkRunner", Runner)

    def save_details(run_id, details, *, results_dir, reservation):
        del run_id, results_dir, reservation
        saved_details.update(details)
        return _reservation(tmp_path).details_path

    monkeypatch.setattr(benchmark_main, "save_async_details", save_details)
    monkeypatch.setattr(
        benchmark_main,
        "save_result",
        lambda **kwargs: kwargs["run_id"],
    )

    runtime = SimpleNamespace(
        get_device_spec=lambda: {"loaded": True},
        unload=lambda: None,
    )
    with pytest.raises(RuntimeError) as raised:
        _execute_with_runtime(_async_args(), tmp_path, runtime)

    assert raised.value is primary
    assert saved_details["run"]["measurement_started"] is True
    assert saved_details["counts"] is None
    assert saved_details["counts_available"] is False


def test_unexpected_failure_persistence_error_cannot_replace_primary(
    monkeypatch, tmp_path, capsys
):
    primary = RuntimeError("warmup failed")
    secondary = OSError("unexpected persistence failure")
    reservation = _reservation(tmp_path)

    class Runner:
        failure_phase = "warmup"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            del kwargs

        def run(self, config, warmup_runs):
            del config, warmup_runs
            raise primary

    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    monkeypatch.setattr(benchmark_main, "AsyncBenchmarkRunner", Runner)
    monkeypatch.setattr(
        benchmark_main,
        "_persist_async_failure",
        lambda **kwargs: (_ for _ in ()).throw(secondary),
    )

    runtime = SimpleNamespace(
        get_device_spec=lambda: {"loaded": True},
        unload=lambda: None,
    )
    with pytest.raises(RuntimeError) as raised:
        _execute_with_runtime(_async_args(), tmp_path, runtime)
    captured = capsys.readouterr()

    assert raised.value is primary
    assert primary.cleanup_secondary_errors[-1]["phase"] == (
        "failure_persistence"
    )
    assert "phase=failure_persistence" in captured.err
    assert captured.out.splitlines().count("RUN_ID_RESERVED=async001") == 1
    assert captured.out.splitlines().count("RUN_ID=async001") == 0


def test_stderr_and_debug_print_failures_preserve_primary_traceback(
    monkeypatch, tmp_path, capsys
):
    primary = RuntimeError("warmup failed")
    sidecar_error = OSError("sidecar failed")
    reservation = _reservation(tmp_path)

    def raise_primary():
        raise primary

    class Runner:
        failure_phase = "warmup"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            del kwargs

        def run(self, config, warmup_runs):
            del config, warmup_runs
            raise_primary()

    real_print = print

    def fail_stderr_print(*values, **kwargs):
        if kwargs.get("file") is benchmark_main.sys.stderr:
            raise OSError("stderr unavailable")
        return real_print(*values, **kwargs)

    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    monkeypatch.setattr(benchmark_main, "AsyncBenchmarkRunner", Runner)
    monkeypatch.setattr(
        benchmark_main,
        "save_async_details",
        lambda *args, **kwargs: (_ for _ in ()).throw(sidecar_error),
    )
    monkeypatch.setattr(
        benchmark_main,
        "save_result",
        lambda **kwargs: kwargs["run_id"],
    )
    monkeypatch.setattr(
        benchmark_main,
        "print",
        fail_stderr_print,
        raising=False,
    )
    runtime = SimpleNamespace(
        get_device_spec=lambda: {
            "backend": "onnxruntime",
            "device": "cpu",
            "active_providers": ["CPUExecutionProvider"],
        },
        unload=lambda: None,
    )

    with pytest.raises(RuntimeError) as raised:
        _execute_with_runtime(_async_args("--debug"), tmp_path, runtime)
    captured = capsys.readouterr()
    traceback_names = []
    traceback = raised.value.__traceback__
    while traceback is not None:
        traceback_names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next

    assert raised.value is primary
    assert traceback_names.count("raise_primary") == 1
    assert captured.out.splitlines().count("RUN_ID_RESERVED=async001") == 1
    assert captured.out.splitlines().count("RUN_ID=async001") == 1


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


def test_uncertain_csv_commit_returns_nonzero_with_only_reserved_run_id(
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
    assert captured.out.splitlines().count(
        f"RUN_ID_RESERVED={reservation.run_id}"
    ) == 1
    assert captured.out.splitlines().count(f"RUN_ID={reservation.run_id}") == 0
    assert "결과 CSV 저장 실패" in captured.err
