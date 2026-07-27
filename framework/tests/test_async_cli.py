import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import main as benchmark_main
import core.async_inference.runner as async_runner_module
import core.result_store as result_store_module
import core.runtime_executor as runtime_executor_module
from core.async_inference import AsyncBenchmarkResult, RunStatus
from core.async_inference.completion import CompletionCoordinator
from core.async_inference.engine import AsyncInferenceEngine
from core.async_inference.metrics import AsyncMetricsCollector
from core.inference_pipeline import InferencePipeline
from core.runtime_executor import (
    NativeAsyncExecutorSnapshot,
    NativeAsyncRuntimeExecutor,
)
from core.targets import get_target


def parse(extra):
    return benchmark_main.build_parser().parse_args(
        ["--model", "resnet50", *extra]
    )


def test_default_inference_mode_is_e2e():
    assert parse([]).inference_mode == "e2e"


def test_debug_help_distinguishes_e2e_samples_from_async_lifecycle():
    debug_action = next(
        action
        for action in benchmark_main.build_parser()._actions
        if "--debug" in action.option_strings
    )

    assert "e2e" in debug_action.help
    assert "샘플" in debug_action.help
    assert "async_queue" in debug_action.help
    assert "lifecycle" in debug_action.help


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


def test_pre_resolution_validation_does_not_apply_furiosa_rules_to_cpu_target():
    args = parse(
        [
            "--target",
            "cpu",
            "--backend",
            "furiosa_llm",
            "--inference-mode",
            "async_queue",
            "--batch-size",
            "2",
        ]
    )

    benchmark_main.validate_async_args(args)


def test_furiosa_async_executor_uses_queue_and_worker_inflight_limit():
    args = _async_args(
        "--backend",
        "furiosa_llm",
        "--worker-count",
        "8",
        "--queue-capacity",
        "4",
    )
    config = benchmark_main.build_async_config(args)
    calls = []

    class Backend:
        def submit_async(self, inputs, callback):
            raise AssertionError("executor construction must not submit work")

    backend = Backend()

    class Runtime:
        def supports_generate(self):
            return True

        def native_async_max_batch_size(self):
            return 1

        def create_native_backend(self, **kwargs):
            calls.append(kwargs)
            return backend

    loader = SimpleNamespace(
        get_metadata=lambda: {"stop_token_ids": [2, 128009]}
    )

    executor = benchmark_main._build_async_runtime_executor(
        args,
        get_target("furiosa-rngd"),
        Runtime(),
        loader,
        config,
    )

    assert isinstance(executor, NativeAsyncRuntimeExecutor)
    assert executor.backend is backend
    assert executor.max_inflight == 4
    assert executor.completion_timeout_sec == config.flush_timeout_sec
    assert calls == [
        {
            "max_new_tokens": args.max_new_tokens,
            "stop_token_ids": [2, 128009],
        }
    ]


def test_mobilint_native_async_executor_uses_runtime_factory_without_generation_args():
    args = _async_args(
        "--target",
        "mobilint-aries",
        "--worker-count",
        "8",
        "--queue-capacity",
        "4",
    )
    config = benchmark_main.build_async_config(args)
    calls = []

    class Backend:
        def submit_async(self, inputs, callback):
            raise AssertionError("executor construction must not submit work")

    backend = Backend()

    class Runtime:
        def supports_generate(self):
            return False

        def native_async_max_batch_size(self):
            return 1

        def create_native_backend(self, **kwargs):
            calls.append(kwargs)
            return backend

    executor = benchmark_main._build_async_runtime_executor(
        args,
        get_target("mobilint-aries"),
        Runtime(),
        SimpleNamespace(get_metadata=lambda: {"stop_token_ids": [2]}),
        config,
    )

    assert isinstance(executor, NativeAsyncRuntimeExecutor)
    assert executor.backend is backend
    assert executor.max_inflight == 4
    assert executor.completion_timeout_sec == config.flush_timeout_sec
    assert calls == [{}]


def test_native_async_target_requires_callable_runtime_factory():
    args = _async_args("--target", "mobilint-regulus")
    config = benchmark_main.build_async_config(args)

    with pytest.raises(RuntimeError, match="create_native_backend"):
        benchmark_main._build_async_runtime_executor(
            args,
            get_target("mobilint-regulus"),
            SimpleNamespace(
                supports_generate=lambda: False,
                native_async_max_batch_size=lambda: 1,
            ),
            SimpleNamespace(get_metadata=lambda: {}),
            config,
        )


def test_native_async_target_requires_runtime_batch_limit():
    args = _async_args("--target", "mobilint-regulus")
    config = benchmark_main.build_async_config(args)

    with pytest.raises(RuntimeError, match="native_async_max_batch_size"):
        benchmark_main._build_async_runtime_executor(
            args,
            get_target("mobilint-regulus"),
            SimpleNamespace(
                supports_generate=lambda: False,
                create_native_backend=lambda: object(),
            ),
            SimpleNamespace(get_metadata=lambda: {}),
            config,
        )


def test_non_native_target_keeps_blocking_runtime_executor_selection():
    args = _async_args("--target", "cpu")
    config = benchmark_main.build_async_config(args)

    assert benchmark_main._build_async_runtime_executor(
        args,
        get_target("cpu"),
        SimpleNamespace(),
        SimpleNamespace(get_metadata=lambda: {}),
        config,
    ) is None


def test_mobilint_aries_llm_does_not_select_native_async_executor():
    args = _async_args("--target", "mobilint-aries-llm")
    config = benchmark_main.build_async_config(args)

    class Runtime:
        def create_native_backend(self):
            raise AssertionError("LLM target must not select native async")

    assert benchmark_main._build_async_runtime_executor(
        args,
        get_target("mobilint-aries-llm"),
        Runtime(),
        SimpleNamespace(get_metadata=lambda: {}),
        config,
    ) is None


def test_mobilint_static_transformer_uses_blocking_executor_fallback():
    args = _async_args("--target", "mobilint-aries")
    config = benchmark_main.build_async_config(args)

    class Runtime:
        def native_async_max_batch_size(self):
            return None

        def create_native_backend(self):
            raise AssertionError("unsupported artifact must not create native backend")

    assert benchmark_main._build_async_runtime_executor(
        args,
        get_target("mobilint-aries"),
        Runtime(),
        SimpleNamespace(get_metadata=lambda: {}),
        config,
    ) is None


def test_native_async_factory_rejects_batch_above_runtime_limit():
    args = _async_args("--target", "mobilint-aries", "--batch-size", "2")
    config = benchmark_main.build_async_config(args)
    runtime = SimpleNamespace(
        native_async_max_batch_size=lambda: 1,
        create_native_backend=lambda: object(),
        supports_generate=lambda: False,
    )

    with pytest.raises(ValueError, match="native async requires max_batch_size<=1"):
        benchmark_main._build_async_runtime_executor(
            args,
            get_target("mobilint-aries"),
            runtime,
            SimpleNamespace(get_metadata=lambda: {}),
            config,
        )


def test_furiosa_native_async_factory_rejects_batch_above_runtime_limit():
    args = parse(
        [
            "--backend",
            "furiosa_llm",
            "--inference-mode",
            "async_queue",
            "--batch-size",
            "2",
        ]
    )
    config = benchmark_main.build_async_config(args)
    runtime = SimpleNamespace(
        native_async_max_batch_size=lambda: 1,
        create_native_backend=lambda **kwargs: object(),
        supports_generate=lambda: True,
    )

    with pytest.raises(ValueError, match="native async requires max_batch_size<=1"):
        benchmark_main._build_async_runtime_executor(
            args,
            get_target("furiosa-rngd"),
            runtime,
            SimpleNamespace(get_metadata=lambda: {}),
            config,
        )


@pytest.mark.parametrize(
    "declared_limit",
    [
        pytest.param(True, id="bool"),
        pytest.param("1", id="string"),
        pytest.param(1.0, id="integral-float"),
        pytest.param(1.5, id="fractional-float"),
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(object(), id="object"),
    ],
)
def test_native_async_factory_requires_exact_positive_int_batch_limit(
    declared_limit,
):
    args = _async_args("--target", "mobilint-regulus")
    config = benchmark_main.build_async_config(args)
    runtime = SimpleNamespace(
        native_async_max_batch_size=lambda: declared_limit,
        create_native_backend=lambda: object(),
        supports_generate=lambda: False,
    )

    with pytest.raises(
        RuntimeError,
        match="mobilint-regulus.*positive int",
    ):
        benchmark_main._build_async_runtime_executor(
            args,
            get_target("mobilint-regulus"),
            runtime,
            SimpleNamespace(get_metadata=lambda: {}),
            config,
        )


def test_async_pipeline_option_is_forced_only_for_mobilint_native_async_queue():
    mobilint_options = {
        "async_pipeline_enabled": False,
        "activation_slots": 1,
    }
    benchmark_main._enable_native_async_pipeline(
        _async_args("--target", "mobilint-aries"),
        get_target("mobilint-aries"),
        mobilint_options,
    )
    assert mobilint_options["async_pipeline_enabled"] is True

    cpu_options = {}
    benchmark_main._enable_native_async_pipeline(
        _async_args("--target", "cpu"),
        get_target("cpu"),
        cpu_options,
    )
    assert cpu_options == {}

    e2e_options = {"async_pipeline_enabled": False}
    benchmark_main._enable_native_async_pipeline(
        parse(["--target", "mobilint-aries"]),
        get_target("mobilint-aries"),
        e2e_options,
    )
    assert e2e_options["async_pipeline_enabled"] is False

    transformer_options = {
        "async_pipeline_enabled": False,
        "native_async_supported": False,
    }
    benchmark_main._enable_native_async_pipeline(
        _async_args("--target", "mobilint-aries"),
        get_target("mobilint-aries"),
        transformer_options,
    )
    assert transformer_options["async_pipeline_enabled"] is False


@pytest.mark.parametrize(
    ("worker_args", "expected_inflight"),
    [([], 1), (["--worker-count", "4"], 4)],
)
def test_rbln_native_async_injects_framework_worker_capacity_only(
    worker_args, expected_inflight
):
    args = _async_args("--target", "rbln-static", *worker_args)
    runtime_options = {
        **get_target("rbln-static").runtime_options,
        "async_parallel": 2,
    }

    benchmark_main._enable_native_async_pipeline(
        args,
        get_target("rbln-static"),
        runtime_options,
    )

    assert runtime_options["max_async_inflight"] == expected_inflight
    assert runtime_options["async_parallel"] == 2


def test_rbln_native_async_executor_uses_bounded_framework_capacity():
    args = _async_args(
        "--target",
        "rbln-static",
        "--worker-count",
        "4",
        "--queue-capacity",
        "3",
    )
    config = benchmark_main.build_async_config(args)
    backend = SimpleNamespace(submit_async=lambda inputs, callback: None)
    factory_calls = []

    class FakeRblnRuntime:
        def supports_generate(self):
            return False

        def native_async_max_batch_size(self):
            return 1

        def create_native_backend(self, **kwargs):
            factory_calls.append(kwargs)
            return backend

    executor = benchmark_main._build_async_runtime_executor(
        args,
        get_target("rbln-static"),
        FakeRblnRuntime(),
        SimpleNamespace(get_metadata=lambda: {}),
        config,
    )

    assert isinstance(executor, NativeAsyncRuntimeExecutor)
    assert executor.backend is backend
    assert executor.max_inflight == 3
    assert factory_calls == [{}]


def test_rbln_native_async_requires_declared_batch_limit_exactly_one():
    args = _async_args("--target", "rbln-static")
    config = benchmark_main.build_async_config(args)
    runtime = SimpleNamespace(
        native_async_max_batch_size=lambda: 2,
        create_native_backend=lambda: object(),
        supports_generate=lambda: False,
    )

    with pytest.raises(RuntimeError, match="rbln-static.*exactly 1"):
        benchmark_main._build_async_runtime_executor(
            args,
            get_target("rbln-static"),
            runtime,
            SimpleNamespace(get_metadata=lambda: {}),
            config,
        )


def test_rbln_native_async_rejects_batch_two_before_backend_creation():
    args = _async_args(
        "--target", "rbln-static", "--batch-size", "2"
    )
    config = benchmark_main.build_async_config(args)
    calls = []
    runtime = SimpleNamespace(
        native_async_max_batch_size=lambda: 1,
        create_native_backend=lambda: calls.append("create"),
        supports_generate=lambda: False,
    )

    with pytest.raises(
        ValueError, match="native async requires max_batch_size<=1"
    ):
        benchmark_main._build_async_runtime_executor(
            args,
            get_target("rbln-static"),
            runtime,
            SimpleNamespace(get_metadata=lambda: {}),
            config,
        )

    assert calls == []


def test_rbln_runtime_capability_accepts_four_workers():
    runtime = SimpleNamespace(
        compiled_model=None,
        max_concurrent_workers=lambda: 4,
        max_dynamic_batch_size=lambda: 1,
        supports_dynamic_batching=lambda: False,
        supports_batch_generation=lambda: False,
    )
    loader = SimpleNamespace(
        get_metadata=lambda: {
            "is_static_batched": False,
            "total_samples": 1,
        }
    )
    executor = object()
    pipeline = InferencePipeline(
        loader,
        runtime,
        runtime_executor=executor,
    )
    config = benchmark_main.AsyncInferenceConfig(
        queue_capacity=4,
        worker_count=4,
        max_batch_size=1,
        min_samples=1,
    )
    metrics = AsyncMetricsCollector(0, worker_count=4)
    coordinator = CompletionCoordinator(
        pipeline,
        object(),
        None,
        metrics,
        queue_capacity=4,
    )

    engine = AsyncInferenceEngine(
        runtime,
        pipeline,
        config,
        coordinator,
        metrics,
        executor=executor,
    )

    assert len(engine.workers) == 4


def test_execute_benchmark_injects_selected_async_runtime_executor(
    monkeypatch,
    tmp_path,
):
    args = _async_args("--backend", "furiosa_llm")
    selected_executor = object()
    monkeypatch.setattr(
        benchmark_main,
        "_build_async_runtime_executor",
        lambda args, target, runtime, loader, config: selected_executor,
    )

    exit_code, events, _, _ = _execute(
        args,
        tmp_path,
        monkeypatch=monkeypatch,
        target=get_target("furiosa-rngd"),
    )

    init_kwargs = next(
        event[1] for event in events if event[0] == "engine_init"
    )
    assert exit_code == 0
    assert init_kwargs["runtime_executor"] is selected_executor


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
        failure_details_path=root / "details" / "async001.failure.json",
        trace_path=root / "traces" / "async001.jsonl",
    )


def _result(
    *,
    status=RunStatus.VALID,
    outstanding=0,
    invalid_reasons=None,
):
    reasons = (
        () if status is RunStatus.VALID else ("synthetic_invalid",)
    )
    if invalid_reasons is not None:
        reasons = tuple(invalid_reasons)
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


class HostileOutstandingCounter:
    def __bool__(self):
        raise AssertionError("SECRET counter truthiness")

    def __repr__(self):
        return "SECRET hostile counter repr"

    def __str__(self):
        return "SECRET hostile counter str"


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
    runtime_spec=None,
    saved=None,
    runtime_unload_safe=True,
    runtime_unload_safe_error=None,
    runtime_unload_safe_reads=None,
    target=None,
):
    events = [] if events is None else events
    reservation = _reservation(tmp_path)
    if runtime_spec is None:
        runtime_spec = {
            "backend": "onnxruntime",
            "active_providers": ["CPUExecutionProvider"],
        }
    runtime = SimpleNamespace(
        unload=lambda: events.append("unload"),
        get_device_spec=lambda: runtime_spec,
    )
    saved = {} if saved is None else saved

    def reserve(*, results_path, run_id=None):
        events.append(("reserve", Path(results_path), run_id))
        return reservation

    class Engine:
        def __init__(self, **kwargs):
            events.append(("engine_init", kwargs))
            self.failure_phase = "created"

        @property
        def runtime_unload_safe_after_failure(self):
            if runtime_unload_safe_reads is not None:
                runtime_unload_safe_reads.append("read")
            if runtime_unload_safe_error is not None:
                raise runtime_unload_safe_error
            return runtime_unload_safe

        def run_async(self, config, warmup_runs, monitor):
            events.append(("async_run", config, warmup_runs, monitor))
            self.failure_phase = "complete"
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

    def save_failure_details(
        run_id,
        details,
        *,
        results_dir,
        reservation,
    ):
        del run_id, results_dir
        saved["failure_details"] = details
        return reservation.failure_details_path

    monkeypatch.setattr(benchmark_main, "reserve_run_artifacts", reserve)
    monkeypatch.setattr(
        benchmark_main,
        "InferenceEngine",
        Engine,
        raising=False,
    )
    monkeypatch.setattr(benchmark_main, "save_async_details", save_details)
    monkeypatch.setattr(
        benchmark_main,
        "save_async_failure_details",
        save_failure_details,
    )
    monkeypatch.setattr(benchmark_main, "save_result", save_csv)

    exit_code = benchmark_main.execute_benchmark(
        args,
        target=get_target("cpu") if target is None else target,
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


def _native_executor_with_snapshot(monkeypatch, snapshot):
    executor = NativeAsyncRuntimeExecutor(
        SimpleNamespace(
            submit_async=lambda inputs, callback: None,
        ),
        max_inflight=1,
        completion_timeout_sec=1.0,
    )
    monkeypatch.setattr(executor, "snapshot", snapshot)
    return executor


def test_native_async_executor_metrics_sanitize_exact_nonnegative_ints(
    monkeypatch,
):
    executor = _native_executor_with_snapshot(
        monkeypatch,
        lambda: NativeAsyncExecutorSnapshot(
            inflight=0,
            duplicate_callbacks=1,
            late_callbacks=2,
            submit_failures=3,
            timeouts=4,
        ),
    )

    assert benchmark_main._safe_native_async_executor_metrics(executor) == {
        "async_native_inflight": 0,
        "async_native_duplicate_callbacks": 1,
        "async_native_late_callbacks": 2,
        "async_native_submit_failures": 3,
        "async_native_timeouts": 4,
    }


def test_native_async_executor_metrics_ignore_nonnative_executors():
    class NonnativeExecutor:
        def snapshot(self):
            raise AssertionError("nonnative snapshot must not be read")

    assert benchmark_main._safe_native_async_executor_metrics(None) is None
    assert (
        benchmark_main._safe_native_async_executor_metrics(
            NonnativeExecutor()
        )
        is None
    )


@pytest.mark.parametrize(
    "invalid_value",
    [True, 1.0, -1],
    ids=("bool", "float", "negative"),
)
def test_native_async_executor_metrics_reject_invalid_exact_types(
    monkeypatch, invalid_value
):
    executor = _native_executor_with_snapshot(
        monkeypatch,
        lambda: NativeAsyncExecutorSnapshot(
            inflight=0,
            duplicate_callbacks=0,
            late_callbacks=0,
            submit_failures=0,
            timeouts=invalid_value,
        ),
    )

    assert benchmark_main._safe_native_async_executor_metrics(executor) == {}


def test_native_async_executor_metrics_never_coerce_hostile_snapshot(
    monkeypatch,
):
    class HostileValue:
        def __lt__(self, other):
            raise AssertionError("hostile value must not be compared")

        def __int__(self):
            raise AssertionError("hostile value must not be coerced")

        def __str__(self):
            raise AssertionError("hostile value must not be stringified")

        def __repr__(self):
            raise AssertionError("hostile value must not be repr'd")

    class HostileSnapshot:
        def __getattribute__(self, name):
            raise AssertionError("hostile snapshot must not be inspected")

        def __str__(self):
            raise AssertionError("hostile snapshot must not be stringified")

        def __repr__(self):
            raise AssertionError("hostile snapshot must not be repr'd")

    hostile_value_executor = _native_executor_with_snapshot(
        monkeypatch,
        lambda: NativeAsyncExecutorSnapshot(
            inflight=0,
            duplicate_callbacks=0,
            late_callbacks=0,
            submit_failures=0,
            timeouts=HostileValue(),
        ),
    )
    hostile_snapshot_executor = _native_executor_with_snapshot(
        monkeypatch,
        lambda: HostileSnapshot(),
    )

    assert (
        benchmark_main._safe_native_async_executor_metrics(
            hostile_value_executor
        )
        == {}
    )
    assert (
        benchmark_main._safe_native_async_executor_metrics(
            hostile_snapshot_executor
        )
        == {}
    )


def test_native_async_executor_metrics_snapshot_failure_is_safe(
    monkeypatch,
):
    def fail_snapshot():
        raise RuntimeError("SECRET native snapshot failure")

    executor = _native_executor_with_snapshot(
        monkeypatch,
        fail_snapshot,
    )

    assert benchmark_main._safe_native_async_executor_metrics(executor) == {}


def test_native_async_executor_metrics_reach_console_csv_and_details(
    monkeypatch, tmp_path, capsys
):
    executor = _native_executor_with_snapshot(
        monkeypatch,
        lambda: NativeAsyncExecutorSnapshot(
            inflight=0,
            duplicate_callbacks=1,
            late_callbacks=2,
            submit_failures=3,
            timeouts=4,
        ),
    )
    monkeypatch.setattr(
        benchmark_main,
        "_build_async_runtime_executor",
        lambda *args, **kwargs: executor,
    )
    result = _result()
    result.metrics["async_timed_out_requests"] = 7

    exit_code, events, saved, _ = _execute(
        _async_args(),
        tmp_path,
        monkeypatch=monkeypatch,
        result=result,
    )

    native_metrics = {
        "async_native_inflight": 0,
        "async_native_duplicate_callbacks": 1,
        "async_native_late_callbacks": 2,
        "async_native_submit_failures": 3,
        "async_native_timeouts": 4,
    }
    assert exit_code == 0
    assert "unload" in events
    assert {
        key: saved["csv"]["metrics"][key]
        for key in native_metrics
    } == native_metrics
    assert saved["details"]["native_async_executor"] == native_metrics
    assert saved["csv"]["metrics"]["async_timed_out_requests"] == 7
    output = capsys.readouterr().out
    for key, value in native_metrics.items():
        assert f"{key}: {value}" in output


@pytest.mark.parametrize("failure_kind", ["read", "schema"])
def test_native_async_snapshot_failure_marks_successful_run_invalid(
    monkeypatch, tmp_path, capsys, failure_kind
):
    class HostileSnapshot:
        def __getattribute__(self, name):
            raise AssertionError("SECRET hostile snapshot access")

        def __str__(self):
            raise AssertionError("SECRET hostile snapshot string")

        def __repr__(self):
            raise AssertionError("SECRET hostile snapshot repr")

    def snapshot():
        if failure_kind == "read":
            raise RuntimeError("SECRET native snapshot read")
        return HostileSnapshot()

    executor = _native_executor_with_snapshot(monkeypatch, snapshot)
    monkeypatch.setattr(
        benchmark_main,
        "_build_async_runtime_executor",
        lambda *args, **kwargs: executor,
    )

    exit_code, events, saved, _ = _execute(
        _async_args(),
        tmp_path,
        monkeypatch=monkeypatch,
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "unload" not in events
    assert saved["details"]["status"] == "invalid"
    assert saved["details"]["invalid_reasons"] == [
        "native_async_executor_snapshot_invalid"
    ]
    assert saved["csv"]["async_run_status"] == "invalid"
    assert saved["csv"]["async_invalid_reasons"] == (
        "native_async_executor_snapshot_invalid"
    )
    assert "native_async_executor" not in saved["details"]
    assert not any(
        key.startswith("async_native_")
        for key in saved["csv"]["metrics"]
    )
    assert "SECRET" not in captured.out
    assert "SECRET" not in captured.err


def test_native_async_nonzero_inflight_marks_run_invalid_and_blocks_unload(
    monkeypatch, tmp_path
):
    executor = _native_executor_with_snapshot(
        monkeypatch,
        lambda: NativeAsyncExecutorSnapshot(
            inflight=1,
            duplicate_callbacks=0,
            late_callbacks=0,
            submit_failures=0,
            timeouts=0,
        ),
    )
    monkeypatch.setattr(
        benchmark_main,
        "_build_async_runtime_executor",
        lambda *args, **kwargs: executor,
    )

    exit_code, events, saved, _ = _execute(
        _async_args(),
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert exit_code == 1
    assert "unload" not in events
    assert saved["details"]["native_async_executor"][
        "async_native_inflight"
    ] == 1
    assert saved["details"]["invalid_reasons"] == [
        "native_async_inflight_nonzero"
    ]
    assert saved["csv"]["async_run_status"] == "invalid"
    assert saved["csv"]["async_invalid_reasons"] == (
        "native_async_inflight_nonzero"
    )


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
    assert names.index("async_run") < names.index("unload")
    assert names.index("unload") < names.index("details") < names.index("csv")
    assert saved["csv"]["reservation"] is reservation
    assert saved["csv"]["run_id"] == reservation.run_id
    assert saved["csv"]["results_path"] == reservation.results_path
    assert saved["csv"]["details_path"] == "results/details/async001.json"
    assert saved["csv"]["inference_mode"] == "async_queue"
    run = saved["details"]["run"]
    assert {
        key: run[key]
        for key in (
            "model_name",
            "task",
            "backend",
            "device",
            "batch_size",
            "warmup_runs",
            "target_id",
            "dataset_path",
            "model_artifact_path",
            "runtime_device_spec",
        )
    } == {
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
    assert run["furiosa_llm_version"] is None
    assert run["python_version"] == ".".join(
        str(value) for value in sys.version_info[:3]
    )
    assert type(run["framework_git_commit"]) is str
    assert type(run["framework_git_dirty"]) is bool
    assert run["percentile_method"] == "numpy.percentile(method=linear)"
    assert run["token_policy"] is None
    assert run["sampling_policy"] is None
    assert run["async_workload"] == {
        "scenario": "offline",
        "target_qps": None,
        "worker_count": 1,
        "queue_capacity": 256,
        "schedule_seed": 0,
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


def test_async_run_metadata_records_paper_reproducibility_fields(monkeypatch):
    args = _async_args(
        "--backend",
        "furiosa_llm",
        "--artifact",
        "/models/llama.fxb",
        "--max-new-tokens",
        "64",
        "--worker-count",
        "4",
        "--queue-capacity",
        "16",
        "--schedule-seed",
        "23",
    )
    monkeypatch.setattr(
        benchmark_main,
        "_framework_git_metadata",
        lambda: {"commit": "abc123", "dirty": False},
        raising=False,
    )
    monkeypatch.setattr(
        benchmark_main,
        "_furiosa_llm_version",
        lambda: "2026.3.0",
        raising=False,
    )

    metadata = benchmark_main._async_run_metadata(
        args,
        "NLP_GENERATION",
        {"target_id": "furiosa-rngd"},
        {},
    )

    assert metadata["furiosa_llm_version"] == "2026.3.0"
    assert metadata["python_version"] == ".".join(
        str(value) for value in sys.version_info[:3]
    )
    assert metadata["framework_git_commit"] == "abc123"
    assert metadata["framework_git_dirty"] is False
    assert metadata["percentile_method"] == "numpy.percentile(method=linear)"
    assert metadata["model_artifact_path"] == "/models/llama.fxb"
    assert metadata["token_policy"] == {
        "input": "attention_mask_non_padding_prompt_tokens",
        "output": "generated_token_ids_excluding_prompt",
    }
    assert metadata["sampling_policy"] == {
        "temperature": 0.0,
        "ignore_eos": False,
        "max_new_tokens": 64,
    }
    assert metadata["async_workload"] == {
        "scenario": "offline",
        "target_qps": None,
        "worker_count": 4,
        "queue_capacity": 16,
        "schedule_seed": 23,
    }


def test_static_mobilint_profile_is_preserved_in_async_and_failure_metadata():
    args = _async_args(
        "--model",
        "bert-base-uncased",
        "--target",
        "mobilint-aries",
        "--artifact",
        "/models/bert-sst2.mxq",
    )
    result_metadata = {
        "mobilint_artifact_profile_id": (
            "mobilint-bert-base-uncased-tensor-v1"
        )
    }

    run = benchmark_main._async_run_metadata(
        args,
        "NLP_CLASSIFICATION",
        {"target_id": "mobilint-aries"},
        {},
        result_metadata=result_metadata,
    )
    failure = benchmark_main._async_failure_details(
        args=args,
        primary=RuntimeError("benchmark failed"),
        phase="measurement",
        measurement_started=True,
        runtime_diagnostics={},
        task_name="NLP_CLASSIFICATION",
        target_meta={"target_id": "mobilint-aries"},
        result_metadata=result_metadata,
    )

    assert run["mobilint_artifact_profile_id"] == (
        "mobilint-bert-base-uncased-tensor-v1"
    )
    assert failure["run"]["mobilint_artifact_profile_id"] == (
        "mobilint-bert-base-uncased-tensor-v1"
    )


def test_furiosa_version_lookup_failure_is_nonfatal_and_warned(monkeypatch):
    args = _async_args("--backend", "furiosa_llm")
    monkeypatch.setattr(
        benchmark_main,
        "_furiosa_llm_version",
        lambda: None,
    )

    details = benchmark_main._async_failure_details(
        args=args,
        primary=RuntimeError("failed"),
        phase="async_run",
        measurement_started=True,
        runtime_diagnostics={"backend": "furiosa_llm"},
        task_name="NLP_GENERATION",
        target_meta={"target_id": "furiosa-rngd"},
    )

    assert details["run"]["furiosa_llm_version"] is None
    assert "furiosa_llm_version_unavailable" in details["warnings"]


def test_empty_runtime_diagnostics_warn_in_normal_sidecar(
    monkeypatch, tmp_path
):
    _, _, saved, _ = _execute(
        _async_args(),
        tmp_path,
        monkeypatch=monkeypatch,
        runtime_spec={},
    )

    assert saved["details"]["warnings"] == [
        "runtime_device_spec_unavailable"
    ]


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


def test_rbln_runtime_diagnostics_use_exact_bounded_scalar_allowlist():
    class HostileValue:
        def __str__(self):
            raise AssertionError("must not stringify hostile diagnostics")

        def __repr__(self):
            raise AssertionError("must not repr hostile diagnostics")

    class Runtime:
        def get_device_spec(self):
            return {
                "backend": "rbln",
                "device": "0",
                "device_id": 0,
                "accelerator_vendor": "Rebellions",
                "accelerator_name": "RBLN-CA22",
                "detected_npu": "RBLN-CA22",
                "execution_mode": "native_async",
                "sdk_version": "0.11.0",
                "artifact_compiler_version": "0.10.2",
                "artifact_npu": "RBLN-CA22",
                "output_binding_source": "sha256-sidecar",
                "tensor_parallel_size": 1,
                "artifact_uuid": "artifact-uuid",
                "async_parallel": 2,
                "max_async_inflight": 4,
                "artifact_alloc_per_node": np.arange(100_000),
                "input_shapes": [(1, 3, 224, 224)] * 10_000,
                "output_shapes": HostileValue(),
                "descriptor": HostileValue(),
            }

    diagnostics = benchmark_main._safe_runtime_diagnostics(Runtime())

    assert diagnostics == {
        "backend": "rbln",
        "device": "0",
        "device_id": 0,
        "accelerator_vendor": "Rebellions",
        "accelerator_name": "RBLN-CA22",
        "detected_npu": "RBLN-CA22",
        "execution_mode": "native_async",
        "sdk_version": "0.11.0",
        "artifact_compiler_version": "0.10.2",
        "artifact_npu": "RBLN-CA22",
        "output_binding_source": "sha256-sidecar",
        "tensor_parallel_size": 1,
        "artifact_uuid": "artifact-uuid",
        "async_parallel": 2,
        "max_async_inflight": 4,
    }


def test_rbln_runtime_diagnostics_omit_invalid_whitelisted_values():
    class HostileValue:
        def __str__(self):
            raise AssertionError("must not stringify hostile diagnostics")

    class Runtime:
        def get_device_spec(self):
            return {
                "backend": "rbln",
                "device": HostileValue(),
                "device_id": True,
                "accelerator_vendor": "V" * 1024,
                "accelerator_name": "RBLN CA22",
                "detected_npu": HostileValue(),
                "execution_mode": "native/async",
                "sdk_version": float("nan"),
                "artifact_compiler_version": ["0.10.2"],
                "artifact_npu": "RBLN-CA22",
                "tensor_parallel_size": 1.0,
                "artifact_uuid": HostileValue(),
                "async_parallel": float("inf"),
                "max_async_inflight": -1,
            }

    diagnostics = benchmark_main._safe_runtime_diagnostics(Runtime())

    assert diagnostics == {
        "backend": "rbln",
        "artifact_npu": "RBLN-CA22",
    }


def test_runtime_diagnostics_never_compares_hostile_backend_objects():
    class HostileBackend:
        def __eq__(self, other):
            raise AssertionError("must not compare hostile backend values")

        def __str__(self):
            raise AssertionError("must not stringify hostile backend values")

        def __repr__(self):
            raise AssertionError("must not repr hostile backend values")

    class Runtime:
        def get_device_spec(self):
            return {
                "backend": HostileBackend(),
                "descriptor": np.arange(100_000),
            }

    assert benchmark_main._safe_runtime_diagnostics(Runtime()) == {}


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


def test_async_engine_lifecycle_callback_is_debug_only(
    monkeypatch, tmp_path
):
    _, normal_events, _, _ = _execute(
        _async_args(), tmp_path, monkeypatch=monkeypatch
    )
    normal_kwargs = next(
        event[1] for event in normal_events if event[0] == "engine_init"
    )
    assert normal_kwargs.get("lifecycle_callback") is None

    _, debug_events, _, _ = _execute(
        _async_args("--debug"), tmp_path, monkeypatch=monkeypatch
    )
    debug_kwargs = next(
        event[1] for event in debug_events if event[0] == "engine_init"
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
        target=get_target("cpu"),
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


def test_e2e_runtime_execution_failure_unloads_and_never_persists_success(
    monkeypatch,
    capsys,
):
    assert hasattr(runtime_executor_module, "RuntimeExecutionError")
    canonical_error = runtime_executor_module.RuntimeExecutionError(
        error_type="DeviceError",
        error_message="failed",
        dispatch_token=41,
    )
    events = []
    runtime = SimpleNamespace(unload=lambda: events.append("unload"))

    class Runner:
        def __init__(self, **kwargs):
            del kwargs

        def run(self, **kwargs):
            del kwargs
            raise canonical_error

    def reject_save(**kwargs):
        del kwargs
        pytest.fail("failed e2e run must not persist success")

    monkeypatch.setattr(benchmark_main, "BenchmarkRunner", Runner)
    monkeypatch.setattr(benchmark_main, "save_result", reject_save)

    with pytest.raises(runtime_executor_module.RuntimeExecutionError) as raised:
        benchmark_main.execute_benchmark(
            parse([]),
            target=get_target("cpu"),
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

    assert raised.value is canonical_error
    assert events == ["unload"]
    output = capsys.readouterr().out
    assert "RUN_ID=" not in output
    assert "Final Metrics" not in output


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
        assert names.index("unload") < names.index("details")


def test_shutdown_failure_with_zero_outstanding_persists_without_unload(
    monkeypatch, tmp_path
):
    result = _result(
        status=RunStatus.INVALID,
        outstanding=0,
        invalid_reasons=("worker_shutdown_failed",),
    )

    exit_code, events, saved, _ = _execute(
        _async_args(),
        tmp_path,
        monkeypatch=monkeypatch,
        result=result,
        runtime_unload_safe=False,
    )

    assert exit_code == 1
    assert "unload" not in events
    assert saved["details"]["status"] == "invalid"
    assert saved["details"]["invalid_reasons"] == [
        "worker_shutdown_failed"
    ]
    assert saved["csv"]["async_run_status"] == "invalid"
    assert saved["csv"]["async_invalid_reasons"] == (
        "worker_shutdown_failed"
    )


@pytest.mark.parametrize(
    "counter",
    ["missing", None, False, 0.0, 2, HostileOutstandingCounter()],
    ids=("missing", "none", "false", "float-zero", "positive", "hostile"),
)
def test_invalid_outstanding_counter_blocks_unload_and_marks_artifacts_invalid(
    monkeypatch, tmp_path, capsys, counter
):
    result = _result()
    if counter == "missing":
        result.metrics.pop("async_outstanding_requests")
    else:
        result.metrics["async_outstanding_requests"] = counter

    exit_code, events, saved, _ = _execute(
        _async_args(),
        tmp_path,
        monkeypatch=monkeypatch,
        result=result,
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "unload" not in events
    assert saved["details"]["status"] == "invalid"
    assert saved["details"]["invalid_reasons"] == [
        "counter_invariant_failed"
    ]
    assert saved["csv"]["async_run_status"] == "invalid"
    assert saved["csv"]["async_invalid_reasons"] == (
        "counter_invariant_failed"
    )
    if counter == 2:
        assert saved["csv"]["metrics"]["async_outstanding_requests"] == 2
    assert "SECRET" not in captured.out
    assert "SECRET" not in captured.err


@pytest.mark.parametrize("failure_boundary", ["details", "csv"])
@pytest.mark.parametrize(
    "counter",
    ["missing", None, 2, HostileOutstandingCounter()],
    ids=("missing", "none", "positive", "hostile"),
)
def test_invalid_outstanding_counter_never_unloads_on_persistence_failure(
    monkeypatch, tmp_path, capsys, counter, failure_boundary
):
    result = _result()
    if counter == "missing":
        result.metrics.pop("async_outstanding_requests")
    else:
        result.metrics["async_outstanding_requests"] = counter
    persistence_error = OSError(f"forced {failure_boundary} failure")
    events = []
    saved = {}

    with pytest.raises(OSError) as raised:
        _execute(
            _async_args(),
            tmp_path,
            monkeypatch=monkeypatch,
            result=result,
            events=events,
            saved=saved,
            detail_error=(
                persistence_error if failure_boundary == "details" else None
            ),
            csv_error=(
                persistence_error if failure_boundary == "csv" else None
            ),
        )
    captured = capsys.readouterr()

    assert raised.value is persistence_error
    assert "unload" not in events
    assert saved["details"]["status"] == "invalid"
    counter_details = (
        result.details
        if failure_boundary == "details"
        else saved["details"]
    )
    assert "counter_invariant_failed" in counter_details[
        "invalid_reasons"
    ]
    assert saved["failure_details"]["failure"]["phase"] == (
        "sidecar_save" if failure_boundary == "details" else "csv_save"
    )
    assert "SECRET" not in captured.out
    assert "SECRET" not in captured.err


def test_invalid_counter_proof_precedes_runtime_safety_getter(
    monkeypatch, tmp_path, capsys
):
    result = _result()
    result.metrics.pop("async_outstanding_requests")
    safety_reads = []
    safety_error = AssertionError("SECRET safety getter")

    exit_code, events, saved, _ = _execute(
        _async_args(),
        tmp_path,
        monkeypatch=monkeypatch,
        result=result,
        runtime_unload_safe_error=safety_error,
        runtime_unload_safe_reads=safety_reads,
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert safety_reads == []
    assert "unload" not in events
    assert saved["details"]["status"] == "invalid"
    assert "counter_invariant_failed" in saved["details"][
        "invalid_reasons"
    ]
    assert "SECRET" not in captured.out
    assert "SECRET" not in captured.err


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

    class Engine:
        failure_phase = "complete"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_async(self, config, warmup_runs, monitor):
            del config, warmup_runs, monitor
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
    monkeypatch.setattr(benchmark_main, "InferenceEngine", Engine)
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
            target=get_target("cpu"),
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


def test_post_run_runtime_unload_baseexception_records_exact_phase(
    monkeypatch, tmp_path, capsys
):
    class FatalRuntimeUnload(BaseException):
        pass

    primary = FatalRuntimeUnload("SECRET unload payload")
    reservation = _reservation(tmp_path)
    unload_calls = []
    detail_calls = []
    csv_calls = []

    class Engine:
        failure_phase = "complete"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_async(self, config, warmup_runs, monitor):
            del config, warmup_runs, monitor
            return _result()

    def save_details(run_id, details, *, results_dir, reservation):
        del run_id, results_dir
        detail_calls.append(details)
        return reservation.details_path

    def save_csv(**kwargs):
        csv_calls.append(kwargs)
        return kwargs["run_id"]

    def unload():
        unload_calls.append("unload")
        raise primary

    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    monkeypatch.setattr(benchmark_main, "InferenceEngine", Engine)
    monkeypatch.setattr(benchmark_main, "save_async_details", save_details)
    monkeypatch.setattr(benchmark_main, "save_result", save_csv)
    runtime = SimpleNamespace(
        get_device_spec=lambda: {
            "backend": "onnxruntime",
            "device": "cpu",
            "active_providers": ["CPUExecutionProvider"],
        },
        unload=unload,
    )

    with pytest.raises(FatalRuntimeUnload) as raised:
        benchmark_main.execute_benchmark(
            _async_args(),
            target=get_target("cpu"),
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
    assert unload_calls == ["unload"]
    assert detail_calls[-1]["failure"] == {
        "phase": "runtime_unload",
        "error_type": "FatalRuntimeUnload",
        "error_message": (
            "benchmark failed during runtime_unload (FatalRuntimeUnload)"
        ),
    }
    assert "SECRET unload payload" not in json.dumps(detail_calls[-1])
    assert [call["run_id"] for call in csv_calls] == [reservation.run_id]
    assert captured.out.splitlines().count("RUN_ID_RESERVED=async001") == 1
    assert captured.out.splitlines().count("RUN_ID=async001") == 1


def test_committed_normal_sidecar_and_csv_baseexception_gets_recovery_record(
    monkeypatch, tmp_path, capsys
):
    class FatalCsvSave(BaseException):
        pass

    primary = FatalCsvSave("SECRET csv payload")
    results_path = tmp_path / "results" / "benchmark_results.csv"
    real_reserve = benchmark_main.reserve_run_artifacts
    real_save_result = benchmark_main.save_result
    reservation = real_reserve(results_path=results_path, run_id="async001")
    committed = {}
    save_calls = 0

    class Engine:
        failure_phase = "complete"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_async(self, config, warmup_runs, monitor):
            del config, warmup_runs, monitor
            return _result()

    def commit_then_raise(**kwargs):
        nonlocal save_calls
        save_calls += 1
        run_id = real_save_result(**kwargs)
        if save_calls == 1:
            committed["details"] = reservation.details_path.read_bytes()
            committed["csv"] = results_path.read_bytes()
            raise primary
        return run_id

    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    monkeypatch.setattr(benchmark_main, "InferenceEngine", Engine)
    monkeypatch.setattr(benchmark_main, "save_result", commit_then_raise)
    runtime = SimpleNamespace(
        get_device_spec=lambda: {
            "backend": "onnxruntime",
            "device": "cpu",
            "active_providers": ["CPUExecutionProvider"],
        },
        unload=lambda: None,
    )

    with pytest.raises(FatalCsvSave) as raised:
        _execute_with_runtime(_async_args(), tmp_path, runtime)
    captured = capsys.readouterr()

    assert raised.value is primary
    assert reservation.details_path.read_bytes() == committed["details"]
    assert results_path.read_bytes() == committed["csv"]
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["run_id"] == reservation.run_id
    assert rows[0]["async_run_status"] == "valid"
    recovery = json.loads(
        reservation.failure_details_path.read_text(encoding="utf-8")
    )
    assert recovery["failure"] == {
        "phase": "csv_save",
        "error_type": "FatalCsvSave",
        "error_message": (
            "benchmark failed during csv_save (FatalCsvSave)"
        ),
    }
    assert "SECRET csv payload" not in json.dumps(recovery, sort_keys=True)
    assert captured.out.splitlines().count("RUN_ID_RESERVED=async001") == 1
    assert captured.out.splitlines().count("RUN_ID=async001") == 1


def test_consumed_normal_csv_without_recovery_emits_no_terminal_run_id(
    monkeypatch, tmp_path, capsys
):
    class FatalCsvSave(BaseException):
        pass

    primary = FatalCsvSave("SECRET consumed csv payload")
    recovery_error = OSError("recovery publication unavailable")
    results_path = tmp_path / "results" / "benchmark_results.csv"
    reservation = benchmark_main.reserve_run_artifacts(
        results_path=results_path,
        run_id="async001",
    )
    real_save_result = benchmark_main.save_result
    committed = {}

    class Engine:
        failure_phase = "complete"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_async(self, config, warmup_runs, monitor):
            del config, warmup_runs, monitor
            return _result()

    def commit_then_raise(**kwargs):
        run_id = real_save_result(**kwargs)
        committed["details"] = reservation.details_path.read_bytes()
        committed["csv"] = results_path.read_bytes()
        raise primary

    def fail_recovery(*args, **kwargs):
        del args, kwargs
        raise recovery_error

    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    monkeypatch.setattr(benchmark_main, "InferenceEngine", Engine)
    monkeypatch.setattr(benchmark_main, "save_result", commit_then_raise)
    monkeypatch.setattr(
        benchmark_main,
        "save_async_failure_details",
        fail_recovery,
    )
    runtime = SimpleNamespace(
        get_device_spec=lambda: {
            "backend": "onnxruntime",
            "device": "cpu",
            "active_providers": ["CPUExecutionProvider"],
        },
        unload=lambda: None,
    )

    with pytest.raises(FatalCsvSave) as raised:
        _execute_with_runtime(_async_args(), tmp_path, runtime)
    captured = capsys.readouterr()

    assert raised.value is primary
    assert reservation.details_path.read_bytes() == committed["details"]
    assert results_path.read_bytes() == committed["csv"]
    assert not reservation.failure_details_path.exists()
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["async_run_status"] == "valid"
    assert captured.out.splitlines().count("RUN_ID_RESERVED=async001") == 1
    assert captured.out.splitlines().count("RUN_ID=async001") == 0


def test_pending_normal_csv_without_recovery_emits_no_terminal_run_id(
    monkeypatch, tmp_path, capsys
):
    class FatalPendingCsvSave(BaseException):
        pass

    primary = FatalPendingCsvSave("SECRET pending csv payload")
    recovery_error = OSError("recovery publication unavailable")
    results_path = tmp_path / "results" / "benchmark_results.csv"
    reservation = benchmark_main.reserve_run_artifacts(
        results_path=results_path,
        run_id="async001",
    )
    real_atomic_write_csv = result_store_module._atomic_write_csv_at
    atomic_calls = 0
    committed = {}

    class Engine:
        failure_phase = "complete"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_async(self, config, warmup_runs, monitor):
            del config, warmup_runs, monitor
            return _result()

    def leave_pending_then_retry(*args, **kwargs):
        nonlocal atomic_calls
        atomic_calls += 1
        if atomic_calls == 1:
            committed["details"] = reservation.details_path.read_bytes()
            committed["pending"] = json.loads(
                reservation.pending_path.read_text(encoding="utf-8")
            )
            raise primary
        return real_atomic_write_csv(*args, **kwargs)

    def fail_recovery(*args, **kwargs):
        del args, kwargs
        raise recovery_error

    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    monkeypatch.setattr(benchmark_main, "InferenceEngine", Engine)
    monkeypatch.setattr(
        result_store_module,
        "_atomic_write_csv_at",
        leave_pending_then_retry,
    )
    monkeypatch.setattr(
        benchmark_main,
        "save_async_failure_details",
        fail_recovery,
    )
    runtime = SimpleNamespace(
        get_device_spec=lambda: {
            "backend": "onnxruntime",
            "device": "cpu",
            "active_providers": ["CPUExecutionProvider"],
        },
        unload=lambda: None,
    )

    with pytest.raises(FatalPendingCsvSave) as raised:
        _execute_with_runtime(_async_args(), tmp_path, runtime)
    captured = capsys.readouterr()

    assert raised.value is primary
    normal_bytes = reservation.details_path.read_bytes()
    assert normal_bytes == committed["details"]
    assert json.loads(normal_bytes)["status"] == "valid"
    assert not reservation.failure_details_path.exists()
    assert not reservation.pending_path.exists()
    assert reservation.consumed_path.exists()
    consumed = json.loads(
        reservation.consumed_path.read_text(encoding="utf-8")
    )
    assert consumed["row_fingerprint"] == committed["pending"][
        "row_fingerprint"
    ]
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["async_run_status"] == "valid"
    assert captured.out.splitlines().count("RUN_ID_RESERVED=async001") == 1
    assert captured.out.splitlines().count("RUN_ID=async001") == 0


def test_committed_normal_sidecar_baseexception_gets_recovery_record(
    monkeypatch, tmp_path, capsys
):
    class FatalSidecarSave(BaseException):
        pass

    primary = FatalSidecarSave("SECRET sidecar payload")
    results_path = tmp_path / "results" / "benchmark_results.csv"
    reservation = benchmark_main.reserve_run_artifacts(
        results_path=results_path,
        run_id="async001",
    )
    real_save_details = benchmark_main.save_async_details
    committed = {}
    save_calls = 0

    class Engine:
        failure_phase = "complete"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_async(self, config, warmup_runs, monitor):
            del config, warmup_runs, monitor
            return _result()

    def commit_then_raise(run_id, details, *, results_dir, reservation):
        nonlocal save_calls
        save_calls += 1
        details_path = real_save_details(
            run_id,
            details,
            results_dir=results_dir,
            reservation=reservation,
        )
        if save_calls == 1:
            committed["details"] = details_path.read_bytes()
            raise primary
        return details_path

    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    monkeypatch.setattr(benchmark_main, "InferenceEngine", Engine)
    monkeypatch.setattr(
        benchmark_main,
        "save_async_details",
        commit_then_raise,
    )
    runtime = SimpleNamespace(
        get_device_spec=lambda: {
            "backend": "onnxruntime",
            "device": "cpu",
            "active_providers": ["CPUExecutionProvider"],
        },
        unload=lambda: None,
    )

    with pytest.raises(FatalSidecarSave) as raised:
        _execute_with_runtime(_async_args(), tmp_path, runtime)
    captured = capsys.readouterr()

    assert raised.value is primary
    assert reservation.details_path.read_bytes() == committed["details"]
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["async_run_status"] == "invalid"
    assert rows[0]["details_path"] == ""
    assert rows[0]["failure_details_path"] == (
        "results/details/async001.failure.json"
    )
    recovery = json.loads(
        reservation.failure_details_path.read_text(encoding="utf-8")
    )
    assert recovery["failure"] == {
        "phase": "sidecar_save",
        "error_type": "FatalSidecarSave",
        "error_message": (
            "benchmark failed during sidecar_save (FatalSidecarSave)"
        ),
    }
    assert "SECRET sidecar payload" not in json.dumps(
        recovery,
        sort_keys=True,
    )
    assert captured.out.splitlines().count("RUN_ID_RESERVED=async001") == 1
    assert captured.out.splitlines().count("RUN_ID=async001") == 1


def test_precommit_normal_sidecar_exception_uses_failure_artifacts(
    monkeypatch, tmp_path, capsys
):
    primary = OSError("SECRET precommit sidecar payload")
    results_path = tmp_path / "results" / "benchmark_results.csv"
    reservation = benchmark_main.reserve_run_artifacts(
        results_path=results_path,
        run_id="async001",
    )
    real_save_details = benchmark_main.save_async_details
    save_calls = 0

    class Engine:
        failure_phase = "complete"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_async(self, config, warmup_runs, monitor):
            del config, warmup_runs, monitor
            return _result()

    def fail_before_commit_once(
        run_id,
        details,
        *,
        results_dir,
        reservation,
    ):
        nonlocal save_calls
        save_calls += 1
        if save_calls == 1:
            raise primary
        return real_save_details(
            run_id,
            details,
            results_dir=results_dir,
            reservation=reservation,
        )

    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    monkeypatch.setattr(benchmark_main, "InferenceEngine", Engine)
    monkeypatch.setattr(
        benchmark_main,
        "save_async_details",
        fail_before_commit_once,
    )
    runtime = SimpleNamespace(
        get_device_spec=lambda: {
            "backend": "onnxruntime",
            "device": "cpu",
            "active_providers": ["CPUExecutionProvider"],
        },
        unload=lambda: None,
    )

    with pytest.raises(OSError) as raised:
        _execute_with_runtime(_async_args(), tmp_path, runtime)
    captured = capsys.readouterr()

    assert raised.value is primary
    failure = json.loads(
        reservation.details_path.read_text(encoding="utf-8")
    )
    assert failure["failure"] == {
        "phase": "sidecar_save",
        "error_type": "OSError",
        "error_message": "benchmark failed during sidecar_save (OSError)",
    }
    assert "SECRET precommit sidecar payload" not in json.dumps(
        failure,
        sort_keys=True,
    )
    assert not reservation.failure_details_path.exists()
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["async_run_status"] == "invalid"
    assert rows[0]["details_path"] == "results/details/async001.json"
    assert rows[0]["failure_details_path"] == ""
    assert captured.out.splitlines().count("RUN_ID_RESERVED=async001") == 1
    assert captured.out.splitlines().count("RUN_ID=async001") == 1


def test_committed_normal_sidecar_exception_uses_immutable_recovery(
    monkeypatch, tmp_path, capsys
):
    primary = OSError("SECRET committed sidecar payload")
    primary.final_file_committed = True
    primary.publication_state_uncertain = False
    results_path = tmp_path / "results" / "benchmark_results.csv"
    reservation = benchmark_main.reserve_run_artifacts(
        results_path=results_path,
        run_id="async001",
    )
    real_save_details = benchmark_main.save_async_details
    committed = {}

    class Engine:
        failure_phase = "complete"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_async(self, config, warmup_runs, monitor):
            del config, warmup_runs, monitor
            return _result()

    def commit_then_raise(
        run_id,
        details,
        *,
        results_dir,
        reservation,
    ):
        details_path = real_save_details(
            run_id,
            details,
            results_dir=results_dir,
            reservation=reservation,
        )
        committed["details"] = details_path.read_bytes()
        raise primary

    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    monkeypatch.setattr(benchmark_main, "InferenceEngine", Engine)
    monkeypatch.setattr(
        benchmark_main,
        "save_async_details",
        commit_then_raise,
    )
    runtime = SimpleNamespace(
        get_device_spec=lambda: {
            "backend": "onnxruntime",
            "device": "cpu",
            "active_providers": ["CPUExecutionProvider"],
        },
        unload=lambda: None,
    )

    with pytest.raises(OSError) as raised:
        _execute_with_runtime(_async_args(), tmp_path, runtime)
    captured = capsys.readouterr()

    assert raised.value is primary
    assert reservation.details_path.read_bytes() == committed["details"]
    normal = json.loads(committed["details"])
    assert normal["status"] == "valid"
    recovery = json.loads(
        reservation.failure_details_path.read_text(encoding="utf-8")
    )
    assert recovery["failure"] == {
        "phase": "sidecar_save",
        "error_type": "OSError",
        "error_message": "benchmark failed during sidecar_save (OSError)",
    }
    assert "SECRET committed sidecar payload" not in json.dumps(
        recovery,
        sort_keys=True,
    )
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["async_run_status"] == "invalid"
    assert rows[0]["details_path"] == "results/details/async001.json"
    assert rows[0]["failure_details_path"] == (
        "results/details/async001.failure.json"
    )
    assert captured.out.splitlines().count("RUN_ID_RESERVED=async001") == 1
    assert captured.out.splitlines().count("RUN_ID=async001") == 1


def test_writable_csv_failure_links_recovery_record_without_normal_overwrite(
    monkeypatch, tmp_path, capsys
):
    class FatalCsvSave(BaseException):
        pass

    primary = FatalCsvSave("SECRET csv payload")
    results_path = tmp_path / "results" / "benchmark_results.csv"
    reservation = benchmark_main.reserve_run_artifacts(
        results_path=results_path,
        run_id="async001",
    )
    real_save_result = benchmark_main.save_result
    save_calls = 0

    class Engine:
        failure_phase = "complete"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_async(self, config, warmup_runs, monitor):
            del config, warmup_runs, monitor
            return _result()

    def fail_before_store_once(**kwargs):
        nonlocal save_calls
        save_calls += 1
        if save_calls == 1:
            raise primary
        return real_save_result(**kwargs)

    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    monkeypatch.setattr(benchmark_main, "InferenceEngine", Engine)
    monkeypatch.setattr(
        benchmark_main,
        "save_result",
        fail_before_store_once,
    )
    runtime = SimpleNamespace(
        get_device_spec=lambda: {
            "backend": "onnxruntime",
            "device": "cpu",
            "active_providers": ["CPUExecutionProvider"],
        },
        unload=lambda: None,
    )

    with pytest.raises(FatalCsvSave) as raised:
        _execute_with_runtime(_async_args(), tmp_path, runtime)
    captured = capsys.readouterr()

    assert raised.value is primary
    with results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["async_run_status"] == "invalid"
    assert rows[0]["failure_details_path"] == (
        "results/details/async001.failure.json"
    )
    normal = json.loads(
        reservation.details_path.read_text(encoding="utf-8")
    )
    recovery = json.loads(
        reservation.failure_details_path.read_text(encoding="utf-8")
    )
    assert normal["status"] == "valid"
    assert recovery["failure"]["phase"] == "csv_save"
    assert "SECRET csv payload" not in json.dumps(recovery, sort_keys=True)
    assert captured.out.splitlines().count("RUN_ID_RESERVED=async001") == 1
    assert captured.out.splitlines().count("RUN_ID=async001") == 1


def test_consumed_normal_artifacts_preserve_runtime_unload_failure_truth(
    tmp_path, capsys
):
    class FatalRuntimeUnload(BaseException):
        pass

    primary = FatalRuntimeUnload("SECRET unload payload")
    args = _async_args()
    config = benchmark_main.build_async_config(args)
    reservation = benchmark_main.reserve_run_artifacts(
        results_path=tmp_path / "results" / "benchmark_results.csv",
        run_id="async001",
    )
    benchmark_main.save_async_details(
        reservation.run_id,
        {"status": "valid", "warnings": []},
        results_dir=reservation.results_root,
        reservation=reservation,
    )
    benchmark_main.save_result(
        metrics={"accuracy": 1.0},
        model_name=args.model,
        task="IMAGE_CLASSIFICATION",
        backend=args.backend,
        device=args.device,
        batch_size=args.batch_size,
        warmup_runs=args.warmup,
        results_path=reservation.results_path,
        run_id=reservation.run_id,
        inference_mode="async_queue",
        scenario=config.scenario.value,
        async_run_status="valid",
        details_path="details/async001.json",
        reservation=reservation,
    )
    original_details = reservation.details_path.read_bytes()
    original_csv = reservation.results_path.read_bytes()
    print(f"RUN_ID_RESERVED={reservation.run_id}")

    with pytest.raises(FatalRuntimeUnload) as raised:
        try:
            raise primary
        except BaseException:
            terminal_ready = benchmark_main._persist_async_failure(
                args=args,
                config=config,
                reservation=reservation,
                primary=primary,
                phase="runtime_unload",
                measurement_started=True,
                runtime_diagnostics={
                    "backend": "onnxruntime",
                    "device": "cpu",
                    "active_providers": ["CPUExecutionProvider"],
                },
                task_name="IMAGE_CLASSIFICATION",
                target_meta={"target_id": "cpu"},
                primary_details_committed=True,
                csv_committed=True,
            )
            if terminal_ready:
                print(f"RUN_ID={reservation.run_id}")
            raise
    captured = capsys.readouterr()

    assert raised.value is primary
    assert reservation.details_path.read_bytes() == original_details
    assert reservation.results_path.read_bytes() == original_csv
    recovery = json.loads(
        reservation.failure_details_path.read_text(encoding="utf-8")
    )
    assert recovery["failure"]["phase"] == "runtime_unload"
    assert "SECRET unload payload" not in json.dumps(recovery, sort_keys=True)
    assert captured.out.splitlines().count("RUN_ID_RESERVED=async001") == 1
    assert captured.out.splitlines().count("RUN_ID=async001") == 1


def test_failure_recovery_records_persistence_error_and_csv_link(
    monkeypatch, tmp_path
):
    primary = RuntimeError("SECRET primary payload")
    secondary = OSError("SECRET sidecar payload")
    args = _async_args()
    config = benchmark_main.build_async_config(args)
    reservation = benchmark_main.reserve_run_artifacts(
        results_path=tmp_path / "results" / "benchmark_results.csv",
        run_id="async001",
    )
    monkeypatch.setattr(
        benchmark_main,
        "save_async_details",
        lambda *args, **kwargs: (_ for _ in ()).throw(secondary),
    )

    with pytest.raises(RuntimeError) as raised:
        try:
            raise primary
        except BaseException:
            assert benchmark_main._persist_async_failure(
                args=args,
                config=config,
                reservation=reservation,
                primary=primary,
                phase="warmup",
                measurement_started=False,
                runtime_diagnostics={},
                task_name="IMAGE_CLASSIFICATION",
                target_meta={"target_id": "cpu"},
                primary_details_committed=False,
                csv_committed=False,
            )
            raise

    assert raised.value is primary
    recovery = json.loads(
        reservation.failure_details_path.read_text(encoding="utf-8")
    )
    assert recovery["cleanup_secondary_errors"] == [
        {
            "phase": "failure_sidecar",
            "error_type": "OSError",
            "error_message": (
                "secondary failure during failure_sidecar (OSError)"
            ),
        }
    ]
    serialized = json.dumps(recovery, sort_keys=True)
    assert "SECRET primary payload" not in serialized
    assert "SECRET sidecar payload" not in serialized
    with reservation.results_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["failure_details_path"] == (
        "results/details/async001.failure.json"
    )


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
            raise RuntimeError("api-token=secondary-trace-secret")

    class Engine:
        failure_phase = "created"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_async(self, config, warmup_runs, monitor):
            del config, warmup_runs, monitor
            raise LookupError("primary runner error")

    monkeypatch.setattr(benchmark_main, "RequestTraceWriter", TraceWriter)
    monkeypatch.setattr(benchmark_main, "InferenceEngine", Engine)

    with pytest.raises(LookupError, match="primary runner error") as raised:
        benchmark_main.execute_benchmark(
            args,
            target=get_target("cpu"),
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
    notes = "\n".join(raised.value.__notes__)
    assert "request_trace_cleanup" in notes
    assert "RuntimeError" in notes
    assert "api-token=secondary-trace-secret" not in notes


def _execute_with_runtime(args, tmp_path, runtime):
    return benchmark_main.execute_benchmark(
        args,
        target=get_target("cpu"),
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
            target=get_target("cpu"),
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

    class Engine:
        failure_phase = "warmup"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_async(self, config, warmup_runs, monitor):
            del config, warmup_runs, monitor
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
    monkeypatch.setattr(benchmark_main, "InferenceEngine", Engine)
    monkeypatch.setattr(benchmark_main, "save_async_details", save_details)
    monkeypatch.setattr(
        benchmark_main,
        "save_result",
        lambda **kwargs: kwargs["run_id"],
    )
    monkeypatch.setattr(
        benchmark_main,
        "save_async_failure_details",
        lambda *args, **kwargs: reservation.failure_details_path,
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


def test_failure_details_snapshot_safe_cleanup_and_runtime_warning(
    monkeypatch, tmp_path
):
    primary = RuntimeError("SECRET primary payload")
    secondary = OSError("SECRET cleanup payload")
    reservation = _reservation(tmp_path)
    saved_details = {}

    class Engine:
        failure_phase = "warmup"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_async(self, config, warmup_runs, monitor):
            del config, warmup_runs, monitor
            raise primary

    def save_details(run_id, details, *, results_dir, reservation):
        del run_id, results_dir
        saved_details.update(details)
        return reservation.details_path

    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    monkeypatch.setattr(benchmark_main, "InferenceEngine", Engine)
    monkeypatch.setattr(benchmark_main, "save_async_details", save_details)
    monkeypatch.setattr(
        benchmark_main,
        "save_result",
        lambda **kwargs: kwargs["run_id"],
    )
    runtime = SimpleNamespace(
        get_device_spec=lambda: {},
        unload=lambda: (_ for _ in ()).throw(secondary),
    )

    with pytest.raises(RuntimeError) as raised:
        _execute_with_runtime(_async_args(), tmp_path, runtime)

    assert raised.value is primary
    assert saved_details["warnings"] == [
        "runtime_device_spec_unavailable"
    ]
    assert saved_details["cleanup_secondary_errors"] == [
        {
            "phase": "runtime_unload",
            "error_type": "OSError",
            "error_message": (
                "secondary failure during runtime_unload (OSError)"
            ),
        }
    ]
    serialized = json.dumps(saved_details, sort_keys=True)
    assert "SECRET primary payload" not in serialized
    assert "SECRET cleanup payload" not in serialized


def test_warmup_failure_persistence_sidecar_error_is_secondary(
    monkeypatch, tmp_path, capsys
):
    primary = RuntimeError("warmup failed")
    secondary = OSError("sidecar failed")
    reservation = _reservation(tmp_path)

    class Engine:
        failure_phase = "warmup"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_async(self, config, warmup_runs, monitor):
            del config, warmup_runs, monitor
            raise primary

    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    monkeypatch.setattr(benchmark_main, "InferenceEngine", Engine)
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
    monkeypatch.setattr(
        benchmark_main,
        "save_async_failure_details",
        lambda *args, **kwargs: reservation.failure_details_path,
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

    class Engine:
        failure_phase = "warmup"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_async(self, config, warmup_runs, monitor):
            del config, warmup_runs, monitor
            raise primary

    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    monkeypatch.setattr(benchmark_main, "InferenceEngine", Engine)
    monkeypatch.setattr(
        benchmark_main,
        "save_async_details",
        lambda *args, **kwargs: (_ for _ in ()).throw(secondary),
    )

    def save_csv(**kwargs):
        saved_csv.update(kwargs)
        return kwargs["run_id"]

    monkeypatch.setattr(benchmark_main, "save_result", save_csv)
    monkeypatch.setattr(
        benchmark_main,
        "save_async_failure_details",
        lambda *args, **kwargs: reservation.failure_details_path,
    )
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

    class Engine:
        failure_phase = "warmup"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_async(self, config, warmup_runs, monitor):
            del config, warmup_runs, monitor
            raise primary

    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    monkeypatch.setattr(benchmark_main, "InferenceEngine", Engine)
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
    monkeypatch.setattr(
        benchmark_main,
        "save_async_failure_details",
        lambda *args, **kwargs: reservation.failure_details_path,
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

    class Engine:
        failure_phase = "measurement"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_async(self, config, warmup_runs, monitor):
            del config, warmup_runs, monitor
            raise primary

    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    monkeypatch.setattr(benchmark_main, "InferenceEngine", Engine)

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
    monkeypatch.setattr(
        benchmark_main,
        "save_async_failure_details",
        lambda *args, **kwargs: reservation.failure_details_path,
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

    class Engine:
        failure_phase = "warmup"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_async(self, config, warmup_runs, monitor):
            del config, warmup_runs, monitor
            raise primary

    monkeypatch.setattr(
        benchmark_main,
        "reserve_run_artifacts",
        lambda **kwargs: reservation,
    )
    monkeypatch.setattr(benchmark_main, "InferenceEngine", Engine)
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

    class Engine:
        failure_phase = "warmup"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_async(self, config, warmup_runs, monitor):
            del config, warmup_runs, monitor
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
    monkeypatch.setattr(benchmark_main, "InferenceEngine", Engine)
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
        "save_async_failure_details",
        lambda *args, **kwargs: reservation.failure_details_path,
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
        def __init__(
            self,
            runtime,
            pipeline,
            config,
            coordinator,
            metrics,
            executor=None,
        ):
            del runtime, pipeline, config, coordinator, metrics, executor

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
            target=get_target("cpu"),
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

    class Engine:
        failure_phase = "created"
        runtime_unload_safe_after_failure = True

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_async(self, config, warmup_runs, monitor):
            del config, warmup_runs, monitor
            raise primary

    monkeypatch.setattr(benchmark_main, "InferenceEngine", Engine)

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
            target=get_target("cpu"),
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
    events = []
    saved = {}

    with pytest.raises(OSError) as raised:
        _execute(
            _async_args(),
            tmp_path,
            monkeypatch=monkeypatch,
            detail_error=error,
            events=events,
            saved=saved,
        )

    assert raised.value is error
    assert saved["csv"]["details_path"] == "results/details/async001.json"
    assert saved["csv"]["failure_details_path"] == (
        "results/details/async001.failure.json"
    )
    assert saved["csv"]["async_run_status"] == "invalid"
    assert saved["failure_details"]["failure"]["phase"] == "sidecar_save"
    names = [event[0] if isinstance(event, tuple) else event for event in events]
    assert names.index("unload") < names.index("details") < names.index("csv")


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
    events = []
    saved = {}

    with pytest.raises(HostilePersistenceError) as raised:
        _execute(
            _async_args(),
            tmp_path,
            monkeypatch=monkeypatch,
            detail_error=error,
            events=events,
            saved=saved,
        )
    captured = capsys.readouterr()

    assert raised.value is error
    assert saved["csv"]["details_path"] == ""
    assert saved["csv"]["failure_details_path"] == (
        "results/details/async001.failure.json"
    )
    assert saved["csv"]["async_run_status"] == "invalid"
    assert saved["failure_details"]["failure"] == {
        "phase": "sidecar_save",
        "error_type": "HostilePersistenceError",
        "error_message": (
            "benchmark failed during sidecar_save "
            "(HostilePersistenceError)"
        ),
    }
    assert "safe detail failure" not in json.dumps(
        saved["failure_details"],
        sort_keys=True,
    )
    names = [event[0] if isinstance(event, tuple) else event for event in events]
    assert names.index("unload") < names.index("details") < names.index("csv")
    assert "save_async_details" in captured.err


def test_uncertain_csv_commit_returns_nonzero_with_only_reserved_run_id(
    monkeypatch, tmp_path, capsys
):
    error = OSError("results directory fsync failed")
    error.publication_state_uncertain = True
    events = []
    reservation = _reservation(tmp_path)

    with pytest.raises(OSError) as raised:
        _execute(
            _async_args(),
            tmp_path,
            monkeypatch=monkeypatch,
            csv_error=error,
            events=events,
        )
    captured = capsys.readouterr()
    names = [event[0] if isinstance(event, tuple) else event for event in events]

    assert raised.value is error
    assert names.index("unload") < names.index("details") < names.index("csv")
    assert captured.out.splitlines().count(
        f"RUN_ID_RESERVED={reservation.run_id}"
    ) == 1
    assert captured.out.splitlines().count(f"RUN_ID={reservation.run_id}") == 0
    assert "결과 CSV 저장 실패" in captured.err
