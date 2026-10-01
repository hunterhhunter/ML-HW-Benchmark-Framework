"""One CLI async run links a quality pass to a runtime-only performance pass."""

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import main as benchmark_main
from core.targets import get_target
from evaluators.latency_evaluator import LatencyOnlyEvaluator


class Loader:
    def __init__(self):
        self.samples = [
            {"input": np.asarray([float(index)]), "label": float(index * 2)}
            for index in range(3)
        ]
        self.current_idx = 0
        self.total_samples = 3

    def get_metadata(self):
        return {"total_samples": 3, "is_static_batched": False}

    def load_by_index(self, index):
        return self.samples[index]

    def load_batch(self, batch_size):
        start = self.current_idx
        self.current_idx = min(start + batch_size, 3)
        return self.samples[start:self.current_idx]


class Runtime:
    compiled_model = None

    def __init__(self):
        self.calls = 0
        self.unloads = 0

    def supports_generate(self):
        return False

    def max_concurrent_workers(self):
        return 1

    def supports_dynamic_batching(self):
        return True

    def max_dynamic_batch_size(self):
        return None

    def run(self, inputs):
        self.calls += 1
        return {"output": inputs["input"] * 2}

    def warmup(self, inputs, num_runs=1):
        return None

    def get_device_spec(self):
        return {"backend": "onnxruntime", "active_providers": ["CPUExecutionProvider"]}

    def unload(self):
        self.unloads += 1


class Evaluator:
    def __init__(self):
        self.samples = 0

    def add_batch(self, outputs, labels, timing_ms):
        assert np.array_equal(outputs["output"].reshape(-1), np.asarray(labels))
        self.samples += len(labels)

    def compute(self):
        return {"accuracy": 1.0, "Total Samples": self.samples}


class Decoder:
    def __init__(self):
        self.calls = 0

    def decode(self, outputs):
        self.calls += 1
        return outputs


def test_cli_runs_quality_then_runtime_only_under_one_result_id(tmp_path):
    results_path = tmp_path / "results.csv"
    args = benchmark_main.build_parser().parse_args([
        "--model", "resnet50",
        "--target", "cpu",
        "--backend", "onnxruntime",
        "--inference-mode", "async_queue",
        "--max-samples", "3",
        "--min-samples", "1",
        "--warmup", "0",
        "--results-path", str(results_path),
    ])
    runtime = Runtime()
    evaluator = Evaluator()
    decoder = Decoder()
    result = benchmark_main.execute_benchmark(
        args,
        target=get_target("cpu"),
        loader=Loader(),
        runtime=runtime,
        evaluator=evaluator,
        decoder=decoder,
        hw_monitor=None,
        task_name="IMAGE_CLASSIFICATION",
        target_meta=benchmark_main.target_metadata(get_target("cpu"), {}),
        results_path=results_path,
    )

    assert result == 0
    assert runtime.calls == 6
    assert runtime.unloads == 1
    assert evaluator.samples == 3
    assert decoder.calls == 3
    with results_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["quality_status"] == "passed"
    assert row["comparison_eligible"] == "True"
    assert row["async_metric_schema_version"] == "2"
    assert float(row["async_runtime_completed_samples_per_sec"]) > 0
    assert not row.get("async_completed_samples_per_sec")
    details = json.loads((tmp_path.parent / row["details_path"]).read_text())
    assert details["quality_phase"]["evaluator_samples"] == 3
    assert details["performance_phase"]["completed_samples"] == 3
    assert details["async_metric_schema_version"] == 2
    assert details["async_measurement_boundary"] == "framework_runtime_call_return"

def _args(results_path):
    return benchmark_main.build_parser().parse_args([
        "--model", "resnet50", "--target", "cpu", "--backend", "onnxruntime",
        "--inference-mode", "async_queue", "--max-samples", "3",
        "--min-samples", "1", "--warmup", "0",
        "--results-path", str(results_path),
    ])

def _execute(results_path, runtime, evaluator, decoder, args=None):
    target = get_target("cpu")
    return benchmark_main.execute_benchmark(
        args or _args(results_path), target=target, loader=Loader(), runtime=runtime,
        evaluator=evaluator, decoder=decoder, hw_monitor=None,
        task_name="IMAGE_CLASSIFICATION",
        target_meta=benchmark_main.target_metadata(target, {}),
        results_path=results_path,
    )

def test_accuracy_failure_stops_before_performance_and_persists_reason(tmp_path):
    class FailingEvaluator(Evaluator):
        def compute(self):
            raise RuntimeError("accuracy computation failed")

    results_path = tmp_path / "results.csv"
    runtime = Runtime()
    with pytest.raises(benchmark_main.AccuracyPhaseFailed):
        _execute(results_path, runtime, FailingEvaluator(), Decoder())

    assert runtime.calls == 3
    with results_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["async_invalid_reasons"] == "accuracy_failed"
    assert rows[0]["quality_status"] == "failed"
    assert rows[0]["comparison_eligible"] == "False"
    assert not rows[0].get("async_runtime_completed_samples_per_sec")
    details = json.loads((tmp_path.parent / rows[0]["details_path"]).read_text())
    assert details["quality_phase"]["status"] == "failed"
    assert details["async_metric_schema_version"] == 2
    assert details["async_measurement_boundary"] == "framework_runtime_call_return"

def test_latency_only_evaluator_runs_performance_only_and_is_ineligible(tmp_path):
    results_path = tmp_path / "results.csv"
    runtime = Runtime()
    decoder = Decoder()
    status = _execute(results_path, runtime, LatencyOnlyEvaluator(), decoder)

    assert status == 0
    assert runtime.calls == 3
    assert decoder.calls == 0
    with results_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["quality_status"] == "unavailable"
    assert rows[0]["comparison_eligible"] == "False"
    assert float(rows[0]["async_runtime_completed_samples_per_sec"]) > 0

def test_runtime_only_option_skips_quality_even_with_evaluator(tmp_path):
    results_path = tmp_path / "results.csv"
    args = _args(results_path)
    args.async_pass = "runtime-only"
    runtime = Runtime()
    evaluator = Evaluator()
    decoder = Decoder()

    assert _execute(results_path, runtime, evaluator, decoder, args) == 0

    assert runtime.calls == 3
    assert runtime.unloads == 1
    assert evaluator.samples == 0
    assert decoder.calls == 0
    with results_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["quality_status"] == "skipped"
    assert row["comparison_eligible"] == "False"
    assert row["async_run_status"] == "valid"
    assert row["async_runtime_completed_samples"] == "3"
    assert row["quality_evaluator_samples"] == ""
    assert float(row["async_runtime_completed_samples_per_sec"]) > 0
    details = json.loads((tmp_path.parent / row["details_path"]).read_text())
    assert details["quality_phase"]["status"] == "skipped"
    assert details["quality_phase"]["reason"] == "requested_runtime_only"
    assert details["performance_phase"]["completed_samples"] == 3


def test_runtime_only_config_failure_keeps_quality_skipped(tmp_path):
    class InvalidMetadataLoader(Loader):
        def get_metadata(self):
            return {"total_samples": 0, "is_static_batched": False}

    results_path = tmp_path / "results.csv"
    args = _args(results_path)
    args.async_pass = "runtime-only"
    args.scenario = "server_like"
    args.target_qps = 100
    args.min_duration_sec = 0
    runtime = Runtime()
    target = get_target("cpu")

    with pytest.raises(ValueError, match="total_samples"):
        benchmark_main.execute_benchmark(
            args,
            target=target,
            loader=InvalidMetadataLoader(),
            runtime=runtime,
            evaluator=Evaluator(),
            decoder=Decoder(),
            hw_monitor=None,
            task_name="IMAGE_CLASSIFICATION",
            target_meta=benchmark_main.target_metadata(target, {}),
            results_path=results_path,
        )

    assert runtime.calls == 0
    with results_path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["quality_status"] == "skipped"
    assert row["comparison_eligible"] == "False"
    assert not row.get("async_runtime_completed_samples_per_sec")
    details = json.loads((tmp_path.parent / row["details_path"]).read_text())
    assert details["quality_phase"]["status"] == "skipped"
    assert details["quality_phase"]["reason"] == "requested_runtime_only"


def test_quality_sample_count_mismatch_prevents_performance_pass(tmp_path):
    class WrongCountEvaluator(Evaluator):
        def compute(self):
            return {"accuracy": 1.0, "Total Samples": self.samples - 1}

    results_path = tmp_path / "results.csv"
    runtime = Runtime()
    with pytest.raises(benchmark_main.AccuracyPhaseFailed):
        _execute(results_path, runtime, WrongCountEvaluator(), Decoder())

    assert runtime.calls == 3
    with results_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["async_invalid_reasons"] == "accuracy_failed"
    assert rows[0]["quality_evaluator_samples"] == "2"
    assert rows[0]["comparison_eligible"] == "False"
    assert not rows[0].get("async_runtime_completed_samples_per_sec")


def test_server_like_uses_same_fixed_sample_count_in_both_passes(tmp_path):
    results_path = tmp_path / "results.csv"
    args = _args(results_path)
    args.scenario = "server_like"
    args.target_qps = 1_000
    args.min_duration_sec = 0.0
    runtime = Runtime()
    status = _execute(results_path, runtime, Evaluator(), Decoder(), args)

    assert status == 0
    assert runtime.calls == 6
    with results_path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["scenario"] == "server_like"
    assert row["quality_evaluator_samples"] == "3"
    assert row["async_runtime_completed_samples"] == "3"
    assert row["comparison_eligible"] == "True"


def test_max_samples_above_dataset_size_uses_actual_sample_count(tmp_path):
    results_path = tmp_path / "results.csv"
    args = _args(results_path)
    args.max_samples = 5
    runtime = Runtime()
    status = _execute(results_path, runtime, Evaluator(), Decoder(), args)

    assert status == 0
    assert runtime.calls == 6
    with results_path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["quality_evaluator_samples"] == "3"
    assert row["async_runtime_completed_samples"] == "3"
    assert row["comparison_eligible"] == "True"


def test_server_like_caps_quality_and_performance_to_unique_samples(tmp_path):
    results_path = tmp_path / "results.csv"
    args = _args(results_path)
    args.scenario = "server_like"
    args.target_qps = 1_000
    args.min_duration_sec = 0.0
    args.max_samples = 5
    runtime = Runtime()
    status = _execute(results_path, runtime, Evaluator(), Decoder(), args)

    assert status == 0
    assert runtime.calls == 6
    with results_path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["quality_evaluator_samples"] == "3"
    assert row["async_runtime_completed_samples"] == "3"
    assert row["comparison_eligible"] == "True"


def test_server_like_quality_never_repeats_an_image_id(tmp_path):
    class UniqueEvaluator(Evaluator):
        def __init__(self):
            super().__init__()
            self.seen_labels = set()

        def add_batch(self, outputs, labels, timing_ms):
            for label in labels:
                value = float(label)
                assert value not in self.seen_labels
                self.seen_labels.add(value)
            super().add_batch(outputs, labels, timing_ms)

    results_path = tmp_path / "results.csv"
    args = _args(results_path)
    args.scenario = "server_like"
    args.target_qps = 1_000
    args.min_duration_sec = 0.0
    args.max_samples = 5
    evaluator = UniqueEvaluator()

    assert _execute(results_path, Runtime(), evaluator, Decoder(), args) == 0
    assert evaluator.seen_labels == {0.0, 2.0, 4.0}


def test_server_like_planner_is_bounded_when_intervals_round_to_zero():
    config = benchmark_main.AsyncInferenceConfig(
        scenario=benchmark_main.AsyncScenario.SERVER_LIKE,
        target_qps=1e20,
        min_samples=1,
        min_duration_sec=10.0,
    )

    planned = benchmark_main._async_performance_config(config, Loader())

    assert planned.max_samples == 3


def test_server_like_without_max_samples_plans_equal_pass_counts(tmp_path):
    results_path = tmp_path / "results.csv"
    args = _args(results_path)
    args.scenario = "server_like"
    args.target_qps = 500
    args.min_duration_sec = 0.005
    args.max_samples = None
    runtime = Runtime()
    status = _execute(results_path, runtime, Evaluator(), Decoder(), args)

    assert status == 0
    with results_path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    count = int(row["async_runtime_completed_samples"])
    assert count > 0
    assert runtime.calls == 2 * count
    assert row["quality_evaluator_samples"] == str(count)
    assert row["comparison_eligible"] == "True"


def test_native_callback_error_in_quality_pass_stops_performance(
    tmp_path, monkeypatch
):
    snapshot = {
        "async_native_inflight": 0,
        "async_native_duplicate_callbacks": 1,
        "async_native_late_callbacks": 0,
        "async_native_submit_failures": 0,
        "async_native_timeouts": 0,
    }
    monkeypatch.setattr(
        benchmark_main,
        "_safe_native_async_executor_metrics",
        lambda executor: snapshot,
    )
    results_path = tmp_path / "results.csv"
    runtime = Runtime()
    with pytest.raises(benchmark_main.NativeAsyncPhaseFailed):
        _execute(results_path, runtime, Evaluator(), Decoder())

    assert runtime.calls == 3
    with results_path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["async_invalid_reasons"] == "native_async_failed"
    assert row["quality_status"] == "failed"
    assert row["comparison_eligible"] == "False"
    assert not row.get("async_runtime_completed_samples_per_sec")
