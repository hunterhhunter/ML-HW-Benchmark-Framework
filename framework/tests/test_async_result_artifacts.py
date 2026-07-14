import csv
import json
import multiprocessing
import re
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

import core.async_inference.trace as trace_module
import core.result_store as result_store_module
from core.async_inference.trace import RequestTraceWriter
from core.async_inference.types import RequestTrace, TerminalStatus
from core.result_store import (
    create_run_id,
    load_results,
    save_async_details,
    save_result,
)
from core.result_store import delete_result


def make_trace(**changes):
    values = {
        "request_id": 1,
        "sample_index": 2,
        "status": TerminalStatus.COMPLETED,
        "scheduled_ns": 1,
        "issued_ns": 2,
        "enqueued_ns": 3,
        "runtime_started_ns": 4,
        "runtime_finished_ns": 5,
        "completed_ns": 6,
        "worker_id": 0,
        "batch_size": 1,
        "timed_out": False,
        "sample_count": 3,
        "error_type": None,
        "error_message": None,
    }
    values.update(changes)
    return RequestTrace(**values)


def save_minimal_result(csv_path, **changes):
    values = {
        "metrics": {"accuracy": 1.0},
        "model_name": "tiny",
        "task": "IMAGE_CLASSIFICATION",
        "backend": "onnxruntime",
        "device": "cpu",
        "batch_size": 1,
        "warmup_runs": 0,
        "results_path": csv_path,
    }
    values.update(changes)
    return save_result(**values)


def save_result_process(csv_path, prefix, start, count):
    assert start.wait(5.0)
    for index in range(count):
        save_minimal_result(
            Path(csv_path),
            run_id=f"{prefix}{index:04d}",
            model_name=f"{prefix}-{index}",
        )


def save_duplicate_result_process(csv_path, start, outcomes):
    assert start.wait(5.0)
    try:
        save_minimal_result(Path(csv_path), run_id="shared123")
    except BaseException as exc:
        outcomes.put(("error", type(exc).__name__))
    else:
        outcomes.put(("ok", "shared123"))


def save_sidecar_process(results_dir, start, generation, outcomes):
    assert start.wait(5.0)
    try:
        save_async_details(
            "shared123",
            {"generation": generation},
            results_dir=Path(results_dir),
        )
    except BaseException as exc:
        outcomes.put(("error", type(exc).__name__))
    else:
        outcomes.put(("ok", generation))


def test_create_run_id_has_stable_path_safe_shape():
    run_ids = {create_run_id() for _ in range(32)}

    assert len(run_ids) == 32
    assert all(re.fullmatch(r"[0-9a-f]{8}", run_id) for run_id in run_ids)


def test_save_result_accepts_exact_preallocated_id_and_protects_async_metadata(
    tmp_path,
):
    csv_path = tmp_path / "results.csv"

    run_id = save_minimal_result(
        csv_path,
        run_id="fixed123",
        metrics={
            "accuracy": 1.0,
            "scenario": "metric-collision",
            "async_run_status": "metric-collision",
        },
        inference_mode="async_queue",
        scenario="offline",
        queue_capacity=8,
        worker_count=2,
        batch_timeout_ms=1.5,
        target_qps=None,
        schedule_seed=7,
        async_run_status="valid",
        async_invalid_reasons="",
        details_path="results/details/fixed123.json",
        request_trace_path="results/traces/fixed123.jsonl",
    )

    assert run_id == "fixed123"
    row = load_results(results_path=csv_path)[0]
    assert row["inference_mode"] == "async_queue"
    assert row["scenario"] == "offline"
    assert row["queue_capacity"] == "8"
    assert row["target_qps"] == ""
    assert row["async_run_status"] == "valid"
    assert row["request_trace_path"] == "results/traces/fixed123.jsonl"


def test_save_result_rejects_supplied_duplicate_run_id(tmp_path):
    csv_path = tmp_path / "results.csv"
    save_minimal_result(csv_path, run_id="fixed123")

    with pytest.raises(ValueError, match="already exists"):
        save_minimal_result(csv_path, run_id="fixed123", model_name="duplicate")

    rows = load_results(results_path=csv_path)
    assert [row["run_id"] for row in rows] == ["fixed123"]


def test_save_result_retries_generated_run_id_collision(tmp_path, monkeypatch):
    csv_path = tmp_path / "results.csv"
    save_minimal_result(csv_path, run_id="fixed123")
    generated = iter(("fixed123", "fresh123"))
    monkeypatch.setattr(result_store_module, "create_run_id", lambda: next(generated))

    run_id = save_minimal_result(csv_path)

    assert run_id == "fresh123"
    assert {row["run_id"] for row in load_results(results_path=csv_path)} == {
        "fixed123",
        "fresh123",
    }


def test_save_result_preserves_legacy_duplicate_ids_but_adds_no_duplicate(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "run_id,model_name,accuracy\n"
        "legacy01,first,0.1\n"
        "legacy01,second,0.2\n",
        encoding="utf-8",
    )

    save_minimal_result(csv_path, run_id="fresh123")

    rows = list(reversed(load_results(results_path=csv_path)))
    assert [row["run_id"] for row in rows] == [
        "legacy01",
        "legacy01",
        "fresh123",
    ]


def test_interprocess_supplied_run_id_has_exactly_one_winner(tmp_path):
    csv_path = tmp_path / "results.csv"
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    outcomes = context.Queue()
    processes = [
        context.Process(
            target=save_duplicate_result_process,
            args=(str(csv_path), start, outcomes),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(10.0)
        assert process.exitcode == 0

    results = sorted(outcomes.get(timeout=1.0) for _ in processes)
    assert results == [("error", "ValueError"), ("ok", "shared123")]
    assert [row["run_id"] for row in load_results(results_path=csv_path)] == [
        "shared123"
    ]


def test_save_result_migrates_legacy_header_without_reordering_data(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "run_id,model_name,legacy_metric,accuracy\n"
        "old00001,first,keep-a,0.1\n"
        "old00002,second,keep-b,0.2\n",
        encoding="utf-8",
    )

    save_minimal_result(
        csv_path,
        inference_mode="async_queue",
        scenario="offline",
        metrics={"accuracy": 1.0, "new_metric": 2.0},
    )

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert reader.fieldnames[:4] == [
        "run_id",
        "model_name",
        "legacy_metric",
        "accuracy",
    ]
    assert reader.fieldnames[-1] == "new_metric"
    assert [(row["model_name"], row["legacy_metric"]) for row in rows[:2]] == [
        ("first", "keep-a"),
        ("second", "keep-b"),
    ]
    assert rows[0]["inference_mode"] == ""
    assert rows[2]["scenario"] == "offline"


@pytest.mark.parametrize(
    "content",
    [
        "run_id,model_name,model_name\nold0001,first,second\n",
        "run_id,,accuracy\nold0001,tiny,1.0\n",
        "run_id,model_name,accuracy\nold0001,tiny,1.0,extra\n",
        "run_id,model_name,accuracy\nold0001,tiny\n",
        'run_id,model_name,accuracy\nold0001,"unterminated,1.0\n',
    ],
)
def test_save_result_rejects_malformed_legacy_csv_without_mutation(
    tmp_path,
    content,
):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(content, encoding="utf-8")
    original = csv_path.read_bytes()

    with pytest.raises(ValueError, match="CSV"):
        save_minimal_result(csv_path, run_id="fresh123")

    assert csv_path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_delete_and_load_reject_malformed_csv_without_mutation(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "run_id,model_name,accuracy\nold0001,tiny,1.0,extra\n",
        encoding="utf-8",
    )
    original = csv_path.read_bytes()

    with pytest.raises(ValueError, match="CSV"):
        load_results(results_path=csv_path)
    with pytest.raises(ValueError, match="CSV"):
        delete_result("old0001", results_path=csv_path)

    assert csv_path.read_bytes() == original


def test_valid_positional_csv_migration_preserves_every_legacy_cell(tmp_path):
    csv_path = tmp_path / "results.csv"
    header = ["run_id", "model_name", "notes", "accuracy"]
    legacy_rows = [
        ["old0001", "tiny,one", "first line\nsecond line", "1.0"],
        ["old0002", "", "", "0.5"],
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(legacy_rows)

    save_minimal_result(
        csv_path,
        run_id="fresh123",
        metrics={"accuracy": 0.9, "new_metric": 7},
    )

    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, strict=True))
    migrated_header = rows[0]
    assert migrated_header[: len(header)] == header
    for migrated, original in zip(rows[1:3], legacy_rows):
        assert migrated[: len(header)] == original
        assert migrated[len(header) :] == [""] * (
            len(migrated_header) - len(header)
        )


def test_save_result_preserves_legacy_positional_results_path(tmp_path):
    csv_path = tmp_path / "positional.csv"

    run_id = save_result(
        {"accuracy": 1.0},
        "tiny",
        "IMAGE_CLASSIFICATION",
        "onnxruntime",
        "cpu",
        1,
        0,
        None,
        "target",
        "vendor",
        "accelerator",
        "runtime",
        "compiler",
        "onnx",
        csv_path,
    )

    result = load_results(results_path=csv_path)[0]
    assert result["run_id"] == run_id
    assert result["inference_mode"] == "e2e"


@pytest.mark.parametrize(
    "run_id",
    ["", ".", "..", "../escape", "a/b", r"a\b", "/absolute", "bad\nline", "한글"],
)
def test_artifact_apis_reject_unsafe_run_ids_without_creating_files(
    tmp_path,
    run_id,
):
    csv_path = tmp_path / "results.csv"
    with pytest.raises(ValueError, match="run_id"):
        save_minimal_result(csv_path, run_id=run_id)
    with pytest.raises(ValueError, match="run_id"):
        save_async_details(run_id, {}, results_dir=tmp_path)

    assert not csv_path.exists()
    assert not (tmp_path / "details").exists()


def test_save_result_validates_run_id_before_directory_or_lock_creation(tmp_path):
    csv_path = tmp_path / "missing" / "results.csv"

    with pytest.raises(ValueError, match="run_id"):
        save_minimal_result(csv_path, run_id="../escape")

    assert not csv_path.parent.exists()
    assert not csv_path.with_suffix(".csv.lock").exists()


def test_csv_migration_replace_failure_preserves_legacy_bytes(
    tmp_path,
    monkeypatch,
):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "run_id,model_name,accuracy\nold00001,tiny,1.0\n",
        encoding="utf-8",
    )
    original = csv_path.read_bytes()

    def fail_replace(source, target):
        raise OSError("replace failed")

    monkeypatch.setattr(result_store_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        save_minimal_result(csv_path, scenario="offline")

    assert csv_path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_csv_rewrite_preserves_existing_file_mode(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "run_id,model_name,accuracy\nold00001,tiny,1.0\n",
        encoding="utf-8",
    )
    csv_path.chmod(0o640)

    save_minimal_result(csv_path, run_id="fresh123", scenario="offline")

    assert stat.S_IMODE(csv_path.stat().st_mode) == 0o640


def test_csv_delete_rewrite_preserves_existing_file_mode(tmp_path):
    csv_path = tmp_path / "results.csv"
    save_minimal_result(csv_path, run_id="first123")
    save_minimal_result(csv_path, run_id="second123")
    csv_path.chmod(0o640)

    assert delete_result("first123", results_path=csv_path) is True

    assert stat.S_IMODE(csv_path.stat().st_mode) == 0o640


def test_new_csv_has_sensible_default_mode(tmp_path):
    csv_path = tmp_path / "results.csv"

    save_minimal_result(csv_path, run_id="fresh123")

    assert stat.S_IMODE(csv_path.stat().st_mode) == 0o644


def test_csv_mode_failure_preserves_original_and_removes_temp(
    tmp_path,
    monkeypatch,
):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "run_id,model_name,accuracy\nold00001,tiny,1.0\n",
        encoding="utf-8",
    )
    csv_path.chmod(0o640)
    original = csv_path.read_bytes()

    monkeypatch.setattr(
        result_store_module.os,
        "fchmod",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("chmod failed")),
    )
    with pytest.raises(OSError, match="chmod failed"):
        save_minimal_result(csv_path, run_id="fresh123", scenario="offline")

    assert csv_path.read_bytes() == original
    assert stat.S_IMODE(csv_path.stat().st_mode) == 0o640
    assert not list(tmp_path.glob("*.tmp"))


def test_csv_primary_replace_error_survives_cleanup_error(
    tmp_path,
    monkeypatch,
):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "run_id,model_name,accuracy\nold00001,tiny,1.0\n",
        encoding="utf-8",
    )
    original = csv_path.read_bytes()

    monkeypatch.setattr(
        result_store_module.os,
        "replace",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("replace primary")),
    )
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cleanup secondary")),
    )
    with pytest.raises(OSError, match="replace primary") as raised:
        save_minimal_result(csv_path, run_id="fresh123")

    assert csv_path.read_bytes() == original
    assert raised.value.persistence_secondary_errors == [
        {
            "phase": "cleanup_temp",
            "error_type": "OSError",
            "error_message": "cleanup secondary",
            "temporary_file_may_remain": True,
        }
    ]
    assert list(tmp_path.glob("*.tmp"))


def test_csv_directory_fsync_error_survives_directory_close_error(
    tmp_path,
    monkeypatch,
):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "run_id,model_name,accuracy\nold00001,tiny,1.0\n",
        encoding="utf-8",
    )
    fsync_calls = 0
    real_fsync = result_store_module.os.fsync
    real_close = result_store_module.os.close

    def fail_directory_fsync(file_descriptor):
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("directory fsync primary")
        return real_fsync(file_descriptor)

    def fail_directory_close(file_descriptor):
        real_close(file_descriptor)
        raise OSError("directory close secondary")

    monkeypatch.setattr(result_store_module.os, "fsync", fail_directory_fsync)
    monkeypatch.setattr(result_store_module.os, "close", fail_directory_close)
    with pytest.raises(OSError, match="directory fsync primary") as raised:
        save_minimal_result(csv_path, run_id="fresh123", scenario="offline")

    assert raised.value.persistence_secondary_errors == [
        {
            "phase": "close_parent_directory",
            "error_type": "OSError",
            "error_message": "directory close secondary",
        }
    ]
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, strict=True))
    assert len(rows) == 3
    assert not list(tmp_path.glob("*.tmp"))


def test_interprocess_result_writes_preserve_every_row(tmp_path):
    csv_path = tmp_path / "results.csv"
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    processes = [
        context.Process(
            target=save_result_process,
            args=(str(csv_path), prefix, start, 12),
        )
        for prefix in ("aa", "bb", "cc")
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(10.0)
        assert process.exitcode == 0

    rows = load_results(results_path=csv_path)
    assert len(rows) == 36
    assert len({row["run_id"] for row in rows}) == 36


class DetailStatus(Enum):
    READY = "ready"


def test_save_async_details_is_deterministic_strict_json(tmp_path):
    path = save_async_details(
        "fixed123",
        {
            "run_id": "cannot-overwrite",
            "schema_version": "cannot-overwrite",
            "quality_metrics": {"accuracy": np.float32(1.0)},
            "array": np.asarray([1, 2], dtype=np.int64),
            "path": Path("models/tiny.onnx"),
            "state": DetailStatus.READY,
            "nested": (np.int64(3), {"x", "y"}),
        },
        results_dir=tmp_path,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path == tmp_path / "details" / "fixed123.json"
    assert payload["schema_version"] == "1.0"
    assert payload["run_id"] == "fixed123"
    assert payload["quality_metrics"]["accuracy"] == 1.0
    assert payload["array"] == [1, 2]
    assert payload["path"] == "models/tiny.onnx"
    assert payload["state"] == "ready"
    assert payload["nested"] == [3, ["x", "y"]]
    assert not list(path.parent.glob("*.tmp"))


def test_save_async_details_repeated_output_is_byte_deterministic(tmp_path):
    details = {"z": {"beta", "alpha"}, "a": np.float64(1.25)}

    first_path = save_async_details(
        "fixed123",
        details,
        results_dir=tmp_path / "first",
    )
    second_path = save_async_details(
        "fixed123",
        details,
        results_dir=tmp_path / "second",
    )

    assert first_path.read_bytes() == second_path.read_bytes()


def test_save_async_details_preserves_task7_result_sections(tmp_path):
    details = {
        "config": {"scenario": "offline", "queue_capacity": 8},
        "measurement_duration_sec": 1.25,
        "invalid_reasons": [],
        "warnings": ["tail_percentile_low_sample_count"],
        "counter_invariants": {"valid": True},
        "timing_ms": {"queue_wait": {"p99": 2.0}},
        "queue": {"depth_max": 4},
        "workers": {"utilization": 0.5},
        "batch_size": {"max": 2},
        "failure_types": {},
        "quality_metrics": {"accuracy": 1.0},
        "hardware_metrics": {"hw_power_watts": 12.5},
        "status": "valid",
    }

    path = save_async_details("fixed123", details, results_dir=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert all(section in payload for section in details)
    assert payload["config"]["scenario"] == "offline"
    assert payload["timing_ms"]["queue_wait"]["p99"] == 2.0
    assert payload["hardware_metrics"]["hw_power_watts"] == 12.5


class HostileDetail:
    def __iter__(self):
        raise AssertionError("must not iterate")

    def item(self):
        raise AssertionError("must not call item")

    def __str__(self):
        raise AssertionError("must not stringify")


def test_save_async_details_rejects_unsupported_objects_without_callbacks(tmp_path):
    with pytest.raises(TypeError, match="HostileDetail"):
        save_async_details(
            "fixed123",
            {"hostile": HostileDetail()},
            results_dir=tmp_path,
        )

    assert not (tmp_path / "details" / "fixed123.json").exists()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_save_async_details_rejects_nonfinite_values(tmp_path, bad_value):
    with pytest.raises(ValueError, match="non-finite"):
        save_async_details(
            "fixed123",
            {"bad": bad_value},
            results_dir=tmp_path,
        )

    assert not (tmp_path / "details" / "fixed123.json").exists()


def test_save_async_details_rejects_cycles(tmp_path):
    cycle = []
    cycle.append(cycle)

    with pytest.raises(ValueError, match="cycles"):
        save_async_details(
            "fixed123",
            {"cycle": cycle},
            results_dir=tmp_path,
        )

    assert not (tmp_path / "details" / "fixed123.json").exists()


def test_save_async_details_rejects_object_array_cycles(tmp_path):
    cycle = np.empty(1, dtype=object)
    cycle[0] = cycle

    with pytest.raises(ValueError, match="cycles"):
        save_async_details(
            "fixed123",
            {"cycle": cycle},
            results_dir=tmp_path,
        )

    assert not (tmp_path / "details" / "fixed123.json").exists()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (list(range(10_001)), "item budget"),
        (np.arange(4_097), "array budget"),
        ("x" * 1_000_001, "string budget"),
    ],
)
def test_save_async_details_rejects_values_over_normalization_budgets(
    tmp_path,
    value,
    message,
):
    with pytest.raises(ValueError, match=message):
        save_async_details(
            "fixed123",
            {"value": value},
            results_dir=tmp_path,
        )

    assert not (tmp_path / "details").exists()


def test_save_async_details_rejects_depth_over_budget(tmp_path):
    value = "leaf"
    for _ in range(34):
        value = [value]

    with pytest.raises(ValueError, match="depth budget"):
        save_async_details(
            "fixed123",
            {"value": value},
            results_dir=tmp_path,
        )

    assert not (tmp_path / "details").exists()


def test_save_async_details_rejects_enum_identity_cycle_without_recursion(tmp_path):
    class CyclicStatus(Enum):
        READY = "ready"

    object.__setattr__(CyclicStatus.READY, "_value_", CyclicStatus.READY)

    with pytest.raises(ValueError, match="cycle"):
        save_async_details(
            "fixed123",
            {"status": CyclicStatus.READY},
            results_dir=tmp_path,
        )

    assert not (tmp_path / "details").exists()


def test_save_async_details_publish_failure_keeps_previous_complete_file(
    tmp_path,
    monkeypatch,
):
    path = save_async_details(
        "fixed123",
        {"generation": 1},
        results_dir=tmp_path,
    )
    original = path.read_bytes()

    def fail_publish(*args, **kwargs):
        raise OSError("publish failed")

    monkeypatch.setattr(result_store_module.os, "link", fail_publish)
    with pytest.raises(OSError, match="publish failed"):
        save_async_details(
            "fixed123",
            {"generation": 2},
            results_dir=tmp_path,
        )

    assert path.read_bytes() == original
    assert not list(path.parent.glob("*.tmp"))


def test_sidecar_primary_publish_error_survives_cleanup_error(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        result_store_module.os,
        "link",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("publish primary")),
    )
    monkeypatch.setattr(
        result_store_module.os,
        "unlink",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cleanup secondary")),
    )

    with pytest.raises(OSError, match="publish primary") as raised:
        save_async_details("fixed123", {"value": 1}, results_dir=tmp_path)

    errors = raised.value.persistence_secondary_errors
    assert len(errors) == 1
    assert errors[0]["phase"] == "cleanup_temp"
    assert errors[0]["error_type"] == "OSError"
    assert errors[0]["error_message"] == "cleanup secondary"
    assert errors[0]["temporary_file_may_remain"] is True
    assert errors[0]["temporary_path"].endswith(".tmp")
    assert list((tmp_path / "details").glob("*.tmp"))


def test_save_async_details_fsync_failure_never_publishes_partial_file(
    tmp_path,
    monkeypatch,
):
    def fail_fsync(file_descriptor):
        raise OSError("fsync failed")

    monkeypatch.setattr(result_store_module.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="fsync failed"):
        save_async_details(
            "fixed123",
            {"generation": 1},
            results_dir=tmp_path,
        )

    details_dir = tmp_path / "details"
    assert not (details_dir / "fixed123.json").exists()
    assert not list(details_dir.glob("*.tmp"))


def test_save_async_details_rejects_symlinked_details_directory(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "details").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError):
        save_async_details("fixed123", {"value": 1}, results_dir=root)

    assert list(outside.iterdir()) == []


def test_save_async_details_rejects_symlinked_results_root(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError):
        save_async_details("fixed123", {"value": 1}, results_dir=root)

    assert list(outside.iterdir()) == []


def test_save_async_details_does_not_follow_existing_final_symlink(tmp_path):
    details = tmp_path / "details"
    details.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("sentinel", encoding="utf-8")
    final_path = details / "fixed123.json"
    final_path.symlink_to(outside)

    with pytest.raises(FileExistsError):
        save_async_details("fixed123", {"value": 1}, results_dir=tmp_path)

    assert final_path.is_symlink()
    assert outside.read_text(encoding="utf-8") == "sentinel"


def test_save_async_details_detects_details_symlink_swap_at_publish(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "root"
    details = root / "details"
    outside = tmp_path / "outside"
    details.mkdir(parents=True)
    outside.mkdir()
    real_link = result_store_module.os.link
    swapped = False

    def swap_then_link(source, target, *args, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            details.rename(root / "relocated")
            details.symlink_to(outside, target_is_directory=True)
        return real_link(source, target, *args, **kwargs)

    monkeypatch.setattr(result_store_module.os, "link", swap_then_link)
    with pytest.raises(OSError, match="changed during publication"):
        save_async_details("fixed123", {"value": 1}, results_dir=root)

    assert list(outside.iterdir()) == []
    assert not (root / "relocated" / "fixed123.json").exists()


def test_concurrent_sidecar_writers_have_exactly_one_winner(tmp_path):
    barrier = threading.Barrier(3)

    def write_sidecar(generation):
        barrier.wait()
        return save_async_details(
            "fixed123",
            {"generation": generation, "values": list(range(100))},
            results_dir=tmp_path,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(write_sidecar, value) for value in (1, 2)]
        barrier.wait()
        outcomes = []
        for future in futures:
            try:
                outcomes.append(("ok", future.result(timeout=5.0)))
            except FileExistsError:
                outcomes.append(("error", None))

    assert sorted(outcome for outcome, _ in outcomes) == ["error", "ok"]
    path = tmp_path / "details" / "fixed123.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["generation"] in (1, 2)
    assert payload["values"] == list(range(100))
    assert payload["run_id"] == "fixed123"
    assert not list(path.parent.glob("*.tmp"))


def test_multiprocess_sidecar_writers_have_exactly_one_winner(tmp_path):
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    outcomes = context.Queue()
    processes = [
        context.Process(
            target=save_sidecar_process,
            args=(str(tmp_path), start, generation, outcomes),
        )
        for generation in (1, 2)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(10.0)
        assert process.exitcode == 0

    results = sorted(outcomes.get(timeout=1.0) for _ in processes)
    assert results[0] == ("error", "FileExistsError")
    assert results[1][0] == "ok"
    payload = json.loads(
        (tmp_path / "details" / "shared123.json").read_text(encoding="utf-8")
    )
    assert payload["generation"] == results[1][1]


def test_trace_writer_publishes_only_exact_trace_fields(tmp_path):
    path = tmp_path / "trace.jsonl"
    writer = RequestTraceWriter(path, capacity=2)
    writer.start()
    writer.write(
        make_trace(
            status=TerminalStatus.FAILED,
            error_type="RuntimeError",
            error_message="failed",
        )
    )

    assert writer.close(timeout=1.0) is True
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row == {
        "request_id": 1,
        "sample_index": 2,
        "status": "failed",
        "scheduled_ns": 1,
        "issued_ns": 2,
        "enqueued_ns": 3,
        "runtime_started_ns": 4,
        "runtime_finished_ns": 5,
        "completed_ns": 6,
        "worker_id": 0,
        "batch_size": 1,
        "timed_out": False,
        "sample_count": 3,
        "error_type": "RuntimeError",
        "error_message": "failed",
    }
    assert writer.dropped == 0
    assert writer.error is None
    assert not list(tmp_path.glob("*.tmp"))
    assert writer._queue.unfinished_tasks == 0


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_trace_writer_requires_positive_integral_capacity(tmp_path, capacity):
    with pytest.raises(ValueError, match="capacity"):
        RequestTraceWriter(tmp_path / "trace.jsonl", capacity=capacity)


def test_trace_close_timeout_abandons_full_queue_and_cleans_temp(
    tmp_path,
    monkeypatch,
):
    entered = threading.Event()
    release = threading.Event()
    real_dumps = trace_module.json.dumps

    def blocked_dumps(*args, **kwargs):
        entered.set()
        assert release.wait(1.0)
        return real_dumps(*args, **kwargs)

    monkeypatch.setattr(trace_module.json, "dumps", blocked_dumps)
    path = tmp_path / "trace.jsonl"
    writer = RequestTraceWriter(path, capacity=1)
    writer.start()
    writer.write(make_trace(request_id=1))
    assert entered.wait(1.0)
    writer.write(make_trace(request_id=2))

    assert writer.close(timeout=0.0) is False
    release.set()
    writer._thread.join(1.0)

    assert not writer._thread.is_alive()
    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert writer.error["phase"] == "close"


def test_trace_writer_lifecycle_contract_is_explicit_and_close_is_idempotent(
    tmp_path,
):
    writer = RequestTraceWriter(tmp_path / "trace.jsonl")
    with pytest.raises(RuntimeError, match="running state"):
        writer.write(make_trace())
    with pytest.raises(RuntimeError, match="requires start"):
        writer.close(timeout=1.0)

    writer.start()
    with pytest.raises(RuntimeError, match="created state"):
        writer.start()
    writer.write(make_trace())
    assert writer.close(timeout=1.0) is True
    assert writer.close(timeout=0.0) is True
    with pytest.raises(RuntimeError, match="running state"):
        writer.write(make_trace())
    with pytest.raises(RuntimeError, match="created state"):
        writer.start()


def test_trace_writer_rejects_payload_like_objects_and_publishes_no_payload(
    tmp_path,
):
    path = tmp_path / "trace.jsonl"
    writer = RequestTraceWriter(path)
    writer.start()

    with pytest.raises(TypeError, match="only exact RequestTrace"):
        writer.write({"input": [1], "output": [2], "sample": "secret"})
    assert writer.close(timeout=1.0) is True

    assert path.read_text(encoding="utf-8") == ""


def test_trace_writer_saturation_is_nonblocking_and_counts_drops(
    tmp_path,
    monkeypatch,
):
    entered = threading.Event()
    release = threading.Event()
    real_dumps = trace_module.json.dumps

    def blocked_dumps(*args, **kwargs):
        entered.set()
        assert release.wait(1.0)
        return real_dumps(*args, **kwargs)

    monkeypatch.setattr(trace_module.json, "dumps", blocked_dumps)
    path = tmp_path / "trace.jsonl"
    writer = RequestTraceWriter(path, capacity=1)
    writer.start()
    writer.write(make_trace(request_id=1))
    assert entered.wait(1.0)
    writer.write(make_trace(request_id=2))
    writer.write(make_trace(request_id=3))

    assert writer.dropped == 1
    release.set()
    assert writer.close(timeout=1.0) is True
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["request_id"] for row in rows] == [1, 2]
    assert writer._queue.unfinished_tasks == 0


def test_trace_writer_accepts_concurrent_writes_without_accounting_loss(tmp_path):
    path = tmp_path / "trace.jsonl"
    writer = RequestTraceWriter(path, capacity=64)
    writer.start()
    barrier = threading.Barrier(5)

    def write_group(group):
        barrier.wait()
        for offset in range(10):
            writer.write(make_trace(request_id=group * 10 + offset))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(write_group, group) for group in range(4)]
        barrier.wait()
        for future in futures:
            future.result(timeout=5.0)

    assert writer.close(timeout=1.0) is True
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert sorted(row["request_id"] for row in rows) == list(range(40))
    assert writer.dropped == 0
    assert writer._queue.unfinished_tasks == 0


def test_concurrent_trace_writers_have_exactly_one_winner(tmp_path):
    path = tmp_path / "trace.jsonl"
    writers = [RequestTraceWriter(path), RequestTraceWriter(path)]
    for writer in writers:
        writer.start()
        writer.write(make_trace(request_id=writers.index(writer) + 1))

    barrier = threading.Barrier(3)

    def close_writer(writer):
        barrier.wait()
        return writer.close(timeout=1.0)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(close_writer, writer) for writer in writers]
        barrier.wait()
        results = [future.result(timeout=5.0) for future in futures]

    assert sorted(results) == [False, True]
    loser = writers[results.index(False)]
    assert loser.error["error_type"] == "FileExistsError"
    assert json.loads(path.read_text())["request_id"] in (1, 2)


def test_trace_start_rejects_existing_final_target(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text("existing\n", encoding="utf-8")
    writer = RequestTraceWriter(path)

    with pytest.raises(FileExistsError):
        writer.start()

    assert path.read_text(encoding="utf-8") == "existing\n"
    assert writer.error["error_type"] == "FileExistsError"


def test_trace_serialization_failure_is_diagnostic_and_never_published(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "trace.jsonl"

    def fail_dumps(*args, **kwargs):
        raise ValueError("serialization failed")

    monkeypatch.setattr(trace_module.json, "dumps", fail_dumps)
    writer = RequestTraceWriter(path)
    writer.start()
    writer.write(make_trace())

    assert writer.close(timeout=1.0) is False
    assert writer.error == {
        "phase": "serialize",
        "error_type": "ValueError",
        "error_message": "serialization failed",
    }
    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert writer._queue.unfinished_tasks == 0


class FailingWriteHandle:
    def __init__(self, handle):
        self.handle = handle

    def write(self, text):
        raise OSError("write failed")

    def flush(self):
        return self.handle.flush()

    def fileno(self):
        return self.handle.fileno()

    def close(self):
        return self.handle.close()


class FailingWriteAndCloseHandle(FailingWriteHandle):
    def close(self):
        self.handle.close()
        raise OSError("close secondary")


def test_trace_write_failure_is_diagnostic_and_cleans_temp(tmp_path, monkeypatch):
    real_fdopen = trace_module.os.fdopen

    def failing_fdopen(*args, **kwargs):
        return FailingWriteHandle(real_fdopen(*args, **kwargs))

    monkeypatch.setattr(trace_module.os, "fdopen", failing_fdopen)
    path = tmp_path / "trace.jsonl"
    writer = RequestTraceWriter(path)
    writer.start()
    writer.write(make_trace())

    assert writer.close(timeout=1.0) is False
    assert writer.error["phase"] == "write"
    assert writer.error["error_message"] == "write failed"
    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_trace_primary_write_error_retains_secondary_close_error(
    tmp_path,
    monkeypatch,
):
    real_fdopen = trace_module.os.fdopen

    def failing_fdopen(*args, **kwargs):
        return FailingWriteAndCloseHandle(real_fdopen(*args, **kwargs))

    monkeypatch.setattr(trace_module.os, "fdopen", failing_fdopen)
    writer = RequestTraceWriter(tmp_path / "trace.jsonl")
    writer.start()
    writer.write(make_trace())

    assert writer.close(timeout=1.0) is False
    assert writer.error["phase"] == "write"
    assert writer.error["error_message"] == "write failed"
    assert writer.error["secondary_errors"] == [
        {
            "phase": "close_file",
            "error_type": "OSError",
            "error_message": "close secondary",
        }
    ]


def test_trace_primary_publish_error_retains_cleanup_error_and_temp_path(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        trace_module.os,
        "link",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("publish primary")),
    )
    monkeypatch.setattr(
        Path,
        "unlink",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cleanup secondary")),
    )
    writer = RequestTraceWriter(tmp_path / "trace.jsonl")
    writer.start()
    writer.write(make_trace())

    assert writer.close(timeout=1.0) is False
    assert writer.error["phase"] == "publish"
    assert writer.error["error_message"] == "publish primary"
    secondary = writer.error["secondary_errors"]
    assert len(secondary) == 1
    assert secondary[0]["phase"] == "cleanup_temp"
    assert secondary[0]["error_type"] == "OSError"
    assert secondary[0]["error_message"] == "cleanup secondary"
    assert secondary[0]["temporary_file_may_remain"] is True
    assert secondary[0]["temporary_path"].endswith(".tmp")
    assert list(tmp_path.glob("*.tmp"))


def test_trace_fdopen_failure_closes_descriptor_and_cleans_temp(
    tmp_path,
    monkeypatch,
):
    descriptors = []

    def fail_fdopen(file_descriptor, *args, **kwargs):
        descriptors.append(file_descriptor)
        raise OSError("fdopen failed")

    monkeypatch.setattr(trace_module.os, "fdopen", fail_fdopen)
    path = tmp_path / "trace.jsonl"
    writer = RequestTraceWriter(path)
    writer.start()

    assert writer.close(timeout=1.0) is False
    assert writer.error["phase"] == "open"
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        trace_module.os.fstat(descriptors[0])
    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    ("operation", "expected_phase"),
    [("fsync", "flush"), ("link", "publish")],
)
def test_trace_finalize_failure_is_diagnostic_and_cleans_temp(
    tmp_path,
    monkeypatch,
    operation,
    expected_phase,
):
    def fail(*args, **kwargs):
        raise OSError(f"{operation} failed")

    monkeypatch.setattr(trace_module.os, operation, fail)
    path = tmp_path / "trace.jsonl"
    writer = RequestTraceWriter(path)
    writer.start()
    writer.write(make_trace())

    assert writer.close(timeout=1.0) is False
    assert writer.error["phase"] == expected_phase
    assert writer.error["error_message"] == f"{operation} failed"
    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_trace_directory_fsync_failure_reports_complete_atomic_file(
    tmp_path,
    monkeypatch,
):
    calls = 0
    real_fsync = trace_module.os.fsync

    def fail_directory_fsync(file_descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync failed")
        return real_fsync(file_descriptor)

    monkeypatch.setattr(trace_module.os, "fsync", fail_directory_fsync)
    path = tmp_path / "trace.jsonl"
    writer = RequestTraceWriter(path)
    writer.start()
    writer.write(make_trace())

    assert writer.close(timeout=1.0) is False
    assert writer.error["phase"] == "directory_fsync"
    assert json.loads(path.read_text())["request_id"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_trace_directory_fsync_error_retains_directory_close_error(
    tmp_path,
    monkeypatch,
):
    fsync_calls = 0
    real_fsync = trace_module.os.fsync
    real_close = trace_module.os.close

    def fail_directory_fsync(file_descriptor):
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("directory fsync primary")
        return real_fsync(file_descriptor)

    def fail_directory_close(file_descriptor):
        real_close(file_descriptor)
        raise OSError("directory close secondary")

    monkeypatch.setattr(trace_module.os, "fsync", fail_directory_fsync)
    monkeypatch.setattr(trace_module.os, "close", fail_directory_close)
    path = tmp_path / "trace.jsonl"
    writer = RequestTraceWriter(path)
    writer.start()
    writer.write(make_trace())

    assert writer.close(timeout=1.0) is False
    assert writer.error["phase"] == "directory_fsync"
    assert writer.error["error_message"] == "directory fsync primary"
    assert writer.error["secondary_errors"] == [
        {
            "phase": "close_directory",
            "error_type": "OSError",
            "error_message": "directory close secondary",
        }
    ]
    assert json.loads(path.read_text())["request_id"] == 1


def test_trace_close_uses_one_remaining_absolute_deadline(tmp_path, monkeypatch):
    class ControlledThread:
        def __init__(self, **kwargs):
            self.join_timeouts = []
            self.alive = False

        def start(self):
            self.alive = True

        def join(self, timeout):
            self.join_timeouts.append(timeout)

        def is_alive(self):
            return self.alive

    ticks = iter((100.0, 100.25))
    monkeypatch.setattr(trace_module.threading, "Thread", ControlledThread)
    monkeypatch.setattr(trace_module.time, "monotonic", lambda: next(ticks))
    writer = RequestTraceWriter(tmp_path / "trace.jsonl")
    writer.start()

    assert writer.close(timeout=1.0) is False
    assert writer._thread.join_timeouts == [0.75]


def test_delete_result_rewrite_failure_preserves_original_file(
    tmp_path,
    monkeypatch,
):
    csv_path = tmp_path / "results.csv"
    run_id = save_minimal_result(csv_path, run_id="fixed123")
    original = csv_path.read_bytes()

    def fail_replace(source, target):
        raise OSError("replace failed")

    monkeypatch.setattr(result_store_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        delete_result(run_id, results_path=csv_path)

    assert csv_path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
