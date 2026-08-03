import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from tools.mobilint_compile_recipes.attempt import (
    STAGES,
    create_attempt,
    execute_stage,
    record_artifact,
    record_quality_csv,
    record_quality_failure,
)


def _result(root):
    return json.loads((root / "result.json").read_text(encoding="utf-8"))


def _execute_stage_in_child_process(root, stage, command):
    execute_stage(root, stage, command)


def test_create_attempt_initializes_immutable_stage_results(tmp_path):
    root = create_attempt(tmp_path, "fixed", "resnet50", "default", {"run": 1})

    result = _result(root)

    assert root == tmp_path / "fixed" / "resnet50" / "default"
    assert result["attempt_id"] == "fixed"
    assert result["metadata"] == {"run": 1}
    assert tuple(result["stages"]) == STAGES
    assert {value["status"] for value in result["stages"].values()} == {"not_run"}
    assert result["compile_status"] == "not_run"
    assert result["runtime_status"] == "not_run"
    assert result["contract_status"] == "not_run"
    assert result["quality_status"] == "not_run"


def test_create_attempt_rejects_duplicate_or_traversal_roots(tmp_path):
    create_attempt(tmp_path, "fixed", "resnet50", "default", {})

    with pytest.raises(FileExistsError, match="already exists"):
        create_attempt(tmp_path, "fixed", "resnet50", "default", {})
    with pytest.raises(ValueError, match="path segment"):
        create_attempt(tmp_path, "../escape", "resnet50", "default", {})


def test_execute_stage_records_output_time_and_success(tmp_path, capsys):
    root = create_attempt(tmp_path, "fixed", "resnet50", "default", {})

    code = execute_stage(
        root, "SOURCE_SMOKE", [sys.executable, "-c", "print('SOURCE_OK')"]
    )

    result = _result(root)
    assert code == 0
    assert result["stages"]["SOURCE_SMOKE"]["status"] == "pass"
    assert result["stages"]["SOURCE_SMOKE"]["elapsed_seconds"] >= 0
    assert "SOURCE_OK" in (root / "compile.log").read_text(encoding="utf-8")
    assert "SOURCE_OK" in capsys.readouterr().out


def test_execute_stage_preserves_first_failure(tmp_path):
    root = create_attempt(tmp_path, "failed", "patchtst-etth1", "stock", {})

    code = execute_stage(
        root,
        "MBLT_COMPILE",
        [sys.executable, "-c", "import sys; print('bad op'); sys.exit(7)"],
    )

    result = _result(root)
    assert code == 7
    assert result["failed_at"] == "MBLT_COMPILE"
    assert result["stages"]["MBLT_COMPILE"]["status"] == "fail"
    assert result["stages"]["MBLT_COMPILE"]["exit_code"] == 7
    assert result["stages"]["MXQ_COMPILE"]["status"] == "not_run"


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX signal exit codes")
def test_execute_stage_records_signal_termination(tmp_path):
    root = create_attempt(tmp_path, "signal", "resnet50", "default", {})

    code = execute_stage(
        root,
        "SOURCE_SMOKE",
        [
            sys.executable,
            "-c",
            "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
        ],
    )

    stage = _result(root)["stages"]["SOURCE_SMOKE"]
    assert code == -signal.SIGTERM
    assert stage["status"] == "fail"
    assert stage["exit_code"] == -signal.SIGTERM
    assert stage["signal"] == signal.SIGTERM


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX flock semantics")
def test_concurrent_stage_writers_preserve_both_stage_records(tmp_path):
    root = create_attempt(tmp_path, "concurrent", "resnet50", "default", {})
    marker = root / "first-started"
    first = [
        sys.executable,
        "-c",
        "from pathlib import Path; import time; "
        f"Path({str(marker)!r}).write_text('started'); "
        "print('FIRST_START', flush=True); time.sleep(0.25); "
        "print('FIRST_DONE', flush=True)",
    ]
    second = [sys.executable, "-c", "print('SECOND_START', flush=True)"]
    context = multiprocessing.get_context("fork")
    first_writer = context.Process(
        target=_execute_stage_in_child_process,
        args=(root, "SOURCE_SMOKE", first),
    )
    second_writer = context.Process(
        target=_execute_stage_in_child_process,
        args=(root, "CALIBRATION_PREPARE", second),
    )

    first_writer.start()
    deadline = time.monotonic() + 3
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert marker.exists()
    second_writer.start()
    first_writer.join(timeout=5)
    second_writer.join(timeout=5)

    assert first_writer.exitcode == 0
    assert second_writer.exitcode == 0
    result = _result(root)
    assert result["stages"]["SOURCE_SMOKE"]["status"] == "pass"
    assert result["stages"]["CALIBRATION_PREPARE"]["status"] == "pass"
    log = (root / "compile.log").read_text(encoding="utf-8")
    assert log.index("FIRST_DONE") < log.index("SECOND_START")


def test_execute_stage_rejects_unknown_stage(tmp_path):
    root = create_attempt(tmp_path, "unknown", "resnet50", "default", {})

    with pytest.raises(ValueError, match="unknown stage"):
        execute_stage(root, "NOT_A_STAGE", [sys.executable, "-c", "pass"])


def test_record_artifact_records_relative_path_size_and_hash(tmp_path):
    root = create_attempt(tmp_path, "artifact", "resnet50", "default", {})
    artifact = root / "mxq" / "resnet50.mxq"
    artifact.parent.mkdir()
    artifact.write_bytes(b"compiled artifact")

    record_artifact(root, artifact)

    result = _result(root)
    assert result["artifacts"] == [
        {
            "path": "mxq/resnet50.mxq",
            "size_bytes": len(b"compiled artifact"),
            "sha256": hashlib.sha256(b"compiled artifact").hexdigest(),
        }
    ]


def test_environment_capture_is_allowlisted_not_process_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ATTEMPT_SECRET", "must-not-be-recorded")

    root = create_attempt(tmp_path, "environment", "resnet50", "default", {})

    environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
    assert set(environment) == {"os", "architecture", "python", "packages", "wheel"}
    assert "ATTEMPT_SECRET" not in json.dumps(environment)
    assert "environ" not in json.dumps(environment).lower()


def test_record_quality_csv_keeps_allowlisted_metrics_and_hash(tmp_path):
    root = create_attempt(tmp_path, "quality", "resnet50", "default", {})
    csv_path = root / "quality.csv"
    csv_path.write_text(
        "total_samples,accuracy,f1,MSE,Top-1,mAP@0.5,ignored\n"
        "32,0.9,0.8,0.1,0.7,0.6,nope\n",
        encoding="utf-8",
    )

    record_quality_csv(root, csv_path)

    result = _result(root)
    assert result["quality_status"] == "pass"
    assert result["quality"] == {
        "result_csv": "quality.csv",
        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "sample_count": "32",
        "metrics": {
            "accuracy": "0.9",
            "f1": "0.8",
            "MSE": "0.1",
            "Top-1": "0.7",
            "mAP@0.5": "0.6",
        },
    }


def test_record_quality_csv_rejects_missing_sample_count(tmp_path):
    root = create_attempt(tmp_path, "missing-samples", "resnet50", "default", {})
    csv_path = root / "quality.csv"
    csv_path.write_text("accuracy\n0.9\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sample-count"):
        record_quality_csv(root, csv_path)


def test_record_quality_failure_only_updates_quality_evidence(tmp_path):
    root = create_attempt(tmp_path, "quality-failure", "resnet50", "default", {})
    execute_stage(root, "MBLT_COMPILE", [sys.executable, "-c", "pass"])
    execute_stage(root, "ARIES_LOAD", [sys.executable, "-c", "pass"])
    execute_stage(root, "CONTRACT_CHECK", [sys.executable, "-c", "pass"])
    execute_stage(root, "TASK_SMOKE", [sys.executable, "-c", "pass"])
    before = _result(root)
    log = root / "framework-e2e.log"
    log.write_text("framework E2E failed\n", encoding="utf-8")

    record_quality_failure(root, 19, log)

    result = _result(root)
    assert result["quality_status"] == "fail"
    assert result["quality_failure"] == {
        "exit_code": 19,
        "log": "framework-e2e.log",
        "size_bytes": len(b"framework E2E failed\n"),
        "sha256": hashlib.sha256(b"framework E2E failed\n").hexdigest(),
    }
    assert result["stages"]["TASK_SMOKE"] == before["stages"]["TASK_SMOKE"]
    for field in ("compile_status", "runtime_status", "contract_status"):
        assert result[field] == before[field]


def test_record_quality_failure_requires_nonzero_exit_code(tmp_path):
    root = create_attempt(tmp_path, "quality-zero", "resnet50", "default", {})

    with pytest.raises(ValueError, match="nonzero"):
        record_quality_failure(root, 0, root / "framework-e2e.log")


def test_record_quality_failure_requires_nonempty_regular_log(tmp_path):
    root = create_attempt(tmp_path, "quality-log", "resnet50", "default", {})
    missing = root / "missing.log"
    empty = root / "empty.log"
    empty.touch()
    directory = root / "log-directory"
    directory.mkdir()

    for invalid_log in (missing, empty, directory):
        with pytest.raises(ValueError, match="non-empty regular file"):
            record_quality_failure(root, 9, invalid_log)


def test_cli_run_returns_the_child_exit_code(tmp_path):
    root = create_attempt(tmp_path, "cli-failure", "resnet50", "default", {})
    environment = os.environ | {"PYTHONPATH": str(Path(__file__).parents[1])}

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.mobilint_compile_recipes.attempt",
            "run",
            "--attempt-root",
            str(root),
            "--stage",
            "SOURCE_SMOKE",
            "--",
            sys.executable,
            "-c",
            "import sys; sys.exit(7)",
        ],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )

    assert completed.returncode == 7
    assert _result(root)["stages"]["SOURCE_SMOKE"]["exit_code"] == 7
