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


def _execute_stage_in_child_process(root, stage, command, artifact=None):
    execute_stage(root, stage, command, artifact=artifact)


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
        artifact=root / "mblt" / "failed.mblt",
    )

    result = _result(root)
    assert code == 7
    assert result["failed_at"] == "MBLT_COMPILE"
    assert result["stages"]["MBLT_COMPILE"]["status"] == "fail"
    assert result["stages"]["MBLT_COMPILE"]["exit_code"] == 7
    assert result["stages"]["MXQ_COMPILE"]["status"] == "not_run"


@pytest.mark.parametrize(
    ("stage", "relative_path", "payload"),
    [
        ("MBLT_COMPILE", "mblt/model.mblt", b"mblt bytes"),
        ("MXQ_COMPILE", "mxq/model.mxq", b"mxq bytes"),
    ],
)
def test_compile_stage_success_atomically_records_exact_artifact_evidence(
    tmp_path, stage, relative_path, payload
):
    root = create_attempt(tmp_path, f"atomic-{stage}", "resnet50", "default", {})
    artifact = root / relative_path
    program = (
        "from pathlib import Path; "
        f"path=Path({str(artifact)!r}); path.parent.mkdir(parents=True); "
        f"path.write_bytes({payload!r})"
    )

    code = execute_stage(
        root,
        stage,
        [sys.executable, "-c", program],
        artifact=artifact,
    )

    result = _result(root)
    assert code == 0
    assert result["stages"][stage]["status"] == "pass"
    assert result["artifacts"] == [
        {
            "path": relative_path,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]


@pytest.mark.parametrize("artifact_state", ("missing", "empty"))
def test_compile_stage_artifact_registration_failure_records_stage_failure(
    tmp_path, artifact_state
):
    root = create_attempt(
        tmp_path, f"artifact-{artifact_state}", "resnet50", "default", {}
    )
    artifact = root / "mblt" / "model.mblt"
    if artifact_state == "empty":
        artifact.parent.mkdir()
        artifact.touch()

    code = execute_stage(
        root,
        "MBLT_COMPILE",
        [sys.executable, "-c", "pass"],
        artifact=artifact,
    )

    result = _result(root)
    assert code != 0
    assert result["failed_at"] == "MBLT_COMPILE"
    assert result["compile_status"] == "fail"
    assert result["stages"]["MBLT_COMPILE"]["status"] == "fail"
    assert "artifact" in result["stages"]["MBLT_COMPILE"]["error"].lower()
    assert result["artifacts"] == []


def test_compile_stage_hash_failure_records_stage_failure(tmp_path, monkeypatch):
    from tools.mobilint_compile_recipes import attempt as attempt_module

    root = create_attempt(tmp_path, "artifact-hash", "resnet50", "default", {})
    artifact = root / "mxq" / "model.mxq"
    program = (
        "from pathlib import Path; "
        f"path=Path({str(artifact)!r}); path.parent.mkdir(parents=True); "
        "path.write_bytes(b'mxq')"
    )
    monkeypatch.setattr(
        attempt_module,
        "sha256_file",
        lambda path: (_ for _ in ()).throw(OSError("hash read failed")),
    )

    code = execute_stage(
        root,
        "MXQ_COMPILE",
        [sys.executable, "-c", program],
        artifact=artifact,
    )

    result = _result(root)
    assert code != 0
    assert result["failed_at"] == "MXQ_COMPILE"
    assert result["compile_status"] == "fail"
    assert result["stages"]["MXQ_COMPILE"]["status"] == "fail"
    assert "hash read failed" in result["stages"]["MXQ_COMPILE"]["error"]
    assert result["artifacts"] == []


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


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX flock semantics")
def test_concurrent_compile_writers_atomically_preserve_both_artifacts(tmp_path):
    root = create_attempt(tmp_path, "compile-concurrent", "resnet50", "default", {})
    marker = root / "mblt-started"
    mblt = root / "mblt" / "model.mblt"
    mxq = root / "mxq" / "model.mxq"
    first = [
        sys.executable,
        "-c",
        "from pathlib import Path; import time; "
        f"Path({str(marker)!r}).write_text('started'); time.sleep(0.2); "
        f"path=Path({str(mblt)!r}); path.parent.mkdir(); path.write_bytes(b'mblt')",
    ]
    second = [
        sys.executable,
        "-c",
        "from pathlib import Path; "
        f"path=Path({str(mxq)!r}); path.parent.mkdir(); path.write_bytes(b'mxq')",
    ]
    context = multiprocessing.get_context("fork")
    first_writer = context.Process(
        target=_execute_stage_in_child_process,
        args=(root, "MBLT_COMPILE", first, mblt),
    )
    second_writer = context.Process(
        target=_execute_stage_in_child_process,
        args=(root, "MXQ_COMPILE", second, mxq),
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
    assert result["compile_status"] == "pass"
    assert result["failed_at"] is None
    assert result["artifacts"] == [
        {
            "path": "mblt/model.mblt",
            "size_bytes": 4,
            "sha256": hashlib.sha256(b"mblt").hexdigest(),
        },
        {
            "path": "mxq/model.mxq",
            "size_bytes": 3,
            "sha256": hashlib.sha256(b"mxq").hexdigest(),
        },
    ]


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


@pytest.mark.parametrize(
    ("header", "row", "expected_metrics"),
    [
        (
            "Total Samples,Top-1 Accuracy,Top-5 Accuracy,ignored\n",
            "64,76.5,93.2,nope\n",
            {"Top-1 Accuracy": "76.5", "Top-5 Accuracy": "93.2"},
        ),
        (
            "total_samples,exact_match,f1,ignored\n",
            "64,68.75,78.8566,nope\n",
            {"exact_match": "68.75", "f1": "78.8566"},
        ),
    ],
)
def test_record_quality_csv_preserves_real_framework_metric_headers(
    tmp_path, header, row, expected_metrics
):
    root = create_attempt(tmp_path, "real-quality", "resnet50", "default", {})
    csv_path = root / "quality.csv"
    csv_path.write_text(header + row, encoding="utf-8")

    record_quality_csv(root, csv_path)

    assert _result(root)["quality"]["metrics"] == expected_metrics


def test_record_quality_csv_rejects_missing_sample_count(tmp_path):
    root = create_attempt(tmp_path, "missing-samples", "resnet50", "default", {})
    csv_path = root / "quality.csv"
    csv_path.write_text("accuracy\n0.9\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sample-count"):
        record_quality_csv(root, csv_path)


def test_record_quality_failure_only_updates_quality_evidence(tmp_path):
    root = create_attempt(tmp_path, "quality-failure", "resnet50", "default", {})
    artifact = root / "mblt" / "resnet50.mblt"
    execute_stage(
        root,
        "MBLT_COMPILE",
        [
            sys.executable,
            "-c",
            "from pathlib import Path; "
            f"path=Path({str(artifact)!r}); path.parent.mkdir(); path.write_bytes(b'mblt')",
        ],
        artifact=artifact,
    )
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


def _write_fake_bert_reports(tmp_path):
    task_root = tmp_path / "sst2"
    (task_root / "weights").mkdir(parents=True)
    (task_root / "mblt").mkdir()
    (task_root / "mxq").mkdir()
    weights = task_root / "weights" / "weight_dict.pth"
    mblt = task_root / "mblt" / "sst2.mblt"
    mxq = task_root / "mxq" / "sst2.mxq"
    weights.write_bytes(b"embedding weights")
    mblt.write_bytes(b"mblt compiler output")
    mxq.write_bytes(b"mxq compiler output")
    manifest = {
        "task": "sst2",
        "model_id": "textattack/bert-base-uncased-SST-2",
        "target_device": "aries-rb",
        "weights": {
            "path": "weights/weight_dict.pth",
            "size_bytes": weights.stat().st_size,
            "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        },
    }
    report = {
        "task": "sst2",
        "model_id": "textattack/bert-base-uncased-SST-2",
        "target_device": "aries-rb",
        "artifacts": {
            "mblt": {
                "path": "mblt/sst2.mblt",
                "size_bytes": mblt.stat().st_size,
                "sha256": hashlib.sha256(mblt.read_bytes()).hexdigest(),
            },
            "mxq": {
                "path": "mxq/sst2.mxq",
                "size_bytes": mxq.stat().st_size,
                "sha256": hashlib.sha256(mxq.read_bytes()).hexdigest(),
            },
        },
    }
    (task_root / "calibration_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (task_root / "compile-report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    return task_root


def test_bert_bridge_maps_only_compile_evidence(tmp_path):
    from tools.mobilint_compile_recipes.bert_bridge import import_bert_compile_result

    output = tmp_path / "result.json"
    result = import_bert_compile_result(_write_fake_bert_reports(tmp_path), output)

    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert result["compile_status"] == "pass"
    assert result["runtime_status"] == "not_run"
    assert result["contract_status"] == "not_run"
    assert result["quality_status"] == "not_run"
    assert result["stages"]["MBLT_COMPILE"]["status"] == "pass"
    assert result["stages"]["MXQ_COMPILE"]["status"] == "pass"
    assert result["stages"]["SOURCE_PREPARE"]["status"] == "not_run"
    assert [entry["path"] for entry in result["artifacts"]] == [
        "mblt/sst2.mblt",
        "mxq/sst2.mxq",
    ]
    legacy_manifest = json.loads(
        (tmp_path / "sst2" / "calibration_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "calibration_artifacts" not in legacy_manifest


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing-manifest", "manifest"),
        ("missing-report", "report"),
        ("empty-mblt", "non-empty"),
        ("mismatched-mxq", "SHA256"),
        ("mismatched-task", "task mismatch"),
    ],
)
def test_bert_bridge_rejects_unproven_compile_evidence(tmp_path, mutation, message):
    from tools.mobilint_compile_recipes.bert_bridge import import_bert_compile_result

    task_root = _write_fake_bert_reports(tmp_path)
    if mutation == "missing-manifest":
        (task_root / "calibration_manifest.json").unlink()
    elif mutation == "missing-report":
        (task_root / "compile-report.json").unlink()
    elif mutation == "empty-mblt":
        (task_root / "mblt" / "sst2.mblt").write_bytes(b"")
    elif mutation == "mismatched-mxq":
        (task_root / "mxq" / "sst2.mxq").write_bytes(b"x" * 19)
    else:
        report_path = task_root / "compile-report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["task"] = "squad1"
        report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises((FileNotFoundError, ValueError), match=message):
        import_bert_compile_result(task_root, tmp_path / "result.json")


def test_experiment_help_is_dependency_free_and_lists_every_model():
    script = Path(__file__).parents[1] / "scripts" / "run_mobilint_compile_experiment.sh"

    completed = subprocess.run(
        ["bash", str(script), "--help"],
        check=True,
        text=True,
        capture_output=True,
        env={"PATH": os.environ["PATH"], "PYTHONPATH": "intentionally-invalid"},
    )

    for name in (
        "bert-sst2",
        "bert-squad1",
        "patchtst-etth1",
        "resnet50",
        "yolov5m",
    ):
        assert name in completed.stdout
    for option in (
        "--wheel",
        "--python",
        "--venv",
        "--model",
        "--variant",
        "--output-root",
        "--dataset",
        "--model-revision",
        "--yolov5-root",
        "--weights",
        "--parent-attempt",
    ):
        assert option in completed.stdout


def _write_fake_compiler_python(tmp_path):
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    log = tmp_path / "fake-python.log"
    fake_python = fake_bin / "python3.10"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -u
printf '%s\\t%s\\n' "$$" "$*" >> "${FAKE_PYTHON_LOG:?}"
if [[ "${1:-}" == "-m" && "${2:-}" == "venv" ]]; then
  mkdir -p -- "$3/bin"
  cp -- "$0" "$3/bin/python"
  chmod +x -- "$3/bin/python"
  exit 0
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "pip" ]]; then
  if [[ "${FAKE_FAIL_PIP:-}" == "1" && "${3:-}" == "install" ]]; then
    exit 29
  fi
  exit 0
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "tools.mobilint_compile_recipes.attempt" ]]; then
  exec "${REAL_TEST_PYTHON:?}" "$@"
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "tools.mobilint_compile_recipes.bert_bridge" ]]; then
  exec "${REAL_TEST_PYTHON:?}" "$@"
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "tools.mobilint_bert_compile.prepare" ]]; then
  task=""
  output_root=""
  shift 2
  while (($#)); do
    case "$1" in
      --task) task="$2"; shift 2 ;;
      --output-root) output_root="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  task_root="$output_root/$task"
  mkdir -p -- "$task_root/calibration_data"
  printf '{}\\n' > "$task_root/calibration_manifest.json"
  printf 'calibration\\n' > "$task_root/calibration_data/000.npy"
  printf 'RECIPE_STAGE=bert-prepare PID=%s\\n' "$$" >> "${FAKE_STAGE_LOG:?}"
  exit 0
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "tools.mobilint_bert_compile.compile" ]]; then
  stage=""
  artifact_root=""
  task=""
  shift 2
  while (($#)); do
    case "$1" in
      --stage) stage="$2"; shift 2 ;;
      --artifact-root) artifact_root="$2"; shift 2 ;;
      --task) task="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  if [[ "$stage" == "mblt" && ! -e "$artifact_root/$task/compile-report.json" ]]; then
    printf 'PRE_MBLT_REPORT=missing\\n' >> "${FAKE_STAGE_LOG:?}"
  fi
  printf 'RECIPE_STAGE=%s PID=%s\\n' "$stage" "$$" >> "${FAKE_STAGE_LOG:?}"
  if [[ "${FAKE_FAIL_STAGE:-}" == "$stage" ]]; then
    exit 23
  fi
  exit 95
fi
if [[ "${1:-}" == "-" ]]; then
  while IFS= read -r _line; do :; done
  exit 0
fi
if [[ "${1:-}" == "-c" ]]; then
  if [[ "${2:-}" == *parent_attempt* ]]; then
    exec "${REAL_TEST_PYTHON:?}" "$@"
  fi
  if [[ "$*" == *CALIBRATION_EVIDENCE* ]]; then
    exec "${REAL_TEST_PYTHON:?}" "$@"
  fi
  exit 0
fi
if [[ "${1:-}" != "-m" ]]; then
  printf 'unexpected fake Python command: %s\\n' "$*" >&2
  exit 97
fi
case "${2:-}" in
  tools.mobilint_compile_recipes.resnet50) artifact_name="resnet50" ;;
  tools.mobilint_compile_recipes.patchtst_etth1) artifact_name="patchtst-etth1" ;;
  *) printf 'unexpected fake Python module: %s\\n' "${2:-}" >&2; exit 97 ;;
esac
stage=""
attempt_root=""
shift 2
while (($#)); do
  case "$1" in
    --stage) stage="$2"; shift 2 ;;
    --attempt-root) attempt_root="$2"; shift 2 ;;
    *) shift ;;
  esac
done
printf 'RECIPE_STAGE=%s PID=%s\\n' "$stage" "$$" >> "${FAKE_STAGE_LOG:?}"
if [[ "${FAKE_FAIL_STAGE:-}" == "$stage" ]]; then
  exit 23
fi
if [[ "${FAKE_SKIP_ARTIFACT_STAGE:-}" == "$stage" ]]; then
  exit 0
fi
case "$stage" in
  prepare)
    mkdir -p -- "$attempt_root/calibration"
    printf '{}\\n' > "$attempt_root/source-manifest.json"
    printf '{}\\n' > "$attempt_root/compile-report.json"
    printf 'calibration\\n' > "$attempt_root/calibration/calibration.json"
    ;;
  source-smoke) ;;
  mblt)
    mkdir -p -- "$attempt_root/mblt"
    printf 'fake mblt\\n' > "$attempt_root/mblt/$artifact_name-mblt.mblt"
    ;;
  mxq)
    mkdir -p -- "$attempt_root/mxq"
    printf 'fake mxq\\n' > "$attempt_root/mxq/$artifact_name-mxq.mxq"
    ;;
  *) exit 96 ;;
esac
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    sha256sum = fake_bin / "sha256sum"
    sha256sum.write_text(
        "#!/usr/bin/env bash\nprintf '%s  %s\\n' "
        "'28f276baef1bff86ed313cb819b53d8abb684a7555cf4c81c459edc09abf1b4b' \"$1\"\n",
        encoding="utf-8",
    )
    sha256sum.chmod(0o755)
    return fake_python, fake_bin, log


def _run_fake_experiment(
    tmp_path,
    *,
    fail_stage="",
    fail_pip=False,
    model="resnet50",
    variant="default",
    model_revision=None,
    parent_attempt=None,
    skip_artifact_stage="",
    expect_attempt=True,
):
    framework = Path(__file__).parents[1]
    script = framework / "scripts" / "run_mobilint_compile_experiment.sh"
    fake_python, fake_bin, python_log = _write_fake_compiler_python(tmp_path)
    stage_log = tmp_path / "stage.log"
    wheel = tmp_path / "qbcompiler-1.2.0-py3-none-any.whl"
    wheel.write_bytes(b"fake wheel; checksum command is isolated by the test")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    injection_marker = tmp_path / "EVAL_WAS_USED"
    output_root = tmp_path / f"attempts; touch {injection_marker}"
    environment = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PYTHONPATH": str(framework),
        "REAL_TEST_PYTHON": sys.executable,
        "FAKE_PYTHON_LOG": str(python_log),
        "FAKE_STAGE_LOG": str(stage_log),
        "FAKE_FAIL_STAGE": fail_stage,
        "FAKE_SKIP_ARTIFACT_STAGE": skip_artifact_stage,
        "FAKE_FAIL_PIP": "1" if fail_pip else "0",
    }
    arguments = [
        "bash",
        str(script),
        "--wheel",
        str(wheel),
        "--python",
        str(fake_python),
        "--venv",
        str(tmp_path / "compiler venv"),
        "--model",
        model,
        "--variant",
        variant,
        "--dataset",
        str(dataset),
        "--output-root",
        str(output_root),
    ]
    if model_revision is not None:
        arguments.extend(("--model-revision", model_revision))
    if parent_attempt is not None:
        arguments.extend(("--parent-attempt", str(parent_attempt)))
    completed = subprocess.run(
        arguments,
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )
    roots = [
        Path(line.removeprefix("ATTEMPT_ROOT="))
        for line in completed.stdout.splitlines()
        if line.startswith("ATTEMPT_ROOT=")
    ]
    if expect_attempt:
        assert len(roots) == 1, completed.stdout + completed.stderr
        root = roots[0]
    else:
        assert roots == [], completed.stdout + completed.stderr
        root = None
    return completed, root, stage_log, python_log, injection_marker


def test_experiment_runs_fresh_ordered_stages_and_records_artifacts(tmp_path):
    completed, root, stage_log, python_log, injection_marker = _run_fake_experiment(
        tmp_path
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.splitlines()[-2:] == [
        f"ATTEMPT_ROOT={root}",
        "EXPERIMENT_EXIT_CODE=0",
    ]
    result = _result(root)
    assert result["compile_status"] == "pass"
    assert [result["stages"][stage]["status"] for stage in STAGES[:5]] == [
        "pass",
        "pass",
        "pass",
        "pass",
        "pass",
    ]
    assert result["artifacts"] == [
        {
            "path": "mblt/resnet50-mblt.mblt",
            "size_bytes": len(b"fake mblt\n"),
            "sha256": hashlib.sha256(b"fake mblt\n").hexdigest(),
        },
        {
            "path": "mxq/resnet50-mxq.mxq",
            "size_bytes": len(b"fake mxq\n"),
            "sha256": hashlib.sha256(b"fake mxq\n").hexdigest(),
        },
    ]
    stage_lines = stage_log.read_text(encoding="utf-8").splitlines()
    assert [line.split()[0] for line in stage_lines] == [
        "RECIPE_STAGE=prepare",
        "RECIPE_STAGE=source-smoke",
        "RECIPE_STAGE=mblt",
        "RECIPE_STAGE=mxq",
    ]
    stage_pids = [line.split("PID=", 1)[1] for line in stage_lines]
    calibration_pids = [
        line.split("\t", 1)[0]
        for line in python_log.read_text(encoding="utf-8").splitlines()
        if line.split("\t", 1)[1].startswith("-c ")
        and "CALIBRATION_EVIDENCE" in line
    ]
    assert len(calibration_pids) == 1
    assert len(set(stage_pids + calibration_pids)) == 5
    assert not injection_marker.exists()


def test_experiment_artifact_registration_failure_never_leaves_compile_pass(tmp_path):
    completed, root, stage_log, _, _ = _run_fake_experiment(
        tmp_path,
        skip_artifact_stage="mblt",
    )

    assert completed.returncode != 0
    result = _result(root)
    assert result["compile_status"] == "fail"
    assert result["failed_at"] == "MBLT_COMPILE"
    assert result["stages"]["MBLT_COMPILE"]["status"] == "fail"
    assert (
        "artifact evidence registration failed"
        in result["stages"]["MBLT_COMPILE"]["error"]
    )
    assert result["stages"]["MXQ_COMPILE"]["status"] == "not_run"
    assert result["artifacts"] == []
    assert [line.split()[0] for line in stage_log.read_text().splitlines()] == [
        "RECIPE_STAGE=prepare",
        "RECIPE_STAGE=source-smoke",
        "RECIPE_STAGE=mblt",
    ]


def test_every_model_compile_path_uses_atomic_stage_artifact_transaction():
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_mobilint_compile_experiment.sh"
    ).read_text(encoding="utf-8")

    assert "record_artifact" not in script
    assert script.count("run_compile_stage MBLT_COMPILE") == 4
    assert script.count("run_compile_stage MXQ_COMPILE") == 4


def test_experiment_stops_at_first_failure_and_preserves_exit_code(tmp_path):
    completed, root, stage_log, _, _ = _run_fake_experiment(
        tmp_path, fail_stage="source-smoke"
    )

    assert completed.returncode == 23
    assert completed.stdout.splitlines()[-2:] == [
        f"ATTEMPT_ROOT={root}",
        "EXPERIMENT_EXIT_CODE=23",
    ]
    result = _result(root)
    assert result["failed_at"] == "SOURCE_SMOKE"
    assert result["stages"]["SOURCE_PREPARE"]["status"] == "pass"
    assert result["stages"]["SOURCE_SMOKE"]["status"] == "fail"
    assert result["stages"]["CALIBRATION_PREPARE"]["status"] == "not_run"
    assert result["stages"]["MBLT_COMPILE"]["status"] == "not_run"
    assert result["stages"]["MXQ_COMPILE"]["status"] == "not_run"
    assert [line.split()[0] for line in stage_log.read_text().splitlines()] == [
        "RECIPE_STAGE=prepare",
        "RECIPE_STAGE=source-smoke",
    ]


def test_experiment_prints_attempt_and_exit_for_bootstrap_failure(tmp_path):
    completed, root, _, _, _ = _run_fake_experiment(tmp_path, fail_pip=True)

    assert completed.returncode == 29
    assert completed.stdout.splitlines()[-2:] == [
        f"ATTEMPT_ROOT={root}",
        "EXPERIMENT_EXIT_CODE=29",
    ]
    assert {stage["status"] for stage in _result(root)["stages"].values()} == {
        "not_run"
    }


def _write_patchtst_parent(tmp_path, revision="a" * 40):
    parent = tmp_path / "parent attempt"
    parent.mkdir()
    (parent / "result.json").write_text(
        json.dumps(
            {
                "attempt_id": "stock-parent-001",
                "model": "patchtst-etth1",
                "variant": "stock",
                "failed_at": "MBLT_COMPILE",
                "stages": {"MBLT_COMPILE": {"status": "fail"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (parent / "source-manifest.json").write_text(
        json.dumps({"resolved_revision": revision}) + "\n",
        encoding="utf-8",
    )
    return parent


def test_patchtst_compat_forwards_exact_sha_and_normalized_parent_identity(tmp_path):
    revision = "a" * 40
    parent = _write_patchtst_parent(tmp_path, revision)
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(parent, target_is_directory=True)

    completed, root, _, _, _ = _run_fake_experiment(
        tmp_path,
        model="patchtst-etth1",
        variant="compat-static-patchifier",
        model_revision=revision,
        parent_attempt=parent_link,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = _result(root)
    assert result["metadata"] == {
        "parent_attempt": str(parent.resolve()),
        "parent_identity": {
            "attempt_id": "stock-parent-001",
            "model": "patchtst-etth1",
            "variant": "stock",
            "failed_at": "MBLT_COMPILE",
            "resolved_revision": revision,
        },
    }
    prepare_command = result["stages"]["SOURCE_PREPARE"]["command"]
    assert prepare_command[prepare_command.index("--model-revision") + 1] == revision


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing-result", "invalid PatchTST parent attempt"),
        ("wrong-model", "invalid PatchTST parent attempt"),
        ("wrong-variant", "invalid PatchTST parent attempt"),
        ("wrong-failed-at", "invalid PatchTST parent attempt"),
        ("stage-not-fail", "invalid PatchTST parent attempt"),
        ("missing-attempt-id", "invalid PatchTST parent attempt"),
        ("missing-manifest", "invalid PatchTST parent attempt"),
        ("mismatched-revision", "invalid PatchTST parent attempt"),
    ],
)
def test_patchtst_compat_rejects_invalid_parent_before_attempt_creation(
    tmp_path, mutation, message
):
    revision = "a" * 40
    parent = _write_patchtst_parent(tmp_path, revision)
    result_path = parent / "result.json"
    manifest_path = parent / "source-manifest.json"
    if mutation == "missing-result":
        result_path.unlink()
    elif mutation == "missing-manifest":
        manifest_path.unlink()
    elif mutation == "mismatched-revision":
        manifest_path.write_text(
            json.dumps({"resolved_revision": "b" * 40}) + "\n",
            encoding="utf-8",
        )
    else:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if mutation == "wrong-model":
            result["model"] = "resnet50"
        elif mutation == "wrong-variant":
            result["variant"] = "compat-static-patchifier"
        elif mutation == "wrong-failed-at":
            result["failed_at"] = "SOURCE_SMOKE"
        elif mutation == "missing-attempt-id":
            result["attempt_id"] = ""
        else:
            result["stages"]["MBLT_COMPILE"]["status"] = "pass"
        result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")

    completed, root, _, _, _ = _run_fake_experiment(
        tmp_path,
        model="patchtst-etth1",
        variant="compat-static-patchifier",
        model_revision=revision,
        parent_attempt=parent,
        expect_attempt=False,
    )

    assert root is None
    assert completed.returncode != 0
    assert message in completed.stderr
    assert "ATTEMPT_ROOT=" not in completed.stdout
    attempted_output_root = tmp_path / f"attempts; touch {tmp_path / 'EVAL_WAS_USED'}"
    assert not attempted_output_root.exists()


@pytest.mark.parametrize("model", ("bert-sst2", "bert-squad1"))
def test_bert_reaches_mblt_before_compile_report_exists(tmp_path, model):
    completed, root, stage_log, _, _ = _run_fake_experiment(
        tmp_path,
        model=model,
        fail_stage="mblt",
    )

    assert completed.returncode == 23
    result = _result(root)
    assert result["failed_at"] == "MBLT_COMPILE"
    assert result["stages"]["CALIBRATION_PREPARE"]["status"] == "pass"
    assert result["stages"]["MBLT_COMPILE"]["status"] == "fail"
    assert "PRE_MBLT_REPORT=missing" in stage_log.read_text(encoding="utf-8")
