"""실제 자식 프로세스로 작업 상태, 결과 연결, 취소를 확인한다."""

import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "framework" / "src"))

from app.schemas.benchmark import BenchmarkRunRequest


def manager_class():
    return importlib.import_module("app.services.job_manager").JobManager


def registered_runner(tmp_path, monkeypatch):
    config = tmp_path / "runners.json"
    config.write_text(json.dumps({"profiles": [{"id": "test", "python": sys.executable}]}))
    monkeypatch.setenv("WEB_RUNNERS_FILE", str(config))
    return importlib.import_module("app.services.runner_profiles").select_runner_profile("cpu", "resnet50")


def await_status(manager, job_id, expected, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if job["status"] == expected:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach {expected}: {manager.get(job_id)}")


def test_completed_job_persists_log_and_csv_evidence(tmp_path, monkeypatch):
    script = tmp_path / "fake_cli.py"
    csv_path = tmp_path / "result.csv"
    script.write_text("""
import csv, sys
print('RUN_ID_RESERVED=run-1', flush=True)
print('hello from cli', flush=True)
with open(sys.argv[1], 'w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['run_id', 'model_name', 'async_run_status'])
    writer.writerow(['run-1', 'resnet50', 'valid'])
print('RUN_ID=run-1', flush=True)
""")
    Manager = manager_class()
    profile = registered_runner(tmp_path, monkeypatch)
    manager = Manager(state_dir=tmp_path / "state", framework_dir=tmp_path)
    manager.start()
    try:
        job = manager.enqueue(
            request=BenchmarkRunRequest(model="resnet50", target_id="cpu"),
            argv=[sys.executable, str(script), str(csv_path)],
            runner_id=profile.id, runner_fingerprint=profile.fingerprint(), results_path=csv_path,
        )
        done = await_status(manager, job["job_id"], "completed")
        assert done["run_id_reserved"] == "run-1"
        assert done["run_id"] == "run-1"
        assert done["result_present"] is True
        log = manager.log(job["job_id"], offset=0, limit=100)
        assert "hello from cli" in log["text"]
        assert log["next_offset"] > 0
    finally:
        manager.stop()

    reopened = Manager(state_dir=tmp_path / "state", framework_dir=tmp_path)
    assert reopened.get(job["job_id"])["status"] == "completed"


def test_running_and_queued_jobs_can_be_cancelled_independently(tmp_path, monkeypatch):
    script = tmp_path / "slow.py"
    script.write_text("import time\nprint('started', flush=True)\ntime.sleep(30)\n")
    Manager = manager_class()
    profile = registered_runner(tmp_path, monkeypatch)
    manager = Manager(state_dir=tmp_path / "state", framework_dir=tmp_path)
    manager.start()
    try:
        request = BenchmarkRunRequest(model="resnet50", target_id="cpu")
        first = manager.enqueue(request=request, argv=[sys.executable, str(script)],
                                runner_id=profile.id, runner_fingerprint=profile.fingerprint())
        await_status(manager, first["job_id"], "running")
        second = manager.enqueue(request=request, argv=[sys.executable, str(script)],
                                 runner_id=profile.id, runner_fingerprint=profile.fingerprint())
        manager.cancel(second["job_id"])
        assert await_status(manager, second["job_id"], "cancelled")["started_at"] is None
        manager.cancel(first["job_id"])
        assert await_status(manager, first["job_id"], "cancelled")["finished_at"]
    finally:
        manager.stop()


def test_timeout_and_restart_recovery_are_distinct(tmp_path, monkeypatch):
    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(30)\n")
    Manager = manager_class()
    profile = registered_runner(tmp_path, monkeypatch)
    manager = Manager(state_dir=tmp_path / "state", framework_dir=tmp_path)
    manager.start()
    try:
        request = BenchmarkRunRequest(model="resnet50", target_id="cpu")
        job = manager.enqueue(request=request, argv=[sys.executable, str(script)],
                              runner_id=profile.id, runner_fingerprint=profile.fingerprint(), timeout_sec=1)
        assert await_status(manager, job["job_id"], "timed_out", timeout=4)["exit_code"] is not None
    finally:
        manager.stop()


def test_cancel_during_claim_never_starts_process(tmp_path, monkeypatch):
    script = tmp_path / "should_not_run.py"
    marker = tmp_path / "started"
    script.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('started')\n")
    profile = registered_runner(tmp_path, monkeypatch)
    manager = manager_class()(state_dir=tmp_path / "state", framework_dir=tmp_path)
    original_claim = manager.store.claim_next

    def cancel_after_claim():
        job = original_claim()
        if job:
            manager.cancel(job["job_id"])
        return job

    monkeypatch.setattr(manager.store, "claim_next", cancel_after_claim)
    try:
        job = manager.enqueue(
            request=BenchmarkRunRequest(model="resnet50", target_id="cpu"),
            argv=[sys.executable, str(script)], runner_id=profile.id,
            runner_fingerprint=profile.fingerprint(),
        )
        await_status(manager, job["job_id"], "cancelled")
        assert not marker.exists()
    finally:
        manager.stop()


def test_server_shutdown_marks_active_job_interrupted(tmp_path, monkeypatch):
    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(30)\n")
    profile = registered_runner(tmp_path, monkeypatch)
    manager = manager_class()(state_dir=tmp_path / "state", framework_dir=tmp_path)
    manager.start()
    job = manager.enqueue(
        request=BenchmarkRunRequest(model="resnet50", target_id="cpu"),
        argv=[sys.executable, str(script)], runner_id=profile.id,
        runner_fingerprint=profile.fingerprint(),
    )
    await_status(manager, job["job_id"], "running")
    manager.stop()
    assert manager.get(job["job_id"])["status"] == "interrupted"


def test_large_output_is_bounded_without_losing_run_ids(tmp_path, monkeypatch):
    script = tmp_path / "verbose.py"
    script.write_text("import os\nos.write(1, b'RUN_ID_RESERVED=big-run\\n' + b'x' * 20000 + b'\\nRUN_ID=big-run\\n')\n")
    profile = registered_runner(tmp_path, monkeypatch)
    manager = manager_class()(state_dir=tmp_path / "state", framework_dir=tmp_path,
                              max_log_bytes=100)
    manager.start()
    try:
        job = manager.enqueue(
            request=BenchmarkRunRequest(model="resnet50", target_id="cpu"),
            argv=[sys.executable, str(script)], runner_id=profile.id,
            runner_fingerprint=profile.fingerprint(),
        )
        done = await_status(manager, job["job_id"], "completed")
        assert done["run_id_reserved"] == "big-run"
        assert done["run_id"] == "big-run"
        assert done["log_truncated"] is True
        assert len(manager.log(job["job_id"])["text"].encode()) <= 100
    finally:
        manager.stop()


def test_malformed_result_does_not_erase_process_outcome_or_run_ids(tmp_path, monkeypatch):
    script = tmp_path / "bad_csv.py"
    csv_path = tmp_path / "bad.csv"
    script.write_text("import sys\nfrom pathlib import Path\n"
                      "print('RUN_ID_RESERVED=bad-run', flush=True)\n"
                      "Path(sys.argv[1]).write_bytes(b'\\xff\\xfe')\n"
                      "print('RUN_ID=bad-run', flush=True)\n")
    profile = registered_runner(tmp_path, monkeypatch)
    manager = manager_class()(state_dir=tmp_path / "state", framework_dir=tmp_path)
    manager.start()
    try:
        job = manager.enqueue(
            request=BenchmarkRunRequest(model="resnet50", target_id="cpu"),
            argv=[sys.executable, str(script), str(csv_path)],
            runner_id=profile.id, runner_fingerprint=profile.fingerprint(), results_path=csv_path,
        )
        done = await_status(manager, job["job_id"], "completed")
        assert done["run_id"] == "bad-run"
        assert done["run_id_reserved"] == "bad-run"
        assert done["result_present"] is False
        assert done["result_validation"] == "unreadable"
    finally:
        manager.stop()
