"""단일 호스트의 영속 웹 작업 queue."""

import fcntl
import os
from pathlib import Path
import threading
import time

from core.result_store import get_result

from ..schemas.benchmark import BenchmarkRunRequest
from .job_store import JobStore, _now
from .process_runner import run_process
from .runner_profiles import select_runner_profile


class JobManager:
    def __init__(self, *, state_dir: Path, framework_dir: Path,
                 max_queued: int = 10, max_log_bytes: int = 10_000_000):
        self.state_dir = Path(state_dir)
        self.framework_dir = Path(framework_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.store = JobStore(self.state_dir / "jobs.sqlite")
        self.max_queued = max_queued
        self.max_log_bytes = max_log_bytes
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._active_cancel = threading.Event()
        self._active_id: str | None = None
        self._thread: threading.Thread | None = None
        self._lock_file = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        lock_file = (self.state_dir / "worker.lock").open("a+")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_file.close()
            raise RuntimeError("웹 작업 worker가 이미 실행 중입니다") from exc
        self._lock_file = lock_file
        self.store.recover()
        self._stop.clear()
        self._thread = threading.Thread(target=self._work, name="benchmark-web-worker", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._active_cancel.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise RuntimeError("웹 작업 worker 종료를 확인할 수 없습니다")
        self.store.recover()
        if self._lock_file:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None

    def enqueue(self, *, request: BenchmarkRunRequest, argv: list[str],
                runner_id: str, runner_fingerprint: str,
                results_path: Path | None = None, result_source_id: str | None = None,
                timeout_sec: int | None = None) -> dict:
        if self._thread is None or not self._thread.is_alive():
            self.start()
        log_path = self.state_dir / "logs" / (os.urandom(12).hex() + ".log")
        job = self.store.create(
            request=request.model_dump(mode="json"), argv=argv,
            runner_id=runner_id, runner_fingerprint=runner_fingerprint,
            results_path=str(results_path) if results_path else None,
            result_source_id=result_source_id, log_path=str(log_path),
            timeout_sec=timeout_sec or request.timeout_sec, max_queued=self.max_queued,
        )
        self._wake.set()
        return job

    def get(self, job_id: str) -> dict | None:
        return self.store.get(job_id)

    def list(self, **kwargs) -> list[dict]:
        return self.store.list(**kwargs)

    def cancel(self, job_id: str) -> dict | None:
        job = self.store.cancel(job_id)
        if job and self._active_id == job_id:
            self._active_cancel.set()
        self._wake.set()
        return job

    def log(self, job_id: str, *, offset: int = 0, limit: int = 65536) -> dict | None:
        job = self.get(job_id)
        if job is None:
            return None
        if offset < 0 or not 1 <= limit <= 65536:
            raise ValueError("offset 또는 limit 범위가 잘못됐습니다")
        path = Path(job["log_path"])
        if not path.exists():
            return {"text": "", "next_offset": 0, "truncated": False}
        with path.open("rb") as stream:
            stream.seek(offset)
            raw = stream.read(limit)
            next_offset = stream.tell()
        return {
            "text": raw.decode("utf-8", errors="replace"),
            "next_offset": next_offset,
            "truncated": job["log_truncated"],
        }

    def _work(self):
        while not self._stop.is_set():
            job = self.store.claim_next()
            if job is None:
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            self._active_cancel.clear()
            self._active_id = job["job_id"]
            if self.store.get(job["job_id"])["cancel_requested"]:
                self.store.update(job["job_id"], status="cancelled", finished_at=_now())
                self._active_id = None
                continue
            try:
                profile = select_runner_profile(
                    job["target_id"], job["model"], job["runner_id"]
                )
                if profile.fingerprint() != job["runner_fingerprint"]:
                    raise ValueError("대기 중 Python 실행 환경 설정이 변경됐습니다")
                result = run_process(
                    job["argv"], cwd=self.framework_dir, env=profile.env,
                    log_path=Path(job["log_path"]), timeout_sec=job["timeout_sec"],
                    cancel_event=self._active_cancel,
                    on_start=lambda pid: self.store.update(job["job_id"], pid=pid),
                    max_log_bytes=self.max_log_bytes,
                )
                path = Path(job["results_path"]) if job["results_path"] else None
                row = None
                result_validation = "missing"
                result_error = None
                if result["run_id"]:
                    try:
                        row = get_result(result["run_id"], path)
                        if row is not None:
                            result_validation = "present"
                    except Exception as exc:
                        result_validation = "unreadable"
                        result_error = f"결과 CSV를 읽을 수 없습니다: {exc}"
                if self._stop.is_set() and result["status"] == "cancelled" and not self.store.get(job["job_id"])["cancel_requested"]:
                    result["status"] = "interrupted"
                self.store.update(
                    job["job_id"], status=result["status"], finished_at=_now(),
                    exit_code=result["exit_code"],
                    run_id_reserved=result["run_id_reserved"], run_id=result["run_id"],
                    result_present=int(row is not None),
                    result_validation=result_validation,
                    log_truncated=int(result["log_truncated"]),
                    error=result_error or (None if result["status"] == "completed" else result["status"]),
                )
            except Exception as exc:
                self.store.update(job["job_id"], status="failed", finished_at=_now(), error=str(exc))
            finally:
                self._active_id = None
                self._active_cancel.clear()
