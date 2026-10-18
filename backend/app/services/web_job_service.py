"""API와 영속 JobManager 사이의 단일 진입점."""

import os
from pathlib import Path
import threading

from ..schemas.benchmark import BenchmarkRunRequest
from .benchmark_service import FRAMEWORK_DIR, get_available_targets
from .job_manager import JobManager
from .preflight_service import preview_benchmark
from .runner_profiles import select_runner_profile
from .result_sources import ResultSources


_manager: JobManager | None = None
_manager_lock = threading.Lock()


def get_manager() -> JobManager:
    global _manager
    state_dir = Path(os.getenv("WEB_STATE_DIR", FRAMEWORK_DIR / "results" / "web")).resolve()
    with _manager_lock:
        if _manager is None or _manager.state_dir != state_dir:
            if _manager is not None:
                _manager.stop()
            _manager = JobManager(
                state_dir=state_dir, framework_dir=FRAMEWORK_DIR,
                max_queued=int(os.getenv("WEB_MAX_QUEUED", "10")),
                max_log_bytes=int(os.getenv("WEB_MAX_LOG_BYTES", "10000000")),
            )
        _manager.start()
        return _manager


def shutdown_all_jobs() -> None:
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.stop()
            _manager = None


def _public(job: dict) -> dict:
    request = job["request"]
    target = next(
        (item for item in get_available_targets() if item["target_id"] == job["target_id"]),
        None,
    )
    return {
        key: job[key] for key in (
            "job_id", "status", "model", "target_id", "runner_id", "runner_fingerprint",
            "created_at", "started_at", "finished_at", "pid", "exit_code",
            "run_id_reserved", "run_id", "result_present", "result_validation",
            "results_path", "result_source_id", "log_truncated", "error", "timeout_sec",
        )
    } | {
        "backend": target["runtime_name"] if target else request.get("backend") or "onnxruntime",
        "device": target["device"] if target else request.get("device") or "cpu",
    }


def start_benchmark(request: BenchmarkRunRequest) -> dict:
    preview = preview_benchmark(request)
    if preview["errors"]:
        raise ValueError("; ".join(preview["errors"]))
    profile = select_runner_profile(request.target_id, request.model, request.runner_profile_id)
    results_path = Path(preview["resolved_results_path"]) if preview["resolved_results_path"] else FRAMEWORK_DIR / "results" / "benchmark_results.csv"
    if not results_path.is_absolute():
        results_path = FRAMEWORK_DIR / results_path
    source = ResultSources().ensure(results_path, must_exist=False)
    job = get_manager().enqueue(
        request=request, argv=preview["argv"], runner_id=profile.id,
        runner_fingerprint=profile.fingerprint(), results_path=results_path.resolve(),
        result_source_id=source["source_id"],
    )
    return _public(job) | {"message": "벤치마크 작업을 대기열에 등록했습니다"}


def get_job_status(job_id: str) -> dict | None:
    job = get_manager().get(job_id)
    return _public(job) if job else None


def list_jobs(*, status: str | None = None, limit: int = 50, offset: int = 0) -> dict:
    return {"jobs": [_public(job) for job in get_manager().list(status=status, limit=limit, offset=offset)]}


def cancel_benchmark(job_id: str) -> dict | None:
    job = get_manager().cancel(job_id)
    return _public(job) if job else None


def get_job_log(job_id: str, *, offset: int, limit: int) -> dict | None:
    return get_manager().log(job_id, offset=offset, limit=limit)
