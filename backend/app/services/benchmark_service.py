import subprocess
import sys
import os
import uuid
import threading
import logging
from typing import Dict, Optional
from pathlib import Path

from ..schemas.benchmark import BenchmarkStatus, BenchmarkRunRequest

logger = logging.getLogger(__name__)

# 프레임워크 루트 경로
FRAMEWORK_DIR = Path(__file__).resolve().parent.parent.parent.parent / "framework"

# framework venv의 Python 인터프리터
FRAMEWORK_PYTHON = FRAMEWORK_DIR / ".venv" / "bin" / "python"

# 진행 중인 벤치마크 작업 저장소
_jobs: Dict[str, dict] = {}
_lock = threading.Lock()


def get_available_profiles() -> list[dict]:
    """framework/src/core/model_profiles.py에서 등록된 프로필 목록을 반환한다."""
    profiles_path = FRAMEWORK_DIR / "src" / "core" / "model_profiles.py"
    if not profiles_path.exists():
        return []

    # framework/src를 sys.path에 추가하여 model_profiles.py의 상대 import 해결
    import types

    framework_src = str(FRAMEWORK_DIR / "src")
    added_to_path = framework_src not in sys.path
    if added_to_path:
        sys.path.insert(0, framework_src)

    # onnx 등 무거운 의존성 스텁 (프로필 딕셔너리 읽기에는 불필요)
    onnx_stub = None
    if "onnx" not in sys.modules:
        onnx_stub = types.ModuleType("onnx")
        sys.modules["onnx"] = onnx_stub

    try:
        # core 패키지로 정상 import
        from core.model_profiles import SUPPORTED_PROFILES
        supported = SUPPORTED_PROFILES
    finally:
        if onnx_stub is not None:
            sys.modules.pop("onnx", None)
        if added_to_path:
            sys.path.remove(framework_src)

    result = []
    for model_name, profile in supported.items():
        task_enum = profile.get("task")
        task_name = task_enum.name if hasattr(task_enum, "name") else str(task_enum)

        # 태스크에 따라 사용 가능한 백엔드 결정
        if task_name == "NLP_GENERATION":
            backends = ["onnxruntime", "vllm"]
        else:
            backends = ["onnxruntime"]

        result.append({
            "model_name": model_name,
            "task": task_name,
            "backends": backends,
            "default_model_path": profile.get("default_model_path"),
            "default_dataset_path": profile.get("default_dataset_path"),
        })

    return result


def start_benchmark(request: BenchmarkRunRequest) -> dict:
    """벤치마크를 비동기로 실행하고 job_id를 반환한다."""
    job_id = uuid.uuid4().hex[:12]

    # CLI 명령어 조립 (framework venv Python 사용)
    python = str(FRAMEWORK_PYTHON) if FRAMEWORK_PYTHON.exists() else sys.executable
    cmd = [
        python, str(FRAMEWORK_DIR / "src" / "main.py"),
        "--model", request.model,
        "--backend", request.backend,
        "--device", request.device,
        "--batch-size", str(request.batch_size),
        "--warmup", str(request.warmup),
        "--layout", request.layout,
        "--max-new-tokens", str(request.max_new_tokens),
    ]

    if request.max_steps is not None:
        cmd.extend(["--max-steps", str(request.max_steps)])
    if request.max_model_len is not None:
        cmd.extend(["--max-model-len", str(request.max_model_len)])
    if request.gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(request.gpu_memory_utilization)])
    if request.enforce_eager:
        cmd.append("--enforce-eager")
    if request.debug:
        cmd.append("--debug")
    if request.monitor:
        cmd.append("--monitor")
        cmd.extend(["--monitor-interval", str(request.monitor_interval)])

    job = {
        "job_id": job_id,
        "status": BenchmarkStatus.RUNNING,
        "model": request.model,
        "backend": request.backend,
        "device": request.device,
        "output": "",
        "error": None,
        "run_id": None,
        "process": None,
        "timeout_sec": request.timeout_sec,
    }

    with _lock:
        _jobs[job_id] = job

    # 백그라운드 스레드에서 실행
    thread = threading.Thread(
        target=_run_benchmark, args=(job_id, cmd, request.timeout_sec), daemon=True
    )
    thread.start()

    return {
        "job_id": job_id,
        "status": BenchmarkStatus.RUNNING,
        "model": request.model,
        "backend": request.backend,
        "device": request.device,
        "message": f"벤치마크 실행 시작: {request.model} ({request.backend}/{request.device})",
    }


def _kill_process(proc: subprocess.Popen, grace_sec: float = 10.0) -> None:
    """SIGTERM → (grace) → SIGKILL 계단식 종료."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=grace_sec)
        except subprocess.TimeoutExpired:
            logger.warning("subprocess %s terminate timeout, sending SIGKILL", proc.pid)
            proc.kill()
            try:
                proc.wait(timeout=grace_sec)
            except subprocess.TimeoutExpired:
                logger.error("subprocess %s SIGKILL도 무응답", proc.pid)
    except Exception:
        logger.exception("프로세스 종료 중 예외")


def _run_benchmark(job_id: str, cmd: list[str], timeout_sec: int):
    """서브프로세스로 벤치마크를 실행한다. timeout_sec 초과 시 강제 종료."""
    timed_out = threading.Event()
    timer: Optional[threading.Timer] = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(FRAMEWORK_DIR),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        with _lock:
            _jobs[job_id]["process"] = proc

        logger.info("job %s started (pid=%s, timeout=%ss)", job_id, proc.pid, timeout_sec)

        def _on_timeout():
            timed_out.set()
            logger.warning("job %s timeout after %ss, killing pid=%s", job_id, timeout_sec, proc.pid)
            _kill_process(proc)

        timer = threading.Timer(timeout_sec, _on_timeout)
        timer.daemon = True
        timer.start()

        output_lines = []
        for line in proc.stdout:
            output_lines.append(line)
            with _lock:
                _jobs[job_id]["output"] = "".join(output_lines)

        proc.wait()

        # RUN_ID=<id> 단일 라인 계약 (framework/src/main.py가 찍음)
        run_id = None
        for line in reversed(output_lines):
            stripped = line.strip()
            if stripped.startswith("RUN_ID="):
                run_id = stripped[len("RUN_ID=") :]
                break

        with _lock:
            if timed_out.is_set():
                _jobs[job_id]["status"] = BenchmarkStatus.FAILED
                _jobs[job_id]["error"] = f"timeout ({timeout_sec}s 초과로 프로세스 강제 종료)"
            elif proc.returncode == 0:
                _jobs[job_id]["status"] = BenchmarkStatus.COMPLETED
                _jobs[job_id]["run_id"] = run_id
                if run_id is None:
                    logger.warning("job %s 완료됐지만 RUN_ID 라인 없음", job_id)
            else:
                _jobs[job_id]["status"] = BenchmarkStatus.FAILED
                _jobs[job_id]["error"] = f"프로세스 종료 코드: {proc.returncode}"

        logger.info("job %s finished (status=%s)", job_id, _jobs[job_id]["status"])

    except Exception as e:
        logger.exception("job %s 실행 중 예외", job_id)
        with _lock:
            _jobs[job_id]["status"] = BenchmarkStatus.FAILED
            _jobs[job_id]["error"] = str(e)
    finally:
        if timer is not None:
            timer.cancel()


def shutdown_all_jobs() -> None:
    """서버 shutdown 시 호출. 진행 중인 모든 subprocess를 강제 종료한다."""
    with _lock:
        procs = [
            (job_id, job.get("process"))
            for job_id, job in _jobs.items()
            if job.get("status") == BenchmarkStatus.RUNNING and job.get("process") is not None
        ]
    for job_id, proc in procs:
        logger.info("shutdown: terminating job %s (pid=%s)", job_id, proc.pid)
        _kill_process(proc)
        with _lock:
            if _jobs[job_id]["status"] == BenchmarkStatus.RUNNING:
                _jobs[job_id]["status"] = BenchmarkStatus.FAILED
                _jobs[job_id]["error"] = "server shutdown"


def get_job_status(job_id: str) -> Optional[dict]:
    """작업 상태를 조회한다."""
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "model": job["model"],
            "backend": job["backend"],
            "device": job["device"],
            "output": job["output"],
            "error": job["error"],
            "run_id": job["run_id"],
        }
