"""프레임워크 CLI 자식 프로세스와 그 process group을 관리한다."""

import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time


RUN_ID_LINE = re.compile(r"^(RUN_ID_RESERVED|RUN_ID)=([A-Za-z0-9._-]+)$")


def terminate_group(process: subprocess.Popen, grace_sec: float = 2.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_process(
    argv: list[str], *, cwd: Path, env: dict[str, str], log_path: Path,
    timeout_sec: int, cancel_event: threading.Event,
    on_start, max_log_bytes: int = 10_000_000,
) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        argv, cwd=cwd, env={**os.environ, **env, "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
    )
    on_start(process.pid)
    captured: dict[str, str] = {}
    truncated = False

    def drain():
        nonlocal truncated
        pending = b""
        written = 0
        with log_path.open("wb") as log:
            while True:
                chunk = os.read(process.stdout.fileno(), 8192)
                if not chunk:
                    break
                remaining = max_log_bytes - written
                if remaining > 0:
                    piece = chunk[:remaining]
                    log.write(piece)
                    log.flush()
                    written += len(piece)
                if len(chunk) > remaining:
                    truncated = True
                pending = (pending + chunk)[-8192:]
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    match = RUN_ID_LINE.fullmatch(line.decode("utf-8", errors="replace").strip())
                    if match:
                        captured[match.group(1)] = match.group(2)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout_sec
    reason = None
    while process.poll() is None:
        if cancel_event.is_set():
            reason = "cancelled"
            terminate_group(process)
            break
        if time.monotonic() >= deadline:
            reason = "timed_out"
            terminate_group(process)
            break
        time.sleep(0.05)
    process.wait()
    reader.join(timeout=3)
    return {
        "status": reason or ("completed" if process.returncode == 0 else "failed"),
        "exit_code": process.returncode,
        "run_id_reserved": captured.get("RUN_ID_RESERVED"),
        "run_id": captured.get("RUN_ID"),
        "log_truncated": truncated,
    }
