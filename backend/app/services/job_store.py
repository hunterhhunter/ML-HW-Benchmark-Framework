"""웹 작업 상태의 SQLite 원장. 성능 수치는 결과 CSV에만 보관한다."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import sqlite3
from datetime import datetime, timedelta, timezone
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY, status TEXT NOT NULL, model TEXT NOT NULL,
                target_id TEXT, request_json TEXT NOT NULL, argv_json TEXT NOT NULL,
                runner_id TEXT NOT NULL, runner_fingerprint TEXT NOT NULL,
                results_path TEXT, result_source_id TEXT, log_path TEXT NOT NULL,
                created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
                pid INTEGER, exit_code INTEGER, run_id_reserved TEXT, run_id TEXT,
                result_present INTEGER NOT NULL DEFAULT 0,
                result_validation TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0,
                log_truncated INTEGER NOT NULL DEFAULT 0, error TEXT, timeout_sec INTEGER NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status, created_at)")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _row(row):
        if row is None:
            return None
        value = dict(row)
        value["request"] = json.loads(value.pop("request_json"))
        value["argv"] = json.loads(value.pop("argv_json"))
        value["result_present"] = bool(value["result_present"])
        value["cancel_requested"] = bool(value["cancel_requested"])
        value["log_truncated"] = bool(value["log_truncated"])
        return value

    def create(self, *, request: dict, argv: list[str], runner_id: str,
               runner_fingerprint: str, results_path: str | None,
               result_source_id: str | None, log_path: str, timeout_sec: int,
               max_queued: int) -> dict:
        job_id = uuid4().hex
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            queued = db.execute("SELECT count(*) FROM jobs WHERE status='queued'").fetchone()[0]
            if queued >= max_queued:
                raise OverflowError("작업 대기열이 가득 찼습니다")
            db.execute("""INSERT INTO jobs (
                job_id,status,model,target_id,request_json,argv_json,runner_id,
                runner_fingerprint,results_path,result_source_id,log_path,created_at,timeout_sec
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                job_id, "queued", request["model"], request.get("target_id"),
                json.dumps(request), json.dumps(argv), runner_id, runner_fingerprint,
                results_path, result_source_id, log_path, _now(), timeout_sec,
            ))
        return self.get(job_id)

    def get(self, job_id: str) -> dict | None:
        with self._connect() as db:
            return self._row(db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())

    def list(self, *, status: str | None = None, limit: int = 50, offset: int = 0) -> list[dict]:
        with self._connect() as db:
            if status:
                rows = db.execute(
                    "SELECT * FROM jobs WHERE status=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (status, limit, offset),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)
                ).fetchall()
            return [self._row(row) for row in rows]

    def update(self, job_id: str, **values) -> dict:
        allowed = {
            "status", "started_at", "finished_at", "pid", "exit_code",
            "run_id_reserved", "run_id", "result_present", "result_validation",
            "cancel_requested", "log_truncated", "error", "result_source_id",
        }
        if not values or set(values) - allowed:
            raise ValueError("허용되지 않은 job 필드")
        clause = ", ".join(f"{key}=?" for key in values)
        with self._connect() as db:
            db.execute(f"UPDATE jobs SET {clause} WHERE job_id=?", (*values.values(), job_id))
        return self.get(job_id)

    def claim_next(self) -> dict | None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            active = db.execute("SELECT 1 FROM jobs WHERE status='running' LIMIT 1").fetchone()
            if active:
                return None
            row = db.execute(
                "SELECT job_id FROM jobs WHERE status='queued' ORDER BY created_at, rowid LIMIT 1"
            ).fetchone()
            if not row:
                return None
            job_id = row[0]
            db.execute(
                "UPDATE jobs SET status='running', started_at=? WHERE job_id=? AND status='queued'",
                (_now(), job_id),
            )
        return self.get(job_id)

    def cancel(self, job_id: str) -> dict | None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                return None
            if row[0] == "queued":
                db.execute("UPDATE jobs SET status='cancelled', finished_at=? WHERE job_id=?", (_now(), job_id))
            elif row[0] == "running":
                db.execute("UPDATE jobs SET cancel_requested=1 WHERE job_id=?", (job_id,))
        return self.get(job_id)

    def recover(self) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE jobs SET status='interrupted', finished_at=?, error=? WHERE status='running'",
                (_now(), "서버 재시작: 이전 프로세스 소유권을 확인할 수 없음"),
            )

    def prune_older_than(self, *, days: int) -> list[str]:
        if days <= 0:
            return []
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        terminal = ("completed", "failed", "cancelled", "timed_out", "interrupted")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT job_id,log_path FROM jobs WHERE status IN (?,?,?,?,?) AND finished_at < ?",
                (*terminal, cutoff),
            ).fetchall()
            db.executemany("DELETE FROM jobs WHERE job_id=?", [(row["job_id"],) for row in rows])
        return [row["log_path"] for row in rows]
