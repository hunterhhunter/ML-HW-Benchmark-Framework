"""허용된 결과 CSV 경로의 영속 source ID."""

import os
from contextlib import contextmanager
from pathlib import Path
import sqlite3
from uuid import uuid4

from core.result_store import DEFAULT_RESULTS_PATH

from .benchmark_service import FRAMEWORK_DIR


def _state_path() -> Path:
    return Path(os.getenv("WEB_STATE_DIR", FRAMEWORK_DIR / "results" / "web")) / "result_sources.sqlite"


def _roots() -> list[Path]:
    raw = os.getenv("WEB_RESULTS_ROOTS")
    return [Path(value).expanduser().resolve() for value in raw.split(os.pathsep) if value] if raw else [(FRAMEWORK_DIR / "results").resolve()]


def validate_results_path(value: str | Path, *, must_exist: bool) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = FRAMEWORK_DIR / path
    path = path.resolve(strict=False)
    if not any(path.is_relative_to(root) for root in _roots()):
        raise ValueError("허용된 결과 경로 밖의 CSV입니다")
    if path.suffix.lower() != ".csv":
        raise ValueError("결과 source는 CSV 파일이어야 합니다")
    if must_exist and not path.is_file():
        raise ValueError("결과 CSV 파일이 없습니다")
    return path


class ResultSources:
    def __init__(self):
        self.db_path = _state_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS sources (source_id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE)")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def ensure(self, value: str | Path, *, must_exist: bool) -> dict:
        path = validate_results_path(value, must_exist=must_exist)
        if path == DEFAULT_RESULTS_PATH.resolve():
            return {"source_id": "default", "path": str(path), "exists": path.is_file()}
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO sources(source_id,path) VALUES (?,?)", (uuid4().hex, str(path)))
            row = db.execute("SELECT source_id FROM sources WHERE path=?", (str(path),)).fetchone()
        return {"source_id": row[0], "path": str(path), "exists": path.is_file()}

    def get(self, source_id: str) -> dict | None:
        if source_id == "default":
            path = DEFAULT_RESULTS_PATH.resolve()
        else:
            with self._connect() as db:
                row = db.execute("SELECT path FROM sources WHERE source_id=?", (source_id,)).fetchone()
            if row is None:
                return None
            path = Path(row[0])
        path = validate_results_path(path, must_exist=False)
        return {"source_id": source_id, "path": str(path), "exists": path.is_file()}

    def list(self) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT source_id,path FROM sources ORDER BY rowid").fetchall()
        result = []
        for source_id, raw_path in [("default", str(DEFAULT_RESULTS_PATH)), *rows]:
            try:
                source = self.get(source_id)
                result.append(source | {"available": True, "reason": None})
            except ValueError as exc:
                result.append({
                    "source_id": source_id, "path": raw_path, "exists": False,
                    "available": False, "reason": str(exc),
                })
        return result
