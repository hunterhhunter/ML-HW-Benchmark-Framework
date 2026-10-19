"""
벤치마크 결과 CSV를 조회/관리하는 서비스 레이어.
framework/src/core/result_store.py를 직접 임포트하여 사용한다.
"""

import sys
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

# framework 모듈을 임포트할 수 있도록 sys.path에 추가
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
_PROJECT_ROOT = _BACKEND_ROOT.parent
_FRAMEWORK_SRC = _PROJECT_ROOT / "framework" / "src"
if str(_FRAMEWORK_SRC) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK_SRC))

from core.result_store import (
    load_results,
    delete_result,
    get_result,
    DEFAULT_RESULTS_PATH,
    META_COLUMNS,
)
from .result_sources import ResultSources
from .artifact_reader import artifact_path


def _row_to_response(row: Dict[str, str]) -> Dict[str, Any]:
    """CSV 행을 API 응답 형식으로 변환한다. 메타데이터와 메트릭을 분리."""
    meta_keys = set(META_COLUMNS)
    metrics = {}
    for k, v in row.items():
        if k is None:
            continue
        if k not in meta_keys and v != "":
            try:
                metrics[k] = float(v)
            except (ValueError, TypeError):
                metrics[k] = v

    def _safe_int(val: str, default: int = 0) -> int:
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    max_steps_raw = row.get("max_steps", "")
    return {
        "run_id": row.get("run_id", ""),
        "timestamp": row.get("timestamp", ""),
        "model_name": row.get("model_name", ""),
        "task": row.get("task", ""),
        "backend": row.get("backend", ""),
        "device": row.get("device", ""),
        "target_id": row.get("target_id", ""),
        "accelerator_vendor": row.get("accelerator_vendor", ""),
        "accelerator_name": row.get("accelerator_name", ""),
        "runtime_name": row.get("runtime_name", ""),
        "compiler_name": row.get("compiler_name", ""),
        "artifact_format": row.get("artifact_format", ""),
        "batch_size": _safe_int(row.get("batch_size", "0")),
        "warmup_runs": _safe_int(row.get("warmup_runs", "0")),
        "max_steps": _safe_int(max_steps_raw) if max_steps_raw else None,
        "metrics": metrics,
    }


def _dedupe_latest_by_run_id(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """newest-first row 목록에서 같은 run_id는 첫 행만 남긴다."""
    seen: set[str] = set()
    deduped: List[Dict[str, str]] = []
    for row in rows:
        run_id = (row.get("run_id") or "").strip()
        if not run_id:
            deduped.append(row)
            continue
        if run_id in seen:
            continue
        seen.add(run_id)
        deduped.append(row)
    return deduped


def list_results(
    model_name: Optional[str] = None,
    task: Optional[str] = None,
    backend: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """결과 목록 조회"""
    # 전체 매칭 건수를 위해 limit 없이 먼저 조회
    all_rows = load_results(
        model_name=model_name,
        task=task,
        backend=backend,
    )
    all_rows = _dedupe_latest_by_run_id(all_rows)
    total = len(all_rows)

    # limit 적용
    if limit:
        all_rows = all_rows[:limit]

    results = [_row_to_response(r) for r in all_rows]
    return {"total": total, "results": results}


def get_result_by_id(run_id: str) -> Optional[Dict[str, Any]]:
    """특정 run_id의 결과 조회"""
    for row in load_results():
        if row.get("run_id") == run_id:
            return _row_to_response(row)
    return None


def delete_result_by_id(run_id: str) -> bool:
    """특정 run_id의 결과 삭제"""
    return delete_result(run_id)


def full_row(row: Dict[str, str], source_id: str, csv_path: Path) -> dict:
    metadata = {key: row.get(key, "") for key in META_COLUMNS}
    metrics: dict[str, Any] = {}
    for key, raw in row.items():
        if key in META_COLUMNS or key is None:
            continue
        if raw in (None, ""):
            metrics[key] = None
            continue
        try:
            number = float(raw)
            metrics[key] = number if math.isfinite(number) else None
        except (ValueError, TypeError):
            metrics[key] = raw
    artifacts = {}
    for kind in ("details", "failure_details", "trace"):
        try:
            path = artifact_path(csv_path, row, kind)
            artifacts[kind] = {"available": path is not None, "reason": None if path else "missing"}
        except ValueError as exc:
            artifacts[kind] = {"available": False, "reason": str(exc)}
    return {
        "source_id": source_id,
        "run_id": row.get("run_id", ""),
        "model_name": row.get("model_name", ""),
        "timestamp": row.get("timestamp", ""),
        "task": row.get("task", ""),
        "backend": row.get("backend", ""),
        "device": row.get("device", ""),
        "target_id": row.get("target_id", ""),
        "batch_size": row.get("batch_size", ""),
        "warmup_runs": row.get("warmup_runs", ""),
        "metadata": metadata,
        "metrics": metrics,
        "raw_row": row,
        "artifacts": artifacts,
    }


def list_source_results(source_id: str, *, model_name: str | None = None,
                        task: str | None = None, backend: str | None = None,
                        target: str | None = None, mode: str | None = None,
                        status: str | None = None, limit: int = 100, offset: int = 0) -> dict:
    source = ResultSources().get(source_id)
    if source is None:
        raise KeyError(source_id)
    path = Path(source["path"])
    rows = _dedupe_latest_by_run_id(load_results(
        results_path=path, model_name=model_name, task=task, backend=backend,
    ))
    if target:
        rows = [row for row in rows if row.get("target_id") == target]
    if mode:
        rows = [row for row in rows if row.get("inference_mode") == mode]
    if status:
        rows = [row for row in rows if row.get("async_run_status") == status]
    return {"total": len(rows), "results": [full_row(row, source_id, path) for row in rows[offset:offset + limit]]}


def get_source_result(source_id: str, run_id: str) -> tuple[dict, Path] | None:
    source = ResultSources().get(source_id)
    if source is None:
        raise KeyError(source_id)
    path = Path(source["path"])
    row = get_result(run_id, path)
    return (full_row(row, source_id, path), path) if row is not None else None


def delete_source_result(source_id: str, run_id: str) -> bool:
    source = ResultSources().get(source_id)
    if source is None:
        raise KeyError(source_id)
    return delete_result(run_id, Path(source["path"]))
