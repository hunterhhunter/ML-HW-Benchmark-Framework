"""
벤치마크 결과 CSV를 조회/관리하는 서비스 레이어.
framework/src/core/result_store.py를 직접 임포트하여 사용한다.
"""

import sys
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
    get_result,
    delete_result,
    DEFAULT_RESULTS_PATH,
    META_COLUMNS,
)


def _row_to_response(row: Dict[str, str]) -> Dict[str, Any]:
    """CSV 행을 API 응답 형식으로 변환한다. 메타데이터와 메트릭을 분리."""
    meta_keys = set(META_COLUMNS)
    metrics = {}
    for k, v in row.items():
        if k not in meta_keys and v != "":
            try:
                metrics[k] = float(v)
            except (ValueError, TypeError):
                metrics[k] = v

    return {
        "run_id": row.get("run_id", ""),
        "timestamp": row.get("timestamp", ""),
        "model_name": row.get("model_name", ""),
        "task": row.get("task", ""),
        "backend": row.get("backend", ""),
        "device": row.get("device", ""),
        "batch_size": row.get("batch_size", ""),
        "warmup_runs": row.get("warmup_runs", ""),
        "max_steps": row.get("max_steps") or None,
        "metrics": metrics,
    }


def list_results(
    model_name: Optional[str] = None,
    task: Optional[str] = None,
    backend: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """결과 목록 조회"""
    rows = load_results(
        model_name=model_name,
        task=task,
        backend=backend,
        limit=limit,
    )
    results = [_row_to_response(r) for r in rows]
    return {"total": len(results), "results": results}


def get_result_by_id(run_id: str) -> Optional[Dict[str, Any]]:
    """특정 run_id의 결과 조회"""
    row = get_result(run_id)
    if row is None:
        return None
    return _row_to_response(row)


def delete_result_by_id(run_id: str) -> bool:
    """특정 run_id의 결과 삭제"""
    return delete_result(run_id)
