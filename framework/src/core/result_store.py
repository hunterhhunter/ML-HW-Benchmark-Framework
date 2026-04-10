"""
벤치마크 결과를 CSV 파일로 저장하고 조회하는 모듈.

하나의 CSV 파일(results/benchmark_results.csv)에 모든 벤치마크 결과를 누적 저장한다.
각 행은 하나의 벤치마크 실행(run)을 나타내며, 공통 메타데이터 컬럼과
태스크별 메트릭 컬럼으로 구성된다. 태스크마다 메트릭이 다르므로
해당 태스크에 없는 메트릭 컬럼은 빈 값으로 남는다.
"""

import csv
import fcntl
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# 공통 메타데이터 컬럼 (순서 보장)
META_COLUMNS = [
    "run_id",
    "timestamp",
    "model_name",
    "task",
    "backend",
    "device",
    "batch_size",
    "warmup_runs",
    "max_steps",
]

# 기본 결과 파일 경로 (framework/results/benchmark_results.csv)
_FRAMEWORK_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULTS_DIR = _FRAMEWORK_ROOT / "results"
DEFAULT_RESULTS_PATH = DEFAULT_RESULTS_DIR / "benchmark_results.csv"


def save_result(
    metrics: Dict[str, Any],
    model_name: str,
    task: str,
    backend: str,
    device: str,
    batch_size: int,
    warmup_runs: int,
    max_steps: Optional[int] = None,
    results_path: Optional[Path] = None,
) -> str:
    """
    벤치마크 결과 한 건을 CSV 파일에 추가(append)한다.

    Args:
        metrics: evaluator.compute()가 반환한 메트릭 딕셔너리
        model_name: 모델 이름 (예: 'resnet50')
        task: 태스크 이름 (예: 'IMAGE_CLASSIFICATION')
        backend: 런타임 백엔드 (예: 'onnxruntime')
        device: 추론 장치 (예: 'cuda')
        batch_size: 배치 크기
        warmup_runs: 웜업 횟수
        max_steps: 최대 스텝 수 (None이면 전체 데이터셋)
        results_path: CSV 파일 경로 (기본: framework/results/benchmark_results.csv)

    Returns:
        생성된 run_id (UUID 문자열)
    """
    if results_path is None:
        results_path = DEFAULT_RESULTS_PATH

    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 메타데이터 행 구성
    row = {
        "run_id": run_id,
        "timestamp": timestamp,
        "model_name": model_name,
        "task": task,
        "backend": backend,
        "device": device,
        "batch_size": batch_size,
        "warmup_runs": warmup_runs,
        "max_steps": max_steps if max_steps is not None else "",
    }

    # 메트릭 값 추가 (메타 컬럼과 겹치는 키는 무시)
    meta_keys = set(META_COLUMNS)
    for key, value in metrics.items():
        if key not in meta_keys:
            row[key] = value

    # 파일 잠금으로 동시 쓰기 보호
    lock_path = results_path.parent / (results_path.name + ".lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            # 기존 CSV의 헤더를 읽어서 새 컬럼이 있으면 병합
            existing_columns = []
            if results_path.exists():
                with open(results_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    try:
                        existing_columns = next(reader)
                    except StopIteration:
                        existing_columns = []

            # 최종 컬럼 목록: 기존 컬럼 + 새로 등장한 메트릭 컬럼
            metric_keys = [k for k in row.keys() if k not in META_COLUMNS]
            if existing_columns:
                new_keys = [k for k in metric_keys if k not in existing_columns]
                all_columns = existing_columns + new_keys
            else:
                all_columns = META_COLUMNS + metric_keys

            if existing_columns and set(all_columns) != set(existing_columns):
                # 새 컬럼이 추가된 경우: 전체 CSV를 다시 써야 함
                existing_rows = _read_all_rows(results_path, existing_columns)
                with open(results_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
                    writer.writeheader()
                    for existing_row in existing_rows:
                        writer.writerow(existing_row)
                    writer.writerow(row)
            else:
                # 컬럼 변경 없음: 단순 append
                file_exists = results_path.exists() and results_path.stat().st_size > 0
                with open(results_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(row)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    return run_id


def load_results(
    results_path: Optional[Path] = None,
    model_name: Optional[str] = None,
    task: Optional[str] = None,
    backend: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    CSV 파일에서 벤치마크 결과를 읽어 반환한다.

    Args:
        results_path: CSV 파일 경로
        model_name: 모델 이름으로 필터링 (None이면 전체)
        task: 태스크로 필터링
        backend: 백엔드로 필터링
        limit: 최대 반환 건수 (최신순, None이면 전체)

    Returns:
        딕셔너리 리스트 (최신순 정렬)
    """
    if results_path is None:
        results_path = DEFAULT_RESULTS_PATH

    results_path = Path(results_path)
    if not results_path.exists():
        return []

    with open(results_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 필터링
    if model_name:
        rows = [r for r in rows if r.get("model_name") == model_name]
    if task:
        rows = [r for r in rows if r.get("task") == task]
    if backend:
        rows = [r for r in rows if r.get("backend") == backend]

    # 최신순 정렬 (timestamp 역순)
    rows.reverse()

    if limit:
        rows = rows[:limit]

    return rows


def delete_result(
    run_id: str,
    results_path: Optional[Path] = None,
) -> bool:
    """
    특정 run_id의 결과를 삭제한다.

    Returns:
        삭제 성공 여부
    """
    if results_path is None:
        results_path = DEFAULT_RESULTS_PATH

    results_path = Path(results_path)
    if not results_path.exists():
        return False

    with open(results_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)

    original_count = len(rows)
    rows = [r for r in rows if r.get("run_id") != run_id]

    if len(rows) == original_count:
        return False

    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    return True


def get_result(
    run_id: str,
    results_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """특정 run_id의 결과를 반환한다."""
    if results_path is None:
        results_path = DEFAULT_RESULTS_PATH

    results_path = Path(results_path)
    if not results_path.exists():
        return None

    with open(results_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("run_id") == run_id:
                return row

    return None


def _read_all_rows(path: Path, columns: list) -> List[Dict[str, str]]:
    """기존 CSV의 모든 데이터 행을 읽어 반환한다."""
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows
