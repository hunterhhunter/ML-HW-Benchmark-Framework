"""
벤치마크 결과를 CSV 파일로 저장하고 조회하는 모듈.

하나의 CSV 파일(results/benchmark_results.csv)에 모든 벤치마크 결과를 누적 저장한다.
각 행은 하나의 벤치마크 실행(run)을 나타내며, 공통 메타데이터 컬럼과
태스크별 메트릭 컬럼으로 구성된다. 태스크마다 메트릭이 다르므로
해당 태스크에 없는 메트릭 컬럼은 빈 값으로 남는다.
"""

import csv
import fcntl
import json
import math
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import (
    Path,
    PosixPath,
    PurePath,
    PurePosixPath,
    PureWindowsPath,
    WindowsPath,
)
from typing import Any, Dict, List, Optional

import numpy as np


@contextmanager
def _csv_lock(results_path: Path):
    """CSV 파일에 대한 프로세스 간 배타 락. 사이드카 .lock 파일을 사용한다."""
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = results_path.with_suffix(results_path.suffix + ".lock")
    with open(lock_path, "w") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


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
    "target_id",
    "accelerator_vendor",
    "accelerator_name",
    "runtime_name",
    "compiler_name",
    "artifact_format",
    "inference_mode",
    "scenario",
    "queue_capacity",
    "worker_count",
    "batch_timeout_ms",
    "target_qps",
    "schedule_seed",
    "async_run_status",
    "async_invalid_reasons",
    "details_path",
    "request_trace_path",
]

# 기본 결과 파일 경로 (framework/results/benchmark_results.csv)
_FRAMEWORK_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULTS_DIR = _FRAMEWORK_ROOT / "results"
DEFAULT_RESULTS_PATH = DEFAULT_RESULTS_DIR / "benchmark_results.csv"

_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", re.ASCII)
_PATH_TYPES = (
    PurePath,
    PurePosixPath,
    PureWindowsPath,
    PosixPath,
    WindowsPath,
)
_NUMPY_INTEGER_TYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
)
_NUMPY_FLOAT_TYPES = (np.float16, np.float32, np.float64, np.longdouble)


def create_run_id() -> str:
    """Return a compact lowercase hexadecimal identifier safe for filenames."""
    return uuid.uuid4().hex[:8]


def _validated_run_id(run_id: str) -> str:
    if type(run_id) is not str or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(
            "run_id must be a non-empty ASCII identifier containing only "
            "letters, digits, underscores, or hyphens"
        )
    return run_id


def _safe_type_name(value: Any) -> str:
    try:
        name = type.__getattribute__(type(value), "__name__")
    except BaseException:
        return "<unknown>"
    return name if type(name) is str else "<unknown>"


def _normalize_json_value(value: Any, active: set[int]) -> Any:
    value_type = type(value)
    if value is None or value_type in (bool, int, str):
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("async details cannot contain non-finite floats")
        return value
    if value_type is np.bool_:
        return bool(value)
    if value_type in _NUMPY_INTEGER_TYPES:
        return int(value)
    if value_type in _NUMPY_FLOAT_TYPES:
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError("async details cannot contain non-finite floats")
        return converted
    if value_type is np.str_:
        return str(value)
    if value_type in _PATH_TYPES:
        return str(value)
    if isinstance(value, Enum):
        enum_value = object.__getattribute__(value, "_value_")
        return _normalize_json_value(enum_value, active)
    if value_type is np.ndarray:
        identity = id(value)
        if identity in active:
            raise ValueError("async details cannot contain container cycles")
        active.add(identity)
        try:
            converted = np.ndarray.tolist(value)
            return _normalize_json_value(converted, active)
        finally:
            active.remove(identity)

    if value_type not in (dict, list, tuple, set, frozenset):
        raise TypeError(
            f"async details type is not supported: {_safe_type_name(value)}"
        )

    identity = id(value)
    if identity in active:
        raise ValueError("async details cannot contain container cycles")
    active.add(identity)
    try:
        if value_type is dict:
            normalized = {}
            for key, item in dict.items(value):
                if type(key) is not str:
                    raise TypeError("async details object keys must be strings")
                normalized[key] = _normalize_json_value(item, active)
            return normalized
        if value_type in (list, tuple):
            length = value_type.__len__(value)
            return [
                _normalize_json_value(value_type.__getitem__(value, index), active)
                for index in range(length)
            ]

        normalized_items = [
            _normalize_json_value(item, active)
            for item in value_type.__iter__(value)
        ]
        return sorted(
            normalized_items,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    finally:
        active.remove(identity)


def _fsync_parent(path: Path) -> None:
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_parent(path)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def save_async_details(
    run_id: str,
    details: Dict[str, Any],
    results_dir: Optional[Path] = None,
) -> Path:
    """Persist strict JSON details using an atomic same-directory replace."""
    run_id = _validated_run_id(run_id)
    if type(details) is not dict:
        raise TypeError("details must be an exact dict")
    normalized = _normalize_json_value(details, set())
    normalized["schema_version"] = "1.0"
    normalized["run_id"] = run_id
    text = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    root = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    path = root / "details" / f"{run_id}.json"
    _atomic_write_text(path, text)
    return path


def save_result(
    metrics: Dict[str, Any],
    model_name: str,
    task: str,
    backend: str,
    device: str,
    batch_size: int,
    warmup_runs: int,
    max_steps: Optional[int] = None,
    target_id: str = "",
    accelerator_vendor: str = "",
    accelerator_name: str = "",
    runtime_name: str = "",
    compiler_name: str = "",
    artifact_format: str = "",
    results_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    inference_mode: str = "e2e",
    scenario: str = "",
    queue_capacity: Optional[int] = None,
    worker_count: Optional[int] = None,
    batch_timeout_ms: Optional[float] = None,
    target_qps: Optional[float] = None,
    schedule_seed: Optional[int] = None,
    async_run_status: str = "",
    async_invalid_reasons: str = "",
    details_path: str = "",
    request_trace_path: str = "",
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
        run_id: 측정 전에 할당한 안전한 실행 ID (None이면 자동 생성)
        inference_mode: 실행 모드 (`e2e` 또는 `async_queue`)
        scenario: async 부하 시나리오
        details_path: JSON sidecar 상대 경로
        request_trace_path: 선택적 JSONL trace 상대 경로

    Returns:
        생성된 run_id (UUID 문자열)
    """
    if results_path is None:
        results_path = DEFAULT_RESULTS_PATH

    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = create_run_id() if run_id is None else _validated_run_id(run_id)
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
        "target_id": target_id,
        "accelerator_vendor": accelerator_vendor,
        "accelerator_name": accelerator_name,
        "runtime_name": runtime_name,
        "compiler_name": compiler_name,
        "artifact_format": artifact_format,
        "inference_mode": inference_mode,
        "scenario": scenario,
        "queue_capacity": "" if queue_capacity is None else queue_capacity,
        "worker_count": "" if worker_count is None else worker_count,
        "batch_timeout_ms": (
            "" if batch_timeout_ms is None else batch_timeout_ms
        ),
        "target_qps": "" if target_qps is None else target_qps,
        "schedule_seed": "" if schedule_seed is None else schedule_seed,
        "async_run_status": async_run_status,
        "async_invalid_reasons": async_invalid_reasons,
        "details_path": details_path,
        "request_trace_path": request_trace_path,
    }

    # 메트릭 값 추가 (메타 컬럼과 겹치는 키는 무시)
    meta_keys = set(META_COLUMNS)
    for key, value in metrics.items():
        if key not in meta_keys:
            row[key] = value

    with _csv_lock(results_path):
        # 기존 CSV의 헤더를 읽어서 새 컬럼이 있으면 병합
        existing_columns = []
        if results_path.exists():
            with open(results_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                try:
                    existing_columns = next(reader)
                except StopIteration:
                    existing_columns = []

        # 최종 컬럼 목록: 기존 컬럼 + 새 메타데이터 + 새 메트릭
        metric_keys = [k for k in row.keys() if k not in META_COLUMNS]
        if existing_columns:
            new_meta_keys = [
                key for key in META_COLUMNS if key not in existing_columns
            ]
            new_metric_keys = [
                key for key in metric_keys if key not in existing_columns
            ]
            all_columns = existing_columns + new_meta_keys + new_metric_keys
        else:
            all_columns = META_COLUMNS + metric_keys

        if not existing_columns or all_columns != existing_columns:
            existing_rows = (
                _read_all_rows(results_path, existing_columns)
                if existing_columns
                else []
            )
            _atomic_write_csv(
                results_path,
                all_columns,
                [*existing_rows, row],
            )
        else:
            # 컬럼 변경 없음: 단순 append
            file_exists = results_path.exists() and results_path.stat().st_size > 0
            with open(results_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())

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

    with _csv_lock(results_path):
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

    with _csv_lock(results_path):
        with open(results_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            rows = list(reader)

        original_count = len(rows)
        rows = [r for r in rows if r.get("run_id") != run_id]

        if len(rows) == original_count:
            return False

        _atomic_write_csv(results_path, columns, rows)

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

    with _csv_lock(results_path):
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


def _atomic_write_csv(
    path: Path,
    columns: List[str],
    rows: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            fd = -1
            writer = csv.DictWriter(
                handle,
                fieldnames=columns,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_parent(path)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
