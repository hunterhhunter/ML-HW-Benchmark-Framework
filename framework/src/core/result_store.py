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
import stat
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

from .artifact_reservation import (
    ArtifactFilesystemUnsupportedError,
    RunArtifactReservation,
    VerifiedReservation,
    consume_reservation,
    create_reservation_marker,
    directory_binding_matches,
    link_no_overwrite,
    open_results_root,
    reservation_binding_matches,
    verify_reservation,
)


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
_DETAILS_MAX_DEPTH = 32
_DETAILS_MAX_ITEMS = 10_000
_DETAILS_MAX_ARRAY_ITEMS = 4_096
_DETAILS_MAX_STRING_LENGTH = 1_000_000


class AsyncDetailsNormalizationError(ValueError):
    """Raised when async-detail normalization exceeds a safety budget."""


def create_run_id() -> str:
    """Return a compact lowercase hexadecimal identifier safe for filenames."""
    return uuid.uuid4().hex[:8]


@contextmanager
def _csv_lock_at(root_fd: int, results_name: str):
    lock_name = f"{results_name}.lock"
    lock_fd = os.open(
        lock_name,
        os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o644,
        dir_fd=root_fd,
    )
    with os.fdopen(lock_fd, "w") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def _validated_run_id(run_id: str) -> str:
    if type(run_id) is not str or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(
            "run_id must be a non-empty ASCII identifier containing only "
            "letters, digits, underscores, or hyphens"
        )
    return run_id


def reserve_run_artifacts(
    results_path: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> RunArtifactReservation:
    """Durably reserve one run ID and owner token before measurement."""
    supplied_run_id = run_id is not None
    if supplied_run_id:
        run_id = _validated_run_id(run_id)
    if results_path is None:
        results_path = DEFAULT_RESULTS_PATH

    with open_results_root(results_path, create=True) as opened_root:
        with _csv_lock_at(
            opened_root.root.file_descriptor,
            opened_root.results_name,
        ):
            columns, rows = _read_csv_structure_at(
                opened_root.root.file_descriptor,
                opened_root.results_name,
            )
            existing_run_ids = set()
            if "run_id" in columns:
                run_id_index = columns.index("run_id")
                existing_run_ids = {row[run_id_index] for row in rows}

            if supplied_run_id:
                if run_id in existing_run_ids:
                    raise ValueError(f"run_id already exists: {run_id}")
                return create_reservation_marker(opened_root, run_id)

            while True:
                candidate = _validated_run_id(create_run_id())
                if candidate in existing_run_ids:
                    continue
                try:
                    return create_reservation_marker(opened_root, candidate)
                except FileExistsError:
                    continue


def _safe_type_name(value: Any) -> str:
    try:
        name = type.__getattribute__(type(value), "__name__")
    except BaseException:
        return "<unknown>"
    return name if type(name) is str else "<unknown>"


def _safe_persistence_error(phase: str, exc: BaseException) -> Dict[str, Any]:
    error_type = _safe_type_name(exc)
    error_message = f"<{error_type}>"
    try:
        args = BaseException.args.__get__(exc, type(exc))
    except BaseException:
        args = ()
    if type(args) is tuple and len(args) == 1 and type(args[0]) is str:
        error_message = args[0]
    return {
        "phase": phase,
        "error_type": error_type,
        "error_message": error_message,
    }


def _attach_secondary_error(
    primary: BaseException,
    phase: str,
    secondary: BaseException,
    *,
    temporary_file_may_remain: bool = False,
    temporary_path: Optional[str] = None,
    publication_state_uncertain: bool = False,
) -> None:
    diagnostic = _safe_persistence_error(phase, secondary)
    if temporary_file_may_remain:
        diagnostic["temporary_file_may_remain"] = True
    if temporary_path is not None:
        diagnostic["temporary_path"] = temporary_path
    if publication_state_uncertain:
        diagnostic["publication_state_uncertain"] = True
    try:
        errors = getattr(primary, "persistence_secondary_errors", None)
    except BaseException:
        errors = None
    if type(errors) is not list:
        errors = []
        try:
            setattr(primary, "persistence_secondary_errors", errors)
        except BaseException:
            pass
    errors.append(diagnostic)
    try:
        primary.add_note(
            f"secondary persistence failure during {phase}: "
            f"{diagnostic['error_type']}: {diagnostic['error_message']}"
        )
    except BaseException:
        pass


def _normalize_json_value(
    value: Any,
    active: set[int],
    *,
    depth: int = 0,
    item_count: Optional[List[int]] = None,
) -> Any:
    if item_count is None:
        item_count = [0]
    if depth > _DETAILS_MAX_DEPTH:
        raise AsyncDetailsNormalizationError(
            "async details exceeded the depth budget"
        )
    if item_count[0] >= _DETAILS_MAX_ITEMS:
        raise AsyncDetailsNormalizationError(
            "async details exceeded the item budget"
        )
    item_count[0] += 1

    value_type = type(value)
    if value is None or value_type in (bool, int):
        return value
    if value_type is str:
        if len(value) > _DETAILS_MAX_STRING_LENGTH:
            raise AsyncDetailsNormalizationError(
                "async details exceeded the string budget"
            )
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
        converted = str(value)
        if len(converted) > _DETAILS_MAX_STRING_LENGTH:
            raise AsyncDetailsNormalizationError(
                "async details exceeded the string budget"
            )
        return converted
    if value_type in _PATH_TYPES:
        converted = str(value)
        if len(converted) > _DETAILS_MAX_STRING_LENGTH:
            raise AsyncDetailsNormalizationError(
                "async details exceeded the string budget"
            )
        return converted
    if isinstance(value, Enum):
        identity = id(value)
        if identity in active:
            raise ValueError("async details cannot contain cycles")
        active.add(identity)
        try:
            enum_value = object.__getattribute__(value, "_value_")
            return _normalize_json_value(
                enum_value,
                active,
                depth=depth + 1,
                item_count=item_count,
            )
        finally:
            active.remove(identity)
    if value_type is np.ndarray:
        if value.size > _DETAILS_MAX_ARRAY_ITEMS:
            raise AsyncDetailsNormalizationError(
                "async details exceeded the array budget"
            )
        if value.ndim + depth > _DETAILS_MAX_DEPTH:
            raise AsyncDetailsNormalizationError(
                "async details exceeded the depth budget"
            )
        identity = id(value)
        if identity in active:
            raise ValueError("async details cannot contain container cycles")
        active.add(identity)
        try:
            converted = np.ndarray.tolist(value)
            return _normalize_json_value(
                converted,
                active,
                depth=depth,
                item_count=item_count,
            )
        finally:
            active.remove(identity)

    if value_type not in (dict, list, tuple, set, frozenset):
        raise TypeError(
            f"async details type is not supported: {_safe_type_name(value)}"
        )

    size = value_type.__len__(value)
    if size > _DETAILS_MAX_ITEMS - item_count[0]:
        raise AsyncDetailsNormalizationError(
            "async details exceeded the item budget"
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
                if len(key) > _DETAILS_MAX_STRING_LENGTH:
                    raise AsyncDetailsNormalizationError(
                        "async details exceeded the string budget"
                    )
                normalized[key] = _normalize_json_value(
                    item,
                    active,
                    depth=depth + 1,
                    item_count=item_count,
                )
            return normalized
        if value_type in (list, tuple):
            return [
                _normalize_json_value(
                    value_type.__getitem__(value, index),
                    active,
                    depth=depth + 1,
                    item_count=item_count,
                )
                for index in range(size)
            ]

        normalized_items = [
            _normalize_json_value(
                item,
                active,
                depth=depth + 1,
                item_count=item_count,
            )
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
    primary = None
    try:
        os.fsync(directory_fd)
    except BaseException as exc:
        primary = exc
    try:
        os.close(directory_fd)
    except BaseException as exc:
        if primary is None:
            primary = exc
        else:
            _attach_secondary_error(primary, "close_parent_directory", exc)
    if primary is not None:
        raise primary


def _sidecar_directories_match(
    verified: VerifiedReservation,
    details_fd: int,
) -> bool:
    return reservation_binding_matches(verified) and directory_binding_matches(
        verified.root.path / "details",
        details_fd,
    )


def _atomic_write_sidecar(
    verified: VerifiedReservation,
    final_name: str,
    text: str,
) -> Path:
    root = verified.root.path
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_fd = verified.root.file_descriptor
    details_fd = None
    file_fd = None
    handle = None
    temporary_name = None
    final_published = False
    committed = False
    handle = None
    primary = None
    try:
        try:
            os.mkdir("details", mode=0o755, dir_fd=root_fd)
        except FileExistsError:
            pass
        else:
            os.fsync(root_fd)
        details_fd = os.open("details", directory_flags, dir_fd=root_fd)
        if not _sidecar_directories_match(verified, details_fd):
            raise OSError("sidecar details directory changed during publication")

        temporary_name = f".{final_name}.{uuid.uuid4().hex}.tmp"
        file_fd = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=details_fd,
        )
        handle = os.fdopen(file_fd, "w", encoding="utf-8")
        file_fd = None
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        if not _sidecar_directories_match(verified, details_fd):
            raise OSError("sidecar details directory changed during publication")
        link_no_overwrite(
            temporary_name,
            final_name,
            source_directory_fd=details_fd,
            target_directory_fd=details_fd,
        )
        final_published = True
        if not _sidecar_directories_match(verified, details_fd):
            raise OSError("sidecar details directory changed during publication")
        os.unlink(temporary_name, dir_fd=details_fd)
        temporary_name = None
        os.fsync(details_fd)
        if not _sidecar_directories_match(verified, details_fd):
            raise OSError("sidecar details directory changed during publication")
        committed = True
    except BaseException as exc:
        primary = exc
    finally:
        if final_published and not committed and details_fd is not None:
            try:
                os.unlink(final_name, dir_fd=details_fd)
            except BaseException as exc:
                if primary is None:
                    primary = exc
                else:
                    _attach_secondary_error(
                        primary,
                        "rollback_final",
                        exc,
                        publication_state_uncertain=True,
                    )
            else:
                try:
                    os.fsync(details_fd)
                except BaseException as exc:
                    if primary is None:
                        primary = exc
                    else:
                        _attach_secondary_error(
                            primary,
                            "rollback_directory_fsync",
                            exc,
                            publication_state_uncertain=True,
                        )
        if handle is not None:
            try:
                handle.close()
            except BaseException as exc:
                if primary is None:
                    primary = exc
                else:
                    _attach_secondary_error(primary, "close_file", exc)
        if file_fd is not None:
            try:
                os.close(file_fd)
            except BaseException as exc:
                if primary is None:
                    primary = exc
                else:
                    _attach_secondary_error(primary, "close_descriptor", exc)
        if temporary_name is not None and details_fd is not None:
            try:
                os.unlink(temporary_name, dir_fd=details_fd)
            except FileNotFoundError:
                pass
            except BaseException as exc:
                temporary_path = str(root / "details" / temporary_name)
                if primary is None:
                    primary = exc
                    try:
                        setattr(
                            primary,
                            "persistence_secondary_errors",
                            [
                                {
                                    **_safe_persistence_error(
                                        "cleanup_temp",
                                        exc,
                                    ),
                                    "temporary_file_may_remain": True,
                                    "temporary_path": temporary_path,
                                }
                            ],
                        )
                    except BaseException:
                        pass
                else:
                    _attach_secondary_error(
                        primary,
                        "cleanup_temp",
                        exc,
                        temporary_file_may_remain=True,
                        temporary_path=temporary_path,
                    )
        if details_fd is not None:
            try:
                os.close(details_fd)
            except BaseException as exc:
                if primary is None:
                    primary = exc
                else:
                    _attach_secondary_error(primary, "close_details_directory", exc)
    if primary is not None:
        raise primary
    return root / "details" / final_name


def save_async_details(
    run_id: str,
    details: Dict[str, Any],
    results_dir: Optional[Path] = None,
    reservation: Optional[RunArtifactReservation] = None,
) -> Path:
    """Persist strict JSON details using an atomic same-directory replace."""
    run_id = _validated_run_id(run_id)
    if reservation is None:
        raise ValueError("a valid run artifact reservation is required")
    root = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    with verify_reservation(
        reservation,
        run_id,
        results_root=root,
    ) as verified:
        return _save_verified_async_details(verified, run_id, details)


def _save_verified_async_details(
    verified: VerifiedReservation,
    run_id: str,
    details: Dict[str, Any],
) -> Path:
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
    return _atomic_write_sidecar(verified, f"{run_id}.json", text)


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
    reservation: Optional[RunArtifactReservation] = None,
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
    supplied_run_id = run_id is not None
    if supplied_run_id:
        run_id = _validated_run_id(run_id)

    async_mode = inference_mode == "async_queue"
    if async_mode:
        if type(reservation) is not RunArtifactReservation:
            raise ValueError(
                "async_queue results require a valid run artifact reservation"
            )
        if not supplied_run_id:
            raise ValueError("async_queue results require reservation run_id")
    elif reservation is not None:
        raise ValueError("RunArtifactReservation is only valid for async_queue")

    if results_path is None:
        results_path = DEFAULT_RESULTS_PATH

    results_path = Path(results_path)
    if not async_mode:
        results_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 메타데이터 행 구성
    row = {
        "run_id": "",
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

    if async_mode:
        with verify_reservation(
            reservation,
            run_id,
            results_path=results_path,
        ) as verified:
            return _save_reserved_result(verified, row)

    with _csv_lock(results_path):
        existing_columns, existing_rows = _read_csv_structure(results_path)
        existing_run_ids = set()
        if "run_id" in existing_columns:
            run_id_index = existing_columns.index("run_id")
            existing_run_ids = {
                existing_row[run_id_index] for existing_row in existing_rows
            }
        if supplied_run_id:
            if run_id in existing_run_ids:
                raise ValueError(f"run_id already exists: {run_id}")
        else:
            while True:
                candidate = _validated_run_id(create_run_id())
                if candidate not in existing_run_ids:
                    run_id = candidate
                    break
        row["run_id"] = run_id

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
            added_columns = len(all_columns) - len(existing_columns)
            migrated_rows = [
                [*existing_row, *([""] * added_columns)]
                for existing_row in existing_rows
            ]
            _atomic_write_csv(
                results_path,
                all_columns,
                [
                    *migrated_rows,
                    [row.get(column, "") for column in all_columns],
                ],
            )
        else:
            # 컬럼 변경 없음: 단순 append
            file_exists = results_path.exists() and results_path.stat().st_size > 0
            with open(results_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(all_columns)
                writer.writerow([row.get(column, "") for column in all_columns])
                f.flush()
                os.fsync(f.fileno())

    return run_id


def _result_columns_and_rows(
    row: Dict[str, Any],
    existing_columns: List[str],
    existing_rows: List[List[str]],
) -> tuple[List[str], List[List[Any]]]:
    metric_keys = [key for key in row if key not in META_COLUMNS]
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
    added_columns = len(all_columns) - len(existing_columns)
    migrated_rows = [
        [*existing_row, *("" for _ in range(added_columns))]
        for existing_row in existing_rows
    ]
    return all_columns, [
        *migrated_rows,
        [row.get(column, "") for column in all_columns],
    ]


def _save_reserved_result(
    verified: VerifiedReservation,
    row: Dict[str, Any],
) -> str:
    root_fd = verified.root.file_descriptor
    with _csv_lock_at(root_fd, verified.results_name):
        if not reservation_binding_matches(verified):
            raise ValueError("reservation path identity changed before CSV save")
        columns, rows = _read_csv_structure_at(root_fd, verified.results_name)
        if "run_id" in columns:
            run_id_index = columns.index("run_id")
            if any(
                existing_row[run_id_index] == verified.reservation.run_id
                for existing_row in rows
            ):
                raise ValueError(
                    f"run_id already exists: {verified.reservation.run_id}"
                )
        row["run_id"] = verified.reservation.run_id
        all_columns, all_rows = _result_columns_and_rows(row, columns, rows)
        consume_reservation(verified)
        _atomic_write_csv_at(
            verified.root,
            verified.results_name,
            all_columns,
            all_rows,
        )
    return verified.reservation.run_id


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
        columns, positional_rows = _read_csv_structure(results_path)
        rows = [dict(zip(columns, row)) for row in positional_rows]

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
        columns, rows = _read_csv_structure(results_path)
        if "run_id" not in columns:
            return False
        run_id_index = columns.index("run_id")

        original_count = len(rows)
        rows = [row for row in rows if row[run_id_index] != run_id]

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
        columns, rows = _read_csv_structure(results_path)
        if "run_id" not in columns:
            return None
        run_id_index = columns.index("run_id")
        for row in reversed(rows):
            if row[run_id_index] == run_id:
                return dict(zip(columns, row))

    return None


def _read_csv_structure(path: Path) -> tuple[List[str], List[List[str]]]:
    if not path.exists() or path.stat().st_size == 0:
        return [], []
    try:
        with open(path, "r", newline="", encoding="utf-8") as handle:
            records = list(csv.reader(handle, strict=True))
    except csv.Error as exc:
        raise ValueError("malformed CSV: invalid quoting") from exc
    return _validate_csv_records(records)


def _read_csv_structure_at(
    root_fd: int,
    results_name: str,
) -> tuple[List[str], List[List[str]]]:
    try:
        file_fd = os.open(
            results_name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
    except FileNotFoundError:
        return [], []
    opened = os.fstat(file_fd)
    if not stat.S_ISREG(opened.st_mode):
        os.close(file_fd)
        raise ValueError("results_path must be a regular CSV file")
    if opened.st_size == 0:
        os.close(file_fd)
        return [], []
    try:
        with os.fdopen(file_fd, "r", newline="", encoding="utf-8") as handle:
            records = list(csv.reader(handle, strict=True))
    except csv.Error as exc:
        raise ValueError("malformed CSV: invalid quoting") from exc
    return _validate_csv_records(records)


def _validate_csv_records(
    records: List[List[str]],
) -> tuple[List[str], List[List[str]]]:
    if not records:
        return [], []
    columns = records[0]
    if not columns or any(column == "" for column in columns):
        raise ValueError("malformed CSV: header columns must be non-empty")
    if len(set(columns)) != len(columns):
        raise ValueError("malformed CSV: header columns must be unique")
    width = len(columns)
    rows = records[1:]
    for row_number, row in enumerate(rows, start=2):
        if len(row) != width:
            raise ValueError(
                f"malformed CSV: row {row_number} has {len(row)} cells; "
                f"expected {width}"
            )
    return columns, rows


def _atomic_write_csv(
    path: Path,
    columns: List[str],
    rows: List[List[Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = (
        stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    )
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    handle = None
    primary = None
    try:
        os.fchmod(fd, file_mode)
        handle = os.fdopen(fd, "w", newline="", encoding="utf-8")
        fd = -1
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_parent(path)
    except BaseException as exc:
        primary = exc
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except BaseException as exc:
                if primary is None:
                    primary = exc
                else:
                    _attach_secondary_error(primary, "close_descriptor", exc)
        if handle is not None:
            try:
                handle.close()
            except BaseException as exc:
                if primary is None:
                    primary = exc
                else:
                    _attach_secondary_error(primary, "close_file", exc)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            except BaseException as exc:
                if primary is None:
                    primary = exc
                else:
                    _attach_secondary_error(
                        primary,
                        "cleanup_temp",
                        exc,
                        temporary_file_may_remain=True,
                    )
    if primary is not None:
        raise primary


def _atomic_write_csv_at(
    root,
    results_name: str,
    columns: List[str],
    rows: List[List[Any]],
) -> None:
    try:
        existing = os.stat(
            results_name,
            dir_fd=root.file_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        file_mode = 0o644
    else:
        if not stat.S_ISREG(existing.st_mode):
            raise ValueError("results_path must be a regular CSV file")
        file_mode = stat.S_IMODE(existing.st_mode)

    temporary_name = f".{results_name}.{uuid.uuid4().hex}.tmp"
    file_fd = None
    handle = None
    primary = None
    try:
        file_fd = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            file_mode,
            dir_fd=root.file_descriptor,
        )
        os.fchmod(file_fd, file_mode)
        handle = os.fdopen(file_fd, "w", newline="", encoding="utf-8")
        file_fd = None
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        os.replace(
            temporary_name,
            results_name,
            src_dir_fd=root.file_descriptor,
            dst_dir_fd=root.file_descriptor,
        )
        temporary_name = None
        os.fsync(root.file_descriptor)
    except BaseException as exc:
        primary = exc
    finally:
        if handle is not None:
            try:
                handle.close()
            except BaseException as exc:
                if primary is None:
                    primary = exc
                else:
                    _attach_secondary_error(primary, "close_file", exc)
        if file_fd is not None:
            try:
                os.close(file_fd)
            except BaseException as exc:
                if primary is None:
                    primary = exc
                else:
                    _attach_secondary_error(primary, "close_descriptor", exc)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=root.file_descriptor)
            except FileNotFoundError:
                pass
            except BaseException as exc:
                if primary is None:
                    primary = exc
                else:
                    _attach_secondary_error(
                        primary,
                        "cleanup_temp",
                        exc,
                        temporary_file_may_remain=True,
                        temporary_path=str(root.path / temporary_name),
                    )
    if primary is not None:
        raise primary
