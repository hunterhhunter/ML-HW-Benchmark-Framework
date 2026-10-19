"""CSV가 가리키는 async details와 trace의 제한된 읽기."""

import itertools
import json
from pathlib import Path, PurePosixPath


KINDS = {
    "details": ("details_path", ".json", "details"),
    "failure_details": ("failure_details_path", ".failure.json", "details"),
    "trace": ("request_trace_path", ".jsonl", "traces"),
}


def artifact_path(csv_path: Path, row: dict, kind: str) -> Path | None:
    field, suffix, directory = KINDS[kind]
    raw = row.get(field)
    if not raw:
        return None
    relative = PurePosixPath(raw)
    run_id = row.get("run_id", "")
    expected = f"{run_id}{suffix}"
    if relative.is_absolute() or ".." in relative.parts or "\\" in raw:
        raise ValueError("부속 파일 경로가 CSV 범위를 벗어납니다")
    parent = csv_path.parent.resolve()
    if len(relative.parts) == 2 and relative.parts[0] == directory:
        candidate = parent / relative
    elif (len(relative.parts) == 3 and relative.parts[0] == parent.name
          and relative.parts[1] == directory):
        candidate = parent.parent / relative
    else:
        raise ValueError("부속 파일 경로와 run ID가 일치하지 않습니다")
    if relative.name != expected:
        raise ValueError("부속 파일 경로와 run ID가 일치하지 않습니다")
    if candidate.parent.is_symlink() or candidate.is_symlink():
        raise ValueError("부속 파일 symlink는 열 수 없습니다")
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(parent):
        raise ValueError("부속 파일이 결과 디렉터리 밖에 있습니다")
    return resolved if resolved.is_file() else None


def read_json_artifact(csv_path: Path, row: dict, kind: str) -> dict | None:
    if kind not in {"details", "failure_details"}:
        raise ValueError("지원하지 않는 부속 파일 종류")
    path = artifact_path(csv_path, row, kind)
    if path is None:
        return None
    if path.stat().st_size > 2_000_000:
        raise ValueError("부속 JSON 파일이 크기 제한을 초과했습니다")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("run_id") != row.get("run_id"):
        raise ValueError("부속 JSON의 run ID가 결과 행과 다릅니다")
    return data


def read_trace(csv_path: Path, row: dict, *, offset: int, limit: int) -> dict | None:
    path = artifact_path(csv_path, row, "trace")
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as stream:
        lines = [json.loads(line) for line in itertools.islice(stream, offset, offset + limit)]
    return {"lines": lines, "next_offset": offset + len(lines)}
