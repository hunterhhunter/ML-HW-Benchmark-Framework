from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from pydantic import BaseModel

from ..schemas.result import (
    BenchmarkResultResponse,
    BenchmarkResultListResponse,
    DeleteResultResponse,
)
from ..services.result_service import (
    list_results,
    get_result_by_id,
    delete_result_by_id,
    list_source_results,
    get_source_result,
    delete_source_result,
)
from ..services.result_sources import ResultSources
from ..services.artifact_reader import read_json_artifact, read_trace

router = APIRouter(prefix="/api/results", tags=["results"])


class RegisterSourceRequest(BaseModel):
    path: str


@router.get("")
async def get_results(
    source_id: Optional[str] = None,
    model_name: Optional[str] = Query(None, description="모델 이름으로 필터링"),
    task: Optional[str] = Query(None, description="태스크로 필터링"),
    backend: Optional[str] = Query(None, description="백엔드로 필터링"),
    limit: Optional[int] = Query(None, description="최대 반환 건수"),
    offset: int = Query(0, ge=0),
    target: Optional[str] = None,
    mode: Optional[str] = None,
    status: Optional[str] = None,
):
    """벤치마크 결과 목록을 조회한다."""
    if source_id is not None:
        try:
            return list_source_results(
                source_id, model_name=model_name, task=task, backend=backend,
                target=target, mode=mode, status=status,
                limit=min(limit or 100, 500), offset=offset,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="결과 source를 찾을 수 없습니다") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return list_results(
        model_name=model_name,
        task=task,
        backend=backend,
        limit=limit,
    )


@router.get("/sources")
async def list_sources():
    return {"sources": ResultSources().list()}


@router.post("/sources")
async def register_source(request: RegisterSourceRequest):
    try:
        return ResultSources().ensure(request.path, must_exist=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/runs/{source_id}/{run_id}")
async def get_source_run(source_id: str, run_id: str):
    try:
        value = get_source_result(source_id, run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="결과 source를 찾을 수 없습니다") from exc
    if value is None:
        raise HTTPException(status_code=404, detail="결과 행을 찾을 수 없습니다")
    return value[0]


@router.get("/runs/{source_id}/{run_id}/artifacts/{kind}")
async def get_run_artifact(source_id: str, run_id: str, kind: str):
    if kind not in {"details", "failure_details"}:
        raise HTTPException(status_code=404, detail="지원하지 않는 부속 파일 종류")
    try:
        value = get_source_result(source_id, run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="결과 source를 찾을 수 없습니다") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if value is None:
        raise HTTPException(status_code=404, detail="결과 행을 찾을 수 없습니다")
    result, csv_path = value
    try:
        data = read_json_artifact(csv_path, result["raw_row"], kind)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if data is None:
        raise HTTPException(status_code=404, detail="부속 파일을 찾을 수 없습니다")
    return data


@router.get("/runs/{source_id}/{run_id}/trace")
async def get_run_trace(source_id: str, run_id: str, offset: int = Query(0, ge=0),
                        limit: int = Query(100, ge=1, le=1000)):
    try:
        value = get_source_result(source_id, run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="결과 source를 찾을 수 없습니다") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if value is None:
        raise HTTPException(status_code=404, detail="결과 행을 찾을 수 없습니다")
    result, csv_path = value
    try:
        data = read_trace(csv_path, result["raw_row"], offset=offset, limit=limit)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if data is None:
        raise HTTPException(status_code=404, detail="trace 파일을 찾을 수 없습니다")
    return data


@router.delete("/runs/{source_id}/{run_id}", response_model=DeleteResultResponse)
async def delete_source_run(source_id: str, run_id: str):
    try:
        removed = delete_source_result(source_id, run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="결과 source를 찾을 수 없습니다") from exc
    if not removed:
        raise HTTPException(status_code=404, detail="결과 행을 찾을 수 없습니다")
    return DeleteResultResponse(success=True, message=f"CSV 행 삭제 완료: {run_id}")


@router.get("/{run_id}", response_model=BenchmarkResultResponse)
async def get_result(run_id: str):
    """특정 run_id의 벤치마크 결과를 조회한다."""
    result = get_result_by_id(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"결과를 찾을 수 없습니다: {run_id}")
    return result


@router.delete("/{run_id}", response_model=DeleteResultResponse)
async def delete_result(run_id: str):
    """특정 run_id의 벤치마크 결과를 삭제한다."""
    success = delete_result_by_id(run_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"결과를 찾을 수 없습니다: {run_id}")
    return DeleteResultResponse(success=True, message=f"결과 삭제 완료: {run_id}")
