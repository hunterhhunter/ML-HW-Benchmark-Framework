from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from ..schemas.result import (
    BenchmarkResultResponse,
    BenchmarkResultListResponse,
    DeleteResultResponse,
)
from ..services.result_service import (
    list_results,
    get_result_by_id,
    delete_result_by_id,
)

router = APIRouter(prefix="/api/results", tags=["results"])


@router.get("", response_model=BenchmarkResultListResponse)
async def get_results(
    model_name: Optional[str] = Query(None, description="모델 이름으로 필터링"),
    task: Optional[str] = Query(None, description="태스크로 필터링"),
    backend: Optional[str] = Query(None, description="백엔드로 필터링"),
    limit: Optional[int] = Query(None, description="최대 반환 건수"),
):
    """벤치마크 결과 목록을 조회한다."""
    return list_results(
        model_name=model_name,
        task=task,
        backend=backend,
        limit=limit,
    )


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
