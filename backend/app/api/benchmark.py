from fastapi import APIRouter, HTTPException

from ..schemas.benchmark import (
    ProfileListResponse,
    ModelProfileResponse,
    TargetListResponse,
    TargetResponse,
    BenchmarkRunRequest,
    BenchmarkJobResponse,
    BenchmarkJobStatusResponse,
)
from ..services.benchmark_service import (
    get_available_profiles,
    get_available_targets,
    start_benchmark,
    get_job_status,
    cancel_benchmark,
)

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])


@router.get("/profiles", response_model=ProfileListResponse)
async def list_profiles():
    """사용 가능한 모델 프로필 목록을 반환한다."""
    profiles = get_available_profiles()
    return ProfileListResponse(
        profiles=[ModelProfileResponse(**p) for p in profiles]
    )


@router.get("/targets", response_model=TargetListResponse)
async def list_targets():
    """사용 가능한 benchmark target 목록을 반환한다."""
    targets = get_available_targets()
    return TargetListResponse(
        targets=[TargetResponse(**target) for target in targets]
    )


@router.post("/run", response_model=BenchmarkJobResponse)
async def run_benchmark(request: BenchmarkRunRequest):
    """벤치마크를 비동기로 실행한다."""
    profiles = get_available_profiles()
    model_names = [p["model_name"] for p in profiles]
    if request.model not in model_names:
        raise HTTPException(
            status_code=400,
            detail=f"지원되지 않는 모델: {request.model}. 사용 가능: {model_names}",
        )
    result = start_benchmark(request)
    return BenchmarkJobResponse(**result)


@router.get("/jobs/{job_id}", response_model=BenchmarkJobStatusResponse)
async def get_benchmark_status(job_id: str):
    """벤치마크 작업 상태를 조회한다."""
    status = get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"작업을 찾을 수 없습니다: {job_id}")
    return BenchmarkJobStatusResponse(**status)


@router.post("/jobs/{job_id}/cancel")
async def cancel_benchmark_job(job_id: str):
    """진행 중인 벤치마크 작업을 중단한다."""
    result = cancel_benchmark(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"작업을 찾을 수 없습니다: {job_id}")
    return result
