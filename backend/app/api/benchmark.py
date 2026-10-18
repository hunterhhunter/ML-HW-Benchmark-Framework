from fastapi import APIRouter, HTTPException, Query

from ..schemas.benchmark import (
    ProfileListResponse,
    ModelProfileResponse,
    TargetListResponse,
    TargetResponse,
    BenchmarkRunRequest,
    BenchmarkJobResponse,
)
from ..services.benchmark_service import (
    get_available_profiles,
    get_available_targets,
)
from ..services.web_job_service import (
    start_benchmark, get_job_status, cancel_benchmark, get_job_log, list_jobs,
)
from ..services.command_builder import SUPPORTED_CLI_FLAGS
from ..services.preflight_service import preview_benchmark
from ..services.runner_profiles import load_runner_profiles

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])


@router.get("/catalog")
async def catalog():
    targets = [
        {**target, "registered": True, "hardware_verified": False}
        for target in get_available_targets()
    ]
    return {
        "profiles": [
            {key: value for key, value in profile.items() if key != "backends"}
            for profile in get_available_profiles()
        ],
        "targets": targets,
        "runner_profiles": [profile.public() for profile in load_runner_profiles()],
        "cli_options": list(SUPPORTED_CLI_FLAGS),
    }


@router.post("/preview")
async def preview(request: BenchmarkRunRequest):
    return preview_benchmark(request)


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
    """기존 API 주소. 새 영속 작업 queue로 위임한다."""
    try:
        return BenchmarkJobResponse(**start_benchmark(request))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OverflowError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


@router.post("/jobs")
async def create_job(request: BenchmarkRunRequest):
    try:
        return start_benchmark(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OverflowError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


@router.get("/jobs")
async def get_jobs(status: str | None = None, limit: int = Query(50, ge=1, le=200),
                   offset: int = Query(0, ge=0)):
    return list_jobs(status=status, limit=limit, offset=offset)


@router.get("/jobs/{job_id}")
async def get_benchmark_status(job_id: str):
    """벤치마크 작업 상태를 조회한다."""
    status = get_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"작업을 찾을 수 없습니다: {job_id}")
    return status


@router.get("/jobs/{job_id}/log")
async def benchmark_log(job_id: str, offset: int = Query(0, ge=0),
                        limit: int = Query(65536, ge=1, le=65536)):
    log = get_job_log(job_id, offset=offset, limit=limit)
    if log is None:
        raise HTTPException(status_code=404, detail=f"작업을 찾을 수 없습니다: {job_id}")
    return log


@router.post("/jobs/{job_id}/cancel")
async def cancel_benchmark_job(job_id: str):
    """진행 중인 벤치마크 작업을 중단한다."""
    result = cancel_benchmark(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"작업을 찾을 수 없습니다: {job_id}")
    return result
