from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .api.results import router as results_router
from .api.benchmark import router as benchmark_router
from .services.benchmark_service import shutdown_all_jobs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST_DIR = Path(
    os.getenv("FRONTEND_DIST_DIR", PROJECT_ROOT / "frontend" / "dist")
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # 서버 종료 시 진행 중인 벤치마크 subprocess를 정리해 고아 GPU 프로세스 방지
    shutdown_all_jobs()


app = FastAPI(
    title="ML HW Benchmark API",
    description="ML 하드웨어 벤치마크 프레임워크 웹 API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://localhost:8000",
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(results_router)
app.include_router(benchmark_router)


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


def _frontend_file(path: str) -> Path:
    requested = (FRONTEND_DIST_DIR / path).resolve()
    dist_root = FRONTEND_DIST_DIR.resolve()
    if requested == dist_root or dist_root in requested.parents:
        return requested
    raise HTTPException(status_code=404, detail="Not found")


@app.get("/", include_in_schema=False)
async def serve_frontend_root():
    index_path = FRONTEND_DIST_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend build not found")
    return FileResponse(index_path)


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API route not found")

    static_path = _frontend_file(full_path)
    if static_path.is_file():
        return FileResponse(static_path)

    index_path = FRONTEND_DIST_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend build not found")
