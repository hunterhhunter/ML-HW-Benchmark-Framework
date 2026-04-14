from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.results import router as results_router
from .api.benchmark import router as benchmark_router
from .services.benchmark_service import shutdown_all_jobs


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
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(results_router)
app.include_router(benchmark_router)


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}
