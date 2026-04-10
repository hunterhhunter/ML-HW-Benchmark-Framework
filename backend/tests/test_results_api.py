"""
Backend 결과 API 테스트

FastAPI TestClient를 사용하여 /api/results 엔드포인트를 검증합니다.
임시 CSV 파일을 사용하므로 실제 결과 파일에 영향을 주지 않습니다.

실행 방법:
    cd backend && python -m pytest tests/test_results_api.py -v
"""

import sys
import os

# backend 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# framework/src도 sys.path에 추가 (result_store 임포트용)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FRAMEWORK_SRC = os.path.join(_PROJECT_ROOT, "framework", "src")
if _FRAMEWORK_SRC not in sys.path:
    sys.path.insert(0, _FRAMEWORK_SRC)

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from pathlib import Path

from app.main import app
from core.result_store import save_result, DEFAULT_RESULTS_PATH


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def tmp_csv(tmp_path):
    """임시 CSV 경로로 result_store의 기본 경로를 오버라이드한다."""
    csv_path = tmp_path / "test_results.csv"
    with patch("core.result_store.DEFAULT_RESULTS_PATH", csv_path), \
         patch("app.services.result_service.DEFAULT_RESULTS_PATH", csv_path):
        yield csv_path


@pytest.fixture
def seeded_csv(tmp_csv):
    """테스트 데이터가 미리 들어있는 CSV를 제공한다."""
    ids = []
    ids.append(save_result(
        metrics={"Top-1 Accuracy": 75.42, "Average Latency (ms)": 12.34},
        model_name="resnet50",
        task="IMAGE_CLASSIFICATION",
        backend="onnxruntime",
        device="cuda",
        batch_size=1,
        warmup_runs=2,
        results_path=tmp_csv,
    ))
    ids.append(save_result(
        metrics={"accuracy": 92.5, "Average Latency (ms)": 8.45},
        model_name="bert-base-uncased",
        task="NLP_CLASSIFICATION",
        backend="onnxruntime",
        device="cpu",
        batch_size=1,
        warmup_runs=2,
        results_path=tmp_csv,
    ))
    ids.append(save_result(
        metrics={"MAE": 0.123, "MSE": 0.234, "RMSE": 0.484},
        model_name="patchtst-fm-r1",
        task="TIME_SERIES_FORECASTING",
        backend="onnxruntime",
        device="cuda",
        batch_size=4,
        warmup_runs=2,
        results_path=tmp_csv,
    ))
    return ids


# ------------------------------------------------------------------
# GET /api/results
# ------------------------------------------------------------------

class TestGetResults:
    def test_empty_results(self, client, tmp_csv):
        """결과가 없으면 빈 리스트를 반환한다."""
        resp = client.get("/api/results")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["results"] == []

    def test_list_all_results(self, client, seeded_csv):
        """전체 결과를 조회한다."""
        resp = client.get("/api/results")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3

    def test_filter_by_model_name(self, client, seeded_csv):
        """model_name 필터가 동작한다."""
        resp = client.get("/api/results", params={"model_name": "resnet50"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["model_name"] == "resnet50"

    def test_filter_by_task(self, client, seeded_csv):
        """task 필터가 동작한다."""
        resp = client.get("/api/results", params={"task": "TIME_SERIES_FORECASTING"})
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["model_name"] == "patchtst-fm-r1"

    def test_filter_by_backend(self, client, seeded_csv):
        """backend 필터가 동작한다."""
        resp = client.get("/api/results", params={"backend": "onnxruntime"})
        data = resp.json()
        assert data["total"] == 3

    def test_limit(self, client, seeded_csv):
        """limit 파라미터가 동작한다."""
        resp = client.get("/api/results", params={"limit": 2})
        data = resp.json()
        assert data["total"] == 2

    def test_metrics_are_separated(self, client, seeded_csv):
        """메트릭이 별도 딕셔너리로 분리된다."""
        resp = client.get("/api/results", params={"model_name": "resnet50"})
        result = resp.json()["results"][0]
        assert "metrics" in result
        assert "Top-1 Accuracy" in result["metrics"]
        assert result["metrics"]["Top-1 Accuracy"] == 75.42


# ------------------------------------------------------------------
# GET /api/results/{run_id}
# ------------------------------------------------------------------

class TestGetResultById:
    def test_get_existing(self, client, seeded_csv):
        """존재하는 run_id로 결과를 조회한다."""
        run_id = seeded_csv[0]
        resp = client.get(f"/api/results/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run_id
        assert data["model_name"] == "resnet50"

    def test_get_nonexistent(self, client, tmp_csv):
        """존재하지 않는 run_id는 404를 반환한다."""
        resp = client.get("/api/results/nonexistent")
        assert resp.status_code == 404


# ------------------------------------------------------------------
# DELETE /api/results/{run_id}
# ------------------------------------------------------------------

class TestDeleteResult:
    def test_delete_existing(self, client, seeded_csv):
        """존재하는 run_id를 삭제한다."""
        run_id = seeded_csv[0]
        resp = client.delete(f"/api/results/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

        # 삭제 후 조회 시 404
        resp = client.get(f"/api/results/{run_id}")
        assert resp.status_code == 404

    def test_delete_nonexistent(self, client, tmp_csv):
        """존재하지 않는 run_id 삭제 시 404를 반환한다."""
        resp = client.delete("/api/results/nonexistent")
        assert resp.status_code == 404

    def test_delete_preserves_others(self, client, seeded_csv):
        """삭제 후 나머지 결과는 보존된다."""
        client.delete(f"/api/results/{seeded_csv[0]}")
        resp = client.get("/api/results")
        assert resp.json()["total"] == 2


# ------------------------------------------------------------------
# GET /api/health (기존 엔드포인트 회귀 테스트)
# ------------------------------------------------------------------

class TestHealth:
    def test_health_check(self, client):
        """기존 health 엔드포인트가 정상 동작한다."""
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
