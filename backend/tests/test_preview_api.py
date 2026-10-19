"""서버에 등록된 실행 환경과 부작용 없는 CLI 미리보기를 검증한다."""

import asyncio
import importlib
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "framework" / "src"))

from app.main import app


def select_runner_profile(*args):
    return importlib.import_module("app.services.runner_profiles").select_runner_profile(*args)


def test_runner_selection_prefers_exact_target_and_model(tmp_path, monkeypatch):
    config = tmp_path / "runners.json"
    config.write_text(json.dumps({"profiles": [
        {"id": "default", "python": sys.executable},
        {"id": "furiosa", "python": sys.executable, "targets": ["furiosa-rngd"]},
        {"id": "furiosa-llama", "python": sys.executable,
         "targets": ["furiosa-rngd"], "models": ["llama-3.2-3b"]},
    ]}))
    monkeypatch.setenv("WEB_RUNNERS_FILE", str(config))

    assert select_runner_profile("furiosa-rngd", "llama-3.2-3b").id == "furiosa-llama"
    assert select_runner_profile("furiosa-rngd", "bert-base-uncased").id == "furiosa"
    assert select_runner_profile("cpu", "resnet50").id == "default"


def test_target_runner_precedes_model_only_runner(tmp_path, monkeypatch):
    config = tmp_path / "runners.json"
    config.write_text(json.dumps({"profiles": [
        {"id": "model-only", "python": sys.executable, "models": ["resnet50"]},
        {"id": "target-only", "python": sys.executable, "targets": ["cpu"]},
        {"id": "default", "python": sys.executable},
    ]}))
    monkeypatch.setenv("WEB_RUNNERS_FILE", str(config))
    assert select_runner_profile("cpu", "resnet50").id == "target-only"


def test_preview_reports_actual_argv_and_never_starts_a_job(tmp_path, monkeypatch):
    asset = tmp_path / "resnet50.onnx"
    asset.write_bytes(b"fixture")
    monkeypatch.setenv("WEB_ASSET_ROOTS", str(tmp_path))
    monkeypatch.setenv("WEB_RESULTS_ROOTS", str(tmp_path))
    monkeypatch.delenv("WEB_RUNNERS_FILE", raising=False)

    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/api/benchmark/preview", json={
                "model": "resnet50", "target_id": "cpu",
                "onnx_path": str(asset), "results_path": str(tmp_path / "run.csv"),
            })

    response = asyncio.run(call())
    assert response.status_code == 200
    body = response.json()
    assert body["errors"] == []
    assert body["argv"][-8:] == [
        "--model", "resnet50", "--target", "cpu", "--onnx", str(asset),
        "--results-path", str(tmp_path / "run.csv"),
    ]
    assert body["runner_profile"]["id"] == "default"


def test_preview_rejects_asset_outside_registered_root(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    other = tmp_path / "other.onnx"
    other.write_bytes(b"fixture")
    monkeypatch.setenv("WEB_ASSET_ROOTS", str(allowed))

    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/api/benchmark/preview", json={
                "model": "resnet50", "target_id": "cpu", "onnx_path": str(other),
            })

    body = asyncio.run(call()).json()
    assert any("허용된 자산 경로" in error for error in body["errors"])


def test_catalog_exposes_registry_without_claiming_device_verification():
    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.get("/api/benchmark/catalog")

    response = asyncio.run(call())
    assert response.status_code == 200
    body = response.json()
    cpu = next(item for item in body["targets"] if item["target_id"] == "cpu")
    assert cpu["registered"] is True
    assert cpu["hardware_verified"] is False
    assert "backends" not in body["profiles"][0]


def test_removed_model_profiles_are_not_exposed_by_web_catalog():
    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            catalog = await client.get("/api/benchmark/catalog")
            profiles = await client.get("/api/benchmark/profiles")
            preview = await client.post("/api/benchmark/preview", json={
                "model": "yolov8s-seg", "target_id": "cpu",
            })
            return catalog, profiles, preview

    catalog, profiles, preview = asyncio.run(call())
    assert catalog.status_code == profiles.status_code == preview.status_code == 200
    removed = {
        "yolov8m", "yolov8m-seg", "yolov8m-pose", "yolov8m_seg",
        "yolov8-seg-m", "yolov8m_pose", "yolov8-pose-m", "yolov8s-seg",
        "patchtst-fm-r1", "patchtst-etth1",
    }
    for response in (catalog, profiles):
        names = {item["model_name"] for item in response.json()["profiles"]}
        assert removed.isdisjoint(names)
        assert {"resnet50", "yolov5m", "yolov8s-pose", "ttm-r2"} <= names
    assert "등록되지 않은 모델: yolov8s-seg" in preview.json()["errors"]


def test_preview_reports_known_target_restrictions():
    async def call(payload):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return (await client.post("/api/benchmark/preview", json=payload)).json()

    rbln = asyncio.run(call({"model": "resnet50", "target_id": "rbln-static", "batch_size": 2}))
    assert any("batch size" in error for error in rbln["errors"])
    furiosa = asyncio.run(call({"model": "resnet50", "target_id": "furiosa-rngd-torch", "compile": False}))
    assert any("--no-compile" in error for error in furiosa["errors"])


def test_explicit_runner_must_match_target_and_model(tmp_path, monkeypatch):
    config = tmp_path / "runners.json"
    config.write_text(json.dumps({"profiles": [
        {"id": "cpu-only", "python": sys.executable, "targets": ["cpu"]},
        {"id": "deepx-only", "python": sys.executable, "targets": ["deepx"]},
    ]}))
    monkeypatch.setenv("WEB_RUNNERS_FILE", str(config))
    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/api/benchmark/preview", json={
                "model": "resnet50", "target_id": "cpu", "runner_profile_id": "deepx-only",
            })
    body = asyncio.run(call()).json()
    assert any("Python 실행 환경" in error for error in body["errors"])


def test_model_path_kind_distinguishes_local_path_from_hub_id(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_ASSET_ROOTS", str(tmp_path))
    async def call(model_path, kind):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return (await client.post("/api/benchmark/preview", json={
                "model": "resnet50", "target_id": "cpu",
                "model_path": model_path, "model_path_kind": kind,
            })).json()

    missing_local = asyncio.run(call(str(tmp_path / "missing"), "local"))
    assert any("model_path: 파일" in error for error in missing_local["errors"])
    hub = asyncio.run(call("vendor/model", "hf_id"))
    assert hub["asset_checks"]["model_path"]["kind"] == "huggingface_id"
    assert not any("model_path:" in error for error in hub["errors"])
