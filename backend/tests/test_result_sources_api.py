"""등록 CSV, 원본 메타데이터, details/trace 접근 경계를 검증한다."""

import asyncio
import csv
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "framework" / "src"))

from app.main import app


def write_csv(path, model, *, details="", trace=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "run_id", "model_name", "inference_mode", "async_run_status",
            "details_path", "request_trace_path", "Average Latency (ms)", "unknown_metric",
        ])
        writer.writeheader()
        writer.writerow({
            "run_id": "same-run", "model_name": model, "inference_mode": "async_queue",
            "async_run_status": "valid", "details_path": details,
            "request_trace_path": trace, "Average Latency (ms)": "NaN",
            "unknown_metric": "vendor-specific",
        })


def test_two_sources_keep_same_run_id_separate_and_preserve_raw_values(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("WEB_RESULTS_ROOTS", str(tmp_path))
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    write_csv(first, "resnet50")
    write_csv(second, "yolov5m")

    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            a = (await client.post("/api/results/sources", json={"path": str(first)})).json()
            b = (await client.post("/api/results/sources", json={"path": str(second)})).json()
            sources = (await client.get("/api/results/sources")).json()["sources"]
            assert {source["source_id"] for source in sources} >= {"default", a["source_id"], b["source_id"]}
            assert a["source_id"] != b["source_id"]
            result_a = (await client.get(f"/api/results/runs/{a['source_id']}/same-run")).json()
            result_b = (await client.get(f"/api/results/runs/{b['source_id']}/same-run")).json()
            assert result_a["model_name"] == "resnet50"
            assert result_b["model_name"] == "yolov5m"
            assert result_a["metadata"]["async_run_status"] == "valid"
            assert result_a["metrics"]["Average Latency (ms)"] is None
            assert result_a["raw_row"]["Average Latency (ms)"] == "NaN"
            assert result_a["metrics"]["unknown_metric"] == "vendor-specific"
            listed = (await client.get("/api/results", params={"source_id": b["source_id"]})).json()
            assert listed["total"] == 1
            assert listed["results"][0]["model_name"] == "yolov5m"

    asyncio.run(call())


def test_details_trace_and_csv_only_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("WEB_RESULTS_ROOTS", str(tmp_path))
    csv_path = tmp_path / "results.csv"
    details = tmp_path / "details" / "same-run.json"
    trace = tmp_path / "traces" / "same-run.jsonl"
    details.parent.mkdir()
    trace.parent.mkdir()
    details.write_text(json.dumps({"run_id": "same-run", "status": "valid"}))
    trace.write_text('{"request_id":1}\n{"request_id":2}\n')
    write_csv(csv_path, "resnet50", details="details/same-run.json",
              trace="traces/same-run.jsonl")

    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            source = (await client.post("/api/results/sources", json={"path": str(csv_path)})).json()["source_id"]
            detail = await client.get(f"/api/results/runs/{source}/same-run/artifacts/details")
            assert detail.status_code == 200
            assert detail.json()["status"] == "valid"
            trace_response = await client.get(f"/api/results/runs/{source}/same-run/trace", params={"offset": 0, "limit": 1})
            assert trace_response.status_code == 200
            assert trace_response.json()["lines"] == [{"request_id": 1}]
            assert trace_response.json()["next_offset"] == 1
            deleted = await client.delete(f"/api/results/runs/{source}/same-run")
            assert deleted.status_code == 200
            assert details.exists() and trace.exists()
            assert (await client.get(f"/api/results/runs/{source}/same-run")).status_code == 404

    asyncio.run(call())


def test_source_registration_rejects_path_outside_root(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.csv"
    write_csv(outside, "resnet50")
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("WEB_RESULTS_ROOTS", str(allowed))

    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/api/results/sources", json={"path": str(outside)})

    assert asyncio.run(call()).status_code == 400


def test_artifact_reader_rejects_traversal_symlink_and_wrong_run_id(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("WEB_RESULTS_ROOTS", str(tmp_path))
    csv_path = tmp_path / "results.csv"
    write_csv(csv_path, "resnet50", details="../outside.json")

    async def read(source):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.get(f"/api/results/runs/{source}/same-run/artifacts/details")

    async def register():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return (await client.post("/api/results/sources", json={"path": str(csv_path)})).json()["source_id"]

    source = asyncio.run(register())
    assert asyncio.run(read(source)).status_code == 400
    details_dir = tmp_path / "details"
    details_dir.mkdir()
    details = details_dir / "same-run.json"
    outside = tmp_path / "other.json"
    outside.write_text(json.dumps({"run_id": "same-run"}))
    details.symlink_to(outside)
    write_csv(csv_path, "resnet50", details="details/same-run.json")
    assert asyncio.run(read(source)).status_code == 400
    details.unlink()
    details.write_text(json.dumps({"run_id": "other-run"}))
    assert asyncio.run(read(source)).status_code == 400


def test_unknown_source_on_artifact_route_is_404():
    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.get("/api/results/runs/not-registered/run-1/artifacts/details")

    assert asyncio.run(call()).status_code == 404


def test_preview_can_select_registered_result_source(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("WEB_RESULTS_ROOTS", str(tmp_path))
    csv_path = tmp_path / "selected.csv"
    write_csv(csv_path, "resnet50")

    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            source = (await client.post("/api/results/sources", json={"path": str(csv_path)})).json()["source_id"]
            return await client.post("/api/benchmark/preview", json={
                "model": "resnet50", "target_id": "cpu", "result_source_id": source,
            })

    response = asyncio.run(call())
    assert response.status_code == 200
    assert response.json()["errors"] == []
    assert response.json()["argv"][-2:] == ["--results-path", str(csv_path)]
