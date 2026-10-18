"""웹 API가 미리보기와 같은 argv를 실제 작업으로 실행하는지 확인한다."""

import asyncio
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "framework" / "src"))

from app.main import app


def test_job_api_runs_registered_host_command_and_exposes_log(tmp_path, monkeypatch):
    runner = tmp_path / "runner.py"
    runner.write_text(f"#!{sys.executable}\n" + """
import csv, os, sys
args = sys.argv
path = args[args.index('--results-path') + 1]
print('RUN_ID_RESERVED=api-run', flush=True)
print('FLAG=' + os.environ.get('WEB_TEST_FLAG', 'missing'), flush=True)
with open(path, 'w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['run_id', 'model_name'])
    writer.writerow(['api-run', 'resnet50'])
print('RUN_ID=api-run', flush=True)
""")
    runner.chmod(0o755)
    config = tmp_path / "runners.json"
    config.write_text(json.dumps({"profiles": [{
        "id": "test", "python": str(runner), "env": {"WEB_TEST_FLAG": "flagvalue"},
    }]}))
    monkeypatch.setenv("WEB_RUNNERS_FILE", str(config))
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("WEB_RESULTS_ROOTS", str(tmp_path))
    results = tmp_path / "run.csv"

    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            create = await client.post("/api/benchmark/jobs", json={
                "model": "resnet50", "target_id": "cpu", "results_path": str(results),
            })
            assert create.status_code == 200, create.text
            job_id = create.json()["job_id"]
            for _ in range(100):
                status = (await client.get(f"/api/benchmark/jobs/{job_id}")).json()
                if status["status"] == "completed":
                    break
                await asyncio.sleep(0.03)
            assert status["status"] == "completed", status
            assert status["run_id"] == "api-run"
            assert status["elapsed_sec"] is not None and status["elapsed_sec"] >= 0
            assert status["result_present"] is True
            assert status["result_source_id"]
            linked = await client.get(f"/api/results/runs/{status['result_source_id']}/api-run")
            assert linked.status_code == 200
            log = (await client.get(f"/api/benchmark/jobs/{job_id}/log")).json()
            assert "RUN_ID=api-run" in log["text"]
            assert "FLAG=flagvalue" in log["text"]
            jobs = (await client.get("/api/benchmark/jobs")).json()
            assert any(job["job_id"] == job_id for job in jobs["jobs"])
            legacy = await client.post("/api/benchmark/run", json={
                "model": "resnet50", "target_id": "cpu", "results_path": str(results),
            })
            assert legacy.status_code == 200, legacy.text
            assert legacy.json()["job_id"] != job_id

    try:
        asyncio.run(call())
    finally:
        from app.services.web_job_service import shutdown_all_jobs
        shutdown_all_jobs()
