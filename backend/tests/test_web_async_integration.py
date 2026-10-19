"""웹 API의 async 작업에서 CSV, details, trace가 연결되는지 검사한다."""

import asyncio
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "framework" / "src"))

from app.main import app


def test_web_async_job_links_csv_details_and_trace(tmp_path, monkeypatch):
    runner = tmp_path / "runner.py"
    runner.write_text(f"#!{sys.executable}\n" + """
import csv, json, pathlib, sys
args = sys.argv
csv_path = pathlib.Path(args[args.index('--results-path') + 1])
assert args[args.index('--inference-mode') + 1] == 'async_queue'
assert args[args.index('--async-pass') + 1] == 'runtime-only'
assert '--save-request-trace' in args
print('RUN_ID_RESERVED=async-web-run', flush=True)
(csv_path.parent / 'details').mkdir(exist_ok=True)
(csv_path.parent / 'traces').mkdir(exist_ok=True)
(csv_path.parent / 'details' / 'async-web-run.json').write_text(json.dumps({
    'run_id': 'async-web-run', 'schema_version': 2, 'quality_status': 'skipped',
}))
(csv_path.parent / 'traces' / 'async-web-run.jsonl').write_text(json.dumps({'request_id': 1}) + '\\n')
with csv_path.open('w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow(['run_id', 'model_name', 'inference_mode', 'async_run_status',
                     'details_path', 'request_trace_path', 'quality_status',
                     'async_runtime_completed_samples_per_sec'])
    writer.writerow(['async-web-run', 'resnet50', 'async_queue', 'valid',
                     'details/async-web-run.json', 'traces/async-web-run.jsonl', 'skipped', '4.5'])
print('RUN_ID=async-web-run', flush=True)
""")
    runner.chmod(0o755)
    config = tmp_path / "runners.json"
    config.write_text(json.dumps({"profiles": [{"id": "test", "python": str(runner)}]}))
    monkeypatch.setenv("WEB_RUNNERS_FILE", str(config))
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("WEB_RESULTS_ROOTS", str(tmp_path))
    csv_path = tmp_path / "async.csv"

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            request = {"model": "resnet50", "target_id": "cpu", "results_path": str(csv_path),
                       "inference_mode": "async_queue", "async_pass": "runtime-only",
                       "save_request_trace": True}
            preview = await client.post("/api/benchmark/preview", json=request)
            assert preview.status_code == 200, preview.text
            assert preview.json()["errors"] == []
            created = await client.post("/api/benchmark/jobs", json=request)
            assert created.status_code == 200, created.text
            job_id = created.json()["job_id"]
            for _ in range(100):
                job = (await client.get(f"/api/benchmark/jobs/{job_id}")).json()
                if job["status"] in {"completed", "failed"}:
                    break
                await asyncio.sleep(0.03)
            assert job["status"] == "completed", job
            assert job["result_present"] is True
            source = job["result_source_id"]
            run = (await client.get(f"/api/results/runs/{source}/async-web-run")).json()
            assert run["metadata"]["async_run_status"] == "valid"
            assert run["metrics"]["quality_status"] == "skipped"
            assert run["metrics"]["async_runtime_completed_samples_per_sec"] == 4.5
            assert run["artifacts"]["details"]["available"] is True
            assert run["artifacts"]["trace"]["available"] is True
            details = await client.get(f"/api/results/runs/{source}/async-web-run/artifacts/details")
            assert details.json()["quality_status"] == "skipped"
            trace = await client.get(f"/api/results/runs/{source}/async-web-run/trace")
            assert trace.json()["lines"] == [{"request_id": 1}]

    try:
        asyncio.run(exercise())
    finally:
        from app.services.web_job_service import shutdown_all_jobs
        shutdown_all_jobs()
