import json
import subprocess
from pathlib import Path

from core.result_store import load_results
from tools.rngd_server_benchmark import (
    ServerBenchmarkConfig,
    build_vllm_argv,
    run_server_benchmark,
)


FIXTURE = Path(__file__).parent / "fixtures" / "vllm_bench_serve_0_16_0.json"
LABELS = 'model_name="llama",engine="npu:0"'


def histogram(name, count, total_sum):
    return "\n".join(
        [
            f"# TYPE {name} histogram",
            f'{name}_bucket{{{LABELS},le="0.01"}} {count}',
            f'{name}_bucket{{{LABELS},le="+Inf"}} {count}',
            f"{name}_sum{{{LABELS}}} {total_sum}",
            f"{name}_count{{{LABELS}}} {count}",
        ]
    )


def metrics_text(*, success, requests, tokens, itls):
    parts = [
        "# TYPE furiosa_llm_request_success_total counter",
        (
            "furiosa_llm_request_success_total"
            f'{{{LABELS},finished_reason="stop"}} {success}'
        ),
        histogram("furiosa_llm_request_generation_tokens", requests, tokens),
        histogram("furiosa_llm_time_to_first_token_seconds", requests, 0.01 * requests),
        histogram("furiosa_llm_inter_token_latency_seconds", itls, 0.01 * itls),
        histogram("furiosa_llm_e2e_request_latency_seconds", requests, 0.02 * requests),
    ]
    return "\n".join(parts) + "\n"


def config(tmp_path, **changes):
    values = {
        "base_url": "http://127.0.0.1:8000",
        "model": "llama",
        "input_tokens": 128,
        "output_tokens": 3,
        "num_prompts": 3,
        "max_concurrency": 2,
        "request_rate": float("inf"),
        "seed": 7,
        "result_dir": tmp_path / "raw",
        "results_path": tmp_path / "results.csv",
        "metrics_labels": {"model_name": "llama", "engine": "npu:0"},
    }
    values.update(changes)
    return ServerBenchmarkConfig(**values)


def test_build_vllm_argv_is_fixed_and_does_not_interpret_user_fragments(tmp_path):
    selected = config(
        tmp_path,
        model="model;touch /tmp/owned",
        request_rate=12.5,
    )
    argv = build_vllm_argv(selected, tmp_path / "result.json")

    assert argv[:3] == ["vllm", "bench", "serve"]
    assert argv[argv.index("--base-url") + 1] == "http://127.0.0.1:8000/v1"
    assert argv[argv.index("--endpoint") + 1] == "/completions"
    assert argv[argv.index("--model") + 1] == "model;touch /tmp/owned"
    assert argv[argv.index("--temperature") + 1] == "0"
    assert argv[argv.index("--metric-percentiles") + 1] == "50,85,90,95,99"
    assert argv[argv.index("--request-rate") + 1] == "12.5"
    assert "--ignore-eos" in argv
    assert "--save-result" in argv
    assert "--save-detailed" in argv


class Response:
    def __init__(self, *, payload=None, text=""):
        self._payload = payload
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class Client:
    def __init__(self, responses, events):
        self.responses = list(responses)
        self.events = events

    def get(self, url, **kwargs):
        self.events.append(("get", url, kwargs))
        return self.responses.pop(0)


def run_with_fakes(
    tmp_path,
    *,
    after_success=2,
    after_tokens=5,
    version_after=None,
    raw_model="llama",
):
    events = []
    version = {"furiosa_llm": "2026.3.0"}
    models = {"data": [{"id": "llama", "artifact_id": "fxb-123"}]}
    before = metrics_text(success=0, requests=0, tokens=0, itls=0)
    after = metrics_text(
        success=after_success,
        requests=2,
        tokens=after_tokens,
        itls=3,
    )
    client = Client(
        [
            Response(payload=version),
            Response(payload=models),
            Response(text=before),
            Response(text=after),
            Response(payload=version if version_after is None else version_after),
            Response(payload=models),
        ],
        events,
    )

    def command_runner(argv, **kwargs):
        events.append(("run", tuple(argv), kwargs))
        result_dir = Path(argv[argv.index("--result-dir") + 1])
        result_filename = argv[argv.index("--result-filename") + 1]
        result_dir.mkdir(parents=True, exist_ok=True)
        raw_result = json.loads(FIXTURE.read_text(encoding="utf-8"))
        raw_result["model_id"] = raw_model
        (result_dir / result_filename).write_text(
            json.dumps(raw_result),
            encoding="utf-8",
        )

    outcome = run_server_benchmark(
        config(tmp_path),
        http_client=client,
        command_runner=command_runner,
    )
    return outcome, events


def test_server_benchmark_lifecycle_persists_valid_evidence(tmp_path):
    outcome, events = run_with_fakes(tmp_path)

    assert outcome["status"] == "valid"
    event_names = [event[0] for event in events]
    assert event_names == ["get", "get", "get", "run", "get", "get", "get"]
    rows = load_results(results_path=tmp_path / "results.csv")
    assert len(rows) == 1
    assert rows[0]["inference_mode"] == "external_server"
    assert rows[0]["backend"] == "furiosa_llm_server"
    assert rows[0]["async_run_status"] == "valid"

    details_path = tmp_path / rows[0]["details_path"]
    details = json.loads(details_path.read_text(encoding="utf-8"))
    assert details["invalid_reasons"] == []
    assert details["raw_evidence"]["vllm_result_sha256"]
    assert details["raw_evidence"]["metrics_before_sha256"]
    assert details["raw_evidence"]["metrics_after_sha256"]
    assert details["client"]["source"] == "vllm_bench_serve_detailed_json"
    assert details["vendor"]["source"] == (
        "furiosa_server_prometheus_histogram_delta"
    )


def test_server_benchmark_counter_mismatch_cannot_be_valid(tmp_path):
    outcome, _ = run_with_fakes(tmp_path, after_success=3)

    assert outcome["status"] == "invalid"
    assert "vendor_metrics_scope_mismatch" in outcome["invalid_reasons"]
    rows = load_results(results_path=tmp_path / "results.csv")
    assert rows[0]["async_run_status"] == "invalid"
    assert "vendor_metrics_scope_mismatch" in rows[0][
        "async_invalid_reasons"
    ]


def test_server_benchmark_token_or_identity_change_cannot_be_valid(tmp_path):
    token_outcome, _ = run_with_fakes(tmp_path / "tokens", after_tokens=6)
    identity_outcome, _ = run_with_fakes(
        tmp_path / "identity",
        version_after={"furiosa_llm": "changed"},
    )
    model_outcome, _ = run_with_fakes(
        tmp_path / "model",
        raw_model="different-model",
    )

    assert token_outcome["status"] == "invalid"
    assert identity_outcome["status"] == "invalid"
    assert model_outcome["status"] == "invalid"
    assert "vendor_metrics_scope_mismatch" in token_outcome["invalid_reasons"]
    assert "vendor_metrics_scope_mismatch" in identity_outcome[
        "invalid_reasons"
    ]
    assert "vendor_metrics_scope_mismatch" in model_outcome[
        "invalid_reasons"
    ]


def test_server_benchmark_failure_persists_available_raw_hashes(tmp_path):
    events = []
    version = {"furiosa_llm": "2026.3.0"}
    models = {"data": [{"id": "llama", "artifact_id": "fxb-123"}]}
    before = metrics_text(success=0, requests=0, tokens=0, itls=0)
    client = Client(
        [
            Response(payload=version),
            Response(payload=models),
            Response(text=before),
        ],
        events,
    )

    def fail(argv, **kwargs):
        raise subprocess.CalledProcessError(2, argv)

    outcome = run_server_benchmark(
        config(tmp_path),
        http_client=client,
        command_runner=fail,
    )

    assert outcome["status"] == "invalid"
    assert outcome["failure_details_path"]
    failure_path = tmp_path / outcome["failure_details_path"]
    details = json.loads(failure_path.read_text(encoding="utf-8"))
    assert details["raw_evidence"]["metrics_before_sha256"]
    assert details["raw_evidence"]["metrics_after_sha256"] is None
    assert details["failure"]["error_type"] == "CalledProcessError"
    rows = load_results(results_path=tmp_path / "results.csv")
    assert rows[0]["failure_details_path"] == outcome["failure_details_path"]
