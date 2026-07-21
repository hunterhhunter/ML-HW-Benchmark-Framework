"""Reproducible Furiosa server benchmark orchestration."""

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from adapters.vllm_bench_result import normalize_vllm_bench_result
from core.result_store import (
    reserve_run_artifacts,
    save_async_details,
    save_async_failure_details,
    save_result,
)
from monitors.furiosa_server_metrics import normalize_furiosa_server_metrics


@dataclass(frozen=True)
class ServerBenchmarkConfig:
    base_url: str
    model: str
    input_tokens: int
    output_tokens: int
    num_prompts: int
    max_concurrency: int
    request_rate: float
    seed: int
    result_dir: Path
    results_path: Path
    metrics_labels: dict[str, str]
    http_timeout_sec: float = 5.0
    benchmark_timeout_sec: float = 3600.0

    def __post_init__(self):
        parsed = urlsplit(self.base_url)
        if (
            type(self.base_url) is not str
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        ):
            raise ValueError("base_url must be an HTTP(S) server origin")
        if type(self.model) is not str or not self.model:
            raise ValueError("model is required")
        for name in (
            "input_tokens",
            "output_tokens",
            "num_prompts",
            "max_concurrency",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact int")
        if type(self.seed) is not int:
            raise ValueError("seed must be an exact int")
        if type(self.request_rate) not in (int, float) or type(self.request_rate) is bool:
            raise ValueError("request_rate must be numeric")
        if math.isnan(float(self.request_rate)) or self.request_rate <= 0:
            raise ValueError("request_rate must be positive or infinity")
        for name in ("http_timeout_sec", "benchmark_timeout_sec"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or type(value) is bool
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if type(self.metrics_labels) is not dict or set(self.metrics_labels) != {
            "model_name",
            "engine",
        }:
            raise ValueError("metrics_labels must contain exact model_name and engine")
        if any(
            type(key) is not str or type(value) is not str or not value
            for key, value in self.metrics_labels.items()
        ):
            raise ValueError("metrics_labels must contain non-empty strings")


def build_vllm_argv(
    config: ServerBenchmarkConfig,
    result_path: Path,
) -> list[str]:
    argv = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--base-url",
        f"{config.base_url.rstrip('/')}/v1",
        "--endpoint",
        "/completions",
        "--model",
        config.model,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(config.input_tokens),
        "--random-output-len",
        str(config.output_tokens),
        "--max-concurrency",
        str(config.max_concurrency),
        "--num-prompts",
        str(config.num_prompts),
        "--seed",
        str(config.seed),
        "--temperature",
        "0",
        "--ignore-eos",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,85,90,95,99",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result_path.parent),
        "--result-filename",
        result_path.name,
    ]
    if math.isfinite(float(config.request_rate)):
        argv.extend(("--request-rate", str(config.request_rate)))
    return argv


def _get(client, url: str, timeout_sec: float):
    response = client.get(
        url,
        follow_redirects=False,
        timeout=timeout_sec,
    )
    response.raise_for_status()
    return response


def _get_json(client, url: str, timeout_sec: float) -> dict:
    payload = _get(client, url, timeout_sec).json()
    if type(payload) is not dict:
        raise TypeError(f"server identity response must be a JSON object: {url}")
    return payload


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_evidence(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def _model_is_present(payload: dict, model: str) -> bool:
    data = payload.get("data")
    return bool(
        type(data) is list
        and any(
            type(item) is dict and item.get("id") == model
            for item in data
        )
    )


def _safe_exception(exc: BaseException) -> dict:
    try:
        error_type = type.__getattribute__(type(exc), "__name__")
    except BaseException:
        error_type = "BaseException"
    if type(error_type) is not str:
        error_type = "BaseException"
    return {
        "error_type": error_type[:128],
        "error_message": f"server benchmark failed ({error_type[:128]})",
    }


def _artifact_reference(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def run_server_benchmark(
    config: ServerBenchmarkConfig,
    *,
    http_client=None,
    command_runner=subprocess.run,
) -> dict:
    reservation = reserve_run_artifacts(results_path=config.results_path)
    config.result_dir.mkdir(parents=True, exist_ok=True)
    raw_result_path = config.result_dir / f"{reservation.run_id}.vllm.json"
    before_path = config.result_dir / f"{reservation.run_id}.metrics.before.prom"
    after_path = config.result_dir / f"{reservation.run_id}.metrics.after.prom"
    client = httpx.Client() if http_client is None else http_client
    owns_client = http_client is None
    before_text = None
    after_text = None
    raw_result = None
    combined_metrics = {}
    invalid_reasons = []
    details = {
        "run": {
            "model": config.model,
            "base_url": config.base_url,
            "input_tokens": config.input_tokens,
            "output_tokens": config.output_tokens,
            "num_prompts": config.num_prompts,
            "max_concurrency": config.max_concurrency,
            "request_rate": (
                config.request_rate
                if math.isfinite(float(config.request_rate))
                else "inf"
            ),
            "seed": config.seed,
            "metrics_labels": dict(sorted(config.metrics_labels.items())),
        }
    }
    failure = None
    try:
        version_before = _get_json(
            client,
            f"{config.base_url.rstrip('/')}/version",
            config.http_timeout_sec,
        )
        models_before = _get_json(
            client,
            f"{config.base_url.rstrip('/')}/v1/models",
            config.http_timeout_sec,
        )
        if not _model_is_present(models_before, config.model):
            raise ValueError("requested model is absent from /v1/models")
        before_text = _get(
            client,
            f"{config.base_url.rstrip('/')}/metrics",
            config.http_timeout_sec,
        ).text
        if type(before_text) is not str:
            raise TypeError("before metrics response must be text")
        _write_evidence(before_path, before_text)

        argv = build_vllm_argv(config, raw_result_path)
        command_runner(
            argv,
            check=True,
            timeout=config.benchmark_timeout_sec,
        )
        if not raw_result_path.is_file():
            raise FileNotFoundError("vLLM benchmark result was not created")
        raw_result = json.loads(raw_result_path.read_text(encoding="utf-8"))
        if type(raw_result) is not dict:
            raise TypeError("vLLM benchmark result must be a JSON object")

        after_text = _get(
            client,
            f"{config.base_url.rstrip('/')}/metrics",
            config.http_timeout_sec,
        ).text
        if type(after_text) is not str:
            raise TypeError("after metrics response must be text")
        _write_evidence(after_path, after_text)
        version_after = _get_json(
            client,
            f"{config.base_url.rstrip('/')}/version",
            config.http_timeout_sec,
        )
        models_after = _get_json(
            client,
            f"{config.base_url.rstrip('/')}/v1/models",
            config.http_timeout_sec,
        )

        client_result = normalize_vllm_bench_result(raw_result)
        vendor_result = normalize_furiosa_server_metrics(
            before_text,
            after_text,
            labels=config.metrics_labels,
        )
        invalid_reasons.extend(client_result["invalid_reasons"])
        invalid_reasons.extend(vendor_result["invalid_reasons"])
        scope_mismatches = []
        if raw_result.get("model_id") != config.model:
            scope_mismatches.append("client_model_identity_mismatch")
        if version_before != version_after or models_before != models_after:
            scope_mismatches.append("server_identity_changed")
        if (
            vendor_result["metrics"]["server_vendor_successful_requests"]
            != client_result["metrics"]["server_successful_requests"]
        ):
            scope_mismatches.append("request_count_mismatch")
        if (
            vendor_result["metrics"]["server_vendor_generation_tokens"]
            != client_result["metrics"]["server_output_tokens"]
        ):
            scope_mismatches.append("generation_token_count_mismatch")
        expected_itls = sum(
            max(output_len - 1, 0)
            for output_len in client_result["details"]["raw_seconds"][
                "output_lens"
            ]
        )
        observed_itls = vendor_result["metrics"][
            "server_vendor_itl_observations"
        ]
        if observed_itls != expected_itls:
            scope_mismatches.append("itl_observation_count_mismatch")
        if scope_mismatches:
            invalid_reasons.append("vendor_metrics_scope_mismatch")

        combined_metrics.update(client_result["metrics"])
        combined_metrics.update(vendor_result["metrics"])
        details.update(
            {
                "client": client_result["details"],
                "vendor": vendor_result["details"],
                "server_identity": {
                    "version_before": version_before,
                    "version_after": version_after,
                    "models_before": models_before,
                    "models_after": models_after,
                },
                "scope_validation": {
                    "mismatches": scope_mismatches,
                    "expected_itl_observations": expected_itls,
                    "observed_itl_observations": observed_itls,
                },
            }
        )
    except BaseException as exc:
        invalid_reasons.append("benchmark_exception")
        failure = _safe_exception(exc)
    finally:
        if owns_client:
            client.close()

    raw_result_bytes = (
        raw_result_path.read_bytes() if raw_result_path.is_file() else b""
    )
    details["raw_evidence"] = {
        "vllm_result_path": str(raw_result_path),
        "vllm_result_sha256": (
            _sha256_bytes(raw_result_bytes) if raw_result_bytes else None
        ),
        "metrics_before_path": str(before_path),
        "metrics_before_sha256": (
            _sha256_bytes(before_text.encode("utf-8"))
            if before_text is not None
            else None
        ),
        "metrics_after_path": str(after_path),
        "metrics_after_sha256": (
            _sha256_bytes(after_text.encode("utf-8"))
            if after_text is not None
            else None
        ),
    }
    invalid_reasons = sorted(set(invalid_reasons))
    details["invalid_reasons"] = invalid_reasons
    status = "invalid" if invalid_reasons else "valid"
    details["status"] = status

    details_path = ""
    failure_details_path = ""
    if failure is None:
        published = save_async_details(
            reservation.run_id,
            details,
            results_dir=reservation.results_root,
            reservation=reservation,
        )
        details_path = _artifact_reference(published, reservation.results_root)
    else:
        details["failure"] = failure
        published = save_async_failure_details(
            reservation.run_id,
            details,
            results_dir=reservation.results_root,
            reservation=reservation,
        )
        failure_details_path = _artifact_reference(
            published,
            reservation.results_root,
        )

    save_result(
        metrics=combined_metrics,
        model_name=config.model,
        task="NLP_GENERATION",
        backend="furiosa_llm_server",
        device=config.metrics_labels["engine"],
        batch_size=1,
        warmup_runs=0,
        target_id="furiosa-rngd",
        accelerator_vendor="FuriosaAI",
        accelerator_name="RNGD",
        runtime_name="furiosa_llm_server",
        artifact_format="fxb",
        results_path=config.results_path,
        run_id=reservation.run_id,
        inference_mode="external_server",
        scenario=(
            "server_like"
            if math.isfinite(float(config.request_rate))
            else "offline_like"
        ),
        queue_capacity=config.max_concurrency,
        target_qps=(
            config.request_rate
            if math.isfinite(float(config.request_rate))
            else None
        ),
        schedule_seed=config.seed,
        async_run_status=status,
        async_invalid_reasons=",".join(invalid_reasons),
        details_path=details_path,
        failure_details_path=failure_details_path,
        reservation=reservation,
    )
    return {
        "run_id": reservation.run_id,
        "status": status,
        "invalid_reasons": invalid_reasons,
        "metrics": combined_metrics,
        "details_path": details_path,
        "failure_details_path": failure_details_path,
    }


def _request_rate(value: str) -> float:
    return math.inf if value == "inf" else float(value)


def _labels(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError("metrics label must use key=value")
        key, selected = value.split("=", 1)
        if not key or not selected or key in result:
            raise argparse.ArgumentTypeError("metrics labels must be unique key=value")
        result[key] = selected
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--max-concurrency", type=int, required=True)
    parser.add_argument("--request-rate", type=_request_rate, default=math.inf)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--metrics-label", action="append", default=[])
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = ServerBenchmarkConfig(
        base_url=args.base_url,
        model=args.model,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        num_prompts=args.num_prompts,
        max_concurrency=args.max_concurrency,
        request_rate=args.request_rate,
        seed=args.seed,
        result_dir=args.result_dir,
        results_path=args.results_path,
        metrics_labels=_labels(args.metrics_label),
    )
    return 0 if run_server_benchmark(config)["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
