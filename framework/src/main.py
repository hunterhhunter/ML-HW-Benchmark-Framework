import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Any

# 프로젝트 루트 경로 추가 (sys.path)
FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
project_root = str(FRAMEWORK_ROOT)
if project_root not in sys.path:
    sys.path.append(project_root)

from core.model_spec import Model_Spec, Task
from core.model_profiles import create_model_spec
from core.compiled_model import CompiledModel
from core.benchmarkrunner import BenchmarkRunner
from core.runtime_executor import NativeAsyncRuntimeExecutor
from core.inference_engine import InferenceEngine
from core.async_inference import (
    AsyncInferenceConfig,
    AsyncScenario,
    RunStatus,
)
from core.async_inference.trace import RequestTraceWriter
from core.result_store import (
    get_reserved_result_state,
    reserve_run_artifacts,
    save_async_details,
    save_async_failure_details,
    save_result,
)
from core.targets import resolve_target, target_metadata

# 구체화된 컴포넌트 임포트 (Facade Pattern 적용)
from dataloader import create_dataloader
from decoders import create_decoder
from evaluators import create_evaluator
from runtimes import create_runtime
from compilers import get_compiler, normalize_compile_result
# from src.runtimes.iree_rt import IREERuntime  # 향후 IREE 백엔드 추가 시 주석 해제


def _resolve_framework_path(path_value: str | None) -> str | None:
    """Resolve profile-owned relative paths from the framework root."""
    if not path_value:
        return path_value
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((FRAMEWORK_ROOT / path).resolve())


def _run_prepare_script(script: str) -> None:
    script_path = _resolve_framework_path(script)
    subprocess.run([sys.executable, script_path], check=True, cwd=str(FRAMEWORK_ROOT))


def _apply_hailo_task_runtime_defaults(
    runtime_kwargs: dict[str, Any],
    cli_runtime_options: dict[str, Any],
    task_enum: Task,
) -> None:
    if "output_format_type" in cli_runtime_options:
        return

    # Hailo's classifier examples force FLOAT32 outputs even when inputs are
    # UINT8. Detection post-processing paths also expect float tensors unless
    # the user explicitly overrides the format for a specific HEF.
    if task_enum in {Task.IMAGE_CLASSIFICATION, Task.OBJECT_DETECTION}:
        runtime_kwargs["output_format_type"] = "float32"


def run_auto_prepare(profile: dict, args: argparse.Namespace, target=None):
    """
    Zero-Config 벤치마크를 위해 누락된 리소스를 감지하고 백그라운드 준비 스크립트를 자동 실행합니다.
    """
    if args.backend in ("vllm", "furiosa_llm", "furiosa", "rngd"):
        model_path = args.model_path
    elif args.backend == "hailort":
        model_path = args.hef or args.artifact
    elif (
        target is not None
        and target.uses_compiler
        and not args.compile
        and target.artifact_format not in ("onnx", "hf_model")
    ):
        model_path = args.artifact
    elif target is not None and not target.uses_compiler and target.artifact_format not in ("onnx", "hf_model"):
        model_path = args.artifact
    else:
        model_path = args.onnx
    dataset_path = args.dataset

    can_auto_prepare_model = (
        args.backend not in ("hailort", "furiosa_llm", "furiosa", "rngd")
        and not (
            target is not None
            and target.uses_compiler
            and not args.compile
            and target.artifact_format not in ("onnx", "hf_model")
        )
        and not (target is not None and not target.uses_compiler and target.artifact_format not in ("onnx", "hf_model"))
    )
    if can_auto_prepare_model and "prepare_model_script" in profile and profile["prepare_model_script"]:
        if not model_path or not os.path.exists(model_path):
            script = profile["prepare_model_script"]
            print(f"[*] 모델 리소스 누락 감지. 자동 준비 스크립트 실행: {script}")
            _run_prepare_script(script)
            
    if "prepare_dataset_script" in profile and profile["prepare_dataset_script"]:
        if not dataset_path or not os.path.exists(dataset_path):
            script = profile["prepare_dataset_script"]
            print(f"[*] 데이터셋 리소스 누락 감지. 자동 준비 스크립트 실행: {script}")
            _run_prepare_script(script)


def _coerce_option_value(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_key_value_options(items: list[str] | None, *, coerce_values: bool = False) -> dict:
    """CLI의 key=value 리스트를 딕셔너리로 변환한다."""
    options = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"옵션은 key=value 형식이어야 합니다: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"옵션 key가 비어 있습니다: {item}")
        options[key] = _coerce_option_value(value) if coerce_values else value
    return options


_FURIOSA_RUNTIME_OPTIONS = frozenset({
    "devices",
    "data_parallel_size",
    "pipeline_parallel_size",
    "max_io_memory_mb",
    "seed",
    "cache_dir",
    "npu_queue_limit",
    "max_processing_samples",
    "spare_blocks_ratio",
})


def _validate_furiosa_runtime_options(options: dict[str, Any]) -> None:
    unknown = sorted(set(options) - _FURIOSA_RUNTIME_OPTIONS)
    if unknown:
        raise ValueError(
            "Furiosa-LLM에서 지원하지 않는 runtime option입니다: "
            + ", ".join(unknown)
        )


def _validate_furiosa_cli(args: argparse.Namespace, task_enum: Task) -> None:
    if task_enum != Task.NLP_GENERATION:
        raise ValueError("furiosa_llm backend supports only NLP_GENERATION tasks.")

    model_path = Path(args.model_path) if args.model_path else None
    if model_path is None or not model_path.is_dir():
        raise ValueError(
            "furiosa_llm backend requires --model-path to be a local Hugging Face directory."
        )

    selected_fxb = args.fxb or args.artifact
    fxb_path = Path(selected_fxb) if selected_fxb else None
    if fxb_path is None or not fxb_path.is_file() or fxb_path.suffix.lower() != ".fxb":
        raise ValueError(
            "furiosa_llm backend requires --fxb (or --artifact) to be an existing .fxb file."
        )

    args.fxb = str(fxb_path)
    args.artifact = str(fxb_path)
    if not args.tokenizer_path:
        args.tokenizer_path = str(model_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified BenchmarkRunner CLI Orchestrator")
    parser.add_argument("--model", type=str, required=True, help="모델 이름 (예: resnet50, llama-3.2-3b)")
    parser.add_argument("--onnx", type=str, default=None, help="ONNX 파일의 절대 또는 상대 경로 (onnxruntime 백엔드 필수)")
    parser.add_argument("--hef", type=str, default=None, help="HailoRT 실행용 HEF 파일 경로 (hailo8/hailo10h target 필수)")
    parser.add_argument("--artifact", type=str, default=None, help="target 전용 사전 컴파일 artifact 경로 (예: DEEPX .dxnn)")
    parser.add_argument("--fxb", type=str, default=None, help="Furiosa RNGD 실행용 FXB 파일 경로 (--artifact fallback 지원)")
    parser.add_argument("--model-path", type=str, default=None, help="HuggingFace 모델 디렉토리 경로 (vLLM 백엔드 필수)")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="HuggingFace 토크나이저 디렉토리 경로 (NLP 모델 필수)")
    parser.add_argument("--dataset", type=str, default=None, help="평가용 데이터셋 최상위 디렉토리 또는 CSV 파일 경로")
    parser.add_argument("--image-dir", type=str, default="", help="(옵션) 데이터셋 내 이미지 하위 폴더 경로")
    parser.add_argument("--label-dir", type=str, default="", help="(옵션) 데이터셋 내 라벨 하위 폴더 경로")
    parser.add_argument("--layout", type=str, default="NCHW", choices=["NCHW", "NHWC"], help="모델 텐서 레이아웃 (기본: NCHW)")
    parser.add_argument("--image-preprocess-mode", type=str, default="auto", choices=["auto", "normalized", "raw"], help="이미지 전처리 dtype 모드. raw는 resize/crop 후 0..255 픽셀을 전달합니다.")
    parser.add_argument("--image-resize-mode", type=str, default="auto", choices=["auto", "direct", "letterbox"], help="객체 탐지 이미지 resize 모드. Hailo object detection은 auto에서 letterbox를 사용합니다.")
    parser.add_argument("--target", type=str, default=None, help="실행 target_id (예: cpu, cuda, hailo8, hailo10h, vendor_mock_npu). 지정 시 backend/device보다 우선합니다.")
    parser.add_argument("--backend", type=str, default="onnxruntime", choices=["onnxruntime", "iree", "vllm", "hailort", "deepx", "furiosa_llm", "furiosa", "rngd"], help="추론을 실행할 백엔드 (기본: onnxruntime)")
    parser.add_argument("--device", type=str, default="cpu", help="추론 장치 (예: cpu, cuda, 기본: cpu)")
    parser.add_argument("--compile", dest="compile", action="store_true", default=True, help="target에 compiler가 있으면 컴파일을 수행합니다.")
    parser.add_argument("--no-compile", dest="compile", action="store_false", help="target compiler를 사용하지 않고 원본 artifact를 runtime에 전달합니다.")
    parser.add_argument("--compile-option", action="append", default=[], help="벤더 compiler 옵션 key=value. 여러 번 지정 가능.")
    parser.add_argument("--runtime-option", action="append", default=[], help="런타임 옵션 key=value. 여러 번 지정 가능 (예: output_format_type=uint8).")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="추론 배치 사이즈 (기본: 1)")
    parser.add_argument("--warmup", "-w", type=int, default=2, help="웜업 횟수 (기본: 2)")
    parser.add_argument("--max-steps", type=int, default=None, help="시간이 지루할 때 쓸 강제 종료 리미트 (옵션)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="LLM 생성 최대 토큰 수 (기본: 256)")
    parser.add_argument("--max-model-len", type=int, default=None, help="vLLM 최대 컨텍스트 길이 (기본: 모델 기본값, 메모리 부족 시 줄이세요)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="vLLM GPU 메모리 사용률 0.0~1.0 (기본: 0.90, OOM 시 낮추세요 예: 0.7)")
    parser.add_argument("--enforce-eager", action="store_true", default=None, help="vLLM CUDA 그래프 캡처 비활성화 (메모리 부족 시 사용)")
    parser.add_argument(
        "--results-path",
        type=str,
        default=None,
        help=(
            "결과 CSV 경로. async details/trace도 이 CSV의 parent 아래에 저장됩니다. "
            "기본: framework/results/benchmark_results.csv"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "e2e: 샘플별 예측/정답/점수 로그, async_queue: coarse lifecycle "
            "단계만 출력 (기본: 비활성)"
        ),
    )
    parser.add_argument("--monitor", action="store_true", help="벤치마크 중 하드웨어 모니터링 활성화 (GPU/CPU/RAM)")
    parser.add_argument("--monitor-interval", type=float, default=0.2, help="모니터링 샘플링 간격 초 (기본: 0.2)")
    parser.add_argument(
        "--inference-mode",
        choices=["e2e", "async_queue"],
        default="e2e",
        help="추론 실행 방식 (기본: e2e)",
    )
    parser.add_argument(
        "--scenario",
        choices=["offline", "server_like"],
        default=None,
        help="async_queue 부하 시나리오 (미지정 시 offline)",
    )
    parser.add_argument("--target-qps", type=float, default=None)
    parser.add_argument("--queue-capacity", type=int, default=None)
    parser.add_argument("--worker-count", type=int, default=None)
    parser.add_argument("--batch-timeout-ms", type=float, default=None)
    parser.add_argument("--submit-timeout-sec", type=float, default=None)
    parser.add_argument("--flush-timeout-sec", type=float, default=None)
    parser.add_argument("--request-timeout-ms", type=float, default=None)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--min-duration-sec", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--schedule-seed", type=int, default=None)
    parser.add_argument("--latency-slo-ms", type=float, default=None)
    parser.add_argument("--save-request-trace", action="store_true")
    return parser


ASYNC_ONLY_ARGUMENTS = {
    "scenario",
    "target_qps",
    "queue_capacity",
    "worker_count",
    "batch_timeout_ms",
    "submit_timeout_sec",
    "flush_timeout_sec",
    "request_timeout_ms",
    "min_samples",
    "min_duration_sec",
    "max_samples",
    "schedule_seed",
    "latency_slo_ms",
}


def validate_async_args(args: argparse.Namespace) -> None:
    if args.inference_mode == "e2e":
        supplied = [
            name
            for name in ASYNC_ONLY_ARGUMENTS
            if getattr(args, name) is not None
        ]
        if args.save_request_trace:
            supplied.append("save_request_trace")
        if supplied:
            rendered = ", ".join(
                f"--{name.replace('_', '-')}" for name in sorted(supplied)
            )
            raise ValueError(
                f"async_queue 전용 옵션입니다: {rendered}"
            )
        return
    if args.max_steps is not None:
        raise ValueError(
            "async_queue에서는 --max-steps 대신 --max-samples를 사용하세요"
        )
    if (args.scenario or "offline") == "server_like" and args.target_qps is None:
        raise ValueError("server_like에는 --target-qps가 필요합니다")
    if (
        args.backend in {"furiosa_llm", "furiosa", "rngd"}
        and args.batch_size != 1
    ):
        raise ValueError(
            "Furiosa native async는 framework 동적 배칭을 사용하지 않습니다. "
            "--batch-size 1을 사용하세요."
        )


def build_async_config(args: argparse.Namespace) -> AsyncInferenceConfig:
    def selected(name, default):
        value = getattr(args, name)
        return default if value is None else value

    scenario_name = args.scenario or "offline"
    config = AsyncInferenceConfig(
        scenario=AsyncScenario(scenario_name),
        queue_capacity=selected("queue_capacity", 256),
        worker_count=selected("worker_count", 1),
        max_batch_size=args.batch_size,
        batch_timeout_ms=selected("batch_timeout_ms", 1.0),
        submit_timeout_sec=selected("submit_timeout_sec", 30.0),
        flush_timeout_sec=selected("flush_timeout_sec", 300.0),
        request_timeout_ms=selected("request_timeout_ms", 0.0),
        min_samples=selected("min_samples", 100),
        min_duration_sec=selected(
            "min_duration_sec",
            10.0 if scenario_name == "server_like" else 0.0,
        ),
        max_samples=args.max_samples,
        target_qps=args.target_qps,
        schedule_seed=selected("schedule_seed", 0),
        latency_slo_ms=args.latency_slo_ms,
    )
    config.validate()
    return config


def _build_async_runtime_executor(args, runtime, loader, config):
    if args.backend not in {"furiosa_llm", "furiosa", "rngd"}:
        return None
    if config.max_batch_size != 1:
        raise ValueError(
            "Furiosa native async requires max_batch_size=1 so that "
            "Furiosa-LLM owns continuous batching."
        )
    metadata = loader.get_metadata()
    backend = runtime.create_native_backend(
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=metadata.get("stop_token_ids"),
    )
    return NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=min(config.worker_count, config.queue_capacity),
        completion_timeout_sec=config.flush_timeout_sec,
    )


def _print_final_metrics(model_name: str, results: dict) -> None:
    print("\n" + "="*40)
    print(f" Final Metrics ({model_name.upper()}) ")
    print("="*40)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("="*40)


def _artifact_reference(path: Path, results_root: Path) -> str:
    return str(Path(path).relative_to(Path(results_root).parent))


def _result_save_kwargs(args, results, task_name, target_meta) -> dict:
    return {
        "metrics": results,
        "model_name": args.model,
        "task": task_name,
        "backend": args.backend,
        "device": args.device,
        "batch_size": args.batch_size,
        "warmup_runs": args.warmup,
        "max_steps": args.max_steps,
        "target_id": target_meta["target_id"],
        "accelerator_vendor": target_meta["accelerator_vendor"],
        "accelerator_name": target_meta["accelerator_name"],
        "runtime_name": target_meta["runtime_name"],
        "compiler_name": target_meta["compiler_name"],
        "artifact_format": target_meta["artifact_format"],
    }


def _safe_persistence_error(phase: str, error) -> dict:
    if type(error) is dict:
        diagnostic = dict(error)
        diagnostic.setdefault("phase", phase)
        return diagnostic
    try:
        error_type = type.__getattribute__(type(error), "__name__")
    except BaseException:
        error_type = "<unknown>"
    if type(error_type) is not str:
        error_type = "<unknown>"
    try:
        args = BaseException.args.__get__(error, type(error))
    except BaseException:
        args = ()
    message = (
        args[0]
        if type(args) is tuple and len(args) == 1 and type(args[0]) is str
        else f"<{error_type}>"
    )
    try:
        state = BaseException.__getattribute__(error, "__dict__")
    except BaseException:
        state = {}
    if type(state) is not dict:
        state = {}
    committed = dict.get(state, "final_file_committed")
    uncertain = dict.get(state, "publication_state_uncertain")
    return {
        "phase": phase,
        "error_type": error_type,
        "error_message": message,
        "final_file_committed": committed is True,
        "publication_state_uncertain": (
            uncertain is True or (committed is True and uncertain is not False)
        ),
    }


_SAFE_RUNTIME_BACKENDS = frozenset(
    {
        "deepx",
        "hailort",
        "iree",
        "mock_npu",
        "onnxruntime",
        "furiosa_llm",
        "vllm",
    }
)
_SAFE_IDENTIFIER_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:+-"
)
_REDACTED_IDENTIFIER = "<redacted>"
_SAFE_FAILURE_PHASES = frozenset(
    {
        "complete",
        "created",
        "csv_save",
        "engine_setup",
        "engine_start",
        "finalization",
        "measurement",
        "reservation",
        "result_shaping",
        "runner_run",
        "runner_setup",
        "runtime_unload",
        "sidecar_save",
        "trace_close",
        "trace_start",
        "validation",
        "warmup",
    }
)
_SAFE_SECONDARY_PHASES = frozenset(
    {
        "failure_csv",
        "failure_details_recovery",
        "failure_persistence",
        "failure_phase",
        "failure_sidecar",
        "normal_csv_recovery",
        "request_trace_cleanup",
        "runtime_unload",
        "runtime_unload_safety",
    }
)
_SAFE_SECONDARY_ERROR_TYPES = frozenset(
    {
        "ArtifactFilesystemUnsupportedError",
        "FileExistsError",
        "InvalidArgument",
        "OSError",
        "PermissionError",
        "RuntimeError",
        "TimeoutError",
        "TypeError",
        "ValueError",
    }
)


def _safe_identifier(value, *, provider=False) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 64
        or any(char not in _SAFE_IDENTIFIER_CHARACTERS for char in value)
        or (provider and not value.endswith("ExecutionProvider"))
    ):
        return _REDACTED_IDENTIFIER
    return value


def _safe_runtime_diagnostics(runtime) -> dict:
    try:
        value = runtime.get_device_spec()
    except BaseException:
        return {}
    if type(value) is not dict:
        return {}

    snapshot = {}
    backend = dict.get(value, "backend")
    if type(backend) is str:
        snapshot["backend"] = (
            backend
            if backend in _SAFE_RUNTIME_BACKENDS
            else _REDACTED_IDENTIFIER
        )
    device = dict.get(value, "device")
    if device is not None:
        snapshot["device"] = _safe_identifier(device)
    providers = dict.get(value, "active_providers")
    if type(providers) in (list, tuple):
        snapshot["active_providers"] = [
            _safe_identifier(provider, provider=True)
            for provider in list(providers)[:32]
        ]
    return snapshot


def _async_run_metadata(
    args, task_name, target_meta, runtime_diagnostics
) -> dict:
    artifact = args.onnx or args.hef or args.artifact or args.model_path or ""
    return {
        "model_name": args.model,
        "task": task_name,
        "backend": args.backend,
        "device": args.device,
        "batch_size": args.batch_size,
        "warmup_runs": args.warmup,
        "target_id": dict.get(target_meta, "target_id", ""),
        "dataset_path": str(args.dataset or ""),
        "model_artifact_path": str(artifact),
        "runtime_device_spec": runtime_diagnostics,
    }


def _safe_print(*values, **kwargs) -> bool:
    try:
        print(*values, **kwargs)
    except BaseException:
        return False
    return True


def _safe_print_final_metrics(model_name: str, results: dict) -> None:
    try:
        _print_final_metrics(model_name, results)
    except BaseException:
        pass


def _debug_lifecycle(
    args, phase, event, reservation=None, **fields
) -> None:
    try:
        if not args.debug:
            return
        parts = [f"phase={phase}", f"event={event}"]
        if reservation is not None:
            parts.append(f"run_id={reservation.run_id}")
        parts.extend(f"{key}={value}" for key, value in fields.items())
        _safe_print(
            "[AsyncDebug] " + " ".join(parts),
            file=sys.stderr,
            flush=True,
        )
    except BaseException:
        return


def _diagnostic_proves_commit(diagnostic) -> bool:
    return (
        type(diagnostic) is dict
        and dict.get(diagnostic, "final_file_committed") is True
        and dict.get(diagnostic, "publication_state_uncertain") is False
    )


def _render_persistence_error(diagnostic) -> str:
    if type(diagnostic) is not dict:
        return "phase=<unknown> error_type=<unknown> error_message=<unknown>"

    def safe_text(name):
        value = dict.get(diagnostic, name)
        return value if type(value) is str else "<unknown>"

    return (
        f"phase={safe_text('phase')} "
        f"error_type={safe_text('error_type')} "
        f"error_message={safe_text('error_message')}"
    )


def _record_async_invalid_reason(async_result, reason: str) -> None:
    reasons = set(async_result.invalid_reasons)
    reasons.update(async_result.details.get("invalid_reasons", []))
    reasons.add(reason)
    normalized_reasons = sorted(reasons)
    async_result.details["invalid_reasons"] = normalized_reasons
    async_result.details["status"] = RunStatus.INVALID.value
    async_result.metrics["async_run_status"] = RunStatus.INVALID.value
    async_result.metrics["async_invalid_reasons"] = ",".join(
        normalized_reasons
    )


def _record_async_outstanding_zero_proof(
    async_result,
    lifecycle_state: dict,
) -> None:
    lifecycle_state["outstanding_zero_proven"] = False
    outstanding = dict.get(
        async_result.metrics,
        "async_outstanding_requests",
    )
    if type(outstanding) is int and outstanding == 0:
        lifecycle_state["outstanding_zero_proven"] = True


def _record_async_persistence_failure(
    async_result,
    reason: str,
    diagnostic: dict,
) -> None:
    _record_async_invalid_reason(async_result, reason)
    async_result.details.setdefault("persistence_errors", []).append(
        diagnostic
    )


def _record_async_warning(async_result, warning: str) -> None:
    warnings = set(async_result.details.get("warnings", []))
    warnings.add(warning)
    async_result.details["warnings"] = sorted(warnings)


def _attach_secondary(primary: BaseException, phase: str, error) -> None:
    normalized = _safe_persistence_error(phase, error)
    normalized["phase"] = phase
    try:
        state = BaseException.__getattribute__(primary, "__dict__")
        if type(state) is dict:
            errors = dict.get(state, "cleanup_secondary_errors")
            if type(errors) is not list:
                errors = []
                dict.__setitem__(state, "cleanup_secondary_errors", errors)
            list.append(errors, normalized)
    except BaseException:
        pass
    try:
        BaseException.add_note(
            primary,
            f"{phase} also failed: {_render_persistence_error(normalized)}",
        )
    except BaseException:
        pass


def _failure_diagnostic(primary, phase) -> dict:
    diagnostic = _safe_persistence_error(phase, primary)
    safe_phase = (
        phase
        if type(phase) is str and phase in _SAFE_FAILURE_PHASES
        else _REDACTED_IDENTIFIER
    )
    safe_error_type = _safe_identifier(
        dict.get(diagnostic, "error_type", "<unknown>")
    )
    return {
        "phase": safe_phase,
        "error_type": safe_error_type,
        "error_message": (
            f"benchmark failed during {safe_phase} ({safe_error_type})"
        ),
    }


def _safe_cleanup_secondary_errors(primary) -> list[dict]:
    try:
        state = BaseException.__getattribute__(primary, "__dict__")
    except BaseException:
        return []
    if type(state) is not dict:
        return []
    errors = dict.get(state, "cleanup_secondary_errors")
    if type(errors) is not list:
        return []

    diagnostics = []
    for error in list(errors)[:32]:
        if type(error) is not dict:
            continue
        phase = dict.get(error, "phase")
        if type(phase) is not str or phase not in _SAFE_SECONDARY_PHASES:
            phase = _REDACTED_IDENTIFIER
        error_type = dict.get(error, "error_type")
        if (
            type(error_type) is not str
            or error_type not in _SAFE_SECONDARY_ERROR_TYPES
        ):
            error_type = _REDACTED_IDENTIFIER
        diagnostics.append(
            {
                "phase": phase,
                "error_type": error_type,
                "error_message": (
                    f"secondary failure during {phase} ({error_type})"
                ),
            }
        )
    return diagnostics


def _async_failure_details(
    *,
    args,
    primary,
    phase,
    measurement_started,
    runtime_diagnostics,
    task_name,
    target_meta,
) -> dict:
    run = _async_run_metadata(
        args,
        task_name,
        target_meta,
        runtime_diagnostics,
    )
    run["measurement_started"] = bool(measurement_started)
    return {
        "status": RunStatus.INVALID.value,
        "invalid_reasons": ["benchmark_exception"],
        "warnings": (
            []
            if runtime_diagnostics
            else ["runtime_device_spec_unavailable"]
        ),
        "run": run,
        "failure": _failure_diagnostic(primary, phase),
        "cleanup_secondary_errors": _safe_cleanup_secondary_errors(primary),
        "counts": (
            {
                "submitted": 0,
                "accepted": 0,
                "completed": 0,
                "failed": 0,
                "rejected": 0,
                "outstanding": 0,
            }
            if not measurement_started
            else None
        ),
        "counts_available": not measurement_started,
    }


def _persist_async_failure(
    *,
    args,
    config,
    reservation,
    primary,
    phase,
    measurement_started,
    runtime_diagnostics,
    task_name,
    target_meta,
    primary_details_committed=False,
    csv_committed=False,
) -> bool:
    details = _async_failure_details(
        args=args,
        primary=primary,
        phase=phase,
        measurement_started=measurement_started,
        runtime_diagnostics=runtime_diagnostics,
        task_name=task_name,
        target_meta=target_meta,
    )
    details_path = ""
    primary_details_available = bool(primary_details_committed)
    recovery_required = bool(primary_details_committed or csv_committed)
    csv_was_committed = bool(csv_committed)
    if primary_details_committed:
        details_path = _artifact_reference(
            reservation.details_path,
            reservation.results_root,
        )
    else:
        _debug_lifecycle(args, "sidecar_save", "start", reservation)
        try:
            details_file = save_async_details(
                reservation.run_id,
                details,
                results_dir=reservation.results_root,
                reservation=reservation,
            )
            details_path = _artifact_reference(
                details_file,
                reservation.results_root,
            )
            primary_details_available = True
        except BaseException as exc:
            recovery_required = True
            diagnostic = _safe_persistence_error("failure_sidecar", exc)
            diagnostic["phase"] = "failure_sidecar"
            if _diagnostic_proves_commit(diagnostic):
                details_path = _artifact_reference(
                    reservation.details_path,
                    reservation.results_root,
                )
                primary_details_available = True
            _attach_secondary(primary, "failure_sidecar", exc)
            _debug_lifecycle(
                args,
                "sidecar_save",
                "failed",
                reservation,
                error_type=dict.get(diagnostic, "error_type", "<unknown>"),
            )
            _safe_print(
                f"[Error] async failure sidecar 저장 실패: "
                f"{_render_persistence_error(diagnostic)}",
                file=sys.stderr,
            )
        else:
            _debug_lifecycle(
                args,
                "sidecar_save",
                "complete",
                reservation,
                details_path=details_file,
            )

    failure_details_path = (
        _artifact_reference(
            reservation.failure_details_path,
            reservation.results_root,
        )
        if recovery_required
        else ""
    )

    save_kwargs = {
        "metrics": {},
        "model_name": args.model,
        "task": task_name,
        "backend": args.backend,
        "device": args.device,
        "batch_size": args.batch_size,
        "warmup_runs": args.warmup,
        "max_steps": None,
        "target_id": dict.get(target_meta, "target_id", ""),
        "accelerator_vendor": dict.get(
            target_meta, "accelerator_vendor", ""
        ),
        "accelerator_name": dict.get(
            target_meta, "accelerator_name", ""
        ),
        "runtime_name": dict.get(target_meta, "runtime_name", ""),
        "compiler_name": dict.get(target_meta, "compiler_name", ""),
        "artifact_format": dict.get(target_meta, "artifact_format", ""),
        "results_path": reservation.results_path,
        "run_id": reservation.run_id,
        "inference_mode": "async_queue",
        "scenario": config.scenario.value,
        "queue_capacity": config.queue_capacity,
        "worker_count": config.worker_count,
        "batch_timeout_ms": config.batch_timeout_ms,
        "target_qps": config.target_qps,
        "schedule_seed": config.schedule_seed,
        "async_run_status": RunStatus.INVALID.value,
        "async_invalid_reasons": "benchmark_exception",
        "details_path": details_path,
        "failure_details_path": failure_details_path,
        "request_trace_path": "",
        "reservation": reservation,
    }
    if not csv_committed:
        _debug_lifecycle(args, "csv_save", "start", reservation)
        try:
            run_id = save_result(**save_kwargs)
            if type(run_id) is not str or run_id != reservation.run_id:
                raise RuntimeError(
                    "reserved CSV save returned an unexpected run ID"
                )
        except BaseException as exc:
            recovery_required = True
            diagnostic = _safe_persistence_error("failure_csv", exc)
            diagnostic["phase"] = "failure_csv"
            _attach_secondary(primary, "failure_csv", exc)
            _debug_lifecycle(
                args,
                "csv_save",
                "failed",
                reservation,
                error_type=dict.get(
                    diagnostic,
                    "error_type",
                    "<unknown>",
                ),
            )
            _safe_print(
                f"[Error] async failure CSV 저장 실패: "
                f"{_render_persistence_error(diagnostic)}",
                file=sys.stderr,
            )
        else:
            csv_committed = True
            _debug_lifecycle(args, "csv_save", "complete", reservation)

    recovery_committed = False
    if recovery_required:
        failure_details_path = _artifact_reference(
            reservation.failure_details_path,
            reservation.results_root,
        )
        recovery_details = _async_failure_details(
            args=args,
            primary=primary,
            phase=phase,
            measurement_started=measurement_started,
            runtime_diagnostics=runtime_diagnostics,
            task_name=task_name,
            target_meta=target_meta,
        )
        recovery_details["recovery"] = {
            "normal_details_preserved": bool(primary_details_committed),
            "csv_already_committed": csv_was_committed,
            "failure_details_path": failure_details_path,
        }
        _debug_lifecycle(
            args,
            "failure_details_save",
            "start",
            reservation,
        )
        try:
            recovery_file = save_async_failure_details(
                reservation.run_id,
                recovery_details,
                results_dir=reservation.results_root,
                reservation=reservation,
            )
        except BaseException as exc:
            diagnostic = _safe_persistence_error(
                "failure_details_recovery",
                exc,
            )
            diagnostic["phase"] = "failure_details_recovery"
            _attach_secondary(primary, "failure_details_recovery", exc)
            _debug_lifecycle(
                args,
                "failure_details_save",
                "failed",
                reservation,
                error_type=dict.get(
                    diagnostic,
                    "error_type",
                    "<unknown>",
                ),
            )
            _safe_print(
                f"[Error] async failure recovery 저장 실패: "
                f"{_render_persistence_error(diagnostic)}",
                file=sys.stderr,
            )
        else:
            recovery_committed = True
            _debug_lifecycle(
                args,
                "failure_details_save",
                "complete",
                reservation,
                failure_details_path=recovery_file,
            )
            _safe_print(
                "[AsyncFailure] "
                f"run_id={reservation.run_id} "
                f"failure_details_path={recovery_file} "
                f"csv_already_committed={str(csv_was_committed).lower()}",
                file=sys.stderr,
            )

    failure_truth_committed = (
        recovery_committed
        if recovery_required
        else primary_details_available
    )
    return bool(csv_committed and failure_truth_committed)


def _close_trace_writer(trace_writer, timeout):
    try:
        closed = trace_writer.close(timeout=timeout)
    except Exception as exc:
        diagnostic = _safe_persistence_error("trace_close", exc)
        return _diagnostic_proves_commit(diagnostic), diagnostic
    if closed:
        return True, None
    diagnostic = _safe_persistence_error(
        "trace_close",
        trace_writer.error
        or {
            "phase": "trace_close",
            "error_type": "TraceCloseError",
            "error_message": "trace writer did not commit",
        },
    )
    return _diagnostic_proves_commit(diagnostic), diagnostic


def _close_trace_after_failure(
    primary,
    trace_writer,
    timeout,
    args,
    reservation,
) -> None:
    if trace_writer is None:
        return
    _debug_lifecycle(args, "trace_close", "start", reservation)
    try:
        _, diagnostic = _close_trace_writer(trace_writer, timeout)
    except BaseException as secondary:
        _debug_lifecycle(
            args,
            "trace_close",
            "failed",
            reservation,
            error_type=_failure_diagnostic(
                secondary, "request_trace_cleanup"
            )["error_type"],
        )
        _attach_secondary(primary, "request_trace_cleanup", secondary)
    else:
        if diagnostic is not None:
            _debug_lifecycle(
                args,
                "trace_close",
                "failed",
                reservation,
                error_type=dict.get(
                    diagnostic, "error_type", "<unknown>"
                ),
            )
            _attach_secondary(primary, "request_trace_cleanup", diagnostic)
        else:
            _debug_lifecycle(
                args,
                "trace_close",
                "complete",
                reservation,
            )


def _cleanup_async_setup(
    primary,
    runtime,
    trace_writer,
    timeout,
    args,
    reservation,
) -> None:
    _close_trace_after_failure(
        primary,
        trace_writer,
        timeout,
        args,
        reservation,
    )
    _debug_lifecycle(args, "runtime_unload", "start", reservation)
    try:
        runtime.unload()
    except BaseException as secondary:
        _debug_lifecycle(
            args,
            "runtime_unload",
            "failed",
            reservation,
            error_type=_failure_diagnostic(
                secondary, "runtime_unload"
            )["error_type"],
        )
        _attach_secondary(primary, "runtime_unload", secondary)
    else:
        _debug_lifecycle(
            args,
            "runtime_unload",
            "complete",
            reservation,
        )


def _cleanup_async_run_failure(
    primary,
    engine,
    runtime,
    trace_writer,
    timeout,
    args,
    reservation,
    outstanding_zero_proven,
) -> None:
    _close_trace_after_failure(
        primary,
        trace_writer,
        timeout,
        args,
        reservation,
    )
    if outstanding_zero_proven is False:
        return
    try:
        unload_safe = getattr(
            engine,
            "runtime_unload_safe_after_failure",
            False,
        )
    except BaseException as secondary:
        _attach_secondary(primary, "runtime_unload_safety", secondary)
        return
    if unload_safe is not True:
        return
    _debug_lifecycle(args, "runtime_unload", "start", reservation)
    try:
        runtime.unload()
    except BaseException as secondary:
        _debug_lifecycle(
            args,
            "runtime_unload",
            "failed",
            reservation,
            error_type=_failure_diagnostic(
                secondary, "runtime_unload"
            )["error_type"],
        )
        _attach_secondary(primary, "runtime_unload", secondary)
    else:
        _debug_lifecycle(
            args,
            "runtime_unload",
            "complete",
            reservation,
        )


def _complete_async_benchmark(
    *,
    args,
    config,
    reservation,
    trace_writer,
    async_result,
    runtime,
    runtime_unload_safe,
    task_name,
    target_meta,
    actual_results_path,
    lifecycle_state,
) -> int:
    trace_path = ""
    persistence_failed = False
    if trace_writer is not None:
        lifecycle_state["phase"] = "trace_close"
        _debug_lifecycle(args, "trace_close", "start", reservation)
        trace_committed, diagnostic = _close_trace_writer(
            trace_writer, config.flush_timeout_sec
        )
        lifecycle_state["trace_closed"] = True
        if diagnostic is not None:
            _debug_lifecycle(
                args,
                "trace_close",
                "failed",
                reservation,
                error_type=dict.get(
                    diagnostic, "error_type", "<unknown>"
                ),
            )
            persistence_failed = True
            _record_async_persistence_failure(
                async_result,
                "request_trace_persistence_failed",
                diagnostic,
            )
        else:
            _debug_lifecycle(
                args,
                "trace_close",
                "complete",
                reservation,
            )
        if trace_committed:
            trace_path = _artifact_reference(
                reservation.trace_path,
                reservation.results_root,
            )
        if trace_writer.dropped:
            _record_async_warning(
                async_result,
                f"request_trace_dropped:{trace_writer.dropped}",
            )

    lifecycle_state["phase"] = "result_shaping"
    results = async_result.metrics
    outstanding = dict.get(results, "async_outstanding_requests")
    outstanding_is_exact_int = type(outstanding) is int
    outstanding_is_zero = outstanding_is_exact_int and outstanding == 0
    if not outstanding_is_zero:
        _record_async_invalid_reason(
            async_result,
            "counter_invariant_failed",
        )
    if not outstanding_is_exact_int:
        results["async_outstanding_requests"] = None
    _safe_print_final_metrics(args.model, results)
    async_result.details["run"] = _async_run_metadata(
        args,
        task_name,
        target_meta,
        dict.get(lifecycle_state, "runtime_diagnostics", {}),
    )
    async_result.details["hardware_metrics"] = {
        key: value for key, value in results.items() if key.startswith("hw_")
    }
    if not dict.get(lifecycle_state, "runtime_diagnostics"):
        _record_async_warning(
            async_result,
            "runtime_device_spec_unavailable",
        )

    if outstanding_is_zero and runtime_unload_safe is True:
        lifecycle_state["phase"] = "runtime_unload"
        lifecycle_state["runtime_unload_attempted"] = True
        _debug_lifecycle(args, "runtime_unload", "start", reservation)
        runtime.unload()
        lifecycle_state["runtime_unloaded"] = True
        _debug_lifecycle(args, "runtime_unload", "complete", reservation)

    lifecycle_state["phase"] = "sidecar_save"
    details_path = ""
    _debug_lifecycle(args, "sidecar_save", "start", reservation)
    try:
        details_file = save_async_details(
            reservation.run_id,
            async_result.details,
            results_dir=reservation.results_root,
            reservation=reservation,
        )
    except Exception as exc:
        _debug_lifecycle(
            args,
            "sidecar_save",
            "failed",
            reservation,
            error_type=_failure_diagnostic(
                exc, "save_async_details"
            )["error_type"],
        )
        persistence_failed = True
        diagnostic = _safe_persistence_error("save_async_details", exc)
        _record_async_persistence_failure(
            async_result,
            "async_details_persistence_failed",
            diagnostic,
        )
        if _diagnostic_proves_commit(diagnostic):
            lifecycle_state["sidecar_committed"] = True
            details_path = _artifact_reference(
                reservation.details_path,
                reservation.results_root,
            )
        _safe_print(
            f"[Error] async detail 저장 실패: "
            f"{_render_persistence_error(diagnostic)}",
            file=sys.stderr,
        )
        raise
    else:
        lifecycle_state["sidecar_committed"] = True
        details_path = _artifact_reference(
            details_file,
            reservation.results_root,
        )
        _debug_lifecycle(
            args,
            "sidecar_save",
            "complete",
            reservation,
            details_path=details_file,
        )

    invalid_reasons = sorted(
        set(async_result.invalid_reasons)
        | set(async_result.details.get("invalid_reasons", []))
    )
    async_status = (
        RunStatus.INVALID.value
        if invalid_reasons or persistence_failed
        else async_result.status.value
    )
    csv_saved = False
    save_kwargs = _result_save_kwargs(
        args, results, task_name, target_meta
    )
    save_kwargs.update(
        max_steps=None,
        results_path=reservation.results_path,
        run_id=reservation.run_id,
        inference_mode="async_queue",
        scenario=config.scenario.value,
        queue_capacity=config.queue_capacity,
        worker_count=config.worker_count,
        batch_timeout_ms=config.batch_timeout_ms,
        target_qps=config.target_qps,
        schedule_seed=config.schedule_seed,
        async_run_status=async_status,
        async_invalid_reasons=",".join(invalid_reasons),
        details_path=details_path,
        request_trace_path=trace_path,
        reservation=reservation,
    )
    lifecycle_state["normal_csv_save_kwargs"] = dict(save_kwargs)
    lifecycle_state["phase"] = "csv_save"
    _debug_lifecycle(args, "csv_save", "start", reservation)
    try:
        run_id = save_result(**save_kwargs)
        csv_saved = (
            type(run_id) is str and run_id == reservation.run_id
        )
        if not csv_saved:
            persistence_failed = True
            diagnostic = {
                "phase": "save_result",
                "error_type": "RunIdMismatch",
                "error_message": (
                    "reserved CSV save returned an unexpected run ID"
                ),
            }
            _debug_lifecycle(
                args,
                "csv_save",
                "failed",
                reservation,
                error_type="RunIdMismatch",
            )
            _safe_print(
                f"[Error] 결과 CSV 저장 실패: "
                f"{_render_persistence_error(diagnostic)}",
                file=sys.stderr,
            )
        else:
            lifecycle_state["csv_committed"] = True
            _debug_lifecycle(
                args,
                "csv_save",
                "complete",
                reservation,
            )
    except Exception as exc:
        _debug_lifecycle(
            args,
            "csv_save",
            "failed",
            reservation,
            error_type=_failure_diagnostic(
                exc, "save_result"
            )["error_type"],
        )
        persistence_failed = True
        run_id = reservation.run_id
        diagnostic = _safe_persistence_error("save_result", exc)
        _safe_print(
            f"[Error] 결과 CSV 저장 실패: "
            f"{_render_persistence_error(diagnostic)}",
            file=sys.stderr,
        )
        raise

    if csv_saved:
        _safe_print(f"\n[ResultStore] 결과 저장 완료 (run_id: {run_id})")
        _safe_print(f"[ResultStore] 파일: {actual_results_path}")
        lifecycle_state["terminal_emitted"] = _safe_print(
            f"RUN_ID={reservation.run_id}", flush=True
        )

    if not outstanding_is_zero:
        _safe_print(
            "[Error] runtime unload skipped: outstanding request state "
            "is unresolved",
            file=sys.stderr,
        )
        return 1
    if async_status == RunStatus.INVALID.value or persistence_failed:
        return 1
    return 0


def execute_benchmark(
    args: argparse.Namespace,
    *,
    loader,
    runtime,
    evaluator,
    decoder,
    hw_monitor,
    task_name: str,
    target_meta: dict,
    results_path: Path | None = None,
) -> int:
    """Run one selected benchmark mode and persist its linked artifacts."""
    validate_async_args(args)
    actual_results_path = (
        Path(results_path)
        if results_path is not None
        else FRAMEWORK_ROOT / "results" / "benchmark_results.csv"
    )
    if args.inference_mode == "e2e":
        runner = BenchmarkRunner(
            dataloader=loader,
            runtime=runtime,
            evaluator=evaluator,
            max_new_tokens=args.max_new_tokens,
            monitor=hw_monitor,
            decoder=decoder,
        )
        results = runner.run(
            warmup_runs=args.warmup,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
        )
        _print_final_metrics(args.model, results)
        save_kwargs = _result_save_kwargs(
            args, results, task_name, target_meta
        )
        if results_path is not None:
            save_kwargs["results_path"] = Path(results_path)
        run_id = save_result(**save_kwargs)
        print(f"\n[ResultStore] 결과 저장 완료 (run_id: {run_id})")
        print(f"[ResultStore] 파일: {actual_results_path}")
        print(f"RUN_ID={run_id}", flush=True)
        runtime.unload()
        return 0

    config = build_async_config(args)
    reservation = None
    engine = None
    trace_writer = None
    phase = "reservation"
    lifecycle_state = {
        "phase": phase,
        "measurement_started": False,
        "trace_closed": False,
        "runtime_unloaded": False,
        "runtime_unload_attempted": False,
        "outstanding_zero_proven": None,
        "sidecar_committed": False,
        "csv_committed": False,
        "terminal_emitted": False,
    }
    _debug_lifecycle(args, phase, "start")
    try:
        reservation = reserve_run_artifacts(results_path=actual_results_path)
        _safe_print(f"RUN_ID_RESERVED={reservation.run_id}", flush=True)
        _debug_lifecycle(
            args,
            "reservation",
            "complete",
            reservation,
            results_path=reservation.results_path,
            details_path=reservation.details_path,
            trace_path=reservation.trace_path,
        )
        if args.save_request_trace:
            phase = "trace_start"
            lifecycle_state["phase"] = phase
            _debug_lifecycle(args, phase, "start", reservation)
            trace_writer = RequestTraceWriter(
                reservation.trace_path,
                reservation=reservation,
            )
            trace_writer.start()
            _debug_lifecycle(args, phase, "complete", reservation)

        phase = "runner_setup"
        lifecycle_state["phase"] = phase
        _debug_lifecycle(args, phase, "start", reservation)
        runtime_executor = _build_async_runtime_executor(
            args,
            runtime,
            loader,
            config,
        )
        engine = InferenceEngine(
            dataloader=loader,
            runtime=runtime,
            evaluator=evaluator,
            max_new_tokens=args.max_new_tokens,
            decoder=decoder,
            trace_callback=(
                trace_writer.write if trace_writer is not None else None
            ),
            lifecycle_callback=(
                (
                    lambda lifecycle_phase: _debug_lifecycle(
                        args,
                        lifecycle_phase,
                        "start",
                        reservation,
                    )
                )
                if args.debug
                else None
            ),
            runtime_executor=runtime_executor,
        )
        _debug_lifecycle(args, phase, "complete", reservation)
        phase = "runner_run"
        lifecycle_state["phase"] = phase
        _debug_lifecycle(args, phase, "start", reservation)
        async_result = engine.run_async(
            config,
            warmup_runs=args.warmup,
            monitor=hw_monitor,
        )
        lifecycle_state["measurement_started"] = True
        _record_async_outstanding_zero_proof(
            async_result,
            lifecycle_state,
        )
        lifecycle_state["runtime_diagnostics"] = (
            _safe_runtime_diagnostics(runtime)
        )
        _debug_lifecycle(args, phase, "complete", reservation)
        runtime_unload_safe = False
        if dict.get(lifecycle_state, "outstanding_zero_proven") is True:
            runtime_unload_safe = (
                engine.runtime_unload_safe_after_failure
            )
        return _complete_async_benchmark(
            args=args,
            config=config,
            reservation=reservation,
            trace_writer=trace_writer,
            async_result=async_result,
            runtime=runtime,
            runtime_unload_safe=runtime_unload_safe,
            task_name=task_name,
            target_meta=target_meta,
            actual_results_path=actual_results_path,
            lifecycle_state=lifecycle_state,
        )
    except BaseException as primary:
        failure_phase = dict.get(lifecycle_state, "phase", phase)
        if engine is not None and failure_phase in {
            "runner_setup",
            "runner_run",
        }:
            try:
                engine_phase = getattr(engine, "failure_phase", None)
            except BaseException as secondary:
                _attach_secondary(primary, "failure_phase", secondary)
            else:
                if type(engine_phase) is str and engine_phase:
                    failure_phase = engine_phase
        _debug_lifecycle(
            args,
            dict.get(lifecycle_state, "phase", phase),
            "failed",
            reservation,
        )
        if failure_phase != dict.get(lifecycle_state, "phase", phase):
            _debug_lifecycle(
                args,
                failure_phase,
                "failed",
                reservation,
            )
        if reservation is None:
            _cleanup_async_setup(
                primary,
                runtime,
                trace_writer,
                config.flush_timeout_sec,
                args,
                reservation,
            )
            raise

        runtime_diagnostics = dict.get(
            lifecycle_state,
            "runtime_diagnostics",
        )
        if type(runtime_diagnostics) is not dict:
            runtime_diagnostics = _safe_runtime_diagnostics(runtime)
        cleanup_trace_writer = (
            None
            if dict.get(lifecycle_state, "trace_closed") is True
            else trace_writer
        )
        if (
            dict.get(lifecycle_state, "runtime_unloaded") is True
            or dict.get(lifecycle_state, "runtime_unload_attempted") is True
        ):
            _close_trace_after_failure(
                primary,
                cleanup_trace_writer,
                config.flush_timeout_sec,
                args,
                reservation,
            )
        elif engine is None:
            _cleanup_async_setup(
                primary,
                runtime,
                cleanup_trace_writer,
                config.flush_timeout_sec,
                args,
                reservation,
            )
        else:
            _cleanup_async_run_failure(
                primary,
                engine,
                runtime,
                cleanup_trace_writer,
                config.flush_timeout_sec,
                args,
                reservation,
                dict.get(
                    lifecycle_state,
                    "outstanding_zero_proven",
                ),
            )
        if (
            failure_phase == "csv_save"
            and dict.get(lifecycle_state, "csv_committed") is not True
        ):
            normal_save_kwargs = dict.get(
                lifecycle_state,
                "normal_csv_save_kwargs",
            )
            try:
                reserved_result_state = get_reserved_result_state(
                    reservation
                )
            except BaseException as secondary:
                _attach_secondary(
                    primary,
                    "normal_csv_recovery",
                    secondary,
                )
                reserved_result_state = "unknown"
            if reserved_result_state == "consumed":
                lifecycle_state["csv_committed"] = True
            elif (
                reserved_result_state == "pending"
                and type(normal_save_kwargs) is dict
            ):
                try:
                    recovered_run_id = save_result(**normal_save_kwargs)
                    if (
                        type(recovered_run_id) is not str
                        or recovered_run_id != reservation.run_id
                    ):
                        raise RuntimeError(
                            "normal CSV recovery returned an unexpected run ID"
                        )
                except BaseException as secondary:
                    _attach_secondary(
                        primary,
                        "normal_csv_recovery",
                        secondary,
                    )
                else:
                    lifecycle_state["csv_committed"] = True
        try:
            failure_csv_saved = _persist_async_failure(
                args=args,
                config=config,
                reservation=reservation,
                primary=primary,
                phase=failure_phase,
                measurement_started=(
                    dict.get(lifecycle_state, "measurement_started") is True
                    or failure_phase
                    in {"measurement", "finalization", "complete"}
                ),
                runtime_diagnostics=runtime_diagnostics,
                task_name=task_name,
                target_meta=target_meta,
                primary_details_committed=(
                    dict.get(lifecycle_state, "sidecar_committed") is True
                ),
                csv_committed=(
                    dict.get(lifecycle_state, "csv_committed") is True
                ),
            )
        except BaseException as secondary:
            diagnostic = _safe_persistence_error(
                "failure_persistence",
                secondary,
            )
            diagnostic["phase"] = "failure_persistence"
            _attach_secondary(
                primary,
                "failure_persistence",
                secondary,
            )
            _debug_lifecycle(
                args,
                "failure_persistence",
                "failed",
                reservation,
                error_type=dict.get(
                    diagnostic, "error_type", "<unknown>"
                ),
            )
            _safe_print(
                f"[Error] async failure persistence 실패: "
                f"{_render_persistence_error(diagnostic)}",
                file=sys.stderr,
            )
            failure_csv_saved = False
        if (
            failure_csv_saved
            and dict.get(lifecycle_state, "terminal_emitted") is not True
        ):
            lifecycle_state["terminal_emitted"] = _safe_print(
                f"RUN_ID={reservation.run_id}", flush=True
            )
        raise

def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_async_args(args)
        if args.inference_mode == "async_queue":
            build_async_config(args)
    except ValueError as exc:
        parser.error(str(exc))
    device_was_default = args.device == parser.get_default("device")
    layout_was_default = args.layout == parser.get_default("layout")
    
    # [설계 개선] CLI 인자(--task)에 의존하지 않고, 레지스트리(SUPPORTED_PROFILES)에서 태스크를 자동 추론 (DRY 원칙)
    from core.model_profiles import SUPPORTED_PROFILES
    profile = SUPPORTED_PROFILES.get(args.model)
    if not profile:
        print(f"[Error] '{args.model}' 프로필을 찾을 수 없습니다. model_profiles.py에 등록되었는지 확인하세요.")
        sys.exit(1)

    try:
        target = resolve_target(args.target, args.backend, args.device)
        # --target이 지정되면 target registry의 runtime/device가 실행 기준이다.
        if args.target:
            args.backend = target.runtime_name
            args.device = target.device
        elif target.runtime_name == "furiosa_llm":
            args.backend = target.runtime_name
            if device_was_default:
                args.device = target.device
        elif target.runtime_name == "hailort" and device_was_default:
            args.device = target.device
        if target.runtime_name == "hailort" and layout_was_default:
            args.layout = "NHWC"
    except Exception as e:
        print(f"[Error] target 해석 실패: {e}")
        sys.exit(1)

    try:
        cli_compile_options = parse_key_value_options(args.compile_option)
        cli_runtime_options = parse_key_value_options(args.runtime_option, coerce_values=True)
        if args.backend == "furiosa_llm":
            _validate_furiosa_runtime_options(cli_runtime_options)
    except ValueError as e:
        print(f"[Error] 옵션 파싱 실패: {e}")
        sys.exit(1)

    compile_options = {**target.compiler_options, **cli_compile_options}

        
    # 누락된 인자(default) 주입 (Zero-Config)
    # onnx 경로: default_onnx_path가 있으면 우선 사용, 없으면 default_model_path로 fallback
    if args.onnx is None:
        if "default_onnx_path" in profile:
            args.onnx = _resolve_framework_path(profile["default_onnx_path"])
        elif "default_model_path" in profile:
            args.onnx = _resolve_framework_path(profile["default_model_path"])
    # vllm 모델 경로: 항상 default_model_path (safetensors 폴더)
    if args.model_path is None and "default_model_path" in profile:
        args.model_path = _resolve_framework_path(profile["default_model_path"])
    if args.dataset is None and "default_dataset_path" in profile:
        args.dataset = _resolve_framework_path(profile["default_dataset_path"])
        
    # 토크나이저 경로 자동 추론 (NLP 태스크용)
    if args.tokenizer_path is None:
        if args.backend in ("vllm", "furiosa_llm") and args.model_path:
            args.tokenizer_path = args.model_path
        elif args.onnx:
            # ONNX 파일 경로면 부모 디렉토리를 토크나이저 경로로 간주
            args.tokenizer_path = os.path.dirname(args.onnx) if args.onnx.endswith(".onnx") else args.onnx

    # 사전 컴파일 artifact target은 모델 자동 다운로드보다 artifact 경로 검증이 먼저다.
    if args.backend == "furiosa_llm":
        try:
            _validate_furiosa_cli(args, profile["task"])
        except ValueError as exc:
            print(f"[Error] {exc}")
            sys.exit(1)
    elif args.backend == "hailort":
        args.hef = args.hef or args.artifact
        if not args.hef:
            print("[Error] hailort 백엔드에는 --hef 또는 --artifact 경로가 필요합니다.")
            sys.exit(1)
        if not os.path.exists(args.hef):
            print(f"[Error] HEF 파일을 찾을 수 없습니다: {args.hef}")
            sys.exit(1)
    elif target.uses_compiler and not args.compile and target.artifact_format not in ("onnx", "hf_model"):
        if not args.artifact:
            print(f"[Error] target '{target.target_id}'에서 --no-compile을 사용하려면 --artifact 경로가 필요합니다. "
                  f"(artifact_format={target.artifact_format})")
            sys.exit(1)
        if not os.path.exists(args.artifact):
            print(f"[Error] artifact 파일을 찾을 수 없습니다: {args.artifact}")
            sys.exit(1)
    elif not target.uses_compiler and target.artifact_format not in ("onnx", "hf_model"):
        if not args.artifact:
            print(f"[Error] target '{target.target_id}'에는 --artifact 경로가 필요합니다. "
                  f"(artifact_format={target.artifact_format})")
            sys.exit(1)
        if not os.path.exists(args.artifact):
            print(f"[Error] artifact 파일을 찾을 수 없습니다: {args.artifact}")
            sys.exit(1)

    # 리소스 누락 시 백그라운드 준비 스크립트 실행 (Auto-Prepare)
    run_auto_prepare(profile, args, target)
    
    # 백엔드별 필수 인자 검증
    if args.backend == "furiosa_llm":
        pass
    elif args.backend == "vllm":
        if not args.model_path:
            print("[Error] vllm 백엔드에는 --model-path가 필요합니다.")
            sys.exit(1)
    elif args.backend == "hailort":
        pass
    elif target.uses_compiler and not args.compile and target.artifact_format not in ("onnx", "hf_model"):
        if args.onnx and os.path.exists(args.onnx) and os.path.isdir(args.onnx):
            candidate = os.path.join(args.onnx, "model.onnx")
            if os.path.exists(candidate):
                print(f"[Info] --onnx에 디렉토리가 지정되었습니다. {candidate} 를 스펙 파싱에 사용합니다.")
                args.onnx = candidate
    elif not target.uses_compiler and target.artifact_format not in ("onnx", "hf_model"):
        pass
    else:
        if not args.onnx:
            print("[Error] onnxruntime/iree 또는 compile target에는 --onnx가 필요합니다.")
            sys.exit(1)
        if not os.path.exists(args.onnx):
            print(f"[Error] 모델 파일을 찾을 수 없습니다: {args.onnx}")
            sys.exit(1)
        # 디렉토리가 넘어온 경우 model.onnx 자동 탐색 (HuggingFace 다운로드 폴더 구조 대응)
        if os.path.isdir(args.onnx):
            candidate = os.path.join(args.onnx, "model.onnx")
            if os.path.exists(candidate):
                print(f"[Info] --onnx에 디렉토리가 지정되었습니다. {candidate} 를 사용합니다.")
                args.onnx = candidate
            else:
                print(f"[Error] 디렉토리 {args.onnx} 에서 model.onnx를 찾을 수 없습니다.")
                sys.exit(1)
    
    task_enum = profile["task"]

    # 백엔드-태스크 호환성 검증: vllm은 NLP_GENERATION 전용
    if args.backend in ("vllm", "furiosa_llm") and task_enum != Task.NLP_GENERATION:
        print(f"[Error] {args.backend} 백엔드는 NLP_GENERATION 태스크만 지원합니다. "
              f"모델 '{args.model}'의 태스크는 {task_enum.name}입니다. "
              f"onnxruntime 백엔드를 사용하세요: --backend onnxruntime")
        sys.exit(1)

    print("\n" + "="*60)
    print(f" BenchmarkRunner CLI ")
    print(f"   Model: {args.model} | Task: {task_enum.name} | Layout: {args.layout}")
    print(f"   Target: {target.target_id} | Runtime: {args.backend} | Device: {args.device}")
    print("="*60)
    
    # 0. DataLoader 공통 인터페이스 규약 및 CoC 해소 (Resolver)
    from utils.dataset_resolver import resolve_dataset_paths
    image_dir, label_path = resolve_dataset_paths(task_enum, args.dataset, args.image_dir, args.label_dir)
    
    loader_kwargs = {}
    if image_dir:
        loader_kwargs["image_dir"] = image_dir
    if label_path:
        loader_kwargs["label_path"] = label_path
    
    # 1. Spec & source artifact 생성
    if args.backend == "furiosa_llm":
        source_artifact_path = Path(args.model_path)
        spec_source_format = "hf_model"
        sniff_onnx = False
    elif args.backend == "vllm":
        source_artifact_path = Path(args.model_path)
        spec_source_format = "hf_model"
        sniff_onnx = False
    elif args.backend == "hailort":
        source_artifact_path = Path(args.hef)
        spec_source_format = "hef"
        sniff_onnx = False
    elif target.uses_compiler and not args.compile and target.artifact_format not in ("onnx", "hf_model"):
        if args.onnx and os.path.exists(args.onnx):
            source_artifact_path = Path(args.onnx)
            spec_source_format = "onnx"
            sniff_onnx = True
        else:
            source_artifact_path = Path(args.artifact)
            spec_source_format = target.artifact_format
            sniff_onnx = False
    elif not target.uses_compiler and target.artifact_format not in ("onnx", "hf_model"):
        source_artifact_path = Path(args.artifact)
        spec_source_format = target.artifact_format
        sniff_onnx = False
    else:
        source_artifact_path = Path(args.onnx)
        spec_source_format = "onnx"
        sniff_onnx = True
    try:
        spec = create_model_spec(
            args.model,
            str(source_artifact_path),
            task=task_enum,
            sniff_onnx=sniff_onnx,
            source_format=spec_source_format,
        )
    except Exception as e:
        print(f"[Error] 스펙 파싱 실패: {e}")
        sys.exit(1)

    compile_metadata = {}
    if args.backend == "furiosa_llm":
        artifact_path = Path(args.fxb)
    else:
        artifact_path = (
            Path(args.artifact)
            if target.compiler_name and not args.compile and args.artifact
            else source_artifact_path
        )
    if target.compiler_name and args.compile:
        try:
            compiler = get_compiler(target.compiler_name, **compile_options)
            compile_dir = Path("artifacts") / target.target_id
            compile_result = normalize_compile_result(compiler.compile(spec, str(compile_dir)))
            artifact_path = Path(compile_result.artifact_path)
            compile_metadata = compile_result.metadata
            print(f"[Compiler] target={target.target_id} artifact={artifact_path}")
        except Exception as e:
            print(f"[Error] 컴파일 실패: {e}")
            sys.exit(1)
    elif target.compiler_name and not args.compile:
        print(f"[Compiler] --no-compile 지정됨. 원본 artifact를 runtime에 전달합니다: {artifact_path}")

    compiled_model = CompiledModel(spec=spec, backend_name=args.backend, artifact_path=artifact_path)
    
    # 2. 컴포넌트(주입 객체) 조립
    print(f"[Factory] Assembling components for {task_enum.name}...")
    # NLP_GENERATION: tokenizer_path 전달, TIME_SERIES_FORECASTING: csv_path로 dataset 직접 전달
    if task_enum == Task.NLP_GENERATION and args.tokenizer_path:
        loader_kwargs["tokenizer_path"] = args.tokenizer_path
    if task_enum == Task.TIME_SERIES_FORECASTING:
        loader_kwargs["csv_path"] = args.dataset
        # csv_path 옆 .cache_npz 폴더를 캐시 디렉토리로 자동 지정
        csv_dir = os.path.dirname(os.path.abspath(args.dataset))
        loader_kwargs["cache_dir"] = os.path.join(csv_dir, ".cache_npz")
    elif task_enum in (
        Task.IMAGE_CLASSIFICATION,
        Task.OBJECT_DETECTION,
        Task.INSTANCE_SEGMENTATION,
        Task.POSE_ESTIMATION,
    ):
        # 이미지 데이터셋 디렉토리 옆에 .cache_npz 자동 지정
        loader_kwargs["cache_dir"] = os.path.join(os.path.abspath(args.dataset), ".cache_npz")
    elif task_enum == Task.NLP_GENERATION:
        # val.json 옆 .cache_npz 자동 지정
        loader_kwargs["cache_dir"] = os.path.join(
            os.path.dirname(os.path.abspath(args.dataset)), ".cache_npz"
        )
    if task_enum == Task.OBJECT_DETECTION:
        loader_kwargs["image_preprocess_mode"] = args.image_preprocess_mode
        loader_kwargs["image_resize_mode"] = args.image_resize_mode

    if args.backend == "deepx":
        loader_kwargs.update({
            "backend": "deepx",
            "artifact_path": str(artifact_path),
            "compile_options": compile_options,
            "compile_enabled": args.compile,
            "image_preprocess_mode": args.image_preprocess_mode,
            "image_resize_mode": args.image_resize_mode,
        })
    elif args.backend == "hailort":
        loader_kwargs.update({
            "backend": "hailort",
            "image_preprocess_mode": args.image_preprocess_mode,
            "image_resize_mode": args.image_resize_mode,
        })

    loader = create_dataloader(
        model_spec=spec,
        dataset_path=args.dataset,
        layout=args.layout,
        **loader_kwargs
    )
    
    # 런타임 팩토리 로직
    try:
        runtime_kwargs = dict(target.runtime_options)
        if args.backend == "vllm":
            if args.max_model_len is not None:
                runtime_kwargs["max_model_len"] = args.max_model_len
            elif "default_max_model_len" in profile:
                runtime_kwargs["max_model_len"] = profile["default_max_model_len"]
            if args.gpu_memory_utilization is not None:
                runtime_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
            elif "default_gpu_memory_utilization" in profile:
                runtime_kwargs["gpu_memory_utilization"] = profile["default_gpu_memory_utilization"]
            if args.enforce_eager:
                runtime_kwargs["enforce_eager"] = True
            elif "default_enforce_eager" in profile:
                runtime_kwargs["enforce_eager"] = profile["default_enforce_eager"]
        if args.backend == "hailort" and "batch_size" not in cli_runtime_options:
            runtime_kwargs["batch_size"] = args.batch_size
        loader_runtime_options = loader.get_metadata().get("runtime_options", {})
        if isinstance(loader_runtime_options, dict) and loader_runtime_options:
            runtime_kwargs.update(loader_runtime_options)
            if args.backend == "deepx":
                print(f"[DeepX] Runtime input options from dataloader: {loader_runtime_options}")
        if args.backend == "hailort":
            _apply_hailo_task_runtime_defaults(runtime_kwargs, cli_runtime_options, task_enum)
        runtime_kwargs.update(cli_runtime_options)
        runtime = create_runtime(args.backend, device=args.device, **runtime_kwargs)
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)
        
    # 3. 하드웨어 모니터 생성 (모델 로드 전에 VRAM 베이스라인 캡처)
    hw_monitor = None
    if args.monitor:
        from monitors import create_hw_monitor
        hw_monitor = create_hw_monitor(
            interval=args.monitor_interval,
            device=args.device,
            collector_names=list(target.monitor_names),
            collector_options=target.monitor_options,
        )

    runtime.load(compiled_model)

    # 모델 로드 후 VRAM 스냅샷 (모델 VRAM = after_load - baseline)
    if hw_monitor:
        hw_monitor.record_after_load_vram()

    # 평가기 팩토리 로직
    evaluator_kwargs = {}
    if task_enum == Task.NLP_GENERATION and args.tokenizer_path:
        evaluator_kwargs["tokenizer_path"] = args.tokenizer_path
    if args.debug and args.inference_mode == "e2e":
        evaluator_kwargs["debug"] = True
    if task_enum == Task.TIME_SERIES_FORECASTING:
        evaluator_kwargs["dataloader"] = loader
    evaluator = create_evaluator(spec, top_k=(1, 5), **evaluator_kwargs)
    decoder_runtime_options = dict(runtime_kwargs)
    if args.inference_mode == "async_queue":
        decoder_runtime_options.pop("debug_tensors", None)
    decoder = create_decoder(
        spec,
        backend=args.backend,
        runtime_options=decoder_runtime_options,
        **evaluator_kwargs,
    )

    target_meta = target_metadata(target, compile_metadata)
    results_path = Path(args.results_path) if args.results_path else None
    return execute_benchmark(
        args,
        loader=loader,
        runtime=runtime,
        evaluator=evaluator,
        decoder=decoder,
        hw_monitor=hw_monitor,
        task_name=task_enum.name,
        target_meta=target_meta,
        results_path=results_path,
    )

if __name__ == "__main__":
    sys.exit(main())
