"""실행 없이 웹 요청의 정적 계약과 경로를 확인한다."""

import os
from pathlib import Path

from ..schemas.benchmark import BenchmarkRunRequest
from .benchmark_service import FRAMEWORK_DIR, get_available_profiles, get_available_targets
from .command_builder import build_cli_args
from .runner_profiles import select_runner_profile


ASSET_FIELDS = (
    "onnx_path", "hef_path", "artifact_path", "mobilint_bert_weights",
    "fxb_path", "tokenizer_path", "dataset_path",
)


def _roots(env_name: str, fallback: Path) -> list[Path]:
    value = os.getenv(env_name)
    return [Path(part).expanduser().resolve() for part in value.split(os.pathsep) if part] if value else [fallback.resolve()]


def _check_path(value: str, roots: list[Path], base: Path) -> tuple[Path, bool]:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve(strict=False)
    return resolved, any(resolved.is_relative_to(root) for root in roots)


def preview_benchmark(request: BenchmarkRunRequest) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    asset_checks: dict[str, dict] = {}
    model_tasks = {
        profile["model_name"]: profile["task"] for profile in get_available_profiles()
    }
    model_names = set(model_tasks)
    if request.model not in model_names:
        errors.append(f"등록되지 않은 모델: {request.model}")
    targets = {target["target_id"]: target for target in get_available_targets()}
    target = targets.get(request.target_id) if request.target_id else None
    if request.target_id and target is None:
        errors.append(f"등록되지 않은 target: {request.target_id}")
    if request.target_id in {"rbln-static", "rbln-vllm"}:
        if request.batch_size not in (None, 1):
            errors.append(f"{request.target_id} requires batch size exactly 1")
        if request.target_id == "rbln-vllm" and model_tasks.get(request.model) != "NLP_GENERATION":
            errors.append("rbln-vllm은 NLP_GENERATION 모델만 지원합니다")
        if request.target_id == "rbln-static" and model_tasks.get(request.model) == "NLP_GENERATION":
            errors.append("rbln-static은 텍스트 생성 모델을 지원하지 않습니다")
    if request.target_id == "furiosa-rngd-torch":
        if request.compile is False:
            errors.append("furiosa-rngd-torch는 --no-compile을 지원하지 않습니다")
        if request.batch_size not in (None, 1):
            errors.append("furiosa-rngd-torch requires batch size exactly 1")
        if request.worker_count not in (None, 1):
            errors.append("furiosa-rngd-torch requires worker count exactly 1")

    profile = None
    try:
        profile = select_runner_profile(request.target_id, request.model, request.runner_profile_id)
    except (ValueError, OSError) as exc:
        errors.append(str(exc))

    asset_roots = _roots("WEB_ASSET_ROOTS", FRAMEWORK_DIR)
    for field in ASSET_FIELDS:
        value = getattr(request, field)
        if not value:
            continue
        resolved, allowed = _check_path(value, asset_roots, FRAMEWORK_DIR)
        exists = resolved.exists()
        asset_checks[field] = {"path": str(resolved), "allowed": allowed, "exists": exists}
        if not allowed:
            errors.append(f"{field}: 허용된 자산 경로 밖입니다")
        elif not exists:
            errors.append(f"{field}: 파일 또는 디렉터리가 없습니다")

    if request.model_path:
        value = request.model_path
        is_local = request.model_path_kind == "local" or (
            request.model_path_kind is None and (
                Path(value).is_absolute() or value.startswith((".", "~"))
                or (FRAMEWORK_DIR / value).exists()
            )
        )
        if is_local:
            resolved, allowed = _check_path(value, asset_roots, FRAMEWORK_DIR)
            asset_checks["model_path"] = {
                "path": str(resolved), "allowed": allowed, "exists": resolved.exists(),
            }
            if not allowed:
                errors.append("model_path: 허용된 자산 경로 밖입니다")
            elif not resolved.exists():
                errors.append("model_path: 파일 또는 디렉터리가 없습니다")
        else:
            if Path(value).is_absolute() or ".." in Path(value).parts:
                errors.append("model_path: Hugging Face ID 형식이 아닙니다")
            asset_checks["model_path"] = {"path": value, "kind": "huggingface_id"}

    if request.results_path:
        results_roots = _roots("WEB_RESULTS_ROOTS", FRAMEWORK_DIR / "results")
        resolved, allowed = _check_path(request.results_path, results_roots, FRAMEWORK_DIR)
        if not allowed:
            errors.append("results_path: 허용된 결과 경로 밖입니다")
        elif resolved.suffix.lower() != ".csv":
            errors.append("results_path: CSV 확장자가 필요합니다")

    warnings.append("SDK와 실제 장치 상태는 실행 전 별도 확인이 필요합니다")
    argv = (
        [str(profile.python), str(FRAMEWORK_DIR / "src" / "main.py"), *build_cli_args(request)]
        if profile else []
    )
    return {
        "argv": argv,
        "errors": errors,
        "warnings": warnings,
        "resolved_target": target,
        "runner_profile": profile.public() if profile else None,
        "asset_checks": asset_checks,
    }
