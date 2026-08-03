"""Core contracts for strict Furiosa RNGD compile-failure reproduction."""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

import numpy as np


@dataclass(frozen=True)
class StageResult:
    name: str
    status: str
    detail: str | None = None


@dataclass(frozen=True)
class CaseResult:
    case: str
    status: str
    stages: tuple[StageResult, ...]
    output_shapes: tuple[tuple[int, ...], ...] = ()
    error_type: str | None = None
    error_line: str | None = None
    matched_known_signature: str | None = None


@dataclass(frozen=True)
class CaseConfig:
    case: str
    model_path: Path | None
    device: str = "furiosa:0"
    seed: int = 0


@dataclass(frozen=True)
class CaseDefinition:
    expected_shapes: tuple[tuple[int, ...], ...]
    loader: Callable[[CaseConfig, Any], tuple[Any, tuple[Any, ...]]]


@dataclass(frozen=True)
class Dependencies:
    torch: Any
    furiosa_torch: Any
    CompilerConfig: Any
    TacticHintConfig: Any


KNOWN_SIGNATURES = (
    "align_up_required (true) != false (false)",
    "EinsumByDpe should be given only a single pass",
    "called `Option::unwrap()` on a `None` value",
    "EdgeIndex(162) has empty transition cost table",
    "mutable op violation",
    "aten._native_batch_norm_legit",
    "Tensor device mismatch! Expected: furiosa:0, Got: cpu",
    "Cannot view a tensor with shape torch.Size([7, 512, 16, 64])",
)


def match_known_signature(text: str) -> str | None:
    """Return the first stable historical Furiosa error signature in text."""
    return next((signature for signature in KNOWN_SIGNATURES if signature in text), None)


def safe_error_line(exc: BaseException) -> str:
    """Return a bounded one-line exception summary suitable for JSON reports."""
    message = next(
        (line.strip() for line in str(exc).splitlines() if line.strip()),
        "",
    )
    summary = type(exc).__name__
    if message:
        summary = f"{summary}: {message}"
    return summary[:500]


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    """Atomically enough for diagnostics: create the directory and write JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, default=_json_default, indent=2, sort_keys=True) + "\n"
    )


def _load_dependencies() -> Dependencies:
    import torch
    import furiosa.torch as furiosa_torch
    from furiosa.torch.config import CompilerConfig, TacticHintConfig

    return Dependencies(
        torch=torch,
        furiosa_torch=furiosa_torch,
        CompilerConfig=CompilerConfig,
        TacticHintConfig=TacticHintConfig,
    )


def _as_outputs(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _validate_outputs(
    value: Any,
    expected_shapes: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    outputs = _as_outputs(value)
    if len(outputs) != len(expected_shapes):
        raise RuntimeError(
            f"output count mismatch: expected {len(expected_shapes)}, got {len(outputs)}"
        )
    shapes = []
    for output, expected_shape in zip(outputs, expected_shapes):
        array = _to_numpy(output)
        shape = tuple(array.shape)
        if shape != tuple(expected_shape):
            raise RuntimeError(
                f"output shape mismatch: expected {tuple(expected_shape)}, got {shape}"
            )
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            raise RuntimeError("output contains non-finite values")
        shapes.append(shape)
    return tuple(shapes)


def _failed_result(
    config: CaseConfig,
    stages: list[StageResult],
    stage_name: str,
    exc: BaseException,
    traceback_sink: TextIO | None,
) -> CaseResult:
    if traceback_sink is not None:
        traceback.print_exception(exc, file=traceback_sink)
    error_line = safe_error_line(exc)
    stages.append(StageResult(stage_name, "failed", error_line))
    return CaseResult(
        case=config.case,
        status="failed",
        stages=tuple(stages),
        error_type=type(exc).__name__,
        error_line=error_line,
        matched_known_signature=match_known_signature(str(exc)),
    )


def run_case(
    config: CaseConfig,
    *,
    dependencies: Dependencies | Any | None = None,
    emit: Callable[[str], None] = print,
    traceback_sink: TextIO | None = None,
) -> CaseResult:
    """Run CPU reference and the first strict RNGD call for one model case."""
    try:
        definition = CASE_DEFINITIONS[config.case]
    except KeyError:
        raise ValueError(f"Unknown Furiosa compile reproduction case: {config.case}") from None
    dependencies = dependencies or _load_dependencies()
    torch = dependencies.torch
    stages: list[StageResult] = []

    emit(f"[{config.case}] model load: START")
    try:
        model, cpu_inputs = definition.loader(config, dependencies)
        model = model.eval()
    except BaseException as exc:
        return _failed_result(config, stages, "model_load", exc, traceback_sink)
    stages.append(StageResult("model_load", "passed"))
    emit(f"[{config.case}] model load: PASS")

    emit(f"[{config.case}] CPU first inference: START")
    try:
        with torch.inference_mode():
            cpu_output = model(*cpu_inputs)
        _validate_outputs(cpu_output, definition.expected_shapes)
    except BaseException as exc:
        return _failed_result(
            config,
            stages,
            "cpu_first_inference",
            exc,
            traceback_sink,
        )
    stages.append(StageResult("cpu_first_inference", "passed"))
    emit(f"[{config.case}] CPU first inference: PASS")

    emit(f"[{config.case}] strict compile setup: START")
    try:
        torch_device = torch.device(config.device)
        model = model.to(torch_device)
        rngd_inputs = tuple(value.to(torch_device) for value in cpu_inputs)
        compiler_config = dependencies.CompilerConfig(
            tactic_hint=dependencies.TacticHintConfig.Default
        )
        backend = dependencies.furiosa_torch.backend.with_config(
            compiler_config,
            eager_fallback=False,
        )
        compiled = torch.compile(
            model,
            backend=backend,
            fullgraph=True,
            dynamic=False,
        )
    except BaseException as exc:
        return _failed_result(
            config,
            stages,
            "strict_compile_setup",
            exc,
            traceback_sink,
        )
    stages.append(StageResult("strict_compile_setup", "passed"))
    emit(f"[{config.case}] strict compile setup: PASS")

    emit(f"[{config.case}] RNGD strict compile + first inference: START")
    try:
        with torch.inference_mode():
            rngd_output = compiled(*rngd_inputs)
        output_shapes = _validate_outputs(rngd_output, definition.expected_shapes)
    except BaseException as exc:
        return _failed_result(
            config,
            stages,
            "rngd_first_inference",
            exc,
            traceback_sink,
        )
    stages.append(StageResult("rngd_first_inference", "passed"))
    emit(f"[{config.case}] RNGD strict compile + first inference: PASS")
    return CaseResult(
        case=config.case,
        status="passed",
        stages=tuple(stages),
        output_shapes=output_shapes,
    )


def fuse_conv_bn_pairs(module: Any, torch_module: Any) -> Any:
    """Fuse adjacent eval Conv2d/BatchNorm2d children without touching ReLU."""
    for _, child in tuple(module.named_children()):
        fuse_conv_bn_pairs(child, torch_module)

    children = tuple(module.named_children())
    for (conv_name, conv), (bn_name, batch_norm) in zip(children, children[1:]):
        if not isinstance(conv, torch_module.nn.Conv2d):
            continue
        if not isinstance(batch_norm, torch_module.nn.BatchNorm2d):
            continue
        fused = torch_module.nn.utils.fusion.fuse_conv_bn_eval(conv, batch_norm)
        setattr(module, conv_name, fused)
        setattr(module, bn_name, torch_module.nn.Identity())
    return module


def _load_resnet50_case(
    config: CaseConfig,
    dependencies: Dependencies | Any,
) -> tuple[Any, tuple[Any, ...]]:
    torch = dependencies.torch
    from torchvision.models import ResNet50_Weights, resnet50

    torch.manual_seed(config.seed)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    fuse_conv_bn_pairs(model, torch)
    input_tensor = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    return model, (input_tensor,)


def _require_model_path(
    config: CaseConfig,
    *,
    label: str,
    directory: bool,
) -> Path:
    if config.model_path is None:
        raise FileNotFoundError(f"{label} path was not provided")
    path = Path(config.model_path).expanduser().resolve()
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{label} {kind} does not exist: {path}")
    return path


def _load_yolov5m_case(
    config: CaseConfig,
    dependencies: Dependencies | Any,
) -> tuple[Any, tuple[Any, ...]]:
    path = _require_model_path(
        config,
        label="YOLOv5 checkpoint",
        directory=False,
    )
    if path.name != "yolov5mu.pt":
        raise ValueError(
            "YOLOv5m reproduction requires the YOLOv5u-medium checkpoint "
            f"named 'yolov5mu.pt', got {path.name!r}"
        )

    torch = dependencies.torch
    from ultralytics import YOLO

    yolo = YOLO(str(path))
    yolo.fuse()
    base = yolo.model.eval()

    class RawPrediction(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, images):
            output = self.model(images)
            if isinstance(output, (tuple, list)):
                return output[0]
            return output

    input_tensor = torch.zeros(1, 3, 640, 640, dtype=torch.float32)
    return RawPrediction(base).eval(), (input_tensor,)


def _load_patchtst_case(
    config: CaseConfig,
    dependencies: Dependencies | Any,
) -> tuple[Any, tuple[Any, ...]]:
    path = _require_model_path(
        config,
        label="PatchTST-FM model",
        directory=True,
    )
    torch = dependencies.torch
    try:
        from tsfm_public.models.patchtst_fm import (
            PatchTSTFMForPrediction,
            modeling_patchtst_fm,
        )
    except ImportError as exc:
        raise RuntimeError(
            "PatchTST-FM reproduction requires granite-tsfm==0.3.6."
        ) from exc

    try:
        ignored_loggers = torch._dynamo.config.ignore_logger_methods
    except AttributeError as exc:
        raise RuntimeError(
            "PatchTST-FM reproduction requires "
            "torch._dynamo.config.ignore_logger_methods."
        ) from exc
    ignored_loggers.add(modeling_patchtst_fm.logger.info)

    base = PatchTSTFMForPrediction.from_pretrained(
        path,
        local_files_only=True,
    ).eval()

    class Prediction(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, past_values, past_observed_mask):
            output = self.model(
                past_values=past_values,
                past_observed_mask=past_observed_mask,
                prediction_length=96,
                return_dict=True,
            )
            return output.prediction_outputs

    values = torch.zeros(1, 512, 7, dtype=torch.float32)
    observed = torch.ones(1, 512, 7, dtype=torch.bool)
    return Prediction(base).eval(), (values, observed)


CASE_DEFINITIONS: dict[str, CaseDefinition] = {
    "resnet50": CaseDefinition(
        expected_shapes=((1, 1000),),
        loader=_load_resnet50_case,
    ),
    "yolov5m": CaseDefinition(
        expected_shapes=((1, 84, 8400),),
        loader=_load_yolov5m_case,
    ),
    "patchtst": CaseDefinition(
        expected_shapes=((1, 96, 7),),
        loader=_load_patchtst_case,
    ),
}


__all__ = [
    "CASE_DEFINITIONS",
    "CaseConfig",
    "CaseDefinition",
    "CaseResult",
    "Dependencies",
    "KNOWN_SIGNATURES",
    "StageResult",
    "match_known_signature",
    "safe_error_line",
    "fuse_conv_bn_pairs",
    "write_json",
]
