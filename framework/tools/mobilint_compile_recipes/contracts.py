"""Immutable contracts for the non-BERT Mobilint compiler recipes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True)
class TensorContract:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class CompileRecipe:
    model: str
    variant: str
    source_id: str
    target_device: str
    inference_scheme: str
    compiler_inputs: tuple[TensorContract, ...]
    runtime_inputs: tuple[TensorContract, ...]
    outputs: tuple[TensorContract, ...]
    calibration_samples: int = 32
    config_preset: str | None = None
    yolo_decode_include: bool | None = None


_PATCHTST_INPUTS = (
    TensorContract("past_values", (1, 512, 7), "float32"),
    TensorContract("past_observed_mask", (1, 512, 7), "bool"),
)
_PATCHTST_OUTPUTS = (
    TensorContract("prediction_outputs", (1, 96, 7), "float32"),
)
_RESNET_COMPILER_INPUTS = (
    TensorContract("input_np", (1, 224, 224, 3), "float32"),
)
_RESNET_RUNTIME_INPUTS = (
    TensorContract("input_np", (1, 224, 224, 3), "uint8"),
)
_RESNET_OUTPUTS = (TensorContract("logits", (1, 1000), "float32"),)
_YOLO_COMPILER_INPUTS = (
    TensorContract("input_np", (1, 640, 640, 3), "float32"),
)
_YOLO_RUNTIME_INPUTS = (
    TensorContract("input_np", (1, 640, 640, 3), "uint8"),
)
_YOLO_OUTPUTS = (
    TensorContract("mobilint_yolov5_stride32", (1, 20, 20, 255), "float32"),
    TensorContract("mobilint_yolov5_stride16", (1, 40, 40, 255), "float32"),
    TensorContract("mobilint_yolov5_stride8", (1, 80, 80, 255), "float32"),
)


_RECIPES = MappingProxyType(
    {
        ("patchtst-etth1", "stock"): CompileRecipe(
            model="patchtst-etth1",
            variant="stock",
            source_id="ibm-granite/granite-timeseries-patchtst",
            target_device="aries-rb",
            inference_scheme="global8",
            compiler_inputs=_PATCHTST_INPUTS,
            runtime_inputs=_PATCHTST_INPUTS,
            outputs=_PATCHTST_OUTPUTS,
        ),
        ("patchtst-etth1", "compat-static-patchifier"): CompileRecipe(
            model="patchtst-etth1",
            variant="compat-static-patchifier",
            source_id="ibm-granite/granite-timeseries-patchtst",
            target_device="aries-rb",
            inference_scheme="global8",
            compiler_inputs=_PATCHTST_INPUTS,
            runtime_inputs=_PATCHTST_INPUTS,
            outputs=_PATCHTST_OUTPUTS,
        ),
        ("resnet50", "default"): CompileRecipe(
            model="resnet50",
            variant="default",
            source_id="torchvision.models.resnet50:IMAGENET1K_V2",
            target_device="aries-rb",
            inference_scheme="global8",
            compiler_inputs=_RESNET_COMPILER_INPUTS,
            runtime_inputs=_RESNET_RUNTIME_INPUTS,
            outputs=_RESNET_OUTPUTS,
            config_preset="classification_torchvision",
        ),
        ("yolov5m", "default"): CompileRecipe(
            model="yolov5m",
            variant="default",
            source_id="ultralytics/yolov5@86fd1ab270cb2f7e53ee7412cd4a0650bf4bcc51",
            target_device="aries-rb",
            inference_scheme="global8",
            compiler_inputs=_YOLO_COMPILER_INPUTS,
            runtime_inputs=_YOLO_RUNTIME_INPUTS,
            outputs=_YOLO_OUTPUTS,
            config_preset="yolo_640",
            yolo_decode_include=False,
        ),
    }
)


def get_recipe(model: str, variant: str) -> CompileRecipe:
    """Return the immutable recipe registered for an exact model/variant key."""
    try:
        return _RECIPES[(model, variant)]
    except KeyError as error:
        raise ValueError(
            f"unsupported Mobilint compile recipe: {model!r}/{variant!r}"
        ) from error


def _tensor_to_dict(tensor: TensorContract) -> dict[str, object]:
    return {"name": tensor.name, "shape": list(tensor.shape), "dtype": tensor.dtype}


def contract_to_dict(recipe: CompileRecipe) -> dict[str, object]:
    """Return a stable, JSON-safe representation of a recipe's ABI."""
    return {
        "model": recipe.model,
        "variant": recipe.variant,
        "source_id": recipe.source_id,
        "target_device": recipe.target_device,
        "inference_scheme": recipe.inference_scheme,
        "compiler_inputs": [
            _tensor_to_dict(tensor) for tensor in recipe.compiler_inputs
        ],
        "runtime_inputs": [
            _tensor_to_dict(tensor) for tensor in recipe.runtime_inputs
        ],
        "outputs": [_tensor_to_dict(tensor) for tensor in recipe.outputs],
        "calibration_samples": recipe.calibration_samples,
        "config_preset": recipe.config_preset,
        "yolo_decode_include": recipe.yolo_decode_include,
    }


def select_even_indices(total: int, count: int) -> tuple[int, ...]:
    """Choose deterministic, endpoint-inclusive indices without rounding drift."""
    if type(total) is not int or total <= 0:
        raise ValueError("total must be a positive integer")
    if type(count) is not int or count <= 0 or count > total:
        raise ValueError("count must be a positive integer no greater than total")
    if count == 1:
        return (0,)
    return tuple(index * (total - 1) // (count - 1) for index in range(count))


def sha256_file(path: str | Path) -> str:
    """Calculate an artifact SHA256 without loading it into memory at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
