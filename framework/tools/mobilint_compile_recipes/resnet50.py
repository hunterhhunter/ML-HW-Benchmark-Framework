"""Reproducible uint8-NHWC Mobilint ResNet-50 compiler recipe."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

import torch

from tools.mobilint_compile_recipes.compiler import run_mblt_compile, run_mxq_compile
from tools.mobilint_compile_recipes.contracts import (
    contract_to_dict,
    get_recipe,
    select_even_indices,
    sha256_file,
)


MODEL = "resnet50"
VARIANT = "default"
WEIGHT_ENUM = "IMAGENET1K_V2"
CALIBRATION_SAMPLES = 32
INPUT_NAME = "input_np"
INPUT_SHAPE = (1, 224, 224, 3)
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _recipe():
    return get_recipe(MODEL, VARIANT)


class ResNet50SourceWrapper(torch.nn.Module):
    """Adapt float unit-range NHWC compiler input to TorchVision ResNet input."""

    def __init__(self, source_model):
        super().__init__()
        self.source_model = source_model.eval()
        self.source_model.requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor(_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(_STD, dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, input_np):
        nchw = input_np.permute(0, 3, 1, 2)
        return self.source_model((nchw - self.mean) / self.std)


def validate_compiler_input(input_np) -> None:
    """Eagerly enforce the compiler's float32, unit-range NHWC boundary."""
    if input_np.dtype != torch.float32:
        raise ValueError("ResNet compiler input must be float32 unit-range NHWC")
    if tuple(input_np.shape) != INPUT_SHAPE:
        raise ValueError(
            "ResNet compiler input must have shape "
            f"{INPUT_SHAPE}, got {tuple(input_np.shape)}"
        )
    if not torch.isfinite(input_np).all():
        raise ValueError("ResNet compiler input must be finite")
    if (input_np < 0).any() or (input_np > 1).any():
        raise ValueError("ResNet compiler input must be in the unit range")


def preprocess_calibration_image(image):
    """Apply the runtime's raw RGB ResNet geometry and return batched NHWC uint8."""
    import numpy as np
    from preprocessor.strategies import MLPerfResNet50RawPreprocess

    raw = MLPerfResNet50RawPreprocess(short_side=232)(
        image,
        target_hw=(224, 224),
        mean=np.asarray(_MEAN, dtype=np.float32),
        std=np.asarray(_STD, dtype=np.float32),
    )
    value = np.transpose(raw, (1, 2, 0)).astype(np.uint8, copy=False)[None, ...]
    return np.ascontiguousarray(value)


def _torchvision_provenance() -> dict[str, object]:
    import torch
    from torchvision.models import ResNet50_Weights

    weights = ResNet50_Weights.IMAGENET1K_V2
    local_path = (
        Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
    )
    return {
        "version": metadata.version("torchvision"),
        "weight_enum": WEIGHT_ENUM,
        "weight_url": weights.url,
        "local_weight_path": str(local_path) if local_path.is_file() else None,
        "local_weight_sha256": sha256_file(local_path) if local_path.is_file() else None,
    }


def _sorted_images(dataset_path: str | Path) -> list[Path]:
    from PIL import Image

    root = Path(dataset_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ImageNet validation directory not found: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if len(files) < CALIBRATION_SAMPLES:
        raise ValueError(
            f"ImageNet validation directory has only {len(files)} files; "
            f"need at least {CALIBRATION_SAMPLES}"
        )
    for path in files:
        try:
            with Image.open(path) as image:
                image.convert("RGB")
        except (OSError, ValueError) as error:
            raise ValueError(f"calibration file is not a readable RGB image: {path}") from error
    return files


def _compile_report(recipe) -> dict[str, object]:
    return {
        **contract_to_dict(recipe),
        "calibration_path": "calibration/calibration.json",
        "compiler_options": {
            "mblt": {
                "target_device": recipe.target_device,
                "backend": "torch",
                "cpu_offload": True,
            },
            "mxq": {
                "target_device": recipe.target_device,
                "backend": "torch",
                "inference_scheme": recipe.inference_scheme,
                "config_preset": recipe.config_preset,
                "uint8_input_config": {
                    "apply": True,
                    "inputs": [INPUT_NAME],
                    "division_factor": 255.0,
                },
                "calibration": {
                    "method": 1,
                    "output": 0,
                    "mode": 1,
                    "max_percentile": 0.999,
                    "topk_ratio": 0.01,
                },
            },
        },
        "source_smoke": None,
        "active_compiler_stage": None,
        "resolved_mxq_preset": None,
        "artifacts": {},
    }


def prepare_calibration(dataset_path: str | Path, attempt_root: str | Path) -> dict[str, object]:
    """Select 32 sorted ImageNet validation images and write raw uint8 NHWC arrays."""
    import numpy as np
    from PIL import Image

    recipe = _recipe()
    root = Path(attempt_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"attempt root does not exist: {root}")
    calibration_root = root / "calibration"
    manifest_path = root / "source-manifest.json"
    report_path = root / "compile-report.json"
    if calibration_root.exists() or manifest_path.exists() or report_path.exists():
        raise FileExistsError(f"ResNet calibration output already exists: {root}")

    files = _sorted_images(dataset_path)
    indices = select_even_indices(len(files), recipe.calibration_samples)
    selected = [files[index] for index in indices]
    values: list[object] = []
    for path in selected:
        with Image.open(path) as image:
            values.append(preprocess_calibration_image(image.convert("RGB")))

    calibration_root.mkdir()
    paths: list[list[str]] = []
    samples: list[dict[str, object]] = []
    for ordinal, (index, path, value) in enumerate(zip(indices, selected, values)):
        array_path = calibration_root / f"{ordinal:03d}.npy"
        np.save(array_path, value, allow_pickle=False)
        relative_path = array_path.relative_to(root).as_posix()
        paths.append([str(array_path)])
        samples.append(
            {
                "ordinal": ordinal,
                "dataset_index": index,
                "source_path": str(path),
                "source_sha256": sha256_file(path),
                "calibration_path": relative_path,
                "calibration_sha256": sha256_file(array_path),
            }
        )
    _write_json_atomic(
        calibration_root / "calibration.json",
        {"info": {"input names": [INPUT_NAME]}, "calib paths": paths},
    )
    manifest: dict[str, object] = {
        "model": recipe.model,
        "variant": recipe.variant,
        "source_id": recipe.source_id,
        "torchvision": _torchvision_provenance(),
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "dataset_file_count": len(files),
        "calibration_indices": list(indices),
        "compiler_input": {
            "name": recipe.compiler_inputs[0].name,
            "shape": list(recipe.compiler_inputs[0].shape),
            "dtype": recipe.compiler_inputs[0].dtype,
        },
        "runtime_input": {
            "name": recipe.runtime_inputs[0].name,
            "shape": list(recipe.runtime_inputs[0].shape),
            "dtype": recipe.runtime_inputs[0].dtype,
        },
        "output": {
            "name": recipe.outputs[0].name,
            "shape": list(recipe.outputs[0].shape),
            "dtype": recipe.outputs[0].dtype,
        },
        "preprocess": {
            "strategy": "MLPerfResNet50RawPreprocess",
            "short_side": 232,
            "crop_hw": [224, 224],
            "color_order": "RGB",
            "layout": "NHWC",
            "dtype": "uint8",
        },
        "samples": samples,
    }
    _write_json_atomic(manifest_path, manifest)
    _write_json_atomic(report_path, _compile_report(recipe))
    return manifest


def load_source_model():
    """Load the frozen official TorchVision IMAGENET1K_V2 model on demand."""
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    model.requires_grad_(False)
    return model


def _read_manifest(root: Path) -> dict[str, object]:
    path = root / "source-manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"ResNet source manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {"model": MODEL, "variant": VARIANT, "source_id": _recipe().source_id}
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"source manifest {field} mismatch")
    return manifest


def _read_report(root: Path) -> dict[str, object]:
    path = root / "compile-report.json"
    if not path.is_file():
        raise FileNotFoundError(f"ResNet compile report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_feed_input(root: Path, manifest: Mapping[str, object]):
    import numpy as np

    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("source manifest has no calibration samples")
    sample = samples[0]
    if not isinstance(sample, Mapping):
        raise ValueError("source manifest calibration sample is invalid")
    candidate = (root / str(sample.get("calibration_path"))).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("source manifest calibration path escapes attempt root") from error
    value = np.load(candidate, allow_pickle=False)
    if value.shape != INPUT_SHAPE or value.dtype != np.uint8:
        raise ValueError("calibration input must be uint8 NHWC [1,224,224,3]")
    compiler_input = torch.from_numpy(np.ascontiguousarray(value).copy()).float() / 255.0
    validate_compiler_input(compiler_input)
    return {INPUT_NAME: compiler_input}


def _validate_logits(output) -> None:
    import torch

    if tuple(output.shape) != (1, 1000):
        raise ValueError(f"ResNet output shape must be (1, 1000), got {tuple(output.shape)}")
    if output.dtype != torch.float32:
        raise ValueError(f"ResNet output dtype must be float32, got {output.dtype}")
    if not torch.isfinite(output).all():
        raise ValueError("ResNet output must contain only finite values")


def _default_official_transform():
    from torchvision.models import ResNet50_Weights

    return ResNet50_Weights.IMAGENET1K_V2.transforms()


def _source_smoke_record(output) -> dict[str, object]:
    return {
        "output_shape": list(output.shape),
        "output_dtype": str(output.detach().cpu().numpy().dtype),
        "finite": True,
        "official_transform_equivalence": {"rtol": 1e-5, "atol": 1e-6},
    }


def source_smoke(
    attempt_root: str | Path,
    *,
    model_loader: Callable[[], object] = load_source_model,
    official_transform=None,
):
    """Validate source logits and exact official-transform normalization pre-import."""
    import torch
    from PIL import Image

    root = Path(attempt_root).expanduser().resolve()
    manifest = _read_manifest(root)
    report = _read_report(root)
    if report.get("source_smoke") is not None:
        raise FileExistsError("ResNet source smoke evidence already exists")
    model = model_loader().eval()
    model.requires_grad_(False)
    wrapper = ResNet50SourceWrapper(model)
    feed_dict = _load_feed_input(root, manifest)
    validate_compiler_input(feed_dict[INPUT_NAME])
    with torch.no_grad():
        output = wrapper(**feed_dict)
    _validate_logits(output)

    samples = manifest["samples"]
    source_path = Path(str(samples[0]["source_path"]))
    transform = _default_official_transform() if official_transform is None else official_transform
    with Image.open(source_path) as image, torch.no_grad():
        official_input = transform(image.convert("RGB"))
        if official_input.ndim == 3:
            official_input = official_input.unsqueeze(0)
        if tuple(official_input.shape) != (1, 3, 224, 224):
            raise ValueError("official TorchVision transform must return [1,3,224,224]")
        mean = torch.tensor(_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(_STD, dtype=torch.float32).view(1, 3, 1, 1)
        wrapper_input = (official_input * std + mean).permute(0, 2, 3, 1).contiguous()
        validate_compiler_input(wrapper_input)
        expected = model(official_input)
        actual = wrapper(wrapper_input)
    _validate_logits(expected)
    _validate_logits(actual)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    report["source_smoke"] = _source_smoke_record(output)
    _write_json_atomic(root / "compile-report.json", report)
    return output.detach().cpu()


def _artifact_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_mxq_preset(name: str):
    """Resolve a qbcompiler preset only for an MXQ compiler attempt."""
    from qbcompiler.configs import get_preset

    return get_preset(name)


def _resolved_preset_evidence(preset_loader) -> dict[str, object]:
    recipe = _recipe()
    preset = preset_loader(recipe.config_preset)
    resolved = preset.model_dump(by_alias=True, exclude_none=True)
    try:
        resolved = json.loads(json.dumps(resolved))
    except (TypeError, ValueError) as error:
        raise ValueError("resolved qbcompiler preset must be JSON serializable") from error
    return {
        "name": recipe.config_preset,
        "resolved": resolved,
        "overrides": {
            "target_device": recipe.target_device,
            "inference_scheme": recipe.inference_scheme,
            "calibration_config": {
                "method": 1,
                "output": 0,
                "mode": 1,
                "max_percentile": {"percentile": 0.999, "topk_ratio": 0.01},
            },
            "uint8_input_config": {
                "apply": True,
                "inputs": [INPUT_NAME],
                "division_factor": 255.0,
            },
        },
    }


def compile_stage(
    stage: str,
    attempt_root: str | Path,
    *,
    model_loader: Callable[[], object] = load_source_model,
    official_transform=None,
    mblt_compiler=None,
    mxq_compiler_api=None,
    preset_loader: Callable[[str], object] = load_mxq_preset,
) -> Path:
    """Run one compiler stage; a failed entered stage can only be retried in a new attempt."""
    if stage not in {"mblt", "mxq"}:
        raise ValueError(f"unsupported ResNet compiler stage: {stage!r}")
    root = Path(attempt_root).expanduser().resolve()
    manifest = _read_manifest(root)
    report = _read_report(root)
    active_stage = report.get("active_compiler_stage")
    if active_stage is not None:
        raise RuntimeError(
            "ResNet compiler attempt already entered stage "
            f"{active_stage!r}; use a fresh attempt root"
        )
    output_path = root / stage / f"resnet50-{stage}.{stage}"
    artifacts = report.get("artifacts")
    if isinstance(artifacts, dict) and stage in artifacts:
        raise FileExistsError(f"ResNet compiler stage already exists: {stage}")
    if output_path.exists():
        raise FileExistsError(f"Mobilint compiler artifact already exists: {output_path}")
    if report.get("source_smoke") is None:
        source_smoke(
            root,
            model_loader=model_loader,
            official_transform=official_transform,
        )
        report = _read_report(root)

    model = model_loader().eval()
    model.requires_grad_(False)
    wrapper = ResNet50SourceWrapper(model)
    feed_dict = _load_feed_input(root, manifest)
    validate_compiler_input(feed_dict[INPUT_NAME])
    report["active_compiler_stage"] = stage
    if stage == "mxq":
        report["resolved_mxq_preset"] = _resolved_preset_evidence(preset_loader)
    _write_json_atomic(root / "compile-report.json", report)

    recipe = _recipe()
    if stage == "mblt":
        artifact = run_mblt_compile(
            recipe=recipe,
            model=wrapper,
            feed_dict=feed_dict,
            output=output_path,
            compiler=mblt_compiler,
        )
    else:
        artifact = run_mxq_compile(
            recipe=recipe,
            model=wrapper,
            feed_dict=feed_dict,
            calibration_path=root / "calibration" / "calibration.json",
            output=output_path,
            compiler_api=mxq_compiler_api,
        )
    report["active_compiler_stage"] = None
    report["artifacts"][stage] = _artifact_record(artifact, root)
    _write_json_atomic(root / "compile-report.json", report)
    return artifact


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and compile the frozen TorchVision ResNet50 experiment",
    )
    parser.add_argument(
        "--stage", required=True, choices=("describe", "prepare", "source-smoke", "mblt", "mxq")
    )
    parser.add_argument("--variant", default=VARIANT, choices=(VARIANT,))
    parser.add_argument("--attempt-root", type=Path)
    parser.add_argument("--dataset", type=Path)
    return parser


def _require_cli_path(parser, value, option: str):
    if value is None:
        parser.error(f"{option} is required for this stage")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = _create_parser()
    args = parser.parse_args(argv)
    if args.stage == "describe":
        print(json.dumps(contract_to_dict(_recipe()), indent=2))
        return 0
    attempt_root = _require_cli_path(parser, args.attempt_root, "--attempt-root")
    if args.stage == "prepare":
        dataset = _require_cli_path(parser, args.dataset, "--dataset")
        print(json.dumps(prepare_calibration(dataset, attempt_root), sort_keys=True))
    elif args.stage == "source-smoke":
        output = source_smoke(attempt_root)
        print(json.dumps({"shape": list(output.shape), "dtype": str(output.numpy().dtype)}))
    else:
        artifact = compile_stage(args.stage, attempt_root)
        print(json.dumps({"artifact": str(artifact)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
