"""Pinned raw-head YOLOv5m compiler recipe for Mobilint ARIES."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

from tools.mobilint_compile_recipes.compiler import run_mblt_compile, run_mxq_compile
from tools.mobilint_compile_recipes.contracts import (
    contract_to_dict,
    get_recipe,
    select_even_indices,
    sha256_file,
)


MODEL = "yolov5m"
VARIANT = "default"
EXPECTED_YOLOV5_REVISION = "86fd1ab270cb2f7e53ee7412cd4a0650bf4bcc51"
REQUIRED_SOURCE_FILES = ("models/experimental.py", "models/yolo.py")
CALIBRATION_SAMPLES = 32
INPUT_NAME = "input_np"
INPUT_SHAPE = (1, 640, 640, 3)
RAW_HEAD_SHAPES = (
    (1, 20, 20, 255),
    (1, 40, 40, 255),
    (1, 80, 80, 255),
)
_RAW_SOURCE_HEAD_SHAPES = (
    (1, 3, 20, 20, 85),
    (1, 3, 40, 40, 85),
    (1, 3, 80, 80, 85),
)


def _recipe():
    return get_recipe(MODEL, VARIANT)


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


def validate_sources(
    yolov5_root: str | Path, weights: str | Path
) -> tuple[Path, Path]:
    """Require the exact local YOLOv5 revision and a non-empty weight file."""
    root = Path(yolov5_root).expanduser().resolve()
    weight_path = Path(weights).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"YOLOv5 source root does not exist: {root}")
    for required_file in REQUIRED_SOURCE_FILES:
        source_file = root / required_file
        if not source_file.is_file():
            raise FileNotFoundError(
                f"YOLOv5 source root is missing required file: {source_file}"
            )
    if not weight_path.is_file():
        raise FileNotFoundError(f"YOLOv5 weight file does not exist: {weight_path}")
    if weight_path.stat().st_size == 0:
        raise ValueError(f"YOLOv5 weight file is empty: {weight_path}")

    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    )
    revision = result.stdout.strip()
    if revision != EXPECTED_YOLOV5_REVISION:
        raise RuntimeError(
            f"YOLOv5 source revision {revision} does not match validated "
            f"{EXPECTED_YOLOV5_REVISION}; run git -C {root} checkout "
            f"{EXPECTED_YOLOV5_REVISION}"
        )
    return root, weight_path


def load_source_model(
    yolov5_root: str | Path,
    weights: str | Path,
    *,
    attempt_loader=None,
):
    """Load the pinned raw YOLO model with the checkout's public loader."""
    root, weight_path = validate_sources(yolov5_root, weights)
    if attempt_loader is None:
        existing = sys.modules.get("models.experimental")
        if existing is not None:
            existing_path = Path(str(getattr(existing, "__file__", ""))).resolve()
            if existing_path != (root / "models" / "experimental.py").resolve():
                raise RuntimeError(
                    "an unrelated models.experimental module is already imported; "
                    "load YOLOv5 in a fresh process"
                )
        sys.path.insert(0, str(root))
        try:
            module = importlib.import_module("models.experimental")
            module_path = Path(str(getattr(module, "__file__", ""))).resolve()
            expected_path = (root / "models" / "experimental.py").resolve()
            if module_path != expected_path:
                raise RuntimeError(
                    f"attempt_load resolved outside the pinned checkout: {module_path}"
                )
            attempt_loader = module.attempt_load
            model = attempt_loader(str(weight_path), map_location="cpu")
        finally:
            try:
                sys.path.remove(str(root))
            except ValueError:
                pass
    else:
        model = attempt_loader(str(weight_path), map_location="cpu")
    model = model.fuse().eval()
    model.requires_grad_(False)
    return model


class YoloV5RawHeadWrapper:
    """Lazily construct the Torch module exposing only raw prediction heads."""

    def __new__(cls, source_model):
        import torch

        class _Wrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.source_model = model.eval()
                self.source_model.requires_grad_(False)

            def forward(self, input_np):
                source_output = self.source_model(input_np.permute(0, 3, 1, 2))
                if not isinstance(source_output, (tuple, list)) or len(source_output) < 2:
                    raise ValueError(
                        "YOLOv5 source must expose undecoded raw heads; decoded-only "
                        "output is unsupported"
                    )
                raw_heads = source_output[1]
                if not isinstance(raw_heads, (tuple, list)) or len(raw_heads) != 3:
                    raise ValueError("YOLOv5 source must expose exactly three raw heads")
                ordered = sorted(raw_heads, key=lambda value: int(value.shape[2]))
                actual_shapes = tuple(tuple(value.shape) for value in ordered)
                if actual_shapes != _RAW_SOURCE_HEAD_SHAPES:
                    raise ValueError(
                        "YOLOv5 raw head shape mismatch: expected "
                        f"{_RAW_SOURCE_HEAD_SHAPES}, got {actual_shapes}"
                    )
                for value in ordered:
                    if value.dtype != torch.float32:
                        raise ValueError(
                            f"YOLOv5 raw heads must be float32, got {value.dtype}"
                        )
                    if not torch.isfinite(value).all():
                        raise ValueError("YOLOv5 raw heads must contain only finite values")
                outputs = tuple(
                    value.permute(0, 2, 3, 1, 4)
                    .contiguous()
                    .reshape(value.shape[0], value.shape[2], value.shape[3], 255)
                    for value in ordered
                )
                return outputs

        return _Wrapper(source_model).eval()


def validate_compiler_input(input_np) -> None:
    """Eagerly enforce the compiler's float32 unit-range NHWC boundary."""
    import torch

    if input_np.dtype != torch.float32:
        raise ValueError("YOLOv5 compiler input must be float32 unit-range NHWC")
    if tuple(input_np.shape) != INPUT_SHAPE:
        raise ValueError(
            f"YOLOv5 compiler input must have shape {INPUT_SHAPE}, "
            f"got {tuple(input_np.shape)}"
        )
    if not torch.isfinite(input_np).all():
        raise ValueError("YOLOv5 compiler input must be finite")
    if (input_np < 0).any() or (input_np > 1).any():
        raise ValueError("YOLOv5 compiler input must be in the unit range")


def preprocess_calibration_image(image):
    """Apply the runtime's exact raw RGB YOLOv5 letterbox preprocessing."""
    import numpy as np
    from dataloader.mobilint_vision_profiles import MOBILINT_YOLOV5M_DEFAULT
    from preprocessor.mobilint_vision import MobilintYoloV5Preprocessor

    value = MobilintYoloV5Preprocessor(MOBILINT_YOLOV5M_DEFAULT).preprocess(image)
    value = np.asarray(value, dtype=np.uint8)[None, ...]
    return np.ascontiguousarray(value)


def _sorted_images(dataset_path: str | Path) -> list[Path]:
    from PIL import Image

    root = Path(dataset_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"COCO128 image directory not found: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if len(files) < CALIBRATION_SAMPLES:
        raise ValueError(
            f"COCO128 image directory has only {len(files)} files; "
            f"need at least {CALIBRATION_SAMPLES}"
        )
    for path in files:
        try:
            with Image.open(path) as image:
                image.convert("RGB")
        except (OSError, ValueError) as error:
            raise ValueError(
                f"calibration file is not a readable RGB image: {path}"
            ) from error
    return files


def _source_manifest(
    recipe,
    *,
    source_root: Path,
    weights: Path,
    dataset_path: Path,
    dataset_count: int,
    indices: Sequence[int],
    samples: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "model": recipe.model,
        "variant": recipe.variant,
        "source_id": recipe.source_id,
        "yolov5": {
            "root": str(source_root),
            "revision": EXPECTED_YOLOV5_REVISION,
            "required_files": {
                name: sha256_file(source_root / name) for name in REQUIRED_SOURCE_FILES
            },
        },
        "weights": {
            "path": str(weights),
            "size_bytes": weights.stat().st_size,
            "sha256": sha256_file(weights),
        },
        "dataset_path": str(dataset_path),
        "dataset_file_count": dataset_count,
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
        "outputs": [
            {"name": output.name, "shape": list(output.shape), "dtype": output.dtype}
            for output in recipe.outputs
        ],
        "preprocess": {
            "strategy": "MobilintYoloV5Preprocessor",
            "input_hw": [640, 640],
            "interpolation": "opencv_linear",
            "resize_rounding": "python_round",
            "padding_rounding": "ultralytics_minus_plus_0_1",
            "padding_value": 114,
            "color_order": "RGB",
            "layout": "NHWC",
            "dtype": "uint8",
        },
        "samples": samples,
    }


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
                "yolo_decode_include": recipe.yolo_decode_include,
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


def prepare_calibration(
    dataset_path: str | Path,
    attempt_root: str | Path,
    yolov5_root: str | Path,
    weights: str | Path,
) -> dict[str, object]:
    """Write 32 endpoint-inclusive sorted COCO128 RGB uint8 NHWC samples."""
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
        raise FileExistsError(f"YOLOv5 calibration output already exists: {root}")

    source_root, weight_path = validate_sources(yolov5_root, weights)
    dataset_root = Path(dataset_path).expanduser().resolve()
    files = _sorted_images(dataset_root)
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
    manifest = _source_manifest(
        recipe,
        source_root=source_root,
        weights=weight_path,
        dataset_path=dataset_root,
        dataset_count=len(files),
        indices=indices,
        samples=samples,
    )
    _write_json_atomic(manifest_path, manifest)
    _write_json_atomic(report_path, _compile_report(recipe))
    return manifest


def _read_manifest(root: Path) -> dict[str, object]:
    path = root / "source-manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"YOLOv5 source manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {"model": MODEL, "variant": VARIANT, "source_id": _recipe().source_id}
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"source manifest {field} mismatch")
    return manifest


def _read_report(root: Path) -> dict[str, object]:
    path = root / "compile-report.json"
    if not path.is_file():
        raise FileNotFoundError(f"YOLOv5 compile report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_feed_input(root: Path, manifest: Mapping[str, object]):
    import numpy as np
    import torch

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
        raise ValueError("calibration input must be uint8 NHWC [1,640,640,3]")
    compiler_input = torch.from_numpy(np.ascontiguousarray(value).copy()).float() / 255.0
    validate_compiler_input(compiler_input)
    return {INPUT_NAME: compiler_input}


def _manifest_model_loader(manifest: Mapping[str, object]):
    source = manifest.get("yolov5")
    weights = manifest.get("weights")
    if not isinstance(source, Mapping) or not isinstance(weights, Mapping):
        raise ValueError("source manifest is missing YOLOv5 source provenance")
    if source.get("revision") != EXPECTED_YOLOV5_REVISION:
        raise ValueError("source manifest YOLOv5 revision mismatch")
    source_root = Path(str(source.get("root"))).expanduser().resolve()
    recorded_files = source.get("required_files")
    if not isinstance(recorded_files, Mapping):
        raise ValueError("source manifest is missing required source file hashes")
    for name in REQUIRED_SOURCE_FILES:
        source_path = source_root / name
        if not source_path.is_file():
            raise FileNotFoundError(f"prepared YOLOv5 source file not found: {source_path}")
        if recorded_files.get(name) != sha256_file(source_path):
            raise ValueError(f"prepared YOLOv5 source SHA256 mismatch: {name}")
    weight_path = Path(str(weights.get("path"))).expanduser().resolve()
    if not weight_path.is_file():
        raise FileNotFoundError(f"prepared YOLOv5 weight file not found: {weight_path}")
    if weights.get("sha256") != sha256_file(weight_path):
        raise ValueError("prepared YOLOv5 weight SHA256 mismatch")
    if weights.get("size_bytes") != weight_path.stat().st_size:
        raise ValueError("prepared YOLOv5 weight size mismatch")
    return load_source_model(source_root, weight_path)


def _head_metadata(model) -> tuple[list[float], list[list[list[float]]]]:
    import torch

    try:
        detect = model.model[-1]
        strides_tensor = torch.as_tensor(detect.stride, dtype=torch.float32).detach().cpu()
        anchors_tensor = torch.as_tensor(detect.anchors, dtype=torch.float32).detach().cpu()
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError("YOLOv5 source is missing Detect anchors or strides") from error
    if tuple(strides_tensor.shape) != (3,):
        raise ValueError("YOLOv5 Detect strides must have shape [3]")
    if tuple(anchors_tensor.shape) != (3, 3, 2):
        raise ValueError("YOLOv5 Detect anchors must have shape [3,3,2]")
    if not torch.isfinite(strides_tensor).all() or not torch.isfinite(anchors_tensor).all():
        raise ValueError("YOLOv5 Detect anchors and strides must be finite")
    if (strides_tensor <= 0).any():
        raise ValueError("YOLOv5 Detect strides must be positive")
    pixel_anchors = anchors_tensor * strides_tensor.view(3, 1, 1)
    return strides_tensor.tolist(), pixel_anchors.tolist()


def source_smoke(
    attempt_root: str | Path,
    *,
    model_loader: Callable[[], object] | None = None,
):
    """Validate pinned source raw heads before any compiler stage is entered."""
    import torch

    root = Path(attempt_root).expanduser().resolve()
    manifest = _read_manifest(root)
    report = _read_report(root)
    if report.get("source_smoke") is not None:
        raise FileExistsError("YOLOv5 source smoke evidence already exists")
    model = _manifest_model_loader(manifest) if model_loader is None else model_loader()
    model = model.eval()
    model.requires_grad_(False)
    wrapper = YoloV5RawHeadWrapper(model)
    feed_dict = _load_feed_input(root, manifest)
    validate_compiler_input(feed_dict[INPUT_NAME])
    with torch.no_grad():
        outputs = wrapper(**feed_dict)
    strides, anchors = _head_metadata(model)
    record = {
        "output_shapes": [list(value.shape) for value in outputs],
        "output_dtypes": [str(value.detach().cpu().numpy().dtype) for value in outputs],
        "finite": True,
        "strides": strides,
        "anchors": anchors,
    }
    report["source_smoke"] = record
    _write_json_atomic(root / "compile-report.json", report)
    return tuple(value.detach().cpu() for value in outputs)


def _artifact_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_mxq_preset(name: str):
    """Resolve a public qbcompiler preset only for an MXQ attempt."""
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
            "yolo_decode_include": recipe.yolo_decode_include,
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
    model_loader: Callable[[], object] | None = None,
    mblt_compiler=None,
    mxq_compiler_api=None,
    preset_loader: Callable[[str], object] = load_mxq_preset,
) -> Path:
    """Run one immutable compiler stage for a prepared YOLOv5 attempt."""
    if stage not in {"mblt", "mxq"}:
        raise ValueError(f"unsupported YOLOv5 compiler stage: {stage!r}")
    root = Path(attempt_root).expanduser().resolve()
    manifest = _read_manifest(root)
    report = _read_report(root)
    active_stage = report.get("active_compiler_stage")
    if active_stage is not None:
        raise RuntimeError(
            "YOLOv5 compiler attempt already entered stage "
            f"{active_stage!r}; use a fresh attempt root"
        )
    output_path = root / stage / f"yolov5m-{stage}.{stage}"
    artifacts = report.get("artifacts")
    if isinstance(artifacts, dict) and stage in artifacts:
        raise FileExistsError(f"YOLOv5 compiler stage already exists: {stage}")
    if output_path.exists():
        raise FileExistsError(f"Mobilint compiler artifact already exists: {output_path}")

    if report.get("source_smoke") is None:
        source_smoke(root, model_loader=model_loader)
        report = _read_report(root)
    model = _manifest_model_loader(manifest) if model_loader is None else model_loader()
    model = model.eval()
    model.requires_grad_(False)
    wrapper = YoloV5RawHeadWrapper(model)
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
        description="Prepare and compile the pinned YOLOv5m raw-head experiment"
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("describe", "prepare", "source-smoke", "mblt", "mxq"),
    )
    parser.add_argument("--variant", default=VARIANT, choices=(VARIANT,))
    parser.add_argument("--attempt-root", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--yolov5-root", type=Path)
    parser.add_argument("--weights", type=Path)
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
        source_root = _require_cli_path(parser, args.yolov5_root, "--yolov5-root")
        weights = _require_cli_path(parser, args.weights, "--weights")
        manifest = prepare_calibration(dataset, attempt_root, source_root, weights)
        print(json.dumps(manifest, sort_keys=True))
    elif args.stage == "source-smoke":
        outputs = source_smoke(attempt_root)
        print(
            json.dumps(
                {
                    "shapes": [list(value.shape) for value in outputs],
                    "dtypes": [str(value.numpy().dtype) for value in outputs],
                }
            )
        )
    else:
        artifact = compile_stage(args.stage, attempt_root)
        print(json.dumps({"artifact": str(artifact)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
