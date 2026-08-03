"""Compile prepared BERT tasks with the Mobilint qbcompiler 1.2 API."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import re
from typing import Mapping

from tools.mobilint_bert_compile.common import (
    TaskSpec,
    contract_to_dict,
    get_task_spec,
    make_compiler_model,
    sha256_file,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")


def validate_calibration_set(
    task_root: str | Path,
    manifest: Mapping[str, object],
    spec: TaskSpec,
) -> list[Path]:
    import numpy as np

    from tools.mobilint_bert_compile.prepare import select_calibration_indices

    root = Path(task_root)
    calibration_dir = root / "calibration_data"
    expected_count = manifest.get("calibration_files")
    if expected_count != spec.calibration_samples:
        raise ValueError(
            "calibration sample contract requires "
            f"{spec.calibration_samples}, manifest declares {expected_count!r}"
        )

    dataset_size = manifest.get("dataset_size")
    if not isinstance(dataset_size, int) or isinstance(dataset_size, bool):
        raise ValueError("calibration manifest dataset_size must be an integer")
    expected_indices = list(
        select_calibration_indices(dataset_size, spec.calibration_samples)
    )
    if manifest.get("calibration_indices") != expected_indices:
        raise ValueError("calibration indices do not match the deterministic contract")

    sequence_lengths = manifest.get("sequence_lengths")
    if not isinstance(sequence_lengths, list) or len(sequence_lengths) != expected_count:
        raise ValueError("calibration manifest sequence_lengths count mismatch")
    if any(
        not isinstance(length, int)
        or isinstance(length, bool)
        or length <= 0
        or length > spec.max_length
        for length in sequence_lengths
    ):
        raise ValueError("calibration manifest contains an invalid sequence length")

    paths = sorted(calibration_dir.glob("*.npy"))
    expected_names = [f"{index:03d}.npy" for index in range(expected_count)]
    actual_names = [path.name for path in paths]
    if actual_names != expected_names or any(path.stat().st_size == 0 for path in paths):
        raise ValueError(
            "calibration file set mismatch: "
            f"expected {expected_names}, got {actual_names}"
        )

    artifacts = manifest.get("calibration_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != expected_count:
        raise ValueError("calibration artifact record count mismatch")
    for path, record in zip(paths, artifacts, strict=True):
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("calibration artifact record schema is invalid")
        expected_path = path.relative_to(root).as_posix()
        digest = record.get("sha256")
        if (
            record.get("path") != expected_path
            or type(record.get("size_bytes")) is not int
            or record["size_bytes"] != path.stat().st_size
        ):
            raise ValueError(f"calibration artifact record mismatch: {expected_path}")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"calibration artifact SHA256 is invalid: {expected_path}")
        if digest != sha256_file(path):
            raise ValueError(f"calibration artifact SHA256 mismatch: {expected_path}")

    expected_width = spec.mxq_inputs[0].shape[-1]
    for path, expected_length in zip(paths, sequence_lengths, strict=True):
        try:
            value = np.load(path, allow_pickle=False, mmap_mode="r")
        except Exception as error:
            raise ValueError(f"invalid calibration array file: {path}") from error
        if value.dtype != np.dtype(np.float32):
            raise ValueError(
                f"calibration array dtype mismatch in {path}: {value.dtype}"
            )
        expected_shape = (1, expected_length, expected_width)
        if value.shape != expected_shape:
            raise ValueError(
                f"calibration array shape mismatch in {path}: "
                f"expected {expected_shape}, got {value.shape}"
            )
        if not bool(np.isfinite(value).all()):
            raise ValueError(f"calibration array contains non-finite values: {path}")
    return paths


def build_feed_dict(
    inputs: Mapping[str, object],
    *,
    wrap_tensor=None,
    set_attention_mask=None,
) -> dict[str, object]:
    import torch

    if wrap_tensor is None or set_attention_mask is None:
        from qbcompiler.model_dict.parser.backend.torch.object_wrapper import (
            set_attention_mask as vendor_set_attention_mask,
        )
        from qbcompiler.model_dict.parser.backend.torch.util import (
            wrap_tensor as vendor_wrap_tensor,
        )

        wrap_tensor = wrap_tensor or vendor_wrap_tensor
        set_attention_mask = set_attention_mask or vendor_set_attention_mask

    required = ("input_ids", "attention_mask")
    missing = [name for name in required if name not in inputs]
    if missing:
        raise ValueError("compiler sample is missing input: " + ", ".join(missing))

    ordered_inputs = dict(inputs)
    if "token_type_ids" not in ordered_inputs:
        ordered_inputs["token_type_ids"] = torch.zeros_like(
            ordered_inputs["input_ids"]
        )

    feed_dict = {}
    for name in ("input_ids", "attention_mask", "token_type_ids"):
        value = ordered_inputs[name]
        if getattr(value, "ndim", None) != 2:
            raise ValueError(f"compiler sample {name} must be rank 2")
        wrapped = wrap_tensor(name, value)
        wrapped.src_shape[1].set_dynamic()
        feed_dict[name] = wrapped
    set_attention_mask(feed_dict["attention_mask"], "padding_mask")
    return feed_dict


def _prepare_artifact_path(output: str | Path, suffix: str) -> Path:
    path = Path(output)
    if path.suffix != suffix:
        raise ValueError(f"Mobilint compiler output must use the {suffix} suffix")
    if path.exists():
        raise FileExistsError(f"Mobilint compiler artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _require_nonempty_artifact(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Mobilint compiler produced an empty artifact: {path}")
    return path


def run_mblt_compile(
    *,
    model: object,
    feed_dict: Mapping[str, object],
    output: str | Path,
    compiler=None,
) -> Path:
    path = _prepare_artifact_path(output, ".mblt")
    if compiler is None:
        from qbcompiler import mblt_compile as compiler

    compiler(
        model=model,
        mblt_save_path=str(path),
        target_device="aries-rb",
        backend="torch",
        feed_dict=dict(feed_dict),
        cpu_offload=True,
    )
    return _require_nonempty_artifact(path)


def run_mxq_compile(
    *,
    model: object,
    feed_dict: Mapping[str, object],
    calibration_dir: str | Path,
    output: str | Path,
    compiler_api=None,
) -> Path:
    path = _prepare_artifact_path(output, ".mxq")
    calibration_path = Path(calibration_dir)
    if not calibration_path.is_dir():
        raise FileNotFoundError(
            f"Mobilint calibration directory not found: {calibration_path}"
        )
    if compiler_api is None:
        import qbcompiler as compiler_api

    calibration_config = compiler_api.CalibrationConfig(
        method=1,
        output=0,
        mode=1,
        max_percentile=compiler_api.CalibrationConfig.MaxPercentile(
            percentile=0.999,
            topk_ratio=0.01,
        ),
    )
    compiler_api.mxq_compile(
        model=model,
        target_device="aries-rb",
        save_path=str(path),
        calib_data_path=str(calibration_path),
        backend="torch",
        feed_dict=dict(feed_dict),
        inference_scheme="all",
        calibration_config=calibration_config,
    )
    return _require_nonempty_artifact(path)


def _sample_inputs(spec: TaskSpec, tokenizer):
    if spec.name == "sst2":
        return tokenizer("This movie was surprisingly good.", return_tensors="pt")
    if spec.name == "squad1":
        return tokenizer(
            "Where was the company founded?",
            "The company was founded in Seoul in 2019.",
            return_tensors="pt",
        )
    raise ValueError(f"unsupported Mobilint BERT compile task: {spec.name}")


def validate_source_output_shapes(
    spec: TaskSpec,
    inputs: Mapping[str, object],
    outputs: Mapping[str, object],
) -> dict[str, list[int]]:
    import torch

    input_ids = inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise RuntimeError("compiler smoke input_ids must be a rank-2 tensor")
    batch_size, sequence_length = input_ids.shape
    if batch_size != 1 or sequence_length <= 0:
        raise RuntimeError("compiler smoke input must use batch 1 and positive length")

    if spec.name == "sst2":
        expected_shapes = {"logits": (1, 2)}
        task_label = "SST-2"
    elif spec.name == "squad1":
        expected_shapes = {
            "start_logits": (1, sequence_length),
            "end_logits": (1, sequence_length),
        }
        task_label = "SQuAD"
    else:
        raise ValueError(f"unsupported Mobilint BERT compile task: {spec.name}")

    shapes = {}
    for name, expected_shape in expected_shapes.items():
        value = outputs.get(name)
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"{task_label} source output {name} must be a tensor")
        actual_shape = tuple(value.shape)
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"{task_label} source output {name} shape mismatch: "
                f"expected {expected_shape}, got {actual_shape}"
            )
        if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()):
            raise RuntimeError(
                f"{task_label} source output {name} must contain finite floating values"
            )
        shapes[name] = list(actual_shape)
    return shapes


def _load_model_and_feed(spec: TaskSpec):
    import torch

    from tools.mobilint_bert_compile.prepare import (
        load_source_model,
        load_tokenizer,
    )

    tokenizer = load_tokenizer(spec)
    source_model = load_source_model(spec).eval()
    inputs = dict(_sample_inputs(spec, tokenizer))
    if "token_type_ids" not in inputs:
        inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])

    with torch.no_grad():
        source_outputs = source_model(**inputs)
    compiler_model = make_compiler_model(spec.name, source_model).eval()
    with torch.no_grad():
        compiler_outputs = compiler_model(**inputs)

    actual_names = tuple(compiler_outputs.keys())
    if actual_names != spec.source_outputs:
        raise RuntimeError(
            "compiler source output mismatch: "
            f"expected {spec.source_outputs}, got {actual_names}"
        )
    if spec.name == "squad1":
        for name in spec.source_outputs:
            if not torch.equal(compiler_outputs[name], getattr(source_outputs, name)):
                raise RuntimeError(f"SQuAD compiler wrapper changed {name}")

    output_shapes = validate_source_output_shapes(spec, inputs, compiler_outputs)
    return compiler_model, build_feed_dict(inputs), output_shapes


def _read_manifest(task_root: Path, spec: TaskSpec) -> dict[str, object]:
    path = task_root / "calibration_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"calibration manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for name, expected in (
        ("task", spec.name),
        ("model_id", spec.model_id),
        ("target_device", spec.target_device),
    ):
        if manifest.get(name) != expected:
            raise ValueError(
                f"calibration manifest {name} mismatch: "
                f"expected {expected!r}, got {manifest.get(name)!r}"
            )
    return manifest


def _artifact_record(path: Path, task_root: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(task_root)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _compiler_version() -> str | None:
    try:
        return importlib.metadata.version("qbcompiler")
    except importlib.metadata.PackageNotFoundError:
        return None


def create_compile_report(
    spec: TaskSpec,
    source_output_shapes: Mapping[str, object],
    *,
    compiler_version: str | None,
) -> dict[str, object]:
    return {
        **contract_to_dict(spec),
        "compiler_version": compiler_version,
        "source_output_shapes": dict(source_output_shapes),
        "compiler_options": {
            "mblt": {
                "target_device": "aries-rb",
                "backend": "torch",
                "cpu_offload": True,
            },
            "mxq": {
                "target_device": "aries-rb",
                "backend": "torch",
                "inference_scheme": "all",
                "calibration": {
                    "method": 1,
                    "output": 0,
                    "mode": 1,
                    "max_percentile": 0.999,
                    "topk_ratio": 0.01,
                },
            },
        },
        "artifacts": {},
    }


def compile_task(
    task: str,
    stage: str,
    artifact_root: str | Path,
) -> dict[str, object]:
    if stage not in {"mblt", "mxq", "all"}:
        raise ValueError(f"unsupported Mobilint compiler stage: {stage}")

    spec = get_task_spec(task)
    task_root = Path(artifact_root).expanduser().resolve() / spec.name
    manifest = _read_manifest(task_root, spec)
    validate_calibration_set(task_root, manifest, spec)

    report_path = task_root / "compile-report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = None

    stages = ("mblt", "mxq") if stage == "all" else (stage,)
    for current_stage in stages:
        model, feed_dict, output_shapes = _load_model_and_feed(spec)
        if report is None:
            report = create_compile_report(
                spec,
                output_shapes,
                compiler_version=_compiler_version(),
            )
        if current_stage == "mblt":
            artifact = run_mblt_compile(
                model=model,
                feed_dict=feed_dict,
                output=task_root / "mblt" / f"{spec.name}.mblt",
            )
        else:
            artifact = run_mxq_compile(
                model=model,
                feed_dict=feed_dict,
                calibration_dir=task_root / "calibration_data",
                output=task_root / "mxq" / f"{spec.name}.mxq",
            )
        report["artifacts"][current_stage] = _artifact_record(artifact, task_root)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    assert report is not None
    return report


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile prepared BERT tasks with Mobilint qbcompiler 1.2",
    )
    parser.add_argument("--task", required=True, choices=("sst2", "squad1"))
    parser.add_argument("--stage", choices=("mblt", "mxq", "all"), default="all")
    parser.add_argument("--artifact-root", type=Path, default=Path("./artifacts"))
    parser.add_argument("--describe", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _create_parser().parse_args(argv)
    if args.describe:
        print(json.dumps(contract_to_dict(get_task_spec(args.task)), indent=2))
        return 0

    report = compile_task(args.task, args.stage, args.artifact_root)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
