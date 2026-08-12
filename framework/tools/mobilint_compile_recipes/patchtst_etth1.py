"""Deterministic Mobilint compiler experiments for PatchTST ETTh1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from tools.mobilint_compile_recipes.compiler import run_mblt_compile, run_mxq_compile
from tools.mobilint_compile_recipes.contracts import (
    contract_to_dict,
    get_recipe,
    select_even_indices,
    sha256_file,
)


MODEL = "patchtst-etth1"
SOURCE_ID = "ibm-granite/granite-timeseries-patchtst"
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 96
NUM_INPUT_CHANNELS = 7
PATCH_LENGTH = 12
PATCH_STRIDE = 12
SEQUENCE_START = 8
NUM_PATCHES = 42
CALIBRATION_SAMPLES = 32
INPUT_ORDER = ("past_values", "past_observed_mask")
VARIANTS = ("stock", "compat-static-patchifier")
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
COMPAT_RECIPE_REVISION = 2
COMPAT_REWRITES = (
    "Tensor.unfold -> fixed slice/stack patchifier",
    "bool observation mask -> past_values dtype inside wrapper",
    "Tensor.clamp_min(1.0) -> Tensor.clamp(min=1.0)",
)


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


def _require_variant(variant: str):
    recipe = get_recipe(MODEL, variant)
    if variant not in VARIANTS:
        raise ValueError(f"unsupported PatchTST variant: {variant!r}")
    return recipe


def _compatibility_provenance() -> dict[str, object]:
    return {
        "recipe_revision": COMPAT_RECIPE_REVISION,
        "rewrites": list(COMPAT_REWRITES),
        "recipe_source_sha256": sha256_file(Path(__file__).resolve()),
    }


def static_patchify(past_values):
    """Reproduce the checkpoint's 42 fixed patches without ``Tensor.unfold``."""
    import torch

    if tuple(past_values.shape[-2:]) != (CONTEXT_LENGTH, NUM_INPUT_CHANNELS):
        raise ValueError(
            "PatchTST past_values must end with shape "
            f"({CONTEXT_LENGTH}, {NUM_INPUT_CHANNELS})"
        )
    trimmed = past_values[:, SEQUENCE_START:, :]
    patches = torch.stack(
        [
            trimmed[:, offset : offset + PATCH_LENGTH, :]
            for offset in range(0, NUM_PATCHES * PATCH_STRIDE, PATCH_STRIDE)
        ],
        dim=1,
    )
    return patches.permute(0, 3, 1, 2).contiguous()


def _require_checkpoint_contract(model) -> None:
    expected = {
        "context_length": CONTEXT_LENGTH,
        "prediction_length": PREDICTION_LENGTH,
        "num_input_channels": NUM_INPUT_CHANNELS,
        "patch_length": PATCH_LENGTH,
        "patch_stride": PATCH_STRIDE,
    }
    config = getattr(model, "config", None)
    for field, value in expected.items():
        observed = getattr(config, field, None)
        if observed != value:
            raise ValueError(
                f"PatchTST {field} mismatch: expected {value}, got {observed!r}"
            )


def build_patchtst_wrapper(model, variant: str):
    """Return a tensor-output wrapper with compat-only compiler lowerings."""
    import torch

    _require_variant(variant)
    _require_checkpoint_contract(model)
    cast_mask = variant == "compat-static-patchifier"

    if cast_mask:
        backbone = getattr(model, "model", None)
        if backbone is None or not hasattr(backbone, "patchifier"):
            raise ValueError("PatchTST checkpoint has no model.patchifier to replace")
        scaler_container = getattr(backbone, "scaler", None)
        stock_scaler = getattr(scaler_container, "scaler", None)
        required_scaler_fields = ("dim", "keepdim", "minimum_scale")
        if stock_scaler is None or any(
            not hasattr(stock_scaler, field) for field in required_scaler_fields
        ):
            raise ValueError(
                "PatchTST checkpoint has no compatible model.scaler.scaler to replace"
            )

        class StaticPatchifier(torch.nn.Module):
            def forward(self, past_values):
                return static_patchify(past_values)

        class CompilerCompatibleStdScaler(torch.nn.Module):
            def __init__(self, source_scaler):
                super().__init__()
                self.dim = source_scaler.dim
                self.keepdim = source_scaler.keepdim
                self.minimum_scale = source_scaler.minimum_scale

            def forward(self, data, observed_indicator):
                denominator = observed_indicator.sum(
                    self.dim, keepdim=self.keepdim
                ).clamp(min=1.0)
                loc = (data * observed_indicator).sum(
                    self.dim, keepdim=self.keepdim
                ) / denominator
                variance = (((data - loc) * observed_indicator) ** 2).sum(
                    self.dim, keepdim=self.keepdim
                ) / denominator
                scale = torch.sqrt(variance + self.minimum_scale)
                return (data - loc) / scale, loc, scale

        backbone.patchifier = StaticPatchifier()
        scaler_container.scaler = CompilerCompatibleStdScaler(stock_scaler)

    class PatchTSTWrapper(torch.nn.Module):
        def __init__(self, source_model, convert_mask):
            super().__init__()
            self.source_model = source_model
            self.convert_mask = convert_mask

        def forward(self, past_values, past_observed_mask):
            mask = past_observed_mask
            if self.convert_mask:
                mask = mask.to(dtype=past_values.dtype)
            return self.source_model(
                past_values=past_values,
                past_observed_mask=mask,
                return_dict=True,
            ).prediction_outputs

    wrapper = PatchTSTWrapper(model, cast_mask).eval()
    wrapper.requires_grad_(False)
    return wrapper


def resolve_model_revision(
    source_id: str,
    requested_revision: str,
    *,
    api=None,
) -> str:
    """Resolve a requested Hugging Face revision to one exact commit SHA."""
    if not isinstance(requested_revision, str) or not requested_revision:
        raise ValueError("requested revision must be a non-empty string")
    normalized = requested_revision.lower()
    if _COMMIT_SHA.fullmatch(normalized):
        return normalized
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    resolved = str(api.model_info(source_id, revision=requested_revision).sha).lower()
    if not _COMMIT_SHA.fullmatch(resolved):
        raise ValueError(
            "Hugging Face revision did not resolve to an exact commit SHA: "
            f"{resolved!r}"
        )
    return resolved


def _validate_calibration_sample(sample: Mapping[str, object], index: int):
    import numpy as np

    if not isinstance(sample, Mapping) or tuple(sample) != INPUT_ORDER:
        raise ValueError(
            f"calibration sample {index} inputs must be ordered as {list(INPUT_ORDER)!r}"
        )
    values = np.asarray(sample["past_values"])
    mask = np.asarray(sample["past_observed_mask"])
    expected_shape = (1, CONTEXT_LENGTH, NUM_INPUT_CHANNELS)
    if values.shape != expected_shape:
        raise ValueError(
            f"calibration sample {index} past_values shape must be {expected_shape}"
        )
    if values.dtype != np.float32:
        raise ValueError(
            f"calibration sample {index} past_values dtype must be float32"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"calibration sample {index} past_values must be finite")
    if mask.shape != expected_shape:
        raise ValueError(
            f"calibration sample {index} past_observed_mask shape must be {expected_shape}"
        )
    if mask.dtype != np.bool_:
        raise ValueError(
            f"calibration sample {index} past_observed_mask dtype must be bool"
        )
    return np.ascontiguousarray(values), np.ascontiguousarray(mask)


def write_multi_input_calibration(
    samples: Sequence[Mapping[str, object]], output_dir: str | Path
) -> Path:
    """Write immutable sample directories and an explicitly ordered JSON index."""
    if not samples:
        raise ValueError("calibration samples must not be empty")
    validated = [
        _validate_calibration_sample(sample, index)
        for index, sample in enumerate(samples)
    ]
    root = Path(output_dir).expanduser().resolve()
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"calibration output already exists: {root}") from error

    import numpy as np

    paths: list[list[str]] = []
    for index, (values, mask) in enumerate(validated):
        sample_dir = root / f"{index:03d}"
        sample_dir.mkdir()
        values_path = sample_dir / "past_values.npy"
        mask_path = sample_dir / "past_observed_mask.npy"
        np.save(values_path, values, allow_pickle=False)
        np.save(mask_path, mask, allow_pickle=False)
        paths.append([str(values_path), str(mask_path)])

    index_path = root / "calibration.json"
    _write_json_atomic(
        index_path,
        {
            "info": {"input names": list(INPUT_ORDER)},
            "calib paths": paths,
        },
    )
    return index_path


def _profile_model_spec():
    from core.model_profiles import SUPPORTED_PROFILES
    from core.model_spec import Model_Spec

    profile = SUPPORTED_PROFILES[MODEL]
    return Model_Spec(
        name=MODEL,
        task=profile["task"],
        input_shapes=dict(profile["input_shapes"]),
        input_dtype=dict(profile["input_dtype"]),
        output_shapes={"prediction_outputs": (1, PREDICTION_LENGTH, NUM_INPUT_CHANNELS)},
    )


def _compile_report(recipe, requested_revision: str, resolved_revision: str):
    report = {
        **contract_to_dict(recipe),
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
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
        "artifacts": {},
    }
    if recipe.variant == "compat-static-patchifier":
        report["compatibility"] = _compatibility_provenance()
    return report


def prepare_calibration(
    dataset_path: str | Path,
    attempt_root: str | Path,
    *,
    variant: str,
    requested_revision: str,
    revision_api=None,
    loader_factory=None,
) -> dict[str, object]:
    """Prepare 32 deterministic ETTh1 validation windows and their provenance."""
    recipe = _require_variant(variant)
    root = Path(attempt_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"attempt root does not exist: {root}")
    calibration_root = root / "calibration"
    manifest_path = root / "source-manifest.json"
    report_path = root / "compile-report.json"
    if calibration_root.exists() or manifest_path.exists() or report_path.exists():
        raise FileExistsError(f"PatchTST calibration output already exists: {root}")
    if variant == "compat-static-patchifier" and (
        not isinstance(requested_revision, str)
        or not _COMMIT_SHA.fullmatch(requested_revision)
    ):
        raise ValueError(
            "compat-static-patchifier preparation requires an exact lowercase "
            "commit SHA from the stock source manifest"
        )

    dataset = Path(dataset_path).expanduser().resolve()
    if not dataset.is_file() or dataset.stat().st_size == 0:
        raise FileNotFoundError(f"ETTh1 dataset not found or empty: {dataset}")
    resolved_revision = resolve_model_revision(
        recipe.source_id,
        requested_revision,
        api=revision_api,
    )

    source_provenance = {
        "model": recipe.model,
        "variant": recipe.variant,
        "source_id": recipe.source_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "dataset_id": "ETTh1",
        "etth1_sha256": sha256_file(dataset),
    }
    if variant == "compat-static-patchifier":
        source_provenance["compatibility"] = _compatibility_provenance()
    _write_json_atomic(manifest_path, source_provenance)
    _write_json_atomic(
        report_path,
        _compile_report(recipe, requested_revision, resolved_revision),
    )

    if loader_factory is None:
        from dataloader import ETTmLoader

        loader_factory = ETTmLoader
    loader = loader_factory(
        _profile_model_spec(),
        csv_path=str(dataset),
        split="val",
        split_boundaries=(8640, 11520),
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        stride=PATCH_STRIDE,
        normalize=True,
    )
    loader_metadata = loader.get_metadata()
    indices = select_even_indices(
        int(loader_metadata["window_count"]), recipe.calibration_samples
    )
    samples: list[dict[str, object]] = []
    normalization_samples: list[dict[str, object]] = []
    sample_records: list[dict[str, object]] = []
    for ordinal, window_index in enumerate(indices):
        loaded = loader.load_by_index(window_index)
        values = loaded["input"]["past_values"]
        mask = loaded["input"]["past_observed_mask"]
        samples.append(
            {
                "past_values": values[None, ...],
                "past_observed_mask": mask[None, ...],
            }
        )
        stats = loaded["label"]["norm_stats"]
        normalization_samples.append(
            {
                "window_index": window_index,
                "mean": stats["mean"].astype(float).tolist(),
                "std": stats["std"].astype(float).tolist(),
            }
        )
        sample_records.append(
            {
                "ordinal": ordinal,
                "window_index": window_index,
                "paths": {
                    "past_values": f"calibration/{ordinal:03d}/past_values.npy",
                    "past_observed_mask": (
                        f"calibration/{ordinal:03d}/past_observed_mask.npy"
                    ),
                },
            }
        )

    write_multi_input_calibration(samples, calibration_root)
    for record in sample_records:
        paths = record["paths"]
        assert isinstance(paths, Mapping)
        record["size_bytes"] = {
            name: (root / str(paths[name])).stat().st_size for name in INPUT_ORDER
        }
        record["sha256"] = {
            name: sha256_file(root / str(paths[name])) for name in INPUT_ORDER
        }
    manifest: dict[str, object] = {
        **source_provenance,
        "calibration_indices": list(indices),
        "input_order": list(INPUT_ORDER),
        "inputs": [
            {"name": item.name, "shape": list(item.shape), "dtype": item.dtype}
            for item in recipe.compiler_inputs
        ],
        "output": {
            "name": recipe.outputs[0].name,
            "shape": list(recipe.outputs[0].shape),
            "dtype": recipe.outputs[0].dtype,
        },
        "loader": {
            "split": "val",
            "split_boundaries": [8640, 11520],
            "context_length": CONTEXT_LENGTH,
            "prediction_length": PREDICTION_LENGTH,
            "stride": PATCH_STRIDE,
            "window_count": int(loader_metadata["window_count"]),
        },
        "normalization": {
            "enabled": True,
            "method": "per-window RevIN",
            "epsilon": 1e-8,
            "samples": normalization_samples,
        },
        "samples": sample_records,
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def load_source_model(source_id: str, resolved_revision: str):
    """Load only the exact resolved checkpoint revision."""
    from transformers import PatchTSTForPrediction

    model = PatchTSTForPrediction.from_pretrained(
        source_id,
        revision=resolved_revision,
    ).eval()
    model.requires_grad_(False)
    return model


def _read_manifest(root: Path, variant: str) -> dict[str, object]:
    _require_variant(variant)
    path = root / "source-manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"PatchTST source manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "model": MODEL,
        "variant": variant,
        "source_id": SOURCE_ID,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"source manifest {field} mismatch: expected {value!r}, "
                f"got {manifest.get(field)!r}"
            )
    resolved = manifest.get("resolved_revision")
    if not isinstance(resolved, str) or not _COMMIT_SHA.fullmatch(resolved):
        raise ValueError("source manifest resolved_revision is not an exact commit SHA")
    return manifest


def _load_feed_input(root: Path, manifest: Mapping[str, object]):
    import numpy as np
    import torch

    sample_records = manifest.get("samples")
    if not isinstance(sample_records, list) or not sample_records:
        raise ValueError("source manifest has no calibration samples")
    paths = sample_records[0].get("paths")
    if not isinstance(paths, dict):
        raise ValueError("source manifest sample paths are invalid")
    arrays: dict[str, object] = {}
    for name in INPUT_ORDER:
        path = (root / str(paths[name])).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"source manifest input path escapes attempt root: {path}") from error
        arrays[name] = np.load(path, allow_pickle=False)
    values, mask = _validate_calibration_sample(arrays, 0)
    return {
        "past_values": torch.from_numpy(values.copy()),
        "past_observed_mask": torch.from_numpy(mask.copy()),
    }


def _wrapper_and_output(root: Path, variant: str, manifest, model_loader):
    import torch

    feed_dict = _load_feed_input(root, manifest)
    model = model_loader(manifest["source_id"], manifest["resolved_revision"])
    if variant == "compat-static-patchifier":
        stock_wrapper = build_patchtst_wrapper(model, "stock")
        with torch.no_grad():
            stock_output = stock_wrapper(**feed_dict)
        wrapper = build_patchtst_wrapper(model, variant)
        with torch.no_grad():
            output = wrapper(**feed_dict)
        torch.testing.assert_close(
            output,
            stock_output,
            rtol=1e-5,
            atol=1e-6,
        )
    else:
        wrapper = build_patchtst_wrapper(model, variant)
        with torch.no_grad():
            output = wrapper(**feed_dict)
    if tuple(output.shape) != (1, PREDICTION_LENGTH, NUM_INPUT_CHANNELS):
        raise ValueError(
            "PatchTST output shape mismatch: expected (1, 96, 7), "
            f"got {tuple(output.shape)}"
        )
    if output.dtype != torch.float32:
        raise ValueError(f"PatchTST output dtype must be float32, got {output.dtype}")
    if not torch.isfinite(output).all():
        raise ValueError("PatchTST output must contain only finite values")
    return wrapper, feed_dict, output.detach().cpu()


def _read_report(root: Path) -> dict[str, object]:
    path = root / "compile-report.json"
    if not path.is_file():
        raise FileNotFoundError(f"PatchTST compile report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _source_smoke_record(output, variant: str) -> dict[str, object]:
    checked = variant == "compat-static-patchifier"
    record: dict[str, object] = {
        "output_shape": list(output.shape),
        "output_dtype": str(output.numpy().dtype),
        "finite": True,
        "equivalence_checked": checked,
    }
    if checked:
        record.update(
            {
                "stock_compat_rtol": 1e-5,
                "stock_compat_atol": 1e-6,
            }
        )
    return record


def source_smoke(
    attempt_root: str | Path,
    variant: str,
    *,
    model_loader=load_source_model,
):
    """Prove the pinned CPU source behavior before any compiler import."""
    root = Path(attempt_root).expanduser().resolve()
    manifest = _read_manifest(root, variant)
    _, _, output = _wrapper_and_output(root, variant, manifest, model_loader)
    report = _read_report(root)
    report["source_smoke"] = _source_smoke_record(output, variant)
    _write_json_atomic(root / "compile-report.json", report)
    return output


def _artifact_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def compile_stage(
    stage: str,
    attempt_root: str | Path,
    variant: str,
    *,
    model_loader=load_source_model,
    mblt_compiler=None,
    mxq_compiler_api=None,
) -> Path:
    """Reload the pinned CPU source and execute one isolated compiler stage."""
    if stage not in {"mblt", "mxq"}:
        raise ValueError(f"unsupported PatchTST compiler stage: {stage!r}")
    root = Path(attempt_root).expanduser().resolve()
    manifest = _read_manifest(root, variant)
    output_path = root / stage / f"patchtst-etth1-{stage}.{stage}"
    report = _read_report(root)
    active_stage = report.get("active_compiler_stage")
    if active_stage is not None:
        raise RuntimeError(
            "PatchTST compiler attempt already entered stage "
            f"{active_stage!r}; use a fresh attempt root"
        )
    artifacts = report.get("artifacts")
    if isinstance(artifacts, dict) and stage in artifacts:
        raise FileExistsError(f"PatchTST compiler stage already exists: {stage}")
    if output_path.exists():
        raise FileExistsError(f"Mobilint compiler artifact already exists: {output_path}")
    wrapper, feed_dict, output = _wrapper_and_output(
        root, variant, manifest, model_loader
    )
    report["source_smoke"] = _source_smoke_record(output, variant)
    report["active_compiler_stage"] = stage
    _write_json_atomic(root / "compile-report.json", report)

    recipe = _require_variant(variant)
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
        description="Prepare and compile pinned PatchTST ETTh1 experiments",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("describe", "prepare", "source-smoke", "mblt", "mxq"),
    )
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--attempt-root", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--model-revision", default="main")
    return parser


def _require_cli_path(parser, value, option: str):
    if value is None:
        parser.error(f"{option} is required for this stage")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = _create_parser()
    args = parser.parse_args(argv)
    if args.stage == "describe":
        print(json.dumps(contract_to_dict(_require_variant(args.variant)), indent=2))
        return 0

    attempt_root = _require_cli_path(parser, args.attempt_root, "--attempt-root")
    if args.stage == "prepare":
        dataset = _require_cli_path(parser, args.dataset, "--dataset")
        result = prepare_calibration(
            dataset,
            attempt_root,
            variant=args.variant,
            requested_revision=args.model_revision,
        )
        print(json.dumps(result, sort_keys=True))
    elif args.stage == "source-smoke":
        output = source_smoke(attempt_root, args.variant)
        print(json.dumps({"shape": list(output.shape), "dtype": str(output.numpy().dtype)}))
    else:
        artifact = compile_stage(args.stage, attempt_root, args.variant)
        print(json.dumps({"artifact": str(artifact)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
