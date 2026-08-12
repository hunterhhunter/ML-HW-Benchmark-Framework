"""Lazy qbcompiler call boundary shared by Mobilint compile recipes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from tools.mobilint_compile_recipes.contracts import CompileRecipe


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


def _uint8_input_names(recipe: CompileRecipe) -> list[str]:
    compiler_dtypes = {tensor.name: tensor.dtype for tensor in recipe.compiler_inputs}
    return [
        tensor.name
        for tensor in recipe.runtime_inputs
        if compiler_dtypes.get(tensor.name) != tensor.dtype
    ]


def run_mblt_compile(
    *,
    recipe: CompileRecipe,
    model: object,
    feed_dict: Mapping[str, object],
    output: str | Path,
    compiler=None,
) -> Path:
    """Compile a new MBLT artifact, importing qbcompiler only when required."""
    path = _prepare_artifact_path(output, ".mblt")
    if compiler is None:
        from qbcompiler import mblt_compile as compiler

    compiler(
        model=model,
        mblt_save_path=str(path),
        target_device=recipe.target_device,
        backend="torch",
        feed_dict=dict(feed_dict),
        cpu_offload=True,
    )
    return _require_nonempty_artifact(path)


def run_mxq_compile(
    *,
    recipe: CompileRecipe,
    model: object,
    feed_dict: Mapping[str, object],
    calibration_path: str | Path,
    output: str | Path,
    compiler_api=None,
) -> Path:
    """Compile a new MXQ artifact with the recipe's explicit runtime ABI."""
    path = _prepare_artifact_path(output, ".mxq")
    calibration = Path(calibration_path)
    if not calibration.exists():
        raise FileNotFoundError(
            f"Mobilint calibration path not found: {calibration}"
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
    kwargs = {
        "model": model,
        "target_device": recipe.target_device,
        "save_path": str(path),
        "calib_data_path": str(calibration),
        "backend": "torch",
        "feed_dict": dict(feed_dict),
        "inference_scheme": recipe.inference_scheme,
        "calibration_config": calibration_config,
    }
    if recipe.config_preset is not None:
        kwargs["config_preset"] = recipe.config_preset
    if recipe.yolo_decode_include is not None:
        kwargs["yolo_decode_include"] = recipe.yolo_decode_include
    uint8_inputs = _uint8_input_names(recipe)
    if uint8_inputs:
        kwargs["uint8_input_config"] = compiler_api.Uint8InputConfig(
            apply=True,
            inputs=uint8_inputs,
            division_factor=255.0,
        )

    compiler_api.mxq_compile(**kwargs)
    return _require_nonempty_artifact(path)
