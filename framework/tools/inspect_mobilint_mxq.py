#!/usr/bin/env python3
"""Inspect a Mobilint MXQ artifact without launching it on an NPU."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any


_CORE_MODE_SETTERS = {
    "auto": "set_auto_core_mode",
    "single": "set_single_core_mode",
    "multi": "set_multi_core_mode",
    "global4": "set_global4_core_mode",
    "global8": "set_global8_core_mode",
}


def _load_qbruntime() -> Any:
    try:
        return importlib.import_module("qbruntime")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "qbruntime is unavailable. Install the Mobilint "
            "mobilint-qb-runtime Python package in this environment."
        ) from exc


def _configure_core_mode(qbruntime: Any, core_mode: str) -> Any:
    normalized = str(core_mode).strip().lower()
    setter_name = _CORE_MODE_SETTERS.get(normalized)
    if setter_name is None:
        choices = ", ".join(_CORE_MODE_SETTERS)
        raise ValueError(f"core_mode must be one of: {choices}")

    config = qbruntime.ModelConfig()
    setter = getattr(config, setter_name)
    if normalized == "single":
        core_id = qbruntime.CoreId(
            qbruntime.Cluster.Cluster0,
            qbruntime.Core.Core0,
        )
        result = setter(None, [core_id])
    else:
        result = setter()
    if result is False:
        raise RuntimeError(f"qbruntime rejected core_mode={normalized}")
    return config


def _dtype_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value)


def _dtype_names(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [_dtype_name(item) for item in value]
    return [_dtype_name(value)]


def _shape_lists(values: Any) -> list[list[int]]:
    return [[int(dimension) for dimension in shape] for shape in values]


def inspect_mxq(
    artifact: str | Path,
    *,
    core_mode: str = "global8",
    qbruntime_module: Any | None = None,
) -> dict[str, Any]:
    """Return the static input/output contract embedded in an MXQ artifact."""

    artifact_path = Path(artifact).expanduser().resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"MXQ artifact does not exist: {artifact_path}")
    if artifact_path.suffix.lower() != ".mxq":
        raise ValueError(f"Expected a .mxq artifact: {artifact_path}")

    qbruntime = qbruntime_module or _load_qbruntime()
    normalized_core_mode = str(core_mode).strip().lower()
    config = _configure_core_mode(qbruntime, normalized_core_mode)
    model = qbruntime.Model(str(artifact_path), config)
    try:
        return {
            "sdk_version": str(getattr(qbruntime, "__version__", "unknown")),
            "artifact": str(artifact_path),
            "core_mode": normalized_core_mode,
            "variants": int(model.get_num_model_variants()),
            "input_dtypes": _dtype_names(model.get_model_input_data_type()),
            "input_shapes": _shape_lists(model.get_model_input_shape()),
            "output_shapes": _shape_lists(model.get_model_output_shape()),
        }
    finally:
        model.dispose()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the static tensor contract embedded in a Mobilint MXQ."
    )
    parser.add_argument("artifact", type=Path, help="Path to the .mxq artifact")
    parser.add_argument(
        "--core-mode",
        choices=tuple(_CORE_MODE_SETTERS),
        default="global8",
        help="ModelConfig core mode used while reading metadata (default: global8)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(json.dumps(inspect_mxq(args.artifact, core_mode=args.core_mode), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
