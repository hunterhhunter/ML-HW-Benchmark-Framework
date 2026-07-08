"""DeepX-specific image classification dataloader.

This module keeps DXNN metadata parsing and dx_app-compatible image input
handling out of the generic image preprocessing strategies.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

from .image_classification_loader import ImageClassificationLoader
from .preprocess_strategies import (
    MLPerfResNet50Preprocess,
    MLPerfResNet50RawPreprocess,
    PreprocessStrategy,
)
from core.model_spec import Model_Spec


class DeepXDirectResizeRawPreprocess(PreprocessStrategy):
    """dx_app SimpleResizePreprocessor equivalent for RGB image tensors."""

    def cache_config(self) -> Dict[str, str]:
        return {"vendor": "deepx", "resize": "direct", "scale": "raw"}

    def __call__(
        self,
        img: Image.Image,
        target_hw: tuple,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> np.ndarray:
        img = img.convert("RGB")
        img = img.resize((target_hw[1], target_hw[0]), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        return np.transpose(arr, (2, 0, 1))


@dataclass
class DeepXImageInputConfig:
    preprocess_mode: str
    short_side: int | None = None
    input_info: dict | None = None
    input_layout: str | None = None
    input_dtype: str | None = None
    runtime_options: dict[str, Any] = field(default_factory=dict)

    @property
    def expects_uint8_image(self) -> bool:
        return self.input_dtype == "UINT8" and self.input_layout is not None


def _deepx_preprocessings_from_config(config: dict) -> list:
    default_loader = config.get("default_loader")
    if isinstance(default_loader, dict):
        preprocessings = default_loader.get("preprocessings")
        if isinstance(preprocessings, list):
            return preprocessings
    preprocessings = config.get("preprocessings")
    return preprocessings if isinstance(preprocessings, list) else []


def _read_deepx_json_config(config_path: str | None) -> dict | None:
    if not config_path:
        return None
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARN] DeepX config를 읽지 못해 전처리 자동 판별을 건너뜁니다: {exc}")
        return None


def read_dxnn_compile_config(artifact_path: str | Path | None) -> dict | None:
    """Extract embedded DX-COM compile_config from a DXNN artifact when available."""
    if not artifact_path:
        return None
    path = Path(artifact_path)
    if not path.exists() or path.suffix.lower() != ".dxnn":
        return None

    try:
        data = path.read_bytes()
        if data[:4] != b"DXNN":
            return None
        header = json.loads(data[8:8192].split(b"\0", 1)[0].decode("utf-8"))
        entry = header.get("data", {}).get("compile_config")
        if not entry:
            return None
        base_offset = int(header.get("size", 8192))
        offset = base_offset + int(entry["offset"])
        size = int(entry["size"])
        return json.loads(data[offset:offset + size].decode("utf-8"))
    except Exception as exc:
        print(f"[WARN] DXNN compile_config를 읽지 못했습니다: {exc}")
        return None


def read_dxnn_rmap_input_info(artifact_path: str | Path | None) -> dict | None:
    """Extract the first NPU RMAP input metadata from a DXNN artifact."""
    if not artifact_path:
        return None
    path = Path(artifact_path)
    if not path.exists() or path.suffix.lower() != ".dxnn":
        return None

    try:
        data = path.read_bytes()
        if data[:4] != b"DXNN":
            return None
        header = json.loads(data[8:8192].split(b"\0", 1)[0].decode("utf-8"))
        base_offset = int(header.get("size", 8192))
        compiled_data = header.get("data", {}).get("compiled_data", {})
        if not isinstance(compiled_data, dict):
            return None

        for npu_targets in compiled_data.values():
            if not isinstance(npu_targets, dict):
                continue
            for graph_data in npu_targets.values():
                if not isinstance(graph_data, dict):
                    continue
                entry = graph_data.get("rmap_info")
                if not isinstance(entry, dict):
                    continue
                offset = base_offset + int(entry["offset"])
                size = int(entry["size"])
                rmap_info = json.loads(data[offset:offset + size].decode("utf-8"))
                inputs = rmap_info.get("inputs")
                if isinstance(inputs, list) and inputs:
                    return inputs[0]
    except Exception as exc:
        print(f"[WARN] DXNN rmap_info를 읽지 못했습니다: {exc}")
    return None


def deepx_rmap_input_dtype(input_info: dict | None) -> str | None:
    if not input_info:
        return None
    dtype = input_info.get("dtype") or input_info.get("dtype_encoded")
    return str(dtype).upper() if dtype is not None else None


def deepx_rmap_image_input_layout(input_info: dict | None) -> str | None:
    if not input_info:
        return None
    shape = input_info.get("shape") or input_info.get("shape_encoded")
    if not isinstance(shape, list) or len(shape) != 4:
        return None
    try:
        dims = [int(dim) for dim in shape]
    except (TypeError, ValueError):
        return None
    if dims[-1] in (1, 3, 4):
        return "NHWC"
    if dims[1] in (1, 3, 4):
        return "NCHW"
    return None


def deepx_config_has_graph_normalization(config: dict | None) -> bool:
    if not config:
        return False
    preprocessings = _deepx_preprocessings_from_config(config)
    return any(
        isinstance(step, dict) and ("div" in step or "normalize" in step)
        for step in preprocessings
    )


def deepx_config_resize_short_side(config: dict | None) -> int | None:
    if not config:
        return None

    preprocessings = _deepx_preprocessings_from_config(config)
    for step in preprocessings:
        if not isinstance(step, dict):
            continue
        resize = step.get("resize")
        if not isinstance(resize, dict):
            continue
        size = resize.get("size")
        if isinstance(size, dict):
            size = size.get("shortest_edge") or size.get("short_side")
        if size is not None:
            return int(size)
    return None


def resolve_deepx_image_input_config(
    *,
    artifact_path: str | Path | None,
    compile_options: dict | None = None,
    requested_mode: str = "auto",
    compile_enabled: bool = True,
) -> DeepXImageInputConfig:
    compile_options = compile_options or {}
    input_info = read_dxnn_rmap_input_info(artifact_path)
    input_layout = deepx_rmap_image_input_layout(input_info)
    input_dtype = deepx_rmap_input_dtype(input_info)

    config = _read_deepx_json_config(compile_options.get("config_path"))
    embedded_config = read_dxnn_compile_config(artifact_path)
    short_side = (
        deepx_config_resize_short_side(config)
        or deepx_config_resize_short_side(embedded_config)
    )

    runtime_options: dict[str, Any] = {}
    if input_layout:
        runtime_options["input_layout"] = input_layout
    if input_dtype in ("FLOAT", "FLOAT32"):
        runtime_options["input_dtype"] = "float32"
    elif input_dtype == "UINT8":
        runtime_options.update({
            "input_dtype": "uint8",
            "input_batch_axis": "squeeze",
            "single_input_run_style": "list",
        })

    expects_uint8_image = input_dtype == "UINT8" and input_layout is not None
    if expects_uint8_image:
        if requested_mode == "normalized":
            print(
                "[WARN] DeepX DXNN input is UINT8 image data; "
                "--image-preprocess-mode normalized를 raw로 전환합니다."
            )
        preprocess_mode = "raw"
    elif requested_mode != "auto":
        preprocess_mode = requested_mode
    elif compile_enabled and deepx_config_has_graph_normalization(config):
        preprocess_mode = "raw"
    else:
        preprocess_mode = "normalized"

    return DeepXImageInputConfig(
        preprocess_mode=preprocess_mode,
        short_side=short_side,
        input_info=input_info,
        input_layout=input_layout,
        input_dtype=input_dtype,
        runtime_options=runtime_options,
    )


class DeepXImageClassificationLoader(ImageClassificationLoader):
    """Image classification loader that follows DeepX DXNN input metadata."""

    def __init__(self, model_spec: Model_Spec, **kwargs):
        kwargs = dict(kwargs)
        self.deepx_input_config = resolve_deepx_image_input_config(
            artifact_path=kwargs.get("artifact_path"),
            compile_options=kwargs.get("compile_options", {}),
            requested_mode=kwargs.get("image_preprocess_mode", "auto"),
            compile_enabled=bool(kwargs.get("compile_enabled", True)),
        )

        if self.deepx_input_config.input_layout:
            kwargs["layout"] = self.deepx_input_config.input_layout

        kwargs["preprocess_strategy"] = self._create_preprocess_strategy()
        super().__init__(model_spec, **kwargs)

    def _create_preprocess_strategy(self) -> PreprocessStrategy:
        short_side = self.deepx_input_config.short_side or 256
        if self.deepx_input_config.preprocess_mode == "raw":
            if self.deepx_input_config.expects_uint8_image:
                layout = self.deepx_input_config.input_layout
                dtype = self.deepx_input_config.input_dtype
                print(
                    "[DataLoader] DeepX image preprocess: raw "
                    f"(dx_app direct resize; DXNN input={layout}/{dtype})"
                )
                return DeepXDirectResizeRawPreprocess()

            print(
                "[DataLoader] DeepX image preprocess: raw "
                f"(resize short-side={short_side}, crop only; graph-side div/normalize expected)"
            )
            return MLPerfResNet50RawPreprocess(short_side=short_side)

        print(
            "[DataLoader] DeepX image preprocess: normalized "
            f"(resize short-side={short_side}, div/normalize in loader)"
        )
        return MLPerfResNet50Preprocess(short_side=short_side)

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata["deepx_input"] = {
            "preprocess_mode": self.deepx_input_config.preprocess_mode,
            "input_layout": self.deepx_input_config.input_layout,
            "input_dtype": self.deepx_input_config.input_dtype,
            "short_side": self.deepx_input_config.short_side,
        }
        metadata["runtime_options"] = dict(self.deepx_input_config.runtime_options)
        return metadata
