"""Shared YOLOv8 letterbox preprocessing for segmentation and pose."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Optional

import numpy as np
from PIL import Image

from .base import BasePreprocessor


class YoloVisionPreprocessor(BasePreprocessor):
    """Convert RGB images to normalized NCHW tensors with letterbox context."""

    CACHE_VERSION = 1
    _CONTEXT_KEYS = (
        "original_height",
        "original_width",
        "input_height",
        "input_width",
        "scale",
        "pad_x",
        "pad_y",
    )

    def __init__(
        self,
        target_hw: tuple[int, int] = (640, 640),
        layout: str = "NCHW",
    ) -> None:
        self.target_hw = tuple(int(value) for value in target_hw)
        self.layout = str(layout).upper()
        if len(self.target_hw) != 2 or any(
            value <= 0 for value in self.target_hw
        ):
            raise ValueError("target_hw must contain two positive dimensions")
        if self.layout != "NCHW":
            raise ValueError(
                "YOLOv8 segmentation and pose require NCHW layout"
            )

    def preprocess(self, raw_input: Any) -> np.ndarray:
        tensor, _ = self.preprocess_with_context(raw_input)
        return tensor

    def preprocess_with_context(
        self, raw_input: Any
    ) -> tuple[np.ndarray, dict[str, int | float]]:
        if isinstance(raw_input, (str, os.PathLike)):
            with Image.open(raw_input) as source:
                image = source.convert("RGB")
        elif isinstance(raw_input, Image.Image):
            image = raw_input.convert("RGB")
        else:
            raise TypeError("raw_input must be an image path or PIL.Image")

        original_width, original_height = image.size
        if original_width <= 0 or original_height <= 0:
            raise ValueError("image dimensions must be positive")

        input_height, input_width = self.target_hw
        scale = min(
            input_width / original_width,
            input_height / original_height,
        )
        resized_width = round(original_width * scale)
        resized_height = round(original_height * scale)
        if image.size != (resized_width, resized_height):
            image = image.resize(
                (resized_width, resized_height), Image.Resampling.BILINEAR
            )

        pad_x = (input_width - resized_width) // 2
        pad_y = (input_height - resized_height) // 2
        canvas = Image.new(
            "RGB", (input_width, input_height), (114, 114, 114)
        )
        canvas.paste(image, (pad_x, pad_y))
        image_array = np.asarray(canvas, dtype=np.float32) / 255.0
        tensor = np.ascontiguousarray(image_array.transpose(2, 0, 1))
        context: dict[str, int | float] = {
            "original_height": original_height,
            "original_width": original_width,
            "input_height": input_height,
            "input_width": input_width,
            "scale": float(scale),
            "pad_x": float(pad_x),
            "pad_y": float(pad_y),
        }
        return tensor, context

    def load_or_preprocess_with_context(
        self,
        cache_path: Optional[str | os.PathLike[str]],
        raw_input: Any,
    ) -> tuple[np.ndarray, dict[str, int | float]]:
        if cache_path and Path(cache_path).exists():
            try:
                return self._read_cache(Path(cache_path))
            except (OSError, ValueError, EOFError):
                pass

        tensor, context = self.preprocess_with_context(raw_input)
        if cache_path:
            self._write_cache_atomic(Path(cache_path), tensor, context)
        return tensor, context

    def get_cache_path(
        self,
        cache_dir: Optional[str | os.PathLike[str]],
        image_filename: str,
    ) -> Optional[str]:
        if not cache_dir:
            return None
        height, width = self.target_hw
        filename = (
            f"{Path(image_filename).stem}_v{self.CACHE_VERSION}_letterbox_"
            f"float32_{self.layout}_{height}x{width}.npz"
        )
        return str(Path(cache_dir) / filename)

    def _read_cache(
        self, cache_path: Path
    ) -> tuple[np.ndarray, dict[str, int | float]]:
        required_keys = {"cache_version", "input", *self._CONTEXT_KEYS}
        with np.load(cache_path, allow_pickle=False) as cached:
            if not required_keys.issubset(cached.files):
                raise ValueError("YOLO vision cache schema is incomplete")
            if self._cache_scalar(cached["cache_version"], "cache_version") != (
                self.CACHE_VERSION
            ):
                raise ValueError("YOLO vision cache version mismatch")

            tensor = np.asarray(cached["input"])
            expected_shape = (3, *self.target_hw)
            if tensor.dtype != np.float32 or tensor.shape != expected_shape:
                raise ValueError(
                    "YOLO vision cache tensor dtype or shape mismatch"
                )
            if not np.isfinite(tensor).all():
                raise ValueError("YOLO vision cache tensor is not finite")

            context = {
                key: self._cache_scalar(cached[key], key)
                for key in self._CONTEXT_KEYS
            }

        integer_keys = {
            "original_height",
            "original_width",
            "input_height",
            "input_width",
        }
        typed_context: dict[str, int | float] = {
            key: int(value) if key in integer_keys else float(value)
            for key, value in context.items()
        }
        self._validate_cached_context(typed_context)
        return np.ascontiguousarray(tensor), typed_context

    @staticmethod
    def _cache_scalar(value: np.ndarray, key: str) -> int | float:
        array = np.asarray(value)
        if array.shape != () or not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"YOLO vision cache field is not scalar: {key}")
        scalar = array.item()
        if not np.isfinite(scalar):
            raise ValueError(f"YOLO vision cache field is not finite: {key}")
        return scalar

    def _validate_cached_context(
        self, context: dict[str, int | float]
    ) -> None:
        if (
            context["original_height"] <= 0
            or context["original_width"] <= 0
            or context["input_height"] != self.target_hw[0]
            or context["input_width"] != self.target_hw[1]
            or context["scale"] <= 0
            or context["pad_x"] < 0
            or context["pad_y"] < 0
        ):
            raise ValueError("YOLO vision cache context is incompatible")

    def _write_cache_atomic(
        self,
        cache_path: Path,
        tensor: np.ndarray,
        context: dict[str, int | float],
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez_compressed(
                    stream,
                    cache_version=np.int64(self.CACHE_VERSION),
                    input=tensor,
                    **context,
                )
            os.replace(temporary_path, cache_path)
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            temporary_path.unlink(missing_ok=True)
            raise
