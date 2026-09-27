"""Hailo YOLOv8-pose input adapter for the nine-head HEF ABI."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image


class HailoYoloV8PosePreprocessor:
    """Apply PIL letterbox and return raw uint8 NHWC RGB input."""

    target_hw = (640, 640)

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
            raise TypeError(
                "Hailo YOLOv8-pose input must be an image path or PIL.Image"
            )

        original_width, original_height = image.size
        input_height, input_width = self.target_hw
        scale = min(
            input_width / original_width,
            input_height / original_height,
        )
        resized_width = int(round(original_width * scale))
        resized_height = int(round(original_height * scale))
        image = image.resize(
            (resized_width, resized_height),
            Image.Resampling.BILINEAR,
        )
        pad_x = int(round((input_width - resized_width) / 2 - 0.1))
        pad_y = int(round((input_height - resized_height) / 2 - 0.1))
        canvas = Image.new(
            "RGB", (input_width, input_height), (114, 114, 114)
        )
        canvas.paste(image, (pad_x, pad_y))
        return np.ascontiguousarray(np.asarray(canvas, dtype=np.uint8)), {
            "original_height": original_height,
            "original_width": original_width,
            "input_height": input_height,
            "input_width": input_width,
            "scale": float(scale),
            "pad_x": float(pad_x),
            "pad_y": float(pad_y),
        }

    def get_cache_path(self, cache_dir, image_filename):
        del cache_dir, image_filename
        return None

    def load_or_preprocess_with_context(self, cache_path, raw_input):
        del cache_path
        return self.preprocess_with_context(raw_input)
