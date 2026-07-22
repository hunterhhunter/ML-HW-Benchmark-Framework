"""Exact image preprocessing contracts for supported Mobilint artifacts."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

from core.mobilint_vision_contracts import (
    MobilintVisionArtifactProfile,
    ResNetCenterCropRecipe,
    YoloV5LetterboxRecipe,
)
from core.model_spec import Task

from .base import BasePreprocessor
from .strategies import PreprocessStrategy


class MobilintResNetCenterCropPreprocess(PreprocessStrategy):
    """Mobilint Model Zoo ResNet-50 raw RGB resize/crop preprocessing."""

    def __init__(self, profile: MobilintVisionArtifactProfile):
        if not isinstance(profile, MobilintVisionArtifactProfile):
            raise ValueError("Mobilint ResNet preprocessing requires a vision profile.")
        if profile.task is not Task.IMAGE_CLASSIFICATION or not isinstance(
            profile.input_recipe, ResNetCenterCropRecipe
        ):
            raise ValueError("Mobilint ResNet preprocessing requires a ResNet vision profile.")
        self.profile = profile
        self.recipe = profile.input_recipe
        self._validate_contract()

    def _validate_contract(self) -> None:
        recipe = self.recipe
        if recipe.interpolation != "pil_bilinear":
            raise ValueError(
                "Mobilint ResNet preprocessing supports only PIL bilinear interpolation."
            )
        if recipe.resize_rounding != "integer_truncation":
            raise ValueError(
                "Mobilint ResNet preprocessing requires integer-truncation resize rounding."
            )
        if recipe.crop_rounding != "python_round":
            raise ValueError(
                "Mobilint ResNet preprocessing requires Python center-crop rounding."
            )
        if recipe.resize_short_side <= 0 or any(size <= 0 for size in recipe.crop_hw):
            raise ValueError("Mobilint ResNet resize and crop dimensions must be positive.")
        if (
            self.profile.preprocess_mode != "raw"
            or self.profile.color_order != "RGB"
            or self.profile.input_layout != "NHWC"
            or self.profile.input_dtype != "uint8"
        ):
            raise ValueError("Mobilint ResNet profile has an unsupported input contract.")

    def __call__(
        self,
        img: Image.Image,
        target_hw: tuple,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> np.ndarray:
        if tuple(target_hw) != self.recipe.crop_hw:
            raise ValueError(
                "Mobilint ResNet target size must match the resolved profile recipe."
            )

        image = img.convert("RGB")
        width, height = image.size
        short_side = self.recipe.resize_short_side
        if width < height:
            resized = (short_side, int(short_side * height / width))
        else:
            resized = (int(short_side * width / height), short_side)
        image = image.resize(resized, Image.Resampling.BILINEAR)

        crop_h, crop_w = self.recipe.crop_hw
        left = round((resized[0] - crop_w) / 2)
        top = round((resized[1] - crop_h) / 2)
        cropped = image.crop((left, top, left + crop_w, top + crop_h))
        chw = np.transpose(np.asarray(cropped, dtype=np.uint8), (2, 0, 1))
        return np.ascontiguousarray(chw)

    def cache_config(self) -> dict[str, object]:
        return {"profile_id": self.profile.profile_id, **asdict(self.recipe)}


class MobilintYoloV5Preprocessor(BasePreprocessor):
    """Mobilint Model Zoo YOLOv5 raw RGB OpenCV letterbox preprocessing."""

    def __init__(self, profile: MobilintVisionArtifactProfile):
        if not isinstance(profile, MobilintVisionArtifactProfile):
            raise ValueError("Mobilint YOLO preprocessing requires a vision profile.")
        if profile.task is not Task.OBJECT_DETECTION or not isinstance(
            profile.input_recipe, YoloV5LetterboxRecipe
        ):
            raise ValueError("Mobilint YOLO preprocessing requires a YOLO vision profile.")
        self.profile = profile
        self.recipe = profile.input_recipe
        self._validate_contract()
        self._cache_signature = self._build_cache_signature()

    def _validate_contract(self) -> None:
        recipe = self.recipe
        if recipe.interpolation != "opencv_linear":
            raise ValueError(
                "Mobilint YOLO preprocessing supports only OpenCV linear interpolation."
            )
        if recipe.resize_rounding != "python_round":
            raise ValueError("Mobilint YOLO preprocessing requires Python resize rounding.")
        if recipe.padding_rounding != "ultralytics_minus_plus_0_1":
            raise ValueError(
                "Mobilint YOLO preprocessing requires Ultralytics padding rounding."
            )
        if any(size <= 0 for size in recipe.input_hw):
            raise ValueError("Mobilint YOLO input dimensions must be positive.")
        if (
            self.profile.preprocess_mode != "raw"
            or self.profile.color_order != "RGB"
            or self.profile.input_layout != "NHWC"
            or self.profile.input_dtype != "uint8"
        ):
            raise ValueError("Mobilint YOLO profile has an unsupported input contract.")

    def _build_cache_signature(self) -> str:
        payload = {
            "profile_id": self.profile.profile_id,
            "recipe_class": type(self.recipe).__name__,
            "recipe_version": self.recipe.version,
            "input_size": list(self.recipe.input_hw),
            "interpolation": self.recipe.interpolation,
            "resize_rounding": self.recipe.resize_rounding,
            "padding_rounding": self.recipe.padding_rounding,
            "pad_color": list(self.recipe.pad_color),
            "layout": self.profile.input_layout,
            "dtype": self.profile.input_dtype,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha1(encoded).hexdigest()[:10]

    def preprocess(self, raw_input: Any) -> np.ndarray:
        tensor, _ = self.preprocess_with_context(raw_input)
        return tensor

    def preprocess_with_context(
        self, raw_input: Any
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if isinstance(raw_input, (str, os.PathLike)):
            with Image.open(raw_input) as source:
                rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        else:
            rgb = np.asarray(raw_input.convert("RGB"), dtype=np.uint8)

        h0, w0 = rgb.shape[:2]
        target_h, target_w = self.recipe.input_hw
        ratio = min(target_h / h0, target_w / w0)
        new_w = int(round(w0 * ratio))
        new_h = int(round(h0 * ratio))
        resized = cv2.resize(
            rgb,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )

        dw = (target_w - new_w) / 2
        dh = (target_h - new_h) / 2
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        tensor = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=self.recipe.pad_color,
        )
        tensor = np.ascontiguousarray(tensor, dtype=np.uint8)

        context = {
            "original_width": int(w0),
            "original_height": int(h0),
            "input_width": int(target_w),
            "input_height": int(target_h),
            "scale": float(ratio),
            "pad_x": int(left),
            "pad_y": int(top),
            "layout": "NHWC",
            "resize_mode": "letterbox",
            "ratio_pad": (
                (float(ratio), float(ratio)),
                (int(left), int(top)),
            ),
            "profile_id": self.profile.profile_id,
        }
        return tensor, context

    def load_or_preprocess_with_context(
        self, cache_path: Optional[str], raw_input: Any
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if cache_path and os.path.exists(cache_path):
            with np.load(cache_path, allow_pickle=False) as cached:
                ratio_pad = np.asarray(cached["ratio_pad"])
                context = {
                    "original_width": int(cached["original_width"]),
                    "original_height": int(cached["original_height"]),
                    "input_width": int(cached["input_width"]),
                    "input_height": int(cached["input_height"]),
                    "scale": float(cached["scale"]),
                    "pad_x": int(cached["pad_x"]),
                    "pad_y": int(cached["pad_y"]),
                    "layout": str(cached["layout"]),
                    "resize_mode": str(cached["resize_mode"]),
                    "ratio_pad": (
                        (float(ratio_pad[0, 0]), float(ratio_pad[0, 1])),
                        (int(ratio_pad[1, 0]), int(ratio_pad[1, 1])),
                    ),
                    "profile_id": str(cached["profile_id"]),
                }
                tensor = np.ascontiguousarray(cached["input"], dtype=np.uint8)
            return tensor, context

        tensor, context = self.preprocess_with_context(raw_input)
        if cache_path:
            cache_dir = os.path.dirname(os.path.abspath(cache_path))
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(cache_path, input=tensor, **context)
        return tensor, context

    def get_cache_path(
        self, cache_dir: Optional[str], img_filename: str
    ) -> Optional[str]:
        if not cache_dir:
            return None
        stem = Path(img_filename).stem
        return str(
            Path(cache_dir)
            / f"{stem}_mobilint_yolov5m_{self._cache_signature}.npz"
        )
