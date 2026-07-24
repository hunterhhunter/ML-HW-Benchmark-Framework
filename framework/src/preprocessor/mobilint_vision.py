"""Exact image preprocessing contracts for supported Mobilint artifacts."""

from dataclasses import asdict
import hashlib
import json
import math
import os
from numbers import Integral, Real
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
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    return self._validated_cache_entry(cached)
            except ValueError as exc:
                if str(exc).startswith("Mobilint YOLO cache"):
                    raise
                raise ValueError("Mobilint YOLO cache is corrupt.") from exc
            except (KeyError, TypeError, IndexError, OverflowError) as exc:
                raise ValueError("Mobilint YOLO cache is corrupt.") from exc

        tensor, context = self.preprocess_with_context(raw_input)
        if cache_path:
            cache_dir = os.path.dirname(os.path.abspath(cache_path))
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(cache_path, input=tensor, **context)
        return tensor, context

    def _validated_cache_entry(
        self,
        cached: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        tensor = np.asarray(cached["input"])
        expected_dtype = np.dtype(self.profile.input_dtype)
        if tensor.dtype != expected_dtype:
            raise ValueError("Mobilint YOLO cache input dtype mismatch.")
        if tuple(tensor.shape) != tuple(self.profile.unbatched_input_shape):
            raise ValueError("Mobilint YOLO cache input shape mismatch.")

        original_width = self._cached_integer(cached, "original_width")
        original_height = self._cached_integer(cached, "original_height")
        input_width = self._cached_integer(cached, "input_width")
        input_height = self._cached_integer(cached, "input_height")
        pad_x = self._cached_integer(cached, "pad_x")
        pad_y = self._cached_integer(cached, "pad_y")
        scale = self._cached_real(cached, "scale")
        layout = self._cached_text(cached, "layout")
        resize_mode = self._cached_text(cached, "resize_mode")
        profile_id = self._cached_text(cached, "profile_id")
        ratio_pad = np.asarray(cached["ratio_pad"])

        target_height, target_width = self.recipe.input_hw
        if profile_id != self.profile.profile_id:
            raise ValueError("Mobilint YOLO cache profile ID mismatch.")
        if layout != self.profile.input_layout:
            raise ValueError("Mobilint YOLO cache layout mismatch.")
        if (input_height, input_width) != (target_height, target_width):
            raise ValueError("Mobilint YOLO cache input size mismatch.")
        if resize_mode != "letterbox":
            raise ValueError("Mobilint YOLO cache resize mode mismatch.")
        if original_width <= 0 or original_height <= 0:
            raise ValueError("Mobilint YOLO cache original size is invalid.")

        expected_scale = min(
            target_height / original_height,
            target_width / original_width,
        )
        new_width = int(round(original_width * expected_scale))
        new_height = int(round(original_height * expected_scale))
        expected_pad_x = int(round((target_width - new_width) / 2 - 0.1))
        expected_pad_y = int(round((target_height - new_height) / 2 - 0.1))
        if not math.isclose(scale, expected_scale, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Mobilint YOLO cache scale mismatch.")
        if (pad_x, pad_y) != (expected_pad_x, expected_pad_y):
            raise ValueError("Mobilint YOLO cache padding mismatch.")
        if ratio_pad.shape != (2, 2) or not np.all(np.isfinite(ratio_pad)):
            raise ValueError("Mobilint YOLO cache ratio_pad is invalid.")
        expected_ratio_pad = np.asarray(
            ((scale, scale), (pad_x, pad_y)),
            dtype=np.float64,
        )
        if not np.array_equal(ratio_pad, expected_ratio_pad):
            raise ValueError("Mobilint YOLO cache ratio_pad mismatch.")

        context = {
            "original_width": original_width,
            "original_height": original_height,
            "input_width": input_width,
            "input_height": input_height,
            "scale": scale,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "layout": layout,
            "resize_mode": resize_mode,
            "ratio_pad": ((scale, scale), (pad_x, pad_y)),
            "profile_id": profile_id,
        }
        return np.ascontiguousarray(tensor), context

    @staticmethod
    def _cached_scalar(cached: Any, name: str) -> Any:
        value = np.asarray(cached[name])
        if value.shape != ():
            raise ValueError(f"Mobilint YOLO cache {name} must be scalar.")
        return value.item()

    @classmethod
    def _cached_integer(cls, cached: Any, name: str) -> int:
        value = cls._cached_scalar(cached, name)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"Mobilint YOLO cache {name} must be an integer.")
        return int(value)

    @classmethod
    def _cached_real(cls, cached: Any, name: str) -> float:
        value = cls._cached_scalar(cached, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"Mobilint YOLO cache {name} must be finite.")
        return float(value)

    @classmethod
    def _cached_text(cls, cached: Any, name: str) -> str:
        value = cls._cached_scalar(cached, name)
        if type(value) is not str:
            raise ValueError(f"Mobilint YOLO cache {name} must be text.")
        return value

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
