"""Raw RGB letterbox for the Regulus nine-head Pose MXQ."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from core.regulus_vision_contracts import RegulusPoseArtifactProfile
from .base import BasePreprocessor


class RegulusPosePreprocessor(BasePreprocessor):
    image_preprocess_mode = "raw"

    def __init__(self, profile: RegulusPoseArtifactProfile):
        if not isinstance(profile, RegulusPoseArtifactProfile) or profile.preprocess_mode != "raw":
            raise ValueError("Regulus Pose requires a raw profile")
        self.profile = profile
        self.target_hw = profile.input_recipe.input_hw

    def get_cache_path(self, cache_dir: str | Path | None, img_filename: str | Path) -> str | None:
        if not cache_dir:
            return None
        source = Path(img_filename).resolve(strict=True)
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        identity = json.dumps([self.profile.profile_id, self.profile.input_recipe.version,
                               str(source), source_hash], separators=(",", ":"))
        key = hashlib.sha256(identity.encode()).hexdigest()
        return str(Path(cache_dir) / f"regulus_pose_{key}.npz")

    def preprocess(self, raw_input: Any) -> np.ndarray:
        return self.preprocess_with_context(raw_input)[0]

    def preprocess_with_context(self, raw_input: Any) -> tuple[np.ndarray, dict[str, Any]]:
        if isinstance(raw_input, (str, os.PathLike)):
            with Image.open(raw_input) as source:
                rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        elif isinstance(raw_input, Image.Image):
            rgb = np.asarray(raw_input.convert("RGB"), dtype=np.uint8)
        else:
            raise TypeError("Regulus Pose input must be an image path or PIL image")
        h0, w0 = rgb.shape[:2]
        target_h, target_w = self.target_hw
        ratio = min(target_h / h0, target_w / w0)
        new_w, new_h = round(w0 * ratio), round(h0 * ratio)
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        dw, dh = (target_w - new_w) / 2, (target_h - new_h) / 2
        left, right = round(dw - 0.1), round(dw + 0.1)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        tensor = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        context = {
            "original_width": int(w0), "original_height": int(h0),
            "input_width": target_w, "input_height": target_h,
            "scale": float(ratio), "pad_x": int(left), "pad_y": int(top),
            "layout": "NHWC", "resize_mode": "letterbox",
            "ratio_pad": ((float(ratio), float(ratio)), (int(left), int(top))),
            "profile_id": self.profile.profile_id,
        }
        return np.ascontiguousarray(tensor, dtype=np.uint8), context

    def load_or_preprocess_with_context(self, cache_path: str | None, raw_input: Any):
        if cache_path and Path(cache_path).exists():
            with np.load(cache_path, allow_pickle=False) as cached:
                tensor = cached["input"]
                context = json.loads(str(cached["context_json"].item()))
            if tensor.shape != self.profile.unbatched_input_shape or tensor.dtype != np.uint8:
                raise ValueError("Regulus Pose cache input contract mismatch")
            if context.get("profile_id") != self.profile.profile_id:
                raise ValueError("Regulus Pose cache profile mismatch")
            context["ratio_pad"] = tuple(tuple(row) for row in context["ratio_pad"])
            return np.ascontiguousarray(tensor), context
        tensor, context = self.preprocess_with_context(raw_input)
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, input=tensor, context_json=json.dumps(context))
        return tensor, context
