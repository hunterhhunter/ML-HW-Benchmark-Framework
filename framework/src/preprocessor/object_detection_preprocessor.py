"""
ObjectDetectionPreprocessor — 객체 탐지(Object Detection)용 전처리기

YOLO 스타일의 이미지를 리사이즈하여 numpy 텐서로 변환합니다.
일반 ONNX 경로는 normalized 0..1 float 입력을, Hailo HEF 경로는 raw 0..255
uint8 입력을 사용할 수 있습니다.
샘플 단위 .npy/.npz 파일로 캐싱하여 반복 실행 시 전처리 비용을 제거합니다.
"""

import os
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
from PIL import Image

from .base import BasePreprocessor


class ObjectDetectionPreprocessor(BasePreprocessor):
    """
    객체 탐지 모델(YOLO 계열)용 전처리기.

    기본적으로 이미지를 비율 왜곡 방식으로 target_hw로 직접 리사이즈한 뒤,
    preprocess_mode에 따라 raw 픽셀 또는 normalized float 텐서를 반환합니다.
    Hailo처럼 YOLO letterbox 계약이 필요한 백엔드는 resize_mode="letterbox"로
    원본 비율 유지, padding, 좌표 복원 context를 함께 사용할 수 있습니다.
    NCHW 또는 NHWC 레이아웃을 지원합니다.

    Args:
        target_hw: 모델 입력 해상도 (H, W). 기본값 (640, 640).
        mean:      채널별 정규화 평균 (shape [3]). 기본값 [0, 0, 0].
        std:       채널별 정규화 표준편차 (shape [3]). 기본값 [1, 1, 1].
        layout:    메모리 레이아웃 "NCHW" 또는 "NHWC". 기본값 "NCHW".
        preprocess_mode: "normalized"는 0..1 float 정규화, "raw"는 0..255 uint8.
        resize_mode: "direct"는 왜곡 resize, "letterbox"는 비율 유지 padding.
    """

    def __init__(
        self,
        target_hw: Tuple[int, int] = (640, 640),
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        layout: str = "NCHW",
        preprocess_mode: str = "normalized",
        resize_mode: str = "direct",
        pad_color: Tuple[int, int, int] = (114, 114, 114),
    ):
        self.target_hw = tuple(target_hw)
        self.mean   = mean   if mean   is not None else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.std    = std    if std    is not None else np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.layout = layout.upper()
        self.preprocess_mode = preprocess_mode.lower()
        self.resize_mode = resize_mode.lower()
        self.pad_color = tuple(int(v) for v in pad_color)
        if self.preprocess_mode not in {"normalized", "raw"}:
            raise ValueError(
                "ObjectDetectionPreprocessor preprocess_mode must be "
                f"'normalized' or 'raw', got {preprocess_mode!r}"
            )
        if self.resize_mode not in {"direct", "letterbox"}:
            raise ValueError(
                "ObjectDetectionPreprocessor resize_mode must be "
                f"'direct' or 'letterbox', got {resize_mode!r}"
            )
        if self.layout not in {"NCHW", "NHWC"}:
            raise ValueError(f"Unsupported object detection layout: {layout!r}")

    def preprocess(self, raw_input: Any) -> np.ndarray:
        """
        이미지를 리사이즈하고 정규화하여 numpy 텐서를 반환합니다.

        Args:
            raw_input: PIL.Image 객체 또는 이미지 파일 경로 문자열.

        Returns:
            np.ndarray:
              normalized NCHW → shape (C, H, W), dtype float32
              normalized NHWC → shape (H, W, C), dtype float32
              raw NCHW → shape (C, H, W), dtype uint8
              raw NHWC → shape (H, W, C), dtype uint8
        """
        tensor, _ = self.preprocess_with_context(raw_input)
        return tensor

    def preprocess_with_context(self, raw_input: Any) -> Tuple[np.ndarray, Dict[str, Any] | None]:
        """전처리 텐서와 optional 좌표계 context를 함께 반환합니다."""
        img = Image.open(raw_input) if isinstance(raw_input, str) else raw_input
        img = img.convert("RGB")
        if self.resize_mode == "letterbox":
            return self._preprocess_letterbox(img)
        return self._preprocess_direct(img), None

    def load_or_preprocess_with_context(
        self, cache_path: Optional[str], raw_input: Any
    ) -> Tuple[np.ndarray, Dict[str, Any] | None]:
        """Context를 보존해야 하는 letterbox 경로용 캐시 로직."""
        if cache_path and os.path.exists(cache_path):
            if cache_path.endswith(".npz"):
                cached = np.load(cache_path, allow_pickle=False)
                ctx = {
                    "original_width": int(cached["original_width"]),
                    "original_height": int(cached["original_height"]),
                    "input_width": int(cached["input_width"]),
                    "input_height": int(cached["input_height"]),
                    "scale": float(cached["scale"]),
                    "pad_x": int(cached["pad_x"]),
                    "pad_y": int(cached["pad_y"]),
                    "layout": str(cached["layout"]),
                    "resize_mode": str(cached["resize_mode"]),
                }
                return np.ascontiguousarray(cached["input"]), ctx
            return np.load(cache_path), None

        tensor, ctx = self.preprocess_with_context(raw_input)

        if cache_path:
            cache_dir = os.path.dirname(os.path.abspath(cache_path))
            os.makedirs(cache_dir, exist_ok=True)
            if ctx is None:
                np.save(cache_path, tensor)
            else:
                np.savez_compressed(cache_path, input=tensor, **ctx)

        return tensor, ctx

    def _preprocess_direct(self, img: Image.Image) -> np.ndarray:
        img = img.resize((self.target_hw[1], self.target_hw[0]), Image.Resampling.BILINEAR)
        return self._format_array(np.asarray(img))

    def _preprocess_letterbox(self, img: Image.Image) -> Tuple[np.ndarray, Dict[str, Any]]:
        orig_w, orig_h = img.size
        target_h, target_w = self.target_hw

        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        if (orig_w, orig_h) != (new_w, new_h):
            img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        canvas = Image.new("RGB", (target_w, target_h), self.pad_color)
        canvas.paste(img, (pad_x, pad_y))

        ctx = {
            "original_width": orig_w,
            "original_height": orig_h,
            "input_width": target_w,
            "input_height": target_h,
            "scale": scale,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "layout": self.layout,
            "resize_mode": self.resize_mode,
        }
        return self._format_array(np.asarray(canvas)), ctx

    def _format_array(self, img_array: np.ndarray) -> np.ndarray:
        if self.preprocess_mode == "raw":
            img_array = img_array.astype(np.uint8, copy=False)
        else:
            img_array = img_array.astype(np.float32) / 255.0
            img_array = (img_array - self.mean) / self.std  # (H, W, C)

        if self.layout == "NCHW":
            img_array = np.transpose(img_array, (2, 0, 1))  # (C, H, W)

        return np.ascontiguousarray(img_array)

    def get_cache_path(self, cache_dir: Optional[str], img_filename: str) -> Optional[str]:
        """
        이미지 파일명 기반으로 .npy 캐시 파일 경로를 생성합니다.

        Args:
            cache_dir:    캐시 디렉토리 경로. None이면 None 반환.
            img_filename: 이미지 파일명 (확장자 포함).

        Returns:
            str 또는 None: 캐시 파일 경로.
        """
        if not cache_dir:
            return None
        stem = Path(img_filename).stem
        h, w = self.target_hw
        suffix = "npz" if self.resize_mode == "letterbox" else "npy"
        return str(
            Path(cache_dir)
            / f"{stem}_{self.resize_mode}_{self.preprocess_mode}_{self.layout}_{h}x{w}.{suffix}"
        )
