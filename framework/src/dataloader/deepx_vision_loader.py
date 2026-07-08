"""DeepX vision dataloaders for YOLO-style tasks.

DX-APP routes object detection, instance segmentation, and pose estimation
through the same LetterboxPreprocessor family. These loaders keep that DeepX
contract separate from the generic object-detection loader, which still uses
the framework's direct-resize preprocessing for non-DeepX backends.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

from .base import DataLoader
from .deepx_image_classification_loader import (
    deepx_rmap_image_input_layout,
    deepx_rmap_input_dtype,
    read_dxnn_rmap_input_info,
)
from core.model_spec import Model_Spec, Task


_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")


@dataclass
class DeepXVisionInputConfig:
    input_layout: str
    input_dtype: str | None = None
    input_info: dict | None = None
    tensor_dtype: str = "uint8"
    runtime_options: dict[str, Any] = field(default_factory=dict)


def _image_layout_from_shape(shape: Tuple[int, ...] | None) -> str | None:
    if not shape or len(shape) != 4:
        return None
    if shape[-1] in (1, 3, 4):
        return "NHWC"
    if shape[1] in (1, 3, 4):
        return "NCHW"
    return None


def _first_input_shape(model_spec: Model_Spec) -> Tuple[int, ...] | None:
    if not model_spec.input_shapes:
        return None
    return tuple(next(iter(model_spec.input_shapes.values())))


def resolve_deepx_vision_input_config(
    model_spec: Model_Spec,
    *,
    artifact_path: str | Path | None,
    requested_layout: str = "NCHW",
) -> DeepXVisionInputConfig:
    input_info = read_dxnn_rmap_input_info(artifact_path)
    requested_layout = requested_layout.upper() if requested_layout else "NCHW"
    input_layout = (
        deepx_rmap_image_input_layout(input_info)
        or requested_layout
        or _image_layout_from_shape(_first_input_shape(model_spec))
    )
    input_dtype = deepx_rmap_input_dtype(input_info)

    runtime_options: dict[str, Any] = {
        "input_layout": input_layout,
        "input_batch_axis": "squeeze",
        "single_input_run_style": "list",
    }
    tensor_dtype = "uint8"
    if input_dtype in ("FLOAT", "FLOAT32"):
        runtime_options["input_dtype"] = "float32"
        tensor_dtype = "float32"
    elif input_dtype == "UINT8":
        runtime_options["input_dtype"] = "uint8"
        tensor_dtype = "uint8"

    return DeepXVisionInputConfig(
        input_layout=input_layout,
        input_dtype=input_dtype,
        input_info=input_info,
        tensor_dtype=tensor_dtype,
        runtime_options=runtime_options,
    )


class DeepXLetterboxPreprocess:
    """dx_app LetterboxPreprocessor equivalent for PIL/numpy DataLoaders."""

    def __init__(
        self,
        target_hw: Tuple[int, int],
        *,
        layout: str = "NCHW",
        tensor_dtype: str = "uint8",
        pad_color: Tuple[int, int, int] = (114, 114, 114),
    ):
        self.target_hw = tuple(target_hw)
        self.layout = layout.upper()
        self.tensor_dtype = tensor_dtype
        self.pad_color = pad_color

    def cache_key(self) -> str:
        h, w = self.target_hw
        return f"deepx_letterbox_{h}x{w}_{self.layout}_{self.tensor_dtype}"

    def __call__(self, img: Image.Image) -> tuple[np.ndarray, Dict[str, Any]]:
        img = img.convert("RGB")
        orig_w, orig_h = img.size
        target_h, target_w = self.target_hw

        gain = min(target_h / orig_h, target_w / orig_w)
        new_w = int(round(orig_w * gain))
        new_h = int(round(orig_h * gain))
        pad_w = (target_w - new_w) / 2
        pad_h = (target_h - new_h) / 2

        if (orig_w, orig_h) != (new_w, new_h):
            img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        left = int(round(pad_w - 0.1))
        right = int(round(pad_w + 0.1))
        top = int(round(pad_h - 0.1))
        bottom = int(round(pad_h + 0.1))

        canvas = Image.new("RGB", (target_w, target_h), self.pad_color)
        canvas.paste(img, (left, top))
        array = np.asarray(canvas, dtype=np.uint8)

        if self.tensor_dtype == "float32":
            array = array.astype(np.float32) / 255.0
        if self.layout == "NCHW":
            array = np.transpose(array, (2, 0, 1))
        elif self.layout != "NHWC":
            raise ValueError(f"Unsupported DeepX vision layout: {self.layout}")

        ctx = {
            "original_width": orig_w,
            "original_height": orig_h,
            "input_width": target_w,
            "input_height": target_h,
            "scale": gain,
            "pad_x": left,
            "pad_y": top,
            "layout": self.layout,
        }
        return np.ascontiguousarray(array), ctx


class DeepXVisionLoader(DataLoader):
    """Common DeepX loader for YOLO-style image-to-vision tasks."""

    deepx_task_type = "vision"

    def __init__(self, model_spec: Model_Spec, **kwargs):
        self.model_spec = model_spec
        self.task = model_spec.task
        self.base_path = kwargs.get("dataset_path", "./data/coco128")
        self.image_dir = kwargs.get("image_dir") or self.base_path
        self.label_path = kwargs.get("label_path")
        self.current_idx = 0

        if not self.image_dir:
            raise ValueError("[DeepXVisionLoader] image_dir 또는 dataset_path가 필요합니다.")

        self.image_files = self._discover_images(self.image_dir)
        self.total_samples = len(self.image_files)
        if self.total_samples == 0:
            raise FileNotFoundError(
                f"[DeepXVisionLoader] '{self.image_dir}' 경로에 이미지가 존재하지 않습니다."
            )

        self.target_hw = self._parse_target_shape(kwargs)
        self.deepx_input_config = resolve_deepx_vision_input_config(
            model_spec,
            artifact_path=kwargs.get("artifact_path"),
            requested_layout=kwargs.get("layout", "NCHW"),
        )
        self.layout = self.deepx_input_config.input_layout
        self.preprocessor = DeepXLetterboxPreprocess(
            self.target_hw,
            layout=self.layout,
            tensor_dtype=self.deepx_input_config.tensor_dtype,
        )

        self.cache_dir = kwargs.get("cache_dir")
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        print(
            "[DataLoader] DeepX vision preprocess: letterbox "
            f"(task={self.deepx_task_type}, input={self.layout}/{self.deepx_input_config.input_dtype or 'auto'}, "
            f"dtype={self.deepx_input_config.tensor_dtype})"
        )

    def _discover_images(self, image_dir: str) -> list[str]:
        if not os.path.exists(image_dir):
            return []
        return sorted(
            item
            for item in os.listdir(image_dir)
            if item.lower().endswith(_IMAGE_EXTENSIONS)
        )

    def _parse_target_shape(self, kwargs: Dict[str, Any]) -> Tuple[int, int]:
        if "target_hw" in kwargs:
            return tuple(kwargs["target_hw"])
        shape = _first_input_shape(self.model_spec)
        if shape:
            spatial_dims = [dim for dim in shape if dim is not None and dim > 4]
            if len(spatial_dims) >= 2:
                return int(spatial_dims[0]), int(spatial_dims[1])
        return 640, 640

    def _cache_path(self, img_filename: str) -> str | None:
        if not self.cache_dir:
            return None
        stem = Path(img_filename).stem
        return str(Path(self.cache_dir) / f"{stem}.{self.preprocessor.cache_key()}.npz")

    def _load_or_preprocess(self, img_path: str, img_filename: str) -> tuple[np.ndarray, Dict[str, Any]]:
        cache_path = self._cache_path(img_filename)
        if cache_path and os.path.exists(cache_path):
            try:
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
                }
                return np.ascontiguousarray(cached["input"]), ctx
            except Exception:
                pass

        with Image.open(img_path) as img:
            tensor, ctx = self.preprocessor(img)

        if cache_path:
            np.savez_compressed(cache_path, input=tensor, **ctx)
        return tensor, ctx

    def _get_label_file(self, img_filename: str) -> str | None:
        if not self.label_path:
            return None
        if os.path.isdir(self.label_path):
            return os.path.join(self.label_path, f"{Path(img_filename).stem}.txt")
        return self.label_path if os.path.isfile(self.label_path) else None

    def _parse_label(self, img_filename: str) -> Any:
        label_file = self._get_label_file(img_filename)
        if not label_file or not os.path.exists(label_file):
            return np.empty((0, 5), dtype=np.float32)

        rows = []
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    rows.append([float(value) for value in parts])
                except ValueError:
                    continue
        if not rows:
            return np.empty((0, 5), dtype=np.float32)
        if len({len(row) for row in rows}) != 1:
            return [np.asarray(row, dtype=np.float32) for row in rows]
        return np.asarray(rows, dtype=np.float32)

    def _sample_at(self, index: int) -> Dict[str, Any]:
        img_filename = self.image_files[index]
        img_path = os.path.join(self.image_dir, img_filename)
        tensor, ctx = self._load_or_preprocess(img_path, img_filename)
        return {
            "input": tensor,
            "label": self._parse_label(img_filename),
            "img_path": img_path,
            "preprocess_context": ctx,
        }

    def load_single(self) -> Dict[str, Any]:
        if self.current_idx >= self.total_samples:
            raise StopIteration("모든 샘플이 소진되었습니다.")
        sample = self._sample_at(self.current_idx)
        self.current_idx += 1
        return sample

    def load_batch(self, batch_size: int) -> list[Dict[str, Any]]:
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(self.load_single())
            except StopIteration:
                break
        return batch

    def load_by_index(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= self.total_samples:
            raise IndexError(f"index {index} is out of range [0, {self.total_samples})")
        return self._sample_at(index)

    def get_labels(self) -> Any:
        return None

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "dataset_path": self.base_path,
            "image_dir": self.image_dir,
            "label_path": self.label_path,
            "target_hw": self.target_hw,
            "preprocessor": type(self.preprocessor).__name__,
            "cache_dir": self.cache_dir,
            "deepx_input": {
                "task_type": self.deepx_task_type,
                "input_layout": self.deepx_input_config.input_layout,
                "input_dtype": self.deepx_input_config.input_dtype,
                "tensor_dtype": self.deepx_input_config.tensor_dtype,
            },
            "runtime_options": dict(self.deepx_input_config.runtime_options),
        }

    def preprocess(self, raw_input: Any) -> np.ndarray:
        if isinstance(raw_input, (str, Path)):
            with Image.open(raw_input) as img:
                tensor, _ = self.preprocessor(img)
                return tensor
        if isinstance(raw_input, Image.Image):
            tensor, _ = self.preprocessor(raw_input)
            return tensor
        if isinstance(raw_input, np.ndarray):
            img = Image.fromarray(raw_input.astype(np.uint8), mode="RGB")
            tensor, _ = self.preprocessor(img)
            return tensor
        raise TypeError(f"Unsupported raw input type for DeepX vision preprocessing: {type(raw_input)!r}")


class DeepXObjectDetectionLoader(DeepXVisionLoader):
    deepx_task_type = "object_detection"


class DeepXInstanceSegmentationLoader(DeepXVisionLoader):
    deepx_task_type = "instance_segmentation"


class DeepXPoseEstimationLoader(DeepXVisionLoader):
    deepx_task_type = "pose_estimation"


def is_deepx_vision_task(task: Task) -> bool:
    return task in {
        Task.OBJECT_DETECTION,
        Task.INSTANCE_SEGMENTATION,
        Task.POSE_ESTIMATION,
    }
