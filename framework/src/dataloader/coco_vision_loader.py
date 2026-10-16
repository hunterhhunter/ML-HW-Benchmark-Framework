"""Validated lazy COCO image loader shared by segmentation and pose."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from core.model_spec import Model_Spec, Task
from preprocessor.yolo_vision_preprocessor import YoloVisionPreprocessor

from .base import DataLoader


class CocoVisionLoader(DataLoader):
    """Index COCO metadata eagerly while decoding images only on demand."""

    expected_task: Task

    def __init__(self, model_spec: Model_Spec, **kwargs: Any) -> None:
        if model_spec.task is not self.expected_task:
            raise ValueError(
                f"{type(self).__name__} requires {self.expected_task.name}"
            )
        self.model_spec = model_spec
        self.dataset_path = str(kwargs.get("dataset_path", ""))
        self.image_dir = self._required_path(kwargs, "image_dir", directory=True)
        self.annotation_file = self._required_path(
            kwargs, "label_path", directory=False
        )
        self.cache_dir = kwargs.get("cache_dir")

        preprocess_mode = str(
            kwargs.get("image_preprocess_mode", "auto") or "auto"
        ).lower()
        if preprocess_mode not in {"auto", "normalized"}:
            raise ValueError(
                "COCO YOLOv8 loaders require normalized float input"
            )
        resize_mode = str(
            kwargs.get("image_resize_mode", "auto") or "auto"
        ).lower()
        if resize_mode not in {"auto", "letterbox"}:
            raise ValueError("COCO YOLOv8 loaders require letterbox resize")

        target_hw = kwargs.get("target_hw") or self._model_target_hw()
        self.preprocessor = kwargs.get(
            "preprocessor"
        ) or YoloVisionPreprocessor(
            target_hw=target_hw,
            layout=kwargs.get("layout", "NCHW"),
        )
        self.target_hw = tuple(self.preprocessor.target_hw)

        payload = self._read_annotation_payload()
        self.images = self._validate_and_index(payload)
        self.category_ids = sorted(
            int(category["id"]) for category in payload["categories"]
        )
        self._validate_task_payload(payload)
        self.total_samples = len(self.images)
        self.current_idx = 0

    @staticmethod
    def _required_path(
        kwargs: dict[str, Any], key: str, *, directory: bool
    ) -> Path:
        value = kwargs.get(key)
        if not value:
            raise ValueError(f"COCO vision loader requires {key}")
        path = Path(value)
        if directory and not path.is_dir():
            raise FileNotFoundError(f"COCO image directory does not exist: {path}")
        if not directory and not path.is_file():
            raise FileNotFoundError(
                f"COCO annotation file does not exist: {path}"
            )
        return path

    def _model_target_hw(self) -> tuple[int, int]:
        shape = tuple(next(iter(self.model_spec.input_shapes.values())))
        if (
            len(shape) != 4
            or shape[1] != 3
            or shape[2] is None
            or shape[3] is None
        ):
            raise ValueError(
                "COCO YOLOv8 loaders require a static NCHW model input"
            )
        return int(shape[2]), int(shape[3])

    def _read_annotation_payload(self) -> dict[str, Any]:
        try:
            with self.annotation_file.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"malformed COCO annotation JSON: {self.annotation_file}: {exc}"
            ) from exc
        except OSError as exc:
            raise ValueError(
                f"cannot read COCO annotation file: {self.annotation_file}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError("COCO annotation root must be an object")
        for key in ("images", "annotations", "categories"):
            if not isinstance(payload.get(key), list):
                raise ValueError(f"COCO annotation requires a list field: {key}")
        return payload

    def _validate_and_index(
        self, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        image_by_id: dict[int, dict[str, Any]] = {}
        image_root = self.image_dir.resolve()
        for image in payload["images"]:
            if not isinstance(image, dict):
                raise ValueError("COCO image entries must be objects")
            image_id = self._required_integer(image, "id", "image")
            if image_id in image_by_id:
                raise ValueError(f"duplicate image id {image_id}")
            file_name = image.get("file_name")
            if not isinstance(file_name, str) or not file_name:
                raise ValueError(
                    f"COCO image {image_id} requires a non-empty file_name"
                )
            image_path = (self.image_dir / file_name).resolve()
            if image_path != image_root and image_root not in image_path.parents:
                raise ValueError(
                    f"COCO image {image_id} points outside image_dir: {file_name}"
                )
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"COCO image {image_id} file does not exist: {image_path}"
                )
            image_by_id[image_id] = {
                **image,
                "id": image_id,
                "file_name": file_name,
            }
        if not image_by_id:
            raise ValueError("COCO annotation contains no images")

        category_ids: set[int] = set()
        for category in payload["categories"]:
            if not isinstance(category, dict):
                raise ValueError("COCO category entries must be objects")
            category_id = self._required_integer(
                category, "id", "category"
            )
            if category_id in category_ids:
                raise ValueError(f"duplicate category id {category_id}")
            category_ids.add(category_id)
        if not category_ids:
            raise ValueError("COCO annotation contains no categories")

        annotation_ids: set[int] = set()
        for annotation in payload["annotations"]:
            if not isinstance(annotation, dict):
                raise ValueError("COCO annotation entries must be objects")
            annotation_id = self._required_integer(
                annotation, "id", "annotation"
            )
            if annotation_id in annotation_ids:
                raise ValueError(f"duplicate annotation id {annotation_id}")
            annotation_ids.add(annotation_id)
            image_id = self._required_integer(
                annotation, "image_id", f"annotation {annotation_id}"
            )
            category_id = self._required_integer(
                annotation, "category_id", f"annotation {annotation_id}"
            )
            if image_id not in image_by_id:
                raise ValueError(
                    f"annotation {annotation_id} references unknown image id "
                    f"{image_id}"
                )
            if category_id not in category_ids:
                raise ValueError(
                    f"annotation {annotation_id} references unknown category id "
                    f"{category_id}"
                )

        return [image_by_id[key] for key in sorted(image_by_id)]

    @staticmethod
    def _required_integer(
        record: dict[str, Any], key: str, description: str
    ) -> int:
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"COCO {description} requires integer {key}")
        return value

    def _validate_task_payload(self, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    def _sample_at(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self.total_samples:
            raise IndexError(
                f"index {index} is out of range [0, {self.total_samples})"
            )
        image = self.images[index]
        image_path = self.image_dir / image["file_name"]
        cache_path = self.preprocessor.get_cache_path(
            self.cache_dir, image["file_name"]
        )
        tensor, context = self.preprocessor.load_or_preprocess_with_context(
            cache_path, image_path
        )
        return {
            "input": tensor,
            "label": {
                "image_id": int(image["id"]),
                "file_name": image["file_name"],
            },
            "preprocess_context": context,
            "img_path": str(image_path),
        }

    def load_single(self) -> dict[str, Any]:
        if self.current_idx >= self.total_samples:
            raise StopIteration("all COCO vision samples have been consumed")
        sample = self._sample_at(self.current_idx)
        self.current_idx += 1
        return sample

    def load_batch(self, batch_size: int) -> list[dict[str, Any]]:
        if batch_size < 0:
            raise ValueError("batch_size cannot be negative")
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(self.load_single())
            except StopIteration:
                break
        return batch

    def load_by_index(self, index: int) -> dict[str, Any]:
        return self._sample_at(index)

    def get_labels(self) -> list[dict[str, int | str]]:
        return [
            {
                "image_id": int(image["id"]),
                "file_name": image["file_name"],
            }
            for image in self.images
        ]

    def get_metadata(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "dataset_path": self.dataset_path,
            "image_dir": str(self.image_dir),
            "annotation_file": str(self.annotation_file),
            "task": self.expected_task.name,
            "target_hw": self.target_hw,
            "category_ids": list(self.category_ids),
            "is_static_batched": False,
        }

    def preprocess(self, raw_input: Any) -> np.ndarray:
        return self.preprocessor.preprocess(raw_input)
