"""COCO instance-segmentation DataLoader."""

from typing import Any

from core.model_spec import Task

from .coco_vision_loader import CocoVisionLoader


class CocoInstanceSegmentationLoader(CocoVisionLoader):
    expected_task = Task.INSTANCE_SEGMENTATION

    def _validate_task_payload(self, payload: dict[str, Any]) -> None:
        for annotation in payload["annotations"]:
            segmentation = annotation.get("segmentation")
            if not isinstance(segmentation, (list, dict)) or not segmentation:
                raise ValueError(
                    f"COCO annotation {annotation['id']} requires segmentation"
                )
