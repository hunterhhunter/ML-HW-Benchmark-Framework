"""COCO 17-keypoint pose-estimation DataLoader."""

from typing import Any

from core.model_spec import Task

from .coco_vision_loader import CocoVisionLoader


class CocoPoseLoader(CocoVisionLoader):
    expected_task = Task.POSE_ESTIMATION

    def _validate_task_payload(self, payload: dict[str, Any]) -> None:
        person_categories = [
            category
            for category in payload["categories"]
            if category["id"] == 1 and category.get("name") == "person"
        ]
        if len(person_categories) != 1 or len(
            person_categories[0].get("keypoints", [])
        ) != 17:
            raise ValueError(
                "COCO pose annotations require person category 1 with 17 keypoint names"
            )
        for annotation in payload["annotations"]:
            keypoints = annotation.get("keypoints")
            if annotation["category_id"] != 1:
                raise ValueError("COCO pose annotations must use person category 1")
            if not isinstance(keypoints, list) or len(keypoints) != 51:
                raise ValueError(
                    f"COCO annotation {annotation['id']} requires 17 keypoints"
                )
