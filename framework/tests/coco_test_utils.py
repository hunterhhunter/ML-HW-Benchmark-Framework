import json
from pathlib import Path

from PIL import Image

from core.model_spec import Model_Spec, Task


def write_coco_fixture(root: Path) -> dict[str, Path]:
    image_dir = root / "images" / "val2017"
    annotation_dir = root / "annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    Image.new("RGB", (8, 6), (255, 255, 255)).save(
        image_dir / "000000000001.jpg"
    )
    images = [
        {
            "id": 1,
            "file_name": "000000000001.jpg",
            "width": 8,
            "height": 6,
        }
    ]
    common = {"info": {}, "licenses": [], "images": images}
    instances = {
        **common,
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [2, 1, 4, 4],
                "area": 16,
                "iscrowd": 0,
                "segmentation": [[2, 1, 6, 1, 6, 5, 2, 5]],
            }
        ],
        "categories": [
            {"id": 1, "name": "person", "supercategory": "person"}
        ],
    }
    pose = {
        **common,
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [2, 1, 4, 4],
                "area": 16,
                "iscrowd": 0,
                "num_keypoints": 17,
                "keypoints": [4, 3, 2] * 17,
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": [f"kpt_{index}" for index in range(17)],
                "skeleton": [],
            }
        ],
    }
    instances_path = annotation_dir / "instances_val2017.json"
    pose_path = annotation_dir / "person_keypoints_val2017.json"
    instances_path.write_text(json.dumps(instances), encoding="utf-8")
    pose_path.write_text(json.dumps(pose), encoding="utf-8")
    return {
        "images": image_dir,
        "instances": instances_path,
        "pose": pose_path,
    }


def make_seg_spec() -> Model_Spec:
    return Model_Spec(
        name="yolov8s-seg",
        task=Task.INSTANCE_SEGMENTATION,
        input_shapes={"images": (1, 3, 640, 640)},
        input_dtype={"images": "float32"},
        output_shapes={
            "output0": (1, 116, 8400),
            "output1": (1, 32, 160, 160),
        },
    )


def make_pose_spec() -> Model_Spec:
    return Model_Spec(
        name="yolov8s-pose",
        task=Task.POSE_ESTIMATION,
        input_shapes={"images": (1, 3, 640, 640)},
        input_dtype={"images": "float32"},
        output_shapes={"output0": (1, 56, 8400)},
    )
