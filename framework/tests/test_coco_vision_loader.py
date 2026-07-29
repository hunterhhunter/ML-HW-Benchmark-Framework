import json

from PIL import Image
import pytest

from coco_test_utils import make_pose_spec, make_seg_spec, write_coco_fixture
from core.model_spec import Task
from dataloader import create_dataloader
from dataloader.coco_instance_segmentation_loader import (
    CocoInstanceSegmentationLoader,
)
from dataloader.coco_pose_loader import CocoPoseLoader


def test_segmentation_loader_returns_tensor_identity_and_context(tmp_path):
    paths = write_coco_fixture(tmp_path)
    loader = CocoInstanceSegmentationLoader(
        make_seg_spec(),
        dataset_path=str(tmp_path),
        image_dir=str(paths["images"]),
        label_path=str(paths["instances"]),
        target_hw=(8, 8),
    )

    sample = loader.load_by_index(0)

    assert sample["input"].shape == (3, 8, 8)
    assert sample["label"] == {
        "image_id": 1,
        "file_name": "000000000001.jpg",
    }
    assert sample["preprocess_context"]["original_width"] == 8
    assert loader.get_metadata() == {
        "total_samples": 1,
        "dataset_path": str(tmp_path),
        "image_dir": str(paths["images"]),
        "annotation_file": str(paths["instances"]),
        "task": Task.INSTANCE_SEGMENTATION.name,
        "target_hw": (8, 8),
        "category_ids": [1],
        "is_static_batched": False,
    }


def test_pose_loader_and_factory_return_pose_samples(tmp_path):
    paths = write_coco_fixture(tmp_path)

    loader = create_dataloader(
        make_pose_spec(),
        dataset_path=str(tmp_path),
        image_dir=str(paths["images"]),
        label_path=str(paths["pose"]),
        target_hw=(8, 8),
    )

    assert isinstance(loader, CocoPoseLoader)
    assert loader.load_single()["label"]["image_id"] == 1
    assert loader.get_metadata()["category_ids"] == [1]


def test_loader_orders_by_image_id_and_random_access_preserves_cursor(tmp_path):
    paths = write_coco_fixture(tmp_path)
    Image.new("RGB", (4, 4), (0, 0, 0)).save(paths["images"] / "zero.jpg")
    payload = json.loads(paths["instances"].read_text(encoding="utf-8"))
    payload["images"].insert(
        0,
        {"id": 3, "file_name": "000000000001.jpg", "width": 8, "height": 6},
    )
    payload["images"].append(
        {"id": 0, "file_name": "zero.jpg", "width": 4, "height": 4}
    )
    paths["instances"].write_text(json.dumps(payload), encoding="utf-8")
    loader = CocoInstanceSegmentationLoader(
        make_seg_spec(),
        dataset_path=str(tmp_path),
        image_dir=str(paths["images"]),
        label_path=str(paths["instances"]),
        target_hw=(8, 8),
    )

    assert loader.load_by_index(2)["label"]["image_id"] == 3
    assert loader.current_idx == 0
    assert [item["label"]["image_id"] for item in loader.load_batch(2)] == [
        0,
        1,
    ]
    assert loader.current_idx == 2
    assert [label["image_id"] for label in loader.get_labels()] == [0, 1, 3]
    assert loader.load_single()["label"]["image_id"] == 3
    with pytest.raises(StopIteration):
        loader.load_single()


def test_loader_rejects_duplicate_image_ids(tmp_path):
    paths = write_coco_fixture(tmp_path)
    payload = json.loads(paths["instances"].read_text(encoding="utf-8"))
    payload["images"].append(dict(payload["images"][0]))
    paths["instances"].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate image id 1"):
        CocoInstanceSegmentationLoader(
            make_seg_spec(),
            dataset_path=str(tmp_path),
            image_dir=str(paths["images"]),
            label_path=str(paths["instances"]),
            target_hw=(8, 8),
        )


def test_loader_rejects_missing_referenced_image(tmp_path):
    paths = write_coco_fixture(tmp_path)
    (paths["images"] / "000000000001.jpg").unlink()

    with pytest.raises(FileNotFoundError, match="000000000001.jpg"):
        CocoInstanceSegmentationLoader(
            make_seg_spec(),
            dataset_path=str(tmp_path),
            image_dir=str(paths["images"]),
            label_path=str(paths["instances"]),
            target_hw=(8, 8),
        )


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("{", "malformed COCO annotation JSON"),
        ("[]", "root must be an object"),
        (json.dumps({"annotations": [], "categories": []}), "images"),
    ],
)
def test_loader_rejects_malformed_coco_json(tmp_path, contents, message):
    paths = write_coco_fixture(tmp_path)
    paths["instances"].write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        CocoInstanceSegmentationLoader(
            make_seg_spec(),
            dataset_path=str(tmp_path),
            image_dir=str(paths["images"]),
            label_path=str(paths["instances"]),
            target_hw=(8, 8),
        )


def test_loaders_reject_task_specific_annotation_schema(tmp_path):
    paths = write_coco_fixture(tmp_path)
    instances = json.loads(paths["instances"].read_text(encoding="utf-8"))
    instances["annotations"][0].pop("segmentation")
    paths["instances"].write_text(json.dumps(instances), encoding="utf-8")
    with pytest.raises(ValueError, match="segmentation"):
        CocoInstanceSegmentationLoader(
            make_seg_spec(),
            dataset_path=str(tmp_path),
            image_dir=str(paths["images"]),
            label_path=str(paths["instances"]),
            target_hw=(8, 8),
        )

    pose = json.loads(paths["pose"].read_text(encoding="utf-8"))
    pose["categories"][0]["keypoints"].pop()
    paths["pose"].write_text(json.dumps(pose), encoding="utf-8")
    with pytest.raises(ValueError, match="17 keypoint"):
        CocoPoseLoader(
            make_pose_spec(),
            dataset_path=str(tmp_path),
            image_dir=str(paths["images"]),
            label_path=str(paths["pose"]),
            target_hw=(8, 8),
        )


def test_loader_rejects_wrong_task_and_preprocessing_modes(tmp_path):
    paths = write_coco_fixture(tmp_path)
    with pytest.raises(ValueError, match="INSTANCE_SEGMENTATION"):
        CocoInstanceSegmentationLoader(
            make_pose_spec(),
            dataset_path=str(tmp_path),
            image_dir=str(paths["images"]),
            label_path=str(paths["instances"]),
        )
    with pytest.raises(ValueError, match="normalized"):
        CocoPoseLoader(
            make_pose_spec(),
            dataset_path=str(tmp_path),
            image_dir=str(paths["images"]),
            label_path=str(paths["pose"]),
            image_preprocess_mode="raw",
        )
    with pytest.raises(ValueError, match="letterbox"):
        CocoPoseLoader(
            make_pose_spec(),
            dataset_path=str(tmp_path),
            image_dir=str(paths["images"]),
            label_path=str(paths["pose"]),
            image_resize_mode="direct",
        )
