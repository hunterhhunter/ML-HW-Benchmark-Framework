import importlib.util
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

import pytest

from core.model_spec import Task
from utils.dataset_resolver import resolve_dataset_paths


def _asset_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "prepare_coco_vision.py"
    )
    spec = importlib.util.spec_from_file_location(
        "prepare_coco_vision", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolver_selects_task_specific_official_annotations(tmp_path):
    image_dir = tmp_path / "images" / "val2017"
    annotation_dir = tmp_path / "annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    instances = annotation_dir / "instances_val2017.json"
    keypoints = annotation_dir / "person_keypoints_val2017.json"
    instances.write_text("{}", encoding="utf-8")
    keypoints.write_text("{}", encoding="utf-8")

    assert resolve_dataset_paths(
        Task.INSTANCE_SEGMENTATION, str(tmp_path), "", ""
    ) == (str(image_dir), str(instances))
    assert resolve_dataset_paths(
        Task.POSE_ESTIMATION, str(tmp_path), "", ""
    ) == (str(image_dir), str(keypoints))


def test_safe_extract_rejects_archive_path_escape(tmp_path):
    module = _asset_module()
    archive_path = tmp_path / "bad.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "bad")

    with ZipFile(archive_path) as archive, pytest.raises(
        ValueError, match="outside"
    ):
        module._safe_extract(archive, tmp_path / "dataset")


def test_asset_validation_names_every_missing_path(tmp_path):
    module = _asset_module()

    with pytest.raises(FileNotFoundError) as raised:
        module.validate_coco_vision_assets(tmp_path)

    message = str(raised.value)
    assert "images/val2017" in message
    assert "instances_val2017.json" in message
    assert "person_keypoints_val2017.json" in message


def test_prepare_extracts_valid_archives_and_second_run_needs_no_download(
    tmp_path,
):
    module = _asset_module()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    image_archive = source_dir / "val2017.zip"
    annotation_archive = source_dir / "annotations_trainval2017.zip"
    with ZipFile(image_archive, "w") as archive:
        archive.writestr("val2017/000000000001.jpg", b"jpeg")
    with ZipFile(annotation_archive, "w") as archive:
        archive.writestr(
            "annotations/instances_val2017.json",
            json.dumps({"images": [], "annotations": [], "categories": []}),
        )
        archive.writestr(
            "annotations/person_keypoints_val2017.json",
            json.dumps({"images": [], "annotations": [], "categories": []}),
        )

    sources = {
        "val2017.zip": image_archive,
        "annotations_trainval2017.zip": annotation_archive,
    }

    def local_download(url, destination):
        shutil.copyfile(sources[url.rsplit("/", 1)[-1]], destination)

    dataset_root = tmp_path / "coco"
    module.prepare_coco_vision(dataset_root, download=local_download)
    assert (dataset_root / "images" / "val2017" / "000000000001.jpg").exists()
    assert (dataset_root / "annotations" / "instances_val2017.json").exists()
    assert (
        dataset_root / "annotations" / "person_keypoints_val2017.json"
    ).exists()

    shutil.rmtree(source_dir)
    module.prepare_coco_vision(
        dataset_root,
        download=lambda url, destination: pytest.fail(
            f"complete dataset attempted another download: {url}"
        ),
    )
