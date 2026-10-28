"""Prepare official COCO 2017 validation assets for segmentation and pose."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
from typing import Callable
import urllib.request
import zipfile


COCO_DOWNLOADS = {
    "val2017.zip": (
        "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip"
    ),
    "annotations_trainval2017.zip": (
        "https://s3.amazonaws.com/images.cocodataset.org/annotations/"
        "annotations_trainval2017.zip"
    ),
}


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract an archive only when every member remains below destination."""
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if target != root and root not in target.parents:
            raise ValueError(
                f"archive member extracts outside dataset root: {member.filename}"
            )
        mode = member.external_attr >> 16
        if stat.S_IFMT(mode) == stat.S_IFLNK:
            raise ValueError(f"archive symlink is not allowed: {member.filename}")
    archive.extractall(destination)


def _validate_annotation_file(path: Path, image_dir: Path) -> set[str]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid COCO annotation JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"COCO annotation root must be an object: {path}")
    required = {"images", "annotations", "categories"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(
            f"COCO annotation JSON is missing keys {missing}: {path}"
        )
    if not all(isinstance(payload[key], list) for key in required):
        raise ValueError(f"COCO annotation fields must be lists: {path}")
    if not payload["images"] or not payload["categories"]:
        raise ValueError(f"COCO annotation contains no images or categories: {path}")
    image_root = image_dir.resolve()
    names: set[str] = set()
    ids: set[int] = set()
    for record in payload["images"]:
        if not isinstance(record, dict) or type(record.get("id")) is not int:
            raise ValueError(f"invalid COCO image record: {path}")
        filename = record.get("file_name")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"COCO image record has no file_name: {path}")
        image_path = (image_dir / filename).resolve()
        if image_path == image_root or image_root not in image_path.parents or not image_path.is_file():
            raise ValueError(f"COCO image reference is missing or outside val2017: {filename}")
        if filename in names or record["id"] in ids:
            raise ValueError(f"duplicate COCO image record: {filename}")
        names.add(filename)
        ids.add(record["id"])
    return names


def validate_coco_vision_assets(dataset_root: Path, *, expected_images: int | None = None) -> None:
    """Validate paths and JSON structure required by both vision tasks."""
    dataset_root = Path(dataset_root)
    image_dir = dataset_root / "images" / "val2017"
    annotation_files = [
        dataset_root / "annotations" / "instances_val2017.json",
        dataset_root / "annotations" / "person_keypoints_val2017.json",
    ]
    required = [image_dir, *annotation_files]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing COCO vision assets: " + ", ".join(missing)
        )
    if not image_dir.is_dir():
        raise ValueError(f"COCO val2017 image path is not a directory: {image_dir}")
    annotation_images = [
        _validate_annotation_file(annotation_file, image_dir)
        for annotation_file in annotation_files
    ]
    if annotation_images[0] != annotation_images[1]:
        raise ValueError("COCO task annotations disagree on val2017 images")
    actual_images = {path.name for path in image_dir.iterdir() if path.is_file()}
    if annotation_images[0] != actual_images:
        raise ValueError("COCO val2017 image directory does not match annotations")
    if expected_images is not None and len(actual_images) != expected_images:
        raise ValueError(f"COCO val2017 expected {expected_images} images, found {len(actual_images)}")


def _download_archive(
    url: str,
    archive_path: Path,
    download: Callable[[str, str], object],
) -> None:
    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    try:
        download(url, str(partial_path))
        os.replace(partial_path, archive_path)
    except BaseException:
        try:
            partial_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def prepare_coco_vision(
    dataset_root: Path,
    download: Callable[[str, str], object] = urllib.request.urlretrieve,
    *,
    expected_images: int | None = None,
) -> None:
    """Download, safely extract, and validate official COCO validation data."""
    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    try:
        validate_coco_vision_assets(dataset_root, expected_images=expected_images)
        return
    except (FileNotFoundError, ValueError):
        pass

    downloads = [
        (
            "val2017.zip",
            COCO_DOWNLOADS["val2017.zip"],
            dataset_root / "images",
        ),
        (
            "annotations_trainval2017.zip",
            COCO_DOWNLOADS["annotations_trainval2017.zip"],
            dataset_root,
        ),
    ]
    for filename, url, extract_root in downloads:
        archive_path = dataset_root / filename
        if not archive_path.exists():
            print(f"[*] Downloading {url} -> {archive_path}")
            _download_archive(url, archive_path, download)
        extract_root.mkdir(parents=True, exist_ok=True)
        print(f"[*] Extracting {archive_path} -> {extract_root}")
        try:
            with zipfile.ZipFile(archive_path) as archive:
                _safe_extract(archive, extract_root)
        except zipfile.BadZipFile as exc:
            raise ValueError(f"invalid COCO archive: {archive_path}") from exc

    validate_coco_vision_assets(dataset_root, expected_images=expected_images)
    print(f"[+] Official COCO vision dataset is ready: {dataset_root}")


def _parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent / "coco"
    parser = argparse.ArgumentParser(
        description="Prepare official COCO val2017 images and annotations"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_root,
        help=f"output dataset root (default: {default_root})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prepare_coco_vision(args.dataset_root, expected_images=5000)


if __name__ == "__main__":
    main()
