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


def _validate_annotation_file(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid COCO annotation JSON: {path}: {exc}") from exc
    required = {"images", "annotations", "categories"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(
            f"COCO annotation JSON is missing keys {missing}: {path}"
        )


def validate_coco_vision_assets(dataset_root: Path) -> None:
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
    for annotation_file in annotation_files:
        _validate_annotation_file(annotation_file)


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
) -> None:
    """Download, safely extract, and validate official COCO validation data."""
    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    try:
        validate_coco_vision_assets(dataset_root)
        return
    except FileNotFoundError:
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

    validate_coco_vision_assets(dataset_root)
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
    prepare_coco_vision(args.dataset_root)


if __name__ == "__main__":
    main()
