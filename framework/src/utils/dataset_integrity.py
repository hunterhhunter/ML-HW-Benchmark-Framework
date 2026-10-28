"""Validate the assets produced by built-in dataset preparation scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _image_files(directory: Path) -> set[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory is missing: {directory}")
    entries = list(directory.iterdir())
    invalid = [entry.name for entry in entries if not entry.is_file() or entry.suffix.lower() not in _IMAGE_SUFFIXES]
    if invalid:
        raise ValueError(f"unexpected entries in image directory {directory}: {invalid[:3]}")
    return {entry.name for entry in entries}


def _validate_labeled_images(root: Path, image_subdir: str, label_name: str, count: int) -> None:
    names = _image_files(root / image_subdir)
    label_path = root / label_name
    if not label_path.is_file():
        raise FileNotFoundError(f"image labels are missing: {label_path}")
    labels = {}
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2 or parts[0] in labels:
            raise ValueError(f"invalid or duplicate image label in {label_path}: {line!r}")
        labels[parts[0]] = int(parts[1])
    if len(names) != count or set(labels) != names:
        raise ValueError(
            f"image/label sample count or names mismatch under {root}: "
            f"expected {count}, images {len(names)}, labels {len(labels)}"
        )


def _validate_coco128(root: Path) -> None:
    names = _image_files(root / "images" / "train2017")
    label_dir = root / "labels" / "train2017"
    if not label_dir.is_dir():
        raise FileNotFoundError(f"COCO128 label directory is missing: {label_dir}")
    if len(names) != 128:
        raise ValueError(f"COCO128 expected 128 image samples, found {len(names)}")
    image_stems = {Path(name).stem for name in names}
    labels = list(label_dir.glob("*.txt"))
    if not any(path.stat().st_size > 0 for path in labels):
        raise ValueError(f"COCO128 has no nonempty label files under {label_dir}")
    if any(path.stem not in image_stems for path in labels):
        raise ValueError(f"COCO128 has label files without images under {label_dir}")


def _validate_coco_vision(root: Path) -> None:
    names = _image_files(root / "images" / "val2017")
    if len(names) != 5000:
        raise ValueError(f"COCO val2017 expected 5000 image samples, found {len(names)}")
    for annotation_name in ("instances_val2017.json", "person_keypoints_val2017.json"):
        path = root / "annotations" / annotation_name
        if not path.is_file():
            raise FileNotFoundError(f"COCO annotation file is missing: {path}")
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        image_records = payload.get("images") if isinstance(payload, dict) else None
        if not isinstance(image_records, list) or {item.get("file_name") for item in image_records if isinstance(item, dict)} != names:
            raise ValueError(f"COCO annotation image list does not match val2017: {path}")


def _validate_squad(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"SQuAD JSON is missing: {path}")
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    articles = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(articles, list) or not any(
        isinstance(article, dict)
        and any(
            isinstance(paragraph, dict) and paragraph.get("qas")
            for paragraph in article.get("paragraphs", [])
        )
        for article in articles
    ):
        raise ValueError(f"SQuAD JSON contains no QA questions: {path}")


def _validate_numpy(root: Path, *, qa: bool) -> None:
    names = ["input_ids", "attention_mask"]
    if qa:
        names += ["token_type_ids", "start_positions", "end_positions"]
    else:
        names += ["labels"]
    arrays = {}
    for name in names:
        path = root / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"NumPy dataset array is missing: {path}")
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    validate_numpy_arrays(arrays)


def validate_numpy_arrays(arrays: dict[str, np.ndarray]) -> None:
    """Check a baked NLP dataset before a loader exposes any sample."""
    input_ids = arrays["input_ids"]
    if input_ids.ndim != 2 or 0 in input_ids.shape:
        raise ValueError("NumPy dataset sample length, shape, or dtype mismatch")
    count, width = input_ids.shape
    for name, array in arrays.items():
        expected_ndim = 2 if name in {"input_ids", "attention_mask", "token_type_ids"} else 1
        if (
            array.ndim != expected_ndim
            or array.shape[0] != count
            or (expected_ndim == 2 and array.shape[1] != width)
            or array.dtype.kind not in "iu"
        ):
            raise ValueError("NumPy dataset sample length, shape, or dtype mismatch")


def _validate_etth1(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"ETTh1 CSV is missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        if "date" not in header or "OT" not in header:
            raise ValueError(f"ETTh1 CSV is missing required columns: {path}")
        count = sum(1 for _ in reader)
    if count < 12116:
        raise ValueError(f"ETTh1 CSV has too few samples: {count}")


def validate_prepared_dataset(script: str, dataset_path: str) -> None:
    """Raise if a built-in script's default output is incomplete."""
    path = Path(dataset_path)
    name = Path(script).name
    if name == "prepare_imagenet_1k.py":
        _validate_labeled_images(path, "val", "val_labels.txt", 3000)
    elif name == "prepare_cifar10.py":
        _validate_labeled_images(path, "test", "test_labels.txt", 3000)
    elif name == "prepare_coco128.py":
        _validate_coco128(path)
    elif name == "prepare_coco_vision.py":
        _validate_coco_vision(path)
    elif name == "prepare_squad2.py":
        _validate_squad(path)
    elif name == "prepare_text_numpy.py":
        _validate_numpy(path, qa=False)
    elif name == "prepare_squad_numpy.py":
        _validate_numpy(path, qa=True)
    elif name == "prepare_etth1.py":
        _validate_etth1(path)
    elif not path.exists():
        raise FileNotFoundError(f"dataset path is missing: {path}")
