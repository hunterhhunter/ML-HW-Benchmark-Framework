"""Offline checks that a failed preparation cannot be mistaken for success."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest
from PIL import Image


SCRIPTS = Path(__file__).resolve().parents[1] / "datasets"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["prepare_imagenet_1k", "prepare_cifar10"])
def test_image_download_failure_propagates(name, tmp_path, monkeypatch):
    module = _load_script(name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "load_dataset", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))

    with pytest.raises(RuntimeError, match="offline"):
        module.main()

    root = tmp_path / "datasets" / ("imagenet_1k" if "imagenet" in name else "cifar10")
    assert not list(root.glob("*labels.txt"))


@pytest.mark.parametrize("name,call", [
    ("prepare_coco128", "main"),
    ("prepare_etth1", "download_etth1"),
    ("prepare_squad2", "prepare_squad2"),
])
def test_url_download_failure_propagates(name, call, tmp_path, monkeypatch):
    module = _load_script(name)
    monkeypatch.chdir(tmp_path)
    if name == "prepare_coco128":
        monkeypatch.setattr(module, "__file__", str(tmp_path / "datasets" / "prepare_coco128.py"))
    monkeypatch.setattr(module.urllib.request, "urlretrieve", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))

    with pytest.raises(RuntimeError, match="offline"):
        if name == "prepare_etth1":
            getattr(module, call)(str(tmp_path / "etth1"))
        else:
            getattr(module, call)()


@pytest.mark.parametrize("name", ["prepare_text_numpy", "prepare_squad_numpy"])
def test_numpy_source_failure_propagates(name, tmp_path, monkeypatch):
    module = _load_script(name)
    monkeypatch.setattr(sys, "argv", [f"{name}.py", "--output-dir", str(tmp_path / "output")])
    monkeypatch.setattr(module, "load_dataset", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))

    with pytest.raises(RuntimeError, match="offline"):
        module.main()


@pytest.mark.parametrize("name", ["prepare_imagenet_1k", "prepare_cifar10"])
def test_short_image_download_does_not_publish_labels(name, tmp_path, monkeypatch):
    module = _load_script(name)
    monkeypatch.chdir(tmp_path)
    items = [{"image": Image.new("RGB", (2, 2)), "label": 0} for _ in range(2)]
    monkeypatch.setattr(module, "load_dataset", lambda *a, **k: items)

    with pytest.raises(RuntimeError, match="Expected 3000"):
        module.main()

    root = tmp_path / "datasets" / ("imagenet_1k" if "imagenet" in name else "cifar10")
    assert not list(root.glob("*labels.txt"))


def test_coco128_short_archive_is_not_reported_ready(tmp_path, monkeypatch):
    module = _load_script("prepare_coco128")
    monkeypatch.setattr(module, "__file__", str(tmp_path / "datasets" / "prepare_coco128.py"))
    source = tmp_path / "source.zip"
    with ZipFile(source, "w") as archive:
        archive.writestr("coco128/images/train2017/one.jpg", b"jpg")
        archive.writestr("coco128/labels/train2017/one.txt", "0 0.5 0.5 0.2 0.2\n")
    monkeypatch.setattr(module.urllib.request, "urlretrieve", lambda url, dest, hook: shutil.copyfile(source, dest))

    with pytest.raises(ValueError, match="128"):
        module.main()


def test_coco128_rejects_all_missing_label_files(tmp_path):
    module = _load_script("prepare_coco128")
    images = tmp_path / "images" / "train2017"
    labels = tmp_path / "labels" / "train2017"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for index in range(128):
        (images / f"{index:012d}.jpg").write_bytes(b"image")
    with pytest.raises(ValueError, match="label"):
        module.validate_coco128(tmp_path)


def test_etth1_existing_empty_file_is_not_accepted(tmp_path, monkeypatch):
    module = _load_script("prepare_etth1")
    (tmp_path / "ETTh1.csv").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        module.urllib.request, "urlretrieve",
        lambda url, path: Path(path).write_text("bad", encoding="utf-8"),
    )
    with pytest.raises(ValueError, match="ETTh1|CSV"):
        module.download_etth1(str(tmp_path))
    assert (tmp_path / "ETTh1.csv").read_text(encoding="utf-8") == ""


def test_etth1_existing_invalid_file_is_redownloaded(tmp_path, monkeypatch):
    module = _load_script("prepare_etth1")
    dest = tmp_path / "output"
    dest.mkdir()
    (dest / "ETTh1.csv").write_text("", encoding="utf-8")
    source = tmp_path / "valid.csv"
    with source.open("w", encoding="utf-8") as stream:
        stream.write("date,OT\n")
        stream.writelines("2020-01-01,1\n" for _ in range(12116))
    monkeypatch.setattr(
        module.urllib.request, "urlretrieve",
        lambda url, path: shutil.copyfile(source, path),
    )

    module.download_etth1(str(dest))
    assert (dest / "ETTh1.csv").read_bytes() == source.read_bytes()


def test_squad_download_rejects_empty_json(tmp_path, monkeypatch):
    module = _load_script("prepare_squad2")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module.urllib.request, "urlretrieve", lambda url, dest: Path(dest).write_text(json.dumps({"data": []})))
    with pytest.raises(ValueError, match="SQuAD|question|QA"):
        module.prepare_squad2()
    assert not (tmp_path / "datasets/squad2/val.json").exists()


def test_squad_numpy_rejects_empty_dataset(tmp_path, monkeypatch):
    module = _load_script("prepare_squad_numpy")
    monkeypatch.setattr(sys, "argv", ["prepare_squad_numpy.py", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(module, "load_dataset", lambda *a, **k: [])
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    with pytest.raises(ValueError, match="zero|empty|sample"):
        module.main()


def test_text_numpy_rejects_empty_dataset(tmp_path, monkeypatch):
    module = _load_script("prepare_text_numpy")
    class EmptyDataset:
        def __len__(self): return 0
        def map(self, *a, **k): return self
        def set_format(self, **k): pass
        def __getitem__(self, key): return []
    monkeypatch.setattr(sys, "argv", ["prepare_text_numpy.py", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(module, "load_dataset", lambda *a, **k: EmptyDataset())
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    with pytest.raises(ValueError, match="zero|empty|sample"):
        module.main()
