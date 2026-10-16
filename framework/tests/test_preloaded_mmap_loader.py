import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from dataloader.preloaded_mmap_loader import PreloadedMmapDataLoader


class Samples:
    def __init__(self, values):
        self.values = values

    def get_metadata(self):
        return {"total_samples": len(self.values), "task": "POSE_ESTIMATION"}

    def get_labels(self):
        return [item["label"] for item in self.values]

    def load_by_index(self, index):
        return self.values[index]

    def preprocess(self, raw):
        return raw


def _samples():
    return Samples([{"input": np.full((2, 3), i, dtype=np.uint8),
                     "label": {"image_id": i + 10},
                     "preprocess_context": {"pad_y": i}} for i in range(3)])


def test_preloaded_loader_preserves_inputs_and_labels(tmp_path):
    inner = _samples()
    loader = PreloadedMmapDataLoader(inner, sample_count=3, directory=tmp_path)
    assert loader.get_metadata()["preloaded_mmap_input_shape"] == [2, 3]
    assert loader.get_metadata()["preloaded_mmap_input_dtype"] == "uint8"
    for i in range(3):
        sample = loader.load_by_index(i)
        np.testing.assert_array_equal(sample["input"], inner.values[i]["input"])
        assert sample["label"] == {"image_id": i + 10}
        assert sample["preprocess_context"] == {"pad_y": i}
    assert loader.get_labels() == inner.get_labels()
    assert [loader.load_single()["label"]["image_id"] for _ in range(3)] == [10, 11, 12]
    loader.close()


def test_preloaded_loader_rejects_invalid_count_and_tensor_contract(tmp_path):
    for count in (0, 4):
        with pytest.raises(ValueError):
            PreloadedMmapDataLoader(_samples(), sample_count=count, directory=tmp_path)
    for bad in (np.ones((3, 3), dtype=np.uint8), np.ones((2, 3), dtype=np.float32),
                {"x": np.ones(1)}, np.array([object()], dtype=object)):
        inner = _samples()
        inner.values[1]["input"] = bad
        with pytest.raises((TypeError, ValueError)):
            PreloadedMmapDataLoader(inner, sample_count=3, directory=tmp_path)


def test_preloaded_loader_cleans_tempfile_on_error_and_close(tmp_path):
    inner = _samples()
    inner.values[2]["input"] = np.ones((3, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        PreloadedMmapDataLoader(inner, sample_count=3, directory=tmp_path)
    assert list(tmp_path.iterdir()) == []
    loader = PreloadedMmapDataLoader(_samples(), sample_count=3, directory=tmp_path)
    assert loader.cache_path.exists()
    loader.close()
    loader.close()
    assert list(tmp_path.iterdir()) == []


def test_preloaded_loader_rejects_insufficient_disk_space(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(OSError, match="space"):
        PreloadedMmapDataLoader(_samples(), sample_count=3, directory=tmp_path)
    assert list(tmp_path.iterdir()) == []
