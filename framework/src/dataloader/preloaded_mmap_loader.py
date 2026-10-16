"""Materialize static NumPy inputs in one mmap before async measurement."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .base import DataLoader


class PreloadedMmapDataLoader(DataLoader):
    def __init__(self, inner: DataLoader, *, sample_count: int,
                 directory: str | Path, progress_interval: int = 0):
        if type(sample_count) is not int:
            raise TypeError("sample_count must be an integer")
        total = inner.get_metadata().get("total_samples")
        if type(total) is not int or not 1 <= sample_count <= total:
            raise ValueError(f"sample_count must be in [1, {total}], got {sample_count}")
        self.inner = inner
        self.total_samples = sample_count
        self.current_idx = 0
        self._samples: list[dict[str, Any]] = []
        self._inputs = None
        self._closed = False
        self.cache_path: Path | None = None
        cache_directory = Path(directory).resolve()
        cache_directory.mkdir(parents=True, exist_ok=True)
        try:
            first = self._validated_sample(inner.load_by_index(0), 0)
            first_input = first["input"]
            required_bytes = sample_count * first_input.nbytes + 4096
            if shutil.disk_usage(cache_directory).free < required_bytes:
                raise OSError(f"insufficient disk space for async preload: need {required_bytes} bytes")
            fd, name = tempfile.mkstemp(prefix="mlhw-preloaded-", suffix=".npy",
                                         dir=cache_directory)
            os.close(fd)
            self.cache_path = Path(name)
            self.cache_path.unlink()
            self._inputs = np.lib.format.open_memmap(self.cache_path, mode="w+",
                dtype=first_input.dtype, shape=(sample_count, *first_input.shape))
            self._store_sample(0, first)
            for index in range(1, sample_count):
                sample = self._validated_sample(inner.load_by_index(index), index)
                value = sample["input"]
                if value.shape != first_input.shape or value.dtype != first_input.dtype:
                    raise ValueError("preloaded async inputs must have one static shape and dtype")
                self._store_sample(index, sample)
                if progress_interval and (index + 1) % progress_interval == 0:
                    print(f"[AsyncPreload] materialized {index + 1}/{sample_count}", flush=True)
            self._inputs.flush()
            atexit.register(self.close)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _validated_sample(sample: Any, index: int) -> dict[str, Any]:
        if not isinstance(sample, dict) or "input" not in sample:
            raise TypeError(f"preloaded async sample {index} must contain input")
        value = sample["input"]
        if not isinstance(value, np.ndarray) or value.ndim < 1 or not np.issubdtype(value.dtype, np.number):
            raise TypeError("preloaded async input must be one numeric NumPy tensor")
        if value.dtype.hasobject:
            raise TypeError("preloaded async input cannot have object dtype")
        return sample

    def _store_sample(self, index: int, sample: dict[str, Any]) -> None:
        self._inputs[index] = np.ascontiguousarray(sample["input"])
        metadata = dict(sample)
        metadata.pop("input")
        self._samples.append(metadata)

    def load_by_index(self, index: int) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("preloaded mmap loader is closed")
        if index < 0 or index >= self.total_samples:
            raise IndexError(index)
        sample = dict(self._samples[index])
        sample["input"] = np.asarray(self._inputs[index])
        return sample

    def load_single(self) -> dict[str, Any]:
        if self.current_idx >= self.total_samples:
            raise StopIteration("all preloaded samples consumed")
        sample = self.load_by_index(self.current_idx)
        self.current_idx += 1
        return sample

    def load_batch(self, batch_size: int) -> list[dict[str, Any]]:
        if batch_size < 0:
            raise ValueError("batch_size cannot be negative")
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(self.load_single())
            except StopIteration:
                break
        return batch

    def get_labels(self) -> Any:
        return self.inner.get_labels()

    def get_metadata(self) -> dict[str, Any]:
        metadata = dict(self.inner.get_metadata())
        metadata.update(total_samples=self.total_samples,
                        preloaded_mmap_sample_count=self.total_samples,
                        preloaded_mmap_input_shape=list(self._inputs.shape[1:]),
                        preloaded_mmap_input_dtype=self._inputs.dtype.name)
        return metadata

    def preprocess(self, raw_input: Any):
        return self.inner.preprocess(raw_input)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        inputs, self._inputs = self._inputs, None
        if inputs is not None:
            try:
                inputs.flush()
            finally:
                if getattr(inputs, "_mmap", None) is not None:
                    inputs._mmap.close()
        if self.cache_path is not None:
            self.cache_path.unlink(missing_ok=True)
        atexit.unregister(self.close)
