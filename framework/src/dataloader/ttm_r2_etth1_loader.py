"""Canonical ETTh1 loader for the fixed TTM-R2 workload."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from core.model_spec import Model_Spec
from ttm_r1.etth1_quality import ETTh1QualityConfig, load_etth1_windows
from ttm_r2.host_adapter import TTMR2HostAdapter
from ttm_r2.profile import (
    TTM_R2_CONTRACT_ID,
    TTM_R2_EXPECTED_WINDOWS,
    validate_dataset,
)

from .base import DataLoader


class TTMR2ETTh1Loader(DataLoader):
    """Stream 240 fixed univariate ETTh1 test windows to TTM-R2."""

    def __init__(self, model_spec: Model_Spec, **kwargs):
        self.model_spec = model_spec
        csv_path = kwargs.get("csv_path")
        if not csv_path:
            raise ValueError("TTMR2ETTh1Loader requires 'csv_path'.")
        self.csv_path = Path(csv_path)

        validate_dataset(self.csv_path)
        self._contexts, self._targets, self._split = load_etth1_windows(
            ETTh1QualityConfig(
                dataset_path=self.csv_path,
                windows=TTM_R2_EXPECTED_WINDOWS,
            )
        )
        self._adapter = TTMR2HostAdapter()
        self.total_samples = TTM_R2_EXPECTED_WINDOWS
        self._current_idx = 0

    def load_single(self) -> Dict[str, Any]:
        if self._current_idx >= self.total_samples:
            raise StopIteration("TTMR2ETTh1Loader exhausted all windows.")
        sample = self.load_by_index(self._current_idx)
        self._current_idx += 1
        return sample

    def load_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("TTMR2ETTh1Loader batch_size must be a positive integer.")
        batch: List[Dict[str, Any]] = []
        for _ in range(batch_size):
            if self._current_idx >= self.total_samples:
                break
            batch.append(self.load_by_index(self._current_idx))
            self._current_idx += 1
        return batch

    def load_by_index(self, index: int) -> Dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise IndexError("TTMR2ETTh1Loader index must be an integer.")
        if not 0 <= index < self.total_samples:
            raise IndexError(
                f"TTMR2ETTh1Loader index {index} out of range "
                f"[0, {self.total_samples})."
            )

        prepared = self._adapter.prepare(self._contexts[index : index + 1])
        return {
            "input": {
                "past_values": prepared.past_values.squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            },
            "label": {
                "future_values": self._targets[index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False),
                "loc": self._as_numpy(prepared.loc),
                "scale": self._as_numpy(prepared.scale),
                "model_loc": self._as_numpy(prepared.model_loc),
                "model_scale": self._as_numpy(prepared.model_scale),
            },
            "window_idx": index,
        }

    def get_labels(self) -> np.ndarray:
        return self._targets.detach().cpu().numpy().astype(np.float32, copy=False)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "split": "test",
            "split_start": self._split["test_start"],
            "split_end": self._split["test_start"] + self._split["test"],
            "total_samples": self.total_samples,
            "window_count": self.total_samples,
            "context_length": 512,
            "prediction_length": 96,
            "stride": 1,
            "feature_cols": ["OT"],
            "num_channels": 1,
            "contract_id": TTM_R2_CONTRACT_ID,
        }

    def preprocess(self, raw_input: Any) -> np.ndarray:
        value = torch.as_tensor(raw_input, dtype=torch.float32)
        if value.ndim == 2:
            value = value.unsqueeze(0)
        prepared = self._adapter.prepare(value)
        return self._as_numpy(prepared.past_values).squeeze(0)

    @staticmethod
    def _as_numpy(value: torch.Tensor) -> np.ndarray:
        return (
            value.squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
