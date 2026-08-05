"""Local-only loading and structural validation for TimesFM 2.5."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch


def load_timesfm25_model(model_path: str) -> torch.nn.Module:
    """Load official local weights without an implicit Hub fallback."""
    checkpoint = Path(model_path)
    if not checkpoint.is_dir() or not (checkpoint / "model.safetensors").is_file():
        raise ValueError(f"TimesFM 2.5 local checkpoint directory is invalid: {checkpoint}")
    model_class = _timesfm_model_class()
    loaded = model_class.from_pretrained(str(checkpoint), local_files_only=True)
    if not isinstance(loaded, torch.nn.Module):
        raise ValueError("TimesFM 2.5 loader did not return a torch module")
    loaded.to(dtype=torch.float32).eval().requires_grad_(False)
    _validate_config(getattr(loaded, "config", None))
    return loaded


def _timesfm_model_class() -> Any:
    try:
        transformers = importlib.import_module("transformers")
        return getattr(transformers, "TimesFm2_5ModelForPrediction")
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "TimesFM 2.5 requires transformers with TimesFm2_5ModelForPrediction support"
        ) from exc


def _validate_config(config: object) -> None:
    expected = {
        "patch_length": 32,
        "horizon_length": 128,
        "num_hidden_layers": 20,
        "hidden_size": 1280,
    }
    actual = {name: getattr(config, name, None) for name in expected}
    if actual != expected:
        raise ValueError(f"TimesFM 2.5 checkpoint config does not match fixed contract: {actual}")
