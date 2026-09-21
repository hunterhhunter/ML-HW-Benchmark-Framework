"""Tensor-only R2 core using the fixed TTM-R1 lowering semantics."""

from __future__ import annotations

from pathlib import Path

import torch

from ttm_r1.core import (
    TTMR1Core,
    _add_missing_tied_weight_metadata,
    _load_ttm_model_class,
    _load_ttm_r1_checkpoint,
    _replace_ttm_r1_patchify,
    _replace_ttm_r1_scaler,
)

from .contracts import TTMR2Contract


class TTMR2Core(TTMR1Core):
    """Expose the fixed R2 forecast while reusing static scaler/patch lowering."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
        self.contract = TTMR2Contract.fixed()


def load_ttm_r2_model(model_path: str) -> torch.nn.Module:
    """Load local R2 main weights without Hub fallback and restore every tensor."""
    checkpoint = Path(model_path)
    if not checkpoint.is_dir() or not (checkpoint / "model.safetensors").is_file():
        raise ValueError(f"TTM-R2 requires a local checkpoint directory: {checkpoint}")
    model_class = _load_ttm_model_class()
    _add_missing_tied_weight_metadata(model_class)
    model = model_class.from_pretrained(str(checkpoint), local_files_only=True).eval()
    model.load_state_dict(_load_ttm_r1_checkpoint(str(checkpoint)), strict=True)
    model.requires_grad_(False)
    _validate_r2_config(model.config)
    return model


def lower_ttm_r2_model(model: torch.nn.Module) -> None:
    """Apply the static R2 scaler and patch lowering after configuration validation."""
    _validate_r2_config(getattr(model, "config", None))
    _replace_ttm_r1_scaler(model)
    _replace_ttm_r1_patchify(model)


def _validate_r2_config(config: object) -> None:
    expected = {
        "context_length": 512,
        "prediction_length": 96,
        "num_input_channels": 1,
        "patch_length": 64,
        "patch_stride": 64,
        "num_patches": 8,
        "scaling": "std",
    }
    actual = {name: getattr(config, name, None) for name in expected}
    if actual != expected:
        raise ValueError(f"TTM-R2 main config does not match 512-96-r2: {actual}")
