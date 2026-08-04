"""Tensor-only TTM-R1 prediction wrapper for static device compilation."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from .contracts import TTMR1Contract


class StaticTTMR1Patchify(torch.nn.Module):
    """Exact [1,512,1] TTM-R1 patchification without ``aten::unfold``."""

    def forward(self, past_values: torch.Tensor) -> torch.Tensor:
        return past_values.reshape(1, 8, 64, 1).permute(0, 3, 1, 2).contiguous()


class TTMR1Core(torch.nn.Module):
    """Expose only TTM-R1's fixed forecast tensor, never a model container."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        if not isinstance(model, torch.nn.Module):
            raise ValueError("TTM-R1 model must be a torch Module")
        _replace_ttm_r1_patchify(model)
        self.model = model
        self.contract = TTMR1Contract.fixed()

    def forward(self, past_values: torch.Tensor) -> torch.Tensor:
        """Return one FP32 `[1,96,1]` forecast from a static TTM input."""
        self._validate("past_values", past_values, self.contract.core_inputs[0].shape)
        output = self.model(past_values=past_values, return_dict=True)
        forecast = self._extract_forecast(output)
        self._validate("forecast", forecast, self.contract.core_output.shape)
        return forecast

    @staticmethod
    def _extract_forecast(output: Any) -> torch.Tensor:
        if isinstance(output, dict):
            forecast = output.get("prediction_outputs")
        else:
            forecast = getattr(output, "prediction_outputs", None)
        if not isinstance(forecast, torch.Tensor):
            raise ValueError("TTM-R1 model output must expose prediction_outputs forecast")
        return forecast

    @staticmethod
    def _validate(name: str, tensor: Any, shape: tuple[int, ...]) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch Tensor")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must use float32")


def load_ttm_r1_model(model_path: str) -> torch.nn.Module:
    """Load local official weights without allowing a Hub fallback download."""
    TinyTimeMixerForPrediction = _load_ttm_model_class()
    _add_missing_tied_weight_metadata(TinyTimeMixerForPrediction)

    model = TinyTimeMixerForPrediction.from_pretrained(
        model_path,
        local_files_only=True,
    ).eval()
    model.requires_grad_(False)
    _validate_checkpoint_config(getattr(model, "config", None))
    return model


def _add_missing_tied_weight_metadata(model_class: Any) -> None:
    """Bridge the pre-5.0 IBM TTM class to the Transformers 5.x loader API."""
    if not hasattr(model_class, "all_tied_weights_keys"):
        model_class.all_tied_weights_keys = {}


def _replace_ttm_r1_patchify(model: torch.nn.Module) -> None:
    """Lower the R1 checkpoint's fixed non-overlapping patchify operation."""
    backbone = getattr(model, "backbone", None)
    patching = getattr(backbone, "patching", None)
    config = getattr(model, "config", None)
    if patching is None or config is None:
        return
    expected = {
        "context_length": 512,
        "patch_length": 64,
        "patch_stride": 64,
        "num_patches": 8,
        "num_input_channels": 1,
    }
    actual = {name: getattr(config, name, None) for name in expected}
    if actual != expected:
        raise ValueError(f"TTM-R1 patchify config does not match fixed ABI: {actual}")
    backbone.patching = StaticTTMR1Patchify()


def _load_ttm_model_class() -> Any:
    """Prefer an upstream Transformers implementation, then IBM's R1 package."""
    try:
        transformers = importlib.import_module("transformers")
        model_class = getattr(transformers, "TinyTimeMixerForPrediction", None)
        if model_class is not None:
            return model_class
    except ImportError:
        pass
    try:
        module = importlib.import_module("tsfm_public.models.tinytimemixer")
        return module.TinyTimeMixerForPrediction
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "TTM-R1 requires TinyTimeMixerForPrediction; install granite-tsfm==0.2.27 "
            "or a Transformers release that exports the class"
        ) from exc


def _validate_checkpoint_config(config: object) -> None:
    values = {
        "context_length": getattr(config, "context_length", None),
        "prediction_length": getattr(config, "prediction_length", None),
        "num_input_channels": getattr(config, "num_input_channels", None),
    }
    expected = {
        "context_length": 512,
        "prediction_length": 96,
        "num_input_channels": 1,
    }
    mismatches = {
        name: value for name, value in values.items() if value != expected[name]
    }
    if mismatches:
        raise ValueError(f"TTM-R1 checkpoint config does not match fixed ABI: {mismatches}")
