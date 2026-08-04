"""Tensor-only TTM-R1 prediction wrapper for static device compilation."""

from __future__ import annotations

from typing import Any

import torch

from .contracts import TTMR1Contract


class TTMR1Core(torch.nn.Module):
    """Expose only TTM-R1's fixed forecast tensor, never a model container."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        if not isinstance(model, torch.nn.Module):
            raise ValueError("TTM-R1 model must be a torch Module")
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
    try:
        from transformers import TinyTimeMixerForPrediction
    except ImportError as exc:
        raise ImportError(
            "TTM-R1 requires transformers with TinyTimeMixerForPrediction support"
        ) from exc

    model = TinyTimeMixerForPrediction.from_pretrained(
        model_path,
        local_files_only=True,
    ).eval()
    model.requires_grad_(False)
    _validate_checkpoint_config(getattr(model, "config", None))
    return model


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
