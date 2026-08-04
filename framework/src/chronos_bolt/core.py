"""Tensor-only T5 encoder/decoder core for Chronos-Bolt compilation."""

from __future__ import annotations

from typing import Any

import torch

from .contracts import ChronosBoltContract


def _config_value(config: object, name: str) -> Any:
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


class ChronosBoltTransformerCore(torch.nn.Module):
    """Expose only the learned Transformer and quantile head as tensors."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        config = getattr(model, "config", None)
        chronos_config = getattr(model, "chronos_config", None)
        d_model = _config_value(config, "d_model")
        prediction_length = _config_value(chronos_config, "prediction_length")
        quantiles = _config_value(chronos_config, "quantiles")
        use_reg_token = _config_value(chronos_config, "use_reg_token")
        if type(d_model) is not int or d_model <= 0:
            raise ValueError("Chronos-Bolt model config must contain a positive d_model")
        if prediction_length != 64:
            raise ValueError("Chronos-Bolt core requires prediction_length=64")
        if not isinstance(quantiles, (list, tuple)) or len(quantiles) != 9:
            raise ValueError("Chronos-Bolt core requires exactly nine quantiles")
        if type(use_reg_token) is not bool:
            raise ValueError("Chronos-Bolt core requires a bool use_reg_token")
        for attribute in ("encoder", "decoder", "output_patch_embedding"):
            if not isinstance(getattr(model, attribute, None), torch.nn.Module):
                raise ValueError(f"Chronos-Bolt model has no {attribute} module")
        self.model = model
        self.contract = ChronosBoltContract.tiny(
            d_model=d_model,
            use_reg_token=use_reg_token,
        )

    def forward(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized FP32 quantiles without a Hugging Face container."""
        self._validate("input_embeds", input_embeds)
        self._validate("attention_mask", attention_mask)
        self._validate("decoder_input_embeds", decoder_input_embeds)

        encoder_outputs = self.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
            return_dict=False,
        )
        encoder_hidden_states = encoder_outputs[0]
        decoder_outputs = self.model.decoder(
            inputs_embeds=decoder_input_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            return_dict=False,
            use_cache=False,
        )
        sequence_output = decoder_outputs[0]
        raw_quantiles = self.model.output_patch_embedding(sequence_output)
        quantiles = raw_quantiles.reshape(self.contract.core_output.shape)
        if quantiles.dtype != torch.float32:
            raise ValueError("Chronos-Bolt core output must use float32")
        return quantiles

    def _validate(self, name: str, tensor: torch.Tensor) -> None:
        expected = next(item for item in self.contract.core_inputs if item.name == name)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch Tensor")
        if tuple(tensor.shape) != expected.shape:
            raise ValueError(f"{name} must have shape {expected.shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must use float32")


def load_chronos_bolt_model(model_path: str) -> torch.nn.Module:
    """Load a local official checkpoint without a Hub download fallback."""
    from chronos.chronos_bolt import ChronosBoltPipeline

    pipeline = ChronosBoltPipeline.from_pretrained(model_path)
    model = pipeline.model.eval()
    model.requires_grad_(False)
    return model
