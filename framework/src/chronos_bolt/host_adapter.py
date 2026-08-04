"""CPU-only Chronos-Bolt preprocessing around the fixed Transformer ABI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .contracts import ChronosBoltContract


@dataclass(frozen=True)
class PreparedChronosBoltInputs:
    """NPU-core inputs plus the normalization state needed for final forecasts."""

    input_embeds: torch.Tensor
    attention_mask: torch.Tensor
    decoder_input_embeds: torch.Tensor
    loc: torch.Tensor
    scale: torch.Tensor
    use_arcsinh: bool

    def restore(self, normalized_quantiles: torch.Tensor) -> torch.Tensor:
        """Apply the original Chronos InstanceNorm inverse in FP32."""
        if not isinstance(normalized_quantiles, torch.Tensor):
            raise ValueError("normalized quantiles must be a torch Tensor")
        if normalized_quantiles.shape != (1, 9, 64):
            raise ValueError(
                "normalized quantiles must have fixed shape (1, 9, 64), "
                f"got {tuple(normalized_quantiles.shape)}"
            )
        if normalized_quantiles.dtype != torch.float32:
            raise ValueError("normalized quantiles must use float32")
        restored = normalized_quantiles
        if self.use_arcsinh:
            restored = torch.sinh(restored)
        return restored * self.scale + self.loc


class ChronosBoltHostAdapter:
    """Preserve dynamic Chronos semantics while emitting one static core ABI."""

    _CONTEXT_LENGTH = 512
    _PATCH_SIZE = 16
    _PATCH_COUNT = 32

    def __init__(self, model: Any) -> None:
        self.model = model
        chronos_config = getattr(model, "chronos_config", None)
        config = getattr(model, "config", None)
        if chronos_config is None or config is None:
            raise ValueError("Chronos-Bolt model must expose config and chronos_config")
        if getattr(chronos_config, "context_length", None) != self._CONTEXT_LENGTH:
            raise ValueError("Chronos-Bolt core requires context_length=512")
        if getattr(chronos_config, "input_patch_size", None) != self._PATCH_SIZE:
            raise ValueError("Chronos-Bolt core requires input_patch_size=16")
        if getattr(chronos_config, "input_patch_stride", None) != self._PATCH_SIZE:
            raise ValueError("Chronos-Bolt core requires input_patch_stride=16")
        if getattr(chronos_config, "use_reg_token", False):
            raise ValueError("Chronos-Bolt core does not support use_reg_token")
        d_model = getattr(config, "d_model", None)
        self.contract = ChronosBoltContract.tiny(d_model=d_model)
        self._eps = float(getattr(getattr(model, "instance_norm", None), "eps", 1e-5))
        self._use_arcsinh = bool(
            getattr(getattr(model, "instance_norm", None), "use_arcsinh", False)
        )

    def prepare(
        self, context: torch.Tensor, observed_mask: torch.Tensor | None = None
    ) -> PreparedChronosBoltInputs:
        """Create static embeddings and masks without adding NPU-incompatible ops."""
        self._validate_context(context)
        context, observed_mask = self._crop_and_left_pad(context, observed_mask)
        loc, scale = self._loc_scale_from_nan_pattern(context)
        normalized = (context - loc) / scale
        if self._use_arcsinh:
            normalized = torch.arcsinh(normalized)

        if observed_mask is None:
            observed_mask = torch.logical_not(torch.isnan(context))
        else:
            observed_mask = observed_mask.to(dtype=torch.bool)
        values, masks = self._fixed_patches(normalized, observed_mask)
        values = torch.where(masks, values, torch.zeros_like(values))
        mask_values = masks.to(dtype=torch.float32)
        patch_input = torch.cat((values, mask_values), dim=-1)
        attention_mask = masks.any(dim=-1).to(dtype=torch.float32)
        input_embeds = self.model.input_patch_embedding(patch_input)
        decoder_input_embeds = self._decoder_start_embedding(context.device)

        self._validate_core_tensor("input_embeds", input_embeds)
        self._validate_core_tensor("attention_mask", attention_mask)
        self._validate_core_tensor("decoder_input_embeds", decoder_input_embeds)
        return PreparedChronosBoltInputs(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            decoder_input_embeds=decoder_input_embeds,
            loc=loc,
            scale=scale,
            use_arcsinh=self._use_arcsinh,
        )

    def _validate_context(self, context: torch.Tensor) -> None:
        if not isinstance(context, torch.Tensor):
            raise ValueError("context must be a torch Tensor")
        if context.ndim != 2 or context.shape[0] != 1 or context.shape[1] <= 0:
            raise ValueError("context must have shape (1, T) with positive T")
        if context.dtype != torch.float32:
            raise ValueError("context must use float32")

    def _crop_and_left_pad(
        self, context: torch.Tensor, observed_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if observed_mask is not None:
            if not isinstance(observed_mask, torch.Tensor):
                raise ValueError("observed_mask must be a torch Tensor")
            if observed_mask.shape != context.shape:
                raise ValueError("observed_mask shape must match context")
            if observed_mask.dtype != torch.bool:
                raise ValueError("observed_mask must use bool")
        if context.shape[-1] > self._CONTEXT_LENGTH:
            context = context[:, -self._CONTEXT_LENGTH :]
            if observed_mask is not None:
                observed_mask = observed_mask[:, -self._CONTEXT_LENGTH :]
        missing = self._CONTEXT_LENGTH - context.shape[-1]
        if missing == 0:
            return context, observed_mask
        padding = torch.full((1, missing), torch.nan, dtype=torch.float32, device=context.device)
        context = torch.cat((padding, context), dim=-1)
        if observed_mask is not None:
            observed_mask = torch.cat(
                (
                    torch.zeros((1, missing), dtype=torch.bool, device=context.device),
                    observed_mask,
                ),
                dim=-1,
            )
        return context, observed_mask

    def _loc_scale_from_nan_pattern(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Match Chronos InstanceNorm without `torch.nanmean` in the static path."""
        finite = torch.isfinite(context)
        count = finite.sum(dim=-1, keepdim=True)
        safe_context = torch.where(finite, context, torch.zeros_like(context))
        denominator = count.clamp_min(1).to(dtype=torch.float32)
        loc = safe_context.sum(dim=-1, keepdim=True) / denominator
        loc = torch.where(count > 0, loc, torch.zeros_like(loc))

        centered = torch.where(finite, safe_context - loc, torch.zeros_like(safe_context))
        variance = centered.square().sum(dim=-1, keepdim=True) / denominator
        scale = variance.sqrt()
        scale = torch.where(count > 0, scale, torch.ones_like(scale))
        scale = torch.where(scale == 0, torch.full_like(scale, self._eps), scale)
        return loc, scale

    def _fixed_patches(
        self, values: torch.Tensor, observed_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        offsets = range(0, self._CONTEXT_LENGTH, self._PATCH_SIZE)
        value_patches = torch.stack(
            [values[:, offset : offset + self._PATCH_SIZE] for offset in offsets], dim=1
        )
        mask_patches = torch.stack(
            [observed_mask[:, offset : offset + self._PATCH_SIZE] for offset in offsets], dim=1
        )
        return value_patches, mask_patches

    def _decoder_start_embedding(self, device: torch.device) -> torch.Tensor:
        token_id = getattr(self.model.config, "decoder_start_token_id", None)
        if type(token_id) is not int:
            raise ValueError("Chronos-Bolt decoder_start_token_id must be an integer")
        token_ids = torch.full((1, 1), token_id, dtype=torch.long, device=device)
        return self.model.shared(token_ids)

    def _validate_core_tensor(self, name: str, tensor: torch.Tensor) -> None:
        expected = next(item for item in self.contract.core_inputs if item.name == name)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch Tensor")
        if tuple(tensor.shape) != expected.shape:
            raise ValueError(f"{name} must have shape {expected.shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must use float32")
