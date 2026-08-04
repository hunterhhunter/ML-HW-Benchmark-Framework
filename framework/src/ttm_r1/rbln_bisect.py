"""Fixed-shape TTM-R1 graph stages for localizing RBLN compile failures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class ProbeStage:
    """One tensor-only model boundary with deterministic CPU input and output."""

    name: str
    module: torch.nn.Module
    inputs: tuple[torch.Tensor, ...]
    output: torch.Tensor


class _ScalerProbe(torch.nn.Module):
    def __init__(self, scaler: torch.nn.Module) -> None:
        super().__init__()
        self.scaler = scaler

    def forward(self, past_values: torch.Tensor) -> torch.Tensor:
        scaled, _, _ = self.scaler(past_values, torch.ones_like(past_values))
        return scaled


class _PatchifyProbe(torch.nn.Module):
    def __init__(self, patching: torch.nn.Module) -> None:
        super().__init__()
        self.patching = patching

    def forward(self, scaled_past_values: torch.Tensor) -> torch.Tensor:
        return self.patching(scaled_past_values)


class _EncoderProbe(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        output = self.encoder(patches, output_hidden_states=False, return_dict=True)
        return _get_tensor(output, "last_hidden_state")


class _DecoderProbe(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, hidden_state: torch.Tensor, patch_input: torch.Tensor) -> torch.Tensor:
        output, _ = self.decoder(
            hidden_state=hidden_state,
            patch_input=patch_input,
            output_hidden_states=False,
            static_categorical_values=None,
        )
        return output


class _HeadProbe(torch.nn.Module):
    def __init__(self, head: torch.nn.Module) -> None:
        super().__init__()
        self.head = head

    def forward(self, decoder_hidden: torch.Tensor, past_values: torch.Tensor) -> torch.Tensor:
        return self.head(decoder_hidden, past_values=past_values, future_values=None)


class _RestoreProbe(torch.nn.Module):
    def forward(self, forecast: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return forecast * scale + loc


def build_probe_stages(model: torch.nn.Module, past_values: torch.Tensor) -> tuple[ProbeStage, ...]:
    """Capture CPU tensors at every static R1 prediction boundary."""
    backbone = getattr(model, "backbone", None)
    scaler = getattr(backbone, "scaler", None)
    patching = getattr(backbone, "patching", None)
    encoder = getattr(backbone, "encoder", None)
    decoder = getattr(model, "decoder", None)
    head = getattr(model, "head", None)
    modules = (scaler, patching, encoder, decoder, head)
    if not all(isinstance(module, torch.nn.Module) for module in modules):
        raise ValueError("TTM-R1 probe requires backbone scaler, patching, encoder, decoder, and head")
    if tuple(past_values.shape) != (1, 512, 1) or past_values.dtype != torch.float32:
        raise ValueError("TTM-R1 probe requires float32 past_values with shape [1,512,1]")

    scaler_probe = _ScalerProbe(scaler)
    patchify_probe = _PatchifyProbe(patching)
    encoder_probe = _EncoderProbe(encoder)
    decoder_probe = _DecoderProbe(decoder)
    head_probe = _HeadProbe(head)
    restore_probe = _RestoreProbe()

    with torch.no_grad():
        scaled, loc, scale = scaler(past_values, torch.ones_like(past_values))
        patches = patching(scaled)
        hidden_state = _get_tensor(
            encoder(patches, output_hidden_states=False, return_dict=True), "last_hidden_state"
        )
        decoder_hidden, _ = decoder(
            hidden_state=hidden_state,
            patch_input=patches,
            output_hidden_states=False,
            static_categorical_values=None,
        )
        forecast = head(decoder_hidden, past_values=past_values, future_values=None)
        restored = restore_probe(forecast, loc, scale)

    return (
        ProbeStage("scaler", scaler_probe, (past_values,), scaled),
        ProbeStage("patchify", patchify_probe, (scaled,), patches),
        ProbeStage("encoder", encoder_probe, (patches,), hidden_state),
        ProbeStage("decoder", decoder_probe, (hidden_state, patches), decoder_hidden),
        ProbeStage("head", head_probe, (decoder_hidden, past_values), forecast),
        ProbeStage("restore", restore_probe, (forecast, loc, scale), restored),
    )


def compile_probe_stages(
    rebel: Any, stages: tuple[ProbeStage, ...], output_dir: str | Path
) -> list[dict[str, object]]:
    """Compile every stage independently so a later failure cannot hide an earlier success."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report = []
    for stage in stages:
        input_info = [
            (f"input_{index}", list(value.shape), "float32")
            for index, value in enumerate(stage.inputs)
        ]
        entry: dict[str, object] = {
            "name": stage.name,
            "inputs": [
                {"name": name, "shape": shape, "dtype": dtype}
                for name, shape, dtype in input_info
            ],
            "output": {"shape": list(stage.output.shape), "dtype": "float32"},
        }
        try:
            artifact = destination / f"{stage.name}.rbln"
            compiled = rebel.compile_from_torch(stage.module, input_info)
            compiled.save(str(artifact))
            entry.update({"status": "compiled", "artifact": str(artifact.resolve())})
        except Exception as error:
            entry.update(
                {
                    "status": "compile_failed",
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            )
        report.append(entry)
    return report


def _get_tensor(output: Any, name: str) -> torch.Tensor:
    value = output.get(name) if isinstance(output, dict) else getattr(output, name, None)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"TTM-R1 probe output must expose tensor {name}")
    return value
