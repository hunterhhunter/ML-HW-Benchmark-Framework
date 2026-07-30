"""Compile the fixed PatchTST ETTh1 contract without ``aten::unfold``."""

import json
from collections.abc import Sequence
from dataclasses import replace

from tools.rbln_compile_recipes.common import (
    RecipeContract,
    TensorContract,
    create_parser,
    emit_description_or_require_output,
    save_and_validate,
)


DEFAULT_MODEL_ID = "ibm-granite/granite-timeseries-patchtst"
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 96
NUM_INPUT_CHANNELS = 7
PATCH_LENGTH = 12
PATCH_STRIDE = 12
SEQUENCE_START = 8
NUM_PATCHES = 42

CONTRACT = RecipeContract(
    recipe="patchtst_etth1",
    model_id=DEFAULT_MODEL_ID,
    inputs=(
        TensorContract(
            "past_values",
            (1, CONTEXT_LENGTH, NUM_INPUT_CHANNELS),
            "float32",
        ),
        TensorContract(
            "past_observed_mask",
            (1, CONTEXT_LENGTH, NUM_INPUT_CHANNELS),
            "bool",
        ),
    ),
    outputs=(
        TensorContract(
            "prediction_outputs",
            (1, PREDICTION_LENGTH, NUM_INPUT_CHANNELS),
            "float32",
        ),
    ),
    allow_unnamed_outputs=True,
    notes=(
        "fixed static patchification: 42 patches shaped (1,42,12,7)",
        "replace PatchTST patchifier to remove unsupported aten::unfold",
        "preserve bool mask ABI and cast to past_values dtype before model math",
        "require original/static and bool/float CPU equivalence before compilation",
        "model_trace_method=jittrace",
    ),
)


def static_patchify(past_values):
    """Build the 42 fixed ETTh1 patches without using ``Tensor.unfold``."""
    import torch

    trimmed = past_values[:, SEQUENCE_START:, :]
    return torch.stack(
        [
            trimmed[:, offset : offset + PATCH_LENGTH, :]
            for offset in range(0, NUM_PATCHES * PATCH_STRIDE, PATCH_STRIDE)
        ],
        dim=1,
    )


def build_static_patchifier(torch_module):
    """Return the channel-first adapter expected by the PatchTST encoder."""

    class StaticPatchifier(torch_module.nn.Module):
        def forward(self, past_values):
            patches = static_patchify(past_values)
            return patches.permute(0, 3, 1, 2).contiguous()

    return StaticPatchifier()


def _require_checkpoint_contract(model) -> None:
    expected = {
        "context_length": CONTEXT_LENGTH,
        "prediction_length": PREDICTION_LENGTH,
        "num_input_channels": NUM_INPUT_CHANNELS,
        "patch_length": PATCH_LENGTH,
        "patch_stride": PATCH_STRIDE,
    }
    config = getattr(model, "config", None)
    for field, value in expected.items():
        observed = getattr(config, field, None)
        if observed != value:
            raise ValueError(
                f"PatchTST {field} mismatch: expected {value}, got {observed!r}"
            )


def _deterministic_samples(torch_module):
    values = torch_module.linspace(
        -1.0,
        1.0,
        CONTEXT_LENGTH * NUM_INPUT_CHANNELS,
        dtype=torch_module.float32,
    ).reshape(1, CONTEXT_LENGTH, NUM_INPUT_CHANNELS)
    positions = torch_module.arange(CONTEXT_LENGTH * NUM_INPUT_CHANNELS).reshape(
        1, CONTEXT_LENGTH, NUM_INPUT_CHANNELS
    )
    observed_mask = positions.remainder(5).ne(0)
    return values, observed_mask


def compile_model(model_id: str):
    import rebel
    import torch
    from transformers import PatchTSTForPrediction

    model = PatchTSTForPrediction.from_pretrained(model_id).eval()
    model.requires_grad_(False)
    _require_checkpoint_contract(model)

    sample_values, sample_mask = _deterministic_samples(torch)
    with torch.no_grad():
        bool_mask_output = model(
            past_values=sample_values,
            past_observed_mask=sample_mask,
            return_dict=True,
        ).prediction_outputs
        float_mask_output = model(
            past_values=sample_values,
            past_observed_mask=sample_mask.to(dtype=sample_values.dtype),
            return_dict=True,
        ).prediction_outputs

    torch.testing.assert_close(
        float_mask_output,
        bool_mask_output,
        rtol=0,
        atol=0,
    )
    original_output = float_mask_output

    backbone = getattr(model, "model", None)
    if backbone is None or not hasattr(backbone, "patchifier"):
        raise ValueError("PatchTST checkpoint has no model.patchifier to replace")
    backbone.patchifier = build_static_patchifier(torch)

    with torch.no_grad():
        static_output = model(
            past_values=sample_values,
            past_observed_mask=sample_mask.to(dtype=sample_values.dtype),
            return_dict=True,
        ).prediction_outputs
    torch.testing.assert_close(
        static_output,
        original_output,
        rtol=1e-5,
        atol=1e-6,
    )

    class PatchTSTETTh1(torch.nn.Module):
        def __init__(self, patchtst):
            super().__init__()
            self.model = patchtst

        def forward(self, past_values, past_observed_mask):
            observed_mask = past_observed_mask.to(dtype=past_values.dtype)
            return self.model(
                past_values=past_values,
                past_observed_mask=observed_mask,
                return_dict=True,
            ).prediction_outputs

    wrapper = PatchTSTETTh1(model).eval()
    wrapper.requires_grad_(False)
    traced = torch.jit.trace(wrapper, (sample_values, sample_mask))
    if "aten::unfold" in str(traced.inlined_graph):
        raise RuntimeError("PatchTST trace still contains unsupported aten::unfold")

    return rebel.compile_from_torch(
        wrapper,
        [
            (
                "past_values",
                [1, CONTEXT_LENGTH, NUM_INPUT_CHANNELS],
                "float32",
            ),
            (
                "past_observed_mask",
                [1, CONTEXT_LENGTH, NUM_INPUT_CHANNELS],
                "bool",
            ),
        ],
        model_trace_method="jittrace",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = parser.parse_args(argv)
    output = emit_description_or_require_output(args, CONTRACT)
    if output is None:
        return 0

    selected_contract = replace(CONTRACT, model_id=args.model_id)
    report = save_and_validate(compile_model(args.model_id), output, selected_contract)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
