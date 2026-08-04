from types import SimpleNamespace

import pytest
import torch

from chronos_bolt.core import ChronosBoltTransformerCore


class _Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.received = None

    def forward(self, *, attention_mask, inputs_embeds, return_dict):
        self.received = {
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "return_dict": return_dict,
        }
        return (inputs_embeds + 10.0,)


class _Decoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.received = None

    def forward(
        self,
        *,
        inputs_embeds,
        encoder_hidden_states,
        encoder_attention_mask,
        return_dict,
        use_cache,
    ):
        self.received = {
            "inputs_embeds": inputs_embeds,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "return_dict": return_dict,
            "use_cache": use_cache,
        }
        return (inputs_embeds + encoder_hidden_states[:, :1, :],)


class _OutputHead(torch.nn.Module):
    def forward(self, sequence_output):
        values = torch.arange(576, dtype=torch.float32, device=sequence_output.device)
        return values.reshape(1, 1, 576).expand(sequence_output.shape[0], -1, -1)


class _FakeChronosBolt(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(d_model=8)
        self.chronos_config = SimpleNamespace(
            prediction_length=64,
            quantiles=[0.1] * 9,
            use_reg_token=True,
        )
        self.encoder = _Encoder()
        self.decoder = _Decoder()
        self.output_patch_embedding = _OutputHead()


def test_core_returns_fixed_tensor_and_passes_tensor_only_t5_calls():
    """Catches a ModelOutput/container boundary or a changed core ABI."""
    model = _FakeChronosBolt()
    core = ChronosBoltTransformerCore(model)
    input_embeds = torch.zeros((1, 33, 8), dtype=torch.float32)
    attention_mask = torch.ones((1, 33), dtype=torch.float32)
    decoder_input_embeds = torch.full((1, 1, 8), 2.0, dtype=torch.float32)

    output = core(input_embeds, attention_mask, decoder_input_embeds)

    assert type(output) is torch.Tensor
    assert output.shape == (1, 9, 64)
    assert output.dtype == torch.float32
    assert output[0, 0, 0].item() == 0.0
    assert output[0, 8, 63].item() == 575.0
    assert model.encoder.received["return_dict"] is False
    assert model.decoder.received["return_dict"] is False
    assert model.decoder.received["use_cache"] is False
    assert torch.equal(model.decoder.received["encoder_attention_mask"], attention_mask)


def test_core_rejects_wrong_named_input_shape_before_t5_execution():
    """Catches vendor execution with an artifact-incompatible embedding shape."""
    core = ChronosBoltTransformerCore(_FakeChronosBolt())

    with pytest.raises(ValueError, match="input_embeds"):
        core(
            torch.zeros((1, 31, 8), dtype=torch.float32),
            torch.ones((1, 33), dtype=torch.float32),
            torch.zeros((1, 1, 8), dtype=torch.float32),
        )


def test_core_rejects_non_float32_output_head_result():
    """Catches a quantized or cast core result being mislabeled as the FP32 ABI."""
    model = _FakeChronosBolt()

    class _HalfOutputHead(torch.nn.Module):
        def forward(self, sequence_output):
            return torch.zeros((1, 1, 576), dtype=torch.float16)

    model.output_patch_embedding = _HalfOutputHead()
    core = ChronosBoltTransformerCore(model)
    with pytest.raises(ValueError, match="float32"):
        core(
            torch.zeros((1, 33, 8), dtype=torch.float32),
            torch.ones((1, 33), dtype=torch.float32),
            torch.zeros((1, 1, 8), dtype=torch.float32),
        )


def test_core_leaves_finite_output_validation_outside_compiled_graph():
    """Catches data-dependent Python control flow inside a fullgraph compiler target."""
    model = _FakeChronosBolt()

    class _NanOutputHead(torch.nn.Module):
        def forward(self, sequence_output):
            return torch.full((1, 1, 576), torch.nan, dtype=torch.float32)

    model.output_patch_embedding = _NanOutputHead()
    output = ChronosBoltTransformerCore(model)(
        torch.zeros((1, 33, 8), dtype=torch.float32),
        torch.ones((1, 33), dtype=torch.float32),
        torch.zeros((1, 1, 8), dtype=torch.float32),
    )
    assert torch.isnan(output).all()
