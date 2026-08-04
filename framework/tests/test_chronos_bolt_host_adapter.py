from types import SimpleNamespace

import pytest
import torch

from chronos_bolt.host_adapter import ChronosBoltHostAdapter


class _PatchEmbedding(torch.nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.last_input = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.last_input = value.detach().clone()
        return value[..., : self.d_model]


class _Shared(torch.nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return torch.arange(
            self.d_model, dtype=torch.float32, device=token_ids.device
        ).reshape(1, 1, self.d_model).expand(token_ids.shape[0], -1, -1)


class _FakeChronosBolt:
    def __init__(self, d_model: int = 4) -> None:
        self.config = SimpleNamespace(
            d_model=d_model,
            decoder_start_token_id=0,
            reg_token_id=1,
        )
        self.chronos_config = SimpleNamespace(
            context_length=2048,
            input_patch_size=16,
            input_patch_stride=16,
            use_reg_token=True,
        )
        self.instance_norm = SimpleNamespace(eps=1e-5, use_arcsinh=False)
        self.input_patch_embedding = _PatchEmbedding(d_model)
        self.shared = _Shared(d_model)


def test_adapter_left_pads_short_context_and_appends_the_learned_reg_token():
    """Catches dropping Tiny's learned REG token from the vendor core ABI."""
    model = _FakeChronosBolt()
    adapter = ChronosBoltHostAdapter(model)

    prepared = adapter.prepare(torch.arange(20, dtype=torch.float32).reshape(1, 20))

    assert prepared.input_embeds.shape == (1, 33, 4)
    assert prepared.attention_mask.shape == (1, 33)
    assert prepared.attention_mask.dtype == torch.float32
    assert prepared.attention_mask.tolist() == [[0.0] * 30 + [1.0, 1.0, 1.0]]
    assert prepared.decoder_input_embeds.shape == (1, 1, 4)
    assert model.input_patch_embedding.last_input.shape == (1, 32, 32)


def test_explicit_mask_zeroes_patch_value_but_does_not_change_instance_norm_stats():
    """Catches using the explicit mask for loc/scale, unlike Chronos InstanceNorm."""
    model = _FakeChronosBolt()
    prepared = ChronosBoltHostAdapter(model).prepare(
        torch.tensor([[1.0, 2.0, 3.0]]),
        observed_mask=torch.tensor([[True, False, True]]),
    )

    expected_scale = torch.sqrt(torch.tensor(2.0 / 3.0))
    assert torch.allclose(prepared.loc, torch.tensor([[2.0]]))
    assert torch.allclose(prepared.scale, torch.tensor([[expected_scale]]))
    assert model.input_patch_embedding.last_input[0, -1, 14].item() == 0.0
    assert model.input_patch_embedding.last_input[0, -1, 30].item() == 0.0


def test_restore_applies_original_loc_and_scale_to_every_quantile():
    """Catches returning normalized NPU output as user-visible predictions."""
    adapter = ChronosBoltHostAdapter(_FakeChronosBolt())
    prepared = adapter.prepare(torch.tensor([[1.0, 2.0, 3.0]]))

    restored = prepared.restore(torch.ones((1, 9, 64), dtype=torch.float32))

    assert restored.shape == (1, 9, 64)
    assert torch.allclose(restored, torch.full((1, 9, 64), 2.0 + (2.0 / 3.0) ** 0.5))


@pytest.mark.parametrize(
    ("context", "mask", "message"),
    [
        (torch.zeros((512,)), None, "shape"),
        (torch.zeros((1, 512), dtype=torch.float64), None, "float32"),
        (torch.zeros((1, 512)), torch.ones((1, 511), dtype=torch.bool), "match"),
    ],
)
def test_adapter_rejects_inputs_that_cannot_match_fixed_core_abi(context, mask, message):
    """Catches malformed host data before a vendor compiler receives it."""
    with pytest.raises(ValueError, match=message):
        ChronosBoltHostAdapter(_FakeChronosBolt()).prepare(context, observed_mask=mask)
