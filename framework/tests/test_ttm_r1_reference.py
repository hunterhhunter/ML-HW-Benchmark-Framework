from types import SimpleNamespace

import torch

from ttm_r1 import reference
from ttm_r1 import core


class _FakeTTM(torch.nn.Module):
    def forward(self, *, past_values, return_dict):
        assert return_dict is True
        return SimpleNamespace(prediction_outputs=past_values[:, -96:, :] + 0.25)


def test_preflight_requires_exact_host_and_reference_forecasts(monkeypatch, tmp_path):
    """Catches vendor compilation starting before a reference-equivalent split exists."""
    monkeypatch.setattr(reference, "load_ttm_r1_model", lambda _: _FakeTTM())

    result = reference.run_preflight(tmp_path)

    assert set(result.core_inputs) == {"finite", "nan"}
    assert set(result.core_outputs) == {"finite", "nan"}
    assert result.host_parity["finite"]["max_abs_error"] == 0.0
    assert result.host_parity["nan"]["max_abs_error"] == 0.0


def test_preflight_requires_an_existing_local_checkpoint_path():
    """Catches a hidden Hugging Face download in a device execution command."""
    try:
        reference.run_preflight("does-not-exist")
    except ValueError as error:
        assert "local checkpoint" in str(error)
    else:
        raise AssertionError("run_preflight accepted a missing checkpoint directory")


def test_preflight_compares_lowered_patchify_core_against_the_original_model(monkeypatch, tmp_path):
    """Catches comparing a lowered core against itself instead of official unfold semantics."""

    class _TTMR1WithPatchify(_FakeTTM):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                context_length=512,
                patch_length=64,
                patch_stride=64,
                num_patches=8,
                num_input_channels=1,
            )
            self.backbone = torch.nn.Module()
            self.backbone.patching = torch.nn.Identity()

    original = _TTMR1WithPatchify()
    monkeypatch.setattr(reference, "load_ttm_r1_model", lambda _: original)

    result = reference.run_preflight(tmp_path)

    assert isinstance(original.backbone.patching, torch.nn.Identity)
    assert isinstance(result.core.model.backbone.patching, core.StaticTTMR1Patchify)
