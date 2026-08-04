from types import SimpleNamespace

import pytest
import torch

from chronos_bolt.contracts import ChronosBoltContract
from tools.chronos_bolt_vendors.furiosa import FuriosaDependencies, run_furiosa


class _Core(torch.nn.Module):
    def forward(self, input_embeds, attention_mask, decoder_input_embeds):
        del input_embeds, attention_mask, decoder_input_embeds
        return torch.zeros((1, 9, 64), dtype=torch.float32)


def test_furiosa_uses_strict_static_fullgraph_without_eager_fallback(monkeypatch):
    """Catches a successful CPU fallback being mislabeled as an RNGD compile."""
    captured = {}

    class _Config:
        def __init__(self, *, tactic_hint):
            captured["tactic_hint"] = tactic_hint

    class _Backend:
        @staticmethod
        def with_config(config, *, eager_fallback):
            captured["config"] = config
            captured["eager_fallback"] = eager_fallback
            return "strict-backend"

    def fake_compile(model, *, backend, fullgraph, dynamic):
        captured["backend"] = backend
        captured["fullgraph"] = fullgraph
        captured["dynamic"] = dynamic
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)
    dependencies = FuriosaDependencies(
        torch=torch,
        furiosa_torch=SimpleNamespace(backend=_Backend()),
        CompilerConfig=_Config,
        TacticHintConfig=SimpleNamespace(Default="default"),
    )
    contract = ChronosBoltContract.tiny(d_model=4, use_reg_token=True)
    inputs = tuple(torch.ones(item.shape, dtype=torch.float32) for item in contract.core_inputs)

    output = run_furiosa(
        _Core(),
        inputs,
        contract,
        device="cpu",
        dependencies=dependencies,
    )

    assert output.shape == (1, 9, 64)
    assert captured == {
        "tactic_hint": "default",
        "config": captured["config"],
        "eager_fallback": False,
        "backend": "strict-backend",
        "fullgraph": True,
        "dynamic": False,
    }


def test_furiosa_rejects_non_finite_first_output(monkeypatch):
    """Catches device output validation being skipped after strict compilation."""
    class _NanCore(_Core):
        def forward(self, *inputs):
            del inputs
            return torch.full((1, 9, 64), torch.nan, dtype=torch.float32)

    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: model)
    dependencies = FuriosaDependencies(
        torch=torch,
        furiosa_torch=SimpleNamespace(
            backend=SimpleNamespace(with_config=lambda config, **kwargs: object())
        ),
        CompilerConfig=lambda **kwargs: object(),
        TacticHintConfig=SimpleNamespace(Default="default"),
    )
    contract = ChronosBoltContract.tiny(d_model=4, use_reg_token=True)
    inputs = tuple(torch.ones(item.shape, dtype=torch.float32) for item in contract.core_inputs)

    with pytest.raises(ValueError, match="non-finite"):
        run_furiosa(
            _NanCore(),
            inputs,
            contract,
            device="cpu",
            dependencies=dependencies,
        )
