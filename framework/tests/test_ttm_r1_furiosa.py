from types import SimpleNamespace

import pytest
import torch

from ttm_r1.contracts import TTMR1Contract
from tools.ttm_r1_vendors.furiosa import FuriosaDependencies, run_furiosa_core


class _Core(torch.nn.Module):
    def forward(self, past_values):
        del past_values
        return torch.zeros((1, 96, 1), dtype=torch.float32)


def test_furiosa_uses_strict_static_fullgraph_without_eager_fallback(monkeypatch):
    """Catches CPU fallback being mislabeled as an RNGD TTM compilation."""
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
        captured.update(backend=backend, fullgraph=fullgraph, dynamic=dynamic)
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)
    dependencies = FuriosaDependencies(
        torch=torch,
        furiosa_torch=SimpleNamespace(backend=_Backend()),
        CompilerConfig=_Config,
        TacticHintConfig=SimpleNamespace(Default="default"),
    )

    output = run_furiosa_core(
        _Core(),
        (torch.ones((1, 512, 1), dtype=torch.float32),),
        TTMR1Contract.fixed(),
        device="cpu",
        dependencies=dependencies,
    )

    assert output.shape == (1, 96, 1)
    assert captured["eager_fallback"] is False
    assert captured["fullgraph"] is True
    assert captured["dynamic"] is False


def test_furiosa_rejects_non_finite_first_output(monkeypatch):
    """Catches output validation being skipped after strict compilation."""
    class _NanCore(_Core):
        def forward(self, past_values):
            del past_values
            return torch.full((1, 96, 1), torch.nan, dtype=torch.float32)

    monkeypatch.setattr(torch, "compile", lambda model, **_kwargs: model)
    dependencies = FuriosaDependencies(
        torch=torch,
        furiosa_torch=SimpleNamespace(
            backend=SimpleNamespace(with_config=lambda *_args, **_kwargs: object())
        ),
        CompilerConfig=lambda **_kwargs: object(),
        TacticHintConfig=SimpleNamespace(Default="default"),
    )

    with pytest.raises(ValueError, match="non-finite"):
        run_furiosa_core(
            _NanCore(),
            (torch.ones((1, 512, 1), dtype=torch.float32),),
            TTMR1Contract.fixed(),
            device="cpu",
            dependencies=dependencies,
        )
