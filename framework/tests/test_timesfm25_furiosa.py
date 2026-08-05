from types import SimpleNamespace

import torch

from timesfm25.contracts import TimesFM25Contract
from tools.timesfm25_vendors.furiosa import FuriosaDependencies, compile_furiosa_runner


class _Core(torch.nn.Module):
    def forward(self, context):
        del context
        return torch.zeros((1, 128), dtype=torch.float32)


def test_furiosa_requests_strict_static_compilation_without_eager_fallback(monkeypatch):
    captured = {}

    class _CompilerConfig:
        def __init__(self, *, tactic_hint):
            captured["tactic_hint"] = tactic_hint

    class _Backend:
        @staticmethod
        def with_config(config, *, eager_fallback):
            captured["config"] = config
            captured["eager_fallback"] = eager_fallback
            return "strict-backend"

    def fake_compile(module, *, backend, fullgraph, dynamic):
        captured.update(backend=backend, fullgraph=fullgraph, dynamic=dynamic)
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)
    dependencies = FuriosaDependencies(
        torch=torch,
        furiosa_torch=SimpleNamespace(backend=_Backend()),
        CompilerConfig=_CompilerConfig,
        TacticHintConfig=SimpleNamespace(Default="default"),
    )

    runner = compile_furiosa_runner(
        _Core(), TimesFM25Contract.fixed(), device="cpu", dependencies=dependencies
    )
    output = runner((torch.ones((1, 1024), dtype=torch.float32),))

    assert output.shape == (1, 128)
    assert captured["eager_fallback"] is False
    assert captured["fullgraph"] is True
    assert captured["dynamic"] is False
