from types import SimpleNamespace

import torch

from ttm_r2.contracts import TTMR2Contract
from tools.ttm_r2_vendors.furiosa import FuriosaDependencies, run_furiosa_core


class _Core(torch.nn.Module):
    def forward(self, values):
        return values[:, :96, :]


def test_r2_furiosa_requests_static_no_fallback_graph(monkeypatch):
    captured = {}

    class _Config:
        def __init__(self, *, tactic_hint): captured["hint"] = tactic_hint

    class _Backend:
        @staticmethod
        def with_config(config, *, eager_fallback):
            captured["fallback"] = eager_fallback
            return config

    def compile_fn(module, *, backend, fullgraph, dynamic):
        captured.update(fullgraph=fullgraph, dynamic=dynamic)
        return module

    monkeypatch.setattr(torch, "compile", compile_fn)
    dependencies = FuriosaDependencies(torch, SimpleNamespace(backend=_Backend()), _Config, SimpleNamespace(Default="default"))
    output = run_furiosa_core(
        _Core(), (torch.zeros((1, 512, 1), dtype=torch.float32),), TTMR2Contract.fixed(),
        device="cpu", dependencies=dependencies,
    )
    assert output.shape == (1, 96, 1)
    assert captured == {"hint": "default", "fallback": False, "fullgraph": True, "dynamic": False}
