from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from timesfm25.contracts import TimesFM25Contract
from tools.timesfm25_vendors.rbln import compile_rbln, run_rbln_artifact


class _Compiled:
    def save(self, path):
        Path(path).write_bytes(b"rbln-artifact")


def _fake_rebel():
    contract = TimesFM25Contract.fixed()
    inspection = SimpleNamespace(
        npu="RBLN-CA22",
        compiler_version="0.11.0",
        inputs=[SimpleNamespace(name="normalized_context", shape=(1, 1024), dtype="float32")],
        outputs=[SimpleNamespace(name="point_forecast", shape=(1, 128), dtype="float32")],
    )
    compile_calls, runtime_calls = [], []

    class _Runtime:
        def __call__(self, *inputs):
            runtime_calls.append(inputs)
            return np.zeros(contract.core_output.shape, dtype=np.float32)

    def compile_from_torch(module, input_info):
        compile_calls.append((module, input_info))
        return _Compiled()

    return SimpleNamespace(
        compile_from_torch=compile_from_torch,
        RBLNCompiledModel=SimpleNamespace(inspect=lambda _: inspection),
        Runtime=lambda *_args, **_kwargs: _Runtime(),
        compile_calls=compile_calls,
        runtime_calls=runtime_calls,
    )


def test_rbln_compile_uses_fixed_timesfm_point_core_abi(monkeypatch, tmp_path):
    fake_rebel = _fake_rebel()
    monkeypatch.setitem(__import__("sys").modules, "rebel", fake_rebel)

    report = compile_rbln(torch.nn.Identity(), TimesFM25Contract.fixed(), tmp_path / "core.rbln")

    assert fake_rebel.compile_calls[0][1] == [("normalized_context", [1, 1024], "float32")]
    assert report["inspection"]["output"]["shape"] == [1, 128]


def test_rbln_runtime_executes_one_finite_timesfm_forecast(monkeypatch, tmp_path):
    fake_rebel = _fake_rebel()
    monkeypatch.setitem(__import__("sys").modules, "rebel", fake_rebel)
    artifact = tmp_path / "core.rbln"
    artifact.write_bytes(b"rbln-artifact")

    output = run_rbln_artifact(
        artifact,
        (np.ones((1, 1024), dtype=np.float32),),
        TimesFM25Contract.fixed(),
    )

    assert output.shape == (1, 128)
    assert len(fake_rebel.runtime_calls) == 1
