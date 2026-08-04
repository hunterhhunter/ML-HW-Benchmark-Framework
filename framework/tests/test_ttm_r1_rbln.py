from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ttm_r1.contracts import TTMR1Contract
from tools.ttm_r1_vendors.rbln import compile_rbln, run_rbln_artifact


class _Compiled:
    def save(self, path):
        Path(path).write_bytes(b"rbln-artifact")


def _fake_rebel(output=None):
    contract = TTMR1Contract.fixed()
    inspection = SimpleNamespace(
        npu="RBLN-CA22",
        compiler_version="0.11.0",
        inputs=[
            SimpleNamespace(name=item.name, shape=item.shape, dtype=item.dtype)
            for item in contract.core_inputs
        ],
        outputs=[
            SimpleNamespace(
                name=contract.core_output.name,
                shape=contract.core_output.shape,
                dtype=contract.core_output.dtype,
            )
        ],
    )
    calls = []
    runtime_calls = []

    class _Runtime:
        def __call__(self, *inputs):
            runtime_calls.append(inputs)
            return (
                np.zeros(contract.core_output.shape, dtype=np.float32)
                if output is None
                else output
            )

    def compile_from_torch(model, inputs):
        calls.append((model, inputs))
        return _Compiled()

    return SimpleNamespace(
        compile_from_torch=compile_from_torch,
        RBLNCompiledModel=SimpleNamespace(inspect=lambda _: inspection),
        Runtime=lambda *_args, **_kwargs: _Runtime(),
        compile_calls=calls,
        runtime_calls=runtime_calls,
    )


def test_rbln_compile_uses_the_exact_single_tensor_contract(monkeypatch, tmp_path):
    """Catches compiling a TTM artifact with a different static ABI."""
    fake_rebel = _fake_rebel()
    monkeypatch.setitem(__import__("sys").modules, "rebel", fake_rebel)

    report = compile_rbln(torch.nn.Identity(), TTMR1Contract.fixed(), tmp_path / "core.rbln")

    assert fake_rebel.compile_calls[0][1] == [
        ("past_values", [1, 512, 1], "float32")
    ]
    assert report["inspection"]["output"]["shape"] == [1, 96, 1]


def test_rbln_first_run_requires_one_finite_contract_matching_output(monkeypatch, tmp_path):
    """Catches treating artifact creation as CA22 execution evidence."""
    fake_rebel = _fake_rebel()
    monkeypatch.setitem(__import__("sys").modules, "rebel", fake_rebel)
    artifact = tmp_path / "core.rbln"
    artifact.write_bytes(b"rbln-artifact")

    output = run_rbln_artifact(
        artifact,
        (np.ones((1, 512, 1), dtype=np.float32),),
        TTMR1Contract.fixed(),
    )

    assert output.shape == (1, 96, 1)
    assert len(fake_rebel.runtime_calls) == 1


def test_rbln_run_rejects_non_finite_output(monkeypatch, tmp_path):
    """Catches invalid device values being presented as a valid forecast."""
    fake_rebel = _fake_rebel(np.full((1, 96, 1), np.nan, dtype=np.float32))
    monkeypatch.setitem(__import__("sys").modules, "rebel", fake_rebel)
    artifact = tmp_path / "core.rbln"
    artifact.write_bytes(b"rbln-artifact")

    with pytest.raises(ValueError, match="non-finite"):
        run_rbln_artifact(
            artifact,
            (np.ones((1, 512, 1), dtype=np.float32),),
            TTMR1Contract.fixed(),
        )
