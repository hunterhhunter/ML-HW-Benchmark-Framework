from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chronos_bolt.contracts import ChronosBoltContract
from tools.chronos_bolt_vendors.rbln import compile_rbln, run_rbln


class _Compiled:
    def __init__(self):
        self.saved = []

    def save(self, path):
        self.saved.append(path)
        Path(path).write_bytes(b"rbln-artifact")


class _Runtime:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def __call__(self, *inputs):
        self.calls.append(inputs)
        return self.output


def _contract():
    return ChronosBoltContract.tiny(d_model=4, use_reg_token=True)


def _fake_rebel(output=None):
    contract = _contract()
    inspection = SimpleNamespace(
        npu="RBLN-CA22",
        compiler_version="0.11.0",
        inputs=[
            SimpleNamespace(name=item.name, shape=item.shape, dtype=item.dtype)
            for item in contract.core_inputs
        ],
        outputs=[
            SimpleNamespace(
                name="quantile_preds",
                shape=contract.core_output.shape,
                dtype=contract.core_output.dtype,
            )
        ],
    )
    compiled = _Compiled()
    runtime = _Runtime(
        np.zeros(contract.core_output.shape, dtype=np.float32)
        if output is None
        else output
    )
    calls = []

    def compile_from_torch(model, inputs):
        calls.append((model, inputs))
        return compiled

    return SimpleNamespace(
        compile_from_torch=compile_from_torch,
        RBLNCompiledModel=SimpleNamespace(inspect=lambda path: inspection),
        Runtime=lambda path, **kwargs: runtime,
        compiled=compiled,
        runtime=runtime,
        compile_calls=calls,
    )


def test_rbln_compile_uses_the_exact_three_tensor_contract(monkeypatch, tmp_path):
    """Catches an artifact compiled with different names, dtypes, or static shapes."""
    fake_rebel = _fake_rebel()
    monkeypatch.setitem(__import__("sys").modules, "rebel", fake_rebel)
    contract = _contract()

    report = compile_rbln(torch.nn.Identity(), contract, tmp_path / "core.rbln")

    assert fake_rebel.compiled.saved == [str(tmp_path / "core.rbln")]
    assert fake_rebel.compile_calls[0][1] == [
        ("input_embeds", [1, 33, 4], "float32"),
        ("attention_mask", [1, 33], "float32"),
        ("decoder_input_embeds", [1, 1, 4], "float32"),
    ]
    assert report["artifact"]["size_bytes"] == len(b"rbln-artifact")
    assert report["inspection"]["npu"] == "RBLN-CA22"


def test_rbln_first_run_requires_one_finite_contract_matching_output(monkeypatch, tmp_path):
    """Catches treating artifact creation as device verification without an NPU call."""
    fake_rebel = _fake_rebel()
    monkeypatch.setitem(__import__("sys").modules, "rebel", fake_rebel)
    artifact = tmp_path / "core.rbln"
    artifact.write_bytes(b"rbln-artifact")
    contract = _contract()
    inputs = tuple(
        np.ones(item.shape, dtype=np.float32) for item in contract.core_inputs
    )

    output = run_rbln(artifact, inputs, contract)

    assert output.shape == (1, 9, 64)
    assert output.dtype == np.float32
    assert len(fake_rebel.runtime.calls) == 1
    assert all(value.flags.c_contiguous for value in fake_rebel.runtime.calls[0])


def test_rbln_run_rejects_non_finite_output(monkeypatch, tmp_path):
    """Catches a device run being reported as valid with invalid forecast values."""
    output = np.full((1, 9, 64), np.nan, dtype=np.float32)
    fake_rebel = _fake_rebel(output=output)
    monkeypatch.setitem(__import__("sys").modules, "rebel", fake_rebel)
    artifact = tmp_path / "core.rbln"
    artifact.write_bytes(b"rbln-artifact")
    contract = _contract()
    inputs = tuple(
        np.ones(item.shape, dtype=np.float32) for item in contract.core_inputs
    )

    with pytest.raises(ValueError, match="non-finite"):
        run_rbln(artifact, inputs, contract)
