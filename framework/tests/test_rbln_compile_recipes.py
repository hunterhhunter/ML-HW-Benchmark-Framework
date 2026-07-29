from pathlib import Path
import json
import sys
import types

import pytest

from tools.rbln_compile_recipes.common import (
    RecipeContract,
    TensorContract,
    contract_to_dict,
    create_parser,
    emit_description_or_require_output,
    prepare_output_path,
    save_and_validate,
)


def _contract():
    return RecipeContract(
        recipe="unit",
        model_id="owner/model",
        inputs=(TensorContract("x", (1, 3), "float32"),),
        outputs=(TensorContract("y", (1, 2), "float32"),),
        allow_unnamed_outputs=True,
        notes=("fixed batch one",),
    )


def test_contract_description_is_json_safe_and_stable():
    assert contract_to_dict(_contract()) == {
        "recipe": "unit",
        "model_id": "owner/model",
        "target_npu": "RBLN-CA22",
        "inputs": [{"name": "x", "shape": [1, 3], "dtype": "float32"}],
        "outputs": [{"name": "y", "shape": [1, 2], "dtype": "float32"}],
        "allow_unnamed_outputs": True,
        "notes": ["fixed batch one"],
    }


def test_prepare_output_path_requires_rbln_and_refuses_overwrite(tmp_path):
    with pytest.raises(ValueError, match=".rbln"):
        prepare_output_path(tmp_path / "model.bin")
    existing = tmp_path / "model.rbln"
    existing.write_bytes(b"do-not-overwrite")
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_output_path(existing)


def test_describe_emits_contract_before_requiring_an_output(capsys):
    args = create_parser().parse_args(["--describe"])

    assert emit_description_or_require_output(args, _contract()) is None
    assert json.loads(capsys.readouterr().out) == contract_to_dict(_contract())


def test_compile_requires_an_explicit_output_after_description_is_not_requested():
    args = create_parser().parse_args([])

    with pytest.raises(ValueError, match="--output"):
        emit_description_or_require_output(args, _contract())


def _inspection(*, npu="RBLN-CA22", inputs=None, outputs=None):
    return {
        "npu": npu,
        "compiler_version": "0.11.0",
        "inputs": inputs
        if inputs is not None
        else [{"name": "x", "shape": [1, 3], "dtype": "float32"}],
        "outputs": outputs
        if outputs is not None
        else [{"name": None, "shape": [1, 2], "dtype": "float32"}],
    }


class _Compiled:
    def __init__(self, content=b"compiled"):
        self.content = content
        self.save_calls = 0

    def save(self, path):
        self.save_calls += 1
        Path(path).write_bytes(self.content)


def _inject_rebel(monkeypatch, inspection):
    fake_rebel = types.SimpleNamespace(
        RBLNCompiledModel=types.SimpleNamespace(inspect=lambda path: inspection)
    )
    monkeypatch.setitem(sys.modules, "rebel", fake_rebel)


def test_save_and_validate_accepts_mapping_inspect_and_reports_sha(monkeypatch, tmp_path):
    compiled = _Compiled()
    _inject_rebel(monkeypatch, _inspection())

    report = save_and_validate(compiled, tmp_path / "model.rbln", _contract())

    assert compiled.save_calls == 1
    assert report["size_bytes"] == 8
    assert len(report["sha256"]) == 64
    assert report["compiler_version"] == "0.11.0"


def test_save_and_validate_accepts_attribute_descriptors(monkeypatch, tmp_path):
    descriptor = lambda name, shape: types.SimpleNamespace(
        name=name, shape=shape, dtype="FLOAT32"
    )
    inspection = types.SimpleNamespace(
        npu="RBLN-CA22",
        compiler_version="0.11.0",
        inputs=[descriptor("x", (1, 3))],
        outputs=[descriptor("y", (1, 2))],
    )
    _inject_rebel(monkeypatch, inspection)

    report = save_and_validate(_Compiled(), tmp_path / "attribute.rbln", _contract())

    assert report["target_npu"] == "RBLN-CA22"


def test_save_and_validate_rejects_unnamed_output_without_permission(monkeypatch, tmp_path):
    contract = RecipeContract(
        recipe="unit",
        model_id="owner/model",
        inputs=(TensorContract("x", (1, 3), "float32"),),
        outputs=(TensorContract("y", (1, 2), "float32"),),
    )
    _inject_rebel(monkeypatch, _inspection())

    with pytest.raises(ValueError, match="output.*name"):
        save_and_validate(_Compiled(), tmp_path / "unnamed.rbln", contract)


def test_save_and_validate_rejects_output_count_mismatch(monkeypatch, tmp_path):
    _inject_rebel(monkeypatch, _inspection(outputs=[]))

    with pytest.raises(ValueError, match="output count"):
        save_and_validate(_Compiled(), tmp_path / "count.rbln", _contract())


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        ([{"name": None, "shape": [1, 3], "dtype": "float32"}], "shape"),
        ([{"name": None, "shape": [1, 2], "dtype": "float16"}], "dtype"),
    ],
)
def test_save_and_validate_rejects_output_abi_mismatch(
    monkeypatch, tmp_path, outputs, message
):
    _inject_rebel(monkeypatch, _inspection(outputs=outputs))

    with pytest.raises(ValueError, match=message):
        save_and_validate(_Compiled(), tmp_path / "mismatch.rbln", _contract())


def test_save_and_validate_rejects_input_name_mismatch(monkeypatch, tmp_path):
    _inject_rebel(
        monkeypatch,
        _inspection(inputs=[{"name": "wrong", "shape": [1, 3], "dtype": "float32"}]),
    )

    with pytest.raises(ValueError, match="input.*name"):
        save_and_validate(_Compiled(), tmp_path / "input-name.rbln", _contract())


def test_save_and_validate_rejects_non_batch_one_contract(monkeypatch, tmp_path):
    contract = RecipeContract(
        recipe="unit",
        model_id="owner/model",
        inputs=(TensorContract("x", (2, 3), "float32"),),
        outputs=(TensorContract("y", (2, 2), "float32"),),
        allow_unnamed_outputs=True,
    )
    _inject_rebel(
        monkeypatch,
        _inspection(
            inputs=[{"name": "x", "shape": [2, 3], "dtype": "float32"}],
            outputs=[{"name": None, "shape": [2, 2], "dtype": "float32"}],
        ),
    )

    with pytest.raises(ValueError, match="batch size 1"):
        save_and_validate(_Compiled(), tmp_path / "batch-two.rbln", contract)


def test_save_and_validate_rejects_npu_mismatch(monkeypatch, tmp_path):
    _inject_rebel(monkeypatch, _inspection(npu="RBLN-CA12"))

    with pytest.raises(ValueError, match="NPU"):
        save_and_validate(_Compiled(), tmp_path / "npu.rbln", _contract())


def test_save_and_validate_rejects_non_ca22_contract(monkeypatch, tmp_path):
    contract = RecipeContract(
        recipe="unit",
        model_id="owner/model",
        inputs=(TensorContract("x", (1, 3), "float32"),),
        outputs=(TensorContract("y", (1, 2), "float32"),),
        allow_unnamed_outputs=True,
        target_npu="RBLN-CA12",
    )
    _inject_rebel(monkeypatch, _inspection(npu="RBLN-CA12"))

    with pytest.raises(ValueError, match="RBLN-CA22"):
        save_and_validate(_Compiled(), tmp_path / "ca12.rbln", contract)


def test_save_and_validate_rejects_zero_byte_artifact_without_sdk(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "rebel", raising=False)

    with pytest.raises(ValueError, match="empty"):
        save_and_validate(_Compiled(b""), tmp_path / "empty.rbln", _contract())
