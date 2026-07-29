from pathlib import Path
import importlib
import json
import os
import subprocess
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


@pytest.mark.parametrize(
    ("module", "model_id", "input_names", "output_shape"),
    [
        (
            "tools.rbln_compile_recipes.resnet50.compile",
            "torchvision/resnet50-imagenet1k-v2",
            ["input_np"],
            [1, 1000],
        ),
        (
            "tools.rbln_compile_recipes.bert_sst2.compile",
            "textattack/bert-base-uncased-SST-2",
            ["input_ids", "attention_mask"],
            [1, 2],
        ),
    ],
)
def test_recipe_describe_needs_no_optional_sdk(
    module, model_id, input_names, output_shape
):
    result = subprocess.run(
        [sys.executable, "-m", module, "--describe"],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["model_id"] == model_id
    assert [item["name"] for item in payload["inputs"]] == input_names
    assert payload["outputs"][0]["shape"] == output_shape


@pytest.mark.parametrize(
    "module",
    [
        "tools.rbln_compile_recipes.resnet50.compile",
        "tools.rbln_compile_recipes.bert_sst2.compile",
    ],
)
def test_recipe_help_does_not_import_optional_sdks(module, tmp_path):
    (tmp_path / "sitecustomize.py").write_text(
        """import builtins

_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split('.', 1)[0] in {'rebel', 'torch', 'torchvision', 'transformers'}:
        raise RuntimeError(f'optional SDK imported: {name}')
    return _import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
"""
    )
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join(
            filter(None, (str(tmp_path), os.environ.get("PYTHONPATH")))
        )
    }

    subprocess.run(
        [sys.executable, "-m", module, "--help"],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )


def test_yolov5_preflight_names_missing_root_and_weight(tmp_path):
    module = importlib.import_module("tools.rbln_compile_recipes.yolov5m.compile")

    with pytest.raises(FileNotFoundError, match="YOLOv5 source root"):
        module.validate_sources(tmp_path / "missing", tmp_path / "yolov5m.pt")


def test_yolov5_preflight_rejects_empty_weight_before_git(monkeypatch, tmp_path):
    module = importlib.import_module("tools.rbln_compile_recipes.yolov5m.compile")
    yolov5_root = tmp_path / "yolov5"
    (yolov5_root / "models").mkdir(parents=True)
    (yolov5_root / "models" / "experimental.py").touch()
    (yolov5_root / "models" / "yolo.py").touch()
    weights = tmp_path / "yolov5m.pt"
    weights.touch()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(
            stdout=module.EXPECTED_YOLOV5_REVISION + "\n"
        ),
    )

    with pytest.raises(ValueError, match="YOLOv5 weight file is empty"):
        module.validate_sources(yolov5_root, weights)


def test_yolov5_describe_records_pinned_revision():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.rbln_compile_recipes.yolov5m.compile",
            "--describe",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert "86fd1ab270cb2f7e53ee7412cd4a0650bf4bcc51" in payload["notes"]
    assert payload["outputs"][0]["shape"] == [1, 25200, 85]


def test_yolov5_preflight_uses_exact_pinned_revision_command(monkeypatch, tmp_path):
    module = importlib.import_module("tools.rbln_compile_recipes.yolov5m.compile")
    yolov5_root = tmp_path / "yolov5"
    (yolov5_root / "models").mkdir(parents=True)
    (yolov5_root / "models" / "experimental.py").touch()
    (yolov5_root / "models" / "yolo.py").touch()
    weights = tmp_path / "yolov5m.pt"
    weights.write_bytes(b"weights")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return types.SimpleNamespace(stdout=module.EXPECTED_YOLOV5_REVISION + "\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module.validate_sources(yolov5_root, weights)

    assert calls == [
        (
            ["git", "-C", str(yolov5_root), "rev-parse", "HEAD"],
            {"check": True, "text": True, "capture_output": True},
        )
    ]


def test_yolov5_preflight_rejects_unpinned_revision_with_checkout_command(
    monkeypatch, tmp_path
):
    module = importlib.import_module("tools.rbln_compile_recipes.yolov5m.compile")
    yolov5_root = tmp_path / "yolov5"
    (yolov5_root / "models").mkdir(parents=True)
    (yolov5_root / "models" / "experimental.py").touch()
    (yolov5_root / "models" / "yolo.py").touch()
    weights = tmp_path / "yolov5m.pt"
    weights.write_bytes(b"weights")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(stdout="different-revision\n"),
    )

    with pytest.raises(RuntimeError, match=rf"checkout {module.EXPECTED_YOLOV5_REVISION}"):
        module.validate_sources(yolov5_root, weights)
