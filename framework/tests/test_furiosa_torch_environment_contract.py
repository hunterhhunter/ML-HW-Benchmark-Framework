import builtins
import importlib
import sys
from importlib.metadata import PackageNotFoundError, metadata, version
from pathlib import Path

import pytest


def test_furiosa_torch_dependency_contract_when_vendor_sdk_is_installed():
    try:
        installed = version("furiosa-torch")
    except PackageNotFoundError:
        pytest.skip("Furiosa Torch is installed only in the RNGD environment")

    assert installed == "2026.3.0"
    requirements = metadata("furiosa-torch").get_all("Requires-Dist") or []
    assert "torch==2.10.0" in requirements
    assert any(requirement.startswith("numpy>=2.2.6") for requirement in requirements)


def test_furiosa_torch_bert_requirements_are_isolated_and_pinned():
    requirements_path = (
        Path(__file__).resolve().parent.parent / "requirements-furiosa-torch.txt"
    )

    assert requirements_path.read_text().splitlines() == [
        "furiosa-torch==2026.3.0",
        "torch==2.10.0",
        "transformers==5.1.0",
        "numpy==2.5.1",
    ]


def test_unverified_model_dependencies_are_not_installed():
    requirements_path = (
        Path(__file__).resolve().parent.parent / "requirements-furiosa-torch.txt"
    )
    requirements = requirements_path.read_text()

    assert "ultralytics" not in requirements
    assert "onnx2torch" not in requirements
    assert "granite-tsfm" not in requirements


def test_furiosa_bert_profile_import_does_not_require_onnx(monkeypatch):
    """The pinned minimal BERT environment must reach the Torch runtime."""
    original_import = builtins.__import__

    def import_without_onnx(name, *args, **kwargs):
        if name == "onnx":
            raise ModuleNotFoundError("No module named 'onnx'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_onnx)
    monkeypatch.delitem(sys.modules, "core.model_profiles", raising=False)

    module = importlib.import_module("core.model_profiles")

    assert "bert-base-uncased" in module.SUPPORTED_PROFILES


def test_main_import_does_not_require_unrelated_optional_packages(monkeypatch):
    """BERT must start without ONNX or image-only dependencies."""
    original_import = builtins.__import__
    blocked_roots = {"PIL", "cv2", "onnx"}

    def import_without_vision_packages(name, *args, **kwargs):
        if name.split(".", 1)[0] in blocked_roots:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return original_import(name, *args, **kwargs)

    for module_name in tuple(sys.modules):
        if (
            module_name in {"main", "core.model_profiles"}
            or module_name.startswith(("dataloader", "preprocessor", "decoders"))
        ):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr(
        builtins,
        "__import__",
        import_without_vision_packages,
    )

    module = importlib.import_module("main")

    assert callable(module.create_dataloader)
    assert callable(module.create_decoder)
