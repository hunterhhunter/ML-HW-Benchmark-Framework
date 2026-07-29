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
