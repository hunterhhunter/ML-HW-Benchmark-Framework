from importlib.metadata import PackageNotFoundError, metadata, version
from pathlib import Path

import pytest


def test_furiosa_torch_dependency_contract():
    try:
        installed = version("furiosa-torch")
    except PackageNotFoundError:
        pytest.skip("Furiosa Torch is only installed in the vendor environment")
    assert installed == "2026.3.0"
    requirements = metadata("furiosa-torch").get_all("Requires-Dist") or []
    assert "torch==2.10.0" in requirements
    assert any(req.startswith("numpy>=2.2.6") for req in requirements)


def test_furiosa_torch_requirements_are_isolated_and_pinned():
    requirements = Path("framework/requirements-furiosa-torch.txt").read_text().splitlines()
    assert requirements == [
        "furiosa-torch[vision,llm]==2026.3.0",
        "onnx==1.20.1",
        "onnx2torch==1.5.15",
    ]
