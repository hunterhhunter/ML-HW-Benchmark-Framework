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
    requirements_path = Path(__file__).resolve().parent.parent / "requirements-furiosa-torch.txt"
    requirements = requirements_path.read_text().splitlines()
    assert requirements == [
        "furiosa-torch[vision,llm]==2026.3.0",
        "ultralytics==8.3.216",
        "onnx==1.20.1",
        "onnx2torch==1.5.15",
        "datasets==4.8.3",
        "scikit-learn==1.7.2",
        "pandas==3.0.1",
        "Deprecated==1.3.1",
        "urllib3==2.6.3",
        "filelock==3.25.2",
        "einops==0.8.2",
        "pytest==9.0.2",
        "psutil==7.2.2",
        "prometheus-client==0.24.1",
    ]


def test_patchtst_fm_dependency_does_not_downgrade_furiosa_transformers():
    requirements_path = Path(__file__).resolve().parent.parent / "requirements-furiosa-torch.txt"

    assert "granite-tsfm" not in requirements_path.read_text()
