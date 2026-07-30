import importlib.util
import sys
import types
from enum import Enum
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "tools" / "inspect_mobilint_mxq.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "inspect_mobilint_mxq",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_qbruntime():
    state = types.SimpleNamespace(models=[], configs=[])

    class DataType(Enum):
        Int64 = "Int64"

    class Cluster(Enum):
        Cluster0 = 0

    class Core(Enum):
        Core0 = 0

    class CoreId:
        def __init__(self, cluster, core):
            self.cluster = cluster
            self.core = core

    class ModelConfig:
        def __init__(self):
            self.calls = []
            state.configs.append(self)

        def set_global8_core_mode(self):
            self.calls.append("global8")
            return True

        def set_single_core_mode(self, num_cores, core_ids):
            self.calls.append(("single", num_cores, core_ids))
            return True

    class Model:
        def __init__(self, path, config):
            self.path = path
            self.config = config
            self.dispose_calls = 0
            self.getter_error = None
            state.models.append(self)

        def get_num_model_variants(self):
            return 2

        def get_model_input_data_type(self):
            return [DataType.Int64, DataType.Int64]

        def get_model_input_shape(self):
            if self.getter_error is not None:
                raise self.getter_error
            return [(128,), (128,)]

        def get_model_output_shape(self):
            return [(1, 2)]

        def dispose(self):
            self.dispose_calls += 1

    module = types.ModuleType("qbruntime")
    module.__version__ = "v1.3.2"
    module.ModelConfig = ModelConfig
    module.Model = Model
    module.Cluster = Cluster
    module.Core = Core
    module.CoreId = CoreId
    return module, state


def test_inspector_returns_json_safe_contract_and_disposes_model(tmp_path):
    module = _load_module()
    artifact = tmp_path / "bert.mxq"
    artifact.write_bytes(b"mxq")
    qbruntime, state = _fake_qbruntime()

    result = module.inspect_mxq(
        artifact,
        core_mode="global8",
        qbruntime_module=qbruntime,
    )

    assert result == {
        "sdk_version": "v1.3.2",
        "artifact": str(artifact.resolve()),
        "core_mode": "global8",
        "variants": 2,
        "input_dtypes": ["Int64", "Int64"],
        "input_shapes": [[128], [128]],
        "output_shapes": [[1, 2]],
    }
    assert state.configs[0].calls == ["global8"]
    assert state.models[0].dispose_calls == 1


def test_inspector_disposes_model_when_metadata_getter_fails(tmp_path):
    module = _load_module()
    artifact = tmp_path / "broken.mxq"
    artifact.write_bytes(b"mxq")
    qbruntime, state = _fake_qbruntime()

    original_model = qbruntime.Model

    class BrokenModel(original_model):
        def __init__(self, path, config):
            super().__init__(path, config)
            self.getter_error = RuntimeError("metadata failed")

    qbruntime.Model = BrokenModel

    with pytest.raises(RuntimeError, match="metadata failed"):
        module.inspect_mxq(artifact, qbruntime_module=qbruntime)

    assert state.models[0].dispose_calls == 1


def test_inspector_single_mode_targets_cluster0_core0_for_qbruntime_v13(
    tmp_path,
):
    module = _load_module()
    artifact = tmp_path / "bert.mxq"
    artifact.write_bytes(b"mxq")
    qbruntime, state = _fake_qbruntime()

    module.inspect_mxq(
        artifact,
        core_mode="single",
        qbruntime_module=qbruntime,
    )

    call = state.configs[0].calls[0]
    assert call[0] == "single"
    assert call[1] is None
    assert len(call[2]) == 1
    assert call[2][0].cluster is qbruntime.Cluster.Cluster0
    assert call[2][0].core is qbruntime.Core.Core0


def test_inspector_imports_sdk_lazily_and_reports_missing_package(
    tmp_path,
    monkeypatch,
):
    module = _load_module()
    artifact = tmp_path / "model.mxq"
    artifact.write_bytes(b"mxq")
    monkeypatch.setitem(sys.modules, "qbruntime", None)

    with pytest.raises(ImportError, match="mobilint-qb-runtime"):
        module.inspect_mxq(artifact)
