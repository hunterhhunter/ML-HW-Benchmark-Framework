from pathlib import Path
import runpy
from types import SimpleNamespace

import torch

from ttm_r2.contracts import TTMR2Contract


def test_r2_rbln_quality_cli_uses_existing_artifact_and_240_windows():
    module = runpy.run_path("framework/tools/ttm_r2_rbln_etth1_quality.py", run_name="not_main")
    args = module["build_parser"]().parse_args([
        "--model-path", "/models/r2",
        "--dataset-path", "/data/ETTh1.csv",
        "--artifact", "/artifacts/r2.rbln",
        "--output-dir", "/results/out",
    ])

    assert args.windows == 240
    assert str(args.artifact) == "/artifacts/r2.rbln"


def test_r2_rbln_quality_runner_reuses_one_artifact_runtime(monkeypatch, tmp_path):
    module = runpy.run_path("framework/tools/ttm_r2_rbln_etth1_quality.py", run_name="not_main")
    artifact = Path(tmp_path) / "core.rbln"
    artifact.write_bytes(b"artifact")
    calls = []

    class Runtime:
        def __call__(self, value):
            calls.append(value)
            return __import__("numpy").zeros((1, 96, 1), dtype="float32")

    monkeypatch.setitem(
        __import__("sys").modules,
        "rebel",
        SimpleNamespace(Runtime=lambda *_args, **_kwargs: Runtime()),
    )
    runner = module["_rbln_runner"](artifact, TTMR2Contract.fixed())

    output = runner((torch.zeros((1, 512, 1), dtype=torch.float32),))

    assert tuple(output.shape) == (1, 96, 1)
    assert len(calls) == 1
