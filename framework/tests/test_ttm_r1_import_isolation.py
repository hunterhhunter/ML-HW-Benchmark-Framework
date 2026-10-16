"""Keep optional time-series dependencies out of unrelated targets."""

import os
from pathlib import Path
import subprocess
import sys


def test_mobilint_tensor_helpers_import_without_torch():
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
import builtins

original_import = builtins.__import__

def reject_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise ModuleNotFoundError("No module named 'torch'")
    return original_import(name, *args, **kwargs)

builtins.__import__ = reject_torch
from runtimes.mobilint_ttm_r2 import MOBILINT_TTM_R2_ADAPTER_ID
assert MOBILINT_TTM_R2_ADAPTER_ID == 'ttm-r2-aries-v1'
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
