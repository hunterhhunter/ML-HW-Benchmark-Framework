import os
from pathlib import Path
import subprocess
import sys


def test_main_import_does_not_require_torch_for_non_torch_backends():
    source_root = Path(__file__).resolve().parent.parent / "src"
    script = """
import builtins

real_import = builtins.__import__

def reject_torch(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("torch intentionally unavailable")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = reject_torch
import main
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=source_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
