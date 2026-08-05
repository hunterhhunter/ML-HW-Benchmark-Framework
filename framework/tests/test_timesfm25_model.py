from types import SimpleNamespace

import pytest
import torch

from timesfm25 import model


class _FakeTimesFM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(
            patch_length=32,
            horizon_length=128,
            num_hidden_layers=20,
            hidden_size=1280,
        )

    @classmethod
    def from_pretrained(cls, path, *, local_files_only):
        assert local_files_only is True
        assert path.endswith("checkpoint")
        return cls()


def test_loader_requires_local_checkpoint_directory(tmp_path):
    with pytest.raises(ValueError, match="local checkpoint directory"):
        model.load_timesfm25_model(str(tmp_path / "missing"))


def test_loader_uses_local_weights_and_freezes_model(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(model, "_timesfm_model_class", lambda: _FakeTimesFM)

    loaded = model.load_timesfm25_model(str(checkpoint))

    assert loaded.training is False
    assert loaded.weight.dtype == torch.float32
    assert loaded.weight.requires_grad is False
