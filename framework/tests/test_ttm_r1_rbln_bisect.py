import importlib
from types import SimpleNamespace

import torch


class _Scaler(torch.nn.Module):
    def forward(self, values, observed_mask):
        assert torch.equal(observed_mask, torch.ones_like(values))
        return values + 1.0, torch.zeros((1, 1, 1)), torch.ones((1, 1, 1))


class _Patching(torch.nn.Module):
    def forward(self, values):
        return values.reshape(1, 8, 64, 1).permute(0, 3, 1, 2).contiguous()


class _Encoder(torch.nn.Module):
    def forward(self, patches, *, output_hidden_states, return_dict):
        assert output_hidden_states is False
        assert return_dict is True
        return SimpleNamespace(last_hidden_state=patches + 2.0)


class _Decoder(torch.nn.Module):
    def forward(self, *, hidden_state, patch_input, output_hidden_states, static_categorical_values):
        assert output_hidden_states is False
        assert static_categorical_values is None
        return hidden_state + patch_input, None


class _Head(torch.nn.Module):
    def forward(self, hidden_features, *, past_values, future_values):
        assert future_values is None
        return past_values[:, :96, :]


class _FakeTTMR1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.scaler = _Scaler()
        self.backbone.patching = _Patching()
        self.backbone.encoder = _Encoder()
        self.decoder = _Decoder()
        self.head = _Head()


def test_bisection_stages_keep_the_fixed_ttm_dataflow_and_shapes():
    """Catches a diagnostic stage changing the data it is meant to localize."""
    bisection = importlib.import_module("ttm_r1.rbln_bisect")
    values = torch.arange(512, dtype=torch.float32).reshape(1, 512, 1)

    stages = bisection.build_probe_stages(_FakeTTMR1(), values)

    assert [stage.name for stage in stages] == [
        "scaler",
        "patchify",
        "encoder",
        "decoder",
        "head",
        "restore",
    ]
    assert [tuple(stage.output.shape) for stage in stages] == [
        (1, 512, 1),
        (1, 1, 8, 64),
        (1, 1, 8, 64),
        (1, 1, 8, 64),
        (1, 96, 1),
        (1, 96, 1),
    ]
    assert torch.equal(stages[-1].output, values[:, :96, :])


def test_bisection_compiles_every_stage_and_preserves_a_failure_boundary(tmp_path):
    """Catches a generic compiler failure hiding which fixed stage first failed."""
    bisection = importlib.import_module("ttm_r1.rbln_bisect")
    values = torch.arange(512, dtype=torch.float32).reshape(1, 512, 1)
    stages = bisection.build_probe_stages(_FakeTTMR1(), values)
    calls = []

    class _Compiled:
        def __init__(self, stage):
            self.stage = stage

        def save(self, path):
            __import__("pathlib").Path(path).write_bytes(self.stage.encode())

    def compile_from_torch(module, input_info):
        calls.append(input_info)
        if len(calls) == 3:
            raise RuntimeError("encoder blocked")
        return _Compiled(stages[len(calls) - 1].name)

    report = bisection.compile_probe_stages(
        SimpleNamespace(compile_from_torch=compile_from_torch), stages, tmp_path
    )

    assert [item["status"] for item in report] == [
        "compiled",
        "compiled",
        "compile_failed",
        "compiled",
        "compiled",
        "compiled",
    ]
    assert report[2]["error"] == {"type": "RuntimeError", "message": "encoder blocked"}
    assert calls[0] == [("input_0", [1, 512, 1], "float32")]


def test_bisection_cli_passes_local_checkpoint_and_fresh_result_directory(monkeypatch, tmp_path, capsys):
    """Catches the diagnostic command compiling an implicit or wrong checkpoint."""
    tool = importlib.import_module("tools.ttm_r1_rbln_bisect")
    captured = {}

    def fake_run(model_path, output_dir):
        captured["model_path"] = model_path
        captured["output_dir"] = output_dir
        return output_dir / "rbln-bisect-result.json"

    monkeypatch.setattr(tool, "run_bisection", fake_run)

    assert tool.main(["--model-path", str(tmp_path / "model"), "--output-dir", str(tmp_path / "out")]) == 0
    assert captured == {"model_path": tmp_path / "model", "output_dir": tmp_path / "out"}
    assert "rbln-bisect-result.json" in capsys.readouterr().out
