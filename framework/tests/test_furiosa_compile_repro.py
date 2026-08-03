import json
import builtins
import logging
import sys
from contextlib import nullcontext
from io import StringIO
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


def test_known_signature_prefers_specific_compiler_message():
    from tools import furiosa_compile_repro as repro

    text = """
    furiosa.UnsupportedOpError: failed to compile the graph
    EdgeIndex(162) has empty transition cost table
    """

    assert repro.match_known_signature(text) == (
        "EdgeIndex(162) has empty transition cost table"
    )


def test_safe_error_line_keeps_only_exception_type_and_first_line():
    from tools import furiosa_compile_repro as repro

    exc = RuntimeError("first line\nsecret prompt and path")

    assert repro.safe_error_line(exc) == "RuntimeError: first line"


def test_write_json_serializes_result_contract(tmp_path):
    from tools import furiosa_compile_repro as repro

    result = repro.CaseResult(
        case="resnet50",
        status="failed",
        stages=(repro.StageResult("rngd_first_inference", "failed"),),
        error_type="RuntimeError",
        error_line="RuntimeError: compiler panic",
    )
    output_path = tmp_path / "nested" / "result.json"

    repro.write_json(output_path, result)

    assert json.loads(output_path.read_text()) == {
        "case": "resnet50",
        "status": "failed",
        "stages": [
            {
                "name": "rngd_first_inference",
                "status": "failed",
                "detail": None,
            }
        ],
        "output_shapes": [],
        "error_type": "RuntimeError",
        "error_line": "RuntimeError: compiler panic",
        "matched_known_signature": None,
    }


class _FakeTensor:
    def __init__(self, value, state, *, device="cpu"):
        self.value = np.asarray(value)
        self.state = state
        self.device = device

    @property
    def shape(self):
        return self.value.shape

    @property
    def dtype(self):
        return self.value.dtype

    def to(self, device):
        self.state.device_moves.append((self.device, str(device)))
        return _FakeTensor(self.value.copy(), self.state, device=str(device))

    def detach(self):
        return self

    def cpu(self):
        return _FakeTensor(self.value.copy(), self.state, device="cpu")

    def numpy(self):
        return self.value.copy()


class _FakeModel:
    def __init__(self, state):
        self.state = state
        self.device = "cpu"

    def eval(self):
        return self

    def to(self, device):
        self.device = str(device)
        self.state.model_device = self.device
        return self

    def __call__(self, *inputs):
        if self.device == "cpu":
            self.state.cpu_calls += 1
        else:
            self.state.rngd_calls += 1
            if self.state.first_call_error is not None:
                raise self.state.first_call_error
        return _FakeTensor(
            np.zeros((1, 1000), dtype=np.float32),
            self.state,
            device=self.device,
        )


def _strict_fake_dependencies():
    state = SimpleNamespace(
        cpu_calls=0,
        rngd_calls=0,
        first_call_error=None,
        model_device=None,
        device_moves=[],
        compiler_hints=[],
    )

    class CompilerConfig:
        def __init__(self, *, tactic_hint):
            state.compiler_hints.append(tactic_hint)

    class TacticHintConfig:
        Default = "default"

    class Backend:
        @staticmethod
        def with_config(compiler_config, *, eager_fallback):
            assert compiler_config is not None
            if eager_fallback is not False:
                raise AssertionError("eager fallback must stay disabled")
            return "strict-backend"

    class Torch:
        @staticmethod
        def inference_mode():
            return nullcontext()

        @staticmethod
        def device(value):
            return value

        @staticmethod
        def compile(model, *, backend, fullgraph, dynamic):
            if backend != "strict-backend":
                raise AssertionError("unexpected backend")
            if fullgraph is not True or dynamic is not False:
                raise AssertionError("compile must remain static and fullgraph")
            return model

    dependencies = SimpleNamespace(
        torch=Torch(),
        furiosa_torch=SimpleNamespace(backend=Backend()),
        CompilerConfig=CompilerConfig,
        TacticHintConfig=TacticHintConfig,
    )
    return state, dependencies


def _fake_resnet_definition(repro, state):
    def loader(config, dependencies):
        del config, dependencies
        model = _FakeModel(state)
        inputs = (
            _FakeTensor(
                np.zeros((1, 3, 224, 224), dtype=np.float32),
                state,
            ),
        )
        return model, inputs

    return repro.CaseDefinition(
        expected_shapes=((1, 1000),),
        loader=loader,
    )


def test_run_case_requires_strict_first_call_before_passing(monkeypatch):
    from tools import furiosa_compile_repro as repro

    state, dependencies = _strict_fake_dependencies()
    monkeypatch.setitem(
        repro.CASE_DEFINITIONS,
        "resnet50",
        _fake_resnet_definition(repro, state),
    )

    result = repro.run_case(
        repro.CaseConfig(case="resnet50", model_path=None),
        dependencies=dependencies,
        emit=lambda message: None,
    )

    assert result.status == "passed"
    assert [(stage.name, stage.status) for stage in result.stages] == [
        ("model_load", "passed"),
        ("cpu_first_inference", "passed"),
        ("strict_compile_setup", "passed"),
        ("rngd_first_inference", "passed"),
    ]
    assert result.output_shapes == ((1, 1000),)
    assert state.cpu_calls == 1
    assert state.rngd_calls == 1
    assert state.model_device == "furiosa:0"
    assert state.compiler_hints == ["default"]


def test_run_case_reports_compiler_failure_at_first_rngd_call(monkeypatch):
    from tools import furiosa_compile_repro as repro

    state, dependencies = _strict_fake_dependencies()
    state.first_call_error = RuntimeError("compiler panic")
    monkeypatch.setitem(
        repro.CASE_DEFINITIONS,
        "resnet50",
        _fake_resnet_definition(repro, state),
    )

    result = repro.run_case(
        repro.CaseConfig(case="resnet50", model_path=None),
        dependencies=dependencies,
        emit=lambda message: None,
    )

    assert result.status == "failed"
    assert result.error_type == "RuntimeError"
    assert result.error_line == "RuntimeError: compiler panic"
    assert result.stages[-1].name == "rngd_first_inference"
    assert result.stages[-1].status == "failed"


def test_run_case_writes_full_traceback_to_requested_sink(monkeypatch):
    from tools import furiosa_compile_repro as repro

    state, dependencies = _strict_fake_dependencies()
    state.first_call_error = RuntimeError("compiler panic")
    monkeypatch.setitem(
        repro.CASE_DEFINITIONS,
        "resnet50",
        _fake_resnet_definition(repro, state),
    )
    traceback_sink = StringIO()

    result = repro.run_case(
        repro.CaseConfig(case="resnet50", model_path=None),
        dependencies=dependencies,
        emit=lambda message: None,
        traceback_sink=traceback_sink,
    )

    assert result.status == "failed"
    assert "Traceback (most recent call last)" in traceback_sink.getvalue()
    assert "RuntimeError: compiler panic" in traceback_sink.getvalue()


def test_case_registry_contains_only_the_three_unsupported_models():
    from tools import furiosa_compile_repro as repro

    assert set(repro.CASE_DEFINITIONS) == {"resnet50", "yolov5m", "patchtst"}


def test_fuse_conv_bn_pairs_preserves_cpu_output_and_removes_batchnorm():
    import torch

    from tools import furiosa_compile_repro as repro

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1, bias=False),
        torch.nn.BatchNorm2d(4),
        torch.nn.ReLU(),
    ).eval()
    input_tensor = torch.randn(1, 3, 8, 8)
    with torch.inference_mode():
        expected = model(input_tensor)

    repro.fuse_conv_bn_pairs(model, torch)

    assert sum(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()) == 0
    with torch.inference_mode():
        actual = model(input_tensor)
    torch.testing.assert_close(actual, expected)


def test_resnet_loader_uses_requested_seed_and_imagenet_v2_weights(monkeypatch):
    import torch
    import torchvision.models

    from tools import furiosa_compile_repro as repro

    weights = object()
    seeds = []
    loaded_weights = []
    monkeypatch.setattr(torch, "manual_seed", seeds.append)
    monkeypatch.setattr(
        torchvision.models,
        "ResNet50_Weights",
        SimpleNamespace(IMAGENET1K_V2=weights),
    )

    def resnet50(*, weights):
        loaded_weights.append(weights)
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(4),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )

    monkeypatch.setattr(torchvision.models, "resnet50", resnet50)

    _, inputs = repro.CASE_DEFINITIONS["resnet50"].loader(
        repro.CaseConfig("resnet50", None, seed=17),
        SimpleNamespace(torch=torch),
    )

    assert seeds == [17]
    assert loaded_weights == [weights]
    assert tuple(inputs[0].shape) == (1, 3, 224, 224)


def test_yolov5_loader_rejects_missing_checkpoint_before_import(
    monkeypatch, tmp_path
):
    from tools import furiosa_compile_repro as repro

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] == "ultralytics":
            raise AssertionError("ultralytics imported before path validation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    state, dependencies = _strict_fake_dependencies()
    del state

    with np.testing.assert_raises_regex(FileNotFoundError, "YOLOv5 checkpoint"):
        repro.CASE_DEFINITIONS["yolov5m"].loader(
            repro.CaseConfig("yolov5m", tmp_path / "missing.pt"),
            dependencies,
        )


def test_yolov5_loader_fuses_and_returns_raw_prediction(monkeypatch, tmp_path):
    import torch

    from tools import furiosa_compile_repro as repro

    raw_prediction = torch.zeros(1, 84, 8400)
    auxiliary = [torch.zeros(1, 3, 80, 80, 85)]

    class Base(torch.nn.Module):
        def forward(self, images):
            assert tuple(images.shape) == (1, 3, 640, 640)
            return raw_prediction, auxiliary

    state = SimpleNamespace(fuse_calls=0)

    class YOLO:
        def __init__(self, path):
            assert Path(path).name == "yolov5mu.pt"
            self.model = Base()

        def fuse(self):
            state.fuse_calls += 1
            return self

    module = ModuleType("ultralytics")
    module.YOLO = YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    model_path = tmp_path / "yolov5mu.pt"
    model_path.touch()
    dependencies = SimpleNamespace(torch=torch)

    model, inputs = repro.CASE_DEFINITIONS["yolov5m"].loader(
        repro.CaseConfig("yolov5m", model_path),
        dependencies,
    )

    assert state.fuse_calls == 1
    with torch.inference_mode():
        assert model(*inputs) is raw_prediction


def test_patchtst_loader_ignores_only_model_logger_and_returns_predictions(
    monkeypatch, tmp_path
):
    import torch

    from tools import furiosa_compile_repro as repro

    ignored = set()
    monkeypatch.setattr(
        torch._dynamo.config,
        "ignore_logger_methods",
        ignored,
        raising=False,
    )
    model_logger = logging.getLogger("patchtst-repro-test")
    prediction = torch.zeros(1, 96, 7)
    load_calls = []

    class Base(torch.nn.Module):
        def forward(self, **kwargs):
            assert kwargs["prediction_length"] == 96
            assert kwargs["return_dict"] is True
            return SimpleNamespace(prediction_outputs=prediction)

    class PatchTSTFMForPrediction:
        @classmethod
        def from_pretrained(cls, path, *, local_files_only):
            load_calls.append((Path(path), local_files_only))
            return Base()

    package = ModuleType("tsfm_public")
    models = ModuleType("tsfm_public.models")
    patchtst = ModuleType("tsfm_public.models.patchtst_fm")
    modeling = ModuleType("tsfm_public.models.patchtst_fm.modeling_patchtst_fm")
    modeling.logger = model_logger
    patchtst.PatchTSTFMForPrediction = PatchTSTFMForPrediction
    patchtst.modeling_patchtst_fm = modeling
    monkeypatch.setitem(sys.modules, "tsfm_public", package)
    monkeypatch.setitem(sys.modules, "tsfm_public.models", models)
    monkeypatch.setitem(sys.modules, "tsfm_public.models.patchtst_fm", patchtst)
    monkeypatch.setitem(
        sys.modules,
        "tsfm_public.models.patchtst_fm.modeling_patchtst_fm",
        modeling,
    )
    model_path = tmp_path / "ibm-research_patchtst-fm-r1"
    model_path.mkdir()

    model, inputs = repro.CASE_DEFINITIONS["patchtst"].loader(
        repro.CaseConfig("patchtst", model_path),
        SimpleNamespace(torch=torch),
    )

    assert load_calls == [(model_path, True)]
    assert ignored == {model_logger.info}
    assert [tuple(value.shape) for value in inputs] == [(1, 512, 7), (1, 512, 7)]
    with torch.inference_mode():
        assert model(*inputs) is prediction


def test_reproduction_cli_parser_has_server_safe_defaults():
    from tools import reproduce_furiosa_compile_failures as cli

    args = cli.build_parser().parse_args(["--case", "yolov5m"])

    assert args.case == "yolov5m"
    assert args.device == "furiosa:0"
    assert args.output_dir == Path("results/furiosa-compile-repro")
    assert args.yolov5_path == Path("models/yolov5m/yolov5mu.pt")
    assert args.patchtst_path == Path("models/ibm-research_patchtst-fm-r1")
    assert args.seed == 0


def test_parent_runner_streams_child_output_and_classifies_log(tmp_path, capsys):
    from tools import reproduce_furiosa_compile_failures as cli

    commands = []

    class FakePopen:
        def __init__(self, command, **kwargs):
            commands.append((command, kwargs))
            self.stdout = iter(
                [
                    "[yolov5m] RNGD strict compile + first inference: START\n",
                    "EdgeIndex(162) has empty transition cost table\n",
                ]
            )

        @staticmethod
        def wait():
            return 1

    args = cli.build_parser().parse_args(
        [
            "--case",
            "yolov5m",
            "--output-dir",
            str(tmp_path),
        ]
    )

    exit_code = cli.run_parent(
        args,
        popen_factory=FakePopen,
        timestamp_factory=lambda: "20260803T120000",
    )

    assert exit_code == 1
    log_path = tmp_path / "20260803T120000-yolov5m.log"
    report_path = tmp_path / "20260803T120000-yolov5m.json"
    assert "empty transition cost table" in log_path.read_text()
    report = json.loads(report_path.read_text())
    assert report["case"] == "yolov5m"
    assert report["status"] == "failed"
    assert report["exit_code"] == 1
    assert report["matched_known_signature"] == (
        "EdgeIndex(162) has empty transition cost table"
    )
    assert str(log_path) == report["log_path"]
    assert "empty transition cost table" in capsys.readouterr().out
    command, kwargs = commands[0]
    assert "--_child" in command
    assert kwargs["stderr"] is cli.subprocess.STDOUT


def test_parent_runner_uses_child_signature_when_generic_log_has_none(tmp_path):
    from tools import reproduce_furiosa_compile_failures as cli

    class FakePopen:
        def __init__(self, command, **kwargs):
            del kwargs
            result_index = command.index("--_child-result") + 1
            child_result_path = Path(command[result_index])
            child_result_path.write_text(
                json.dumps(
                    {
                        "invocation": {
                            "case": "patchtst",
                            "device": "furiosa:0",
                            "seed": 0,
                        },
                        "result": {
                            "status": "failed",
                            "matched_known_signature": (
                                "Cannot view a tensor with shape "
                                "torch.Size([7, 512, 16, 64])"
                            ),
                        }
                    }
                )
            )
            self.stdout = iter(["[patchtst] result=failed\n"])

        @staticmethod
        def wait():
            return 1

    args = cli.build_parser().parse_args(
        ["--case", "patchtst", "--output-dir", str(tmp_path)]
    )

    exit_code = cli.run_parent(
        args,
        popen_factory=FakePopen,
        timestamp_factory=lambda: "20260803T121000",
    )

    assert exit_code == 1
    report = json.loads(
        (tmp_path / "20260803T121000-patchtst.json").read_text()
    )
    assert report["matched_known_signature"] == (
        "Cannot view a tensor with shape torch.Size([7, 512, 16, 64])"
    )
    assert report["invocation"] == {
        "case": "patchtst",
        "device": "furiosa:0",
        "seed": 0,
    }


def test_parent_marks_aborted_preflight_child_as_failed(tmp_path):
    from tools import reproduce_furiosa_compile_failures as cli

    class FakePopen:
        def __init__(self, command, **kwargs):
            del kwargs
            result_index = command.index("--_child-result") + 1
            Path(command[result_index]).write_text(
                json.dumps(
                    {
                        "invocation": {"case": "resnet50"},
                        "result": {"case": "resnet50", "status": "running"},
                    }
                )
            )
            self.stdout = iter(["native compiler aborted\n"])

        @staticmethod
        def wait():
            return -6

    args = cli.build_parser().parse_args(
        ["--case", "resnet50", "--output-dir", str(tmp_path)]
    )

    assert cli.run_parent(
        args,
        popen_factory=FakePopen,
        timestamp_factory=lambda: "20260803T122000",
    ) == 1

    report = json.loads(
        (tmp_path / "20260803T122000-resnet50.json").read_text()
    )
    assert report["exit_code"] == -6
    assert report["status"] == "failed"
    assert report["child"]["result"]["status"] == "running"


def test_child_runner_writes_environment_and_case_result(
    monkeypatch, tmp_path
):
    from tools import furiosa_compile_repro as repro
    from tools import reproduce_furiosa_compile_failures as cli

    result = repro.CaseResult(
        case="resnet50",
        status="failed",
        stages=(repro.StageResult("rngd_first_inference", "failed"),),
        error_type="RuntimeError",
        error_line="RuntimeError: compiler panic",
    )
    monkeypatch.setattr(cli, "run_case", lambda config, **kwargs: result)
    monkeypatch.setattr(
        cli,
        "collect_environment",
        lambda: {"python": "3.12.13", "furiosa_torch": "2026.3.0"},
    )
    result_path = tmp_path / "child.json"
    args = cli.build_parser().parse_args(
        [
            "--case",
            "resnet50",
            "--_child",
            "--_child-result",
            str(result_path),
        ]
    )

    exit_code = cli.run_child(args)

    assert exit_code == 1
    payload = json.loads(result_path.read_text())
    assert payload["environment"] == {
        "python": "3.12.13",
        "furiosa_torch": "2026.3.0",
    }
    assert payload["result"]["case"] == "resnet50"
    assert payload["result"]["status"] == "failed"


def test_child_runner_persists_evidence_before_native_execution(
    monkeypatch, tmp_path
):
    from tools import furiosa_compile_repro as repro
    from tools import reproduce_furiosa_compile_failures as cli

    result_path = tmp_path / "child.json"
    monkeypatch.setattr(
        cli,
        "collect_environment",
        lambda: {"furiosa_torch": "2026.3.0"},
    )
    monkeypatch.setattr(
        cli,
        "build_invocation_evidence",
        lambda config: {
            "case": config.case,
            "device": config.device,
            "seed": config.seed,
        },
    )

    def inspect_preflight(config, **kwargs):
        del kwargs
        preflight = json.loads(result_path.read_text())
        assert preflight["environment"] == {"furiosa_torch": "2026.3.0"}
        assert preflight["invocation"] == {
            "case": "resnet50",
            "device": "furiosa:0",
            "seed": 0,
        }
        assert preflight["result"]["status"] == "running"
        return repro.CaseResult(
            case=config.case,
            status="failed",
            stages=(repro.StageResult("rngd_first_inference", "failed"),),
        )

    monkeypatch.setattr(cli, "run_case", inspect_preflight)
    args = cli.build_parser().parse_args(
        [
            "--case",
            "resnet50",
            "--_child",
            "--_child-result",
            str(result_path),
        ]
    )

    assert cli.run_child(args) == 1


def test_invocation_evidence_identifies_model_input_and_revision(
    monkeypatch, tmp_path
):
    from tools import furiosa_compile_repro as repro
    from tools import reproduce_furiosa_compile_failures as cli

    model_path = tmp_path / "yolov5mu.pt"
    model_path.touch()
    monkeypatch.setattr(cli, "_git_revision", lambda: "deadbeef")

    evidence = cli.build_invocation_evidence(
        repro.CaseConfig(
            case="yolov5m",
            model_path=model_path,
            device="furiosa:3",
            seed=17,
        )
    )

    assert evidence == {
        "case": "yolov5m",
        "model_path": str(model_path.resolve()),
        "device": "furiosa:3",
        "seed": 17,
        "git_revision": "deadbeef",
        "inputs": [
            {
                "name": "images",
                "shape": [1, 3, 640, 640],
                "dtype": "float32",
            }
        ],
    }


def test_child_runner_writes_prerequisite_failure_when_sdk_import_fails(
    monkeypatch, tmp_path
):
    from tools import reproduce_furiosa_compile_failures as cli

    def missing_sdk(config, **kwargs):
        del config, kwargs
        raise ImportError("furiosa.torch is not installed")

    monkeypatch.setattr(cli, "run_case", missing_sdk)
    monkeypatch.setattr(cli, "collect_environment", lambda: {"furiosa_torch": None})
    result_path = tmp_path / "missing-sdk.json"
    args = cli.build_parser().parse_args(
        [
            "--case",
            "resnet50",
            "--_child",
            "--_child-result",
            str(result_path),
        ]
    )

    exit_code = cli.run_child(args)

    assert exit_code == 1
    payload = json.loads(result_path.read_text())
    assert payload["result"] == {
        "case": "resnet50",
        "status": "failed",
        "stages": [
            {
                "name": "prerequisites",
                "status": "failed",
                "detail": "ImportError: furiosa.torch is not installed",
            }
        ],
        "output_shapes": [],
        "error_type": "ImportError",
        "error_line": "ImportError: furiosa.torch is not installed",
        "matched_known_signature": None,
    }
