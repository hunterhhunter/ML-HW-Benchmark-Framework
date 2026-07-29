from pathlib import Path
import importlib
import json
import os
import re
import subprocess
import sys
import types

import pytest

from tools.rbln_compile_recipes.common import (
    RecipeContract,
    TensorContract,
    contract_to_dict,
    create_parser,
    emit_description_or_require_output,
    prepare_output_path,
    save_and_validate,
)


def _contract():
    return RecipeContract(
        recipe="unit",
        model_id="owner/model",
        inputs=(TensorContract("x", (1, 3), "float32"),),
        outputs=(TensorContract("y", (1, 2), "float32"),),
        allow_unnamed_outputs=True,
        notes=("fixed batch one",),
    )


def test_contract_description_is_json_safe_and_stable():
    assert contract_to_dict(_contract()) == {
        "recipe": "unit",
        "model_id": "owner/model",
        "target_npu": "RBLN-CA22",
        "inputs": [{"name": "x", "shape": [1, 3], "dtype": "float32"}],
        "outputs": [{"name": "y", "shape": [1, 2], "dtype": "float32"}],
        "allow_unnamed_outputs": True,
        "notes": ["fixed batch one"],
    }


def test_prepare_output_path_requires_rbln_and_refuses_overwrite(tmp_path):
    with pytest.raises(ValueError, match=".rbln"):
        prepare_output_path(tmp_path / "model.bin")
    existing = tmp_path / "model.rbln"
    existing.write_bytes(b"do-not-overwrite")
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_output_path(existing)


def test_describe_emits_contract_before_requiring_an_output(capsys):
    args = create_parser().parse_args(["--describe"])

    assert emit_description_or_require_output(args, _contract()) is None
    assert json.loads(capsys.readouterr().out) == contract_to_dict(_contract())


def test_compile_requires_an_explicit_output_after_description_is_not_requested():
    args = create_parser().parse_args([])

    with pytest.raises(ValueError, match="--output"):
        emit_description_or_require_output(args, _contract())


def _inspection(*, npu="RBLN-CA22", inputs=None, outputs=None):
    return {
        "npu": npu,
        "compiler_version": "0.11.0",
        "inputs": inputs
        if inputs is not None
        else [{"name": "x", "shape": [1, 3], "dtype": "float32"}],
        "outputs": outputs
        if outputs is not None
        else [{"name": None, "shape": [1, 2], "dtype": "float32"}],
    }


class _Compiled:
    def __init__(self, content=b"compiled"):
        self.content = content
        self.save_calls = 0

    def save(self, path):
        self.save_calls += 1
        Path(path).write_bytes(self.content)


def _inject_rebel(monkeypatch, inspection):
    fake_rebel = types.SimpleNamespace(
        RBLNCompiledModel=types.SimpleNamespace(inspect=lambda path: inspection)
    )
    monkeypatch.setitem(sys.modules, "rebel", fake_rebel)


def test_save_and_validate_accepts_mapping_inspect_and_reports_sha(monkeypatch, tmp_path):
    compiled = _Compiled()
    _inject_rebel(monkeypatch, _inspection())

    report = save_and_validate(compiled, tmp_path / "model.rbln", _contract())

    assert compiled.save_calls == 1
    assert report["size_bytes"] == 8
    assert len(report["sha256"]) == 64
    assert report["compiler_version"] == "0.11.0"


def test_save_and_validate_accepts_attribute_descriptors(monkeypatch, tmp_path):
    descriptor = lambda name, shape: types.SimpleNamespace(
        name=name, shape=shape, dtype="FLOAT32"
    )
    inspection = types.SimpleNamespace(
        npu="RBLN-CA22",
        compiler_version="0.11.0",
        inputs=[descriptor("x", (1, 3))],
        outputs=[descriptor("y", (1, 2))],
    )
    _inject_rebel(monkeypatch, inspection)

    report = save_and_validate(_Compiled(), tmp_path / "attribute.rbln", _contract())

    assert report["target_npu"] == "RBLN-CA22"


def test_save_and_validate_rejects_unnamed_output_without_permission(monkeypatch, tmp_path):
    contract = RecipeContract(
        recipe="unit",
        model_id="owner/model",
        inputs=(TensorContract("x", (1, 3), "float32"),),
        outputs=(TensorContract("y", (1, 2), "float32"),),
    )
    _inject_rebel(monkeypatch, _inspection())

    with pytest.raises(ValueError, match="output.*name"):
        save_and_validate(_Compiled(), tmp_path / "unnamed.rbln", contract)


def test_save_and_validate_rejects_output_count_mismatch(monkeypatch, tmp_path):
    _inject_rebel(monkeypatch, _inspection(outputs=[]))

    with pytest.raises(ValueError, match="output count"):
        save_and_validate(_Compiled(), tmp_path / "count.rbln", _contract())


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        ([{"name": None, "shape": [1, 3], "dtype": "float32"}], "shape"),
        ([{"name": None, "shape": [1, 2], "dtype": "float16"}], "dtype"),
    ],
)
def test_save_and_validate_rejects_output_abi_mismatch(
    monkeypatch, tmp_path, outputs, message
):
    _inject_rebel(monkeypatch, _inspection(outputs=outputs))

    with pytest.raises(ValueError, match=message):
        save_and_validate(_Compiled(), tmp_path / "mismatch.rbln", _contract())


def test_save_and_validate_rejects_input_name_mismatch(monkeypatch, tmp_path):
    _inject_rebel(
        monkeypatch,
        _inspection(inputs=[{"name": "wrong", "shape": [1, 3], "dtype": "float32"}]),
    )

    with pytest.raises(ValueError, match="input.*name"):
        save_and_validate(_Compiled(), tmp_path / "input-name.rbln", _contract())


def test_save_and_validate_rejects_non_batch_one_contract(monkeypatch, tmp_path):
    contract = RecipeContract(
        recipe="unit",
        model_id="owner/model",
        inputs=(TensorContract("x", (2, 3), "float32"),),
        outputs=(TensorContract("y", (2, 2), "float32"),),
        allow_unnamed_outputs=True,
    )
    _inject_rebel(
        monkeypatch,
        _inspection(
            inputs=[{"name": "x", "shape": [2, 3], "dtype": "float32"}],
            outputs=[{"name": None, "shape": [2, 2], "dtype": "float32"}],
        ),
    )

    with pytest.raises(ValueError, match="batch size 1"):
        save_and_validate(_Compiled(), tmp_path / "batch-two.rbln", contract)


def test_save_and_validate_rejects_npu_mismatch(monkeypatch, tmp_path):
    _inject_rebel(monkeypatch, _inspection(npu="RBLN-CA12"))

    with pytest.raises(ValueError, match="NPU"):
        save_and_validate(_Compiled(), tmp_path / "npu.rbln", _contract())


def test_save_and_validate_rejects_non_ca22_contract(monkeypatch, tmp_path):
    contract = RecipeContract(
        recipe="unit",
        model_id="owner/model",
        inputs=(TensorContract("x", (1, 3), "float32"),),
        outputs=(TensorContract("y", (1, 2), "float32"),),
        allow_unnamed_outputs=True,
        target_npu="RBLN-CA12",
    )
    _inject_rebel(monkeypatch, _inspection(npu="RBLN-CA12"))

    with pytest.raises(ValueError, match="RBLN-CA22"):
        save_and_validate(_Compiled(), tmp_path / "ca12.rbln", contract)


def test_save_and_validate_rejects_zero_byte_artifact_without_sdk(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "rebel", raising=False)

    with pytest.raises(ValueError, match="empty"):
        save_and_validate(_Compiled(b""), tmp_path / "empty.rbln", _contract())


@pytest.mark.parametrize(
    ("module", "model_id", "input_names", "output_shape"),
    [
        (
            "tools.rbln_compile_recipes.resnet50.compile",
            "torchvision/resnet50-imagenet1k-v2",
            ["input_np"],
            [1, 1000],
        ),
        (
            "tools.rbln_compile_recipes.bert_sst2.compile",
            "textattack/bert-base-uncased-SST-2",
            ["input_ids", "attention_mask"],
            [1, 2],
        ),
        (
            "tools.rbln_compile_recipes.patchtst_etth1.compile",
            "ibm-granite/granite-timeseries-patchtst",
            ["past_values", "past_observed_mask"],
            [1, 96, 7],
        ),
    ],
)
def test_recipe_describe_needs_no_optional_sdk(
    module, model_id, input_names, output_shape
):
    result = subprocess.run(
        [sys.executable, "-m", module, "--describe"],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["model_id"] == model_id
    assert [item["name"] for item in payload["inputs"]] == input_names
    assert payload["outputs"][0]["shape"] == output_shape


@pytest.mark.parametrize(
    "module",
    [
        "tools.rbln_compile_recipes.resnet50.compile",
        "tools.rbln_compile_recipes.bert_sst2.compile",
        "tools.rbln_compile_recipes.bert_squad.compile",
        "tools.rbln_compile_recipes.yolov5m.compile",
        "tools.rbln_compile_recipes.patchtst_etth1.compile",
    ],
)
def test_recipe_help_does_not_import_optional_sdks(module, tmp_path):
    (tmp_path / "sitecustomize.py").write_text(
        """import builtins

_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split('.', 1)[0] in {'models', 'rebel', 'torch', 'torchvision', 'transformers'}:
        raise RuntimeError(f'optional SDK imported: {name}')
    return _import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
"""
    )
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join(
            filter(None, (str(tmp_path), os.environ.get("PYTHONPATH")))
        )
    }

    subprocess.run(
        [sys.executable, "-m", module, "--help"],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )


def test_bert_squad_describe_has_three_inputs_and_two_ordered_outputs():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.rbln_compile_recipes.bert_squad.compile",
            "--describe",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["model_id"] == "csarron/bert-base-uncased-squad-v1"
    assert payload["inputs"] == [
        {"name": "input_ids", "shape": [1, 384], "dtype": "int64"},
        {"name": "attention_mask", "shape": [1, 384], "dtype": "int64"},
        {"name": "token_type_ids", "shape": [1, 384], "dtype": "int64"},
    ]
    assert payload["outputs"] == [
        {"name": "start_logits", "shape": [1, 384], "dtype": "float32"},
        {"name": "end_logits", "shape": [1, 384], "dtype": "float32"},
    ]
    assert payload["allow_unnamed_outputs"] is True
    assert any("CPU/NPU" in note and "mapping" in note for note in payload["notes"])


def test_bert_squad_compiles_three_inputs_and_preserves_output_order(monkeypatch):
    module = importlib.import_module(
        "tools.rbln_compile_recipes.bert_squad.compile"
    )
    observed = {}

    class FakeModule:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def eval(self):
            return self

        def requires_grad_(self, value):
            observed["requires_grad"] = value
            return self

    class FakeQuestionAnsweringModel:
        def eval(self):
            return self

        def __call__(self, **kwargs):
            observed["model_kwargs"] = kwargs
            return types.SimpleNamespace(
                start_logits="start-logits",
                end_logits="end-logits",
            )

    fake_transformers = types.SimpleNamespace(
        AutoModelForQuestionAnswering=types.SimpleNamespace(
            from_pretrained=lambda model_id: observed.setdefault(
                "model_id", model_id
            )
            and FakeQuestionAnsweringModel()
        )
    )

    def compile_from_torch(model, input_info):
        observed["outputs"] = model("ids", "mask", "types")
        observed["input_info"] = input_info
        return "compiled-model"

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(nn=types.SimpleNamespace(Module=FakeModule)),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(
        sys.modules,
        "rebel",
        types.SimpleNamespace(compile_from_torch=compile_from_torch),
    )

    compiled = module.compile_model("owner/qa-model")

    assert compiled == "compiled-model"
    assert observed["model_id"] == "owner/qa-model"
    assert observed["input_info"] == [
        ("input_ids", [1, 384], "int64"),
        ("attention_mask", [1, 384], "int64"),
        ("token_type_ids", [1, 384], "int64"),
    ]
    assert observed["outputs"] == ("start-logits", "end-logits")
    assert observed["model_kwargs"] == {
        "input_ids": "ids",
        "attention_mask": "mask",
        "token_type_ids": "types",
        "return_dict": True,
    }
    assert observed["requires_grad"] is False


def test_patchtst_describe_preserves_bool_mask_artifact_abi():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.rbln_compile_recipes.patchtst_etth1.compile",
            "--describe",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["inputs"] == [
        {"name": "past_values", "shape": [1, 512, 7], "dtype": "float32"},
        {
            "name": "past_observed_mask",
            "shape": [1, 512, 7],
            "dtype": "bool",
        },
    ]
    assert payload["outputs"] == [
        {
            "name": "prediction_outputs",
            "shape": [1, 96, 7],
            "dtype": "float32",
        }
    ]
    assert any("aten::unfold" in note for note in payload["notes"])


def test_static_patchify_matches_unfold_without_aten_unfold():
    torch = pytest.importorskip("torch")
    module = importlib.import_module(
        "tools.rbln_compile_recipes.patchtst_etth1.compile"
    )
    values = torch.arange(512 * 7, dtype=torch.float32).reshape(1, 512, 7)
    expected = values[:, 8:, :].unfold(1, 12, 12).transpose(2, 3).contiguous()

    actual = module.static_patchify(values)

    assert actual.shape == (1, 42, 12, 7)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    replacement = module.build_static_patchifier(torch)
    channel_first = replacement(values)
    torch.testing.assert_close(
        channel_first,
        expected.permute(0, 3, 1, 2).contiguous(),
        rtol=0,
        atol=0,
    )
    traced = torch.jit.trace(replacement, (values,))
    assert "aten::unfold" not in str(traced.inlined_graph)


def test_patchtst_compile_runs_equivalence_gates_and_uses_jittrace(monkeypatch):
    torch = pytest.importorskip("torch")
    module = importlib.import_module(
        "tools.rbln_compile_recipes.patchtst_etth1.compile"
    )
    observed = {}

    class OriginalPatchifier(torch.nn.Module):
        def forward(self, values):
            return values[:, 8:, :].unfold(1, 12, 12).transpose(1, 2).contiguous()

    class FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patchifier = OriginalPatchifier()

    class FakePatchTST(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeBackbone()
            self.config = types.SimpleNamespace(
                context_length=512,
                prediction_length=96,
                num_input_channels=7,
                patch_length=12,
                patch_stride=12,
            )

        def forward(self, past_values, past_observed_mask, return_dict=True):
            assert return_dict is True
            masked = past_values * past_observed_mask
            patches = self.model.patchifier(masked)
            prediction = patches.mean(dim=(2, 3)).unsqueeze(1).expand(-1, 96, -1)
            return types.SimpleNamespace(prediction_outputs=prediction)

    fake_model = FakePatchTST().eval()
    fake_transformers = types.SimpleNamespace(
        PatchTSTForPrediction=types.SimpleNamespace(
            from_pretrained=lambda model_id: observed.setdefault("model_id", model_id)
            and fake_model
        )
    )

    def compile_from_torch(model, input_info, model_trace_method):
        observed["input_info"] = input_info
        observed["model_trace_method"] = model_trace_method
        observed["graph"] = str(
            torch.jit.trace(
                model,
                (
                    torch.zeros((1, 512, 7), dtype=torch.float32),
                    torch.ones((1, 512, 7), dtype=torch.bool),
                ),
            ).inlined_graph
        )
        return "compiled-model"

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(
        sys.modules,
        "rebel",
        types.SimpleNamespace(compile_from_torch=compile_from_torch),
    )

    compiled = module.compile_model("owner/patchtst")

    assert compiled == "compiled-model"
    assert observed["model_id"] == "owner/patchtst"
    assert observed["input_info"] == [
        ("past_values", [1, 512, 7], "float32"),
        ("past_observed_mask", [1, 512, 7], "bool"),
    ]
    assert observed["model_trace_method"] == "jittrace"
    assert "aten::unfold" not in observed["graph"]


def test_yolov5_preflight_names_missing_root_and_weight(tmp_path):
    module = importlib.import_module("tools.rbln_compile_recipes.yolov5m.compile")

    with pytest.raises(FileNotFoundError, match="YOLOv5 source root"):
        module.validate_sources(tmp_path / "missing", tmp_path / "yolov5m.pt")


def test_yolov5_preflight_rejects_empty_weight_before_git(monkeypatch, tmp_path):
    module = importlib.import_module("tools.rbln_compile_recipes.yolov5m.compile")
    yolov5_root = tmp_path / "yolov5"
    (yolov5_root / "models").mkdir(parents=True)
    (yolov5_root / "models" / "experimental.py").touch()
    (yolov5_root / "models" / "yolo.py").touch()
    weights = tmp_path / "yolov5m.pt"
    weights.touch()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(
            stdout=module.EXPECTED_YOLOV5_REVISION + "\n"
        ),
    )

    with pytest.raises(ValueError, match="YOLOv5 weight file is empty"):
        module.validate_sources(yolov5_root, weights)


def test_yolov5_describe_records_pinned_revision():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.rbln_compile_recipes.yolov5m.compile",
            "--describe",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert "86fd1ab270cb2f7e53ee7412cd4a0650bf4bcc51" in payload["notes"]
    assert payload["outputs"][0]["shape"] == [1, 25200, 85]


def test_yolov5_preflight_uses_exact_pinned_revision_command(monkeypatch, tmp_path):
    module = importlib.import_module("tools.rbln_compile_recipes.yolov5m.compile")
    yolov5_root = tmp_path / "yolov5"
    (yolov5_root / "models").mkdir(parents=True)
    (yolov5_root / "models" / "experimental.py").touch()
    (yolov5_root / "models" / "yolo.py").touch()
    weights = tmp_path / "yolov5m.pt"
    weights.write_bytes(b"weights")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return types.SimpleNamespace(stdout=module.EXPECTED_YOLOV5_REVISION + "\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module.validate_sources(yolov5_root, weights)

    assert calls == [
        (
            ["git", "-C", str(yolov5_root), "rev-parse", "HEAD"],
            {"check": True, "text": True, "capture_output": True},
        )
    ]


def test_yolov5_preflight_rejects_unpinned_revision_with_checkout_command(
    monkeypatch, tmp_path
):
    module = importlib.import_module("tools.rbln_compile_recipes.yolov5m.compile")
    yolov5_root = tmp_path / "yolov5"
    (yolov5_root / "models").mkdir(parents=True)
    (yolov5_root / "models" / "experimental.py").touch()
    (yolov5_root / "models" / "yolo.py").touch()
    weights = tmp_path / "yolov5m.pt"
    weights.write_bytes(b"weights")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(stdout="different-revision\n"),
    )

    with pytest.raises(RuntimeError, match=rf"checkout {module.EXPECTED_YOLOV5_REVISION}"):
        module.validate_sources(yolov5_root, weights)


def test_compilation_runbook_references_every_recipe_and_llama_tool():
    text = Path("docs/rbln-compilation.md").read_text(encoding="utf-8")
    for module in (
        "tools.rbln_compile_recipes.resnet50.compile",
        "tools.rbln_compile_recipes.yolov5m.compile",
        "tools.rbln_compile_recipes.bert_sst2.compile",
        "tools.rbln_compile_recipes.bert_squad.compile",
        "tools.rbln_compile_recipes.patchtst_etth1.compile",
    ):
        assert module in text
    assert text.count("tools/prepare_rbln_vllm_model.py") >= 2
    assert "unsupported_single_npu_experiment" in text


def test_compilation_runbook_is_linked_from_operator_entrypoints():
    entrypoints = (
        Path("docs/rbln-setup.md"),
        Path("docs/rbln-vllm-setup.md"),
        Path("README.md"),
        Path("..").resolve() / "README.md",
    )

    for path in entrypoints:
        assert "rbln-compilation.md" in path.read_text(encoding="utf-8"), path


def _runbook_shell_function(name):
    text = Path("docs/rbln-compilation.md").read_text(encoding="utf-8")
    match = re.search(rf"(?ms)^{name}\(\) \{{\n.*?^\}}\n", text)
    assert match is not None
    return match.group(0)


def test_copy_verified_rejects_existing_destination_without_overwriting(tmp_path):
    source = tmp_path / "source.rbln"
    destination = tmp_path / "destination.rbln"
    source.write_bytes(b"new artifact")
    destination.write_bytes(b"existing artifact")

    result = subprocess.run(
        [
            "bash",
            "-c",
            _runbook_shell_function("copy_verified")
            + '\ncopy_verified "$1" "$2"',
            "copy-verified-test",
            str(source),
            str(destination),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert destination.read_bytes() == b"existing artifact"


def test_copy_verified_rejects_missing_source_without_creating_destination(tmp_path):
    source = tmp_path / "missing.rbln"
    destination = tmp_path / "destination.rbln"

    result = subprocess.run(
        [
            "bash",
            "-c",
            _runbook_shell_function("copy_verified")
            + '\ncopy_verified "$1" "$2"',
            "copy-verified-test",
            str(source),
            str(destination),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert not destination.exists()


def _write_fake_command(directory, name, body):
    directory.mkdir(exist_ok=True)
    command = directory / name
    command.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    command.chmod(0o755)
    return command


def test_copy_verified_atomically_rejects_destination_created_during_publish(tmp_path):
    source = tmp_path / "source.rbln"
    destination = tmp_path / "destination.rbln"
    source.write_bytes(b"new artifact")
    fake_bin = tmp_path / "bin"
    _write_fake_command(
        fake_bin,
        "ln",
        'destination="${@: -1}"\n'
        'printf "racing destination" > "$destination"\n'
        'exec /usr/bin/ln "$@"\n',
    )
    environment = os.environ | {
        "PATH": os.pathsep.join((str(fake_bin), os.environ["PATH"]))
    }

    result = subprocess.run(
        [
            "bash",
            "-c",
            _runbook_shell_function("copy_verified")
            + '\ncopy_verified "$1" "$2"',
            "copy-verified-test",
            str(source),
            str(destination),
        ],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode != 0
    assert destination.read_bytes() == b"racing destination"
    assert list(tmp_path.glob(f".{destination.name}.copy.*")) == []


def test_copy_verified_cleans_temp_and_does_not_publish_on_hash_mismatch(tmp_path):
    source = tmp_path / "source.rbln"
    destination = tmp_path / "destination.rbln"
    source.write_bytes(b"new artifact")
    fake_bin = tmp_path / "bin"
    _write_fake_command(
        fake_bin,
        "sha256sum",
        'path="${@: -1}"\n'
        'if [[ "$path" == *".copy."* ]]; then\n'
        '  printf "%064d  %s\\n" 0 "$path"\n'
        '  exit 0\n'
        'fi\n'
        'exec /usr/bin/sha256sum "$@"\n',
    )
    environment = os.environ | {
        "PATH": os.pathsep.join((str(fake_bin), os.environ["PATH"]))
    }

    result = subprocess.run(
        [
            "bash",
            "-c",
            _runbook_shell_function("copy_verified")
            + '\ncopy_verified "$1" "$2"',
            "copy-verified-test",
            str(source),
            str(destination),
        ],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode != 0
    assert not destination.exists()
    assert list(tmp_path.glob(f".{destination.name}.copy.*")) == []


def test_copy_verified_rejects_sha256sum_failure_without_publishing(tmp_path):
    source = tmp_path / "source.rbln"
    destination = tmp_path / "destination.rbln"
    source.write_bytes(b"new artifact")
    fake_bin = tmp_path / "bin"
    _write_fake_command(fake_bin, "sha256sum", "exit 23\n")
    environment = os.environ | {
        "PATH": os.pathsep.join((str(fake_bin), os.environ["PATH"]))
    }

    result = subprocess.run(
        [
            "bash",
            "-c",
            _runbook_shell_function("copy_verified")
            + '\ncopy_verified "$1" "$2"',
            "copy-verified-test",
            str(source),
            str(destination),
        ],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode != 0
    assert not destination.exists()
    assert list(tmp_path.glob(f".{destination.name}.copy.*")) == []


def test_copy_verified_rejects_malformed_digest_without_publishing(tmp_path):
    source = tmp_path / "source.rbln"
    destination = tmp_path / "destination.rbln"
    source.write_bytes(b"new artifact")
    fake_bin = tmp_path / "bin"
    _write_fake_command(
        fake_bin,
        "sha256sum",
        'printf "not-a-sha256  %s\\n" "$1"\n',
    )
    environment = os.environ | {
        "PATH": os.pathsep.join((str(fake_bin), os.environ["PATH"]))
    }

    result = subprocess.run(
        [
            "bash",
            "-c",
            _runbook_shell_function("copy_verified")
            + '\ncopy_verified "$1" "$2"',
            "copy-verified-test",
            str(source),
            str(destination),
        ],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode != 0
    assert not destination.exists()
    assert list(tmp_path.glob(f".{destination.name}.copy.*")) == []


def test_copy_verified_rejects_destination_directory_race_without_linking_inside(
    tmp_path,
):
    source = tmp_path / "source.rbln"
    destination = tmp_path / "destination.rbln"
    source.write_bytes(b"new artifact")
    fake_bin = tmp_path / "bin"
    _write_fake_command(
        fake_bin,
        "ln",
        'destination="${@: -1}"\n'
        'mkdir -- "$destination"\n'
        'exec /usr/bin/ln "$@"\n',
    )
    environment = os.environ | {
        "PATH": os.pathsep.join((str(fake_bin), os.environ["PATH"]))
    }

    result = subprocess.run(
        [
            "bash",
            "-c",
            _runbook_shell_function("copy_verified")
            + '\ncopy_verified "$1" "$2"',
            "copy-verified-test",
            str(source),
            str(destination),
        ],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode != 0
    assert destination.is_dir()
    assert list(destination.iterdir()) == []
    assert list(tmp_path.glob(f".{destination.name}.copy.*")) == []


def test_copy_verified_rejects_destination_symlink_race_without_following_it(tmp_path):
    source = tmp_path / "source.rbln"
    destination = tmp_path / "destination.rbln"
    symlink_target = tmp_path / "symlink-target"
    source.write_bytes(b"new artifact")
    symlink_target.mkdir()
    fake_bin = tmp_path / "bin"
    _write_fake_command(
        fake_bin,
        "ln",
        'destination="${@: -1}"\n'
        '/usr/bin/ln -s -- "$RACE_TARGET" "$destination"\n'
        'exec /usr/bin/ln "$@"\n',
    )
    environment = os.environ | {
        "PATH": os.pathsep.join((str(fake_bin), os.environ["PATH"])),
        "RACE_TARGET": str(symlink_target),
    }

    result = subprocess.run(
        [
            "bash",
            "-c",
            _runbook_shell_function("copy_verified")
            + '\ncopy_verified "$1" "$2"',
            "copy-verified-test",
            str(source),
            str(destination),
        ],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode != 0
    assert destination.is_symlink()
    assert destination.resolve() == symlink_target
    assert list(symlink_target.iterdir()) == []
    assert list(tmp_path.glob(f".{destination.name}.copy.*")) == []


def test_squad_mapping_script_records_all_assignment_metrics():
    text = Path("docs/rbln-setup.md").read_text(encoding="utf-8")

    assert 'print_metrics("direct assignment", direct)' in text
    assert 'print_metrics("swapped assignment", swapped)' in text
    for key in ('"mae"', '"rmse"', '"corr"'):
        assert key in text
