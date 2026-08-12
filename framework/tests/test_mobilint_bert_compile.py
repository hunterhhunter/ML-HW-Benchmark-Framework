from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertForQuestionAnswering, BertForSequenceClassification


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK_ROOT))
sys.path.insert(0, str(FRAMEWORK_ROOT / "src"))

from tools.mobilint_bert_compile.common import (
    TASK_SPECS,
    TensorContract,
    contract_to_dict,
    extract_embedding_weights,
    get_task_spec,
    make_compiler_model,
    sha256_file,
)
from tools.mobilint_bert_compile import prepare as prepare_recipe
from tools.mobilint_bert_compile import compile as compile_recipe


def _calibration_artifacts(task_root, paths):
    return [
        {
            "path": path.relative_to(task_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def _bert_config(*, num_labels=2):
    return BertConfig(
        vocab_size=17,
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=8,
        max_position_embeddings=16,
        type_vocab_size=2,
        num_labels=num_labels,
    )


def test_task_contracts_describe_the_reproduced_compiler_boundary():
    sst2 = contract_to_dict(get_task_spec("sst2"))
    squad1 = contract_to_dict(get_task_spec("squad1"))

    assert sst2 == {
        "task": "sst2",
        "model_id": "textattack/bert-base-uncased-SST-2",
        "dataset": {
            "name": "glue",
            "config": "sst2",
            "split": "validation",
        },
        "max_length": 128,
        "calibration_samples": 32,
        "target_device": "aries-rb",
        "compiler_inputs": [
            {"name": "input_ids", "shape": [1, -1], "dtype": "int64"},
            {"name": "attention_mask", "shape": [1, -1], "dtype": "int64"},
            {"name": "token_type_ids", "shape": [1, -1], "dtype": "int64"},
        ],
        "mxq_inputs": [
            {"name": "embeddings", "shape": [1, -1, 768], "dtype": "float32"}
        ],
        "source_outputs": ["logits"],
        "verified_runtime_outputs": ["logits"],
    }
    assert squad1["model_id"] == "csarron/bert-base-uncased-squad-v1"
    assert squad1["dataset"] == {
        "name": "squad",
        "config": None,
        "split": "validation",
    }
    assert squad1["max_length"] == 384
    assert squad1["source_outputs"] == ["start_logits", "end_logits"]
    assert squad1["verified_runtime_outputs"] == ["end_logits", "start_logits"]


def test_unknown_compile_task_is_rejected():
    with pytest.raises(ValueError, match="unsupported Mobilint BERT compile task"):
        get_task_spec("squad2")


def test_task_contract_is_immutable():
    spec = get_task_spec("sst2")

    with pytest.raises((AttributeError, TypeError)):
        spec.max_length = 64
    with pytest.raises(TypeError):
        TASK_SPECS["replacement"] = spec


def test_extract_embedding_weights_copies_the_finetuned_model_boundary():
    model = BertForSequenceClassification(_bert_config())

    weights = extract_embedding_weights(model)

    assert set(weights) == {
        "word_embeddings",
        "token_type_embeddings",
        "position_embeddings",
        "layernorm_weight",
        "layernorm_bias",
    }
    assert tuple(weights["word_embeddings"].shape) == (17, 4)
    assert tuple(weights["token_type_embeddings"].shape) == (2, 4)
    assert tuple(weights["position_embeddings"].shape) == (16, 4)
    assert tuple(weights["layernorm_weight"].shape) == (4,)
    assert all(value.device.type == "cpu" for value in weights.values())
    assert all(not value.requires_grad for value in weights.values())


def test_squad_compiler_wrapper_preserves_original_logits_bitwise():
    original = BertForQuestionAnswering(_bert_config()).eval()
    compiler_model = make_compiler_model("squad1", original).eval()
    inputs = {
        "input_ids": torch.tensor([[1, 4, 7, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "token_type_ids": torch.tensor([[0, 0, 1, 1]]),
    }

    with torch.no_grad():
        expected = original(**inputs)
        actual = compiler_model(**inputs)

    assert tuple(actual) == ("start_logits", "end_logits")
    assert actual.__class__ is dict
    assert torch.equal(actual["start_logits"], expected.start_logits)
    assert torch.equal(actual["end_logits"], expected.end_logits)
    assert compiler_model.__class__.__module__ == (
        "tools.mobilint_bert_compile.compiler_models"
    )


def test_sst2_compiler_model_keeps_the_original_model():
    original = BertForSequenceClassification(_bert_config()).eval()

    assert make_compiler_model("sst2", original) is original


def test_sha256_file_reports_binary_content(tmp_path):
    artifact = tmp_path / "model.mxq"
    artifact.write_bytes(b"compiled")

    assert sha256_file(artifact) == hashlib.sha256(b"compiled").hexdigest()
    assert json.loads(json.dumps(contract_to_dict(get_task_spec("sst2"))))


def test_calibration_indices_are_deterministic_and_cover_the_split():
    first = prepare_recipe.select_calibration_indices(101, 32)
    second = prepare_recipe.select_calibration_indices(101, 32)

    assert first == second
    assert len(first) == 32
    assert first[0] == 0
    assert first[-1] == 100
    assert tuple(sorted(set(first))) == first


def test_calibration_indices_reject_invalid_counts():
    with pytest.raises(ValueError, match="positive"):
        prepare_recipe.select_calibration_indices(10, 0)
    with pytest.raises(ValueError, match="only 3"):
        prepare_recipe.select_calibration_indices(3, 4)


class _TinyTokenizer:
    def __call__(self, *texts, return_tensors, max_length, truncation):
        assert return_tensors == "pt"
        assert max_length == 8
        assert truncation is True
        length = 3 + sum(len(str(text)) for text in texts) % 3
        return {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5][:length]]),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
            "token_type_ids": torch.zeros((1, length), dtype=torch.long),
        }


def _tiny_prepare_spec():
    spec = get_task_spec("sst2")
    return type(spec)(
        name=spec.name,
        model_id="unit/tiny-sst2",
        dataset_name="unit",
        dataset_config="sst2",
        dataset_split="validation",
        max_length=8,
        source_outputs=spec.source_outputs,
        verified_runtime_outputs=spec.verified_runtime_outputs,
        calibration_samples=4,
        target_device=spec.target_device,
        compiler_inputs=spec.compiler_inputs,
        mxq_inputs=(TensorContract("embeddings", (1, -1, 4), "float32"),),
    )


def test_prepare_task_writes_runtime_compatible_calibration_and_manifest(
    monkeypatch, tmp_path
):
    model = BertForSequenceClassification(_bert_config()).eval()
    dataset = [
        {"sentence": f"sample {index}", "label": index % 2}
        for index in range(10)
    ]
    monkeypatch.setattr(prepare_recipe, "get_task_spec", lambda task: _tiny_prepare_spec())
    monkeypatch.setattr(prepare_recipe, "load_source_model", lambda spec: model)
    monkeypatch.setattr(prepare_recipe, "load_tokenizer", lambda spec: _TinyTokenizer())
    monkeypatch.setattr(prepare_recipe, "load_dataset_split", lambda spec: dataset)

    manifest = prepare_recipe.prepare_task("sst2", tmp_path)

    task_root = tmp_path / "sst2"
    calibration_files = sorted((task_root / "calibration_data").glob("*.npy"))
    assert [path.name for path in calibration_files] == [
        "000.npy",
        "001.npy",
        "002.npy",
        "003.npy",
    ]
    values = [__import__("numpy").load(path) for path in calibration_files]
    assert all(value.dtype.name == "float32" for value in values)
    assert all(value.ndim == 3 and value.shape[0] == 1 for value in values)
    assert all(value.shape[2] == 4 for value in values)
    assert all(value.flags.c_contiguous for value in values)
    weights_path = task_root / "weights" / "weight_dict.pth"
    assert weights_path.is_file()
    assert manifest["task"] == "sst2"
    assert manifest["dataset_size"] == 10
    assert manifest["calibration_indices"] == [0, 3, 6, 9]
    assert manifest["calibration_files"] == 4
    assert manifest["calibration_artifacts"] == _calibration_artifacts(
        task_root, calibration_files
    )
    assert manifest["weights"]["path"] == "weights/weight_dict.pth"
    assert manifest["weights"]["sha256"] == sha256_file(weights_path)
    assert json.loads((task_root / "calibration_manifest.json").read_text()) == manifest
    environment = json.loads((tmp_path / "compile-environment.json").read_text())
    assert environment["python"]["version_info"][:2] == [
        sys.version_info.major,
        sys.version_info.minor,
    ]
    assert "platform" in environment


def test_environment_report_records_the_validated_vendor_wheel(monkeypatch):
    monkeypatch.setenv(
        "MOBILINT_QBCOMPILER_WHEEL_SHA256",
        "a" * 64,
    )
    monkeypatch.setenv(
        "MOBILINT_QBCOMPILER_WHEEL_NAME",
        "qbcompiler-1.2.0-py3-none-any.whl",
    )

    report = prepare_recipe._environment_report()

    assert report["qbcompiler_wheel"] == {
        "filename": "qbcompiler-1.2.0-py3-none-any.whl",
        "sha256": "a" * 64,
    }


def test_prepare_task_refuses_an_existing_task_directory(tmp_path):
    (tmp_path / "sst2").mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        prepare_recipe.prepare_task("sst2", tmp_path)


def test_prepare_describe_does_not_need_huggingface_or_vendor_packages():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(FRAMEWORK_ROOT), str(FRAMEWORK_ROOT / "src"))
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.mobilint_bert_compile.prepare",
            "--task",
            "squad1",
            "--describe",
        ],
        check=True,
        cwd=FRAMEWORK_ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["task"] == "squad1"
    assert payload["target_device"] == "aries-rb"


def test_validate_calibration_set_requires_exact_manifest_files(tmp_path):
    task_root = tmp_path / "sst2"
    calibration_dir = task_root / "calibration_data"
    calibration_dir.mkdir(parents=True)
    spec = _tiny_prepare_spec()
    sequence_lengths = [3, 4, 5, 3]
    for index, sequence_length in enumerate(sequence_lengths):
        np.save(
            calibration_dir / f"{index:03d}.npy",
            np.zeros((1, sequence_length, 4), dtype=np.float32),
        )
    paths = sorted(calibration_dir.glob("*.npy"))
    manifest = {
        "dataset_size": 10,
        "calibration_files": 4,
        "calibration_indices": [0, 3, 6, 9],
        "sequence_lengths": sequence_lengths,
        "calibration_artifacts": _calibration_artifacts(task_root, paths),
    }

    paths = compile_recipe.validate_calibration_set(
        task_root,
        manifest,
        spec,
    )

    assert [path.name for path in paths] == [
        "000.npy",
        "001.npy",
        "002.npy",
        "003.npy",
    ]
    np.save(calibration_dir / "extra.npy", np.zeros((1, 3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="calibration file set mismatch"):
        compile_recipe.validate_calibration_set(
            task_root,
            manifest,
            spec,
        )


def test_validate_calibration_set_rejects_manifest_count_drift(tmp_path):
    task_root = tmp_path / "sst2"
    (task_root / "calibration_data").mkdir(parents=True)

    with pytest.raises(ValueError, match="contract requires 4"):
        compile_recipe.validate_calibration_set(
            task_root,
            {
                "dataset_size": 1,
                "calibration_files": 1,
                "calibration_indices": [0],
                "sequence_lengths": [3],
            },
            _tiny_prepare_spec(),
        )


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (np.zeros((1, 3, 4), dtype=np.float64), "dtype"),
        (np.zeros((1, 3, 5), dtype=np.float32), "shape"),
        (np.full((1, 3, 4), np.nan, dtype=np.float32), "finite"),
    ],
)
def test_validate_calibration_set_rejects_invalid_arrays(tmp_path, array, message):
    task_root = tmp_path / "sst2"
    calibration_dir = task_root / "calibration_data"
    calibration_dir.mkdir(parents=True)
    spec = replace(_tiny_prepare_spec(), calibration_samples=1)
    path = calibration_dir / "000.npy"
    np.save(path, array)

    with pytest.raises(ValueError, match=message):
        compile_recipe.validate_calibration_set(
            task_root,
            {
                "dataset_size": 1,
                "calibration_files": 1,
                "calibration_indices": [0],
                "sequence_lengths": [3],
                "calibration_artifacts": _calibration_artifacts(task_root, [path]),
            },
            spec,
        )


@pytest.mark.parametrize("mutation", ["missing", "mismatch"])
def test_validate_calibration_set_requires_exact_file_hashes(tmp_path, mutation):
    task_root = tmp_path / "sst2"
    calibration_dir = task_root / "calibration_data"
    calibration_dir.mkdir(parents=True)
    spec = replace(_tiny_prepare_spec(), calibration_samples=1)
    path = calibration_dir / "000.npy"
    np.save(path, np.zeros((1, 3, 4), dtype=np.float32))
    artifacts = _calibration_artifacts(task_root, [path])
    if mutation == "missing":
        artifacts[0].pop("sha256")
    else:
        artifacts[0]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="calibration.*SHA256|artifact record"):
        compile_recipe.validate_calibration_set(
            task_root,
            {
                "dataset_size": 1,
                "calibration_files": 1,
                "calibration_indices": [0],
                "sequence_lengths": [3],
                "calibration_artifacts": artifacts,
            },
            spec,
        )


def test_build_feed_dict_marks_dynamic_sequence_and_padding_mask():
    wrapped_by_name = {}
    attention_calls = []

    class _Dimension:
        def __init__(self):
            self.dynamic = False

        def set_dynamic(self):
            self.dynamic = True

    def wrap_tensor(name, value):
        wrapped = SimpleNamespace(name=name, value=value, src_shape=[_Dimension(), _Dimension()])
        wrapped_by_name[name] = wrapped
        return wrapped

    def set_attention_mask(value, semantic):
        attention_calls.append((value, semantic))

    feed = compile_recipe.build_feed_dict(
        {
            "input_ids": torch.ones((1, 5), dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
        },
        wrap_tensor=wrap_tensor,
        set_attention_mask=set_attention_mask,
    )

    assert tuple(feed) == ("input_ids", "attention_mask", "token_type_ids")
    assert all(wrapped.src_shape[1].dynamic for wrapped in wrapped_by_name.values())
    assert attention_calls == [(wrapped_by_name["attention_mask"], "padding_mask")]
    assert torch.equal(
        wrapped_by_name["token_type_ids"].value,
        torch.zeros((1, 5), dtype=torch.long),
    )


def test_mblt_compile_uses_the_fixed_aries_target_and_refuses_overwrite(tmp_path):
    calls = []

    def fake_compile(**kwargs):
        calls.append(kwargs)
        Path(kwargs["mblt_save_path"]).write_bytes(b"mblt")

    output = tmp_path / "mblt" / "sst2.mblt"
    result = compile_recipe.run_mblt_compile(
        model="model",
        feed_dict={"x": "wrapped"},
        output=output,
        compiler=fake_compile,
    )

    assert result == output
    assert calls == [
        {
            "model": "model",
            "mblt_save_path": str(output),
            "target_device": "aries-rb",
            "backend": "torch",
            "feed_dict": {"x": "wrapped"},
            "cpu_offload": True,
        }
    ]
    with pytest.raises(FileExistsError, match="already exists"):
        compile_recipe.run_mblt_compile(
            model="model",
            feed_dict={},
            output=output,
            compiler=fake_compile,
        )


class _FakeCalibrationConfig:
    class MaxPercentile:
        def __init__(self, *, percentile, topk_ratio):
            self.percentile = percentile
            self.topk_ratio = topk_ratio

    def __init__(self, *, method, output, mode, max_percentile):
        self.method = method
        self.output = output
        self.mode = mode
        self.max_percentile = max_percentile


def test_mxq_compile_uses_the_verified_calibration_options(tmp_path):
    calls = []

    def fake_compile(**kwargs):
        calls.append(kwargs)
        Path(kwargs["save_path"]).write_bytes(b"mxq")

    calibration_dir = tmp_path / "calibration_data"
    calibration_dir.mkdir()
    output = tmp_path / "mxq" / "squad1.mxq"
    api = SimpleNamespace(
        CalibrationConfig=_FakeCalibrationConfig,
        mxq_compile=fake_compile,
    )

    result = compile_recipe.run_mxq_compile(
        model="model",
        feed_dict={"x": "wrapped"},
        calibration_dir=calibration_dir,
        output=output,
        compiler_api=api,
    )

    assert result == output
    assert len(calls) == 1
    call = calls[0]
    assert call["model"] == "model"
    assert call["target_device"] == "aries-rb"
    assert call["save_path"] == str(output)
    assert call["calib_data_path"] == str(calibration_dir)
    assert call["backend"] == "torch"
    assert call["feed_dict"] == {"x": "wrapped"}
    assert call["inference_scheme"] == "all"
    config = call["calibration_config"]
    assert (config.method, config.output, config.mode) == (1, 0, 1)
    assert config.max_percentile.percentile == 0.999
    assert config.max_percentile.topk_ratio == 0.01


def test_compile_report_records_source_runtime_order_and_exact_options():
    report = compile_recipe.create_compile_report(
        get_task_spec("squad1"),
        {"start_logits": [1, 9], "end_logits": [1, 9]},
        compiler_version="1.2.0",
    )

    assert report["source_outputs"] == ["start_logits", "end_logits"]
    assert report["verified_runtime_outputs"] == ["end_logits", "start_logits"]
    assert report["compiler_options"] == {
        "mblt": {
            "target_device": "aries-rb",
            "backend": "torch",
            "cpu_offload": True,
        },
        "mxq": {
            "target_device": "aries-rb",
            "backend": "torch",
            "inference_scheme": "all",
            "calibration": {
                "method": 1,
                "output": 0,
                "mode": 1,
                "max_percentile": 0.999,
                "topk_ratio": 0.01,
            },
        },
    }


def test_source_output_shape_validation_rejects_wrong_task_heads():
    with pytest.raises(RuntimeError, match="SST-2.*shape"):
        compile_recipe.validate_source_output_shapes(
            get_task_spec("sst2"),
            {"input_ids": torch.ones((1, 7), dtype=torch.long)},
            {"logits": torch.zeros((1, 3))},
        )

    with pytest.raises(RuntimeError, match="SQuAD.*shape"):
        compile_recipe.validate_source_output_shapes(
            get_task_spec("squad1"),
            {"input_ids": torch.ones((1, 7), dtype=torch.long)},
            {
                "start_logits": torch.zeros((1, 7, 1)),
                "end_logits": torch.zeros((1, 7)),
            },
        )


def test_source_output_shape_validation_accepts_the_task_contracts():
    assert compile_recipe.validate_source_output_shapes(
        get_task_spec("sst2"),
        {"input_ids": torch.ones((1, 7), dtype=torch.long)},
        {"logits": torch.zeros((1, 2))},
    ) == {"logits": [1, 2]}
    assert compile_recipe.validate_source_output_shapes(
        get_task_spec("squad1"),
        {"input_ids": torch.ones((1, 7), dtype=torch.long)},
        {
            "start_logits": torch.zeros((1, 7)),
            "end_logits": torch.zeros((1, 7)),
        },
    ) == {"start_logits": [1, 7], "end_logits": [1, 7]}


def test_all_stage_reloads_the_source_model_for_each_compiler(monkeypatch, tmp_path):
    spec = get_task_spec("sst2")
    task_root = tmp_path / "sst2"
    task_root.mkdir()
    (task_root / "calibration_manifest.json").write_text(
        json.dumps(
            {
                "task": spec.name,
                "model_id": spec.model_id,
                "target_device": spec.target_device,
                "calibration_files": 1,
            }
        )
    )
    load_calls = []

    monkeypatch.setattr(compile_recipe, "validate_calibration_set", lambda *args: [])

    def fake_load(loaded_spec):
        load_calls.append(loaded_spec.name)
        return "model", {"x": "wrapped"}, {"logits": [1, 2]}

    def fake_mblt(**kwargs):
        path = Path(kwargs["output"])
        path.parent.mkdir(parents=True)
        path.write_bytes(b"mblt")
        return path

    def fake_mxq(**kwargs):
        path = Path(kwargs["output"])
        path.parent.mkdir(parents=True)
        path.write_bytes(b"mxq")
        return path

    monkeypatch.setattr(compile_recipe, "_load_model_and_feed", fake_load)
    monkeypatch.setattr(compile_recipe, "run_mblt_compile", fake_mblt)
    monkeypatch.setattr(compile_recipe, "run_mxq_compile", fake_mxq)

    report = compile_recipe.compile_task("sst2", "all", tmp_path)

    assert load_calls == ["sst2", "sst2"]
    assert set(report["artifacts"]) == {"mblt", "mxq"}


@pytest.mark.parametrize("kind", ["mblt", "mxq"])
def test_compiler_helpers_reject_zero_byte_artifacts(tmp_path, kind):
    output = tmp_path / kind / f"model.{kind}"

    if kind == "mblt":
        def empty_compile(**kwargs):
            Path(kwargs["mblt_save_path"]).touch()

        call = lambda: compile_recipe.run_mblt_compile(
            model="model",
            feed_dict={},
            output=output,
            compiler=empty_compile,
        )
    else:
        calibration_dir = tmp_path / "calibration"
        calibration_dir.mkdir()

        def empty_compile(**kwargs):
            Path(kwargs["save_path"]).touch()

        call = lambda: compile_recipe.run_mxq_compile(
            model="model",
            feed_dict={},
            calibration_dir=calibration_dir,
            output=output,
            compiler_api=SimpleNamespace(
                CalibrationConfig=_FakeCalibrationConfig,
                mxq_compile=empty_compile,
            ),
        )

    with pytest.raises(RuntimeError, match="empty artifact"):
        call()


def test_compile_describe_does_not_need_qbcompiler():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(FRAMEWORK_ROOT), str(FRAMEWORK_ROOT / "src"))
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.mobilint_bert_compile.compile",
            "--task",
            "sst2",
            "--describe",
        ],
        check=True,
        cwd=FRAMEWORK_ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["task"] == "sst2"
    assert payload["target_device"] == "aries-rb"


def _compiler_script_path():
    return FRAMEWORK_ROOT / "scripts" / "compile_mobilint_bert.sh"


def test_compiler_shell_script_has_valid_syntax_and_help():
    script = _compiler_script_path()

    assert script.is_file()
    subprocess.run(["bash", "-n", str(script)], check=True)
    help_result = subprocess.run(
        ["bash", str(script), "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--wheel" in help_result.stdout
    assert "--python" in help_result.stdout
    assert "--task" in help_result.stdout
    assert "--output-root" in help_result.stdout
    assert help_result.stdout.startswith("Mobilint BERT")
    assert "set -euo" not in help_result.stdout


def test_compiler_shell_script_rejects_missing_and_unknown_arguments():
    script = _compiler_script_path()

    missing = subprocess.run(
        ["bash", str(script)],
        check=False,
        text=True,
        capture_output=True,
    )
    unknown = subprocess.run(
        ["bash", str(script), "--unknown"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert missing.returncode != 0
    assert "--wheel" in missing.stderr
    assert unknown.returncode != 0
    assert "unknown argument" in unknown.stderr


def test_compiler_shell_script_is_compiler_only_and_pins_the_host_contract():
    text = _compiler_script_path().read_text(encoding="utf-8")

    assert "22.04" in text
    assert "x86_64" in text
    assert "3.10" in text
    assert "qbcompiler-1.2.0-py3-none-any.whl" in text
    assert "28f276baef1bff86ed313cb819b53d8abb684a7555cf4c81c459edc09abf1b4b" in text
    assert "torch==2.7.1" in text
    assert "onnxruntime==1.19.2" in text
    assert "tools.mobilint_bert_compile.prepare" in text
    assert "tools.mobilint_bert_compile.compile" in text
    assert '--stage mblt' in text
    assert '--stage mxq' in text
    assert '--stage all' not in text
    for forbidden in ("qbruntime", "mobilint-cli", "/dev/aries"):
        assert forbidden not in text


def test_compile_runbook_documents_the_real_entrypoints_and_contracts():
    repository_root = FRAMEWORK_ROOT.parent
    runbook = repository_root / "docs" / "mobilint-bert-compilation.md"

    assert runbook.is_file()
    text = runbook.read_text(encoding="utf-8")
    assert "framework/scripts/compile_mobilint_bert.sh" in text
    assert "tools.mobilint_bert_compile.prepare" in text
    assert "tools.mobilint_bert_compile.compile" in text
    assert "qbcompiler-1.2.0-py3-none-any.whl" in text
    assert "Ubuntu 22.04" in text
    assert "Python 3.10" in text
    assert "aries-rb" in text
    assert "start_logits" in text
    assert "end_logits" in text
    assert "Docker가 필요하지" in text
    assert "ARIES 장치가 필요하지" in text
    assert "`.mblt`는 `.mxq`의 입력" in text


def test_aries_transformer_runbook_links_the_compile_runbook():
    aries_runbook = FRAMEWORK_ROOT.parent / "docs" / "mobilint-aries-transformers.md"

    text = aries_runbook.read_text(encoding="utf-8")
    assert "(mobilint-bert-compilation.md)" in text


@pytest.mark.parametrize(
    "generated_path",
    [
        ".venv-qbcompiler-1.2-py310/bin/python",
        "mobilint-bert-artifacts/sst2/mxq/sst2.mxq",
        "mobilint-bert-artifacts-retry-20260803/squad1/mxq/squad1.mxq",
        "mobilint-bert-compile.log",
    ],
)
def test_documented_compiler_outputs_are_gitignored(generated_path):
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", generated_path],
        check=False,
        cwd=FRAMEWORK_ROOT.parent,
    )

    assert result.returncode == 0
