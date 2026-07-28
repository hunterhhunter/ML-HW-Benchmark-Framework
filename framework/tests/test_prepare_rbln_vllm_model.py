import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import prepare_rbln_vllm_model as prepare


def test_resolve_defaults_for_official_models():
    llama32 = prepare.resolve_compile_contract(
        model="llama-3.2-3b",
        model_id=None,
        num_devices=8,
        max_seq_len=None,
        block_size=None,
        batch_size=1,
        allow_unsupported_single_npu=False,
    )
    llama31 = prepare.resolve_compile_contract(
        model="llama-3.1-8b",
        model_id=None,
        num_devices=8,
        max_seq_len=None,
        block_size=None,
        batch_size=1,
        allow_unsupported_single_npu=False,
    )

    assert llama32 == {
        "model": "llama-3.2-3b",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "num_devices": 8,
        "max_seq_len": 4096,
        "block_size": 4096,
        "batch_size": 1,
        "decoder_batch_sizes": [1],
        "support_classification": "official",
    }
    assert llama31 == {
        "model": "llama-3.1-8b",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "num_devices": 8,
        "max_seq_len": 131072,
        "block_size": 16384,
        "batch_size": 1,
        "decoder_batch_sizes": [1],
        "support_classification": "official",
    }


def test_single_npu_llama_3_2_requires_opt_in_and_short_context():
    with pytest.raises(ValueError, match="allow-unsupported-single-npu"):
        prepare.resolve_compile_contract(
            model="llama-3.2-3b",
            model_id=None,
            num_devices=1,
            max_seq_len=512,
            block_size=512,
            batch_size=1,
            allow_unsupported_single_npu=False,
        )

    contract = prepare.resolve_compile_contract(
        model="llama-3.2-3b",
        model_id=None,
        num_devices=1,
        max_seq_len=512,
        block_size=512,
        batch_size=1,
        allow_unsupported_single_npu=True,
    )
    assert contract["support_classification"] == (
        "unsupported_single_npu_experiment"
    )
    assert contract["decoder_batch_sizes"] == [1]

    with pytest.raises(ValueError, match="at most 1024"):
        prepare.resolve_compile_contract(
            model="llama-3.2-3b",
            model_id=None,
            num_devices=1,
            max_seq_len=2048,
            block_size=512,
            batch_size=1,
            allow_unsupported_single_npu=True,
        )


def test_single_npu_llama_3_1_8b_is_always_rejected():
    with pytest.raises(ValueError, match="cannot fit"):
        prepare.resolve_compile_contract(
            model="llama-3.1-8b",
            model_id=None,
            num_devices=1,
            max_seq_len=512,
            block_size=512,
            batch_size=1,
            allow_unsupported_single_npu=True,
        )


def test_official_model_accepts_dynamic_decoder_batch_contract():
    contract = prepare.resolve_compile_contract(
        model="llama-3.2-3b",
        model_id=None,
        num_devices=8,
        max_seq_len=4096,
        block_size=4096,
        batch_size=4,
        decoder_batch_sizes="1,2,4",
        allow_unsupported_single_npu=False,
    )

    assert contract["batch_size"] == 4
    assert contract["decoder_batch_sizes"] == [4, 2, 1]


def test_single_npu_experiment_rejects_batch_greater_than_one():
    with pytest.raises(ValueError, match="batch_size=1"):
        prepare.resolve_compile_contract(
            model="llama-3.2-3b",
            model_id=None,
            num_devices=1,
            max_seq_len=512,
            block_size=512,
            batch_size=2,
            decoder_batch_sizes="1,2",
            allow_unsupported_single_npu=True,
        )


@pytest.mark.parametrize(
    ("max_seq_len", "block_size", "message"),
    [
        (1000, 512, "divide"),
        (512, 1024, "not exceed"),
    ],
)
def test_compile_contract_validates_attention_shape(
    max_seq_len, block_size, message
):
    with pytest.raises(ValueError, match=message):
        prepare.resolve_compile_contract(
            model="llama-3.2-3b",
            model_id=None,
            num_devices=8,
            max_seq_len=max_seq_len,
            block_size=block_size,
            batch_size=1,
            allow_unsupported_single_npu=False,
        )


def test_compile_and_save_forwards_optimum_options_and_writes_manifest(
    monkeypatch, tmp_path
):
    state = {
        "from_pretrained": [],
        "save": [],
        "tokenizer_from_pretrained": [],
        "tokenizer_save": [],
    }

    class Model:
        def save_pretrained(self, output_dir):
            output = Path(output_dir)
            output.mkdir(parents=True)
            (output / "config.json").write_text(
                json.dumps({"model_type": "llama"}), encoding="utf-8"
            )
            artifact = output / "decoder.rbln"
            artifact.write_bytes(b"fake-rbln")
            state["save"].append(str(output))

    class RBLNLlamaForCausalLM:
        @classmethod
        def from_pretrained(cls, **kwargs):
            state["from_pretrained"].append(kwargs)
            return Model()

    class Tokenizer:
        def save_pretrained(self, output_dir):
            output = Path(output_dir)
            (output / "tokenizer_config.json").write_text(
                json.dumps({"model_max_length": 32768}), encoding="utf-8"
            )
            (output / "tokenizer.json").write_text(
                json.dumps({"version": "1.0"}), encoding="utf-8"
            )
            state["tokenizer_save"].append(str(output))

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id):
            state["tokenizer_from_pretrained"].append(model_id)
            return Tokenizer()

    optimum = types.ModuleType("optimum")
    optimum_rbln = types.ModuleType("optimum.rbln")
    optimum_rbln.RBLNLlamaForCausalLM = RBLNLlamaForCausalLM
    monkeypatch.setitem(sys.modules, "optimum", optimum)
    monkeypatch.setitem(sys.modules, "optimum.rbln", optimum_rbln)
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        prepare,
        "package_versions",
        lambda: {
            "rebel-compiler": "0.11.0.post1",
            "optimum-rbln": "0.11.0.post1",
            "vllm-rbln": "0.11.0",
        },
    )
    output_dir = tmp_path / "prepared"
    contract = prepare.resolve_compile_contract(
        model="llama-3.1-8b",
        model_id=None,
        num_devices=8,
        max_seq_len=32768,
        block_size=4096,
        batch_size=1,
        allow_unsupported_single_npu=False,
    )

    manifest = prepare.compile_and_save(contract, output_dir)

    assert state["from_pretrained"] == [
        {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "export": True,
            "rbln_batch_size": 1,
            "rbln_max_seq_len": 32768,
            "rbln_num_devices": 8,
            "rbln_kvcache_partition_len": 4096,
            "rbln_decoder_batch_sizes": [1],
            "rbln_create_runtimes": False,
        }
    ]
    assert state["save"] == [str(output_dir.resolve())]
    assert state["tokenizer_from_pretrained"] == [
        "meta-llama/Llama-3.1-8B-Instruct"
    ]
    assert state["tokenizer_save"] == [str(output_dir.resolve())]
    saved = json.loads(
        (output_dir / "rbln-vllm-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest == saved
    assert saved["model"] == "llama-3.1-8b"
    assert saved["num_devices"] == 8
    assert saved["block_size"] == 4096
    assert saved["batch_size"] == 1
    assert saved["decoder_batch_sizes"] == [1]
    assert saved["package_versions"]["vllm-rbln"] == "0.11.0"
    assert saved["files"] == [
        {
            "path": "config.json",
            "sha256": hashlib.sha256(
                json.dumps({"model_type": "llama"}).encode("utf-8")
            ).hexdigest(),
            "size_bytes": len(json.dumps({"model_type": "llama"})),
        },
        {
            "path": "decoder.rbln",
            "sha256": hashlib.sha256(b"fake-rbln").hexdigest(),
            "size_bytes": len(b"fake-rbln"),
        },
        {
            "path": "tokenizer.json",
            "sha256": hashlib.sha256(
                json.dumps({"version": "1.0"}).encode("utf-8")
            ).hexdigest(),
            "size_bytes": len(json.dumps({"version": "1.0"})),
        },
        {
            "path": "tokenizer_config.json",
            "sha256": hashlib.sha256(
                json.dumps({"model_max_length": 32768}).encode("utf-8")
            ).hexdigest(),
            "size_bytes": len(json.dumps({"model_max_length": 32768})),
        },
    ]


def test_compile_and_save_rejects_existing_output(tmp_path):
    output_dir = tmp_path / "prepared"
    output_dir.mkdir()
    contract = prepare.resolve_compile_contract(
        model="llama-3.2-3b",
        model_id=None,
        num_devices=8,
        max_seq_len=4096,
        block_size=4096,
        batch_size=1,
        allow_unsupported_single_npu=False,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        prepare.compile_and_save(contract, output_dir)
