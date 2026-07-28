"""Download and compile a Llama model into a stable Optimum RBLN directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

MODEL_DEFAULTS = {
    "llama-3.2-3b": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "num_devices": 8,
        "max_seq_len": 4096,
        "block_size": 4096,
    },
    "llama-3.1-8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "num_devices": 8,
        "max_seq_len": 131072,
        "block_size": 16384,
    },
}
PACKAGE_NAMES = (
    "rebel-compiler",
    "optimum-rbln",
    "vllm-rbln",
    "vllm",
    "transformers",
)
MANIFEST_NAME = "rbln-vllm-manifest.json"


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _decoder_batch_sizes(value: object, batch_size: int) -> list[int]:
    if value is None or value == "":
        values: list[object] = [batch_size]
    elif isinstance(value, str):
        try:
            values = [
                int(part.strip())
                for part in value.split(",")
                if part.strip()
            ]
        except ValueError as exc:
            raise ValueError(
                "decoder_batch_sizes must contain positive integers"
            ) from exc
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise ValueError(
            "decoder_batch_sizes must be a list or comma-separated integers"
        )
    result = [_positive_int(item, "decoder_batch_sizes") for item in values]
    if not result or any(item > batch_size for item in result):
        raise ValueError(
            "decoder_batch_sizes values must not exceed batch_size"
        )
    result.append(batch_size)
    return sorted(set(result), reverse=True)


def resolve_compile_contract(
    *,
    model: str,
    model_id: str | None,
    num_devices: int,
    max_seq_len: int | None,
    block_size: int | None,
    batch_size: int,
    allow_unsupported_single_npu: bool,
    decoder_batch_sizes: object = None,
) -> dict[str, Any]:
    if model not in MODEL_DEFAULTS:
        raise ValueError(f"unsupported model alias: {model}")
    defaults = MODEL_DEFAULTS[model]
    selected_num_devices = _positive_int(num_devices, "num_devices")
    selected_batch_size = _positive_int(batch_size, "batch_size")
    if type(allow_unsupported_single_npu) is not bool:
        raise ValueError("allow_unsupported_single_npu must be a bool")

    if model == "llama-3.1-8b" and selected_num_devices == 1:
        raise ValueError(
            "Llama 3.1 8B cannot fit on one 15.7 GiB ATOM NPU with "
            "runtime and KV-cache memory"
        )
    if selected_num_devices == 8:
        support_classification = "official"
    elif (
        model == "llama-3.2-3b"
        and selected_num_devices == 1
        and allow_unsupported_single_npu
    ):
        support_classification = "unsupported_single_npu_experiment"
    elif model == "llama-3.2-3b" and selected_num_devices == 1:
        raise ValueError(
            "the one-NPU Llama 3.2 3B experiment requires "
            "--allow-unsupported-single-npu"
        )
    else:
        raise ValueError(
            f"{model} is officially supported with 8 NPU chips"
        )
    if (
        support_classification == "unsupported_single_npu_experiment"
        and selected_batch_size != 1
    ):
        raise ValueError(
            "the one-NPU Llama 3.2 3B experiment requires batch_size=1"
        )
    selected_decoder_batch_sizes = _decoder_batch_sizes(
        decoder_batch_sizes, selected_batch_size
    )

    if max_seq_len is None:
        selected_max_seq_len = (
            512
            if support_classification == "unsupported_single_npu_experiment"
            else defaults["max_seq_len"]
        )
    else:
        selected_max_seq_len = _positive_int(max_seq_len, "max_seq_len")
    if block_size is None:
        selected_block_size = (
            selected_max_seq_len
            if support_classification == "unsupported_single_npu_experiment"
            else defaults["block_size"]
        )
    else:
        selected_block_size = _positive_int(block_size, "block_size")
    if selected_block_size > selected_max_seq_len:
        raise ValueError("block_size must not exceed max_seq_len")
    if selected_max_seq_len % selected_block_size != 0:
        raise ValueError("block_size must divide max_seq_len")
    if (
        support_classification == "unsupported_single_npu_experiment"
        and selected_max_seq_len > 1024
    ):
        raise ValueError(
            "the one-NPU Llama 3.2 3B experiment allows at most 1024 tokens"
        )
    selected_model_id = model_id or defaults["model_id"]
    if not isinstance(selected_model_id, str) or not selected_model_id.strip():
        raise ValueError("model_id must be a non-empty string")

    return {
        "model": model,
        "model_id": selected_model_id,
        "num_devices": selected_num_devices,
        "max_seq_len": selected_max_seq_len,
        "block_size": selected_block_size,
        "batch_size": selected_batch_size,
        "decoder_batch_sizes": selected_decoder_batch_sizes,
        "support_classification": support_classification,
    }


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_files(output_dir: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted(
        (
            item
            for item in output_dir.rglob("*")
            if item.is_file() and item.name != MANIFEST_NAME
        ),
        key=lambda item: item.relative_to(output_dir).as_posix(),
    ):
        result.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return result


def compile_and_save(
    contract: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        from optimum.rbln import RBLNLlamaForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "model preparation requires optimum-rbln and transformers in "
            "the RBLN SDK environment"
        ) from exc

    compile_kwargs = {
        "model_id": contract["model_id"],
        "export": True,
        "rbln_batch_size": contract["batch_size"],
        "rbln_max_seq_len": contract["max_seq_len"],
        "rbln_num_devices": contract["num_devices"],
        "rbln_decoder_batch_sizes": contract["decoder_batch_sizes"],
        "rbln_create_runtimes": False,
    }
    if contract["block_size"] < contract["max_seq_len"]:
        compile_kwargs["rbln_kvcache_partition_len"] = contract["block_size"]

    tokenizer = AutoTokenizer.from_pretrained(contract["model_id"])
    model = RBLNLlamaForCausalLM.from_pretrained(**compile_kwargs)
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    if not (output / "config.json").is_file():
        raise RuntimeError("saved RBLN model is missing config.json")
    if not any(path.is_file() for path in output.rglob("*.rbln")):
        raise RuntimeError("saved RBLN model contains no .rbln artifacts")
    if not (output / "tokenizer_config.json").is_file() or not any(
        (output / name).is_file()
        for name in ("tokenizer.json", "tokenizer.model")
    ):
        raise RuntimeError("saved RBLN model is missing tokenizer artifacts")

    manifest = {
        **dict(contract),
        "compile_api": "RBLNLlamaForCausalLM.from_pretrained",
        "compile_kwargs": compile_kwargs,
        "package_versions": package_versions(),
        "files": artifact_files(output),
    }
    (output / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download and compile a supported Llama model into a prepared "
            "Optimum RBLN directory"
        )
    )
    parser.add_argument("--model", required=True, choices=sorted(MODEL_DEFAULTS))
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-devices", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--decoder-batch-sizes",
        default=None,
        help=(
            "comma-separated decoder graph batch sizes; the maximum "
            "--batch-size is included automatically"
        ),
    )
    parser.add_argument(
        "--allow-unsupported-single-npu",
        action="store_true",
        help=(
            "allow the unsupported Llama 3.2 3B one-NPU experiment; "
            "never enables one-NPU Llama 3.1 8B"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    contract = resolve_compile_contract(
        model=args.model,
        model_id=args.model_id,
        num_devices=args.num_devices,
        max_seq_len=args.max_seq_len,
        block_size=args.block_size,
        batch_size=args.batch_size,
        allow_unsupported_single_npu=args.allow_unsupported_single_npu,
        decoder_batch_sizes=args.decoder_batch_sizes,
    )
    manifest = compile_and_save(contract, args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
