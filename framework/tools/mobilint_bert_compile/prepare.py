"""Prepare deterministic calibration embeddings for Mobilint BERT compilation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

from tools.mobilint_bert_compile.common import (
    TaskSpec,
    contract_to_dict,
    extract_embedding_weights,
    get_task_spec,
    sha256_file,
)


PRIMARY_PACKAGES = (
    "qbcompiler",
    "torch",
    "torchvision",
    "numpy",
    "tensorflow",
    "onnx",
    "onnxruntime",
    "opencv-python",
    "transformers",
    "datasets",
)


def select_calibration_indices(
    dataset_size: int, count: int = 32
) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("calibration sample count must be positive")
    if dataset_size < count:
        raise ValueError(
            f"requested {count} calibration samples, but dataset has only {dataset_size}"
        )

    import numpy as np

    return tuple(
        int(value)
        for value in np.linspace(0, dataset_size - 1, num=count, dtype=np.int64)
    )


def load_source_model(spec: TaskSpec):
    from transformers import (
        AutoModelForQuestionAnswering,
        AutoModelForSequenceClassification,
    )

    if spec.name == "sst2":
        return AutoModelForSequenceClassification.from_pretrained(spec.model_id).eval()
    if spec.name == "squad1":
        return AutoModelForQuestionAnswering.from_pretrained(spec.model_id).eval()
    raise ValueError(f"unsupported Mobilint BERT compile task: {spec.name}")


def load_tokenizer(spec: TaskSpec):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec.model_id)


def load_dataset_split(spec: TaskSpec):
    from datasets import load_dataset

    return load_dataset(
        spec.dataset_name,
        spec.dataset_config,
        split=spec.dataset_split,
    )


def _tokenize_example(spec: TaskSpec, tokenizer, example: dict):
    if spec.name == "sst2":
        return tokenizer(
            example["sentence"],
            return_tensors="pt",
            max_length=spec.max_length,
            truncation=True,
        )
    if spec.name == "squad1":
        return tokenizer(
            example["question"],
            example["context"],
            return_tensors="pt",
            max_length=spec.max_length,
            truncation="only_second",
        )
    raise ValueError(f"unsupported Mobilint BERT compile task: {spec.name}")


def _numpy_token_inputs(batch: dict) -> dict[str, object]:
    values = {}
    for name in ("input_ids", "attention_mask", "token_type_ids"):
        value = batch.get(name)
        if value is None:
            continue
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        values[name] = value
    return values


def _environment_report() -> dict[str, object]:
    packages = {}
    for name in PRIMARY_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "version_info": list(sys.version_info[:3]),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "qbcompiler_wheel": {
            "filename": os.environ.get("MOBILINT_QBCOMPILER_WHEEL_NAME"),
            "sha256": os.environ.get("MOBILINT_QBCOMPILER_WHEEL_SHA256"),
        },
    }


def _write_environment_report(output_root: Path) -> None:
    path = output_root / "compile-environment.json"
    if path.exists():
        return
    path.write_text(
        json.dumps(_environment_report(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_task(task: str, output_root: str | Path) -> dict[str, object]:
    import numpy as np
    import torch

    from preprocessor.mobilint_bert_embedding import (
        MobilintBertEmbeddingTransform,
    )

    spec = get_task_spec(task)
    root = Path(output_root).expanduser().resolve()
    task_root = root / spec.name
    if task_root.exists():
        raise FileExistsError(f"Mobilint BERT task output already exists: {task_root}")

    calibration_dir = task_root / "calibration_data"
    weights_path = task_root / "weights" / "weight_dict.pth"
    calibration_dir.mkdir(parents=True)
    weights_path.parent.mkdir(parents=True)
    _write_environment_report(root)

    model = load_source_model(spec)
    tokenizer = load_tokenizer(spec)
    dataset = load_dataset_split(spec)
    indices = select_calibration_indices(len(dataset), spec.calibration_samples)

    weights = extract_embedding_weights(model)
    torch.save(weights, weights_path)
    embedding_width = spec.mxq_inputs[0].shape[-1]
    transform = MobilintBertEmbeddingTransform(
        weights_path,
        expected_width=embedding_width,
    )

    sequence_lengths = []
    for output_index, dataset_index in enumerate(indices):
        batch = _tokenize_example(spec, tokenizer, dataset[dataset_index])
        embedded = transform(_numpy_token_inputs(dict(batch)))["embeddings"]
        embedded = np.ascontiguousarray(embedded, dtype=np.float32)
        if embedded.ndim != 3 or embedded.shape[0] != 1:
            raise ValueError(
                f"calibration embedding must have shape [1,L,{embedding_width}]"
            )
        if embedded.shape[2] != embedding_width:
            raise ValueError(
                "calibration embedding width mismatch: "
                f"expected {embedding_width}, got {embedded.shape[2]}"
            )
        np.save(calibration_dir / f"{output_index:03d}.npy", embedded)
        sequence_lengths.append(int(embedded.shape[1]))

    manifest = {
        **contract_to_dict(spec),
        "dataset_size": len(dataset),
        "calibration_indices": list(indices),
        "calibration_files": len(sequence_lengths),
        "sequence_lengths": sequence_lengths,
        "sequence_length_min": min(sequence_lengths),
        "sequence_length_max": max(sequence_lengths),
        "weights": {
            "path": "weights/weight_dict.pth",
            "size_bytes": weights_path.stat().st_size,
            "sha256": sha256_file(weights_path),
        },
    }
    (task_root / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare calibration inputs for Mobilint BERT qbcompiler 1.2",
    )
    parser.add_argument("--task", required=True, choices=("sst2", "squad1"))
    parser.add_argument("--output-root", type=Path, default=Path("./artifacts"))
    parser.add_argument("--describe", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _create_parser().parse_args(argv)
    if args.describe:
        print(json.dumps(contract_to_dict(get_task_spec(args.task)), indent=2))
        return 0

    manifest = prepare_task(args.task, args.output_root)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
