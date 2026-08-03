"""Validate recorded Mobilint compiler artifacts with one ARIES inference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

from tools.mobilint_compile_recipes.attempt import (
    STAGES,
    _attempt_lock,
    _attempt_root,
    _load_result,
    _refresh_independent_statuses,
    _save_result,
    _stage_status,
)
from tools.mobilint_compile_recipes.contracts import (
    TensorContract,
    get_recipe,
    sha256_file,
)


_HARDWARE_STAGES = ("ARIES_LOAD", "CONTRACT_CHECK", "TASK_SMOKE")
_STAGE_FIELDS = {
    "status",
    "started_at",
    "finished_at",
    "elapsed_seconds",
    "exit_code",
    "signal",
    "error",
}
_STATUS_VALUES = {"not_run", "pass", "fail"}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_QBRUNTIME_VERSIONS = {"1.3.2", "v1.3.2"}


@dataclass(frozen=True)
class _RuntimeSpec:
    model: str
    variant: str
    core_mode: str
    inputs: tuple[TensorContract, ...]
    outputs: tuple[TensorContract, ...]
    bert_task: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strict_result(result: Mapping[str, Any]) -> None:
    if not isinstance(result, Mapping):
        raise ValueError("attempt result must be a JSON object")
    if not isinstance(result.get("model"), str) or not result["model"]:
        raise ValueError("attempt result model is missing")
    if not isinstance(result.get("variant"), str) or not result["variant"]:
        raise ValueError("attempt result variant is missing")
    stages = result.get("stages")
    if not isinstance(stages, Mapping) or tuple(stages) != STAGES:
        raise ValueError("attempt result has invalid stage schema")
    for stage in STAGES:
        record = stages[stage]
        if not isinstance(record, Mapping) or not _STAGE_FIELDS.issubset(record):
            raise ValueError(f"attempt result stage {stage} has invalid schema")
        if record.get("status") not in _STATUS_VALUES:
            raise ValueError(f"attempt result stage {stage} has invalid status")
        status = record["status"]
        if status == "not_run":
            if any(record.get(field) is not None for field in _STAGE_FIELDS - {"status"}):
                raise ValueError(
                    f"attempt result stage {stage} not_run fields are incoherent"
                )
            continue
        if (
            not isinstance(record.get("started_at"), str)
            or not record["started_at"]
            or not isinstance(record.get("finished_at"), str)
            or not record["finished_at"]
            or isinstance(record.get("elapsed_seconds"), bool)
            or not isinstance(record.get("elapsed_seconds"), (int, float))
            or not math.isfinite(record["elapsed_seconds"])
            or record["elapsed_seconds"] < 0
            or isinstance(record.get("exit_code"), bool)
            or not isinstance(record.get("exit_code"), int)
        ):
            raise ValueError(
                f"attempt result stage {stage} completed fields are incoherent"
            )
        if status == "pass" and (
            record["exit_code"] != 0
            or record.get("signal") is not None
            or record.get("error") is not None
        ):
            raise ValueError(f"attempt result stage {stage} pass fields are incoherent")
        if status == "fail":
            expected_signal = -record["exit_code"] if record["exit_code"] < 0 else None
            if (
                record["exit_code"] == 0
                or record.get("signal") != expected_signal
                or not isinstance(record.get("error"), str)
                or not record["error"]
            ):
                raise ValueError(
                    f"attempt result stage {stage} fail fields are incoherent"
                )
    for field in (
        "compile_status",
        "runtime_status",
        "contract_status",
        "quality_status",
    ):
        if result.get(field) not in _STATUS_VALUES:
            raise ValueError(f"attempt result {field} has invalid status")
    derived = {
        "compile_status": _stage_status(result, ("MBLT_COMPILE", "MXQ_COMPILE")),
        "runtime_status": _stage_status(result, ("ARIES_LOAD", "TASK_SMOKE")),
        "contract_status": _stage_status(result, ("CONTRACT_CHECK",)),
    }
    for field, expected in derived.items():
        if result[field] != expected:
            raise ValueError(
                f"attempt result {field} is inconsistent with stage status: "
                f"expected {expected}, got {result[field]}"
            )
    failed_stages = [stage for stage in STAGES if stages[stage]["status"] == "fail"]
    expected_failed_at = failed_stages[0] if failed_stages else None
    if result.get("failed_at") != expected_failed_at:
        raise ValueError(
            "attempt result failed_at is inconsistent with failed stages: "
            f"expected {expected_failed_at!r}, got {result.get('failed_at')!r}"
        )
    hardware_statuses = [stages[stage]["status"] for stage in _HARDWARE_STAGES]
    has_runtime_evidence = "runtime_verification" in result
    if all(status == "not_run" for status in hardware_statuses):
        if has_runtime_evidence:
            raise ValueError(
                "attempt result has runtime evidence while hardware stages are not_run"
            )
    elif "not_run" in hardware_statuses:
        raise ValueError("attempt result has partial hardware stage evidence")
    elif not has_runtime_evidence or not isinstance(
        result.get("runtime_verification"), Mapping
    ):
        raise ValueError("attempt result completed hardware stages lack runtime evidence")
    if not isinstance(result.get("artifacts"), list):
        raise ValueError("attempt result artifacts must be a list")
    artifact_paths: set[str] = set()
    for record in result["artifacts"]:
        if not isinstance(record, Mapping):
            raise ValueError("attempt result artifact record schema is invalid")
        path = record.get("path")
        size = record.get("size_bytes")
        digest = record.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or type(size) is not int
            or size <= 0
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or path in artifact_paths
        ):
            raise ValueError("attempt result artifact record schema is invalid")
        artifact_paths.add(path)


def _require_runtime_not_recorded(result: Mapping[str, Any]) -> None:
    recorded = [
        stage
        for stage in _HARDWARE_STAGES
        if result["stages"][stage]["status"] != "not_run"
    ]
    if recorded:
        raise ValueError(
            "ARIES runtime evidence is already recorded for this immutable attempt: "
            + ", ".join(recorded)
        )


def _require_compiled(result: Mapping[str, Any]) -> None:
    if (
        result.get("compile_status") != "pass"
        or result["stages"]["MXQ_COMPILE"]["status"] != "pass"
    ):
        raise ValueError(
            "runtime verification requires MXQ_COMPILE status pass and "
            "compile_status pass"
        )


def _finish_stage(
    result: dict[str, Any],
    stage: str,
    *,
    status: str,
    started_at: str,
    started: float,
    error: BaseException | None = None,
    finished_at: str | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    record = result["stages"][stage]
    if record["status"] != "not_run":
        raise ValueError(f"stage already recorded: {stage}")
    message = None
    if error is not None:
        message = f"{type(error).__name__}: {error}"
    record.update(
        {
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at or _utc_now(),
            "elapsed_seconds": (
                max(0.0, elapsed_seconds)
                if elapsed_seconds is not None
                else max(0.0, time.monotonic() - started)
            ),
            "exit_code": 0 if status == "pass" else 1,
            "signal": None,
            "error": message,
        }
    )
    if status == "fail" and result.get("failed_at") is None:
        result["failed_at"] = stage
    _refresh_independent_statuses(result)


def _record_failure(
    root: Path,
    result: dict[str, Any],
    stage: str,
    error: BaseException,
    *,
    started_at: str,
    started: float,
) -> None:
    _finish_stage(
        result,
        stage,
        status="fail",
        started_at=started_at,
        started=started,
        error=error,
    )
    result["runtime_verification"] = {
        **dict(result.get("runtime_verification", {})),
        "error_stage": stage,
        "error": f"{type(error).__name__}: {error}",
    }
    _save_result(root, result)


def _relative_path(root: Path, path: str | Path, label: str) -> tuple[Path, str]:
    candidate = Path(path).expanduser().resolve()
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"{label} must be inside the attempt root") from error
    return candidate, relative


def _validate_artifact(
    root: Path, result: Mapping[str, Any], artifact: str | Path
) -> tuple[Path, dict[str, object]]:
    path, relative = _relative_path(root, artifact, "MXQ artifact")
    if path.suffix.lower() != ".mxq":
        raise ValueError("runtime artifact must use the .mxq suffix")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"runtime artifact must be a non-empty file: {path}")
    matches = [
        record
        for record in result["artifacts"]
        if isinstance(record, Mapping) and record.get("path") == relative
    ]
    if len(matches) != 1:
        raise ValueError(
            f"MXQ must have exactly one recorded artifact entry: {relative}"
        )
    record = matches[0]
    if type(record.get("size_bytes")) is not int or record["size_bytes"] <= 0:
        raise ValueError("recorded artifact size is invalid")
    if record["size_bytes"] != path.stat().st_size:
        raise ValueError("MXQ artifact size does not match recorded evidence")
    actual_hash = sha256_file(path)
    if record.get("sha256") != actual_hash:
        raise ValueError("MXQ artifact SHA256 does not match recorded evidence")
    return path, {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": actual_hash,
    }


def _runtime_spec(result: Mapping[str, Any]) -> _RuntimeSpec:
    model = result["model"]
    variant = result["variant"]
    if model in {"bert-sst2", "bert-squad1"}:
        from tools.mobilint_bert_compile.common import get_task_spec

        task = "sst2" if model == "bert-sst2" else "squad1"
        source = get_task_spec(task)
        if variant != "default":
            raise ValueError(f"{model} runtime variant must be default")
        if task == "sst2":
            outputs = (TensorContract("logits", (1, 1, 2), "float32"),)
        else:
            outputs = (
                TensorContract("end_logits", (1, -1, 1), "float32"),
                TensorContract("start_logits", (1, -1, 1), "float32"),
            )
        return _RuntimeSpec(
            model=model,
            variant=variant,
            core_mode="single",
            inputs=tuple(
                TensorContract(value.name, value.shape, value.dtype)
                for value in source.mxq_inputs
            ),
            outputs=outputs,
            bert_task=task,
        )
    recipe = get_recipe(model, variant)
    return _RuntimeSpec(
        model=model,
        variant=variant,
        core_mode=recipe.inference_scheme,
        inputs=recipe.runtime_inputs,
        outputs=recipe.outputs,
    )


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_array(
    root: Path,
    relative: object,
    tensor: TensorContract,
    *,
    stored_hash: object,
):
    import numpy as np

    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"saved input {tensor.name} path must be attempt-relative")
    path, normalized = _relative_path(root, root / relative, f"saved input {tensor.name}")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"saved input {tensor.name} is missing or empty: {path}")
    if not isinstance(stored_hash, str) or _SHA256.fullmatch(stored_hash) is None:
        raise ValueError(
            f"saved input {tensor.name} requires a lowercase SHA256 digest"
        )
    if stored_hash != sha256_file(path):
        raise ValueError(f"saved input {tensor.name} SHA256 mismatch")
    try:
        value = np.load(path, allow_pickle=False)
    except Exception as error:
        raise ValueError(f"saved input {tensor.name} is not a valid NumPy array") from error
    if tuple(value.shape) != tensor.shape:
        raise ValueError(
            f"saved input {tensor.name} shape mismatch: expected {tensor.shape}, "
            f"actual {tuple(value.shape)}"
        )
    if value.dtype.name != tensor.dtype:
        raise ValueError(
            f"saved input {tensor.name} dtype mismatch: expected {tensor.dtype}, "
            f"actual {value.dtype.name}"
        )
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"saved input {tensor.name} must contain only finite values")
    contiguous = np.ascontiguousarray(value)
    return contiguous, {
        "name": tensor.name,
        "path": normalized,
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.name,
        "sha256": sha256_file(path),
    }


def _load_recipe_inputs(
    root: Path, spec: _RuntimeSpec
) -> tuple[list[Any], list[dict[str, object]]]:
    manifest = _read_json_object(root / "source-manifest.json", "source manifest")
    for field, expected in (
        ("model", spec.model),
        ("variant", spec.variant),
        ("source_id", get_recipe(spec.model, spec.variant).source_id),
    ):
        if manifest.get(field) != expected:
            raise ValueError(f"source manifest {field} mismatch")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples or not isinstance(samples[0], Mapping):
        raise ValueError("source manifest has no saved smoke input")
    sample = samples[0]
    values: list[Any] = []
    evidence: list[dict[str, object]] = []
    if spec.model == "patchtst-etth1":
        paths = sample.get("paths")
        hashes = sample.get("sha256")
        if not isinstance(paths, Mapping) or tuple(paths) != tuple(
            tensor.name for tensor in spec.inputs
        ):
            raise ValueError("saved PatchTST inputs must follow contract input order")
        if not isinstance(hashes, Mapping) or tuple(hashes) != tuple(
            tensor.name for tensor in spec.inputs
        ):
            raise ValueError("saved PatchTST input SHA256 records must follow contract order")
        for tensor in spec.inputs:
            value, record = _load_array(
                root,
                paths.get(tensor.name),
                tensor,
                stored_hash=hashes.get(tensor.name),
            )
            values.append(value)
            evidence.append(record)
    else:
        if len(spec.inputs) != 1:
            raise ValueError("vision runtime contract must contain one input")
        tensor = spec.inputs[0]
        value, record = _load_array(
            root,
            sample.get("calibration_path"),
            tensor,
            stored_hash=sample.get("calibration_sha256"),
        )
        values.append(value)
        evidence.append(record)
    return values, evidence


def _load_bert_inputs(
    root: Path, result: Mapping[str, Any], spec: _RuntimeSpec
) -> tuple[list[Any], list[dict[str, object]]]:
    from tools.mobilint_bert_compile.common import contract_to_dict, get_task_spec
    from tools.mobilint_bert_compile.compile import validate_calibration_set

    provenance = result.get("bert_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("BERT runtime requires bert_provenance")
    task = spec.bert_task
    assert task is not None
    task_root, relative_task_root = _relative_path(
        root, provenance.get("task_root", ""), "BERT task root"
    )
    if relative_task_root != task:
        raise ValueError("BERT task root does not match the attempt task")
    manifest_name = provenance.get("calibration_manifest")
    if manifest_name != "calibration_manifest.json":
        raise ValueError("BERT calibration manifest provenance is invalid")
    manifest = _read_json_object(
        task_root / manifest_name, "BERT calibration manifest"
    )
    task_spec = get_task_spec(task)
    expected_contract = contract_to_dict(task_spec)
    for field, expected in (
        ("task", task),
        ("model_id", task_spec.model_id),
        ("target_device", task_spec.target_device),
        ("mxq_inputs", expected_contract["mxq_inputs"]),
        ("verified_runtime_outputs", expected_contract["verified_runtime_outputs"]),
    ):
        if manifest.get(field) != expected:
            raise ValueError(f"BERT calibration manifest {field} mismatch")
    paths = validate_calibration_set(task_root, manifest, task_spec)
    tensor = spec.inputs[0]
    sequence_lengths = manifest.get("sequence_lengths")
    concrete = TensorContract(
        tensor.name,
        (1, int(sequence_lengths[0]), tensor.shape[-1]),
        tensor.dtype,
    )
    value, record = _load_array(
        root,
        paths[0].relative_to(root).as_posix(),
        concrete,
        stored_hash=manifest["calibration_artifacts"][0]["sha256"],
    )
    return [value], [record]


def _load_inputs(
    root: Path, result: Mapping[str, Any], spec: _RuntimeSpec
) -> tuple[list[Any], list[dict[str, object]]]:
    if spec.bert_task is not None:
        return _load_bert_inputs(root, result, spec)
    return _load_recipe_inputs(root, spec)


def _resolved_contracts(
    spec: _RuntimeSpec, inputs: Sequence[Any]
) -> tuple[tuple[TensorContract, ...], tuple[TensorContract, ...]]:
    resolved_inputs: list[TensorContract] = []
    dynamic_values: dict[int, int] = {}
    for tensor, value in zip(spec.inputs, inputs, strict=True):
        shape = tuple(int(dimension) for dimension in value.shape)
        resolved_inputs.append(TensorContract(tensor.name, shape, tensor.dtype))
        for index, dimension in enumerate(tensor.shape):
            if dimension == -1:
                dynamic_values[index] = shape[index]
    resolved_outputs: list[TensorContract] = []
    for tensor in spec.outputs:
        shape = tuple(
            dynamic_values.get(index, dimension) if dimension == -1 else dimension
            for index, dimension in enumerate(tensor.shape)
        )
        if -1 in shape:
            raise ValueError(f"cannot resolve dynamic output shape for {tensor.name}")
        resolved_outputs.append(TensorContract(tensor.name, shape, tensor.dtype))
    return tuple(resolved_inputs), tuple(resolved_outputs)


def _dtype_name(value: object) -> str:
    raw = getattr(value, "name", value)
    token = str(raw).rsplit(".", 1)[-1].lower().replace("_", "")
    aliases = {
        "float": "float32",
        "float32": "float32",
        "fp32": "float32",
        "bool": "bool",
        "boolean": "bool",
        "uint8": "uint8",
    }
    try:
        return aliases[token]
    except KeyError as error:
        raise ValueError(f"unsupported qbruntime input dtype: {value!r}") from error


def _input_dtypes(model: object, expected_count: int) -> tuple[str, ...]:
    getter = getattr(model, "get_input_dtypes", None)
    if not callable(getter):
        getter = getattr(model, "get_model_input_data_type", None)
    if not callable(getter):
        raise ValueError("qbruntime model has no supported input dtype metadata API")
    values = getter()
    if isinstance(values, (list, tuple)):
        raw = tuple(values)
    else:
        raw = (values,)
    if len(raw) != expected_count:
        raise ValueError(
            f"qbruntime input dtype count mismatch: expected {expected_count}, "
            f"actual {len(raw)}"
        )
    return tuple(_dtype_name(value) for value in raw)


def _metadata_shapes(model: object, getter_name: str, label: str) -> tuple[tuple[int, ...], ...]:
    getter = getattr(model, getter_name, None)
    if not callable(getter):
        raise ValueError(f"qbruntime model has no {label} metadata API")
    values = getter()
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"qbruntime {label} must be a sequence")
    shapes: list[tuple[int, ...]] = []
    for shape in values:
        if not isinstance(shape, (list, tuple)) or not shape:
            raise ValueError(f"qbruntime {label} contains an invalid shape")
        dimensions: list[int] = []
        for dimension in shape:
            if isinstance(dimension, bool):
                raise ValueError(f"qbruntime {label} contains an invalid dimension")
            try:
                value = int(dimension)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"qbruntime {label} contains an invalid dimension"
                ) from error
            if value == 0 or value < -1:
                raise ValueError(f"qbruntime {label} contains an invalid dimension")
            dimensions.append(value)
        shapes.append(tuple(dimensions))
    return tuple(shapes)


def _boundary_singleton_candidates(shape: tuple[int, ...]):
    leading = 0
    while leading < len(shape) and shape[leading] == 1:
        leading += 1
    trailing = 0
    while trailing < len(shape) and shape[len(shape) - 1 - trailing] == 1:
        trailing += 1
    seen: set[tuple[int, ...]] = set()
    # qbruntime v1.3.2 may add the observed batch/variant singleton axes at
    # the front and one tensor wrapper singleton at the back.  Do not collapse
    # an arbitrary same-sized tensor merely because more axes happen to be 1.
    for left in range(min(leading, 2) + 1):
        for right in range(min(trailing, 1) + 1):
            end = len(shape) - right if right else len(shape)
            if left > end:
                continue
            candidate = shape[left:end]
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


def _shape_representation_matches(
    actual: tuple[int, ...],
    declared: tuple[int, ...],
    resolved: tuple[int, ...],
) -> bool:
    expected_forms = [(declared, resolved)]
    if resolved and resolved[0] == 1:
        expected_forms.append((declared[1:], resolved[1:]))
    for candidate in _boundary_singleton_candidates(actual):
        for declared_form, resolved_form in expected_forms:
            if len(candidate) != len(resolved_form):
                continue
            if all(
                observed == wanted
                or (dynamic == -1 and observed in {-1, wanted})
                for observed, dynamic, wanted in zip(
                    candidate, declared_form, resolved_form, strict=True
                )
            ):
                return True
    return False


def _validate_metadata(
    model: object,
    spec: _RuntimeSpec,
    inputs: tuple[TensorContract, ...],
    outputs: tuple[TensorContract, ...],
) -> dict[str, object]:
    dtypes = _input_dtypes(model, len(inputs))
    expected_dtypes = tuple(tensor.dtype for tensor in inputs)
    if dtypes != expected_dtypes:
        raise ValueError(
            f"qbruntime input dtype mismatch: expected {expected_dtypes}, actual {dtypes}"
        )
    input_shapes = _metadata_shapes(model, "get_model_input_shape", "input shapes")
    if len(input_shapes) != len(inputs):
        raise ValueError(
            f"qbruntime input shape count mismatch: expected {len(inputs)}, "
            f"actual {len(input_shapes)}"
        )
    for actual, declared, resolved in zip(
        input_shapes, spec.inputs, inputs, strict=True
    ):
        if not _shape_representation_matches(actual, declared.shape, resolved.shape):
            raise ValueError(
                f"qbruntime input shape mismatch for {declared.name}: "
                f"expected {resolved.shape}, actual {actual}"
            )
    output_shapes = _metadata_shapes(model, "get_model_output_shape", "output shapes")
    if len(output_shapes) != len(outputs):
        raise ValueError(
            f"qbruntime output count mismatch: expected {len(outputs)}, "
            f"actual {len(output_shapes)}"
        )
    for actual, declared, resolved in zip(
        output_shapes, spec.outputs, outputs, strict=True
    ):
        if not _shape_representation_matches(actual, declared.shape, resolved.shape):
            raise ValueError(
                f"qbruntime output shape mismatch for {declared.name}: "
                f"expected {resolved.shape}, actual {actual}"
            )
    return {
        "input_dtypes": list(dtypes),
        "input_shapes": [list(shape) for shape in input_shapes],
        "output_shapes": [list(shape) for shape in output_shapes],
    }


def _validate_outputs(
    raw_outputs: object,
    declared: tuple[TensorContract, ...],
    resolved: tuple[TensorContract, ...],
) -> tuple[list[Any], list[dict[str, object]]]:
    import numpy as np

    if not isinstance(raw_outputs, (list, tuple)):
        raise ValueError("qbruntime outputs must be a list of arrays")
    if len(raw_outputs) != len(resolved):
        raise ValueError(
            f"qbruntime output count mismatch: expected {len(resolved)}, "
            f"actual {len(raw_outputs)}"
        )
    arrays: list[Any] = []
    evidence: list[dict[str, object]] = []
    for raw, declared_tensor, resolved_tensor in zip(
        raw_outputs, declared, resolved, strict=True
    ):
        value = np.asarray(raw)
        if value.dtype.name != resolved_tensor.dtype:
            raise ValueError(
                f"qbruntime output dtype mismatch for {resolved_tensor.name}: "
                f"expected {resolved_tensor.dtype}, actual {value.dtype.name}"
            )
        if not _shape_representation_matches(
            tuple(value.shape), declared_tensor.shape, resolved_tensor.shape
        ):
            raise ValueError(
                f"qbruntime output shape mismatch for {resolved_tensor.name}: "
                f"expected {resolved_tensor.shape}, actual {tuple(value.shape)}"
            )
        if not bool(np.isfinite(value).all()):
            raise ValueError(
                f"qbruntime output {resolved_tensor.name} must contain only finite values"
            )
        arrays.append(value)
        evidence.append(
            {
                "name": resolved_tensor.name,
                "shape": list(value.shape),
                "dtype": value.dtype.name,
                "finite": True,
            }
        )
    return arrays, evidence


def _configure(sdk: object, core_mode: str):
    config = sdk.ModelConfig()
    if core_mode == "single":
        core_id = sdk.CoreId(sdk.Cluster.Cluster0, sdk.Core.Core0)
        result = config.set_single_core_mode(None, [core_id])
    elif core_mode == "global8":
        result = config.set_global8_core_mode()
    else:
        raise ValueError(f"unsupported recorded core mode: {core_mode}")
    if result is False:
        raise RuntimeError(f"qbruntime core setter rejected core_mode={core_mode}")
    return config


def _require_sdk_version(sdk: object) -> str:
    version = getattr(sdk, "__version__", None)
    if not isinstance(version, str) or version not in _QBRUNTIME_VERSIONS:
        raise ValueError(f"qbruntime 1.3.2 is required; observed {version!r}")
    return version


def _dispose(model: object | None) -> BaseException | None:
    if model is None:
        return None
    try:
        model.dispose()
    except BaseException as error:  # preserve the original runtime failure
        return error
    return None


def verify_runtime(
    attempt_root: str | Path,
    artifact: str | Path,
    qbruntime_module: object | None = None,
) -> dict[str, Any]:
    """Verify one immutable compiler attempt on ARIES and extend its result."""
    root = _attempt_root(attempt_root)
    with _attempt_lock(root):
        result = _load_result(root)
        _strict_result(result)
        _require_compiled(result)
        _require_runtime_not_recorded(result)

        artifact_started_at = _utc_now()
        artifact_started = time.monotonic()
        try:
            artifact_path, artifact_evidence = _validate_artifact(
                root, result, artifact
            )
        except BaseException as error:
            _record_failure(
                root,
                result,
                "ARIES_LOAD",
                error,
                started_at=artifact_started_at,
                started=artifact_started,
            )
            raise

        contract_started_at = _utc_now()
        contract_started = time.monotonic()
        try:
            spec = _runtime_spec(result)
            inputs, input_evidence = _load_inputs(root, result, spec)
            resolved_inputs, resolved_outputs = _resolved_contracts(spec, inputs)
        except BaseException as error:
            _record_failure(
                root,
                result,
                "CONTRACT_CHECK",
                error,
                started_at=contract_started_at,
                started=contract_started,
            )
            raise

        result["runtime_verification"] = {
            "sdk_version": None,
            "artifact": artifact_evidence,
            "core_mode": spec.core_mode,
            "inputs": input_evidence,
        }

        # Saved-input preparation is part of the end-to-end contract check.
        # ARIES timing begins only when the SDK/config/model lifecycle begins.
        aries_started_at = _utc_now()
        aries_started = time.monotonic()
        model = None
        primary_error: BaseException | None = None
        primary_traceback = None
        failure_stage: str | None = None
        dispose_error: BaseException | None = None
        aries_loaded = False
        metadata_validated = False
        inference_completed = False
        outputs_validated = False
        aries_finished_at: str | None = None
        aries_elapsed: float | None = None
        contract_finished_at: str | None = None
        contract_elapsed: float | None = None
        smoke_started_at: str | None = None
        smoke_started: float | None = None
        smoke_finished_at: str | None = None
        smoke_elapsed: float | None = None
        try:
            failure_stage = "ARIES_LOAD"
            sdk = (
                qbruntime_module
                if qbruntime_module is not None
                else importlib.import_module("qbruntime")
            )
            result["runtime_verification"]["sdk_version"] = _require_sdk_version(sdk)
            config = _configure(sdk, spec.core_mode)
            model = sdk.Model(str(artifact_path), config)
            model.launch()
            aries_loaded = True
            aries_finished_at = _utc_now()
            aries_elapsed = time.monotonic() - aries_started

            failure_stage = "CONTRACT_CHECK"
            metadata = _validate_metadata(
                model, spec, resolved_inputs, resolved_outputs
            )
            result["runtime_verification"]["metadata"] = metadata
            metadata_validated = True

            smoke_started_at = _utc_now()
            smoke_started = time.monotonic()
            failure_stage = "TASK_SMOKE"
            payload = inputs[0] if len(inputs) == 1 else list(inputs)
            raw_outputs = model.infer(payload)
            inference_completed = True
            smoke_finished_at = _utc_now()
            smoke_elapsed = time.monotonic() - smoke_started

            failure_stage = "CONTRACT_CHECK"
            _, output_evidence = _validate_outputs(
                raw_outputs, spec.outputs, resolved_outputs
            )
            result["runtime_verification"]["outputs"] = output_evidence
            outputs_validated = True
            contract_finished_at = _utc_now()
            contract_elapsed = time.monotonic() - contract_started
        except BaseException as error:
            primary_error = error
            primary_traceback = error.__traceback__
            if failure_stage == "ARIES_LOAD":
                aries_finished_at = _utc_now()
                aries_elapsed = time.monotonic() - aries_started
            elif failure_stage == "CONTRACT_CHECK":
                contract_finished_at = _utc_now()
                contract_elapsed = time.monotonic() - contract_started
            elif failure_stage == "TASK_SMOKE":
                smoke_finished_at = _utc_now()
                smoke_elapsed = time.monotonic() - smoke_started
        finally:
            dispose_error = _dispose(model)

        if aries_loaded:
            _finish_stage(
                result,
                "ARIES_LOAD",
                status="pass",
                started_at=aries_started_at,
                started=aries_started,
                finished_at=aries_finished_at,
                elapsed_seconds=aries_elapsed,
            )
        else:
            assert primary_error is not None
            _finish_stage(
                result,
                "ARIES_LOAD",
                status="fail",
                started_at=aries_started_at,
                started=aries_started,
                error=primary_error,
                finished_at=aries_finished_at,
                elapsed_seconds=aries_elapsed,
            )

        if metadata_validated and (outputs_validated or inference_completed):
            contract_error = (
                primary_error
                if failure_stage == "CONTRACT_CHECK" and not outputs_validated
                else None
            )
            _finish_stage(
                result,
                "CONTRACT_CHECK",
                status="fail" if contract_error is not None else "pass",
                started_at=contract_started_at,
                started=contract_started,
                error=contract_error,
                finished_at=contract_finished_at,
                elapsed_seconds=contract_elapsed,
            )
        elif aries_loaded and not metadata_validated:
            assert primary_error is not None
            _finish_stage(
                result,
                "CONTRACT_CHECK",
                status="fail",
                started_at=contract_started_at,
                started=contract_started,
                error=primary_error,
                finished_at=contract_finished_at,
                elapsed_seconds=contract_elapsed,
            )

        if smoke_started_at is not None and smoke_started is not None:
            if inference_completed and dispose_error is None:
                _finish_stage(
                    result,
                    "TASK_SMOKE",
                    status="pass",
                    started_at=smoke_started_at,
                    started=smoke_started,
                    finished_at=smoke_finished_at,
                    elapsed_seconds=smoke_elapsed,
                )
            elif failure_stage == "TASK_SMOKE" or dispose_error is not None:
                task_error = (
                    primary_error
                    if failure_stage == "TASK_SMOKE" and primary_error is not None
                    else dispose_error
                )
                assert task_error is not None
                _finish_stage(
                    result,
                    "TASK_SMOKE",
                    status="fail",
                    started_at=smoke_started_at,
                    started=smoke_started,
                    error=task_error,
                    finished_at=(
                        _utc_now() if dispose_error is not None else smoke_finished_at
                    ),
                    elapsed_seconds=(
                        time.monotonic() - smoke_started
                        if dispose_error is not None
                        else smoke_elapsed
                    ),
                )

        verification = result["runtime_verification"]
        if primary_error is not None:
            verification["error_stage"] = failure_stage
            verification["error"] = (
                f"{type(primary_error).__name__}: {primary_error}"
            )
        if dispose_error is not None:
            verification["dispose_error"] = (
                f"{type(dispose_error).__name__}: {dispose_error}"
            )

        try:
            _save_result(root, result)
        except BaseException:
            if primary_error is not None:
                raise primary_error.with_traceback(primary_traceback)
            raise

        if primary_error is not None:
            raise primary_error.with_traceback(primary_traceback)
        if dispose_error is not None:
            raise dispose_error
        return result


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _create_parser().parse_args(argv)
    result = verify_runtime(args.attempt_root, args.artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
