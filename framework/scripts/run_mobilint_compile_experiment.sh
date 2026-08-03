#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd -- "$FRAMEWORK_DIR/.." && pwd)"

EXPECTED_WHEEL_NAME="qbcompiler-1.2.0-py3-none-any.whl"
EXPECTED_WHEEL_SHA256="28f276baef1bff86ed313cb819b53d8abb684a7555cf4c81c459edc09abf1b4b"

usage() {
  cat <<'EOF'
Mobilint qbcompiler 1.2 multi-model compile experiment

Usage:
  bash framework/scripts/run_mobilint_compile_experiment.sh \
    --wheel PATH --model NAME --output-root PATH [options]

Models:
  bert-sst2       Existing BERT SST-2 package (variant: default)
  bert-squad1     Existing BERT SQuAD v1 package (variant: default)
  patchtst-etth1  PatchTST ETTh1 (variant: stock or compat-static-patchifier)
  resnet50        TorchVision ResNet-50 (variant: default)
  yolov5m         Pinned YOLOv5m raw heads (variant: default)

Required:
  --wheel PATH          Official qbcompiler 1.2.0 wheel
  --model NAME          One exact model name listed above
  --output-root PATH    Parent directory for immutable timestamp attempts

Options:
  --python PATH         CPython 3.10 (default: python3.10 on PATH)
  --venv PATH           Compiler venv (default: REPO/.venv-qbcompiler-1.2-py310)
  --variant NAME        Recipe variant (model-specific default)
  --dataset PATH        ETTh1 CSV, ImageNet root, or COCO image root
  --model-revision REV  PatchTST revision (exact SHA for compat retry)
  --yolov5-root PATH    Pinned YOLOv5 checkout
  --weights PATH        Non-empty yolov5m.pt
  --parent-attempt PATH Parent attempt provenance for a retry
  -h, --help            Show this dependency-free help
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

WHEEL_PATH=""
PYTHON_BIN=""
VENV_DIR="$REPO_DIR/.venv-qbcompiler-1.2-py310"
MODEL=""
VARIANT=""
OUTPUT_ROOT=""
DATASET=""
MODEL_REVISION="main"
YOLOV5_ROOT=""
WEIGHTS=""
PARENT_ATTEMPT=""
ATTEMPT_ROOT=""
EXPERIMENT_EXIT_CODE=0

while (($#)); do
  case "$1" in
    --wheel)
      (($# >= 2)) || die "--wheel requires a path"
      WHEEL_PATH="$2"
      shift 2
      ;;
    --python)
      (($# >= 2)) || die "--python requires a path"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv)
      (($# >= 2)) || die "--venv requires a path"
      VENV_DIR="$2"
      shift 2
      ;;
    --model)
      (($# >= 2)) || die "--model requires a name"
      MODEL="$2"
      shift 2
      ;;
    --variant)
      (($# >= 2)) || die "--variant requires a name"
      VARIANT="$2"
      shift 2
      ;;
    --output-root)
      (($# >= 2)) || die "--output-root requires a path"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --dataset)
      (($# >= 2)) || die "--dataset requires a path"
      DATASET="$2"
      shift 2
      ;;
    --model-revision)
      (($# >= 2)) || die "--model-revision requires a revision"
      MODEL_REVISION="$2"
      shift 2
      ;;
    --yolov5-root)
      (($# >= 2)) || die "--yolov5-root requires a path"
      YOLOV5_ROOT="$2"
      shift 2
      ;;
    --weights)
      (($# >= 2)) || die "--weights requires a path"
      WEIGHTS="$2"
      shift 2
      ;;
    --parent-attempt)
      (($# >= 2)) || die "--parent-attempt requires a path"
      PARENT_ATTEMPT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "$WHEEL_PATH" ]] || die "--wheel is required"
[[ -n "$MODEL" ]] || die "--model is required"
[[ -n "$OUTPUT_ROOT" ]] || die "--output-root is required"

case "$MODEL" in
  bert-sst2|bert-squad1)
    [[ -n "$VARIANT" ]] || VARIANT="default"
    [[ "$VARIANT" == "default" ]] || die "$MODEL requires --variant default"
    ;;
  patchtst-etth1)
    [[ -n "$VARIANT" ]] || VARIANT="stock"
    case "$VARIANT" in
      stock|compat-static-patchifier) ;;
      *) die "patchtst-etth1 variant must be stock or compat-static-patchifier" ;;
    esac
    [[ -n "$DATASET" ]] || die "--dataset is required for patchtst-etth1"
    if [[ "$VARIANT" == "compat-static-patchifier" ]]; then
      [[ "$MODEL_REVISION" =~ ^[0-9a-f]{40}$ ]] ||
        die "compat-static-patchifier requires an exact lowercase --model-revision SHA"
      [[ -n "$PARENT_ATTEMPT" ]] ||
        die "compat-static-patchifier requires --parent-attempt"
      [[ -s "$PARENT_ATTEMPT/result.json" ]] ||
        die "--parent-attempt must contain result.json"
    fi
    ;;
  resnet50)
    [[ -n "$VARIANT" ]] || VARIANT="default"
    [[ "$VARIANT" == "default" ]] || die "resnet50 requires --variant default"
    [[ -n "$DATASET" ]] || die "--dataset is required for resnet50"
    ;;
  yolov5m)
    [[ -n "$VARIANT" ]] || VARIANT="default"
    [[ "$VARIANT" == "default" ]] || die "yolov5m requires --variant default"
    [[ -n "$DATASET" ]] || die "--dataset is required for yolov5m"
    [[ -n "$YOLOV5_ROOT" ]] || die "--yolov5-root is required for yolov5m"
    [[ -n "$WEIGHTS" ]] || die "--weights is required for yolov5m"
    ;;
  *)
    die "unsupported --model: $MODEL"
    ;;
esac

if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3.10 || true)"
fi
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || die "CPython 3.10 executable not found"

[[ -r /etc/os-release ]] || die "/etc/os-release is unavailable"
# shellcheck disable=SC1091
. /etc/os-release
[[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "22.04" ]] ||
  die "qbcompiler reproduction requires Ubuntu 22.04"
[[ "$(uname -m)" == "x86_64" ]] ||
  die "qbcompiler reproduction requires x86_64"

"$PYTHON_BIN" - <<'PY' || die "qbcompiler reproduction requires CPython 3.10"
import platform
import sys

if platform.python_implementation() != "CPython" or sys.version_info[:2] != (3, 10):
    raise SystemExit(1)
PY

[[ -f "$WHEEL_PATH" ]] || die "wheel not found: $WHEEL_PATH"
[[ "$(basename -- "$WHEEL_PATH")" == "$EXPECTED_WHEEL_NAME" ]] ||
  die "wheel filename must be $EXPECTED_WHEEL_NAME"
ACTUAL_WHEEL_SHA256="$(sha256sum "$WHEEL_PATH" | awk '{print $1}')"
[[ "$ACTUAL_WHEEL_SHA256" == "$EXPECTED_WHEEL_SHA256" ]] ||
  die "wheel SHA256 mismatch: $ACTUAL_WHEEL_SHA256"
export MOBILINT_QBCOMPILER_WHEEL_NAME="$EXPECTED_WHEEL_NAME"
export MOBILINT_QBCOMPILER_WHEEL_SHA256="$ACTUAL_WHEEL_SHA256"
export PYTHONPATH="$FRAMEWORK_DIR:$FRAMEWORK_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

METADATA_JSON="{}"
if [[ -n "$PARENT_ATTEMPT" ]]; then
  METADATA_JSON="$("$PYTHON_BIN" -c \
    'import json,sys; print(json.dumps({"parent_attempt": sys.argv[1]}))' \
    "$PARENT_ATTEMPT")" || die "could not encode attempt metadata"
fi
ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%S%NZ)-$$"
ATTEMPT_ROOT="$("$PYTHON_BIN" -m tools.mobilint_compile_recipes.attempt create \
  --output-root "$OUTPUT_ROOT" \
  --attempt-id "$ATTEMPT_ID" \
  --model "$MODEL" \
  --variant "$VARIANT" \
  --metadata-json "$METADATA_JSON")" || die "could not create compile attempt"
ATTEMPT_ROOT="$(cd -- "$ATTEMPT_ROOT" && pwd)"

finish() {
  local shell_code=$?
  if ((EXPERIMENT_EXIT_CODE == 0 && shell_code != 0)); then
    EXPERIMENT_EXIT_CODE=$shell_code
  fi
  printf 'ATTEMPT_ROOT=%s\n' "$ATTEMPT_ROOT"
  printf 'EXPERIMENT_EXIT_CODE=%s\n' "$EXPERIMENT_EXIT_CODE"
}
trap finish EXIT

fail_after_attempt() {
  local code=$1
  shift
  EXPERIMENT_EXIT_CODE=$code
  printf 'ERROR: %s\n' "$*" >&2
  exit "$code"
}

if [[ -e "$VENV_DIR" && ! -x "$VENV_DIR/bin/python" ]]; then
  fail_after_attempt 1 "existing --venv is not a usable Python environment: $VENV_DIR"
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR" ||
    fail_after_attempt $? "failed to create compiler venv"
fi
COMPILER_PY="$VENV_DIR/bin/python"
"$COMPILER_PY" - <<'PY' || fail_after_attempt 1 "compiler venv must use CPython 3.10"
import platform
import sys

if platform.python_implementation() != "CPython" or sys.version_info[:2] != (3, 10):
    raise SystemExit(1)
PY

if ! "$COMPILER_PY" -m pip --version >/dev/null 2>&1; then
  "$COMPILER_PY" -m ensurepip --upgrade ||
    fail_after_attempt $? "ensurepip failed"
fi

export PIP_DISABLE_PIP_VERSION_CHECK=1
"$COMPILER_PY" -m pip install --upgrade pip setuptools wheel ||
  fail_after_attempt $? "pip bootstrap failed"
"$COMPILER_PY" -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 \
  torchvision==0.22.1 ||
  fail_after_attempt $? "PyTorch dependency installation failed"
"$COMPILER_PY" -m pip install \
  "$WHEEL_PATH" \
  numpy==1.26.0 \
  tensorflow==2.17.0 \
  onnx==1.16.2 \
  onnxruntime==1.19.2 \
  opencv-python==4.11.0.86 \
  transformers==4.57.1 \
  datasets==3.6.0 ||
  fail_after_attempt $? "compiler dependency installation failed"
"$COMPILER_PY" -m pip check || fail_after_attempt $? "pip check failed"

"$COMPILER_PY" - <<'PY' || fail_after_attempt 1 "qbcompiler signature check failed"
import inspect
import platform
import sys

import onnxruntime
import qbcompiler

print("python:", sys.version)
print("platform:", platform.platform())
print("onnxruntime:", onnxruntime.__version__)
print("qbcompiler:", getattr(qbcompiler, "__version__", "unknown"))
print("mblt_compile:", inspect.signature(qbcompiler.mblt_compile))
print("mxq_compile:", inspect.signature(qbcompiler.mxq_compile))
PY

run_stage() {
  local stage=$1
  shift
  "$COMPILER_PY" -m tools.mobilint_compile_recipes.attempt run \
    --attempt-root "$ATTEMPT_ROOT" \
    --stage "$stage" \
    -- "$@"
  local code=$?
  if ((code != 0)); then
    EXPERIMENT_EXIT_CODE=$code
    exit "$code"
  fi
}

record_artifact() {
  local path=$1
  "$COMPILER_PY" -m tools.mobilint_compile_recipes.attempt artifact \
    --attempt-root "$ATTEMPT_ROOT" \
    --artifact "$path" || fail_after_attempt $? "failed to record artifact: $path"
}

validate_calibration=(
  "$COMPILER_PY"
  -c
  'import json,sys; from pathlib import Path; marker,root,model=sys.argv[1:4]; p=Path(root); manifest=p/("calibration_manifest.json" if model.startswith("bert-") else "source-manifest.json"); report=p/"compile-report.json"; calibration=p/("calibration_data" if model.startswith("bert-") else "calibration"); [(json.loads(x.read_text(encoding="utf-8"))) for x in (manifest,report)]; assert calibration.is_dir() and any(calibration.iterdir())'
  CALIBRATION_EVIDENCE
)

case "$MODEL" in
  bert-sst2|bert-squad1)
    if [[ "$MODEL" == "bert-sst2" ]]; then
      BERT_TASK="sst2"
    else
      BERT_TASK="squad1"
    fi
    BERT_TASK_ROOT="$ATTEMPT_ROOT/$BERT_TASK"
    run_stage SOURCE_PREPARE \
      "$COMPILER_PY" -m tools.mobilint_bert_compile.prepare \
      --task "$BERT_TASK" --output-root "$ATTEMPT_ROOT"
    run_stage SOURCE_SMOKE \
      "$COMPILER_PY" -c \
      'import sys; from tools.mobilint_bert_compile.common import get_task_spec; from tools.mobilint_bert_compile.compile import _load_model_and_feed; _load_model_and_feed(get_task_spec(sys.argv[1]))' \
      "$BERT_TASK"
    run_stage CALIBRATION_PREPARE \
      "${validate_calibration[@]}" "$BERT_TASK_ROOT" "$MODEL"
    run_stage MBLT_COMPILE \
      "$COMPILER_PY" -m tools.mobilint_bert_compile.compile \
      --task "$BERT_TASK" --stage mblt --artifact-root "$ATTEMPT_ROOT"
    record_artifact "$BERT_TASK_ROOT/mblt/$BERT_TASK.mblt"
    run_stage MXQ_COMPILE \
      "$COMPILER_PY" -m tools.mobilint_bert_compile.compile \
      --task "$BERT_TASK" --stage mxq --artifact-root "$ATTEMPT_ROOT"
    record_artifact "$BERT_TASK_ROOT/mxq/$BERT_TASK.mxq"
    "$COMPILER_PY" -m tools.mobilint_compile_recipes.bert_bridge \
      --task-root "$BERT_TASK_ROOT" --output "$ATTEMPT_ROOT/result.json" ||
      fail_after_attempt $? "BERT compile evidence import failed"
    ;;
  patchtst-etth1)
    RECIPE_MODULE="tools.mobilint_compile_recipes.patchtst_etth1"
    run_stage SOURCE_PREPARE \
      "$COMPILER_PY" -m "$RECIPE_MODULE" --stage prepare \
      --variant "$VARIANT" --attempt-root "$ATTEMPT_ROOT" \
      --dataset "$DATASET" --model-revision "$MODEL_REVISION"
    run_stage SOURCE_SMOKE \
      "$COMPILER_PY" -m "$RECIPE_MODULE" --stage source-smoke \
      --variant "$VARIANT" --attempt-root "$ATTEMPT_ROOT"
    run_stage CALIBRATION_PREPARE \
      "${validate_calibration[@]}" "$ATTEMPT_ROOT" "$MODEL"
    run_stage MBLT_COMPILE \
      "$COMPILER_PY" -m "$RECIPE_MODULE" --stage mblt \
      --variant "$VARIANT" --attempt-root "$ATTEMPT_ROOT"
    record_artifact "$ATTEMPT_ROOT/mblt/patchtst-etth1-mblt.mblt"
    run_stage MXQ_COMPILE \
      "$COMPILER_PY" -m "$RECIPE_MODULE" --stage mxq \
      --variant "$VARIANT" --attempt-root "$ATTEMPT_ROOT"
    record_artifact "$ATTEMPT_ROOT/mxq/patchtst-etth1-mxq.mxq"
    ;;
  resnet50)
    RECIPE_MODULE="tools.mobilint_compile_recipes.resnet50"
    run_stage SOURCE_PREPARE \
      "$COMPILER_PY" -m "$RECIPE_MODULE" --stage prepare \
      --variant "$VARIANT" --attempt-root "$ATTEMPT_ROOT" --dataset "$DATASET"
    run_stage SOURCE_SMOKE \
      "$COMPILER_PY" -m "$RECIPE_MODULE" --stage source-smoke \
      --variant "$VARIANT" --attempt-root "$ATTEMPT_ROOT"
    run_stage CALIBRATION_PREPARE \
      "${validate_calibration[@]}" "$ATTEMPT_ROOT" "$MODEL"
    run_stage MBLT_COMPILE \
      "$COMPILER_PY" -m "$RECIPE_MODULE" --stage mblt \
      --variant "$VARIANT" --attempt-root "$ATTEMPT_ROOT"
    record_artifact "$ATTEMPT_ROOT/mblt/resnet50-mblt.mblt"
    run_stage MXQ_COMPILE \
      "$COMPILER_PY" -m "$RECIPE_MODULE" --stage mxq \
      --variant "$VARIANT" --attempt-root "$ATTEMPT_ROOT"
    record_artifact "$ATTEMPT_ROOT/mxq/resnet50-mxq.mxq"
    ;;
  yolov5m)
    RECIPE_MODULE="tools.mobilint_compile_recipes.yolov5m"
    run_stage SOURCE_PREPARE \
      "$COMPILER_PY" -m "$RECIPE_MODULE" --stage prepare \
      --variant "$VARIANT" --attempt-root "$ATTEMPT_ROOT" --dataset "$DATASET" \
      --yolov5-root "$YOLOV5_ROOT" --weights "$WEIGHTS"
    run_stage SOURCE_SMOKE \
      "$COMPILER_PY" -m "$RECIPE_MODULE" --stage source-smoke \
      --variant "$VARIANT" --attempt-root "$ATTEMPT_ROOT"
    run_stage CALIBRATION_PREPARE \
      "${validate_calibration[@]}" "$ATTEMPT_ROOT" "$MODEL"
    run_stage MBLT_COMPILE \
      "$COMPILER_PY" -m "$RECIPE_MODULE" --stage mblt \
      --variant "$VARIANT" --attempt-root "$ATTEMPT_ROOT"
    record_artifact "$ATTEMPT_ROOT/mblt/yolov5m-mblt.mblt"
    run_stage MXQ_COMPILE \
      "$COMPILER_PY" -m "$RECIPE_MODULE" --stage mxq \
      --variant "$VARIANT" --attempt-root "$ATTEMPT_ROOT"
    record_artifact "$ATTEMPT_ROOT/mxq/yolov5m-mxq.mxq"
    ;;
esac

EXPERIMENT_EXIT_CODE=0
exit 0
