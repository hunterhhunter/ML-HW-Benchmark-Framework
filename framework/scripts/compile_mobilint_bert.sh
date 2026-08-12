#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd -- "$FRAMEWORK_DIR/.." && pwd)"

EXPECTED_WHEEL_NAME="qbcompiler-1.2.0-py3-none-any.whl"
EXPECTED_WHEEL_SHA256="28f276baef1bff86ed313cb819b53d8abb684a7555cf4c81c459edc09abf1b4b"

usage() {
  cat <<'EOF'
Mobilint BERT SST-2/SQuAD v1 compiler-only reproduction

Usage:
  bash framework/scripts/compile_mobilint_bert.sh \
    --wheel ~/Downloads/qbcompiler-1.2.0-py3-none-any.whl \
    --python "$(command -v python3.10)" \
    --task all \
    --output-root "$PWD/mobilint-bert-artifacts"

Required:
  --wheel PATH        Official qbcompiler 1.2.0 wheel
  --output-root PATH  Artifact root; existing task directories are rejected

Optional:
  --python PATH       CPython 3.10 interpreter (default: python3.10 on PATH)
  --venv PATH         Compiler venv (default: REPO/.venv-qbcompiler-1.2-py310)
  --task NAME         sst2, squad1, or all (default: all)
  -h, --help          Show this help
EOF
}

# Mobilint BERT SST-2/SQuAD v1 compiler-only reproduction
#
# Usage:
#   bash framework/scripts/compile_mobilint_bert.sh \
#     --wheel ~/Downloads/qbcompiler-1.2.0-py3-none-any.whl \
#     --python "$(command -v python3.10)" \
#     --task all \
#     --output-root "$PWD/mobilint-bert-artifacts"
#
# Required:
#   --wheel PATH        Official qbcompiler 1.2.0 wheel
#   --output-root PATH  Artifact root; existing task directories are rejected
#
# Optional:
#   --python PATH       CPython 3.10 interpreter (default: python3.10 on PATH)
#   --venv PATH         Compiler venv (default: REPO/.venv-qbcompiler-1.2-py310)
#   --task NAME         sst2, squad1, or all (default: all)
#   -h, --help          Show this help

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

WHEEL_PATH=""
PYTHON_BIN=""
VENV_DIR="$REPO_DIR/.venv-qbcompiler-1.2-py310"
TASK="all"
OUTPUT_ROOT=""

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
    --task)
      (($# >= 2)) || die "--task requires sst2, squad1, or all"
      TASK="$2"
      shift 2
      ;;
    --output-root)
      (($# >= 2)) || die "--output-root requires a path"
      OUTPUT_ROOT="$2"
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
[[ -n "$OUTPUT_ROOT" ]] || die "--output-root is required"
case "$TASK" in
  sst2|squad1|all) ;;
  *) die "--task must be sst2, squad1, or all" ;;
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

if [[ -e "$OUTPUT_ROOT" && ! -d "$OUTPUT_ROOT" ]]; then
  die "--output-root exists and is not a directory: $OUTPUT_ROOT"
fi
if [[ "$TASK" == "all" ]]; then
  TASKS=(sst2 squad1)
else
  TASKS=("$TASK")
fi
for task_name in "${TASKS[@]}"; do
  [[ ! -e "$OUTPUT_ROOT/$task_name" ]] ||
    die "task output already exists: $OUTPUT_ROOT/$task_name"
done
mkdir -p "$OUTPUT_ROOT"

if [[ -e "$VENV_DIR" && ! -x "$VENV_DIR/bin/python" ]]; then
  die "existing --venv is not a usable Python environment: $VENV_DIR"
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
COMPILER_PY="$VENV_DIR/bin/python"
"$COMPILER_PY" - <<'PY' || die "compiler venv must use CPython 3.10"
import platform
import sys

if platform.python_implementation() != "CPython" or sys.version_info[:2] != (3, 10):
    raise SystemExit(1)
PY

if ! "$COMPILER_PY" -m pip --version >/dev/null 2>&1; then
  "$COMPILER_PY" -m ensurepip --upgrade
fi

export PIP_DISABLE_PIP_VERSION_CHECK=1
"$COMPILER_PY" -m pip install --upgrade pip setuptools wheel
"$COMPILER_PY" -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 \
  torchvision==0.22.1
"$COMPILER_PY" -m pip install \
  "$WHEEL_PATH" \
  numpy==1.26.0 \
  tensorflow==2.17.0 \
  onnx==1.16.2 \
  onnxruntime==1.19.2 \
  opencv-python==4.11.0.86 \
  transformers==4.57.1 \
  datasets==3.6.0
"$COMPILER_PY" -m pip check

"$COMPILER_PY" - <<'PY'
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

export PYTHONPATH="$FRAMEWORK_DIR:$FRAMEWORK_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
for task_name in "${TASKS[@]}"; do
  printf '\n===== Preparing %s =====\n' "$task_name"
  "$COMPILER_PY" -m tools.mobilint_bert_compile.prepare \
    --task "$task_name" \
    --output-root "$OUTPUT_ROOT"

  printf '\n===== Compiling %s for aries-rb =====\n' "$task_name"
  "$COMPILER_PY" -m tools.mobilint_bert_compile.compile \
    --task "$task_name" \
    --stage mblt \
    --artifact-root "$OUTPUT_ROOT"
  "$COMPILER_PY" -m tools.mobilint_bert_compile.compile \
    --task "$task_name" \
    --stage mxq \
    --artifact-root "$OUTPUT_ROOT"
done

printf '\n===== Generated compiler artifacts =====\n'
find "$OUTPUT_ROOT" -type f \
  \( -name '*.mblt' -o -name '*.mxq' -o -name 'weight_dict.pth' \) \
  -printf '%p\t%s bytes\n' | sort
find "$OUTPUT_ROOT" -type f \
  \( -name '*.mblt' -o -name '*.mxq' -o -name 'weight_dict.pth' \) \
  -print0 | sort -z | xargs -0 -r sha256sum
printf '\nCOMPILATION_COMPLETE=%s\n' "$OUTPUT_ROOT"
