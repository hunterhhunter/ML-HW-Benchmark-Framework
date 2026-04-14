#!/bin/bash
# 환경 설치 스크립트
# onnxruntime-gpu vs onnxruntime (CPU) 충돌 문제를 해결하기 위해
# constraints.txt와 후처리 제거를 조합합니다.
#
# 문제: ultralytics가 의존성으로 onnxruntime (CPU)을 설치하면
#       onnxruntime-gpu의 CUDA 바인딩을 덮어써서 CUDAExecutionProvider가 사라짐.
# 해결: 3중 방어
#   1. constraints.txt로 CPU 버전 설치 차단 시도
#   2. 설치 후 CPU 버전 감지 시 GPU 버전으로 force-reinstall
#   3. ultralytics 자동 설치 비활성화 환경변수 설정

set -e

# 현재 스크립트 위치 기준으로 framework 디렉토리 결정
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRAMEWORK_VENV="${SCRIPT_DIR}/.venv"

if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON="${VIRTUAL_ENV}/bin/python"
elif [ -d "${FRAMEWORK_VENV}" ]; then
    PYTHON="${FRAMEWORK_VENV}/bin/python"
elif [ -n "$CONDA_PREFIX" ]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
else
    echo "[Error] 활성화된 환경(conda/venv)이 없습니다."
    echo "  framework/.venv도 찾을 수 없습니다. 'python -m venv .venv' 로 생성하세요."
    exit 1
fi

echo "[Info] Python: ${PYTHON}"
UV="uv pip"

echo "[1/4] 패키지 설치 (constraints 적용)..."
${UV} install -r "${SCRIPT_DIR}/requirements.txt" -c "${SCRIPT_DIR}/constraints.txt" --python "${PYTHON}"

echo "[2/4] onnxruntime CPU/GPU 충돌 검사 및 복구..."
# CUDAExecutionProvider가 사용 가능한지 직접 확인
HAS_CUDA=$("${PYTHON}" -c "
try:
    import onnxruntime as ort
    print('yes' if 'CUDAExecutionProvider' in ort.get_available_providers() else 'no')
except:
    print('no')
" 2>/dev/null)

if [ "$HAS_CUDA" = "no" ]; then
    echo "  -> CUDAExecutionProvider 없음. onnxruntime-gpu force-reinstall..."
    ${UV} install onnxruntime-gpu==1.24.4 --force-reinstall --no-deps --python "${PYTHON}"
else
    echo "  -> CUDAExecutionProvider 정상."
fi

echo "[3/4] ultralytics 자동 설치 비활성화 확인..."
# ultralytics가 런타임에 pip install onnxruntime을 실행하지 않도록 환경변수 안내
echo "  [참고] 다음 환경변수를 설정하면 ultralytics의 자동 패키지 설치를 막을 수 있습니다:"
echo "    export YOLO_AUTOINSTALL=false"

echo "[4/4] 최종 검증..."
"${PYTHON}" -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print('  사용 가능한 Provider:', providers)
if 'CUDAExecutionProvider' in providers:
    print('  [OK] CUDAExecutionProvider 활성화됨')
else:
    print('  [WARN] CUDAExecutionProvider 없음 — CUDA 환경을 확인하세요')
"

echo "설치 완료."
