import sys
import os
from pathlib import Path

# 프레임워크 루트 경로. Docker에서는 /app/framework, 로컬에서는 repo/framework.
FRAMEWORK_DIR = Path(
    os.getenv(
        "FRAMEWORK_DIR",
        Path(__file__).resolve().parent.parent.parent.parent / "framework",
    )
)

# framework 실행 Python. Docker에서는 /opt/venv/bin/python, 로컬에서는 framework/.venv 우선.
FRAMEWORK_PYTHON = Path(
    os.getenv("FRAMEWORK_PYTHON", FRAMEWORK_DIR / ".venv" / "bin" / "python")
)

def get_available_profiles() -> list[dict]:
    """framework/src/core/model_profiles.py에서 등록된 프로필 목록을 반환한다."""
    profiles_path = FRAMEWORK_DIR / "src" / "core" / "model_profiles.py"
    if not profiles_path.exists():
        return []

    # framework/src를 sys.path에 추가하여 model_profiles.py의 상대 import 해결
    import types

    framework_src = str(FRAMEWORK_DIR / "src")
    added_to_path = framework_src not in sys.path
    if added_to_path:
        sys.path.insert(0, framework_src)

    # onnx 등 무거운 의존성 스텁 (프로필 딕셔너리 읽기에는 불필요)
    onnx_stub = None
    if "onnx" not in sys.modules:
        onnx_stub = types.ModuleType("onnx")
        sys.modules["onnx"] = onnx_stub

    try:
        # core 패키지로 정상 import
        from core.model_profiles import SUPPORTED_PROFILES
        supported = SUPPORTED_PROFILES
    finally:
        if onnx_stub is not None:
            sys.modules.pop("onnx", None)
        if added_to_path:
            sys.path.remove(framework_src)

    result = []
    for model_name, profile in supported.items():
        task_enum = profile.get("task")
        task_name = task_enum.name if hasattr(task_enum, "name") else str(task_enum)

        # 태스크에 따라 사용 가능한 백엔드 결정
        if task_name == "NLP_GENERATION":
            backends = ["onnxruntime", "vllm"]
        else:
            backends = ["onnxruntime"]

        result.append({
            "model_name": model_name,
            "task": task_name,
            "backends": backends,
            "default_model_path": profile.get("default_model_path"),
            "default_dataset_path": profile.get("default_dataset_path"),
        })

    return result


def get_available_targets() -> list[dict]:
    """framework/src/core/targets.py에서 등록된 target 목록을 반환한다."""
    framework_src = str(FRAMEWORK_DIR / "src")
    added_to_path = framework_src not in sys.path
    if added_to_path:
        sys.path.insert(0, framework_src)

    try:
        from core.targets import list_targets
        return [target.to_response() for target in list_targets()]
    finally:
        if added_to_path:
            sys.path.remove(framework_src)
