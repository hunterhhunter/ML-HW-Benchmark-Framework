import os
import shutil
from pathlib import Path

import onnx
from google.protobuf.message import DecodeError


MODELS = {
    "yolov8m": {
        "weights": "yolov8m.pt",
        "output_dir": "yolov8m",
        "onnx_name": "yolov8m.onnx",
        "output_count": 1,
    },
    "yolov8m-seg": {
        "weights": "yolov8m-seg.pt",
        "output_dir": "yolov8m-seg",
        "onnx_name": "yolov8m-seg.onnx",
        "output_count": 2,
    },
    "yolov8m-pose": {
        "weights": "yolov8m-pose.pt",
        "output_dir": "yolov8m-pose",
        "onnx_name": "yolov8m-pose.onnx",
        "output_count": 1,
    },
    "yolov8s-seg": {
        "weights": "yolov8s-seg.pt",
        "output_dir": "yolov8s-seg",
        "onnx_name": "yolov8s-seg.onnx",
        "output_count": 2,
    },
    "yolov8s-pose": {
        "weights": "yolov8s-pose.pt",
        "output_dir": "yolov8s-pose",
        "onnx_name": "yolov8s-pose.onnx",
        "output_count": 1,
    },
}


def _valid_onnx_export(path: Path, expected_output_count: int) -> bool:
    """완전히 파싱·검증되고 예상 I/O 개수를 가진 export인지 확인합니다."""
    try:
        model = onnx.load(path)
        onnx.checker.check_model(model)
    except (OSError, DecodeError, onnx.checker.ValidationError):
        return False
    return len(model.graph.input) == 1 and len(model.graph.output) == expected_output_count


def _export_model(YOLO, root_dir: Path, model_name: str, info: dict) -> None:
    output_dir = root_dir / info["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    final_onnx_path = output_dir / info["onnx_name"]
    if _valid_onnx_export(final_onnx_path, info["output_count"]):
        print(f"[*] 이미 파이프라인 대상 모델이 존재합니다: {final_onnx_path}")
        return

    weights = info["weights"]
    print(f"[*] {model_name}: Ultralytics weights 로드: {weights}")
    model = YOLO(weights)
    export_path = Path(model.export(format="onnx", imgsz=640, dynamic=True))
    print(f"[+] {model_name}: ONNX export 완료: {export_path}")

    if not _valid_onnx_export(export_path, info["output_count"]):
        raise RuntimeError(f"{model_name}: export된 ONNX 구조가 예상과 다릅니다: {export_path}")

    shutil.move(str(export_path), final_onnx_path)
    if not _valid_onnx_export(final_onnx_path, info["output_count"]):
        raise RuntimeError(f"{model_name}: 최종 ONNX 검증에 실패했습니다: {final_onnx_path}")
    print(f"[+] {model_name}: 모델 파일 정리 완료: {final_onnx_path}")

    pt_path = Path(weights)
    if pt_path.exists():
        shutil.move(str(pt_path), output_dir / weights)


def main():
    print("=" * 60)
    print(" Ultralytics YOLOv8 vision ONNX Export ")
    print("=" * 60)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] ultralytics 패키지가 없습니다. 터미널에서 아래 명령어로 설치해주세요:")
        print("    uv pip install ultralytics")
        return

    root_dir = Path(os.path.dirname(__file__))
    for model_name, info in MODELS.items():
        try:
            _export_model(YOLO, root_dir, model_name, info)
        except Exception as exc:
            print(f"[!] {model_name}: ONNX Export 중 오류 발생: {exc}")


if __name__ == "__main__":
    main()
