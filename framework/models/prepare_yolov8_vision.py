import os
import shutil
from pathlib import Path


MODELS = {
    "yolov8m": {
        "weights": "yolov8m.pt",
        "output_dir": "yolov8m",
        "onnx_name": "yolov8m.onnx",
    },
    "yolov8m-seg": {
        "weights": "yolov8m-seg.pt",
        "output_dir": "yolov8m-seg",
        "onnx_name": "yolov8m-seg.onnx",
    },
    "yolov8m-pose": {
        "weights": "yolov8m-pose.pt",
        "output_dir": "yolov8m-pose",
        "onnx_name": "yolov8m-pose.onnx",
    },
}


def _export_model(YOLO, root_dir: Path, model_name: str, info: dict) -> None:
    output_dir = root_dir / info["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    final_onnx_path = output_dir / info["onnx_name"]
    if final_onnx_path.exists():
        print(f"[*] 이미 파이프라인 대상 모델이 존재합니다: {final_onnx_path}")
        return

    weights = info["weights"]
    print(f"[*] {model_name}: Ultralytics weights 로드: {weights}")
    model = YOLO(weights)
    export_path = Path(model.export(format="onnx", imgsz=640, dynamic=True))
    print(f"[+] {model_name}: ONNX export 완료: {export_path}")

    if export_path.exists():
        shutil.move(str(export_path), final_onnx_path)
        print(f"[+] {model_name}: 모델 파일 정리 완료: {final_onnx_path}")

    pt_path = Path(weights)
    if pt_path.exists():
        shutil.move(str(pt_path), output_dir / weights)


def main():
    print("=" * 60)
    print(" Ultralytics YOLOv8m / YOLOv8m-seg / YOLOv8m-pose ONNX Export ")
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
