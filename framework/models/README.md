# 모델 준비 스크립트

이 디렉터리에는 모델 다운로드·ONNX export·양자화 보조 스크립트가 있다. 명령은
`framework/`에서 실행한다. [`model_profiles.py`](../src/core/model_profiles.py)에
등록된 `prepare_model_script`는 해당 profile의 모델 경로가 없고 target에서
자동 준비가 허용될 때만 `main.py`가 호출한다. HEF·DXNN·MXQ·RBLN 같은
NPU 실행 artifact는 이 보조 스크립트가 대신 준비하지 않는다.

| 목적 | 스크립트 | 주요 산출물 |
|---|---|---|
| Hugging Face repository snapshot | `prepare_hf_model.py --name <repo-id>` | 기본 `models/<repo-id의 /를 _로 바꾼 이름>/` |
| 기본 YOLOv5m ONNX export | `prepare_yolov5m.py` | `models/yolov5m/yolov5m.onnx` |
| Optimum CLI ONNX export | `prepare_onnx_hf.py --model <repo-id-or-path> --task <task> --output <dir>` | 지정한 디렉터리의 ONNX 자산 |
| PatchTST ONNX export | `prepare_patchtst.py --model <repo-id-or-path> --output <file.onnx>` | 지정한 ONNX 파일 |
| ONNX 동적 INT8 양자화 | `quantize_onnx_int8.py --input <file.onnx> --output <dir>` | `model.onnx` 및 부속 JSON |

예를 들어 기본 ONNX 준비와 범용 export는 다음처럼 실행한다.

```bash
cd /path/to/ML-HW-Benchmark-Framework/framework
python models/prepare_yolov5m.py
python models/prepare_onnx_hf.py \
  --model microsoft/resnet-50 --task image-classification \
  --output models/microsoft_resnet-50-ONNX
```

`prepare_hf_model.py`의 `--format`은 요청한 확장자와 기본 설정 파일만 내려받는
필터다. `prepare_patchtst.py`는 로컬 `patchtst_fm` 디렉터리에 weight 파일이
없어도 export를 진행할 수 있으므로 결과를 사용하기 전에 원본 weight 로딩
로그와 출력 검증을 확인한다. 양자화 스크립트의 산출물도 곧바로 NPU artifact나
정확도 승인 결과로 취급하지 않는다.

모델별로 실제 사용한 NPU artifact와 입출력 계약은
[ResNet50](../docs/model-execution-specs/resnet50.md),
[YOLOv5m](../docs/model-execution-specs/yolov5m.md),
[YOLOv8s-pose](../docs/model-execution-specs/yolov8s-pose.md) 명세를 따른다.
REGULUS는 직접 컴파일한 MXQ를, 해당 명세의 Hailo·DEEPX 조합은 벤더 제공
artifact를 사용한다. SHA-256 값은 각 명세가 연결한 별도 기록에 있다.
