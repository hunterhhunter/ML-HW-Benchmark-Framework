# YOLOv8s Segmentation/Pose ONNX 검증 기록

- 검증일: 2026-07-29
- GPU: NVIDIA GeForce RTX 5070 Ti (16,303 MiB)
- NVIDIA 드라이버: 580.173.02
- ONNX Runtime GPU: 1.24.4
- pycocotools: 2.0.11

## 모델과 데이터

Ultralytics의 정확한 `yolov8s-seg`, `yolov8s-pose` 가중치를 동적 배치/해상도 ONNX로 내보내고 ONNX Runtime에서 직접 실행했다.

| 모델 | ONNX 크기 | 입력 | 출력 |
| --- | ---: | --- | --- |
| yolov8s-seg | 47,751,907 bytes | `images [batch,3,height,width]` | `output0 [batch,116,anchors]`, `output1 [batch,32,mask_height,mask_width]` |
| yolov8s-pose | 46,985,780 bytes | `images [batch,3,height,width]` | `output0 [batch,56,anchors]` |

공식 COCO 2017 validation 데이터와 annotation을 HTTPS 인증서 검증이 가능한 공식 S3 엔드포인트에서 받았다.

- 이미지: 5,000개
- instance segmentation annotation: 36,781개, category 80개
- person keypoint annotation: 11,004개, category 1개
- 실제 smoke test 입력: image ID 139, `000000000139.jpg`, 640x426

## 실제 런타임 검증

실제 이미지 한 장을 DataLoader → preprocessor → ONNX Runtime → decoder/postprocessor → 공식 `pycocotools.COCOeval` 전체 경로로 실행했다. CUDA 검증은 세션의 활성 provider에 `CUDAExecutionProvider`가 없으면 실패하도록 구성해 CPU fallback을 성공으로 오인하지 않게 했다.

```text
.venv/bin/python -m pytest tests/test_yolov8_onnx_runtime.py -v -k cpu
2 passed, 2 deselected in 1.83s

.venv/bin/python -m pytest tests/test_yolov8_onnx_runtime.py -v -k cuda
2 passed, 2 deselected in 27.92s
```

CLI smoke test 결과:

| 모델/런타임 | 핵심 정확도 | 평균 지연시간 | 처리량 |
| --- | --- | ---: | ---: |
| yolov8s-seg / CPU | Mask mAP 0.2549, AP50 0.3781, AP75 0.3005 | 35.0878 ms | 28.4999 FPS |
| yolov8s-seg / CUDA | Mask mAP 0.2549, AP50 0.3781, AP75 0.3005 | 6.8019 ms | 147.0167 FPS |
| yolov8s-pose / CPU | OKS mAP 0.5000, AP50 1.0000, AP75 0.0000 | 29.5798 ms | 33.8068 FPS |
| yolov8s-pose / CUDA | OKS mAP 0.5000, AP50 1.0000, AP75 0.0000 | 6.0512 ms | 165.2567 FPS |

위 정확도 수치는 `--max-steps 1`로 얻은 end-to-end smoke 결과이며 전체 COCO 정확도가 아니다. `--max-steps`를 생략하면 COCO validation 전체 평가를 실행한다. CUDA 세션 생성 시 ONNX Runtime이 memcpy-node 성능 경고를 출력했지만, 활성 CUDA provider 검증은 통과했다.

## 회귀 검증

```text
.venv/bin/python -m pytest tests/ -q
1322 passed, 5 skipped, 1 warning in 35.90s
```

경고 한 건은 기존 `test_ettm_loader.py`의 미등록 `integration` pytest mark다.
