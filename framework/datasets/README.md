# 데이터셋 준비

이 디렉터리의 `prepare_*.py`는 모델 profile에 연결된 데이터셋을 준비한다. 명령은
`framework/`에서 실행한다. `main.py`는 선택한 profile의
`prepare_dataset_script`가 있고 기본 데이터셋이 없거나 불완전하면 해당 스크립트를
호출하고 산출물의 개수·정답 파일을 검증한다. 사용자가 `--dataset`으로 다른 경로를
지정했다면 그 경로는 직접 준비해야 한다. 기본 스크립트가 임의 경로에 자산을
만들지는 않는다. 기본 경로와 자동 준비 조건은
[`model_profiles.py`](../src/core/model_profiles.py) 및
[`run_auto_prepare()`](../src/main.py)를 기준으로 확인한다.

| 데이터셋 | 준비 명령 | 기본 산출물 |
|---|---|---|
| ImageNet validation 일부 | `python datasets/prepare_imagenet_1k.py` | `datasets/imagenet_1k/val/` 이미지 3,000장, `val_labels.txt` |
| COCO128 | `python datasets/prepare_coco128.py` | `datasets/coco128/` |
| COCO val2017 vision | `python datasets/prepare_coco_vision.py` | `datasets/coco/images/val2017/`, `annotations/` |
| SQuAD 2.0 | `python datasets/prepare_squad2.py` | `datasets/squad2/val.json` |
| SST-2 분류 | `python datasets/prepare_text_numpy.py` | `datasets/sst2_numpy/`의 `input_ids.npy`, `attention_mask.npy`, `labels.npy` |
| SQuAD v1 QA | `python datasets/prepare_squad_numpy.py` | `datasets/squad_numpy/`의 입력·시작·끝 위치 `.npy` 파일 |
| ETTh1 | `python datasets/prepare_etth1.py` | `datasets/etth1/ETTh1.csv` |

SQuAD 2.0의 `val.json`은 Llama 생성 평가에 사용한다. BERT SQuAD v1 QA는
`squad_numpy/`를 사용한다. COCO128은 소량 점검용이고, COCO val2017의 이미지와
annotation은 별도 경로다. 이미지와 정답 파일의 조합은 선택한 모델·평가기에 맞춰
확인한다.

위 ImageNet 스크립트의 3,000장은 에지 NPU 실행 명세의 5,000장 고정 cohort가
아니다. 해당 실험은 [ResNet50 실행 명세](../docs/model-execution-specs/resnet50.md)의
이미지 목록·정답 파일·별도 SHA-256 기록을 사용한다. YOLOv5m과
YOLOv8s-pose도 각각 [탐지](../docs/model-execution-specs/yolov5m.md)·
[자세 추정](../docs/model-execution-specs/yolov8s-pose.md) 명세의 데이터셋을 따른다.

자산을 직접 준비한 경우에는 `--dataset`과 필요 시 `--image-dir`, `--label-dir`에
실제 경로를 지정한다. 전체 재현 순서는 [사용자 설명서](../../docs/user-guide/06-reproduction.md)에
정리되어 있다.

모델·데이터 파일은 Git에 포함되지 않는다. 새 체크아웃에서 디렉터리나 심볼릭 링크가
보인다는 이유만으로 준비가 완료된 것은 아니다. 준비 스크립트가 실패하면 비정상
종료하며, 기존 기본 데이터가 불완전하면 다시 준비를 시도한다. 사용자 지정
데이터셋은 선택한 loader의 이미지·정답 계약에 맞는지 실행 전에 확인한다.
