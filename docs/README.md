# 문서 안내

현재 코드의 사용·확장 절차는 두 설명서에서 시작한다. 모델과 장치의 실행 조건은
실행 명세를 확인하고, 과거 검증 기록의 결과는 당시 자산·환경에 한정해 읽는다.
target 등록만으로 모든 모델·장치 조합의 실행이 확인되는 것은 아니다.

| 목적 | 문서 |
|---|---|
| 설치, 실행, 결과 확인과 재현 | [사용자 설명서](user-guide/README.md) |
| 구조, 공용 인터페이스와 확장 | [개발자 설명서](developer-guide/README.md) |
| 동기·비동기 측정 경계 | [측정 계약](../framework/docs/measurement-contract.md) |
| 비동기 지표·trace 상세 | [비동기 추론 큐 측정 가이드](../framework/docs/async-inference-queue.md) |
| 벤더별 설치·실행·문제 해결 기록 | [벤더 문서](vendors/README.md) |

## 모델별 실행 명세

- 에지 NPU: [ResNet50](../framework/docs/model-execution-specs/resnet50.md) ·
  [YOLOv5m](../framework/docs/model-execution-specs/yolov5m.md) ·
  [YOLOv8s-pose](../framework/docs/model-execution-specs/yolov8s-pose.md).
  자산 해시는 [ResNet50](../framework/docs/model-execution-specs/resnet50-sha256.md) ·
  [YOLOv5m](../framework/docs/model-execution-specs/yolov5m-sha256.md) ·
  [YOLOv8s-pose](../framework/docs/model-execution-specs/yolov8s-pose-sha256.md)에 분리했다.
- TTM-R2: [고정 계약과 당시 서버 검증 기록](../framework/docs/ttm-r2-framework.md).

벤더별 Runtime 호출과 수정 위치는 [개발자 구현 지도](developer-guide/10-vendor-implementation-map.md),
모델·데이터 준비는 [모델](../framework/models/README.md)과
[데이터셋](../framework/datasets/README.md) README를 따른다.

## 검증 기록

- [Cross-NPU Transformer·시계열 검증 현황](../framework/docs/cross-npu-model-validation.md): 작성 당시의 자산·SDK·장치 조건과 판정.

벤더별 검증 기록은 [벤더 문서 목차](vendors/README.md)에서 찾는다.

코드 옆의 짧은 안내는 [Runtime](../framework/src/runtimes/README.md),
[프레임워크](../framework/README.md), [웹 화면](../frontend/README.md) README에 있다.
