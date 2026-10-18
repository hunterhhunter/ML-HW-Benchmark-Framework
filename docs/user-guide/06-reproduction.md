# 06. 실험 재현

[목차](README.md) · [이전: 오류 대응](05-troubleshooting.md)

같은 실험을 다시 실행하려면 명령뿐 아니라 실행 자산과 측정 조건도 맞춘다.
이 문서는 재현 순서를 안내한다. 실제 결과값과 실행 증거는 별도 결과·보고서에
기록한다.

1. 실행 코드의 commit과 사용한 Python·벤더 Runtime 버전을 기록한다.
   `git rev-parse HEAD`와 해당 SDK의 버전 확인 명령을 사용한다.
2. [모델별 실행 명세](02-configuration.md)에 적힌 target, artifact, 모델·데이터
   cohort, 정답 파일과 입력 형식을 확인한다. SHA-256 값은 각 명세가 연결한
   별도 기록과 대조하고, 실행에 사용한 실제 파일 경로를 남긴다.
3. 같은 실행 명령과 batch, warmup, worker, queue, 표본 수를 적용한다.
   먼저 소량 실행을 통과한 뒤 본 실행을 수행하고 `--results-path`를 기록한다.
4. `RUN_ID=<id>`와 CSV 행이 일치하는지 확인한다. 비동기는 상태·품질 패스와
   완료·실패 sample 수도 [결과 확인 절차](04-results.md)대로 확인한다.
5. 성능 지표를 인용할 때는 [측정 계약](../../framework/docs/measurement-contract.md)의
   시간 경계와 단위를 함께 적는다. 전력 결과가 있다면 측정 지점과 원시 trace를
   별도 증거와 연결한다.

에지 NPU의 실제 명령은
[ResNet50](../../framework/docs/model-execution-specs/resnet50.md),
[YOLOv5m](../../framework/docs/model-execution-specs/yolov5m.md),
[YOLOv8s-pose](../../framework/docs/model-execution-specs/yolov8s-pose.md)에 있다.
서버 조합은 [설정 문서](02-configuration.md)의 벤더별 가이드를 따른다.
