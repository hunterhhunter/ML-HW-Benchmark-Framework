# YOLOv8s-pose 실행 자산 SHA-256 기록

[YOLOv8s-pose 실행 명세](yolov8s-pose.md)에서 사용한 파일의 실행 당시 checksum이다.
새 장비에서는 `sha256sum`으로 대조한다. 근거는 Git에 포함되지 않은 로컬
`results/` archive에 있다.

| 자산 | 파일명 | SHA-256 | 기록 근거 |
|---|---|---|---|
| COCO 2017 pose annotation | `person_keypoints_val2017.json` | `788e2dae83c86bd547be7fab269d6399df5671063d29a61360cdb2cc370d2b14` | `results/2026-10-17-hailo10h-ina3221-5000/setup/dataset-hashes.txt` |
| Hailo-8 벤더 제공 HEF | `yolov8s_pose_8.hef` | `456f96e5ed7674796d403d9fa24ff06e2aab37d274a846d6e5fff54da81d2859` | `results/2026-09-27-edge-npu-5000/setup.md` |
| Hailo-10H 벤더 제공 HEF | `yolov8s_pose_10.hef` | `d6cebc9a7bfc71d711000aa7d23b2e1a8aa634cbd90279df110a51d5309b1c58` | `results/2026-10-17-hailo10h-ina3221-5000/setup/artifact-sha256.txt` |
| DEEPX 벤더 제공 DXNN | `yolov8-s-pose_640x640.dxnn` | `65a96d04728f4ba3be1e28ef59c616811e57ce43e55bc31ce49c21655c6a20b4` | `results/2026-09-27-edge-npu-5000/setup.md` |
| Mobilint Regulus 직접 컴파일 모델 MXQ | `model.mxq` | `5164d64d326ab963db45313e46a42707a6763ef57c591e9db611d6ccfb9f3170` | `results/2026-10-01-regulus-vg-e2e-async-runtime-v2-preloaded-5000/environment.txt` |

Regulus 직접 컴파일 기록은 `vendor-native-execution` 브랜치의
`framework/artifacts/vendor-native-e2e/vendor-reference-single-stream-v1/targets/mobilint-regulus/yolov8s-pose/`
아래 `artifact-manifest.json`과 `compile-report.json`에서 확인했다.
