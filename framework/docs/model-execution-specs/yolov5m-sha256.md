# YOLOv5m 실행 자산 SHA-256 기록

[YOLOv5m 실행 명세](yolov5m.md)에서 사용한 artifact의 실행 당시 checksum이다.
새 장비에서는 `sha256sum`으로 대조한다. 근거는 Git에 포함되지 않은 로컬
`results/` archive에 있다.

| 자산 | 파일명 | SHA-256 | 기록 근거 |
|---|---|---|---|
| COCO detection 원본 annotation | `instances_val2017.json` | `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f` | `results/2026-09-27-edge-npu-5000/setup.md` |
| Hailo-8 벤더 제공 HEF | `yolov5m_8.hef` | `f26aa85836d94a4b37639ee031e374389ba5fa47a2f8c3d7e5369bf625f0f3a1` | `results/2026-09-27-edge-npu-5000/setup.md` |
| Hailo-10H 벤더 제공 HEF | `yolov5m_10.hef` | `92893bad750a4d7384817402ee5b447e05f2eaf4fb7c9933afae8cb2457a3dba` | `results/2026-10-17-hailo10h-ina3221-5000/setup/artifact-sha256.txt` |
| DEEPX 벤더 제공 DXNN | `yolov5-m_640x640.dxnn` | `7a5c213eefbd424651d15c59b5d063da3c9d1073d34c19e3ea579504850edd61` | `results/2026-09-27-edge-npu-5000/setup.md` |
| Mobilint Regulus 연구팀 컴파일 MXQ | `model.mxq` | `45da2a948a54085743b0fdb88fa50a6d34d457919d84e68e45f8fc6857a22dde` | `results/2026-10-01-regulus-vg-e2e-async-runtime-v2-preloaded-5000/environment.txt` |

YOLO 형식 라벨은 위 annotation에서 변환한 5,000개 파일이다. 라벨 내용 digest
`d574c4a407b59f413deefe0ccfaec1c68dfb9382d0f55a1e8e932c4ea86187a8`와
변환 기준은 `results/2026-09-27-edge-npu-5000/setup.md`에 기록돼 있다.
이 digest는 개별 파일에 `sha256sum`을 적용한 값이 아니다.
