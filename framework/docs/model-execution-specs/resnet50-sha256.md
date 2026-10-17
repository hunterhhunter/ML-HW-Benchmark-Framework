# ResNet50 실행 자산 SHA-256 기록

[ResNet50 실행 명세](resnet50.md)에서 사용한 파일의 실행 당시 checksum이다.
새 장비에서는 `sha256sum`으로 파일을 확인한다. 아래 근거 경로는 Git에 포함되지 않은
로컬 `results/` archive 기준이다.

| 자산 | 파일명 | SHA-256 | 기록 근거 |
|---|---|---|---|
| ImageNet-1K protocol v2 label | `val_protocol_v2_5000_labels.txt` | `f3bcd181daa7d04115e3872fe6332151b855658840fab9dbfc4c00367535d9ac` | `results/2026-09-27-edge-npu-5000/setup.md` |
| Hailo-8 벤더 제공 HEF | `resnet_v1_50_8.hef` | `6dc70520ce6306ff1082041d1e796ff3165813e1e003603019c9e27cc11de55c` | `results/2026-09-27-edge-npu-5000/setup.md` |
| Hailo-10H 벤더 제공 HEF | `resnet_v1_50_10.hef` | `9ecbca3ac07420968fdd013bd82597483f5b8a1a9839c22ab92c0a995e93ab50` | `results/2026-10-17-hailo10h-ina3221-5000/setup/artifact-sha256.txt` |
| DEEPX 벤더 제공 DXNN | `resnet50_224x224.dxnn` | `c95e4d35430957c3d9faa524a719fdbd276602e575c8b1f9731cbc28311ba7ee` | `results/2026-09-27-edge-npu-5000/setup.md` |
| Mobilint Regulus 연구팀 컴파일 MXQ | `model.mxq` | `ca967bb5b972a7cb38f1bf490ca335630b8d5710d4a916432587032a87b85fd0` | `results/2026-10-01-regulus-vg-e2e-async-runtime-v2-preloaded-5000/environment.txt` |

DEEPX async 실행은 `deepx-official-smoke-resnet50.dxnn`이라는 파일명을
사용했다. 해당 실행의 `results/2026-10-16-deepx-ina3221-runtime-return-w4/setup/artifact-sha256.txt`에도
위 DXNN과 같은 SHA-256이 기록돼 있다. 파일명만으로 서로 다른 모델이라고 판단하지 않는다.

Regulus의 source·compiler 정보는 별도 `vendor-native-execution` 브랜치의
`framework/artifacts/vendor-native-e2e/vendor-native-e2e-resnet50-v2-source-cal1024-v1/targets/mobilint-regulus/resnet50/`
아래 `artifact-manifest.json`과 `compile-report.json`에서 같은 MXQ SHA-256을
대조해 확인했다. 이 컴파일 기록은 현재 `main`의 Git 문서에 포함되지 않는다.
