# 실행 조합과 증거 현황

> 코드 기준: 원격 `main` `30c4db75e62231ac91795224fc5af5ab5f3f35df` (2026-10-16 병합). 조사일: 2026-10-17 (Asia/Seoul). 이 문서는 **현재 코드의 등록 상태**와 **서로 다른 리비전에서 수집한 실행 증거**를 구분한다. 이번 문서 작성 중 NPU를 새로 실행하지 않았다.

## 이 표의 범위와 판정 방법

[`targets.py`](../src/core/targets.py)의 target 등록과 [`model_profiles.py`](../src/core/model_profiles.py)의 모델 등록은 실행 조합의 후보를 만든다. 특정 모델·artifact·mode가 실제 장치에서 동작하거나 품질 기준을 통과했다는 증거는 아니다. 같은 모델 이름에도 원천 checkpoint, 벤더 artifact, 입력과 출력 계약이 다를 수 있다.

각 행은 다음 게이트를 따로 읽는다.

| 게이트 | 이 문서에서 확인하는 것 |
|---|---|
| 산출물 `A` | 해당 장치용 파일 또는 준비 디렉터리의 신원·SHA-256·필수 sidecar가 기록됐는가. 해시 기록만으로 현재 파일의 존재를 보장하지 않는다. |
| 컴파일 `C` | 해당 graph의 전체 컴파일 완료 증거가 있는가. 제공받은 precompiled 파일은 `제공`으로 적고 현지 컴파일 성공으로 바꾸지 않는다. |
| 실장비 `D` | 실제 장치가 유효한 출력을 반환하고 실행이 종료됐는가. 작은 smoke와 전체 실행은 구분한다. |
| 태스크 품질 `Q` | 고정 데이터·evaluator에서 지표를 계산했는가. `quality_status=passed`는 평가 실행과 표본 검증을 뜻하며, 사전 정확도 허용 기준의 승인은 별개다. |
| 전체 실행 `F` | 명시한 표본·요청이 완료되고 결과가 저장됐는가. `runtime-only`의 전체 완료는 품질 포함 전체 검증이 아니다. |
| 전력 `P` | 측정점·시간 경계·원시 trace가 확인됐는가. 시스템 DC 입력 전력을 NPU 단독 전력으로 부르지 않는다. |

`확인`은 아래 근거에서 확인한 범위에만 적용한다. `미수집`은 실패 또는 미지원의 동의어가 아니다. `품질 이상`과 `품질 미측정`은 모두 성능 수치의 정식 품질 승인과 구분한다. 비동기 schema v1 완료 처리량, schema v2 Runtime-call 반환 처리량, 동기 `e2e` 모드의 Runtime-call latency, 별도 System E2E는 서로 다른 경계다. 지표 정의는 [비동기 측정 가이드](../../docs/async-inference-queue.md)를 따른다.

### 증거의 보관 위치

- **Git 추적 근거:** 이 문서가 연결하는 `framework/docs/`, `docs/`의 코드·가이드·과거 검증 기록. `git ls-files results result docs/evidence framework/results`로 조회하면 현재 결과 바이너리·CSV는 포함되지 않는다.
- **로컬 원본:** 기본 체크아웃 `/home/swlab-youngjin/ML-HW-Benchmark-Framework/` 아래의 `results/`, `result/`, `docs/evidence/`, `framework/results/`. 아래에서 `로컬:`로 시작하는 경로는 이 문서 브랜치에 포함되지 않으므로 인수인계 패키지에 자동으로 따라오지 않는다. 표의 일부 CSV와 sidecar를 읽어 run ID·표본·상태를 대조했다. 2026-10-16 Hailo-8·DeepX 전력 묶음의 `manifest.sha256`과 REGULUS 10건 smoke의 `source-sha256.txt`는 로컬 사본에서 검사를 통과했다. 모든 원격 보드에서 재실행하거나 모든 artifact 파일을 다시 해시하지는 않았다.
- **원격 전용 근거:** [서버 모델 검증 현황](cross-npu-model-validation.md)과 [TTM-R2 실행 기록](ttm-r2-framework.md)이 보존한 서버 run ID·해시·환경. 원격 CSV와 artifact를 이번 조사에서 직접 다시 열지 않았다.

## 코드에 등록된 범위

기준 코드에는 target **15개**, 기본 모델 프로필 **14개**와 이름 별칭 **4개**가 있다. 아래 `native async`는 target capability의 선언값이다. 모델별 지원이나 SDK의 실제 동시 실행 능력을 뜻하지 않는다.

| target | runtime / artifact 형식 | target compiler | `native_async` 선언 |
|---|---|---|---|
| `cpu` | `onnxruntime` / ONNX | 없음 | 없음 |
| `cuda` | `onnxruntime` / ONNX | 없음 | 없음 |
| `vllm-cpu` | `vllm` / HF 모델 | 없음 | 없음 |
| `vllm-cuda` | `vllm` / HF 모델 | 없음 | 없음 |
| `furiosa-rngd` | `furiosa_llm` / FXB | 없음 | 있음 |
| `furiosa-rngd-torch` | `furiosa_torch` / PyTorch 모델 | 없음; runtime 경로에서 strict compile | 없음 |
| `rbln-static` | `rbln` / RBLN | 없음 | 있음 |
| `rbln-vllm` | `rbln_vllm` / 준비된 RBLN LLM 디렉터리 | 없음 | 있음 |
| `mobilint-aries` | `mobilint` / MXQ | 없음 | 있음 |
| `mobilint-regulus` | `mobilint` / MXQ | 없음 | 있음 |
| `mobilint-aries-llm` | `mobilint_llm` / 준비된 Model Zoo 디렉터리 | 없음 | 없음 |
| `hailo8` | `hailort` / HEF | 없음 | 있음 |
| `hailo10h` | `hailort` / HEF | 없음 | 있음 |
| `deepx` | `deepx` / DXNN | `deepx` | 있음 |
| `vendor_mock_npu` | `mock_npu` / mockbin | `mock_npu` | 없음 |

기본 프로필은 `resnet50`, `yolov5m`, `yolov8m`, `yolov8m-seg`, `yolov8m-pose`, `yolov8s-seg`, `yolov8s-pose`, `bert-base-uncased`, `bert-base-uncased-squad-v1`, `patchtst-fm-r1`, `patchtst-etth1`, `ttm-r2`, `llama-3.1-8b`, `llama-3.2-3b`다. `yolov8m_seg`, `yolov8-seg-m`, `yolov8m_pose`, `yolov8-pose-m`은 기존 프로필을 가리키는 별칭이며 독립 검증 조합으로 세지 않는다. 프로필에 있는 기본 ONNX·데이터 경로를 벤더 HEF·DXNN·MXQ·RBLN·FXB의 출처로 추정하지 않는다.

## 엣지 artifact 식별자

이 표의 해시는 로컬 기록에 보존된 **실행 artifact의 SHA-256**이다. 실제 파일은 Git에 포함되지 않는다. Hailo·DeepX 값은 `로컬: results/2026-10-01-edge-async-runtime-v2-5000/RUN.md`, Hailo-10H 값은 `로컬: results/2026-09-28-edge-npu-5000/setup.md`, REGULUS 값은 `로컬: results/2026-10-16-regulus-pr60-reapply-smoke-10-v2/environment.txt`와 `results/regulus-async-full-framework-20261001/summary.md`에서 대조했다.

| ID | target / 모델 | 형식 | 기록된 SHA-256 | 준비·컴파일 게이트 `A / C` |
|---|---|---|---|---|
| H8-R | `hailo8` / `resnet50` | HEF | `6dc70520ce6306ff1082041d1e796ff3165813e1e003603019c9e27cc11de55c` | hash 기록 / 벤더 제공 |
| H8-Y | `hailo8` / `yolov5m` | HEF | `f26aa85836d94a4b37639ee031e374389ba5fa47a2f8c3d7e5369bf625f0f3a1` | hash 기록 / 벤더 제공 |
| H8-P | `hailo8` / `yolov8s-pose` | HEF | `456f96e5ed7674796d403d9fa24ff06e2aab37d274a846d6e5fff54da81d2859` | hash 기록 / 벤더 제공 |
| D-R | `deepx` / `resnet50` | DXNN | `c95e4d35430957c3d9faa524a719fdbd276602e575c8b1f9731cbc28311ba7ee` | hash 기록 / 벤더 제공; 이 run에서 DX-COM 미수행 |
| D-Y | `deepx` / `yolov5m` | DXNN | `7a5c213eefbd424651d15c59b5d063da3c9d1073d34c19e3ea579504850edd61` | hash 기록 / 벤더 제공; 이 run에서 DX-COM 미수행 |
| D-P | `deepx` / `yolov8s-pose` | DXNN | `65a96d04728f4ba3be1e28ef59c616811e57ce43e55bc31ce49c21655c6a20b4` | hash 기록 / 벤더 제공; 이 run에서 DX-COM 미수행 |
| H10-R | `hailo10h` / `resnet50` | HEF | `9ecbca3ac07420968fdd013bd82597483f5b8a1a9839c22ab92c0a995e93ab50` | hash 기록 / 벤더 제공 |
| H10-Y | `hailo10h` / `yolov5m` | HEF | `92893bad750a4d7384817402ee5b447e05f2eaf4fb7c9933afae8cb2457a3dba` | hash 기록 / 벤더 제공 |
| H10-P | `hailo10h` / `yolov8s-pose` | HEF | `d6cebc9a7bfc71d711000aa7d23b2e1a8aa634cbd90279df110a51d5309b1c58` | hash 기록 / 벤더 제공 |
| R-R | `mobilint-regulus` / `resnet50` | MXQ | `ca967bb5b972a7cb38f1bf490ca335630b8d5710d4a916432587032a87b85fd0` | hash 기록 / 현지 compiler attempt 별도 |
| R-Y | `mobilint-regulus` / `yolov5m` | MXQ | `45da2a948a54085743b0fdb88fa50a6d34d457919d84e68e45f8fc6857a22dde` | hash 기록 / 현지 compiler attempt 별도 |
| R-P | `mobilint-regulus` / `yolov8s-pose` | MXQ | `5164d64d326ab963db45313e46a42707a6763ef57c591e9db611d6ccfb9f3170` | hash 기록 / 현지 compiler attempt 별도 |

`A`의 hash 기록과 `C`의 compiler attempt·원천 모델·calibration 연결은 다음 모델별 실행 명세에서 파일 단위로 확정한다. 같은 모델명에 다른 파일을 대입하면 이 표의 실행 증거가 이어지지 않는다.

## 엣지 모델 × target × artifact × mode

이 표의 `Q`는 **계산된 품질 또는 명확한 미측정 상태**다. `F`의 5,000건 완료는 당시 코드·입력·mode의 실행 범위다. Hailo-8·DeepX의 과거 동기 행과 Hailo-10H의 2026-09-28 동기 행은 `main@6868f6ba3320b382f9b385be45a794526fb2ca5b` 기록이다. Hailo-8·DeepX의 2026-10-16 비동기·전력 행은 `main@30c4db75e62231ac91795224fc5af5ab5f3f35df`의 격리 클론에 측정용 marker patch를 적용했다. REGULUS 2026-10-01 5,000건 행의 기록된 코드 기준은 `5b286a243e32a4d6bcda3e68fa37331f9f7ed196`이다. `P`가 `DC 입력`인 회차는 로컬 INA3221 원시 trace와 marker patch를 사용했으며 Jetson 전체 입력 전력이다. 아래의 이전 `e2e`와 별도 System E2E 결과를 같은 측정값으로 보지 않는다.

| 모델 | target / artifact | mode / 근거 run ID | 실장비 `D` | 품질 `Q` | 전체 `F` | 전력 `P` |
|---|---|---|---|---|---|---|
| `resnet50` | `hailo8` / H8-R | `e2e` `a91b3f06` | 5,000건 | Top-1 73.78%, 계산 | 5,000건, 과거 main 계열 | 미수집 |
| `resnet50` | `hailo8` / H8-R | `async_queue/offline`, W4 `4e4f2b57` | 5,000건 | 5,000건 평가, `passed` | schema v2 `valid`, 5,000건 | DC 입력, 로컬 원본 |
| `yolov5m` | `hailo8` / H8-Y | `e2e` `0e4665c0` | 5,000건 | mAP@0.5 계산 | 5,000건, 과거 main 계열 | 미수집 |
| `yolov5m` | `hailo8` / H8-Y | `async_queue/offline`, W4 `7fb5f6e4` | 5,000건 | 5,000건 평가, `passed` | schema v2 `valid`, 5,000건 | DC 입력, 로컬 원본 |
| `yolov8s-pose` | `hailo8` / H8-P | `e2e` `ad7bde86` | 5,000건 | OKS mAP 계산 | 5,000건, 과거 main 계열 | 미수집 |
| `yolov8s-pose` | `hailo8` / H8-P | `async_queue/offline`, W4 `1bc2b13b` | 5,000건 | 5,000건 평가, `passed` | schema v2 `valid`, 5,000건 | DC 입력, 로컬 원본 |
| `resnet50` | `deepx` / D-R | `e2e` `0747ae23` | 5,000건 | Top-1 79.36%, 계산 | 5,000건, 과거 main 계열 | 미수집 |
| `resnet50` | `deepx` / D-R | `async_queue/offline`, W4 `671faf17` | 5,000건 | 5,000건 평가, `passed` | schema v2 `valid`, 5,000건 | DC 입력, 로컬 원본 |
| `yolov5m` | `deepx` / D-Y | `e2e` `8eb73f71` | 5,000건 | mAP@0.5 계산 | 5,000건, 과거 main 계열 | 미수집 |
| `yolov5m` | `deepx` / D-Y | `async_queue/offline`, W4 `5f450e9b` | 5,000건 | 5,000건 평가, `passed` | schema v2 `valid`, 5,000건 | DC 입력, 로컬 원본 |
| `yolov8s-pose` | `deepx` / D-P | `e2e` `15f2ca6a` | 5,000건 | OKS mAP 계산 | 5,000건, 과거 main 계열 | 미수집 |
| `yolov8s-pose` | `deepx` / D-P | `async_queue/offline`, W4 `80f58a72` | 5,000건 | 5,000건 평가, `passed` | schema v2 `valid`, 5,000건 | DC 입력, 로컬 원본 |
| `resnet50` | `hailo10h` / H10-R | `e2e` `4b1ade8c` | 5,000건 | Top-1 28.08%, **품질 이상** | 실행 완료; 교정 결과 원본 미확보 | 미수집 |
| `resnet50` | `hailo10h` / H10-R | `async_queue/offline`, runtime-only W4 `d9e06502` | 5,000건 | `skipped` | schema v2 `valid`, 5,000건; 품질 별도 | 미수집 |
| `yolov5m` | `hailo10h` / H10-Y | `e2e` `4dd46424` | 5,000건 | mAP@0.5 계산 | 5,000건, 과거 main 계열 | 미수집 |
| `yolov5m` | `hailo10h` / H10-Y | `async_queue/offline`, runtime-only W4 `a8dc0110` | 5,000건 | `skipped` | schema v2 `valid`, 5,000건; 품질 별도 | 미수집 |
| `yolov8s-pose` | `hailo10h` / H10-P | `e2e` `be2aa9de` | 5,000건 | OKS mAP 계산 | 5,000건, 과거 main 계열 | 미수집 |
| `yolov8s-pose` | `hailo10h` / H10-P | `async_queue/offline`, runtime-only W4 `9328f9ca` | 5,000건 | `skipped` | schema v2 `valid`, 5,000건; 품질 별도 | 미수집 |
| `resnet50` | `mobilint-regulus` / R-R | `async_queue/offline`, runtime-only W4 `946a58d1` | 5,000건, 이전 코드 | `skipped` | schema v2 `valid`; 현 main 재검증 미수집 | 미수집 |
| `yolov5m` | `mobilint-regulus` / R-Y | `async_queue/offline`, runtime-only W4 `191cd5c7` | 5,000건, 이전 코드 | `skipped` | schema v2 `valid`; 현 main 재검증 미수집 | 미수집 |
| `yolov8s-pose` | `mobilint-regulus` / R-P | `async_queue/offline`, runtime-only W4 `c7e5960a` | 5,000건, 이전 코드 | `skipped` | schema v2 `valid`; 현 main 재검증 미수집 | 미수집 |
| `resnet50` | `mobilint-regulus` / R-R | `async_queue/offline`, runtime-only W4 `a85f44a6` | 10건, 현재 main의 조상 `9548fc6` | `skipped` | 10건 smoke, `comparison_eligible=false` | 미수집 |
| `yolov5m` | `mobilint-regulus` / R-Y | `async_queue/offline`, runtime-only W4 `8964766e` | 10건, 현재 main의 조상 `9548fc6` | `skipped` | 10건 smoke, `comparison_eligible=false` | 미수집 |
| `yolov8s-pose` | `mobilint-regulus` / R-P | `async_queue/offline`, runtime-only W4 `d6f47e24` | 10건, 현재 main의 조상 `9548fc6` | `skipped` | 10건 smoke, `comparison_eligible=false` | 미수집 |
| `yolov8s-pose` | `mobilint-regulus` / R-P | `e2e` `ef06906d` | 10건, 현재 main의 조상 `9548fc6` | OKS mAP 0.652551, **10건만** | 10건 smoke; 5,000건 품질 실행 아님 | 미수집 |

근거: Hailo-8·DeepX 과거 동기 결과는 `로컬: results/2026-09-27-edge-npu-5000/{hailo8,deepx}/results.csv`와 Hailo-8 분류 교정 `results/2026-09-28-hailo8-worker4-rerun/results.csv`; 최신 W4 결과와 전력은 `results/2026-10-16-{hailo8,deepx}-ina3221-runtime-return-w4/`의 모델별 CSV·details·sensor·marker와 `RUN.md`; Hailo-10H는 `results/2026-09-28-edge-npu-5000/hailo10h/preflight/precondition.csv`와 `results/2026-10-02-hailo10h-runtime-only-5000/`(ResNet은 반드시 `corrected/resnet50/`); REGULUS는 `results/2026-10-01-regulus-vg-e2e-async-runtime-v2-preloaded-5000/` 및 `results/2026-10-16-regulus-pr60-reapply-smoke-10-v2/`이다. 후자는 병합된 `9548fc6`에서 실행됐으나 병합 커밋 `30c4db7` 자체를 보드에서 실행한 기록은 아니다.

REGULUS의 별도 `vendor_native_e2e` 5,000건 결과는 `로컬: results/results-20260927-regulus-5000-runtime-call/`에 있다. 모델별 MXQ는 R-R/R-Y/R-P와 연결되지만 **프레임워크 `--inference-mode e2e` 결과가 아니다**. 해당 CSV의 NPU bundle은 `unforced`로 기록돼 있으므로 현재 프레임워크의 강제 bundle 0·core 확인 증거로 대신하지 않는다. 또한 이전 `async_queue` 품질 포함 시험 `results/regulus-async-full-framework-20261001/summary.md`는 500건을 완료해도 세 모델의 출력 정확도가 실패했고, 직접 SDK async에서도 출력 반복이 관찰됐다. 이 실패 이력을 위 runtime-only `valid` 결과로 덮어쓰지 않는다. 현재 main의 10건 smoke 이후 전체 품질 재검증은 미수집이다. 기존 [REGULUS 운영 가이드](../../docs/mobilint-regulus-runtime.md)의 async 상태 설명도 이 최신 코드·증거에 맞춰 후속 단계에서 갱신해야 한다.

## 서버 모델 × target × artifact × mode

아래는 [중앙 검증 현황](cross-npu-model-validation.md)에 기록된 실행 범위다. 표의 run·artifact 원본은 이번 조사에서 직접 재확인하지 않았으므로 **문서상 기록**으로 표시한다. TTM-R2의 세 실행은 동일 ETTh1 `OT` 240-window 계약을 사용한다. `full`은 당시 실행 범위이며 현재 `main@30c4db7`을 세 서버에서 새로 돌렸다는 뜻이 아니다.

| 모델 | target / artifact | mode·실장비 `D` | 컴파일 `C` | 품질 `Q` | 전체 `F` / 전력 `P` |
|---|---|---|---|---|---|
| `ttm-r2` | `furiosa-rngd-torch` / 고정 PyTorch checkpoint | `e2e`, RNGD 240 window | strict compile 기록 | MAE·RMSE 계산, tensor parity 실패 별도 | 240 완료 / 비교 전력 미수집 |
| `ttm-r2` | `rbln-static` / RBLN `4159ce14…bf172` | `e2e`, CA22 240 window | 사전 컴파일 artifact | MAE·RMSE 계산, tensor parity 실패 별도 | 240 완료 / 비교 전력 미수집 |
| `ttm-r2` | `mobilint-aries` / MXQ `208958c8…2e217` | `e2e`, ARIES 240 window | 사전 컴파일 artifact | MAE·RMSE 계산, CPU 대비 허용 기준 미승인 | 240 완료 / 비교 전력 미수집 |
| `bert-base-uncased-squad-v1` | `furiosa-rngd-torch` / 로컬 checkpoint | `e2e`, `c5113b3b`, 1 sample | Transformers 4.57.6에서 strict compile | 1 sample EM/F1 계산 | full 미수행 / 비교 전력 미수집 |
| `bert-base-uncased-squad-v1` | `rbln-static` / 3-input RBLN `caada10a…0b33` | 단일 sample 출력·span 확인 | 사전 컴파일 artifact | context mask 포함 전체 evaluator 미승인 | full 미수행 / 비교 전력 미수집 |
| `bert-base-uncased-squad-v1` | `mobilint-aries` / legacy MXQ `5d1ff5a2…4d03a2` | standalone 64 sample 기록 | 사전 컴파일 artifact | legacy 문자열 EM/F1; 현재 CLI 판정과 분리 | 현재 CLI full 근거 미수집 / 비교 전력 미수집 |
| `llama-3.1-8b` | `furiosa-rngd` / 배포 FXB·HF 모델, 정확한 hash 미수집 | `e2e`·native async 각 1,000건, 후속 `9f06dae7` 2-token smoke | 제공 AOT | 실행·생성 확인; 생성 품질 승인 근거 미수집 | 두 모드 1,000건 기록 / 비교 전력 미수집 |
| `llama-3.2-3b` | `furiosa-rngd` / custom FXB, 정확한 hash 미수집 | `e2e`·native async 완료 기록 | custom compile, nearest-preset fallback | 생성 품질 승인 근거 미수집 | 조건부 full 기록 / 비교 전력 미수집 |
| `llama-3.1-8b` | `rbln-vllm` / one-NPU 준비 디렉터리·manifest | sync `a3168997` 1 sample; async `9dd3bf7a` 4 sample | 사전 준비, 공식 8-NPU 구성 아님 | 1-token 수준, 품질 미판정 | full 아님 / 비교 전력 미수집 |
| `llama-3.2-3b` | `rbln-vllm` / one-NPU 준비 디렉터리·manifest | sync `b7808504` 1 sample; async `a307b84f` 4 sample | 사전 준비, 공식 8-NPU 구성 아님 | 생성 smoke, 품질 미판정 | full 아님 / 비교 전력 미수집 |
| `llama-3.1-8b` | `mobilint-aries-llm` / Model Zoo W4V8 snapshot | `1076af18`, 8-token lifecycle smoke | 벤더 제공 모델 | 생성 smoke, 품질 미판정 | full 근거 미수집 / 비교 전력 미수집 |
| `llama-3.2-3b` | `mobilint-aries-llm` / Model Zoo snapshot | 성공 보고, run bundle 미수집 | 벤더 제공 모델 | 미수집 | full 근거 미수집 / 비교 전력 미수집 |

TTM-R2의 checkpoint·데이터·RBLN·MXQ **전체 SHA-256**은 [TTM-R2 고정 계약](ttm-r2-framework.md#고정-계약)에 있다. BERT와 Llama의 남은 artifact·환경·run ID 결손, RBLN one-card 제한은 [중앙 검증 현황](cross-npu-model-validation.md)에 유지한다. LLM의 `generate()` timing을 위 정적 모델의 Runtime-call latency와 같은 열로 환산하지 않는다.

## 아직 실행 조합으로 승인하지 않은 범위와 다음 증거

다음은 코드 또는 실행 가이드에는 있지만 위 표와 같은 **artifact hash + run ID + mode + 품질/완료 상태**의 묶음을 이 조사에서 확정하지 못한 조합이다. 명령 예제는 성공 기록이 아니다.

| 문서상 조합 | 확인된 위치 | 이 증거표에 남은 일 |
|---|---|---|
| ONNX Runtime `cpu`/`cuda`의 ResNet·YOLO·BERT·PatchTST | [사용 안내](../README.md#지원-모델), [기본 프로필](../src/core/model_profiles.py) | 모델별 ONNX hash, 실행 CSV와 정확도 조건을 결합. CUDA provider 나열과 실제 GPU 실행을 구분 |
| `rbln-static`의 `resnet50`, `yolov5m`, SST-2 BERT, `patchtst-fm-r1` | [RBLN artifact 계약과 예제](rbln-setup.md#5-model-contract) | 각 최종 RBLN·sidecar hash, 실장비 run ID·표본·평가 범위를 확인 |
| `mobilint-aries`의 `resnet50`, `yolov5m`, SST-2 BERT, `patchtst-etth1` | [ARIES 실행 예제](../../docs/mobilint-aries-transformers.md#5-static-transformer-실행), [컴파일 시도 기록](../../docs/mobilint-compilation-experiments.md) | model-specific MXQ·host weights·compiler attempt와 현재 CLI 결과를 연결. SST-2의 과거 64개 직접 평가는 현재 CLI full 결과가 아님 |
| `yolov8m`, `yolov8m-seg`, `yolov8m-pose`, `yolov8s-seg` | [기본 프로필](../src/core/model_profiles.py) | target·artifact별 실제 출력 ABI, decoder, evaluator, run 원본을 확인 |
| `vllm-cpu`, `vllm-cuda` | [target registry](../src/core/targets.py) | 환경별 준비 모델과 실제 생성 run을 확인 |
| `vendor_mock_npu` | [target registry](../src/core/targets.py) | SDK 없는 구조 시험으로 유지; 실장비 결과로 표시하지 않음 |

- Hailo-10H ResNet-50 교정 동기 원본은 로컬 보관본에서 찾지 못했다. 28.08% 과거 행을 73.82% 교정 결과로 대체하지 않는다. 로컬 `result/final-result/edge-report-5.5/README.md`는 해당 원본을 `source_unavailable`로 표시하지만 이 파일은 Git 추적 대상이 아니다.
- 최신 main 조상에서 REGULUS Pose 10건 품질·세 모델 10건 runtime-only smoke가 생겼다. 같은 artifact와 현재 코드의 5,000건 품질 패스, 종료·오류 회계, 정확도 기준을 새로 확인해야 과거 async 품질 실패 상태를 갱신할 수 있다.
- Hailo-8·DeepX의 INA3221 로컬 전력 기록은 Jetson DC 입력단과 두 모드별 시간 경계를 보존한다. Hailo-10H·REGULUS 및 서버 NPU의 동일 경계 전력은 이 문서에서 미수집이다. `0 W` 또는 다른 telemetry domain으로 채우지 않는다.
- 제출용 증거 묶음에는 각 행의 코드 revision, 실행 명령, SDK/driver, 모델 원천·compiler recipe, artifact SHA-256, 데이터 manifest, CSV·details·trace, 종료·오류 카운터와 전력 측정점을 묶어야 한다. 지금의 로컬 `results/`는 Git에 포함되지 않으므로 인수받은 사람이 이 문서만으로 원본을 재검증할 수 없다.

문서 갱신 시에는 기존 run을 덮어쓰지 않고 같은 조합에 날짜·revision·환경·run ID를 새 행으로 추가한다. 지원 여부를 요약할 때도 `A/C/D/Q/F/P` 중 확인된 게이트와 미수집 게이트를 함께 제시한다.
