# Mobilint ARIES 다중 모델 컴파일 실험 설계

## 목적

Mobilint `qbcompiler==1.2.0`을 실제로 어떻게 사용했는지와 각 시도가 어느 단계에서
성공하거나 실패했는지를 재현 가능한 형태로 남긴다. 대상은 BERT SST-2, BERT SQuAD
v1, PatchTST ETTh1, ResNet50 ImageNet1K V2, YOLOv5m이다. 기존 BERT 성공 경로는
보존하고 나머지 세 모델에는 실제 compiler recipe와 공통 실험 기록 규약을 추가한다.

모든 모델의 성공을 전제로 하지 않는다. 끝까지 컴파일되지 않은 모델도 source,
calibration, 명령, 환경, 실패 단계, 오류, 재시도 변경점을 보존하면 유효한 실험
결과다.

## 플랫폼과 고정 조건

| 항목 | 값 |
|---|---|
| compiler host | Ubuntu 22.04, Intel x86-64 |
| Python | CPython 3.10 |
| compiler | `qbcompiler-1.2.0-py3-none-any.whl` |
| wheel SHA256 | `28f276baef1bff86ed313cb819b53d8abb684a7555cf4c81c459edc09abf1b4b` |
| target device | `aries-rb` |
| runtime device | Mobilint ARIES |
| runtime 기준 환경 | `qbruntime v1.3.2`, ARIES driver `1.13.0`, firmware `1.2` |

공식 compiler 1.2 문서에서 `aries-rb`는 ARIES target 문자열이며, MXQ의 core mode는
컴파일할 때 결정되고 나중에 바꿀 수 없다고 명시한다. 따라서 BERT는 `single`,
PatchTST·ResNet50·YOLOv5m은 `global8`로 고정한다.

## 선택한 구조

### 기존 BERT 경로 보존

현재 `framework/tools/mobilint_bert_compile`과
`framework/scripts/compile_mobilint_bert.sh`는 실제 성공한 BERT 전용 embedding
경계를 표현하므로 일반화 과정에서 변경하지 않는다. 공통 실험 실행기는 이 기존
entrypoint를 호출하고 결과를 같은 attempt 형식으로 수집한다.

### 모델별 recipe

새 package `framework/tools/mobilint_compile_recipes`에 공통 계약·기록 코드와 다음
recipe를 둔다.

- `patchtst_etth1`
- `resnet50`
- `yolov5m`

각 recipe는 `describe`, `prepare`, `source-smoke`, `mblt`, `mxq` 단계를 독립적으로
실행할 수 있어야 한다. 한 단계가 실패해도 그 전 단계의 manifest와 로그는 남긴다.
compiler import는 실제 `mblt` 또는 `mxq` 단계까지 지연해 일반 개발 환경에서 계약과
준비 로직을 테스트할 수 있게 한다.

### 공통 attempt 실행기

`framework/scripts/run_mobilint_compile_experiment.sh`가 모델과 variant를 받아 새
timestamp attempt를 만든다. 기존 attempt는 덮어쓰지 않는다.

```text
mobilint-compile-attempts/<timestamp>/<model>/<variant>/
├── environment.json
├── source-manifest.json
├── command.txt
├── compile.log
├── result.json
├── calibration/
├── mblt/
└── mxq/
```

shell 실행기는 Python stage를 별도 process로 호출하고 시작·종료 시각, elapsed time,
exit code와 signal 종료를 기록한다. 환경 기록에는 OS, architecture, Python과 package
version, wheel filename/checksum만 포함하며 token, credential, 전체 환경 변수는 기록하지
않는다.

## 모델 계약과 source

### BERT SST-2와 SQuAD v1

- source: `textattack/bert-base-uncased-SST-2`,
  `csarron/bert-base-uncased-squad-v1`
- 현재 recipe의 source revision, validation dataset 32개 deterministic index,
  embedding weight와 calibration manifest를 그대로 사용한다.
- MXQ 입력: `Float32 [1,-1,768]`
- SST-2 출력: logits `[1,1,2]`
- SQuAD runtime 출력 순서: `end_logits`, `start_logits`; 각 `[1,-1,1]`
- core mode: `single`

### PatchTST ETTh1

- source: `ibm-granite/granite-timeseries-patchtst`
- 최초 준비 실행은 요청 revision을 Hugging Face commit SHA로 해석해
  `source-manifest.json`에 기록한다. 동일 실험의 재시도는 그 SHA만 사용한다.
- 입력: `past_values float32 [1,512,7]`,
  `past_observed_mask bool [1,512,7]`
- 출력: `prediction_outputs float32 [1,96,7]`
- calibration: ETTh1에서 균등하게 고른 32개 고정 window
- 다중 입력 calibration은 sample별 디렉터리와 입력 순서를 명시한 JSON manifest를
  사용한다. 같은 shape인 values와 mask를 파일명 추측으로 매칭하지 않는다.
- core mode: `global8`

먼저 Hugging Face 원본 모델을 변경하지 않은 `stock` variant로 source smoke와
컴파일을 시도한다. `aten::unfold` 또는 boolean mask lowering 같은 실제 실패가
관측된 경우에만 `compat-static-patchifier` variant를 별도 attempt로 실행한다. 호환
wrapper는 fixed 42-patch 구성을 사용하고 원본 CPU 출력과 허용 오차 안에서 같음을
compiler 호출 전에 검사한다. stock 실패 기록은 성공한 호환 attempt가 생겨도
삭제하지 않는다.

### ResNet50 ImageNet1K V2

- source: TorchVision `resnet50(weights=IMAGENET1K_V2)`
- source manifest에는 installed TorchVision version과 weight enum, weight file
  SHA256을 기록한다.
- runtime 입력: `uint8 NHWC [1,224,224,3]`
- 출력: class logits 1000개
- calibration: ImageNet validation에서 균등하게 고른 RGB uint8 crop 32개
- core mode: `global8`

source wrapper는 NHWC를 NCHW로 변환하고 ImageNet1K V2 mean/std 정규화를 수행한다.
compiler에는 `Uint8InputConfig`를 명시해 runtime ABI가 uint8이 되게 한다. CPU source
smoke는 wrapper 출력과 TorchVision 공식 transform 경로의 logits가 같은지 검사한다.

### YOLOv5m

- source: 고정된 YOLOv5 Git commit
  `86fd1ab270cb2f7e53ee7412cd4a0650bf4bcc51`
- weight: 사용자가 제공하거나 공식 release에서 받은 `yolov5m.pt`; non-empty 검사와
  SHA256을 source manifest에 기록한다.
- runtime 입력: `uint8 NHWC [1,640,640,3]`
- 출력: `(20,20,255)`, `(40,40,255)`, `(80,80,255)` raw head 세 개
- calibration: COCO128에서 정렬 순서로 균등하게 고른 RGB uint8 letterbox 32개
- core mode: `global8`

source wrapper는 uint8 입력을 NCHW float `[0,1]`로 변환하고 NMS, AutoShape, decoded
prediction을 포함하지 않는다. detect layer의 세 scale을 기존 runtime 순서와 NHWC
shape로 반환한다. compiler를 시작하기 전에 raw head 개수, shape, 유한값과 고정 source
revision을 검사한다.

## compiler 설정

모든 attempt는 `target_device="aries-rb"`, `backend="torch"`, 고정 wheel checksum과
명시적인 `CalibrationConfig`를 기록한다. BERT는 현재 검증된 percentile 설정을
보존한다. 새 recipe의 첫 attempt도 같은 보수적 설정에서 시작하되 모델별로 다음을
추가한다.

- PatchTST: explicit multi-input calibration JSON
- ResNet50: `Uint8InputConfig`와 `classification_torchvision` preset의 resolved dump
- YOLOv5m: `Uint8InputConfig`와 `yolo_640` preset의 resolved dump

실제 오류 때문에 설정을 바꾸면 기존 attempt를 수정하지 않고 새 attempt에 전체
설정 dump, 부모 attempt ID와 변경 이유를 기록한다. `use_random_calib`은 source parser
경로 확인용으로 별도 variant에서만 허용하며 품질용 MXQ 성공으로 판정하지 않는다.

## 단계와 판정

모든 모델은 다음 stage를 순서대로 수행한다.

1. `SOURCE_PREPARE`
2. `SOURCE_SMOKE`
3. `CALIBRATION_PREPARE`
4. `MBLT_COMPILE`
5. `MXQ_COMPILE`
6. `ARIES_LOAD`
7. `CONTRACT_CHECK`
8. `TASK_SMOKE`

`result.json`은 각 stage에 `not_run`, `pass`, `fail` 중 하나와 시작·종료 시각,
elapsed seconds, exit code, 핵심 오류를 기록한다. 최종 판정은 다음 독립 필드로 둔다.

- `compile_status=pass`: non-empty MBLT와 MXQ 및 SHA256 생성
- `runtime_status=pass`: ARIES load, launch, single inference, dispose 성공
- `contract_status=pass`: core mode, 입력·출력 개수, dtype, 논리 shape와 순서 일치
- `quality_status=recorded`: framework metric과 sample 수를 기록

어느 단계든 실패하면 `failed_at`에 첫 실패 stage를 기록하고 이후 stage는
`not_run`으로 남긴다. compiler가 segmentation fault나 signal로 종료돼도 shell
실행기가 process return code와 `compile.log`를 보존한다.

`quality_status`는 컴파일 성공의 전제조건이 아니다. 컴파일과 배포 호환 성공을 먼저
판정하고, BERT accuracy/EM/F1, PatchTST error metric, ResNet Top-1/Top-5, YOLO mAP는
관측값과 sample 수로 별도 기록한다.

## ARIES 검증 흐름

컴파일은 사용자가 서버에서 수행한다. 성공한 MXQ는 같은 서버의 ARIES에서 다음
순서로 검사한다.

1. `inspect_mobilint_mxq.py` 또는 동등한 qbruntime metadata 검사
2. 지정 core mode로 construct와 launch
3. 준비 단계가 저장한 smoke input으로 한 번 추론
4. output shape/dtype/유한값 검사
5. model dispose와 장치 memory 회수 확인
6. framework `e2e` 실행으로 task metric 기록

서버 command는 `tee`와 명시적인 `PIPESTATUS[0]`을 사용해 터미널이 종료돼도 로그와
실제 exit code가 남게 한다. 사용자가 `result.json`, 요약 출력과 필요한 오류 구간을
전달하면 recipe 변경은 새 variant 또는 새 attempt로 추가한다.

## 저장소 기록

대용량 `.mblt`, `.mxq`, calibration array, model weight, 전체 compiler log와 dataset은
Git에 커밋하지 않는다. 저장소에는 다음만 남긴다.

- compiler recipe와 shell entrypoint
- pure contract와 wrapper 단위 테스트
- 모델별 source·calibration·compiler 옵션 문서
- 실행한 attempt의 환경, 명령, elapsed time, artifact SHA256, stage 판정
- 실패 오류의 필요한 부분과 해결 과정
- ARIES contract와 task smoke 요약

민감한 경로, credential, token은 결과를 문서화하기 전에 제거한다. 절대 경로는
`$REPO`, `$HOME`, `$ATTEMPT_ROOT` 같은 역할 기반 표기로 정규화한다.

## 테스트 전략

일반 개발 환경에서 다음을 자동 검사한다.

- pure 모델 계약과 CLI `--describe`
- calibration index와 manifest 순서, shape, dtype, 유한값
- source revision과 weight checksum guard
- fake compiler를 사용한 exact `mblt_compile`·`mxq_compile` argument
- 작은 fake model을 사용한 NHWC/normalization과 output reorder
- 실제 source dependency가 있을 때 PatchTST 원본/호환 wrapper CPU equivalence,
  ResNet official transform equivalence, YOLO raw-head 계약
- attempt state transition, signal/exit-code 기록, 기존 경로 overwrite 거부

실제 qbcompiler 호출과 ARIES 검증은 외부 하드웨어 실험으로 분리한다. 자동 테스트가
compiler 또는 장치 성공을 대신했다고 표현하지 않는다.

## 완료 조건

1. BERT 기존 성공 이력이 공통 결과 형식으로 연결된다.
2. PatchTST, ResNet50, YOLOv5m 각각 실제 compiler server attempt가 한 번 이상
   수행되고 성공 또는 실패 결과가 기록된다.
3. 생성된 MXQ가 있는 모델은 ARIES load와 contract 검사를 실제로 수행한다.
4. 성공·실패를 포함한 모든 실행 명령과 핵심 옵션을 문서에서 재현할 수 있다.
5. 실패 모델을 성공으로 표시하지 않고, random calibration 성공을 품질용 성공으로
   표시하지 않는다.
6. 관련 자동 테스트가 통과하고 생성 artifact가 Git에 포함되지 않는다.
