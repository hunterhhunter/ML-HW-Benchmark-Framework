# TTM-R1 ARIES ETTh1 품질 평가 설계

## 목적

IBM Granite TTM-R1의 고정 scaler-free core를 Mobilint ARIES에서 실행하고, INT8
post-training quantization(PTQ)이 ETTh1 zero-shot OT 예측 품질에 미치는 영향을 CPU
reference와 비교한다. random smoke calibration이 만든 49.8% input saturation을 실제
train distribution 기반 calibration으로 대체한다.

## 범위

- checkpoint: `ibm-granite/granite-timeseries-ttm-r1`.
- 데이터: ETTh1 CSV의 단변량 `OT`, float32.
- calibration: ETTh1 train split `[0, 8640)` 안에서 과거 512시간 context만 쓰는
  균등 추출 256개 origin.
- task evaluation: standard test split 시작점 `11520`부터 순차적인 240개 origin,
  context 512시간과 horizon 96시간.
- target: `aries-rb`, local `mobilint/qbcompiler:1.2-cpu-ubuntu22.04` Docker에서
  `.mxq`를 만들고 원격 ARIES/qbruntime v1.3.2에서 실행한다.

이번 단계는 모델 fine-tuning, test data calibration, 7채널 평가, RBLN 재평가를
포함하지 않는다.

## 구성과 데이터 경계

1. 공통 ETTh1 모듈은 train origin을 균등 선택하고, 각 raw `[1,512,1]` context를
   기존 `TTMR1HostAdapter`로 standard scaling한다. 생성된 256개의 core input은
   `[1,512,1]` float32 `.npy` 파일과 manifest로 저장한다.
2. 로컬 Docker compile 도구는 동일 CPU core를 ONNX로 export하고, 첫 calibration
   input을 `feed_dict`로, calibration directory를 `calib_data_path`로 전달하여
   `qbcompiler.mxq_compile_V2`를 호출한다. `target_device="aries-rb"`,
   `device="cpu"`, `cpu_offload=False`, `use_random_calib=False`가 고정이다.
3. QBC가 생성한 `.mxq`와 compile evidence, 대표 finite fixture를 SSH로 원격
   ARIES 서버에 전송한다. runtime에는 compiler wheel이나 CUDA가 필요 없다.
4. 원격 evaluator는 qbruntime이 노출한 실제 artifact input/output shape와 scales를
   읽는다. 현재 artifact 형태는 input `[1,8,64]` int8, output `[1,1,96]`이며,
   host-prepared `[1,512,1]` input은 static patch layout으로 reshape한 뒤 input
   scale로 quantize한다. runtime output은 `infer_to_float`으로 받고 `[1,96,1]`로
   transpose한 뒤 host adapter로 original OT scale로 복원한다.
5. 같은 240 context를 CPU core와 ARIES artifact에 각각 전달하고, CPU/ARIES task
   MAE·RMSE와 prediction delta를 기록한다.

## 증거와 판정

로컬 compile result에는 checkpoint/dataset/calibration manifest SHA-256,
calibration origin 범위와 수, ONNX SHA-256, `.mxq` SHA-256, compiler target/options를
쓴다. 원격 result에는 qbruntime version, input/output ABI, scales 요약, saturation
수, CPU/ARIES task MAE·RMSE, prediction delta, degradation percentage를 쓴다.

`compile_success`, `runtime_success`, `quantization_status`, `task_quality_status`를
분리한다. `task_quality_status`는 threshold pass/fail이 아닌 `measured`다. Saturation
비율과 task metric을 함께 보고 채택 여부를 판단한다.

## 오류 처리

- train/test split 밖 origin, 누락 CSV/model/calibration file, non-finite tensor,
  빈 calibration directory는 compile 전에 실패한다.
- `.mxq` artifact ABI가 기대한 static input element 수 512 또는 output element 수
  96과 다르면 remote inference 전에 실패한다.
- scale list와 runtime input shape가 맞지 않거나 quantization saturation이 발생하면
  count를 결과에 기록한다. saturation 자체는 artifact 실행을 막지 않지만 결과를
  `quantization_status: saturated`로 표시한다.
- compiler/runtime exception은 각각 immutable JSON에 저장하고 fallback을 사용하지
  않는다.

## 검증

- train origin selection, calibration sample shape, no-test-leakage, scale quantization,
  layout conversion, metric 계산은 SDK-free unit test로 검증한다.
- compiler invocation은 fake qbcompiler로 target/device/random-calibration/cpu-offload
  계약을 테스트한다.
- remote evaluator의 qbruntime integration은 fake model로 input shape, int8 scale,
  output transpose, saturation count를 테스트한다.

## 성공 정의

로컬에서 실제 train-calibrated ARIES `.mxq`를 생성하고, 원격 ARIES에서 240개 ETTh1
test origin을 실행해 CPU 및 device task metrics를 보존하면 완료다. strict core parity
결과와 task-quality 결과는 서로 대체하지 않는다.
