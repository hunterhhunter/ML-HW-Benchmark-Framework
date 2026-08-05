# TTM-R1 Furiosa ETTh1 품질 평가 설계

## 목적

IBM Granite TTM-R1의 scaler-free prediction core가 Furiosa RNGD에서 strict graph로
실행되는지와 별개로, 실제 ETTh1 zero-shot 예측 품질이 CPU reference 대비 어느 정도
변화하는지 기록한다. 이 평가는 tensor-level strict parity를 완화하거나 대체하지
않는다. 결과는 `runtime_success`, `strict_parity`, `task_quality`를 서로 다른 상태로
보존한다.

## 범위

- 데이터: ETTh1 CSV의 `OT` 단변량.
- 평가: 표준 ETTh1 test split의 첫 240 rolling origins.
- 한 origin의 입력/정답: 직전 512시간의 context와 직후 96시간의 target.
- 모델: 로컬에 이미 내려받은
  `ibm-granite/granite-timeseries-ttm-r1` checkpoint.
- 장치: `furiosa:0` RNGD, `fullgraph=True`, `dynamic=False`,
  `eager_fallback=False`.

이번 단계는 다변량 7채널 평가, latency benchmark, ARIES calibration, RBLN 품질 평가를
포함하지 않는다.

## 데이터 흐름

1. CSV를 시간 순서대로 읽고 `OT`를 float32로 변환한다.
2. ETTh1의 관례적 split `(train=8640, validation=2880, test=2880)`를 사용한다.
   각 test origin은 test 시작점부터 순차적으로 선택하며, context가 split 경계 이전을
   참조하는 것은 허용한다. 미래 target은 입력이나 calibration에 사용하지 않는다.
3. 기존 `TTMR1HostAdapter`가 각 `[1,512,1]` context의 standard scaling, NaN 처리,
   scaler-free core 입력 준비, prediction restoration을 CPU에서 담당한다.
4. CPU reference core와 Furiosa compiled core는 정확히 같은 prepared core input을
   받는다. Furiosa graph는 첫 origin에서 한 번만 compile하고 나머지 origins에서
   재사용한다.
5. 두 core 출력은 각각 host adapter로 원래 OT scale의 `[96]` prediction으로
   복원하고, 동일한 ground truth와 비교한다.

## 산출물과 지표

단일 결과 JSON은 다음을 포함한다.

- dataset 경로와 SHA-256, column, split 및 window 계약
- checkpoint manifest/weight hash와 Furiosa execution mode
- CPU와 RNGD 각각의 task `mae`, `rmse`
- CPU와 RNGD prediction 간 `mae`, `rmse`, `max_abs_error`
- `(rngd_metric / cpu_metric - 1) * 100` 형태의 MAE/RMSE 열화율(분모가 0이면 null)
- `runtime_success`, `strict_parity_status`, `task_quality_status` 및 오류 정보

`task_quality_status`는 이 단계에서 임의의 수치 threshold로 pass/fail하지 않는다.
두 task metric과 열화율을 기록하는 `measured` 상태만 사용한다. strict 1e-3 tensor
parity의 기존 실패는 그대로 `parity_failed`로 기록한다.

## 오류 처리

- CSV에 `OT`가 없거나 window 수가 부족하면 NPU compile 전에 실패한다.
- CPU prediction, NPU prediction, target 중 하나라도 non-finite이면 해당 run을
  실패시키고 JSON에 원인을 남긴다.
- Furiosa compile/runtime exception은 eager fallback 없이 즉시 실패로 기록한다.
- dataset/checkpoint를 자동 다운로드하지 않는다. 누락 시 필요한 경로를 명시해
  실패한다.

## 검증

- 순수 데이터 window/split/metric 계산은 작은 synthetic series unit test로 검증한다.
- evaluator가 CPU와 Furiosa 입력에 같은 prepared tensor를 전달하는지는 fake Furiosa
  runner test로 검증한다.
- 실제 RNGD runbook은 ETTh1 준비, unit test, 결과 JSON pretty-print 순서를 제공한다.

## 성공 정의

이 기능의 완료는 결과 JSON이 실제 CPU와 RNGD 예측·정답 기반 지표를 보존하는 것이다.
모델의 채택 여부는 생성된 CPU 대비 ETTh1 MAE/RMSE 열화율을 사용자가 검토한 뒤
결정한다.
