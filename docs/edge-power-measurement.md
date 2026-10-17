# 엣지 NPU 입력 전력 측정

## 측정 경계

INA3221은 Jetson 어댑터의 약 19 V DC 입력 +선에 직렬로 연결되어 있다. 채널 1의 0.05 Ω 션트에서 전류를 계산하고 버스 전압을 곱해 W를 기록한다. I²C 버스 7, 주소 `0x40`이다. 센서 ID는 `0x5449` / `0x3220`, 션트 전압은 40 µV/LSB, 버스 전압은 8 mV/LSB다. −선은 센서를 우회한다.

측정값은 **Jetson 시스템 전체 입력 전력**이다. 캐리어보드, 팬, CPU, NPU, 전원 변환 손실과 측정 프로세스의 영향을 포함한다. `추론 평균 W − 같은 회차 idle 중앙값 W`는 시스템 증가분이며 NPU 단독 전력이 아니다. NPU를 장착하거나 교체하면 idle 기준선을 다시 수집한다. 전류가 3.0 A를 넘으면 소프트웨어가 **측정을 중단**한다. 전원을 차단하는 보호 장치는 아니다.

## 프레임워크 경로

`--power-trace`를 켜면 `Ina3221Collector`가 별도 Python 프로세스에서 `smbus2`로 센서를 읽는다. 프레임워크는 워밍업 이후 60초 idle 기준선을 기록한 뒤 추론 구간을 기록한다. 기본 샘플 간격은 약 0.1초다. `--monitor`를 추가하면 벤더/시스템 telemetry도 수집하지만 원시 전력의 출처는 INA3221이다. 옵션을 끄면 센서를 열지 않는다. 센서가 없을 때 결과의 `power_trace_status`는 `unavailable`이며, 추론 결과를 0 W로 대체하지 않는다.

`power/<run_id>.power.csv`에는 `baseline`과 `inference` phase, 측정 시각, W, 샘플 상태와 오류 코드를 남긴다. 결과 CSV에는 동일한 run ID와 전력 CSV 경로·SHA-256·측정 범위를 남긴다. 원시 trace 분석 스크립트는 완결된 trace와 해시를 확인한 후 다음을 계산한다.

- `idle_median_w`: baseline 전력 샘플의 중앙값
- `inference_mean_w`: 추론 구간 전력을 사다리꼴 적분한 에너지 ÷ 실측 시간
- `incremental_j_per_sample`: (추론 구간 에너지 − idle 중앙값 × 실측 시간) ÷ 완료 샘플 수

비동기의 처리량은 **Runtime-call 반환 완료 샘플 수 ÷ 측정 시간**이다. 요청별 Runtime-call 평균 지연시간의 역수와 구분한다. J/건도 `async_runtime_completed_samples`를 분모로 쓰고 전체 완료 수와 일치하는 유효 회차만 분석한다. 입출력 준비가 느리면 처리량과 J/건 모두 입력 공급의 영향을 받는다. 0.1초보다 짧은 추론 한 건의 순간 전력은 분리할 수 없으므로 충분히 긴 연속 실행을 쓴다. 실행 명령, 의존성, 결과 검증 절차는 [INA3221 사용 가이드](../framework/docs/edge-ina3221-power.md)에 있다.

## 2026-10-16 Hailo-8 250건 기능 검사: E2E와 async

Hailo-8 장착 상태에서 ResNet-50, YOLOv8s-pose, YOLOv5m을 **System E2E와 async/offline으로 각각 250건씩** 측정했다. 당시 `feat/power_measurment` 코드 `65f28e67538aa1a30a7aee422ccc4003147d7b9e`의 `--power-trace` 경로를 사용했다. `MAXN_SUPER` mode 2에서 CPU 1,728 MHz, GPU 1,020 MHz, EMC 3,199 MHz를 고정하고, warmup 10회 뒤 매 회차 60초 idle을 측정했다. async worker는 ResNet-50 4개, 나머지 두 모델 2개다.

| 모델 | 모드 | run ID | 추론 구간 (s) | Idle (W) | 추론 평균 (W) | 증가분 (W) | 증가분 (J/건) |
|---|---|---|---:|---:|---:|---:|---:|
| ResNet-50 | E2E | `8c02aeae` | 1.282 | 10.736 | 11.928 | 1.192 | 0.0061 |
| ResNet-50 | async | `8bb446c4` | 0.699 | 10.823 | 13.477 | 2.653 | 0.0074 |
| YOLOv8s-pose | E2E | `1453fc47` | 10.171 | 10.819 | 12.693 | 1.875 | 0.0763 |
| YOLOv8s-pose | async | `b8762202` | 5.372 | 10.843 | 13.632 | 2.789 | 0.0599 |
| YOLOv5m | E2E | `ea846966` | 9.447 | 10.880 | 12.463 | 1.583 | 0.0598 |
| YOLOv5m | async | `365ed726` | 4.056 | 10.819 | 13.553 | 2.734 | 0.0444 |

여섯 회차 모두 250건 완료와 `power_trace_status=complete`를 확인했다. async 세 회차는 `valid`이며 요청 실패·거절·타임아웃·미완료가 0건이다. 원시 결과와 명령, 해시 검증 기록은 로컬 `results/2026-10-16-hailo8-ina3221-250-smoke/`에 보존했으며 이 PR에는 포함하지 않는다. 각 조건 1회 측정이고 ResNet-50의 전력 표본은 E2E 14개, async 8개뿐이므로 대표 전력·J/건으로 사용하지 않는다. **DeepX E2E 전력은 현재 보관 기록상 미측정이다.**

## 2026-10-16 Hailo-8·DeepX 5,000건 async 전력 측정

Jetson Orin Nano Super의 `MAXN_SUPER`(mode 2)에서 CPU 1,728 MHz, GPU 1,020 MHz, EMC 3,199 MHz를 고정했다. 팬은 동적 제어였다. Hailo-8과 DeepX DX-M1을 같은 보드에 번갈아 장착했다. 아래는 `async_queue`, offline, batch 1, worker 4, queue 256, warmup 10, 모델당 5,000건, 60초 idle 기준선의 선택 회차다. 품질 패스가 완료된 뒤 Runtime-call 패스를 전력 측정했다. 세 모델은 ResNet-50, YOLOv8s-pose, YOLOv5m이다.

| NPU | 모델 | Runtime-call 반환 처리량 (samples/s) | Idle (W) | 추론 평균 (W) | 증가분 (W) | 증가분 (J/건) | run ID |
|---|---|---:|---:|---:|---:|---:|---|
| Hailo-8 | ResNet-50 | 391.753 | 10.869 | 13.964 | 3.095 | 0.007900 | `4e4f2b57` |
| Hailo-8 | YOLOv8s-pose | 111.192 | 10.854 | 14.992 | 4.138 | 0.037214 | `1bc2b13b` |
| Hailo-8 | YOLOv5m | 62.929 | 10.775 | 13.792 | 3.017 | 0.047946 | `7fb5f6e4` |
| DeepX DX-M1 | ResNet-50 | 406.840 | 10.953 | 14.597 | 3.643 | 0.008955 | `671faf17` |
| DeepX DX-M1 | YOLOv8s-pose | 65.905 | 10.912 | 14.812 | 3.900 | 0.059170 | `80f58a72` |
| DeepX DX-M1 | YOLOv5m | 62.132 | 11.008 | 16.278 | 5.270 | 0.084825 | `5f450e9b` |

각 선택 회차는 5,000건 완료, 품질 평가 통과, 실패·거절·타임아웃 0건이었다. DeepX Pose 첫 회차 `4030d8cd`는 51.113 samples/s, 0.063081 J/건이었다. 입력 공급에 95.96초가 걸리고 대기열 최대 깊이가 1이었다. 재측정은 입력 공급 73.94초, 65.905 samples/s였고 대기열 최대 깊이가 1이었다. 두 회차의 idle 차이는 0.021 W였다. Pose 수치의 변동을 감안해 이 표를 NPU 포화 성능 순위로 읽지 않는다. ResNet-50과 YOLOv5m은 각 1회차이므로 분산은 산출하지 않았다.

이 표의 **보드 실행은 독립 INA3221 sampler와 프레임워크 Runtime-call 구간 마커 패치**로 수행했다. 이 PR에 추가한 `--power-trace` 경로의 보드 실행 결과로 표기하지 않는다. 원시 CSV, 프레임워크 상세 JSON, 명령, 코드/아티팩트 해시와 manifest는 로컬 측정 보관본에 보존했다. Hailo-8과 DeepX의 모델 artifact 및 실행 코드는 각각 보관본에서 SHA-256으로 확인했다.

로컬 보관 경로는 `results/2026-10-16-hailo8-ina3221-runtime-return-w4/`와 `results/2026-10-16-deepx-ina3221-runtime-return-w4/`이다. 각 디렉터리의 `selected-summary.csv`, `manifest.sha256`, 모델별 `sensor.csv`·`markers.csv`·`results.csv`·`details/<run_id>.json`을 대조할 수 있다. 원시 측정 파일은 이 PR에 포함하지 않는다.

## 2026-10-17 새 프레임워크 경로 기능 확인

같은 Jetson에 DeepX DX-M1을 장착한 상태에서 이 PR의 `--power-trace` 경로로 ResNet-50 native async Runtime-call 500건을 실행했다. 설정은 offline, batch 1, worker 4, queue 256, warmup 10, idle 1초다. DeepX 실행 환경에 `smbus2`가 없어 기존 Hailo 실행 환경의 패키지를 별도 `PYTHONPATH`로 제공했다. 수집기는 I²C 버스 7의 INA3221을 읽었고 결과 `run_id=c4d1f1f0`은 500건 완료, 실패·거절·타임아웃 0건, `power_trace_status=complete`, `power_scope=jetson_dc_input`이었다. trace에는 baseline 11개, inference 26개 표본이 있으며 SHA-256은 `2d689e179f033c76b7f6909f5a2ece03affd7e015146a923e660ffa52c9da644`이다. 분석 스크립트는 idle 중앙값 9.036 W, 추론 시간 가중 평균 11.555 W, 시스템 증가분 0.012119 J/건을 반환했다.

이 회차는 **코드 경로 기능 확인용**이다. idle 1초, 500건, 동적 클럭이고 품질 평가는 건너뛰었으므로 위 5,000건 고정 클럭 결과나 보고서의 성능·전력 대표값으로 사용하지 않는다. 원시 결과는 보드의 `/home/swlab2/benchmark-data/runs/2026/2026-10-17-deepx-ina3221-framework-smoke/`에, 실행 로그는 `/home/swlab2/power-measure/edge-power-pr-smoke-20261017/deepx-smoke.log`에 보존했다.

## 반복 측정과 보고

전원 인가 전 노출 구리선 절연을 확인한다. `nvpmodel -q`, 클럭, 팬 상태, SDK, 모델 artifact SHA-256, 데이터셋과 전처리를 기록한다. 각 조건에서 60초 이상 idle을 먼저 측정하고, 조건당 3회 이상 반복해 기준선과 증가분 편차를 제시한다. 보고서에서는 품질, Runtime-call 지연시간, 완료 처리량, 시스템 입력 전력, 시스템 증가분 J/건을 별도 지표로 표기한다.
