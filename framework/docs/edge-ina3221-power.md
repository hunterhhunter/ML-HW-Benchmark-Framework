# Jetson INA3221 입력 전력 측정

## 측정 범위와 실행 경계

`hailo8`, `hailo10h`, `deepx` target에서 `--power-trace`를 켜면 프레임워크가
INA3221 reader subprocess를 시작한다. 자식 프로세스만 `smbus2`로 I²C 버스 7,
주소 `0x40`을 열어 채널 1의 버스 전압과 션트 전압을 읽는다. 션트 저항은
`0.05 Ω`이고, 계산값은 `bus_V × current_A`이다. 기본 표본 간격은 0.1초다.
센서가 없는 환경에서는 `power_trace_status=unavailable`로 남는다.

측정점은 **Jetson 어댑터의 DC 입력**이다. 결과에는 캐리어보드, 팬,
전원 변환 손실, Jetson CPU, NPU 및 측정 프로세스의 영향이 함께 들어간다.
따라서 idle 차감값도 NPU 코어만의 전력으로 표기하지 않는다. NPU 미장착
idle과 NPU 장착 idle도 서로 다른 상태이므로 장착 후 기준 구간을 다시 측정한다.

## 실행 전 확인

전원 인가 전에 케이블의 노출된 구리선을 절연했는지 확인한다. 벤치마크에
사용할 **같은 Python 환경**에 `smbus2`를 설치하고 I²C 접근 권한을 확인한다.

```bash
cd "$HOME/ML-HW-Benchmark-Framework/framework"
python3 -m pip install smbus2
python3 -c 'import smbus2; print(smbus2.__file__)'
test -r /dev/i2c-7 && test -w /dev/i2c-7
i2cdetect -y -r 7
nvpmodel -q
```

`python3`은 실제 HailoRT/DX-RT 실행에 사용하는 인터프리터로 바꾼다.
`i2cdetect`의 `0x40`은 연결 확인에만 사용하고, 측정 중에는 실행하지 않는다.
전력 모드, 클럭 설정, 팬 상태, 모델 artifact, 데이터셋과 반복 횟수를 기록한다.

## 프레임워크 실행

아래는 이미 컴파일한 DEEPX 모델의 native async 예시다. artifact와 데이터셋
경로를 실제 파일로 바꾼다. `--power-trace`만 쓰면 전력 수집기를 실행하며,
`--monitor`를 추가하면 기존 벤더/시스템 모니터도 함께 실행한다. 전력 trace의
출처는 두 옵션을 함께 써도 INA3221로 고정된다. 기준 구간은 워밍업 뒤에
시작하며, 엣지 기본값은 60초다. 짧은 단발 추론을 피하려면 5,000건처럼 연속 실행해 추론 구간을 충분히 확보한다.

```bash
python3 src/main.py \
  --model resnet50 --target deepx --no-compile \
  --artifact /path/to/ResNet50.dxnn \
  --dataset /path/to/imagenet_1k \
  --layout NCHW --batch-size 1 \
  --inference-mode async_queue --scenario offline \
  --worker-count 4 --queue-capacity 256 \
  --min-duration-sec 0 --min-samples 5000 --max-samples 5000 \
  --power-trace --power-baseline-sec 60 \
  --results-path results/deepx_edge_power.csv
```

Hailo-10H의 동기 E2E 예시는 다음과 같다. 데이터셋이 짧으면 측정 구간도
짧아지므로 충분한 샘플을 준비한다.

```bash
python3 src/main.py \
  --model resnet50 --target hailo10h \
  --hef /path/to/ResNet50.hef \
  --dataset /path/to/imagenet_1k \
  --layout NHWC --batch-size 1 --warmup 2 \
  --power-trace --power-baseline-sec 60 \
  --results-path results/hailo10h_edge_power.csv
```

`--power-baseline-sec`는 실험 계획에 맞춰 변경할 수 있다. 아래 표의 2026-10-16 보드 측정은 독립 센서 sampler와 프레임워크 마커 패치를 사용했으며, 이 CLI 경로의 새 실행 결과로 혼동하지 않는다. 해당 원시 실행 방법과 수치는 [측정 기록](../../docs/edge-power-measurement.md)에 정리했다. 실행마다
센서가 채널 1만 연속 변환하도록 설정하고 ID를 확인한다. 전류가 3.0 A를
넘거나 음수가 되면 reader가 전력 수집을 멈추고 해당 표본에 오류를 기록한다.
이는 **전원 차단 기능이 아니다**.

## 결과 검증과 에너지 계산

결과 CSV의 `run_id`, `power_trace_status`, `power_trace_path`,
`power_trace_sha256`, `power_scope`를 먼저 확인한다. 전력 원본은 결과 CSV와
같은 디렉터리 아래 `power/<run_id>.power.csv`에 있다. `phase=baseline`은
워밍업 뒤 idle이고 `phase=inference`는 프레임워크 실행 구간이다.
`power_scope`는 `jetson_dc_input`이어야 한다.

```bash
python3 - results/deepx_edge_power.csv <<'PY'
import csv
import sys

with open(sys.argv[1], newline='', encoding='utf-8') as handle:
    row = list(csv.DictReader(handle))[-1]
for key in ('run_id', 'power_trace_status', 'power_trace_path',
            'power_trace_sha256', 'power_scope', 'async_completed_samples'):
    print(f'{key}={row.get(key, "")}')
PY

RUN_ID=$(python3 -c 'import csv; print(list(csv.DictReader(open("results/deepx_edge_power.csv", newline="", encoding="utf-8")))[-1]["run_id"])')
python3 scripts/analyze_edge_power.py results/deepx_edge_power.csv \
  --run-id "$RUN_ID"
```

분석기는 run ID와 SHA-256이 맞고 모든 전력 표본이 유효한 `complete` trace만
받는다. async 결과는 `async_run_status=valid`이고
`async_runtime_completed_samples`와 `async_completed_samples`가 일치해야 한다.
J/sample의 분모에는 Runtime-call 반환 완료 표본 수를 사용한다.
baseline 표본의 중앙값을 idle W로 사용한다. 추론 구간의 시스템
에너지는 전력 표본 사이를 사다리꼴로 적분하며, 증분 에너지는
`시스템 에너지 − idle 중앙값 × 추론 구간 시간`으로 계산한다. `J/sample`은
해당 에너지를 완료된 표본 수로 나눈 값이다. 배치가 1보다 크면 이것은
호출당 J가 아닌 **완료 표본당 J**다. `inference_mean_w`는 적분 에너지를
실측 시간으로 나눈 시간 가중 평균이다. 적분 범위는 추론 phase의 첫 표본부터
마지막 표본까지로, phase 경계와 이 표본 사이의 짧은 시간은 포함하지 않는다.

센서/프로세스의 소비 전력과 백그라운드 작업도 측정에 포함된다. 같은
`nvpmodel`, 클럭, 팬 상태에서 조건당 최소 3회 반복하고 기준선 변동을
함께 보고한다. 0.1초보다 짧은 단발 추론의 전력은 이 trace로 분리할 수
없으므로 연속 실행의 평균 전력과 완료 처리량으로 해석한다.
