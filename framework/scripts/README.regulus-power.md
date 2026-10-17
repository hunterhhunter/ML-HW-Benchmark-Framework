# REGULUS 입력 전력: Jetson INA3221 원격 측정

`serve_regulus_power.py`는 Jetson의 INA3221 채널 1을 읽고, `run_regulus_power.py`는
REGULUS에서 기존 프레임워크의 E2E/async 실행을 차례로 호출한다. 두 스크립트는
같은 공유기의 TCP 8765 포트로 `READY → GO → DONE → RESULT`를 주고받는다.
Jetson의 monotonic 시계로 idle·추론 경계를 기록하므로 두 보드의 시계를
맞출 필요가 없다. REGULUS 프레임워크의 `src` 파일은 수정하지 않는다.

## 배선과 사전 확인

- REGULUS 12V 어댑터 **플러스**를 채널 1 션트에 직렬로 연결한다:
  어댑터 쪽 `VIN1+`, REGULUS 쪽 `VIN1−`.
- 어댑터 **마이너스**는 REGULUS로 직접 연결한다. INA3221의 `GND`와도
  공통 접지로 이어져야 한다. Jetson의 I²C GND는 INA3221 GND에 연결한다.
  별도 어댑터를 사용하는 REGULUS에서는 이전 Jetson 측정 배선을 그대로
  옮겨도 이 공통 접지가 생기지 않을 수 있다.
- 전원을 끈 상태에서 극성, 접지 연속성, 노출 도체 절연을 확인한다. 전원을
  켠 뒤 채널 1 버스 전압이 12V 부근인지 확인한다. 기본 허용 범위는 10~14V다.
  범위를 벗어나면 서버가 측정을 시작하지 않는다.
- INA3221의 3A 한계 설정은 **로깅 중단 조건**이며 전원 차단 회로가 아니다.

## Jetson: 센서 서버

Jetson에서는 `regulus_power_link.py`, `serve_regulus_power.py`, 기존
`src/monitors/ina3221_worker.py`를 같은 디렉터리로 복사한다. 아래 예시는
`~/regulus-power`에 세 파일을 두고 가상 환경을 만든 경우다.

```bash
python3 -m venv "$HOME/regulus-power-venv"
"$HOME/regulus-power-venv/bin/pip" install smbus2
"$HOME/regulus-power-venv/bin/python" "$HOME/regulus-power/serve_regulus_power.py" \
  --bind 192.168.0.97 --allow-client 192.168.0.79 \
  --worker-path "$HOME/regulus-power/ina3221_worker.py" \
  --output-root "$HOME/regulus-power/results" --check-sensor
```

`--check-sensor`가 채널 1에서 12V 부근의 양수 전류·전력을 출력한 후 같은
명령에서 `--check-sensor`만 빼고 서버를 계속 실행한다. Jetson 주소는
현재 Wi‑Fi LAN 주소로 갱신한다. `--allow-client`는 REGULUS의 LAN 주소다.
서버는 센서 원본 CSV, 마커 JSONL, SHA-256이 든 manifest를 Jetson에도
저장한다.

## REGULUS: 6개 조건 오케스트레이션

REGULUS 프레임워크의 `scripts`에 `regulus_power_link.py`,
`run_regulus_power.py`, `regulus_power_suite.example.json`을 둔다.
예시 JSON은 ResNet50, YOLOv5m, YOLOv8s-pose 각각 E2E 5,000건과
async runtime-only 5,000건(worker 4, batch 1)을 담고 있다.
`framework_root`, artifact·dataset 경로, Jetson 주소, 새 `output_root`를
현재 보드에 맞춰 확인한다.

```bash
cd /root/ml-hw-benchmark-regulus-pr60-20261016/framework
python3 scripts/run_regulus_power.py \
  --config scripts/regulus_power_suite.example.json --dry-run
python3 scripts/run_regulus_power.py \
  --config scripts/regulus_power_suite.example.json
```

각 조건은 별도 프레임워크 프로세스로 실행된다. 워밍업 후 `start()`가
Jetson의 60초 idle 측정을 요청하고, `GO` 이후 추론 구간이 시작된다.
`stop()`에서 `DONE`을 보내고, 프레임워크 결과의 완료 건수가 요청 건수와
일치할 때만 전력 상태를 `complete`로 채택한다. 실패한 디렉터리를 자동으로
덮어쓰지 않으므로 재실행에는 새 `output_root`를 사용한다.

각 조건의 `results.csv`, `run.log`, `sensor.csv`, `events.jsonl`,
`power-summary.json`을 함께 보관한다. 마지막 파일에는 `framework_run_id`,
결과 CSV SHA-256, Jetson 원본 SHA-256, idle 중앙값 W, 추론 평균 W,
시스템/idle 차감 J per sample이 포함된다. 측정 범위는 **REGULUS 12V
어댑터 DC 입력 전체**다. idle 차감값도 NPU 코어만의 전력이 아니다.
async 전력 적분 구간에는 프레임워크의 입력 공급, 대기열, flush가 포함된다.
처리량은 별도 `results.csv`의 완료 건수와 측정 구간을 기준으로 읽는다.
