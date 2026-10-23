# 엣지 NPU 입력 전력 최종 결과

`power-results.csv`는 2026-10-22 선정 측정 24회(4개 NPU × 3개 모델 × 동기·비동기)를 한 표로 모은 것이다. 조건당 5,000건을 한 번 실행했다. `power-validation.csv`에는 각 실행의 검증 성공·실패만 담았다. 수치를 재계산하지 않고 로컬 원본의 `selected-summary.csv` 행을 그대로 옮겼다.

## 측정 경계와 계산

- INA3221 채널 1로 약 0.1초 간격에 전력을 기록했다. Hailo-8·Hailo-10H·DeepX는 **Jetson DC 입력 전체**, REGULUS는 **REGULUS 12V DC 입력 전체**가 측정 범위다. 값은 NPU 칩 단독 전력이 아니다.
- Idle은 벤치마크 프로세스가 시작한 뒤 **모델·데이터 로드 전 60초**의 전력 중앙값이다. 로드와 웜업은 Idle과 추론 적분 구간 사이에 둔다.
- 동기 `e2e`는 프레임워크의 System E2E 구간이며 호스트 전처리·후처리를 포함한다. 비동기 `async`는 offline, runtime-only, worker 4개 조건이다. 비동기 성능 지연시간의 Runtime-call 반환 경계와 전력 적분의 완료 구간은 다르다.
- `inference_mean_w`는 추론 구간의 실제 샘플 시각을 반영한 평균이다. `incremental_mean_w`는 그 값에서 Idle 기준을 뺀 값이고, `incremental_j_per_sample`은 Idle 차감 전력의 시간 적분을 해당 실행에서 완료한 5,000건으로 나눈 값이다. `observed_peak_w`는 수집된 표본의 최대값이다.

Jetson 세 구성은 `MAXN_SUPER` mode 2와 CPU 1,728 MHz·GPU 1,020 MHz·EMC 3,199 MHz 고정 조건이다. REGULUS는 CPU 1,200 MHz·NPU core/NOC 1,000 MHz 조건이다. ResNet-50은 ImageNet, YOLOv5m·YOLOv8s-pose는 COCO 평가 입력을 사용했다. 세 Jetson 구성의 측정 코드 기준 커밋은 `94114db671544889c5fd7e4b80f33f1cb57d3d4d`이다. REGULUS는 독립 호스트에서 추론하고 Jetson에서 외부 센서를 읽는 TCP 방식이며 실행 환경과 스크립트 해시는 로컬 원본의 `setup/`에 있다.

Hailo-8·Hailo-10H·DeepX의 YOLOv8s-pose 동기 실행은 프레임워크 내부 trace가 partial이었다. 이 세 행은 같은 실행의 독립 INA3221 로그와 구간 마커를 채택한 결과다. REGULUS의 선정된 6회 전력 로그는 모두 complete다. 각 실행의 채택 근거는 로컬 원본의 `selection.json`과 `validation.csv`에 남아 있다.

## 원본과 재현

로컬 원본은 저장소 최상위 `result/final-result/edge-power/<npu>/2026-10-22-preload-idle-w4/`에 보관한다. 실패한 예비 실행을 포함한 자료는 로컬 `results/`와 `result/archive/`에 보관한다. 이 디렉터리들은 Git에서 제외하므로 GitHub의 CSV만으로 원시 전력 적분을 독립 재계산할 수는 없다. `power-source-files.sha256`은 네 선정 원본의 요약·선정·검증 파일 해시를 고정한다. 원본이 있는 작업 공간에서 저장소 최상위에서 확인한다.

```bash
sha256sum -c result/final-result/edge-NPU/power-source-files.sha256
```

각 원본의 `manifest.sha256`, `setup/`, `sensor.csv` 또는 프레임워크 `power/` trace, 구간 마커, 결과 CSV를 사용하면 실행 인자와 채택 전력 구간을 다시 확인할 수 있다. 측정 절차와 실행 명령 예시는 [엣지 전력 측정 문서](../../../docs/edge-power-measurement.md) 및 [REGULUS 전력 측정 문서](../../../framework/scripts/README.regulus-power.md)에 있다.

기존 `result/final-result/edge-benchmark-results.csv`는 별도로 구성된 웹용 성능·품질 결과다. 이 문서의 2026-10-22 preload Idle 전력 수치와 측정 회차가 다르므로 두 CSV의 행을 같은 실행으로 연결하지 않는다.
