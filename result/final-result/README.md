# 웹용 엣지 벤치마크 결과

`edge-benchmark-results.csv`는 모델 3종 × NPU 4종 × 실행 모드 2종의 24행이다. Hailo-8, Hailo-10H, DeepX의 18행은 `edge-power/selected-results.csv`에서 가져오고, REGULUS의 6행은 다음 보관 원본에서 직접 가져온다. `source_csv`와 `source_detail_json`은 보관 디렉터리(`result/final-result/`) 기준 상대경로이며 각 SHA-256 열은 그 파일의 해시다.

| REGULUS 실행 | 원본 (`edge-report-5.5/evidence/results/` 기준) | 웹에 표시하는 값 |
| --- | --- | --- |
| 동기 E2E | `results-20260927-regulus-5000-runtime-call/vendor-native-e2e/vendor-reference-single-stream-v1/mobilint-regulus/<model>/<model>-seed2026081301.csv` | `System E2E Average/P99 Latency`, `System E2E Throughput`, 동일 실행의 품질, 측정 표본 5,000개 |
| 비동기 | `2026-10-01-regulus-vg-e2e-async-runtime-v2-preloaded-5000/<model>/results.csv`와 `details/<run_id>.json` | `service_time` 평균/P99, Runtime 완료 처리량, 완료 표본 5,000개 |

REGULUS 동기 E2E 처리량은 평균 System E2E 지연시간의 역수다. 비동기 실행에는 품질 평가가 없으므로 표시된 품질은 `quality_source_csv`와 `quality_run_id`가 가리키는 **별도 동기 실행**의 값이다. REGULUS 전력 자료는 없다. YOLOv5m 품질은 원본의 `COCO AP50`이며 다른 장치의 `mAP@0.5`로 바꾸어 부르지 않는다. 긴 REGULUS 동기 Run ID는 CSV에 원형대로 보관하고 웹 표에는 끝의 식별자만 표시한다.

원본 보관 디렉터리가 있는 환경에서 다시 생성한다.

```bash
PYTHONPATH=backend python3 -m app.services.edge_results_consolidate \
  /path/to/archive/result/final-result \
  result/final-result/edge-benchmark-results.csv
cp result/final-result/edge-benchmark-results.csv framework/results/benchmark_results.csv
```

웹 기본 결과 CSV는 `framework/results/benchmark_results.csv`다. 원본 보관본의 `edge-report-5.5/SHA256SUMS`는 해당 디렉터리에서 `sha256sum -c SHA256SUMS --quiet`로 검증한다.
