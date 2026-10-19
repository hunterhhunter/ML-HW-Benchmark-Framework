# 웹용 엣지 벤치마크 결과

`edge-benchmark-results.csv`는 모델 3종 × NPU 4종 × 실행 모드 2종의 24행이다. Hailo-8, Hailo-10H, DeepX의 18행은 `edge-power/selected-results.csv`에서 가져오고, REGULUS의 6행은 다음 보관 원본에서 직접 가져온다. `source_csv`와 `source_detail_json`은 보관 디렉터리(`result/final-result/`) 기준 상대경로이며 각 SHA-256 열은 그 파일의 해시다.

| REGULUS 실행 | 원본 (`edge-report-5.5/evidence/results/` 기준) | 웹에 표시하는 값 |
| --- | --- | --- |
| 동기 E2E | `results-20260927-regulus-5000-runtime-call/vendor-native-e2e/vendor-reference-single-stream-v1/mobilint-regulus/<model>/<model>-seed2026081301.csv` | Runtime 호출 `Average/P99 Latency`, `System E2E Throughput`, 동일 실행의 품질, 측정 표본 5,000개 |
| 비동기 | `2026-10-01-regulus-vg-e2e-async-runtime-v2-preloaded-5000/<model>/results.csv`와 `details/<run_id>.json` | `service_time` 평균/P99, Runtime 완료 처리량, 완료 표본 5,000개 |

모든 행의 지연시간은 동기·비동기 모두 Runtime 호출 평균·P99다. REGULUS 동기 E2E 처리량은 별도로 측정된 평균 System E2E 지연시간의 역수이므로, 표시된 Runtime 호출 평균 지연시간의 역수와 다르다. 비동기 실행에는 품질 평가가 없으므로 표시된 품질은 `quality_source_csv`와 `quality_run_id`가 가리키는 **별도 동기 실행**의 값이다. REGULUS 전력 자료는 없다. 긴 REGULUS 동기 Run ID는 CSV에 원형대로 보관하고 웹 표에는 끝의 식별자만 표시한다.

웹 통합 CSV는 REGULUS 품질에도 다른 NPU와 같은 열 이름을 쓴다. ResNet50은 `Top-1 Accuracy`, YOLOv8s-pose는 `OKS mAP`, YOLOv5m은 `mAP@0.5`다. REGULUS YOLOv5m 값 `0.5379820373275688`은 동기·비동기 행 모두 `mAP@0.5` 열에 기록한다. 원본 자료의 위치와 해시는 `source_csv`와 `source_csv_sha256`에 남기며, 비동기 품질의 원본 실행은 `quality_source_csv`와 `quality_run_id`로 추적한다.

원본 보관 디렉터리가 있는 환경에서 다시 생성한다.

```bash
PYTHONPATH=backend python3 -m app.services.edge_results_consolidate \
  /path/to/archive/result/final-result \
  result/final-result/edge-benchmark-results.csv
cp result/final-result/edge-benchmark-results.csv framework/results/benchmark_results.csv
```

웹 기본 결과 CSV는 `framework/results/benchmark_results.csv`다. 원본 보관본의 `edge-report-5.5/SHA256SUMS`는 해당 디렉터리에서 `sha256sum -c SHA256SUMS --quiet`로 검증한다.
