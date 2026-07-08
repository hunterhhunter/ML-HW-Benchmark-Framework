# 벤치마크 결과 저장 설계 문서

## 개요

벤치마크 실행 결과를 **CSV 파일** 형식으로 저장하고, 웹 API를 통해 조회·관리한다.

## 저장 방식: CSV

별도 데이터베이스(SQLite 등)를 사용하지 않고, 단일 CSV 파일에 모든 벤치마크 결과를 누적 저장한다.

이번 MVP에서는 CSV 하위 호환을 유지하면서 핵심 target metadata 컬럼을 먼저 확장한다. 실행별 상세 compile log, vendor raw metadata, 환경 snapshot을 담는 JSON sidecar는 다음 단계로 남겨둔다.

### CSV를 선택한 이유

| 항목 | CSV | SQLite |
|------|-----|--------|
| 의존성 | 없음 (Python 표준 라이브러리) | sqlite3 모듈 필요 |
| 가독성 | 텍스트 에디터, Excel 등으로 직접 열람 가능 | 전용 도구 필요 |
| 버전 관리 | git diff로 변경 내역 추적 가능 | 바이너리 파일이라 diff 불가 |
| 데이터 규모 | 벤치마크 결과는 수백~수천 건 수준으로 CSV로 충분 | 대규모 데이터에 유리 |
| 이식성 | 어떤 환경에서든 바로 사용 가능 | 파일 포맷 호환성 문제 가능 |

벤치마크 결과는 실행 빈도가 낮고(일 수회~수십회), 데이터 크기가 작으며, 직접 열람·공유가 빈번하므로 CSV가 적합하다.

## 파일 위치

```
framework/results/benchmark_results.csv
```

`.gitignore`에 추가하지 않으므로, 결과를 git으로 추적할 수 있다.

## CSV 스키마

### 공통 메타데이터 컬럼 (모든 행에 존재)

| 컬럼명 | 타입 | 설명 | 예시 |
|--------|------|------|------|
| `run_id` | string | 실행 고유 ID (UUID 앞 8자리) | `a1b2c3d4` |
| `timestamp` | string | 실행 시각 (YYYY-MM-DD HH:MM:SS) | `2026-04-10 14:30:00` |
| `model_name` | string | 모델 이름 | `resnet50` |
| `task` | string | 태스크 타입 (Task enum 이름) | `IMAGE_CLASSIFICATION` |
| `backend` | string | 추론 백엔드 | `onnxruntime` |
| `device` | string | 추론 장치 | `cuda` |
| `batch_size` | int | 배치 크기 | `1` |
| `warmup_runs` | int | 웜업 횟수 | `2` |
| `max_steps` | int/빈값 | 최대 스텝 수 (없으면 빈 값) | `100` |
| `target_id` | string | 실행 타겟 ID | `cuda` |
| `accelerator_vendor` | string | 가속기 벤더 | `NVIDIA` |
| `accelerator_name` | string | 가속기 이름 | `CUDA GPU` |
| `runtime_name` | string | registry runtime 이름 | `onnxruntime` |
| `compiler_name` | string | registry compiler 이름 | `mock_npu` |
| `artifact_format` | string | 실행 아티팩트 포맷 | `onnx` |

### 태스크별 메트릭 컬럼 (동적 확장)

태스크마다 생성되는 메트릭이 다르다. 해당 태스크에 없는 메트릭 컬럼은 빈 값으로 남는다.

#### IMAGE_CLASSIFICATION
| 컬럼명 | 설명 |
|--------|------|
| `Total Samples` | 평가 샘플 수 |
| `Top-1 Accuracy` | Top-1 정확도 (%) |
| `Top-5 Accuracy` | Top-5 정확도 (%) |
| `Precision (macro)` | 매크로 정밀도 |
| `Recall (macro)` | 매크로 재현율 |
| `F1-Score (macro)` | 매크로 F1 |
| `Average Latency (ms)` | 평균 추론 지연 |
| `P99 Latency (ms)` | P99 추론 지연 |

#### NLP_CLASSIFICATION (BERT)
| 컬럼명 | 설명 |
|--------|------|
| `accuracy` | 정확도 (%) |
| `total_samples` | 평가 샘플 수 |
| `Average Latency (ms)` | 평균 추론 지연 |
| `P99 Latency (ms)` | P99 추론 지연 |

#### NLP_GENERATION (LLaMA)
| 컬럼명 | 설명 |
|--------|------|
| `Total Samples` | 평가 샘플 수 |
| `Exact Match` | 정확 일치율 (%) |
| `F1` | F1 점수 |
| `TTFT (ms)` | Time To First Token |
| `TPOT (ms)` | Time Per Output Token |
| `Throughput (tokens/sec)` | 토큰 처리량 |
| `Average Latency (ms)` | 평균 추론 지연 |
| `P99 Latency (ms)` | P99 추론 지연 |

#### TIME_SERIES_FORECASTING (PatchTST)
| 컬럼명 | 설명 |
|--------|------|
| `MAE` | Mean Absolute Error |
| `MSE` | Mean Squared Error |
| `RMSE` | Root Mean Squared Error |
| `Average Latency (ms)` | 평균 추론 지연 |
| `P99 Latency (ms)` | P99 추론 지연 |
| `Total Samples` | 평가 윈도우 수 |

### CSV 예시

```csv
run_id,timestamp,model_name,task,backend,device,batch_size,warmup_runs,max_steps,target_id,accelerator_vendor,accelerator_name,runtime_name,compiler_name,artifact_format,Total Samples,Top-1 Accuracy,Top-5 Accuracy,Average Latency (ms),P99 Latency (ms),accuracy,MAE,MSE,RMSE
a1b2c3d4,2026-04-10 14:30:00,resnet50,IMAGE_CLASSIFICATION,onnxruntime,cuda,1,2,,cuda,NVIDIA,CUDA GPU,onnxruntime,,onnx,5000,75.42,92.18,12.34,15.67,,,,
e5f6g7h8,2026-04-10 15:00:00,bert-base-uncased,NLP_CLASSIFICATION,onnxruntime,cpu,1,2,,cpu,Generic,CPU,onnxruntime,,onnx,1000,,,8.45,10.23,92.5,,,
```

## 아키텍처

```
┌──────────────┐    save_result()    ┌──────────────────────┐
│  Framework   │ ──────────────────> │  CSV 파일            │
│  main.py     │                     │  results/            │
│  (벤치마크)   │                     │  benchmark_results.csv│
└──────────────┘                     └──────────┬───────────┘
                                                │
                                     load_results()
                                     get_result()
                                     delete_result()
                                                │
┌──────────────┐    REST API         ┌──────────┴───────────┐
│  Frontend    │ <────────────────── │  Backend (FastAPI)    │
│  (React)     │                     │  /api/results         │
└──────────────┘                     └──────────────────────┘
```

### 모듈 구성

| 모듈 | 경로 | 역할 |
|------|------|------|
| `ResultStore` | `framework/src/core/result_store.py` | CSV 저장/조회/삭제 핵심 로직 |
| `result_service` | `backend/app/services/result_service.py` | ResultStore를 호출하는 서비스 레이어 |
| `results API` | `backend/app/api/results.py` | REST API 엔드포인트 |
| `result schemas` | `backend/app/schemas/result.py` | Pydantic 응답 모델 |
| `target registry` | `framework/src/core/targets.py` | runtime/compiler/monitor target 조합 |

## API 엔드포인트

| Method | Path | 설명 | 쿼리 파라미터 |
|--------|------|------|--------------|
| GET | `/api/results` | 결과 목록 조회 | `model_name`, `task`, `backend`, `limit` |
| GET | `/api/results/{run_id}` | 단일 결과 조회 | - |
| DELETE | `/api/results/{run_id}` | 결과 삭제 | - |

### 관련 Benchmark API

| Method | Path | 설명 |
|--------|------|------|
| GET | `/api/benchmark/targets` | 사용 가능한 `target_id`와 capability 목록 조회 |
| POST | `/api/benchmark/run` | `target_id`, `compile`, `compile_options`를 포함한 벤치마크 실행 |

### 응답 예시

**GET /api/results?model_name=resnet50&limit=5**
```json
{
  "total": 2,
  "results": [
    {
      "run_id": "a1b2c3d4",
      "timestamp": "2026-04-10 14:30:00",
      "model_name": "resnet50",
      "task": "IMAGE_CLASSIFICATION",
      "backend": "onnxruntime",
      "device": "cuda",
      "target_id": "cuda",
      "accelerator_vendor": "NVIDIA",
      "accelerator_name": "CUDA GPU",
      "runtime_name": "onnxruntime",
      "compiler_name": "",
      "artifact_format": "onnx",
      "batch_size": "1",
      "warmup_runs": "2",
      "max_steps": null,
      "metrics": {
        "Total Samples": 5000.0,
        "Top-1 Accuracy": 75.42,
        "Top-5 Accuracy": 92.18,
        "Average Latency (ms)": 12.34,
        "P99 Latency (ms)": 15.67
      }
    }
  ]
}
```

## 동작 흐름

1. 사용자가 `python src/main.py --model resnet50` 실행
2. `BenchmarkRunner.run()`이 메트릭 딕셔너리 반환
3. `main.py`가 `save_result()`를 호출하여 CSV에 한 행 추가
4. 웹 UI에서 `GET /api/results`로 결과 목록 조회
5. 특정 결과 상세 조회: `GET /api/results/{run_id}`
6. 불필요한 결과 삭제: `DELETE /api/results/{run_id}`

## 새 컬럼 자동 확장

새로운 태스크나 메트릭이 추가되면, `save_result()`가 기존 CSV 헤더에 없는 컬럼을 감지하고 자동으로 헤더를 확장한다. 기존 행의 해당 컬럼은 빈 값으로 유지된다.

## Target Registry 확장

NPU/가속기 확장을 위해 실행 단위는 `target_id`를 중심으로 기록한다. target은 runtime, compiler, monitor, artifact format을 묶는 registry 항목이며, 특정 벤더 SDK가 없어도 프레임워크 전체 import가 깨지지 않도록 lazy loading 방식으로 연결한다.

1차 구현은 `cpu`, `cuda`, `vllm-cuda`, `vllm-cpu`, `vendor_mock_npu` target을 제공한다. 실제 벤더 NPU는 `Runtime`, `Compiler`, `Collector` 구현체를 추가하고 target registry에 조합을 등록하는 방식으로 확장한다.

### 하드웨어 가속기 Metric Prefix

NPU 계열 collector는 벤더별 raw key가 달라도 CSV/API에서 비교하기 쉽도록 공통 key에 `hw_accel_*` prefix를 사용한다.

| Key 예시 | 설명 |
|----------|------|
| `hw_accel_util` | 순간 가속기 utilization |
| `hw_accel_mem_used_mb` | 가속기 메모리 사용량 |
| `hw_accel_power_w` | 전력 사용량 |
| `hw_accel_temp_c` | 온도 |
| `hw_accel_voltage_mv` | 전압 |
| `hw_accel_clock_mhz` | 클럭 |

`HWMonitor.summary()`는 수집된 key를 기반으로 `hw_accel_util_avg`, `hw_accel_mem_used_mb_max` 같은 요약 metric을 생성할 수 있다. CSV는 새 metric 컬럼을 자동 확장하므로 벤더별 monitor 추가 시 별도 migration 없이 저장된다.

### NVIDIA GPU Process-level Metric

NVIDIA collector는 NVML process API가 제공되는 환경에서 현재 benchmark 프로세스와 자식 프로세스(vLLM worker 등)의 VRAM 사용량을 별도로 기록한다.

| Key | 설명 |
|-----|------|
| `hw_gpu_mem_proc_peak_mb` | benchmark 프로세스 트리의 peak VRAM 사용량 |
| `hw_gpu_mem_proc_peak_pct` | benchmark 프로세스 트리 VRAM / GPU 총 VRAM |
| `hw_gpu_mem_proc_of_used_peak_pct` | benchmark 프로세스 트리 VRAM / 현재 GPU 전체 사용 VRAM |
| `hw_gpu_proc_count_max` | 수집된 benchmark GPU process 수의 최대값 |

`hw_gpu_util`, `hw_gpu_mem_peak_mb`, 온도, 전력은 GPU 전체 지표이고, `hw_gpu_*_proc*` 계열은 process-level 지표다. 드라이버, 권한, MIG, 컨테이너 PID namespace 상태에 따라 process-level 값은 비어 있을 수 있으므로, 결과 신뢰도 판단에는 두 계열을 함께 본다.
