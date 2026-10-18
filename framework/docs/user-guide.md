# 벤치마크 프레임워크 사용자 설명서

이 문서는 `main@30c4db75`의 실행 방법을 기준으로 모델·target 선택부터 결과 확인까지
안내한다. 모델별 재현 명령과 벤더 SDK 설치 명령은 아래에 연결한 문서를 따른다.
실행 모드와 지표의 의미는 [측정 계약](measurement-contract.md)을 먼저 확인한다.

## 1. 실행 경로 선택

| 경로 | 사용할 때 | 설정할 수 있는 내용 |
|---|---|---|
| CLI (`framework/src/main.py`) | 모델·데이터·artifact 경로를 지정하거나 `async_queue`를 실행할 때 | 모든 공개 CLI 옵션. 이 문서의 실행 절차는 CLI를 기준으로 한다. |
| 웹 화면 | 기본 모델·target을 고르고 실행 상태와 저장된 결과를 볼 때 | 현재 실행 폼은 모델, target, batch, warmup, `max_steps`, HEF 경로·컴파일·모니터 옵션을 보낸다. `async_queue`, 데이터·일반 artifact 경로, 세부 런타임 옵션은 폼에서 설정할 수 없다. |

`--target`은 Runtime과 장치 등을 묶은 실행 단위이며, 지정하면
`--backend`/`--device`보다 우선한다. target 목록과 지원 모델은
[프레임워크 README](../README.md)에 있다. 목록에 표시되는 것만으로 해당
모델·장비 조합의 실행이 확인된 것은 아니다.

## 2. 환경과 실행 자산 준비

CPU 기본 실행의 출발점은 다음과 같다. NPU·GPU는 해당 장비의 SDK, 드라이버,
Python 환경을 먼저 준비한다. 벤더별 환경을 공통 `requirements.txt` 설치만으로
대체할 수 없다.

```bash
cd /path/to/ML-HW-Benchmark-Framework/framework
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

1. [지원 모델과 target](../README.md)을 고른다. 해당 장비에서 Runtime을 찾을 수
   있는지 확인한다.
2. 모델, 데이터셋, 정답 파일을 준비한다. 기본 프로필은 자산이 없으면 준비
   스크립트를 실행할 수 있어 네트워크 접근이나 별도 인증이 필요할 수 있다.
   준비 스크립트는 [모델](../models/README.md)과
   [데이터셋](../datasets/README.md) 문서에 있다.
3. NPU에서 사전 컴파일 모델을 쓰는 경우 artifact 파일, 대상 장비, 입력 형식
   (shape·layout·dtype), 모델과 데이터의 관계를 해당 실행 명세와 대조한다.
   target에 따라 `--hef` 또는 `--artifact --no-compile`을 사용한다.
   `--no-compile` 지원 여부와 필요한 옵션은 target별 문서를 따른다.

에지 NPU의 확인된 조합과 명령은
[ResNet50](model-execution-specs/resnet50.md),
[YOLOv5m](model-execution-specs/yolov5m.md),
[YOLOv8s-pose](model-execution-specs/yolov8s-pose.md)에 있다.
서버 NPU는 [TTM-R2](ttm-r2-framework.md),
[Rebellions 정적 모델](rbln-setup.md),
[Rebellions LLM](rbln-vllm-setup.md),
[Furiosa](../../docs/furiosa-rngd-setup.md),
[Mobilint ARIES](../../docs/mobilint-aries-transformers.md) 절차를 따른다.

## 3. 소량 실행 후 전체 실행

먼저 같은 환경·artifact·데이터로 소량 실행해 준비 단계와 Runtime 호출이
끝나는지 확인한다. 다음 CPU 예시는 CLI 동작을 확인하는 출발점이다. ResNet50
프로필의 모델과 ImageNet 데이터가 없으면 자동 준비가 시작될 수 있다. 한 샘플
실행의 정확도·처리량을 최종 결과로 사용하지 않는다.

```bash
cd /path/to/ML-HW-Benchmark-Framework/framework
python src/main.py --model resnet50 --target cpu \
  --inference-mode e2e --warmup 0 --max-steps 1 \
  --results-path results/user-guide-smoke.csv
```

본 실행에는 선택한 모델·target 실행 문서의 **동일한** 모델·데이터·artifact
설정을 사용하고, 해당 문서의 표본 수와 warmup 조건을 적용한다. 동기 `e2e`의
반복 step 상한은 `--max-steps`다. CPU 프로필의 준비된 전체 데이터를 실행하는
기본 예시는 다음과 같다. `e2e`는 실행 모드 이름이며, 저장되는 Runtime
지연시간을 System E2E 지연시간으로 읽지 않는다.

```bash
python src/main.py --model resnet50 --target cpu \
  --inference-mode e2e --warmup 2 \
  --results-path results/resnet50-cpu-e2e.csv
```

`async_queue`에서는 `--max-steps` 대신 `--max-samples`를 쓴다. 기본 패스는
`quality-and-runtime`이며, `runtime-only`는 정확도 평가를 건너뛴다.
`--worker-count`는 target이 지원하는 범위에서 정하고, worker 수만으로 NPU
내부 병렬성을 판단하지 않는다. 실행 옵션과 trace 분석은
[비동기 추론 큐 가이드](../../docs/async-inference-queue.md)를 따른다.
소량 확인과 본 실행의 CSV 경로를 분리하면 결과를 찾기 쉽다.

## 4. 결과 확인

CLI가 종료되면 출력의 `RUN_ID=<id>`와 결과 파일 경로를 확인한다. 기본 CSV는
`framework/results/benchmark_results.csv`이며, `--results-path`로 바꿀 수 있다.
CSV 한 행이 한 실행이다. `RUN_ID_RESERVED=<id>`만 출력되고 `RUN_ID=<id>`가
없으면 완료된 결과로 간주하지 않고 오류 출력과 예약 상태를 확인한다.

다음은 위의 소량 실행이 남긴 마지막 CSV 행에서 주요 필드를 읽는 예시다.
다른 결과를 볼 때는 `path`를 해당 `--results-path` 값으로 바꾼다.

```bash
python3 - <<'PY'
import csv
from pathlib import Path

path = Path("results/user-guide-smoke.csv")
with path.open(newline="") as file:
    rows = list(csv.DictReader(file))
row = rows[-1]
for key in (
    "run_id", "model_name", "target_id", "inference_mode",
    "async_run_status", "quality_status", "comparison_eligible",
    "async_runtime_completed_samples_per_sec", "details_path",
):
    if row.get(key):
        print(f"{key}: {row[key]}")
PY
```

비동기 결과는 `async_run_status`, `quality_status`, `comparison_eligible`,
완료·실패 sample 수를 함께 본다. `quality_status=passed`는 평가기가 정상 집계한
상태이며 정확도 목표 통과를 뜻하지 않는다. 공식 완료 처리량은 schema v2의
`async_runtime_completed_samples_per_sec`다. `details_path`와 선택적
`request_trace_path`는 결과 CSV의 상위 디렉터리를 기준으로 열 수 있다.
지연시간·처리량·품질 상태의 해석은 [측정 계약](measurement-contract.md)을 따른다.

## 5. 웹 화면에서 실행할 때

백엔드는 프레임워크 의존성이 설치된 Python에서 실행한다. 아래 명령은 2절에서
만든 `framework/.venv`를 사용한다. 프론트엔드는 별도 터미널에서 실행한다.

```bash
cd /path/to/ML-HW-Benchmark-Framework/backend
../framework/.venv/bin/python -m pip install -r requirements.txt
../framework/.venv/bin/python -m uvicorn app.main:app --reload
```

```bash
cd /path/to/ML-HW-Benchmark-Framework/frontend
npm install
VITE_API_BASE=http://localhost:8000/api npm run dev
```

`http://localhost:5173`에서 모델과 target을 선택하고 실행한다. 화면의 작업 상태와
`run_id`를 확인한 뒤 결과 목록에서 저장된 행을 연다. 웹 결과 목록은 기본
`framework/results/benchmark_results.csv`를 읽는다. CLI에서 별도 CSV 경로를
지정했다면 그 파일은 CLI로 확인한다. 화면 설정과 API 주소는
[프론트엔드 README](../../frontend/README.md)를 참고한다.

## 6. 실행이 막힐 때

- 모델·데이터 준비에서 멈추면 준비 스크립트의 출력, 네트워크 접근, 필요한 인증,
  실제 파일 경로를 확인한다.
- NPU의 load·compile 단계에서 실패하면 해당 target의 SDK·장치 상태와 artifact
  형식, 입력 계약을 먼저 확인한다.
- 비동기 실행이 `invalid`이면 CSV의 `async_invalid_reasons`와
  `details_path`를 확인한 다음 [비동기 가이드](../../docs/async-inference-queue.md)의
  lifecycle 항목을 대조한다.
