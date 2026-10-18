# 03. 벤치마크 실행

[목차](README.md) · [이전: NPU·모델·데이터셋 설정](02-configuration.md) · [다음: 결과 확인 및 해석](04-results.md)

## CLI: 소량 확인

선택한 장비에서 준비, Runtime load, 추론 호출이 끝나는지 먼저 확인한다.
다음은 CPU에서 ResNet50 한 step을 실행하는 예시다. 모델과 ImageNet 데이터가
없으면 자동 준비가 시작될 수 있다. 이 실행의 정확도·처리량은 최종 결과로
사용하지 않는다.

```bash
cd /path/to/ML-HW-Benchmark-Framework/framework
source .venv/bin/activate
python src/main.py --model resnet50 --target cpu \
  --inference-mode e2e --warmup 0 --max-steps 1 \
  --results-path results/user-guide-smoke.csv
```

## CLI: 본 실행

모델·NPU 조합은 [설정 문서](02-configuration.md)에 연결된 실행 명세의
모델·데이터·artifact 설정과 표본 수, warmup 조건을 따른다. 다음 명령은 준비된
CPU 프로필의 전체 데이터를 실행하는 기본 예시다.

```bash
python src/main.py --model resnet50 --target cpu \
  --inference-mode e2e --warmup 2 \
  --results-path results/resnet50-cpu-e2e.csv
```

동기 `e2e`의 `--max-steps`는 반복 step 상한이다. `async_queue`에서는
`--max-steps` 대신 `--max-samples`를 사용한다. 비동기 기본 패스는
`quality-and-runtime`이며, `runtime-only`는 정확도 평가를 건너뛴다.
worker 수는 target이 지원하는 범위에서 정한다. 소량 확인과 본 실행의
CSV 경로를 분리한다. 비동기 부하 옵션과 trace는
[비동기 추론 큐 가이드](../async-inference-queue.md)를 따른다.

## 웹 화면

[설치 절차](01-installation.md)를 마친 뒤, 두 터미널에서 각각 실행한다.

```bash
cd /path/to/ML-HW-Benchmark-Framework/backend
../framework/.venv/bin/python -m uvicorn app.main:app --reload
```

```bash
cd /path/to/ML-HW-Benchmark-Framework/frontend
VITE_API_BASE=http://localhost:8000/api npm run dev
```

`http://localhost:5173`에서 모델과 target을 선택하고 실행한다. 현재 폼은
batch, warmup, `max_steps`, HEF 경로, 컴파일·모니터 옵션을 보낸다.
`async_queue`, 데이터·일반 artifact 경로와 세부 런타임 옵션은 CLI를 사용한다.
웹 설정은 [프론트엔드 README](../../frontend/README.md)를 참고한다.

실행이 끝나면 [결과 확인 및 해석](04-results.md)으로 이동한다.
