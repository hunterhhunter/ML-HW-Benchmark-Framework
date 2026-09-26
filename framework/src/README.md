# Benchmark Framework

이 디렉토리는 다양한 AI 모델(ResNet50, MobileNet 등)과 런타임 백엔드(ONNX, vLLM, IREE)의 추론 성능을 일관성 있게 측정하기 위한 벤치마크 프레임워크 소스 코드를 담고 있습니다. NPU 확장을 위해 runtime, compiler, monitor를 `target_id`로 묶는 plugin registry도 포함합니다.

## 주요 패키지 구성 요소
- **`main.py`**: 터미널 커맨드라인(CLI) 파라미터를 파싱하여 프레임워크를 구동하는 **통합 진입점(Entrypoint)**.
- **`core/`**: 벤치마크 전체 루프를 총괄하는 엔진(`BenchmarkRunner`)과 메타데이터/DTO 클래스(`Model_Spec`, `InferenceResult`).
- **`dataloader/`**: 다양한 타겟 벤치마크 데이터셋에서 입력 데이터를 전처리하고 공급하는 모듈.
- **`runtimes/`**: 하드웨어 가속기 및 추론 엔진 백엔드(ONNX, vLLM, IREE, mock NPU 등)를 통합된 규격(인터페이스)으로 제어하는 모듈.
- **`compilers/`**: target별 native artifact를 생성하고 cache metadata를 반환하는 compiler adapter 모듈.
- **`monitors/`**: 시스템, NVIDIA GPU, mock NPU 등 하드웨어 metric collector를 registry로 조립하는 모듈.
- **`evaluators/`**: 추론된 결괏값의 레이블을 분석하고 검증하여 성능 메트릭(Top-1, Top-5, Latency 등)을 채점하는 모듈.

## 🚀 CLI 실행 가이드 (Zero-Config)

본 벤치마크 프레임워크는 **Zero-Config Auto-Prepare** 아키텍처를 적용하여, 모델명만 지정하면 내장된 프로필 레지스트리가 평가용 모델 스펙, 데이터셋, 배치 경로 등을 자동 탐지하고 누락 시 알아서 다운로드 및 변환합니다.

### 🖼️ 1. 이미지 분류 (Image Classification)
```bash
# ResNet50 모델 평가 (ImageNet 1K 데이터셋 디렉토리 자동 스니핑)
uv run src/main.py --model resnet50 --target cpu
```

### 🎯 2. 객체 탐지 (Object Detection)
```bash
# YOLOv5m 성능 평가 (COCO128 데이터셋)
uv run src/main.py --model yolov5m --target cpu
```

### ⚙️ Hailo HEF 실행
```bash
# Jetson Orin Nano + Hailo-8 M.2에서 precompiled HEF를 sync inference로 실행
uv run src/main.py --model resnet50 --target hailo8 --hef /path/to/resnet50.hef --layout NHWC --monitor

# Hailo-10H에서 HailoRT 5.x용 HEF를 sync inference로 실행
uv run src/main.py --model resnet50 --target hailo10h --hef /path/to/resnet50_10h.hef --layout NHWC --monitor
```

### 📊 3. 자연어 분류 (NLP Classification)
```bash
# BERT Base 모델을 이용한 SST-2 감성 분석
uv run src/main.py --model bert-base-uncased --target cpu
```

### 🧠 4. 기계 독해 (Question Answering)
```bash
# SQuAD v1 정답 도출 성능 평가
uv run src/main.py --model bert-base-uncased-squad-v1 --target cpu
```

### 📈 5. 시계열 예측 (Time-Series Forecasting)
```bash
# PatchTST 모델을 활용한 ETTh1 시계열 벤치마크
uv run src/main.py --model patchtst-fm-r1 --target cpu
```

### 💬 6. 언어 생성 (LLM Generation)
Llama 등의 생성형 대형 언어 모델은 `onnxruntime` 대신 메모리 컨트롤이 뛰어난 전용 가속기인 `vLLM` 백엔드를 필수로 지정해야 합니다.
*(주의: Llama 계열은 접근이 제한된 Gated 모델이므로 사전에 터미널에서 `uv run huggingface-cli login` 인증을 완료해야 다운로드가 승인됩니다.)*
```bash
# Llama 3.2 3B 모델 가동 (SQuAD v2 데이터셋 기반)
uv run src/main.py --model llama-3.2-3b --target vllm-cuda

# Llama 3.1 8B 모델 가동
uv run src/main.py --model llama-3.1-8b --target vllm-cuda
```

> **💡 부가 설정 팁**:
> * 전체 데이터셋 평가 시간이 너무 오래 걸릴 때는 `--max-steps 1` 과 같은 인자를 맨 뒤에 붙여 1사이클만 돌려볼 수 있습니다.
> * 하드웨어 장치를 변경하거나 커스텀 모델을 테스트하고 싶다면, `--target cuda`, `--device cuda` 또는 `--onnx custom.onnx` 처럼 원하는 인자만 수동으로 타이핑하세요. `--target`이 지정되면 내부 runtime/device 선택보다 우선합니다.
> * `vendor_mock_npu` target은 실제 SDK 없이 NPU plugin, compile cache, monitor wiring을 확인하기 위한 개발용 target입니다.
> * `hailo8`/`hailo10h` target은 HailoRT Python package(`hailo_platform`)와 해당 장치용 `.hef` 파일이 있는 환경에서만 실제 추론을 실행합니다.

## 원시 전력 추적

전력 표본을 보존하려면 실행 명령에 `--power-trace`를 명시합니다. 이 옵션은
`--monitor`와 독립적입니다. `--monitor`는 기존 하드웨어 요약 지표를 만들고,
`--power-trace`는 워밍업이 끝난 뒤 3초 기준 구간(`baseline`)과 실제 측정
구간(`inference`)의 장치 전력 W를 별도 CSV로 저장합니다.

```bash
python framework/src/main.py \
  --model resnet50 \
  --target rbln-static \
  --artifact /path/to/model.rbln \
  --dataset /path/to/imagenet \
  --warmup 2 \
  --max-steps 10 \
  --power-trace \
  --results-path framework/results/rbln-resnet50.csv
```

원시 파일은 결과 CSV의 디렉터리를 기준으로
`power/<run_id>.power.csv`에 생성됩니다. 결과 행에는 원시 파일을 연결하는
`power_trace_status`, `power_trace_path`, `power_trace_sha256`,
`power_trace_sample_count`, `power_monitor_source`, `power_scope`만 기록합니다.
기본 수집 주기는 0.2초이며 기준 구간 시작, 추론 시작, 추론 종료에서는 경계
표본을 추가로 남깁니다. 같은 값이 반복돼도 제거하지 않습니다.

원시 CSV 스키마는 다음과 같습니다.

```text
schema_version,run_id,sample_index,phase,target_id,collector,monitor_source,device_id,power_scope,scheduled_elapsed_ms,observed_elapsed_ms,query_latency_ms,power_w,sample_status,error_code
```

현재 지원하는 출처와 측정 범위는 Furiosa `furiosa-smi-py`/`device`,
Rebellions `rbln-smi-json`/`whole_card`, Mobilint `mbltml`/`device_total`입니다.
측정 범위가 서로 다르므로 원시 수치를 곧바로 하드웨어 간 효율 순위로
해석하면 안 됩니다. 사용할 수 없는 전력은 0이 아니라 `unavailable`로 남깁니다.
프레임워크는 J, idle 차감, `J/inference`를 계산하지 않습니다. 에너지 분석은
보존된 원시 CSV와 결과 행의 run ID·SHA-256을 사용해 별도 단계에서 수행합니다.
