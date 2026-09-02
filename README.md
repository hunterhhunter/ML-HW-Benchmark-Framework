# ML HW Benchmark Framework

ML 하드웨어 벤치마크 프레임워크에 웹 인터페이스를 올려 쉽게 사용할 수 있도록 만든 프로젝트입니다.

## 프로젝트 구조

```
ML-HW-Benchmark-Framework/
├── framework/      # 기존 ML 벤치마크 프레임워크 (ONNX/vLLM 추론 성능 측정)
├── backend/        # FastAPI 웹 API 서버
└── frontend/       # React + Vite + TypeScript 웹 클라이언트
```

- **framework/** — 모델 이름 하나로 다운로드부터 추론, 평가까지 자동 실행하는 벤치마크 엔진. 자세한 내용은 [framework/README.md](framework/README.md) 참조.
- **backend/** — FastAPI 기반 REST API. framework를 호출하여 웹에서 벤치마크를 실행하고 결과를 조회할 수 있습니다.
- **frontend/** — React + TypeScript 대시보드. 벤치마크 결과 시각화 및 실행 관리 UI.

## Target / NPU Plugin Registry

현재 프레임워크는 `--backend`/`--device` 직접 선택 방식과 함께 `target_id` 중심 실행을 지원합니다. `target_id`는 runtime, compiler, monitor, artifact format, device selector, capability를 하나로 묶는 실행 단위입니다.

- 기본 target: `cpu`, `cuda`, `vllm-cpu`, `vllm-cuda`, `furiosa-rngd`, `vendor_mock_npu`, `hailo8`, `hailo10h`, `deepx`, `mobilint-aries`, `mobilint-regulus`, `rbln-static`, `rbln-vllm`
- `vendor_mock_npu`는 실제 벤더 SDK 없이 registry, compile cache, monitor wiring을 검증하기 위한 mock NPU plugin입니다.
- 실제 벤더 NPU는 core 실행 흐름 수정 없이 Runtime/Compiler/Monitor adapter를 추가하고 target registry에 조합을 등록하는 방식으로 확장합니다.

세부 구조와 벤더 adapter 추가 절차는 [docs/npu-plugin-registry.md](docs/npu-plugin-registry.md)를 참조하세요.<br>
DEEPX DX-COM/DX-RT/driver 설치와 `deepx` target 실행 절차는 [docs/deepx-setup.md](docs/deepx-setup.md)를 참조하세요.<br>
Hailo-8/10H native async 규약과 ResNet50·YOLOv5m 실행 예시는 [docs/hailo-async-runtime.md](docs/hailo-async-runtime.md)를 참조하세요.<br>
Regulus의 qbruntime E2E 실행과 NPU-only binding 증적은 [docs/mobilint-regulus-runtime.md](docs/mobilint-regulus-runtime.md)를 참조하세요.<br>
Furiosa-LLM 전용 환경과 RNGD 실행 절차는 [docs/furiosa-rngd-setup.md](docs/furiosa-rngd-setup.md)를 참조하세요.<br>
Rebellions ATOM의 일곱 모델 artifact 생성은 [RBLN 컴파일 재현 가이드](framework/docs/rbln-compilation.md), 정적 모델 실행은 [RBLN-CA22 운영 가이드](framework/docs/rbln-setup.md), Llama 생성은 [RBLN vLLM 실행 가이드](framework/docs/rbln-vllm-setup.md)를 참조하세요.<br>

## 시작하기

### 백엔드

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
# http://localhost:8000 에서 API 서버 시작
```

### 프론트엔드

```bash
cd frontend
npm install
npm run dev
# http://localhost:5173 에서 개발 서버 시작
```

### 벤치마크 프레임워크 (단독 실행)

```bash
cd framework
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py --model resnet50 --target cpu

# 기존 호환 경로도 유지됩니다.
python src/main.py --model resnet50 --backend onnxruntime --device cpu

# SDK 없이 plugin/compile/monitor wiring을 확인하는 mock NPU target
python src/main.py --model resnet50 --target vendor_mock_npu --max-steps 1 --warmup 0 --monitor

# Hailo-8/8L HEF sync inference (Jetson + HailoRT 설치 환경)
python src/main.py --model resnet50 --target hailo8 --hef /path/to/resnet50.hef --layout NHWC --monitor

# Hailo-10H HEF sync inference (HailoRT 5.x 설치 환경)
python src/main.py --model resnet50 --target hailo10h --hef /path/to/resnet50_10h.hef --layout NHWC --monitor

# Hailo InferModel native async inference (worker 수는 SDK async queue 크기 이하)
python src/main.py --model resnet50 --target hailo8 --hef /path/to/resnet50.hef --inference-mode async_queue --worker-count 2 --max-samples 100 --min-samples 100 --monitor
python src/main.py --model yolov5m --target hailo10h --hef /path/to/yolov5m_10h.hef --inference-mode async_queue --worker-count 2 --max-samples 100 --min-samples 100 --monitor

# DEEPX DX-COM compile + DX-RT inference (DX-COM/DX-RT 설치 환경)
python src/main.py --model resnet50 --target deepx --compile-option config_path=/path/to/resnet50_config.json --monitor

# 사전 컴파일된 DEEPX DXNN 실행
python src/main.py --model resnet50 --target deepx --no-compile --artifact /path/to/resnet50.dxnn --monitor
```

지원 모델 및 CLI 옵션은 [framework/README.md](framework/README.md)를 참조하세요.

### Target API 확인

백엔드 서버 실행 후 사용 가능한 target 목록을 확인할 수 있습니다.

```bash
curl http://localhost:8000/api/benchmark/targets
```

## Docker 배포 (GPU 서버)

프론트엔드는 Docker build 단계에서 정적 파일로 빌드되고, 운영 시에는 FastAPI가 API와 React UI를 같은 포트에서 서빙합니다. 모델, 데이터셋, 결과 CSV는 이미지에 포함하지 않고 호스트 볼륨으로 유지합니다.

```bash
# Hugging Face gated 모델을 쓸 경우
export HF_TOKEN=hf_xxx

docker compose -f docker-compose.gpu.yml up -d --build
```

- Web UI: `http://<server-ip>:8000`
- API health check: `http://<server-ip>:8000/api/health`
- 볼륨: `framework/models`, `framework/datasets`, `framework/results`, `$HOME/.cache/huggingface`
