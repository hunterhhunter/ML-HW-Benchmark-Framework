# ML HW Benchmark Framework

ML 하드웨어 벤치마크 프레임워크와 실행·결과 조회용 웹 인터페이스를 제공하는 프로젝트입니다.

모델·target 선택부터 실행과 결과 확인까지는 [사용자 설명서](docs/user-guide/README.md)를 참고하세요.
구조·공용 인터페이스와 모델·NPU 확장은 [개발자 설명서](docs/developer-guide/README.md)를 참고하세요.

## 프로젝트 구조

```
ML-HW-Benchmark-Framework/
├── framework/      # ONNX, LLM, NPU 실행과 측정 엔진
├── backend/        # FastAPI 웹 API 서버
└── frontend/       # React + Vite + TypeScript 웹 클라이언트
```

- **framework/** — 모델·target을 선택하고 추론·평가·결과 저장을 수행하는 엔진. 일부 기본 모델·데이터는 프로필의 준비 스크립트로 생성할 수 있다. NPU용 SDK와 장치 artifact는 해당 벤더 절차에 따라 준비한다. 자세한 내용은 [framework/README.md](framework/README.md) 참조.
- **backend/** — FastAPI 기반 REST API. framework를 호출하여 웹에서 벤치마크를 실행하고 결과를 조회할 수 있습니다.
- **frontend/** — React + TypeScript 대시보드. 벤치마크 결과 시각화 및 실행 관리 UI.

## Target / NPU Plugin Registry

현재 프레임워크는 `--backend`/`--device` 직접 선택 방식과 함께 `target_id` 중심 실행을 지원합니다. `target_id`는 runtime, compiler, monitor, artifact format, device selector, capability를 하나로 묶는 실행 단위입니다.

- 일반·개발 target: `cpu`, `cuda`, `vllm-cpu`, `vllm-cuda`, `vendor_mock_npu`
- 벤더 NPU target: `hailo8`, `hailo10h`, `deepx`, `mobilint-aries`, `mobilint-regulus`, `mobilint-aries-llm`, `furiosa-rngd`, `furiosa-rngd-torch`, `rbln-static`, `rbln-vllm`
- `vendor_mock_npu`는 실제 벤더 SDK 없이 registry, compile cache, monitor wiring을 검증하기 위한 mock NPU plugin입니다.
- 실제 벤더 NPU는 core 실행 흐름 수정 없이 Runtime/Compiler/Monitor adapter를 추가하고 target registry에 조합을 등록하는 방식으로 확장합니다.

target 등록은 해당 host의 SDK·장치·artifact 준비나 모델 조합의 검증을 보증하지 않습니다.
벤더 adapter 확장은 [개발자 설명서](docs/developer-guide/04-npu-adapter-extension.md),
실행 조건은 [사용자 설명서](docs/user-guide/02-configuration.md)를 참조하세요.

벤더별 환경과 실행 절차는 다음 문서를 참조하세요.<br>
DEEPX DX-COM/DX-RT/driver 설치와 `deepx` target 실행 절차는 [docs/deepx-setup.md](docs/deepx-setup.md)를 참조하세요.<br>
Hailo-8/10H native async 규약과 ResNet50·YOLOv5m 실행 예시는 [docs/hailo-async-runtime.md](docs/hailo-async-runtime.md)를 참조하세요.<br>
Regulus의 qbruntime E2E 실행과 NPU-only binding 증적은 [docs/mobilint-regulus-runtime.md](docs/mobilint-regulus-runtime.md)를 참조하세요.<br>
Furiosa-LLM 전용 환경과 RNGD 실행 절차는 [docs/furiosa-rngd-setup.md](docs/furiosa-rngd-setup.md)를 참조하세요.<br>
Rebellions ATOM의 모델 artifact 생성은 [RBLN 컴파일 재현 가이드](framework/docs/rbln-compilation.md),<br> 정적 모델 실행은 [RBLN-CA22 운영 가이드](framework/docs/rbln-setup.md), Llama 생성은 [RBLN vLLM 실행 가이드](framework/docs/rbln-vllm-setup.md)를 참조하세요.<br>

## 시작하기

웹을 실행할 때도 먼저 아래 프레임워크 절차로 `framework/.venv`를 준비합니다.
각 명령 블록은 저장소 루트에서 시작합니다. 벤더 NPU는
[전용 환경 설치 절차](docs/user-guide/01-installation.md)를 따릅니다.

### 백엔드

```bash
cd backend
../framework/.venv/bin/python -m pip install -r requirements.txt
../framework/.venv/bin/python -m uvicorn app.main:app --reload
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
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/main.py --model resnet50 --target cpu --warmup 0 --max-steps 1

# 기존 호환 경로도 유지됩니다.
python src/main.py --model resnet50 --backend onnxruntime --device cpu --warmup 0 --max-steps 1
```

모델·데이터가 없으면 이 CPU 예시에서 준비 스크립트가 실행될 수 있습니다. NPU 실행에는
벤더별 Python·SDK 환경과 장치에 맞는 artifact가 필요합니다. 확인된 에지 조합의 명령은
[모델별 실행 명세](framework/docs/model-execution-specs/resnet50.md), 다른 target의
절차와 결과 확인 방법은 [사용자 설명서](docs/user-guide/README.md)를 따르세요.
지원 모델과 주요 CLI 옵션은 [framework/README.md](framework/README.md)에 있습니다.

### Target API 확인

백엔드 서버 실행 후 등록된 target 목록을 확인할 수 있습니다. 목록은 현재 host의
장치·SDK 준비 상태를 검사한 결과가 아닙니다.

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
