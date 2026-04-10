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
python src/main.py --model resnet50
```

지원 모델 및 CLI 옵션은 [framework/README.md](framework/README.md)를 참조하세요.
