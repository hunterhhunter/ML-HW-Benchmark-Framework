# ML HW Benchmark Framework

ML 하드웨어 벤치마크 프레임워크에 웹 UI를 올린 프로젝트입니다.

## Project Structure

```
ML-HW-Benchmark-Framework/
├── framework/          # 기존 ML 벤치마크 프레임워크 (ONNX/vLLM)
├── backend/            # FastAPI 웹 API
│   └── app/
│       ├── main.py     # FastAPI 엔트리포인트
│       ├── api/        # API 라우터
│       ├── schemas/    # Pydantic 모델
│       └── services/   # 비즈니스 로직 (framework 연동)
├── frontend/           # React + Vite + TypeScript
│   └── src/
│       └── App.tsx     # 메인 앱 컴포넌트
├── docs/               # 설계 문서
│   └── result-storage-design.md  # 결과 저장 설계 (CSV 기반)
└── CLAUDE.md
```

## Commands

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload        # http://localhost:8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev                          # http://localhost:5173
```

### Framework (기존 벤치마크)
```bash
cd framework
source .venv/bin/activate
python src/main.py --model resnet50
# 결과는 framework/results/benchmark_results.csv에 자동 저장됨
```

## 결과 저장

벤치마크 결과는 CSV 파일로 저장된다. 상세 설계는 [docs/result-storage-design.md](docs/result-storage-design.md) 참조.

- **저장 위치**: `framework/results/benchmark_results.csv`
- **저장 모듈**: `framework/src/core/result_store.py`
- **API 조회**: `GET /api/results`, `GET /api/results/{run_id}`, `DELETE /api/results/{run_id}`

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
- Save progress, checkpoint, resume → invoke checkpoint
- Code quality, health check → invoke health
