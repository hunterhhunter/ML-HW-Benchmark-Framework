# Frontend

React·Vite 화면에서 백엔드의 벤치마크 작업을 시작하고 기본 결과 CSV의 행을 조회한다.
백엔드 실행과 Python 환경 설정은 [사용자 실행 설명서](../docs/user-guide/03-execution.md)를
따른다.

## 개발 서버

```bash
cd /path/to/ML-HW-Benchmark-Framework/frontend
npm install
npm run dev
```

기본 주소는 `http://localhost:5173`이고 API 기본 경로는 `/api`다. 다른 백엔드
주소를 사용할 때는 `VITE_API_BASE=http://localhost:8000/api npm run dev`처럼
지정한다.

## 실행 화면의 범위

화면은 `GET /api/benchmark/targets`와 모델 profile API에서 선택지를 가져온다.
target 목록은 [registry](../framework/src/core/targets.py)의 등록 상태를 보여주며,
현재 host의 SDK·장치·artifact 준비 상태를 검사한 목록은 아니다. target을
선택하면 runtime·device가 설정되고 compiler가 연결된 target에는 compile
checkbox가 표시된다.

현재 실행 폼은 Hailo용 HEF 경로를 입력받지만 일반 NPU `artifact_path`와
비동기 `scenario`·worker·queue 옵션은 제공하지 않는다. 준비된 DXNN·MXQ·RBLN
artifact나 `async_queue`를 지정하려면 [CLI 실행 절차](../docs/user-guide/03-execution.md)를
사용한다. 작업 상태가 `completed`여도 결과의 유효성·품질 판정은 별도로
[CSV 결과 확인](../docs/user-guide/04-results.md)에 따라 확인한다.

결과 목록·비교 차트는 백엔드 결과 API가 반환한 행을 사용한다. 별도
`--results-path`의 CSV나 비동기 details·trace 파일은 현재 웹 결과 조회 범위에
포함되지 않는다. 코드 연결 위치와 확장 절차는
[웹·CLI 개발 설명서](../docs/developer-guide/11-web-cli-integration.md)에 있다.

## 검증

```bash
npm run lint
npm run build
```
