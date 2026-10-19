# 웹 벤치마크 콘솔

React, Vite, TypeScript로 만든 단일 호스트용 화면이다. FastAPI가 설치된 NPU 호스트에서
프레임워크 CLI를 별도 프로세스로 실행하고 결과 CSV를 읽는다. 브라우저에서 입력하는
모델·데이터·결과 경로는 **백엔드 호스트의 경로**다.

## 실행

저장된 결과 조회만 할 때도 백엔드가 필요하다. NPU SDK가 필요한 실행은
해당 SDK가 설치된 Python 환경을 backend runner profile에 등록한다.

```bash
cd /path/to/ML-HW-Benchmark-Framework/frontend
npm ci
npm run dev
```

기본 주소는 `http://localhost:5173`이며 Vite가 `/api`를 로컬
`http://localhost:8000`으로 전달한다. API 주소가 다르면
`VITE_API_BASE=http://localhost:8000/api npm run dev`처럼 지정한다.
정적 배포는 `npm run build` 후 백엔드가 `frontend/dist`를 제공한다.

## 화면 사용

1. **실행**에서 모델과 target을 선택한다. 자산·전처리·비동기 옵션은 필요한
   값만 입력한다. 빈 값은 CLI 기본값을 사용한다. `layout`을 비워 두면 CLI의
   target별 기본 처리가 유지된다.
2. **설정 검토**에서 Python 환경, 최종 argv, 결과 CSV, 자산 경로의 정적 검사를
   확인한다. 이 검사는 SDK·장치 동작이나 모델 품질을 보증하지 않는다.
3. **실행 시작** 후 **작업**에서 queued/running/종료 상태, 로그, 취소,
   `RUN_ID_RESERVED`와 `RUN_ID`, CSV 존재 여부를 확인한다. 화면을 새로고침해도
   서버에 저장된 작업을 다시 조회할 수 있다.
4. **결과**에서 source를 골라 CSV 행을 읽는다. 별도 CLI 결과 CSV는 서버에서
   접근 가능한 허용 결과 root 안의 파일을 등록한다. 상세 화면에서 원본
   metadata·metric과 사용 가능한 details·trace를 연다. 삭제는 CSV 행만
   제거하고 부속 파일은 보존한다.
5. **비교**에서는 결과 행과 원본 측정 항목을 직접 선택한다. 여러 항목을
   고르면 key별 가로 막대 그래프가 나란히 나타난다. 색상 강조는 같은 key의
   수치상 최저·최고만 표시하며 비교 가능 여부를 판단하지 않는다. 실행 조건
   표를 보고 해석한다. 없는 값은 `기록 없음` 또는 `—`로 남긴다.

`e2e`의 Runtime 지연, 비동기 완료 처리량, 품질 상태, 전력 범위는 서로 다른
계약이다. 수치의 의미는 `framework/docs/measurement-contract.md`를 따른다.

## 검증

```bash
npm test
npm run lint
npm run build
```

API 환경 변수와 runner profile 형식은
[`웹 구현 계획`](../docs/plans/2026-10-18-web-cli-parity.md)과
[`runner profile 예시`](../backend/config/runner_profiles.example.json)에 설명한다.
