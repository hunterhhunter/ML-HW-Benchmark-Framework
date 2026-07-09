# ML HW Benchmark Frontend

React + Vite + TypeScript 기반 벤치마크 대시보드입니다. 백엔드 API를 통해 결과를 조회하고, `target_id` 중심으로 새 벤치마크 실행을 요청합니다.

## 실행

```bash
npm install
npm run dev
```

개발 서버 기본 주소는 `http://localhost:5173`입니다. API 주소는 기본적으로 `/api`를 사용하며, 필요하면 `.env`에서 `VITE_API_BASE`로 바꿀 수 있습니다.

```bash
VITE_API_BASE=http://localhost:8000/api npm run dev
```

## Target 중심 실행 UI

Run form은 백엔드의 `GET /api/benchmark/targets`를 호출해 사용 가능한 target을 표시합니다.

- `cpu`, `cuda`, `vllm-cpu`, `vllm-cuda`, `vendor_mock_npu`, `hailo8`, `hailo10h` 같은 target을 선택할 수 있습니다.
- target 선택 시 runtime, device, compiler 정보가 함께 표시됩니다.
- compiler가 있는 target은 compile checkbox가 노출됩니다.
- 기존 backend/device 값은 호환을 위해 유지되지만, UI에서는 target 선택을 우선합니다.

결과 목록과 비교 차트는 target metadata를 함께 사용합니다. 상세 모달은 `target_id`, accelerator vendor/name, runtime, compiler, artifact format, `hw_accel_*` metric을 보여줍니다.

## 검증

```bash
npm run lint
npm run build
```

이번 MVP에서 `CompareChart.tsx`의 conditional Hook lint 문제를 함께 수정했으며, 위 두 명령이 통과하는 것을 확인했습니다.
