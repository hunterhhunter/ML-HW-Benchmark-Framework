# 05. 오류 대응

[목차](README.md) · [이전: 결과 확인 및 해석](04-results.md) · [다음: 실험 재현](06-reproduction.md)

| 증상 | 먼저 확인할 곳 |
|---|---|
| Python import 오류 | 설치에 사용한 Python과 실행 Python이 같은지 확인한다. `python -m pip show <패키지명>`으로 해당 환경을 점검한다. |
| 모델·데이터 준비 실패 | 준비 스크립트 출력, 네트워크 접근·인증, 파일 경로와 정답 파일을 확인한다. |
| NPU compile·load 실패 | 해당 target의 SDK·장치 상태, artifact 형식, 입력 shape·layout·dtype을 [설정 문서](02-configuration.md)와 대조한다. |
| `RUN_ID_RESERVED`만 출력 | 오류 출력과 결과 디렉터리의 예약 상태를 확인한다. `RUN_ID`가 없으면 완료 결과로 취급하지 않는다. |
| 비동기 `async_run_status=invalid` | CSV의 `async_invalid_reasons`, `details_path`, 필요 시 trace를 [비동기 가이드](../async-inference-queue.md)의 lifecycle 항목과 대조한다. |
| CLI 결과가 웹 목록에 없음 | CLI의 `--results-path`를 확인한다. 웹 목록은 기본 CSV를 읽는다. |

벤더 Runtime별 설치·장치 점검 명령은 [환경 구축 및 설치](01-installation.md)에
연결한 가이드를 따른다. 실패 원인을 확인한 뒤에는
[소량 실행](03-execution.md)으로 다시 시작한다.
