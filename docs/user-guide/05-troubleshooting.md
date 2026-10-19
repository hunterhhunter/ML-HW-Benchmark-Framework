# 05. 오류 대응

[목차](README.md) · [이전: 결과 확인 및 해석](04-results.md) · [다음: 실험 재현](06-reproduction.md)

| 증상 | 먼저 확인할 곳 |
|---|---|
| Python import 오류 | 설치에 사용한 Python과 실행 Python이 같은지 확인한다. `python -m pip show <패키지명>`으로 해당 환경을 점검한다. |
| 모델·데이터 준비 실패 | 준비 스크립트 출력, 네트워크 접근·인증, 파일 경로와 정답 파일을 확인한다. |
| NPU compile·load 실패 | 해당 target의 SDK·장치 상태, artifact 형식, 입력 shape·layout·dtype을 [설정 문서](02-configuration.md)와 대조한다. |
| 출력 shape 또는 decoder 오류 | 실제 artifact 출력과 [모델별 출력 계약](02-configuration.md#조합별-설정과-재현-명령)을 대조한다. raw head와 NMS·후처리된 출력을 혼용하지 않는다. |
| `RUN_ID_RESERVED`만 출력 | 오류 출력과 결과 디렉터리의 예약 상태를 확인한다. `RUN_ID`가 없으면 완료 결과로 취급하지 않는다. |
| 비동기 `async_run_status=invalid` | CSV의 `async_invalid_reasons`, `details_path`, 필요 시 trace를 [비동기 가이드](../../framework/docs/async-inference-queue.md)의 lifecycle 항목과 대조한다. |
| CLI 결과가 웹 목록에 없음 | CLI의 `--results-path`를 확인하고 웹 결과 화면에 해당 CSV source를 등록한다. 파일이 `WEB_RESULTS_ROOTS` 안에 있고 backend가 읽을 수 있는지 확인한다. |
| 웹 preview에서 runner 없음 | `WEB_RUNNERS_FILE`의 Python 경로와 target/model 조건을 확인한다. target 등록은 SDK·장치 가용성을 보증하지 않는다. |
| 웹 작업은 `completed`인데 결과가 없음 | job의 `RUN_ID`, `result_present`, `result_validation`, 결과 CSV 경로를 따로 확인한다. exit code 0만으로 CSV 저장을 추정하지 않는다. |

벤더별 오류는 다음 문서에서 점검한다.

| 장비 | 점검 문서 |
|---|---|
| Hailo-8/10H | [Runtime·HEF 조건](../vendors/hailo/hailo-async-runtime.md#지원-범위) |
| DEEPX | [설치·검증](../vendors/deepx/deepx-setup.md) |
| Mobilint Regulus | [보드 사전 확인](../vendors/mobilint/mobilint-regulus-runtime.md#보드-사전-확인) |
| Furiosa | [오류 대응](../vendors/furiosa/furiosa-rngd-troubleshooting.md) |
| Rebellions | [정적 모델 오류 대응](../vendors/rebellions/rbln-troubleshooting.md) · [LLM 자주 발생하는 문제](../vendors/rebellions/rbln-vllm-setup.md#12-자주-발생하는-문제) |
| Mobilint ARIES | [오류 대응](../vendors/mobilint/mobilint-aries-troubleshooting.md) |

실패 원인을 확인한 뒤에는 [소량 실행](03-execution.md)으로 다시 시작한다.
