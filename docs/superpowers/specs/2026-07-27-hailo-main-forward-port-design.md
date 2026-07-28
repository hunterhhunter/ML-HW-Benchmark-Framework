# Hailo Native Async 최신 main Forward-port 설계

## 배경과 확인된 원인

`feat/hailo-native-async`의 HailoRT `InferModel.run_async()` 연동 자체는 Hailo10H와 HailoRT 5.3.0에서 정상 동작한다. 독립 SDK 테스트는 제출, 콜백, `AsyncInferJob.wait()`까지 완료됐고, 프레임워크에서도 단일 샘플은 정상 종료했다.

하지만 `worker_count=1`, native executor의 `max_inflight=1`인 상태에서 두 번째 요청부터 교착된다. Jetson의 스택 덤프에서 worker는 `runtime_executor.py`의 permit 획득에서 대기했고, 첫 요청의 completion handoff는 worker-local 목록에 남아 terminal ACK 뒤에도 retire되지 않았다. 첫 요청의 물리 콜백은 끝났지만 permit이 반환되지 않아 두 번째 제출이 영구 대기하는 구조다.

이는 `origin/main`의 Mobilint 문제 해결 문서에 기록된 SDK 자원과 프레임워크 요청 수명 분리 문제와 같은 계열이다. 최신 main에는 다음 공용 수정이 이미 포함돼 있다.

- `d3873ec`: 연속 native 요청의 permit 교착 회귀 수정
- `df5f3a2`: one-shot async retirement lease 도입
- `0f069b5`: terminal commit 이후 completion lease retire
- `224d981`: completion terminal 경로에서 native dispatch retire

따라서 Hailo 런타임에 별도 우회 로직을 추가하지 않고, 최신 main의 공용 retirement 계약 위로 Hailo 기능을 forward-port한다.

## 결정

현재 공개된 `feat/hailo-native-async` 브랜치에 최신 `origin/main`을 일반 merge commit으로 병합한다. 브랜치 이력을 재작성하거나 force-push하지 않는다. 병합과 검증은 현재 사용자의 미커밋 변경과 분리된 clean worktree에서 수행한다.

이 방식의 목적은 다음과 같다.

- Jetson이 이미 추적 중인 같은 원격 브랜치를 계속 사용한다.
- Hailo뿐 아니라 Mobilint, DeepX, Furiosa 등 모든 native executor가 같은 permit/ACK/terminal 수명주기를 사용한다.
- 최신 main에 축적된 회귀 테스트와 후속 안정화 수정 전체를 보존한다.

## 병합 및 충돌 처리 원칙

충돌이 생기면 다음 우선순위를 적용한다.

1. `origin/main`의 `NativeAsyncRuntimeExecutor`, completion handoff, retirement lease, terminal commit 규약을 기준 구현으로 유지한다.
2. 기존 Hailo 커밋 `89a5b09`에서 Hailo 전용 기능만 보존한다.
   - HailoRT `InferModel` 기반 native async submit/callback/wait 처리
   - Hailo 8/Hailo10H 대상 설정과 ResNet50/YOLOv5m 연동
   - Hailo runtime option, CLI, 문서 및 Hailo 전용 테스트
3. main에 존재하는 다른 backend의 동작과 공용 async 지표·검증 계약을 퇴행시키지 않는다.
4. 현재 기본 작업트리의 DeepX 및 사용자 문서 변경은 읽거나 이동하거나 커밋하지 않는다.
5. 공용 수명주기와 Hailo 전용 로직 사이의 의미가 불명확한 충돌은 임의로 해결하지 않고 증거를 추가 확인한다.

## 요청 수명주기

병합 후 Hailo 요청은 다음 순서를 따라야 한다.

1. scheduler가 native executor permit을 획득한다.
2. HailoRT가 `run_async()`로 요청을 제출하고 물리 job을 보관한다.
3. Hailo 콜백이 성공 또는 실패를 terminal 상태로 전달한다.
4. 프레임워크 completion 경로가 결과 반영과 terminal commit을 끝낸다.
5. one-shot retirement lease가 해당 dispatch를 ACK하고 permit을 정확히 한 번 반환한다.
6. 그 뒤 대기 중인 다음 요청이 permit을 획득한다.

콜백 이전 조기 반환과 terminal 이후 미반환을 모두 금지한다. 실패, timeout, 취소 경로도 동일한 one-shot retirement 규약을 따라야 한다.

## 검증

로컬 검증은 clean worktree에서 다음 순서로 진행한다.

1. `max_inflight=1`, `worker_count=1`, 연속 2요청 회귀 테스트를 실행해 첫 terminal 완료 뒤 두 번째 요청이 제출되는지 확인한다.
2. 공용 native executor, async completion/runner, Hailo runtime 및 CLI 테스트를 실행한다.
3. 전체 테스트 스위트를 실행하고 기존 CUDA 비가용 환경처럼 장비 의존 skip/failure는 원인과 함께 분리 기록한다.
4. 병합 결과가 최신 `origin/main`과 Hailo 변경을 모두 포함하고, 사용자 미커밋 파일이 포함되지 않았는지 diff와 commit graph로 확인한다.

Jetson Hailo10H에서는 같은 HEF와 데이터셋으로 다음 단계를 수행한다.

1. YOLOv5m 단일 샘플 smoke test
2. `worker_count=1`, 20샘플 연속 async test
3. ResNet50 단일 및 20샘플 test
4. 필요 시 queue size 이하의 worker/inflight 조합 확대 테스트

각 실행에서 submitted, accepted, completed, failed, timed-out, outstanding 수와 종료 여부를 확인한다. Hailo 8은 호환 코드를 로컬 테스트로 검증하되, 실제 장비 smoke test는 Hailo 8 장비에서 별도로 수행한다.

## 배포와 Jetson 갱신

검증이 끝나면 merge commit을 기존 `origin/feat/hailo-native-async`에 일반 push한다. Jetson에서는 로컬 결과 파일을 먼저 보존한 뒤 해당 브랜치를 `git pull --ff-only`로 갱신한다. 이후 위의 Hailo10H 테스트를 재실행해 20샘플 교착이 해소됐는지 확인한다.
