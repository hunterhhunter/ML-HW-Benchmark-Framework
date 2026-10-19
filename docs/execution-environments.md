# 실행 환경 기록

이 문서는 사용자 설명서와 제출용 보고서에 사용할 **환경 정보의 출처**를 한곳에
모은다. 아래 버전은 링크된 문서의 **검증 당시 기록**이며, 2026-10-19 현재 각
장비에 설치된 버전을 새로 조회한 값은 아니다. 설치 가능 버전, 실제 실행에 사용한
버전, 제출 시점의 설치 상태를 서로 구분한다.

## 공통 소프트웨어 구성

| 구성 | 저장소에 있는 기준 | 최종 제출 시 추가로 기록할 값 |
|---|---|---|
| CLI·공통 프레임워크 | [`framework/requirements.txt`](../framework/requirements.txt), [`main.py`](../framework/src/main.py) | 제출 commit, 실제 실행 Python 경로·버전과 설치 package 목록 |
| 웹 API | [`backend/requirements.txt`](../backend/requirements.txt), [runner profile 예시](../backend/config/runner_profiles.example.json) | API Python 환경, target별로 선택한 실제 runner와 package 버전 |
| 웹 화면 | [`frontend/package-lock.json`](../frontend/package-lock.json) | 빌드에 사용한 Node.js·npm 버전 |

공통 requirements만으로 벤더 SDK·드라이버·firmware가 설치되지는 않는다. 특히 웹
백엔드의 Python과 실제 NPU 작업에 선택된 runner Python이 다를 수 있다. 모델,
데이터셋, 벤더 artifact의 실제 파일은 대부분 Git에 포함되지 않으므로, 각 실행의
취득 방법과 SHA-256은 [모델별 명세](README.md#모델별-실행-명세) 및 실행 기록과
함께 확인한다.

## 문서에 남아 있는 검증 환경

| 장치·경로 | 검증 기록에 적힌 환경 | 상세 근거 |
|---|---|---|
| Hailo-8 | HailoRT `4.24.0`, 해당 장치용 HEF | [모델별 실행 명세](../framework/docs/model-execution-specs/resnet50.md), [Hailo 가이드](vendors/hailo/hailo-async-runtime.md) |
| Hailo-10H | HailoRT `5.3.0`, 해당 장치용 HEF | [모델별 실행 명세](../framework/docs/model-execution-specs/resnet50.md), [Hailo 가이드](vendors/hailo/hailo-async-runtime.md) |
| DEEPX DX-M1 | DX-RT `3.4.0`, 해당 DXNN | [모델별 실행 명세](../framework/docs/model-execution-specs/resnet50.md), [DEEPX 설치 가이드](vendors/deepx/deepx-setup.md) |
| Mobilint Regulus | Yocto 보드, qbruntime `1.2.0`, Regulus용 직접 컴파일 MXQ | [모델별 실행 명세](../framework/docs/model-execution-specs/resnet50.md), [Regulus 가이드](vendors/mobilint/mobilint-regulus-runtime.md) |
| Mobilint ARIES | 과거 BERT 실행: qbruntime `1.3.2`, driver `1.13.0`. 전체 package freeze는 중앙 기록에 없음 | [교차 NPU 검증 현황](../framework/docs/cross-npu-model-validation.md), [ARIES 가이드](vendors/mobilint/mobilint-aries-transformers.md) |
| Furiosa RNGD LLM | Python `3.12.13`, Furiosa-LLM `2026.3.0`, PyTorch `2.5.1` 계열 | [교차 NPU 검증 현황](../framework/docs/cross-npu-model-validation.md), [RNGD 가이드](vendors/furiosa/furiosa-rngd-setup.md) |
| Furiosa RNGD Torch | Python `3.12.13`, Furiosa Torch `2026.3.0`, PyTorch `2.10.0+cpu`, Transformers `4.57.6`, NumPy `2.5.1` | [교차 NPU 검증 현황](../framework/docs/cross-npu-model-validation.md), [RNGD 가이드](vendors/furiosa/furiosa-rngd-setup.md) |
| Rebellions RBLN-CA22 | Ubuntu `22.04.5`, Python `3.10.12`, `rebel-compiler==0.11.0`, KMD·firmware `3.2.2` | [정적 모델 운영 가이드](vendors/rebellions/rbln-setup.md) |
| Rebellions vLLM | 위 서버의 별도 실행 환경: `optimum-rbln==0.11.0.post1`, `vllm-rbln==0.11.0` | [ATOM vLLM 가이드](vendors/rebellions/rbln-vllm-setup.md), [교차 NPU 검증 현황](../framework/docs/cross-npu-model-validation.md) |

이 표만으로 에지 보드별 OS·kernel·driver·firmware·Python 전체 목록,
ARIES와 RBLN 정적 실행 환경의 전체 package 목록, 웹 빌드의 Node.js·npm 버전,
실제 웹 runner 설정을 확정할 수 없다. 제출에 사용한 실행에서 별도로 수집한다.

컴파일 환경은 실행 환경과 별도로 확인한다. DEEPX DX-COM의 설치 요구사항은
[DEEPX 가이드](vendors/deepx/deepx-setup.md)에, Mobilint `qbcompiler==1.2.0`의
Ubuntu 22.04·x86-64·Python 3.10 컴파일 호스트 조건은
[컴파일 실험 기록](vendors/mobilint/mobilint-compilation-experiments.md)에 있다.
Furiosa Torch의 검증 환경은 저장소
[`requirements-furiosa-torch.txt`](../framework/requirements-furiosa-torch.txt)에 적힌
Transformers `5.1.0`과 다르다. 설치할 때는
[RNGD 가이드의 overlay 절차](vendors/furiosa/furiosa-rngd-setup.md#bert-전용-furiosa-torch-환경)를
따르고 실제 설치 버전을 다시 남긴다.

## 제출 버전에서 다시 확보할 기록

위 표를 그대로 현재 설치 상태로 제출하지 않는다. **실제 벤치마크를 수행하는 각
호스트와 Python 환경**에서 다음을 실행 기록과 함께 남긴다.

1. 실행 날짜, 코드 commit, target·장치 모델, OS·kernel, driver·firmware 버전.
2. 실행 Python의 경로·버전, 벤더 Runtime/SDK 버전, package 목록과 `pip check` 결과.
   웹 실행이라면 backend Python과 선택한 runner ID·Python을 각각 기록한다.
3. 모델·컴파일 artifact·데이터 cohort의 식별자와 SHA-256, batch·warmup·실행 모드.
4. 결과 CSV 경로, `RUN_ID`, 완료·품질 상태와 필요한 details·trace의 위치.

공통 항목은 해당 장비에서 다음처럼 확인할 수 있다. `PYTHON`은 **실제 실행에
사용할 interpreter**로 바꾼다. 장치별 SDK·driver 확인 명령은 위 가이드를 따른다.

```bash
cd /path/to/ML-HW-Benchmark-Framework
git rev-parse HEAD
uname -srm
cat /etc/os-release
PYTHON=/absolute/path/to/benchmark-python
"$PYTHON" -VV
"$PYTHON" -m pip list --format=freeze
"$PYTHON" -m pip check
```

최종 보고서에는 이 출력 전체를 옮기기보다 장치·OS·Python·SDK·driver·firmware의
핵심 버전과 코드 commit을 표로 제시하고, package 목록과 실행 근거 파일을
해당 결과의 보관 위치에 연결한다. 측정값의 경계와 유효성은
[측정 계약](../framework/docs/measurement-contract.md)을 따른다.
