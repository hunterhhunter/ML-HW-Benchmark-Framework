# 01. 환경 구축 및 설치

[목차](README.md) · [다음: NPU·모델·데이터셋 설정](02-configuration.md)

## 기본 Python 환경

CPU 실행의 출발점은 프레임워크 디렉터리의 Python 환경이다.

```bash
cd /path/to/ML-HW-Benchmark-Framework/framework
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

NPU·GPU는 장비의 드라이버와 Runtime/SDK가 설치된 Python 환경을 사용한다.
공통 `requirements.txt`만 설치해서 벤더 환경이 준비되는 것은 아니다. 실행할
장비의 설치 문서를 먼저 따른다. 검증 당시 버전과 제출 시 다시 기록할 항목은
[실행 환경 기록](../execution-environments.md)에 모았다.

| 장비 | 설치·환경 문서 | 실행 전 장치 확인 |
|---|---|---|
| Hailo-8/10H | [Hailo 가이드](../vendors/hailo/hailo-async-runtime.md) | `python -c 'from hailo_platform import Device; print(Device.scan())'` |
| DEEPX | [DEEPX 가이드](../vendors/deepx/deepx-setup.md) | [설치 검증](../vendors/deepx/deepx-setup.md#37-설치-검증) |
| Mobilint Regulus | [Regulus 가이드](../vendors/mobilint/mobilint-regulus-runtime.md) | [보드 사전 확인](../vendors/mobilint/mobilint-regulus-runtime.md#보드-사전-확인) |
| Rebellions | [정적 모델](../vendors/rebellions/rbln-setup.md) · [LLM](../vendors/rebellions/rbln-vllm-setup.md) | [정적 모델](../vendors/rebellions/rbln-setup.md#3-실행-전-점검) · [LLM](../vendors/rebellions/rbln-vllm-setup.md#2-준비된-atom-서버에서-시작) |
| Furiosa | [Furiosa 가이드](../vendors/furiosa/furiosa-rngd-setup.md) | `furiosa-smi status` |
| Mobilint ARIES | [ARIES 가이드](../vendors/mobilint/mobilint-aries-transformers.md) | [환경·장치 확인](../vendors/mobilint/mobilint-aries-transformers.md#1-실행-환경과-장치-확인) |

## 설치 확인

실제로 벤치마크를 실행할 Python에서 버전과 패키지 설치 위치를 확인한다.

```bash
cd /path/to/ML-HW-Benchmark-Framework/framework
python -V
python -c 'import sys; print(sys.executable)'
python -m pip --version
python src/main.py --help >/dev/null && echo 'framework CLI: OK'
```

마지막 명령은 CLI가 시작되는지만 확인한다. NPU 실행 전에는 위 표의 장치 확인과
해당 가이드의 SDK 버전·artifact 조건도 점검한다. Furiosa는 모델에 따라
[전용 Python 환경](../vendors/furiosa/furiosa-rngd-setup.md#전용-python-환경)이 다르다.

## 웹 화면을 사용할 때

백엔드는 프레임워크 의존성이 설치된 Python에서 실행한다. 아래 명령은 위에서
만든 `framework/.venv`를 사용해 백엔드 패키지를 추가한다. 프론트엔드도 설치한다.

```bash
cd /path/to/ML-HW-Benchmark-Framework/backend
../framework/.venv/bin/python -m pip install -r requirements.txt
cd ../frontend
npm ci
```

NPU 전용 Python 환경은 `WEB_RUNNERS_FILE`의 runner profile에 등록한다.
설정 파일이 없으면 `FRAMEWORK_PYTHON`이 기본 실행 환경이다. backend 자체는
프레임워크 코드를 읽을 수 있는 Python 환경에서 실행한다. 실제 서버 시작
명령은 [벤치마크 실행](03-execution.md)에 있다.
