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
장비의 설치 문서를 먼저 따른다.

| 장비 | 설치·환경 문서 |
|---|---|
| Hailo-8/10H | [Hailo 가이드](../hailo-async-runtime.md) |
| DEEPX | [DEEPX 가이드](../deepx-setup.md) |
| Mobilint Regulus | [Regulus 가이드](../mobilint-regulus-runtime.md) |
| Rebellions | [정적 모델](../../framework/docs/rbln-setup.md) · [LLM](../../framework/docs/rbln-vllm-setup.md) |
| Furiosa | [Furiosa 가이드](../furiosa-rngd-setup.md) |
| Mobilint ARIES | [ARIES 가이드](../mobilint-aries-transformers.md) |

## 웹 화면을 사용할 때

백엔드는 프레임워크 의존성이 설치된 Python에서 실행한다. 아래 명령은 위에서
만든 `framework/.venv`를 사용해 백엔드 패키지를 추가한다. 프론트엔드도 설치한다.

```bash
cd /path/to/ML-HW-Benchmark-Framework/backend
../framework/.venv/bin/python -m pip install -r requirements.txt
cd ../frontend
npm install
```

NPU 전용 Python 환경에서 웹 실행을 요청하려면 백엔드가 사용하는
`FRAMEWORK_PYTHON`도 그 환경의 Python으로 지정해야 한다. 실제 서버 시작
명령은 [벤치마크 실행](03-execution.md)에 있다.
