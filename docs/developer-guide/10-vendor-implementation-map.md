# 10. 벤더 구현 지도

[목차](README.md) · [이전: 결과 저장과 산출물 수명](09-result-persistence.md) · [다음: 웹과 CLI 연결](11-web-cli-integration.md)

이 장은 현재 target에서 **어느 Runtime과 SDK 호출로 들어가는지** 찾는 지도다.
벤더별 설치·실행 명령은 각 운영 가이드, 모델·artifact별 입력과 출력은
[모델별 실행 명세](../../framework/docs/model-execution-specs/resnet50.md)를
따른다. target 등록 정보는 [`targets.py`](../../framework/src/core/targets.py),
Runtime class 등록은 [`runtimes/__init__.py`](../../framework/src/runtimes/__init__.py),
Loader·Decoder 선택과 옵션 병합은 [`main.py`](../../framework/src/main.py)를
확인한다.

## target에서 실제 호출까지

| target | Runtime과 실행 경계 | 모델별 연결·상세 가이드 |
|---|---|---|
| `hailo8`, `hailo10h` | [`HailoRuntime`](../../framework/src/runtimes/hailo_rt.py)이 HEF를 로드한다. 사용 가능한 SDK에서는 InferModel을 구성하고, 그 API가 없는 동기 경로는 InferVStreams를 쓴다. native async는 InferModel의 `run_async()` 경로다. | Hailo Loader, [NMS Decoder](../../framework/src/decoders/object_detection.py)와 [Pose Decoder](../../framework/src/decoders/hailo_yolov8_pose.py); [Hailo 가이드](../hailo-async-runtime.md) |
| `deepx` | [`DeepXRuntime`](../../framework/src/runtimes/deepx_rt.py)이 DXNN과 tensor metadata를 읽는다. 동기는 SDK `run` 계열, native async는 `DeepXNativeBackend`의 DX-RT callback 경로다. | [DEEPX Loader](../../framework/src/dataloader/deepx_vision_loader.py), [Pose Decoder](../../framework/src/decoders/deepx_yolov8_pose.py); [DEEPX 가이드](../deepx-setup.md) |
| `mobilint-aries`, `mobilint-regulus` | [`MobilintRuntime`](../../framework/src/runtimes/mobilint_rt.py)이 MXQ와 장치 family·입출력 계약을 확인한다. 기본 추론은 `Model.infer()`, 지원 artifact에 async pipeline을 활성화한 경우 `infer_async()`를 callback 계약으로 연결한다. | [vision profile](../../framework/src/dataloader/mobilint_vision_profiles.py), [Regulus Pose profile](../../framework/src/dataloader/regulus_vision_profiles.py), [YOLOv5 Decoder](../../framework/src/decoders/mobilint_yolov5.py); [Regulus 가이드](../mobilint-regulus-runtime.md) |
| `mobilint-aries-llm` | [`MobilintLlmRuntime`](../../framework/src/runtimes/mobilint_llm_rt.py)이 Model Zoo/Transformers 모델을 로드하고 blocking `generate()`를 호출한다. MXQ Runtime의 native async 경로와 별개다. | [ARIES Transformer·LLM 가이드](../mobilint-aries-transformers.md) |
| `furiosa-rngd` | [`FuriosaLlmRuntime`](../../framework/src/runtimes/furiosa_llm_rt.py)이 Furiosa-LLM `generate()`를 사용한다. `FuriosaNativeBackend`는 async 생성 stream을 framework callback으로 전달한다. | FXB와 모델 repository 조건은 [RNGD 가이드](../furiosa-rngd-setup.md) |
| `furiosa-rngd-torch` | [`FuriosaTorchRuntime`](../../framework/src/runtimes/furiosa_torch_rt.py)이 고정 shape PyTorch 모델을 `torch.compile()`의 Furiosa backend로 컴파일한 뒤 호출한다. 이 target은 `furiosa-rngd`의 LLM Runtime과 환경·artifact가 다르다. | [모델 adapter](../../framework/src/runtimes/furiosa_torch_models.py), [RNGD 가이드](../furiosa-rngd-setup.md) |
| `rbln-static` | [`RblnRuntime`](../../framework/src/runtimes/rbln_rt.py)이 사전 컴파일한 정적 RBLN artifact를 실행하고 `RblnNativeBackend`로 callback 경로를 제공한다. | [RBLN 정적 운영 가이드](../../framework/docs/rbln-setup.md) |
| `rbln-vllm` | [`RblnVllmRuntime`](../../framework/src/runtimes/rbln_vllm_rt.py)이 준비된 RBLN 모델 디렉터리로 vLLM 생성을 수행한다. 동기 `generate()`와 async stream backend가 있다. | [RBLN vLLM 가이드](../../framework/docs/rbln-vllm-setup.md) |

표의 호출 경계는 **현재 코드 경로**를 뜻한다. 실제 artifact가 어떤 tensor를
내고 전처리·후처리의 어느 단계를 포함하는지는 target 이름으로 결정되지
않는다. Loader metadata, artifact metadata, Runtime 반환값과 선택된 Decoder를
한 조합으로 확인한다. [실행 설정과 조립](07-configuration-and-assembly.md)은
이 선택이 `main.py`에서 이루어지는 순서를 설명한다.

## 수정 위치를 고르는 기준

| 바뀌는 계약 | 시작할 파일 |
|---|---|
| SDK 호출, 동기화, callback, buffer·context 수명 | 해당 `runtimes/*_rt.py`와 [RuntimeExecutor](../../framework/src/core/runtime_executor.py) 경계 |
| 장치 선택, artifact 형식, capability, 기본 옵션 | `core/targets.py`와 Runtime registry |
| 모델 입력 shape·dtype·layout·전처리 | 해당 Loader·Preprocessor·모델 profile, `main.py`의 Loader kwargs |
| 출력 tensor 이름·의미·좌표 복원 | 해당 Runtime의 출력 정규화, `decoders/` factory와 Decoder |
| 계측 API와 전력 측정 지점 | 해당 `monitors/` adapter와 [측정 계약](../../framework/docs/measurement-contract.md) |

예를 들어 YOLOv5m의 Hailo NMS 출력 HEF와 Mobilint raw-head MXQ는
다른 Decoder로 연결된다. 새 모델이나 artifact가 기존
profile과 다르면 [모델·태스크 확장](03-model-and-task-extension.md)의
입출력 확인부터 진행한다. 새 장치는 [NPU 어댑터 확장](04-npu-adapter-extension.md)의
등록·검증 순서를 따른다. SDK 없이 구조만 검사할 때는
[검증 가이드](05-verification.md)를 사용하고, 실제 장치 검증 범위는
각 실행 명세와 결과 근거로 따로 기록한다.
