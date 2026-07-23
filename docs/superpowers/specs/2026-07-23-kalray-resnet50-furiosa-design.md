# Kalray ResNet50 Furiosa 연동 설계

## 목표

모든 가속기 벤치마크에서 `Kalray/resnet50`의 FP32 ONNX 파일
`resnet50-v1-7s.onnx`을 동일한 모델 가중치로 사용한다. Furiosa RNGD 경로만
ONNX 그래프를 PyTorch `nn.Module`로 변환한 뒤 `furiosa.torch.compile()`에
전달한다.

초기 통합에서는 `resnet50-v1-7s-q.onnx` INT8 모델을 사용하지 않는다. 실행
중 모델 다운로드와 다른 ResNet50 가중치로의 자동 대체도 허용하지 않는다.

## 모델 준비와 입력 계약

모델은 기존 준비 스크립트로 받는다.

```bash
cd ~/ML-HW-Benchmark-Framework/framework
../.venv-furiosa-torch/bin/python models/prepare_resnet50_kalray.py \
  --format onnx \
  --output models
```

준비 결과는 다음 경로에 있어야 한다.

```text
models/Kalray_resnet50/resnet50-v1-7s.onnx
```

ResNet50 모델 프로필의 공통 모델 경로와 Furiosa Torch 모델 경로는 모두 이
파일을 가리킨다. 로더는 변환 라이브러리를 불러오기 전에 경로가 실제 파일인지,
확장자가 `.onnx`인지 검증한다.

입력은 ImageNet 전처리를 거친 NCHW FP32 텐서이며 shape는 `[N, 3, 224,
224]`이다. 출력 계약은 `[N, 1000]` logits 하나다. 전처리의 resize, center
crop, 채널 순서, 정규화 값은 모든 런타임에서 동일하게 유지한다.

## Furiosa 로딩 경로

1. `onnx.load()`로 FP32 ONNX 모델을 읽는다.
2. `onnx2torch.convert()`로 PyTorch 모듈로 변환하고 `eval()` 상태로 둔다.
3. 어댑터가 변환 결과를 공통 `logits` 출력 계약으로 정규화한다.
4. Furiosa 런타임이 이 모듈을 `furiosa.torch.compile()`로 컴파일한다.
5. 컴파일과 워밍업은 측정 구간에서 제외하고 실제 추론만 측정한다.

변환 또는 컴파일이 지원되지 않는 연산자로 실패하면 해당 오류를 명확히
노출한다. `torchvision`, Microsoft ResNet50, 랜덤 초기화 모델 등으로
fallback하지 않는다. 그래야 벤더 간 모델 동일성이 깨지지 않는다.

## 의존성

Furiosa Torch 전용 요구사항에 다음 버전을 고정한다.

```text
onnx==1.20.1
onnx2torch==1.5.15
```

`onnx2torch`는 런타임 모듈의 최상단에서 import하지 않고 ResNet50을 실제로
선택했을 때만 지연 import한다. 따라서 LLM이나 다른 모델 실행에는 변환기
설치/초기화 비용이 생기지 않는다.

## 검증

자동 테스트에서는 다음을 확인한다.

- 모델 프로필의 공통 경로와 Furiosa Torch 경로가 같은 FP32 ONNX 파일이다.
- 파일 누락, 디렉터리 입력, 잘못된 확장자는 변환 전에 이해 가능한 오류가 난다.
- 변환기는 지연 import되며 네트워크 다운로드를 시도하지 않는다.
- 변환된 모듈은 `eval()` 상태이고 공통 입력/출력 이름을 지킨다.

실제 모델을 내려받은 서버 검증에서는 고정된 동일 입력에 대해 ONNX Runtime
CPU 출력과 변환된 PyTorch CPU 출력을 `rtol=1e-3`, `atol=1e-4`로 비교한다.
그 다음 RNGD에서 단일 샘플 추론과 ImageNet 벤치마크를 실행하고
`furiosa-smi status`로 NPU 사용을 확인한다.

모델 파일 SHA-256도 결과 메타데이터에 기록해 서버와 런타임이 달라도 같은
가중치를 사용했는지 확인할 수 있게 한다.

## 범위 밖 항목

- INT8 ONNX 정확도 및 성능 검증
- ONNX 변환이 실패할 때의 대체 모델 제공
- 실행 중 Hugging Face 다운로드
- ResNet50 외 모델의 변환 방식 변경

## 참고

- [Kalray/resnet50](https://huggingface.co/Kalray/resnet50)
- [onnx2torch](https://pypi.org/project/onnx2torch/)
