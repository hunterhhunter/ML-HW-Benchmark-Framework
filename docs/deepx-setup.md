# DEEPX 설치 및 프레임워크 연동 가이드

이 문서는 ML HW Benchmark Framework에서 `deepx` target을 사용하기 위해 필요한
DEEPX 컴파일러, 런타임, 드라이버 설치 순서와 검증 방법을 정리한다.

기준 매뉴얼:

- `DEEPX_DX-COM_UM_v2.3.0_MAR_2026.pdf`
- `DEEPX_DX-RT_UM_v3.3.0_Apr_2026.pdf`
- `DEEPX_DX-STREAM_UM_v3.0.0_MAR_2026.pdf`
- `DEEPX_DX-APP_UM_v3.1.0_MAR_2026.pdf`

## 구성 요소

| 구성 요소 | 역할 | 이 프레임워크와의 관계 |
|---|---|---|
| DX-COM | ONNX와 calibration config를 `.dxnn`으로 컴파일 | `framework/src/compilers/deepx_compiler.py`가 `dxcom` CLI 호출 |
| DX-RT | `.dxnn`을 DEEPX NPU에서 실행 | `framework/src/runtimes/deepx_rt.py`가 `dx_engine` Python API 호출 |
| DX driver | PCIe/M.2 NPU 커널 드라이버 | target 장비에서 `/dev/dxrt*`, `dx_dma`, `dxrt_driver` 제공 |
| DX-APP | DEEPX 예제 앱 모음 | 프레임워크 필수는 아니지만 NPU 설치 검증에 유용 |
| DX-STREAM | GStreamer 기반 스트리밍 pipeline | 프레임워크 필수는 아니며 영상 pipeline 검증용 |

권장 설치 순서는 `driver -> DX-RT -> dx_engine Python package -> DX-COM -> framework 실행`이다.
DX-COM은 x86_64 host에서만 지원되므로, ARM target에서는 PC에서 `.dxnn`을 만든 뒤 target으로 복사해 실행하는 구성이 안전하다.

## 0. DX-RT가 설치된 Jetson에서 바로 실행

Jetson에 DEEPX driver, DX-RT, `dx_engine`이 이미 설치돼 있다면 아래 빠른 경로만
수행하면 된다. DX-COM은 Jetson에 설치하지 않아도 되며, x86_64 host에서 만든
`.dxnn`을 내려받아 `--no-compile`로 실행한다.

먼저 기존 설치를 확인한다.

```bash
uname -m
dxrt-cli -s
ls -l /dev/dxrt*
python3 -c "import dx_engine; print(getattr(dx_engine, '__version__', 'unknown'))"
python3 -c "import dx_engine; print('run_async=', hasattr(dx_engine.InferenceEngine, 'run_async')); print('callback=', hasattr(dx_engine.InferenceEngine, 'register_callback'))"
```

`uname -m`은 `aarch64`, 장치는 보통 `/dev/dxrt0`으로 보여야 한다. 마지막 두 값이
모두 `True`이면 이 프레임워크가 DX-RT native async 경로를 선택할 수 있다.

Jetson에서 repo와 Python 환경을 준비한다. 시스템에 설치된 `dx_engine`을 그대로
보이게 하기 위해 venv는 `--system-site-packages`로 만든다.

```bash
git clone https://github.com/hunterhhunter/ML-HW-Benchmark-Framework.git
cd ML-HW-Benchmark-Framework

python3 -m venv --system-site-packages framework/.venv
framework/.venv/bin/python -m pip install -r framework/requirements.txt
framework/.venv/bin/python -c "import dx_engine; print(getattr(dx_engine, '__version__', 'unknown'))"
```

기존 checkout을 갱신할 때는 repo 루트에서 `git pull --ff-only`를 사용한다. 이미 다른
venv가 있다면 새로 만들 필요는 없지만, 그 Python에서 `import dx_engine`이 성공해야
한다.

사전 컴파일된 모델을 Jetson으로 복사하거나 artifact URL에서 받는다.

```bash
mkdir -p framework/models/deepx

# x86_64 컴파일 host에서 복사하는 예
scp user@compile-host:/path/to/ResNet50.dxnn framework/models/deepx/ResNet50.dxnn

# 사내 artifact URL이 있는 경우의 예
curl -fL "https://artifact.example.com/deepx/ResNet50.dxnn" \
  -o framework/models/deepx/ResNet50.dxnn
```

아래 두 명령은 같은 `.dxnn`과 데이터셋으로 실행 모드만 비교한다. `<DATASET>`은
ImageNet root처럼 현재 Jetson에 준비된 실제 경로로 바꾼다.

E2E(기존 blocking `run()` 경로):

```bash
cd framework
./.venv/bin/python src/main.py \
  --model resnet50 \
  --target deepx \
  --no-compile \
  --artifact models/deepx/ResNet50.dxnn \
  --dataset <DATASET> \
  --inference-mode e2e \
  --batch-size 1 \
  --warmup 2 \
  --max-steps 100 \
  --runtime-option device_ids=0 \
  --runtime-option bound_option=NPU_ALL \
  --runtime-option buffer_count=6 \
  --monitor
```

Native async(`run_async()` + DX-RT callback 경로):

```bash
./.venv/bin/python src/main.py \
  --model resnet50 \
  --target deepx \
  --no-compile \
  --artifact models/deepx/ResNet50.dxnn \
  --dataset <DATASET> \
  --inference-mode async_queue \
  --scenario offline \
  --batch-size 1 \
  --worker-count 4 \
  --queue-capacity 256 \
  --min-samples 100 \
  --max-samples 100 \
  --warmup 2 \
  --runtime-option device_ids=0 \
  --runtime-option bound_option=NPU_ALL \
  --runtime-option buffer_count=6 \
  --runtime-option async_completion_timeout_sec=30 \
  --monitor
```

DX-RT v3.3 native async는 한 submission에 batch를 넣는 방식을 지원하지 않으므로
`--batch-size 1`을 유지한다. 처리량은 여러 job을 동시에 제출해 확보하며 실제
in-flight 수는 `--worker-count`로 정한다. `worker-count`는 `buffer_count`보다 클 수
없다. `buffer_count`의 유효 범위는 1~100이고 기본값은 6이다.

프레임워크는 load 시 `register_callback()`과 `run_async()`를 실제로 확인하고 callback
등록에 성공한 경우에만 native async executor를 선택한다. callback을 지원하지 않는
구형/부분 설치에서는 E2E는 계속 동작하고 `async_queue`는 기존 blocking executor
경로를 사용한다. 이 fallback은 동시 `run()` 호출을 허용하지 않으므로
`--worker-count 1`로 실행해야 한다.

## 1. DX-COM 설치

DX-COM v2.3.0은 wheel 기반 설치 흐름을 사용한다. 공식 매뉴얼 기준 요구사항은 x86_64 CPU, Ubuntu 20.04/22.04/24.04 등, Python 3.8~3.12, LDD 2.28 이상이다. aarch64는 DX-COM 컴파일 host로 지원되지 않는다.

필수 OS 라이브러리:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 make
```

이 repo의 framework venv에 설치:

```bash
cd /home/swlab-youngjin/ML-HW-Benchmark-Framework

# DEEPX SDK bundle에서 받은 wheel을 쓰는 공식 흐름
uv pip install --python framework/.venv/bin/python /path/to/dx_com-<VERSION>-cp312-cp312-linux_x86_64.whl

# 패키지 인덱스 접근이 가능한 환경에서는 다음도 동작할 수 있다.
uv pip install --python framework/.venv/bin/python dx_com==2.3.0
```

검증:

```bash
framework/.venv/bin/dxcom --version
framework/.venv/bin/python -c "import dx_com; print(dx_com.__version__)"
```

## 2. DX-COM config 작성

`deepx` compiler는 compile mode에서 반드시 `config_path`를 요구한다.
DX-COM CLI JSON config는 단일 입력 모델만 지원하며, 입력 batch는 1로 고정해야 한다.
컴파일 중 quantization calibration이 수행되므로 `calibration_method`, `calibration_num`,
`default_loader`를 준비해야 한다.

ResNet50/ImageNet 예시:

```json
{
  "inputs": {
    "input": [1, 3, 224, 224]
  },
  "calibration_method": "ema",
  "calibration_num": 100,
  "default_loader": {
    "dataset_path": "/path/to/imagenet_1k/val",
    "file_extensions": ["jpg", "jpeg", "png", "JPEG"],
    "preprocessings": [
      {"resize": {"mode": "torchvision", "size": 232, "interpolation": "BILINEAR"}},
      {"centercrop": {"width": 224, "height": 224}},
      {"convertColor": {"form": "BGR2RGB"}},
      {"div": {"x": 255}},
      {"normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
      {"transpose": {"axis": [2, 0, 1]}}
    ]
  }
}
```

`default_loader`가 이미지를 HWC로 읽는 환경에서는 NCHW ONNX 입력과 맞추기 위해 마지막 `transpose`가 필요하다.
모델 입력 이름은 ONNX graph의 실제 input 이름과 정확히 같아야 한다.

DX-COM 직접 실행:

```bash
cd /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework

./.venv/bin/dxcom \
  -m artifacts/deepx_resnet50_imagenet1k_v2/source/resnet50_imagenet1k_v2_opset17.onnx \
  -c artifacts/deepx_resnet50_imagenet1k_v2/resnet50_imagenet1k_v2_dxcom_config.json \
  -o artifacts/deepx_resnet50_direct \
  --opt_level 1 \
  --gen_log
```

주요 옵션:

| 옵션 | 의미 |
|---|---|
| `--opt_level 0` | 빠른 컴파일, 실행 latency는 늘 수 있음 |
| `--opt_level 1` | 기본값, 성능 중심 최적화 |
| `--gen_log` | output directory에 `compiler.log` 생성 |
| `--float64_calibration` | calibration/offset 계산에 float64 사용 |
| `--aggressive_partitioning` | 더 많은 op를 NPU에 올리려는 실험적 옵션 |

## 3. DX-RT 및 드라이버 설치

DX-RT는 target 장비에서 `.dxnn`을 실행하는 런타임이다. Linux 기준 DX-RT는 Ubuntu 18.04/20.04/22.04/24.04를 지원하며, driver는 M1 AI Accelerator PCIe/M.2 장치에 필요하다.

### 3.1 DX-RT build dependency 설치

DX-RT source tree에서:

```bash
cd /path/to/dx_rt
./install.sh --all
```

아키텍처를 명시해야 하는 target이면:

```bash
./install.sh --arch aarch64 --onnxruntime
```

### 3.2 DX-RT build/install

일반 설치:

```bash
cd /path/to/dx_rt
./build.sh --clean
./build.sh --install /usr/local
sudo ldconfig
```

aarch64 target build 예:

```bash
./build.sh --clean
./build.sh --arch aarch64 --install /usr/local
sudo ldconfig
```

release package가 제공되면 `.deb`로 설치할 수도 있다.

```bash
sudo dpkg --install release/libdxrt_<version>_all.deb
```

### 3.3 드라이버 설치 방법 A: DKMS Debian package

배포/운영 환경에서는 DKMS package 설치가 가장 단순하다.

```bash
sudo apt update
sudo apt install ./dxrt-driver-dkms_<version>_all.deb
dkms status
```

`sudo apt install ./...deb` 사용 시 `_apt` permission 경고가 나올 수 있는데, local file 설치 과정의 정보 메시지일 수 있다. 실제 성공 여부는 `dkms status`, `lsmod`, `SanityCheck.sh`로 확인한다.

### 3.4 드라이버 설치 방법 B: source build

커널 모듈을 직접 빌드해야 하는 환경에서는 source build를 사용한다. 기존 DKMS driver가 설치되어 있으면 먼저 제거해야 한다.

```bash
cd /path/to/dx_rt/module/rt_npu_linux_driver

# build
./build.sh -d m1 -m deepx

# install modules and modprobe config
sudo ./build.sh -d m1 -m deepx -c install

# verify module load
sudo modprobe dx_dma
lsmod | grep dx
```

수동 설치가 필요하면 `modules/` 아래에서 다음 흐름을 따른다.

```bash
cd modules
make DEVICE=m1 PCIE=deepx
sudo make DEVICE=m1 PCIE=deepx install
sudo depmod -A
sudo cp dx_dma.conf /etc/modprobe.d/
sudo modprobe dx_dma
```

### 3.5 DX-RT Python package 설치

이 프레임워크의 `deepx` runtime은 `dx_engine` Python module을 import한다. target 장비에서 framework venv에 설치한다.

```bash
cd /path/to/dx_rt/python_package
uv pip install --python /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python .
```

검증:

```bash
/home/swlab-youngjin/ML-HW-Benchmark-Framework/framework/.venv/bin/python -c "import dx_engine; print(dx_engine.__version__)"
```

Python C++ extension mismatch 오류가 나면 DX-RT Python package를 실행에 쓰는 Python 버전과 같은 Python으로 다시 빌드/설치한다.

### 3.6 dxrtd service 등록

DX-RT를 service 지원 옵션으로 빌드한 경우 `dxrtd` systemd service를 등록한다.

```bash
cd /path/to/dx_rt
sudo cp ./service/dxrt.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable dxrt.service
sudo systemctl start dxrt.service
sudo systemctl status dxrt.service
```

로그 확인:

```bash
sudo journalctl -u dxrt.service
```

`dxrtd`는 장치당 하나만 실행한다. systemd의 `dxrt.service`를 사용 중이면 별도의
터미널에서 `dxrtd`를 다시 실행하지 않는다. 충돌이 의심되면 아래처럼 service와
수동 프로세스를 함께 확인한다.

```bash
systemctl status dxrt.service
pgrep -a dxrtd
```

firmware update 시 문제가 있으면 service를 잠시 내리고 업데이트한 뒤 다시 올린다.

```bash
sudo systemctl stop dxrt.service
dxrt-cli -u fw.bin
sudo systemctl start dxrt.service
```

### 3.7 설치 검증

DX-RT source tree에서:

```bash
sudo ./SanityCheck.sh
sudo ./SanityCheck.sh dx_rt
sudo ./SanityCheck.sh dx_driver
```

수동 확인:

```bash
lspci -nn | grep -i 1ff4
ls -l /dev/dxrt*
lsmod | grep dx
dkms status
dxrt-cli --status
dxrt-cli --info
```

정상 상태에서는 `/dev/dxrt0`, `dxrt_driver`, `dx_dma`, `dxrt-cli`, `dxrtd`가 확인되어야 한다.

## 4. 프레임워크에서 실행

### 4.1 Compile + run

`--target deepx`는 기본적으로 DX-COM compile을 먼저 수행한 뒤 DX-RT runtime에 `.dxnn`을 전달한다.
`--monitor`를 활성화하면 framework는 `dx_engine.dev_status.DeviceStatus`를 사용해 DEEPX NPU 온도(`hw_accel_temp_c`), 전압(`hw_accel_voltage_mv`), 클럭(`hw_accel_clock_mhz`)을 수집하고, `system` collector로 CPU/RAM도 함께 기록한다. DX-RT의 `dxtop`은 utilization과 DRAM 사용량을 실시간으로 보여주지만, 현재 framework collector는 문서상 Python API로 확인되는 DeviceStatus 항목만 직접 저장한다.

```bash
cd /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework

./.venv/bin/python src/main.py \
  --model resnet50 \
  --target deepx \
  --compile-option config_path=artifacts/deepx_resnet50_imagenet1k_v2/resnet50_imagenet1k_v2_dxcom_config.json \
  --compile-option dxcom_bin=./.venv/bin/dxcom \
  --runtime-option device_ids=0 \
  --runtime-option bound_option=NPU_ALL \
  --runtime-option input_layout=NCHW \
  --monitor
```

지원하는 compile option:

| `--compile-option` | DX-COM CLI 대응 |
|---|---|
| `config_path=/path/config.json` | `-c` |
| `dxcom_bin=/path/dxcom` | 실행파일 경로 |
| `opt_level=0` 또는 `opt_level=1` | `--opt_level` |
| `gen_log=true` | `--gen_log` |
| `float64_calibration=true` | `--float64_calibration` |
| `aggressive_partitioning=true` | `--aggressive_partitioning` |
| `compile_input_nodes=...` | `--compile_input_nodes` |
| `compile_output_nodes=...` | `--compile_output_nodes` |

지원하는 runtime option:

| `--runtime-option` | 의미 |
|---|---|
| `device_ids=0` 또는 `device_ids=0,1` | 사용할 NPU device id |
| `bound_option=NPU_ALL` | DX-RT bound option |
| `use_ort=true` | DX-RT option에서 ORT 사용 |
| `buffer_count=6` | DX-RT runtime buffer 수이자 native async 최대 in-flight 상한. 기본 6, 범위 1~100 |
| `async_completion_timeout_sec=30` | native async callback 완료를 기다리는 논리 timeout(초) |
| `input_layout=NCHW` 또는 `input_layout=NHWC` | framework 입력을 runtime 입력 layout에 맞춤 |
| `batch_mode=sdk_batch` 또는 `batch_mode=microbatch` | batch 처리 방식 |

### 4.2 Precompiled `.dxnn` E2E 실행

컴파일 host와 실행 target을 분리하는 경우, PC에서 만든 `.dxnn`을 target으로 복사한 뒤 compile을 건너뛴다.

```bash
cd /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework

./.venv/bin/python src/main.py \
  --model resnet50 \
  --target deepx \
  --no-compile \
  --artifact /path/to/resnet50.dxnn \
  --runtime-option device_ids=0 \
  --runtime-option bound_option=NPU_ALL \
  --monitor
```

`--inference-mode e2e`가 기본값이다. 명시적으로 비교 실행할 때는 옵션을 추가해도
동일하며, 이 경로는 DX-RT `run()`/`run_multi_input()`을 사용한다.

백엔드 API에서는 `artifact_path`로 precompiled artifact를 전달한다.

```json
{
  "model": "resnet50",
  "target_id": "deepx",
  "backend": "deepx",
  "device": "npu0",
  "compile": false,
  "artifact_path": "/path/to/resnet50.dxnn",
  "batch_size": 1,
  "warmup": 2,
  "max_steps": 100,
  "monitor": true
}
```

### 4.3 Precompiled `.dxnn` native async 실행

DX-RT v3.3의 native callback API를 쓰려면 `async_queue`를 선택하고 batch를 1로
유지한다. `worker-count`가 `buffer_count`보다 크면 프레임워크가 실행 전에 잘못된
구성으로 거부한다.

```bash
cd /home/swlab-youngjin/ML-HW-Benchmark-Framework/framework

./.venv/bin/python src/main.py \
  --model resnet50 \
  --target deepx \
  --no-compile \
  --artifact /path/to/resnet50.dxnn \
  --dataset /path/to/imagenet_1k \
  --inference-mode async_queue \
  --scenario offline \
  --batch-size 1 \
  --worker-count 4 \
  --queue-capacity 256 \
  --min-samples 100 \
  --max-samples 100 \
  --runtime-option buffer_count=6 \
  --runtime-option async_completion_timeout_sec=30 \
  --monitor
```

adapter는 DX-RT `user_arg`에 submission token을 전달하므로 callback 순서가 제출
순서와 달라도 올바른 요청에 결과를 연결한다. DX-RT가 소유한 출력 메모리는 callback
안에서 복사한 뒤 프레임워크에 전달하고, 완료되지 않은 job이 있으면 runtime unload를
거부한다.

### 4.4 사전컴파일 DXNN 이미지 입력 처리

사전컴파일된 `.dxnn`은 원본 ONNX 입력 shape만 보고 전처리 방식을 결정하면 안 된다. DX-COM이 만든 DXNN 안에는 `compile_config`와 NPU `rmap_info`가 포함될 수 있고, 실제 DX-RT 입력 ABI는 ONNX graph의 입력과 다를 수 있다.

ResNet50 사전컴파일 모델(`models/deepx/ResNet50.dxnn`)에서 확인한 실제 입력은 다음과 같았다.

```json
{
  "name": "input.1",
  "dtype": "UINT8",
  "shape": [1, 224, 224, 3],
  "layout": "PRE_IM2COL"
}
```

즉 framework의 일반 ResNet50 경로처럼 `NCHW float32 normalized`를 넣으면 안 되고, DX-APP 예제와 같은 `NHWC uint8` 이미지 입력을 넣어야 한다. 이 mismatch가 있으면 출력 shape은 `(1, 1000)`으로 정상처럼 보여도 예측이 한 클래스에 쏠려 Top-1/Top-5가 거의 0으로 떨어질 수 있다.

DX-APP ResNet50 Python 예제에서 참고한 흐름은 다음이다.

- `cv2.imread()`로 BGR 이미지를 읽는다.
- `SimpleResizePreprocessor`가 BGR을 RGB로 바꾸고, aspect ratio 보존 없이 모델 입력 크기(`224x224`)로 direct resize한다.
- ImageNet mean/std 정규화는 하지 않는다.
- DXNN 입력 dtype이 `UINT8`이면 `uint8` tensor를 유지한다.
- 입력 shape이 NHWC이면 batch 차원 없는 `(H, W, C)` tensor를 `ie.run([input_tensor])`에 전달한다.
- `ClassificationPostprocessor`는 `outputs[0]`을 flatten한 뒤 softmax/top-k를 계산한다. class index 기준은 framework의 `ImageClassificationEvaluator`와 같다.

이 repo에서는 이 동작을 generic 이미지 로더나 공용 preprocessor에 섞지 않고 `DeepXDataLoader` 진입점 아래 task별 로더로 분리했다. 구조는 DX-APP의 `task -> model family -> common runner` 방향을 따른다.

| 파일 | 역할 |
|---|---|
| `framework/src/dataloader/deepx_loader.py` | `backend=deepx` 진입점, `Task`별 concrete loader 내부 라우팅 |
| `framework/src/dataloader/deepx_image_classification_loader.py` | DXNN `rmap_info`/`compile_config` 파싱, DeepX 이미지 전처리 선택, runtime input option 생성 |
| `framework/src/dataloader/deepx_vision_loader.py` | YOLO 계열 object detection / instance segmentation / pose estimation용 DX-APP 호환 letterbox 입력 생성 |
| `framework/src/dataloader/__init__.py` | `backend=deepx`인 모든 지원 task를 `DeepXDataLoader`로 라우팅 |
| `framework/src/runtimes/deepx_rt.py` | DataLoader가 만든 runtime option에 따라 `NHWC`, `uint8`, batch-axis squeeze, `run([tensor])` 호출 지원 |
| `framework/src/evaluators/image_classification_evaluator.py` | DeepX 출력 logits를 기존 이미지 분류 metric으로 평가 |
| `framework/src/evaluators/latency_evaluator.py` | instance segmentation / pose estimation의 정확도 metric 연결 전 latency-only 실행 지원 |

DeepX 전용 로더의 자동 처리 규칙:

| DXNN metadata | Loader 처리 | Runtime option |
|---|---|---|
| `dtype=UINT8`, `shape=[1,H,W,C]` | DX-APP 호환 direct resize raw RGB | `input_layout=NHWC`, `input_dtype=uint8`, `input_batch_axis=squeeze`, `single_input_run_style=list` |
| compile config에 `div`/`normalize` 포함 | resize/crop만 수행하는 raw pixel mode | graph-side 전처리 기대 |
| 위 조건 없음 | 일반 ResNet50 normalized mode | loader-side ImageNet normalize |

YOLO 계열 DeepX vision 로더의 처리 규칙:

- `OBJECT_DETECTION`, `INSTANCE_SEGMENTATION`, `POSE_ESTIMATION`은 `DeepXDataLoader` 아래에서 `DeepXObjectDetectionLoader`, `DeepXInstanceSegmentationLoader`, `DeepXPoseEstimationLoader`로 라우팅된다.
- 세 로더 모두 DX-APP의 `LetterboxPreprocessor`와 같은 방식으로 aspect ratio를 유지하고 `(114,114,114)` padding을 적용한다.
- DXNN rmap metadata가 `UINT8/NHWC`이면 batch 없는 `HWC uint8` 샘플을 만들고 runtime에는 `input_batch_axis=squeeze` 옵션을 전달한다.
- rmap metadata가 `FLOAT/FLOAT32`이면 DX-APP runner와 맞춰 letterbox 결과를 `float32` `0..1` 범위로 만든다.
- `yolov8m`, `yolov8m-seg`/`yolov8-seg-m`, `yolov8m-pose`/`yolov8-pose-m` 프로필이 준비되어 있다. seg/pose는 현재 정확도 metric 대신 latency-only evaluator로 끝까지 실행된다.

따라서 사전컴파일 `.dxnn`을 실행할 때는 `input_layout`, `input_dtype`, `input_batch_axis`, `single_input_run_style`을 CLI에서 직접 지정하지 않는 것을 권장한다. 필요한 값은 DeepX DataLoader가 DXNN metadata를 읽어 runtime에 전달한다.

ResNet50/ImageNet 빠른 확인 명령:

```bash
cd /home/swlab-jetson/ML-HW-Benchmark-Framework/framework

python src/main.py \
  --model resnet50 \
  --target deepx \
  --no-compile \
  --artifact models/deepx/ResNet50.dxnn \
  --dataset datasets/imagenet_1k \
  --layout NCHW \
  --runtime-option device_ids=0 \
  --runtime-option bound_option=NPU_ALL \
  --batch-size 1 \
  --warmup 2 \
  --max-steps 100 \
  --monitor \
  --debug
```

정상 동작 시 로그에는 DeepX 전용 로더가 선택한 전처리와 runtime input option이 출력된다.

compile mode API 요청은 `compile_options.config_path`를 포함한다.

```json
{
  "model": "resnet50",
  "target_id": "deepx",
  "backend": "deepx",
  "device": "npu0",
  "compile": true,
  "compile_options": {
    "config_path": "/path/to/resnet50_config.json",
    "opt_level": "1",
    "gen_log": "true"
  },
  "monitor": true
}
```

## 5. DX-APP 설치와 검증

DX-APP은 프레임워크 실행에 필수는 아니지만, driver/DX-RT 설치 후 NPU가 실제 예제 앱에서 동작하는지 확인하는 데 유용하다.

```bash
cd /path/to/dx_app

# 먼저 NPU와 DX-RT 상태 확인
dxrt-cli -s

# build tools, CMake, OpenCV 등 설치
./install.sh --all

# 모델과 sample media 준비
./setup.sh --all

# C++ binary와 Python postprocess binding build
./build.sh
```

예제 실행:

```bash
./bin/yolov9s_sync \
  -m assets/models/YoloV9S.dxnn \
  -i sample/img/sample_kitchen.jpg

python src/python_example/object_detection/yolov9s/yolov9s_sync.py \
  --model assets/models/YoloV9S.dxnn \
  --image sample/img/sample_kitchen.jpg
```

## 6. DX-STREAM 설치와 검증

DX-STREAM은 GStreamer pipeline에서 DEEPX NPU inference를 쓰기 위한 선택 구성이다. 프레임워크의 `deepx` target과 직접 연결되지는 않는다.

```bash
git clone https://github.com/DEEPX-AI/dx_stream.git
cd dx_stream

# dependencies 설치
./install.sh

# 기본 prefix는 /usr/local
./build.sh

# 현재 shell에 환경변수 반영
source ~/.bashrc

# GStreamer plugin 인식 확인
gst-inspect-1.0 dxstream
```

custom prefix를 쓰면 `PKG_CONFIG_PATH`, `GST_PLUGIN_PATH`, `LD_LIBRARY_PATH`, `PATH`에 설치 경로를 반영해야 한다.

```bash
export PKG_CONFIG_PATH="/path/to/install/lib/pkgconfig:${PKG_CONFIG_PATH}"
export GST_PLUGIN_PATH="/path/to/install/lib/x86_64-linux-gnu/gstreamer-1.0:${GST_PLUGIN_PATH}"
export LD_LIBRARY_PATH="/path/to/install/lib/x86_64-linux-gnu/gstreamer-1.0:/path/to/install/share/gstdxstream/lib:${LD_LIBRARY_PATH}"
export PATH="/path/to/install/share/gstdxstream/bin:${PATH}"
```

demo 실행:

```bash
./setup.sh
./run_demo.sh
```

plugin이 보이지 않으면 GStreamer cache를 지우고 다시 확인한다.

```bash
rm -rf ~/.cache/gstreamer-1.0/
gst-inspect-1.0 dxstream
```

## 7. 문제 해결 체크리스트

| 증상 | 확인할 것 |
|---|---|
| `dxcom: command not found` | DX-COM wheel이 framework venv에 설치됐는지, `dxcom_bin`을 지정했는지 확인 |
| `DeepX compiler requires config_path` | compile mode에서 `--compile-option config_path=/path/config.json` 추가 |
| DX-COM input shape 오류 | JSON `inputs` 이름/shape와 ONNX graph 입력을 비교, HWC/NCHW transpose 확인 |
| `import dx_engine` 실패 | target 장비의 framework venv에 DX-RT Python package 설치 |
| `/dev/dxrt0` 없음 | driver 설치, `sudo modprobe dx_dma`, `dkms status`, `SanityCheck.sh dx_driver` 확인 |
| driver build에서 kernel header 오류 | `uname -r`과 `/lib/modules/$(uname -r)/build`를 확인하고 현재 Jetson L4T kernel과 정확히 일치하는 header/source를 설치한 뒤 driver를 다시 빌드 |
| `dxrtd` 연결 실패 | `sudo systemctl status dxrt.service`, `sudo journalctl -u dxrt.service` 확인 |
| `dxrtd` bind/device busy | systemd service와 수동 `dxrtd`가 중복 실행되지 않았는지 `pgrep -a dxrtd`로 확인하고 하나만 유지 |
| restricted container에서 DX-COM multiprocessing 오류 | local socket/seccomp 제한이 없는 host shell 또는 권한이 허용된 container에서 컴파일 |
| `async_queue`가 native 경로를 선택하지 않음 | 같은 venv에서 `InferenceEngine.run_async`/`register_callback` 존재 여부와 callback 등록 가능한 DX-RT 버전인지 확인 |
| native async에서 batch 오류 | DX-RT v3.3은 async batch를 지원하지 않으므로 `--batch-size 1` 사용 |
| native async timeout | `dxrt-cli -s`, worker/buffer 과부하를 확인하고 필요 시 `async_completion_timeout_sec` 조정 |

## 8. 현재 repo 기준 빠른 확인 명령

```bash
cd /home/swlab-youngjin/ML-HW-Benchmark-Framework

# target registry
framework/.venv/bin/python -c "import sys; sys.path.insert(0, 'framework/src'); from core.targets import get_target; print(get_target('deepx'))"

# compiler registry
framework/.venv/bin/python -c "import sys; sys.path.insert(0, 'framework/src'); from compilers import list_compilers; print(list_compilers())"

# DX-COM
framework/.venv/bin/dxcom --version
framework/.venv/bin/python -c "import dx_com; print(dx_com.__version__)"

# DX-RT Python runtime
framework/.venv/bin/python -c "import dx_engine; print(dx_engine.__version__)"
```
