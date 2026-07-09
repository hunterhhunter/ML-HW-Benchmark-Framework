import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Any

# 프로젝트 루트 경로 추가 (sys.path)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.model_spec import Model_Spec, Task
from core.model_profiles import create_model_spec
from core.compiled_model import CompiledModel
from core.benchmarkrunner import BenchmarkRunner
from core.result_store import save_result
from core.targets import resolve_target, target_metadata

# 구체화된 컴포넌트 임포트 (Facade Pattern 적용)
from dataloader import create_dataloader
from evaluators import create_evaluator
from runtimes import create_runtime
from compilers import get_compiler, normalize_compile_result
# from src.runtimes.iree_rt import IREERuntime  # 향후 IREE 백엔드 추가 시 주석 해제

def run_auto_prepare(profile: dict, args: argparse.Namespace, target=None):
    """
    Zero-Config 벤치마크를 위해 누락된 리소스를 감지하고 백그라운드 준비 스크립트를 자동 실행합니다.
    """
    if args.backend == "vllm":
        model_path = args.model_path
    elif args.backend == "hailort":
        model_path = args.hef or args.artifact
    elif (
        target is not None
        and target.uses_compiler
        and not args.compile
        and target.artifact_format not in ("onnx", "hf_model")
    ):
        model_path = args.artifact
    elif target is not None and not target.uses_compiler and target.artifact_format not in ("onnx", "hf_model"):
        model_path = args.artifact
    else:
        model_path = args.onnx
    dataset_path = args.dataset

    can_auto_prepare_model = (
        args.backend != "hailort"
        and not (
            target is not None
            and target.uses_compiler
            and not args.compile
            and target.artifact_format not in ("onnx", "hf_model")
        )
        and not (target is not None and not target.uses_compiler and target.artifact_format not in ("onnx", "hf_model"))
    )
    if can_auto_prepare_model and "prepare_model_script" in profile and profile["prepare_model_script"]:
        if not model_path or not os.path.exists(model_path):
            script = profile["prepare_model_script"]
            print(f"[*] 모델 리소스 누락 감지. 자동 준비 스크립트 실행: {script}")
            subprocess.run([sys.executable, script], check=True)
            
    if "prepare_dataset_script" in profile and profile["prepare_dataset_script"]:
        if not dataset_path or not os.path.exists(dataset_path):
            script = profile["prepare_dataset_script"]
            print(f"[*] 데이터셋 리소스 누락 감지. 자동 준비 스크립트 실행: {script}")
            subprocess.run([sys.executable, script], check=True)


def _coerce_option_value(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_key_value_options(items: list[str] | None, *, coerce_values: bool = False) -> dict:
    """CLI의 key=value 리스트를 딕셔너리로 변환한다."""
    options = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"옵션은 key=value 형식이어야 합니다: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"옵션 key가 비어 있습니다: {item}")
        options[key] = _coerce_option_value(value) if coerce_values else value
    return options


def main():
    parser = argparse.ArgumentParser(description="Unified BenchmarkRunner CLI Orchestrator")
    parser.add_argument("--model", type=str, required=True, help="모델 이름 (예: resnet50, llama-3.2-3b)")
    parser.add_argument("--onnx", type=str, default=None, help="ONNX 파일의 절대 또는 상대 경로 (onnxruntime 백엔드 필수)")
    parser.add_argument("--hef", type=str, default=None, help="HailoRT 실행용 HEF 파일 경로 (hailo8 target 필수)")
    parser.add_argument("--artifact", type=str, default=None, help="target 전용 사전 컴파일 artifact 경로 (예: DEEPX .dxnn)")
    parser.add_argument("--model-path", type=str, default=None, help="HuggingFace 모델 디렉토리 경로 (vLLM 백엔드 필수)")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="HuggingFace 토크나이저 디렉토리 경로 (NLP 모델 필수)")
    parser.add_argument("--dataset", type=str, default=None, help="평가용 데이터셋 최상위 디렉토리 또는 CSV 파일 경로")
    parser.add_argument("--image-dir", type=str, default="", help="(옵션) 데이터셋 내 이미지 하위 폴더 경로")
    parser.add_argument("--label-dir", type=str, default="", help="(옵션) 데이터셋 내 라벨 하위 폴더 경로")
    parser.add_argument("--layout", type=str, default="NCHW", choices=["NCHW", "NHWC"], help="모델 텐서 레이아웃 (기본: NCHW)")
    parser.add_argument("--image-preprocess-mode", type=str, default="auto", choices=["auto", "normalized", "raw"], help="이미지 분류 전처리 모드. raw는 resize/crop 후 0..255 픽셀을 전달합니다.")
    parser.add_argument("--target", type=str, default=None, help="실행 target_id (예: cpu, cuda, vendor_mock_npu). 지정 시 backend/device보다 우선합니다.")
    parser.add_argument("--backend", type=str, default="onnxruntime", choices=["onnxruntime", "iree", "vllm", "hailort", "deepx"], help="추론을 실행할 백엔드 (기본: onnxruntime)")
    parser.add_argument("--device", type=str, default="cpu", help="추론 장치 (예: cpu, cuda, 기본: cpu)")
    parser.add_argument("--compile", dest="compile", action="store_true", default=True, help="target에 compiler가 있으면 컴파일을 수행합니다.")
    parser.add_argument("--no-compile", dest="compile", action="store_false", help="target compiler를 사용하지 않고 원본 artifact를 runtime에 전달합니다.")
    parser.add_argument("--compile-option", action="append", default=[], help="벤더 compiler 옵션 key=value. 여러 번 지정 가능.")
    parser.add_argument("--runtime-option", action="append", default=[], help="런타임 옵션 key=value. 여러 번 지정 가능 (예: output_format_type=uint8).")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="추론 배치 사이즈 (기본: 1)")
    parser.add_argument("--warmup", "-w", type=int, default=2, help="웜업 횟수 (기본: 2)")
    parser.add_argument("--max-steps", type=int, default=None, help="시간이 지루할 때 쓸 강제 종료 리미트 (옵션)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="LLM 생성 최대 토큰 수 (기본: 256)")
    parser.add_argument("--max-model-len", type=int, default=None, help="vLLM 최대 컨텍스트 길이 (기본: 모델 기본값, 메모리 부족 시 줄이세요)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="vLLM GPU 메모리 사용률 0.0~1.0 (기본: 0.90, OOM 시 낮추세요 예: 0.7)")
    parser.add_argument("--enforce-eager", action="store_true", default=None, help="vLLM CUDA 그래프 캡처 비활성화 (메모리 부족 시 사용)")
    parser.add_argument("--debug", action="store_true", help="샘플별 예측/정답/점수 로그 출력 (기본: 비활성)")
    parser.add_argument("--monitor", action="store_true", help="벤치마크 중 하드웨어 모니터링 활성화 (GPU/CPU/RAM)")
    parser.add_argument("--monitor-interval", type=float, default=0.2, help="모니터링 샘플링 간격 초 (기본: 0.2)")
    
    args = parser.parse_args()
    device_was_default = args.device == parser.get_default("device")
    layout_was_default = args.layout == parser.get_default("layout")
    
    # [설계 개선] CLI 인자(--task)에 의존하지 않고, 레지스트리(SUPPORTED_PROFILES)에서 태스크를 자동 추론 (DRY 원칙)
    from core.model_profiles import SUPPORTED_PROFILES
    profile = SUPPORTED_PROFILES.get(args.model)
    if not profile:
        print(f"[Error] '{args.model}' 프로필을 찾을 수 없습니다. model_profiles.py에 등록되었는지 확인하세요.")
        sys.exit(1)

    try:
        target = resolve_target(args.target, args.backend, args.device)
        # --target이 지정되면 target registry의 runtime/device가 실행 기준이다.
        if args.target:
            args.backend = target.runtime_name
            args.device = target.device
        elif target.target_id == "hailo8" and device_was_default:
            args.device = target.device
        if target.target_id == "hailo8" and layout_was_default:
            args.layout = "NHWC"
    except Exception as e:
        print(f"[Error] target 해석 실패: {e}")
        sys.exit(1)

    try:
        cli_compile_options = parse_key_value_options(args.compile_option)
        cli_runtime_options = parse_key_value_options(args.runtime_option, coerce_values=True)
    except ValueError as e:
        print(f"[Error] 옵션 파싱 실패: {e}")
        sys.exit(1)

    compile_options = {**target.compiler_options, **cli_compile_options}

        
    # 누락된 인자(default) 주입 (Zero-Config)
    # onnx 경로: default_onnx_path가 있으면 우선 사용, 없으면 default_model_path로 fallback
    if args.onnx is None:
        if "default_onnx_path" in profile:
            args.onnx = profile["default_onnx_path"]
        elif "default_model_path" in profile:
            args.onnx = profile["default_model_path"]
    # vllm 모델 경로: 항상 default_model_path (safetensors 폴더)
    if args.model_path is None and "default_model_path" in profile:
        args.model_path = profile["default_model_path"]
    if args.dataset is None and "default_dataset_path" in profile:
        args.dataset = profile["default_dataset_path"]
        
    # 토크나이저 경로 자동 추론 (NLP 태스크용)
    if args.tokenizer_path is None:
        if args.backend == "vllm" and args.model_path:
            args.tokenizer_path = args.model_path
        elif args.onnx:
            # ONNX 파일 경로면 부모 디렉토리를 토크나이저 경로로 간주
            args.tokenizer_path = os.path.dirname(args.onnx) if args.onnx.endswith(".onnx") else args.onnx

    # 사전 컴파일 artifact target은 모델 자동 다운로드보다 artifact 경로 검증이 먼저다.
    if args.backend == "hailort":
        args.hef = args.hef or args.artifact
        if not args.hef:
            print("[Error] hailort 백엔드에는 --hef 또는 --artifact 경로가 필요합니다.")
            sys.exit(1)
        if not os.path.exists(args.hef):
            print(f"[Error] HEF 파일을 찾을 수 없습니다: {args.hef}")
            sys.exit(1)
    elif target.uses_compiler and not args.compile and target.artifact_format not in ("onnx", "hf_model"):
        if not args.artifact:
            print(f"[Error] target '{target.target_id}'에서 --no-compile을 사용하려면 --artifact 경로가 필요합니다. "
                  f"(artifact_format={target.artifact_format})")
            sys.exit(1)
        if not os.path.exists(args.artifact):
            print(f"[Error] artifact 파일을 찾을 수 없습니다: {args.artifact}")
            sys.exit(1)
    elif not target.uses_compiler and target.artifact_format not in ("onnx", "hf_model"):
        if not args.artifact:
            print(f"[Error] target '{target.target_id}'에는 --artifact 경로가 필요합니다. "
                  f"(artifact_format={target.artifact_format})")
            sys.exit(1)
        if not os.path.exists(args.artifact):
            print(f"[Error] artifact 파일을 찾을 수 없습니다: {args.artifact}")
            sys.exit(1)

    # 리소스 누락 시 백그라운드 준비 스크립트 실행 (Auto-Prepare)
    run_auto_prepare(profile, args, target)
    
    # 백엔드별 필수 인자 검증
    if args.backend == "vllm":
        if not args.model_path:
            print("[Error] vllm 백엔드에는 --model-path가 필요합니다.")
            sys.exit(1)
    elif args.backend == "hailort":
        pass
    elif target.uses_compiler and not args.compile and target.artifact_format not in ("onnx", "hf_model"):
        if args.onnx and os.path.exists(args.onnx) and os.path.isdir(args.onnx):
            candidate = os.path.join(args.onnx, "model.onnx")
            if os.path.exists(candidate):
                print(f"[Info] --onnx에 디렉토리가 지정되었습니다. {candidate} 를 스펙 파싱에 사용합니다.")
                args.onnx = candidate
    elif not target.uses_compiler and target.artifact_format not in ("onnx", "hf_model"):
        pass
    else:
        if not args.onnx:
            print("[Error] onnxruntime/iree 또는 compile target에는 --onnx가 필요합니다.")
            sys.exit(1)
        if not os.path.exists(args.onnx):
            print(f"[Error] 모델 파일을 찾을 수 없습니다: {args.onnx}")
            sys.exit(1)
        # 디렉토리가 넘어온 경우 model.onnx 자동 탐색 (HuggingFace 다운로드 폴더 구조 대응)
        if os.path.isdir(args.onnx):
            candidate = os.path.join(args.onnx, "model.onnx")
            if os.path.exists(candidate):
                print(f"[Info] --onnx에 디렉토리가 지정되었습니다. {candidate} 를 사용합니다.")
                args.onnx = candidate
            else:
                print(f"[Error] 디렉토리 {args.onnx} 에서 model.onnx를 찾을 수 없습니다.")
                sys.exit(1)
    
    task_enum = profile["task"]

    # 백엔드-태스크 호환성 검증: vllm은 NLP_GENERATION 전용
    if args.backend == "vllm" and task_enum != Task.NLP_GENERATION:
        print(f"[Error] vllm 백엔드는 NLP_GENERATION 태스크만 지원합니다. "
              f"모델 '{args.model}'의 태스크는 {task_enum.name}입니다. "
              f"onnxruntime 백엔드를 사용하세요: --backend onnxruntime")
        sys.exit(1)

    print("\n" + "="*60)
    print(f" BenchmarkRunner CLI ")
    print(f"   Model: {args.model} | Task: {task_enum.name} | Layout: {args.layout}")
    print(f"   Target: {target.target_id} | Runtime: {args.backend} | Device: {args.device}")
    print("="*60)
    
    # 0. DataLoader 공통 인터페이스 규약 및 CoC 해소 (Resolver)
    from utils.dataset_resolver import resolve_dataset_paths
    image_dir, label_path = resolve_dataset_paths(task_enum, args.dataset, args.image_dir, args.label_dir)
    
    loader_kwargs = {}
    if image_dir:
        loader_kwargs["image_dir"] = image_dir
    if label_path:
        loader_kwargs["label_path"] = label_path
    
    # 1. Spec & source artifact 생성
    if args.backend == "vllm":
        source_artifact_path = Path(args.model_path)
        spec_source_format = "hf_model"
        sniff_onnx = False
    elif args.backend == "hailort":
        source_artifact_path = Path(args.hef)
        spec_source_format = "hef"
        sniff_onnx = False
    elif target.uses_compiler and not args.compile and target.artifact_format not in ("onnx", "hf_model"):
        if args.onnx and os.path.exists(args.onnx):
            source_artifact_path = Path(args.onnx)
            spec_source_format = "onnx"
            sniff_onnx = True
        else:
            source_artifact_path = Path(args.artifact)
            spec_source_format = target.artifact_format
            sniff_onnx = False
    elif not target.uses_compiler and target.artifact_format not in ("onnx", "hf_model"):
        source_artifact_path = Path(args.artifact)
        spec_source_format = target.artifact_format
        sniff_onnx = False
    else:
        source_artifact_path = Path(args.onnx)
        spec_source_format = "onnx"
        sniff_onnx = True
    try:
        spec = create_model_spec(
            args.model,
            str(source_artifact_path),
            task=task_enum,
            sniff_onnx=sniff_onnx,
            source_format=spec_source_format,
        )
    except Exception as e:
        print(f"[Error] 스펙 파싱 실패: {e}")
        sys.exit(1)

    compile_metadata = {}
    artifact_path = (
        Path(args.artifact)
        if target.compiler_name and not args.compile and args.artifact
        else source_artifact_path
    )
    if target.compiler_name and args.compile:
        try:
            compiler = get_compiler(target.compiler_name, **compile_options)
            compile_dir = Path("artifacts") / target.target_id
            compile_result = normalize_compile_result(compiler.compile(spec, str(compile_dir)))
            artifact_path = Path(compile_result.artifact_path)
            compile_metadata = compile_result.metadata
            print(f"[Compiler] target={target.target_id} artifact={artifact_path}")
        except Exception as e:
            print(f"[Error] 컴파일 실패: {e}")
            sys.exit(1)
    elif target.compiler_name and not args.compile:
        print(f"[Compiler] --no-compile 지정됨. 원본 artifact를 runtime에 전달합니다: {artifact_path}")

    compiled_model = CompiledModel(spec=spec, backend_name=args.backend, artifact_path=artifact_path)
    
    # 2. 컴포넌트(주입 객체) 조립
    print(f"[Factory] Assembling components for {task_enum.name}...")
    # NLP_GENERATION: tokenizer_path 전달, TIME_SERIES_FORECASTING: csv_path로 dataset 직접 전달
    if task_enum == Task.NLP_GENERATION and args.tokenizer_path:
        loader_kwargs["tokenizer_path"] = args.tokenizer_path
    if task_enum == Task.TIME_SERIES_FORECASTING:
        loader_kwargs["csv_path"] = args.dataset
        # csv_path 옆 .cache_npz 폴더를 캐시 디렉토리로 자동 지정
        csv_dir = os.path.dirname(os.path.abspath(args.dataset))
        loader_kwargs["cache_dir"] = os.path.join(csv_dir, ".cache_npz")
    elif task_enum in (
        Task.IMAGE_CLASSIFICATION,
        Task.OBJECT_DETECTION,
        Task.INSTANCE_SEGMENTATION,
        Task.POSE_ESTIMATION,
    ):
        # 이미지 데이터셋 디렉토리 옆에 .cache_npz 자동 지정
        loader_kwargs["cache_dir"] = os.path.join(os.path.abspath(args.dataset), ".cache_npz")
    elif task_enum == Task.NLP_GENERATION:
        # val.json 옆 .cache_npz 자동 지정
        loader_kwargs["cache_dir"] = os.path.join(
            os.path.dirname(os.path.abspath(args.dataset)), ".cache_npz"
        )

    if args.backend == "deepx":
        loader_kwargs.update({
            "backend": "deepx",
            "artifact_path": str(artifact_path),
            "compile_options": compile_options,
            "compile_enabled": args.compile,
            "image_preprocess_mode": args.image_preprocess_mode,
        })
    elif args.backend == "hailort":
        loader_kwargs.update({
            "backend": "hailort",
            "image_preprocess_mode": args.image_preprocess_mode,
        })

    loader = create_dataloader(
        model_spec=spec,
        dataset_path=args.dataset,
        layout=args.layout,
        **loader_kwargs
    )
    
    # 런타임 팩토리 로직
    try:
        runtime_kwargs = dict(target.runtime_options)
        if args.max_model_len is not None:
            runtime_kwargs["max_model_len"] = args.max_model_len
        elif "default_max_model_len" in profile:
            runtime_kwargs["max_model_len"] = profile["default_max_model_len"]
        if args.gpu_memory_utilization is not None:
            runtime_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
        elif "default_gpu_memory_utilization" in profile:
            runtime_kwargs["gpu_memory_utilization"] = profile["default_gpu_memory_utilization"]
        if args.enforce_eager:
            runtime_kwargs["enforce_eager"] = True
        elif "default_enforce_eager" in profile:
            runtime_kwargs["enforce_eager"] = profile["default_enforce_eager"]
        if args.backend == "hailort" and "batch_size" not in cli_runtime_options:
            runtime_kwargs["batch_size"] = args.batch_size
        loader_runtime_options = loader.get_metadata().get("runtime_options", {})
        if isinstance(loader_runtime_options, dict) and loader_runtime_options:
            runtime_kwargs.update(loader_runtime_options)
            if args.backend == "deepx":
                print(f"[DeepX] Runtime input options from dataloader: {loader_runtime_options}")
        runtime_kwargs.update(cli_runtime_options)
        runtime = create_runtime(args.backend, device=args.device, **runtime_kwargs)
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)
        
    # 3. 하드웨어 모니터 생성 (모델 로드 전에 VRAM 베이스라인 캡처)
    hw_monitor = None
    if args.monitor:
        from monitors import create_hw_monitor
        hw_monitor = create_hw_monitor(
            interval=args.monitor_interval,
            device=args.device,
            collector_names=list(target.monitor_names),
            collector_options=target.monitor_options,
        )

    runtime.load(compiled_model)

    # 모델 로드 후 VRAM 스냅샷 (모델 VRAM = after_load - baseline)
    if hw_monitor:
        hw_monitor.record_after_load_vram()

    # 평가기 팩토리 로직
    evaluator_kwargs = {}
    if task_enum == Task.NLP_GENERATION and args.tokenizer_path:
        evaluator_kwargs["tokenizer_path"] = args.tokenizer_path
    if args.debug:
        evaluator_kwargs["debug"] = True
    if task_enum == Task.TIME_SERIES_FORECASTING:
        evaluator_kwargs["dataloader"] = loader
    evaluator = create_evaluator(spec, top_k=(1, 5), **evaluator_kwargs)

    # 4. 오케스트레이터 구동
    runner = BenchmarkRunner(
        dataloader=loader, runtime=runtime, evaluator=evaluator,
        max_new_tokens=args.max_new_tokens,
        monitor=hw_monitor,
    )
    results = runner.run(warmup_runs=args.warmup, batch_size=args.batch_size, max_steps=args.max_steps)
    
    # 5. 최종 결과 리포팅
    print("\n" + "="*40)
    print(f" Final Metrics ({args.model.upper()}) ")
    print("="*40)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("="*40)

    # 6. 결과를 CSV에 자동 저장
    target_meta = target_metadata(target, compile_metadata)
    run_id = save_result(
        metrics=results,
        model_name=args.model,
        task=task_enum.name,
        backend=args.backend,
        device=args.device,
        batch_size=args.batch_size,
        warmup_runs=args.warmup,
        max_steps=args.max_steps,
        target_id=target_meta["target_id"],
        accelerator_vendor=target_meta["accelerator_vendor"],
        accelerator_name=target_meta["accelerator_name"],
        runtime_name=target_meta["runtime_name"],
        compiler_name=target_meta["compiler_name"],
        artifact_format=target_meta["artifact_format"],
    )
    print(f"\n[ResultStore] 결과 저장 완료 (run_id: {run_id})")
    print(f"[ResultStore] 파일: results/benchmark_results.csv")
    # 기계 판독용 계약 (backend가 파싱). 포맷 변경 시 benchmark_service.py도 함께 수정.
    print(f"RUN_ID={run_id}", flush=True)

    runtime.unload()

if __name__ == "__main__":
    main()
