"""웹 요청을 쉘을 통하지 않는 프레임워크 CLI argv로 변환한다."""

from ..schemas.benchmark import BenchmarkRunRequest


VALUE_FIELDS = (
    ("onnx_path", "--onnx"),
    ("hef_path", "--hef"),
    ("artifact_path", "--artifact"),
    ("mobilint_bert_weights", "--mobilint-bert-weights"),
    ("fxb_path", "--fxb"),
    ("model_path", "--model-path"),
    ("tokenizer_path", "--tokenizer-path"),
    ("dataset_path", "--dataset"),
    ("image_dir", "--image-dir"),
    ("label_dir", "--label-dir"),
    ("layout", "--layout"),
    ("image_preprocess_mode", "--image-preprocess-mode"),
    ("image_preprocess_profile", "--image-preprocess-profile"),
    ("image_resize_mode", "--image-resize-mode"),
    ("batch_size", "--batch-size"),
    ("warmup", "--warmup"),
    ("max_steps", "--max-steps"),
    ("max_new_tokens", "--max-new-tokens"),
    ("max_model_len", "--max-model-len"),
    ("gpu_memory_utilization", "--gpu-memory-utilization"),
    ("results_path", "--results-path"),
    ("monitor_interval", "--monitor-interval"),
)

ASYNC_VALUE_FIELDS = (
    ("inference_mode", "--inference-mode"),
    ("scenario", "--scenario"),
    ("async_pass", "--async-pass"),
    ("target_qps", "--target-qps"),
    ("queue_capacity", "--queue-capacity"),
    ("worker_count", "--worker-count"),
    ("batch_timeout_ms", "--batch-timeout-ms"),
    ("submit_timeout_sec", "--submit-timeout-sec"),
    ("flush_timeout_sec", "--flush-timeout-sec"),
    ("request_timeout_ms", "--request-timeout-ms"),
    ("min_samples", "--min-samples"),
    ("min_duration_sec", "--min-duration-sec"),
    ("max_samples", "--max-samples"),
    ("schedule_seed", "--schedule-seed"),
    ("latency_slo_ms", "--latency-slo-ms"),
)

SUPPORTED_CLI_FLAGS = (
    "--model", "--target", "--backend", "--device",
    *(flag for _, flag in VALUE_FIELDS),
    "--compile", "--no-compile", "--compile-option", "--runtime-option",
    "--enforce-eager", "--debug", "--monitor",
    *(flag for _, flag in ASYNC_VALUE_FIELDS),
    "--save-request-trace", "--async-preload-inputs",
)


def build_cli_args(request: BenchmarkRunRequest) -> list[str]:
    args = ["--model", request.model]
    if request.target_id:
        args.extend(("--target", request.target_id))
    else:
        for value, flag in ((request.backend, "--backend"), (request.device, "--device")):
            if value is not None:
                args.extend((flag, value))

    for field, flag in VALUE_FIELDS:
        value = getattr(request, field)
        if value is not None:
            args.extend((flag, str(value)))

    if request.debug:
        args.append("--debug")
    if request.monitor:
        args.append("--monitor")
    if request.compile is not None:
        args.append("--compile" if request.compile else "--no-compile")
    for key, value in request.compile_options.items():
        args.extend(("--compile-option", f"{key}={value}"))
    if request.enforce_eager:
        args.append("--enforce-eager")
    for key, value in request.runtime_options.items():
        args.extend(("--runtime-option", f"{key}={value}"))

    for field, flag in ASYNC_VALUE_FIELDS:
        value = getattr(request, field)
        if value is not None:
            args.extend((flag, str(value)))
    if request.save_request_trace:
        args.append("--save-request-trace")
    if request.async_preload_inputs:
        args.append("--async-preload-inputs")
    return args
