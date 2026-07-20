import time
from threading import Lock

from .async_inference.completion import CompletionCoordinator
from .async_inference.types import BatchCompletion, InferenceRequest
from .inference_pipeline import InferencePipeline
from .runtime_executor import RuntimeExecutor


class _InlineCompletionMetrics:
    def record_generation(self, generated_tokens, timing_ms):
        del generated_tokens, timing_ms

    def record_terminal(self, trace):
        del trace

    def add_invalid_reason(self, reason):
        del reason

    def add_warning(self, warning):
        del warning


class InferenceEngine:
    def __init__(
        self,
        dataloader,
        runtime,
        evaluator,
        *,
        decoder=None,
        max_new_tokens: int = 256,
        runtime_executor: RuntimeExecutor | None = None,
        trace_callback=None,
        lifecycle_callback=None,
    ):
        self.dataloader = dataloader
        self.runtime = runtime
        self.evaluator = evaluator
        self.decoder = decoder
        self.trace_callback = trace_callback
        self.lifecycle_callback = lifecycle_callback
        self.max_new_tokens = max_new_tokens
        self._pipeline = None
        self._pipeline_lock = Lock()
        self._runtime_executor = runtime_executor
        self.completion = None
        self._run_claim_lock = Lock()
        self._run_mode = None
        self._async_controller = None

    @property
    def pipeline(self):
        with self._pipeline_lock:
            if self._pipeline is None:
                self._pipeline = InferencePipeline(
                    self.dataloader,
                    self.runtime,
                    max_new_tokens=self.max_new_tokens,
                    runtime_executor=self._runtime_executor,
                )
                self._runtime_executor = self._pipeline._compat_executor
            return self._pipeline

    @property
    def runtime_executor(self):
        if self._runtime_executor is None:
            return self.pipeline._compat_executor
        return self._runtime_executor

    @runtime_executor.setter
    def runtime_executor(self, executor):
        self._runtime_executor = executor
        if self._pipeline is not None:
            self._pipeline._compat_executor = executor

    def _prepare_async_diagnostics(self, controller):
        with self._run_claim_lock:
            if self._run_mode is None:
                self._async_controller = controller

    def _claim_run(self, mode, *, async_controller=None):
        with self._run_claim_lock:
            if self._run_mode is not None:
                if mode == "e2e" and self._run_mode == "e2e":
                    raise RuntimeError("run_e2e() may only be called once")
                raise RuntimeError("InferenceEngine may only be run once")
            self._run_mode = mode
            if async_controller is not None:
                self._async_controller = async_controller

    def warmup(self, runs, batch_size):
        try:
            if runs <= 0:
                return
            batch = self.dataloader.load_batch(batch_size)
            if not batch:
                return
            collated = self.pipeline.collate_batch(batch)
            runtime_input = self.pipeline.prepare_runtime_input(
                collated["input"]
            )
            self.runtime.warmup(runtime_input, num_runs=runs)
        finally:
            self.pipeline.reset_dataloader_cursor()

    def run_e2e(
        self,
        batch_size=1,
        max_steps=None,
        event_callback=None,
    ):
        self._claim_run("e2e")

        def emit(event, **details):
            if event_callback is not None:
                event_callback(event, **details)

        if self.completion is None:
            self.completion = CompletionCoordinator(
                pipeline=self.pipeline,
                evaluator=self.evaluator,
                decoder=self.decoder,
                metrics=_InlineCompletionMetrics(),
                queue_capacity=None,
                trace_callback=self.trace_callback,
                raise_callback_errors=True,
            )
        self.completion.start()

        request_id = 0
        sample_index = 0
        steps = 0
        try:
            while True:
                if max_steps is not None and steps >= max_steps:
                    emit("limit_reached", max_steps=max_steps)
                    break

                batch = self.dataloader.load_batch(batch_size)
                if not batch:
                    break

                collated = self.pipeline.collate_batch(batch)
                actual_batch_size = self.pipeline.batch_size(collated)
                issued_ns = time.monotonic_ns()
                request = InferenceRequest(
                    request_id=request_id,
                    sample_index=sample_index,
                    sample=collated,
                    scheduled_ns=issued_ns,
                    issued_ns=issued_ns,
                    enqueued_ns=issued_ns,
                    sample_count=actual_batch_size,
                )
                self.completion.register(request)
                runtime_input = self.pipeline.prepare_runtime_input(
                    collated["input"]
                )
                runtime_started_ns = time.monotonic_ns()
                try:
                    execution = self.runtime_executor.execute(runtime_input)
                except BaseException as primary:
                    runtime_finished_ns = max(
                        time.monotonic_ns(),
                        runtime_started_ns,
                    )
                    self.completion.submit(
                        BatchCompletion(
                            requests=[request],
                            collated=collated,
                            outputs=None,
                            timing_ms=None,
                            runtime_started_ns=runtime_started_ns,
                            runtime_finished_ns=runtime_finished_ns,
                            worker_id=0,
                            batch_size=actual_batch_size,
                            error_type=type(primary).__name__,
                            error_message=str(primary),
                        )
                    )
                    raise
                runtime_finished_ns = max(
                    time.monotonic_ns(),
                    runtime_started_ns,
                )
                completed = BatchCompletion(
                    requests=[request],
                    collated=collated,
                    outputs=execution.outputs,
                    timing_ms=execution.timing_ms,
                    runtime_started_ns=runtime_started_ns,
                    runtime_finished_ns=runtime_finished_ns,
                    worker_id=0,
                    batch_size=actual_batch_size,
                    generated_tokens=execution.generated_tokens,
                    error_type=execution.error_type,
                    error_message=execution.error_message,
                )
                try:
                    self.completion.submit(completed)
                except BaseException as primary:
                    if request_id not in self.completion.snapshot_outstanding():
                        try:
                            self.runtime_executor.acknowledge(execution)
                        except BaseException:
                            raise primary
                    raise
                else:
                    self.runtime_executor.acknowledge(execution)

                request_id += 1
                sample_index += actual_batch_size
                steps += 1
                emit(
                    "batch_complete",
                    batch_idx=steps,
                    actual_batch_size=actual_batch_size,
                    timing_ms=execution.timing_ms,
                )
        finally:
            self.completion.stop(timeout=0.0)

        emit("before_compute")
        return self.evaluator.compute()

    def run_async(self, config, warmup_runs=1, monitor=None):
        from .async_inference.runner import _AsyncRunController
        from .async_inference.metrics import AsyncMetricsCollector

        controller = _AsyncRunController(
            self.dataloader,
            self.runtime,
            self.evaluator,
            max_new_tokens=self.max_new_tokens,
            monitor=monitor,
            decoder=self.decoder,
            trace_callback=self.trace_callback,
            lifecycle_callback=self.lifecycle_callback,
            runtime_executor=self._runtime_executor,
        )
        self._prepare_async_diagnostics(controller)
        controller.validate(config, warmup_runs)
        self._claim_run("async", async_controller=controller)

        pipeline = self.pipeline
        runtime_executor = self.runtime_executor
        metrics = AsyncMetricsCollector(
            time.monotonic_ns(),
            config.worker_count,
            latency_slo_ms=config.latency_slo_ms,
        )
        self.completion = CompletionCoordinator(
            pipeline=pipeline,
            evaluator=self.evaluator,
            decoder=self.decoder,
            metrics=metrics,
            queue_capacity=config.worker_count,
            request_timeout_ms=config.request_timeout_ms,
            trace_callback=self.trace_callback,
        )

        controller.bind_async_resources(
            pipeline=pipeline,
            runtime_executor=runtime_executor,
            metrics=metrics,
            completion=self.completion,
        )
        return controller.run(config, warmup_runs=warmup_runs)
