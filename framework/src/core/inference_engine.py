import time
from threading import Lock

from .async_inference.completion import (
    CompletionCoordinator,
    _safe_error_type_name,
)
from .async_inference.types import BatchCompletion, InferenceRequest
from .inference_pipeline import InferencePipeline
from .runtime_executor import RuntimeExecutionError, RuntimeExecutor


class _InlineCompletionMetrics:
    def record_generation(
        self,
        generated_tokens,
        timing_ms,
        *,
        observation=None,
        requests=(),
    ):
        del generated_tokens, timing_ms, observation, requests

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
        self._dataloader = dataloader
        self._runtime = runtime
        self._evaluator = evaluator
        self._decoder = decoder
        self._trace_callback = trace_callback
        self._lifecycle_callback = lifecycle_callback
        self._max_new_tokens = max_new_tokens
        self._pipeline = None
        self._pipeline_lock = Lock()
        self._runtime_executor = runtime_executor
        self._runtime_executor_explicit = runtime_executor is not None
        self.completion = None
        self._run_claim_lock = Lock()
        self._run_mode = None
        self._warmup_active = False
        self._async_controller = None

    @property
    def dataloader(self):
        return self._dataloader

    @dataloader.setter
    def dataloader(self, dataloader):
        self._set_dependency("dataloader", dataloader)

    @property
    def runtime(self):
        return self._runtime

    @runtime.setter
    def runtime(self, runtime):
        self._set_dependency("runtime", runtime)

    @property
    def evaluator(self):
        return self._evaluator

    @evaluator.setter
    def evaluator(self, evaluator):
        self._set_dependency("evaluator", evaluator)

    @property
    def decoder(self):
        return self._decoder

    @decoder.setter
    def decoder(self, decoder):
        self._set_dependency("decoder", decoder)

    @property
    def max_new_tokens(self):
        return self._max_new_tokens

    @max_new_tokens.setter
    def max_new_tokens(self, max_new_tokens):
        self._set_dependency("max_new_tokens", max_new_tokens)

    @property
    def trace_callback(self):
        return self._trace_callback

    @trace_callback.setter
    def trace_callback(self, trace_callback):
        self._set_dependency("trace_callback", trace_callback)

    @property
    def lifecycle_callback(self):
        return self._lifecycle_callback

    @lifecycle_callback.setter
    def lifecycle_callback(self, lifecycle_callback):
        self._set_dependency("lifecycle_callback", lifecycle_callback)

    @property
    def pipeline(self):
        with self._pipeline_lock:
            return self._materialize_pipeline_locked()

    def _materialize_pipeline_locked(self):
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
        with self._pipeline_lock:
            if self._runtime_executor is None:
                self._materialize_pipeline_locked()
            return self._runtime_executor

    @runtime_executor.setter
    def runtime_executor(self, executor):
        self._set_dependency("runtime_executor", executor)

    def _set_dependency(self, name, value):
        with self._run_claim_lock:
            if self._run_mode is not None:
                raise RuntimeError(
                    "InferenceEngine dependencies cannot be changed "
                    "after a run is claimed"
                )
            if self._warmup_active:
                raise RuntimeError(
                    "InferenceEngine dependencies cannot be changed "
                    "during active warmup"
                )
            with self._pipeline_lock:
                if name == "runtime_executor":
                    self._runtime_executor_explicit = value is not None
                    self._runtime_executor = value
                    if self._pipeline is not None:
                        if value is None:
                            self._pipeline = None
                        else:
                            self._pipeline._compat_executor = value
                    return

                setattr(self, f"_{name}", value)
                if name in {"dataloader", "runtime", "max_new_tokens"}:
                    self._pipeline = None
                    if not self._runtime_executor_explicit:
                        self._runtime_executor = None

    def _prepare_async_diagnostics(self, controller):
        with self._run_claim_lock:
            if self._run_mode is None:
                self._async_controller = controller

    @property
    def failure_phase(self) -> str:
        controller = self._async_controller
        return "created" if controller is None else controller.failure_phase

    @property
    def runtime_unload_safe_after_failure(self) -> bool:
        controller = self._async_controller
        if controller is None:
            return True
        return controller.runtime_unload_safe_after_failure

    def _claim_run(self, mode, *, async_controller=None):
        with self._run_claim_lock:
            if self._run_mode is not None:
                if mode == "e2e" and self._run_mode == "e2e":
                    raise RuntimeError("run_e2e() may only be called once")
                raise RuntimeError("InferenceEngine may only be run once")
            if self._warmup_active:
                raise RuntimeError(
                    "InferenceEngine run cannot be claimed during active warmup"
                )
            with self._pipeline_lock:
                self._run_mode = mode
                if async_controller is not None:
                    async_controller.bind_dependencies(
                        dataloader=self._dataloader,
                        runtime=self._runtime,
                        evaluator=self._evaluator,
                        max_new_tokens=self._max_new_tokens,
                        decoder=self._decoder,
                        trace_callback=self._trace_callback,
                        lifecycle_callback=self._lifecycle_callback,
                        runtime_executor=self._runtime_executor,
                    )
                    self._async_controller = async_controller

    def warmup(self, runs, batch_size):
        with self._run_claim_lock:
            if self._run_mode is not None:
                raise RuntimeError(
                    "InferenceEngine warmup cannot start after a run is claimed"
                )
            if self._warmup_active:
                raise RuntimeError(
                    "InferenceEngine warmup cannot start during active warmup"
                )
            self._warmup_active = True
        try:
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
        finally:
            with self._run_claim_lock:
                self._warmup_active = False

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
        primary = None
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
                runtime_input = self.pipeline.prepare_runtime_input(
                    collated["input"]
                )
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
                runtime_started_ns = time.monotonic_ns()
                try:
                    execution = self.runtime_executor.execute(runtime_input)
                except BaseException as execution_error:
                    runtime_finished_ns = max(
                        time.monotonic_ns(),
                        runtime_started_ns,
                    )
                    primary = execution_error
                    try:
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
                                error_type=_safe_error_type_name(
                                    execution_error
                                ),
                                error_message=execution_error,
                            )
                        )
                    except BaseException:
                        pass
                    break
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
                    generation_observation=(
                        execution.generation_observation
                    ),
                )
                execution_error = None
                if execution.error_type is not None:
                    execution_error = RuntimeExecutionError(
                        error_type=execution.error_type,
                        error_message=execution.error_message,
                        dispatch_token=execution.dispatch_token,
                    )
                try:
                    self.completion.submit(completed)
                except BaseException as completion_error:
                    primary = (
                        execution_error
                        if execution_error is not None
                        else completion_error
                    )
                else:
                    primary = execution_error
                try:
                    self.runtime_executor.acknowledge(execution)
                except BaseException as acknowledge_error:
                    if primary is None:
                        primary = acknowledge_error

                if primary is not None:
                    break

                request_id += 1
                sample_index += actual_batch_size
                steps += 1
                emit(
                    "batch_complete",
                    batch_idx=steps,
                    actual_batch_size=actual_batch_size,
                    timing_ms=execution.timing_ms,
                )
        except BaseException as loop_error:
            if primary is None:
                primary = loop_error
        finally:
            try:
                coordinator_stopped = self.completion.stop(timeout=0.0)
            except BaseException as coordinator_error:
                if primary is None:
                    primary = coordinator_error
            else:
                if coordinator_stopped is not True and primary is None:
                    primary = RuntimeError(
                        "e2e completion coordinator stop failed"
                    )

            try:
                executor_stopped = self.runtime_executor.shutdown(timeout=0.0)
            except BaseException as shutdown_error:
                if primary is None:
                    primary = shutdown_error
            else:
                if executor_stopped is not True and primary is None:
                    primary = RuntimeError(
                        "e2e runtime executor shutdown failed"
                    )

        if primary is not None:
            raise primary

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
        try:
            controller.validate(
                config,
                warmup_runs,
                publish_lifecycle=False,
            )
        except BaseException:
            self._prepare_async_diagnostics(controller)
            controller.publish_lifecycle_phase()
            raise
        self._claim_run("async", async_controller=controller)
        controller.publish_lifecycle_phase()

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
            dataloader=self.dataloader,
            runtime=self.runtime,
            evaluator=self.evaluator,
            max_new_tokens=self.max_new_tokens,
            decoder=self.decoder,
            trace_callback=self.trace_callback,
            lifecycle_callback=self.lifecycle_callback,
            pipeline=pipeline,
            runtime_executor=runtime_executor,
            metrics=metrics,
            completion=self.completion,
        )
        return controller.run(config, warmup_runs=warmup_runs)
