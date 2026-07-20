import time

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
        self.pipeline = InferencePipeline(
            dataloader,
            runtime,
            max_new_tokens=max_new_tokens,
            runtime_executor=runtime_executor,
        )
        self.runtime_executor = self.pipeline._compat_executor
        self.completion = None
        self._e2e_started = False

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

    def run_e2e(self, batch_size=1, max_steps=None):
        if self._e2e_started:
            raise RuntimeError("run_e2e() may only be called once")
        self._e2e_started = True

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
            while max_steps is None or steps < max_steps:
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
                except BaseException:
                    if request_id not in self.completion.snapshot_outstanding():
                        self.runtime_executor.acknowledge(execution)
                    raise
                else:
                    self.runtime_executor.acknowledge(execution)

                request_id += 1
                sample_index += actual_batch_size
                steps += 1
        finally:
            self.completion.stop(timeout=0.0)

        return self.evaluator.compute()
