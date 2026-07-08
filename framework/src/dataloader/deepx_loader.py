"""DeepX DataLoader facade with task-level routing.

DX-APP organizes examples by task and keeps the common runner layer responsible
for selecting task-specific processors. This facade mirrors that shape for the
benchmark framework: `backend=deepx` enters here first, then the facade delegates
to the concrete task loader that already owns the dataset contract.
"""

from typing import Any, Dict

from .base import DataLoader
from core.model_spec import Model_Spec, Task


class DeepXDataLoader(DataLoader):
    """Route DeepX datasets to task-specific DataLoader implementations."""

    def __init__(self, model_spec: Model_Spec, **kwargs):
        self.model_spec = model_spec
        self.task = model_spec.task
        self.backend = "deepx"

        delegate_kwargs = dict(kwargs)
        delegate_kwargs["backend"] = "deepx"
        self._delegate = self._create_delegate(model_spec, delegate_kwargs)

    @property
    def delegate(self) -> DataLoader:
        return self._delegate

    @classmethod
    def supported_tasks(cls) -> tuple[Task, ...]:
        return (
            Task.IMAGE_CLASSIFICATION,
            Task.OBJECT_DETECTION,
            Task.INSTANCE_SEGMENTATION,
            Task.POSE_ESTIMATION,
            Task.NLP_CLASSIFICATION,
            Task.NLP_GENERATION,
            Task.QUESTION_ANSWERING,
            Task.TIME_SERIES_FORECASTING,
        )

    @classmethod
    def _create_delegate(cls, model_spec: Model_Spec, kwargs: Dict[str, Any]) -> DataLoader:
        loader_cls = cls._loader_class_for_task(model_spec.task)
        return loader_cls(model_spec, **kwargs)

    @classmethod
    def _loader_class_for_task(cls, task: Task):
        if task == Task.IMAGE_CLASSIFICATION:
            from .deepx_image_classification_loader import DeepXImageClassificationLoader

            return DeepXImageClassificationLoader
        if task == Task.OBJECT_DETECTION:
            from .deepx_vision_loader import DeepXObjectDetectionLoader

            return DeepXObjectDetectionLoader
        if task == Task.INSTANCE_SEGMENTATION:
            from .deepx_vision_loader import DeepXInstanceSegmentationLoader

            return DeepXInstanceSegmentationLoader
        if task == Task.POSE_ESTIMATION:
            from .deepx_vision_loader import DeepXPoseEstimationLoader

            return DeepXPoseEstimationLoader
        if task == Task.NLP_CLASSIFICATION:
            from .bert_classification_loader import BertClassificationLoader

            return BertClassificationLoader
        if task == Task.NLP_GENERATION:
            from .llama_loader import LlamaLoader

            return LlamaLoader
        if task == Task.QUESTION_ANSWERING:
            from .bert_qa_loader import BertQALoader

            return BertQALoader
        if task == Task.TIME_SERIES_FORECASTING:
            from .ettm_loader import ETTmLoader

            return ETTmLoader

        supported = ", ".join(item.name for item in cls.supported_tasks())
        raise ValueError(
            f"DeepXDataLoader does not support task '{task.name}'. "
            f"Supported tasks: {supported}"
        )

    def load_single(self) -> Dict[str, Any]:
        return self._delegate.load_single()

    def load_batch(self, batch_size: int) -> Any:
        return self._delegate.load_batch(batch_size)

    def get_labels(self) -> Any:
        return self._delegate.get_labels()

    def get_metadata(self) -> Dict[str, Any]:
        metadata = dict(self._delegate.get_metadata())
        deepx_metadata = metadata.get("deepx", {})
        if not isinstance(deepx_metadata, dict):
            deepx_metadata = {}
        metadata["deepx"] = {
            **deepx_metadata,
            "task": self.task.name,
            "delegate_loader": type(self._delegate).__name__,
        }
        return metadata

    def preprocess(self, raw_input: Any) -> Any:
        return self._delegate.preprocess(raw_input)

    def load_by_index(self, index: int) -> Dict[str, Any]:
        return self._delegate.load_by_index(index)

    def __getattr__(self, name: str) -> Any:
        if "_delegate" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self._delegate, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            name in ("current_idx", "_current_idx")
            and "_delegate" in self.__dict__
            and hasattr(self._delegate, name)
        ):
            setattr(self._delegate, name, value)
            return
        super().__setattr__(name, value)
