"""
Evaluator Package Initialization & Factory

이 모듈은 벤치마크 프레임워크의 다양한 Task(이미지 분류, 객체 탐지 등)에 맞는
평가기(Evaluator) 인스턴스를 동적으로 생성하고 접근할 수 있는 단일 진입점 API를 제공합니다.
"""

from core.model_spec import Model_Spec, Task
from .base import Evaluator

_LAZY_EXPORTS = {
    "ImageClassificationEvaluator": ".image_classification_evaluator",
    "LlamaEvaluator": ".llama_evaluator",
    "ObjectDetectionEvaluator": ".object_detection_evaluator",
    "BertClassificationEvaluator": ".bert_classification_evaluator",
    "BertQAEvaluator": ".bert_qa_evaluator",
    "TimeSeriesForecastingEvaluator": ".time_series_forecasting_evaluator",
    "LatencyOnlyEvaluator": ".latency_evaluator",
    "InstanceSegmentationEvaluator": ".instance_segmentation_evaluator",
    "PoseEstimationEvaluator": ".pose_estimation_evaluator",
}

_LEGACY_LATENCY_VISION_MODELS = {
    "yolov8m-seg",
    "yolov8m_seg",
    "yolov8-seg-m",
    "yolov8m-pose",
    "yolov8m_pose",
    "yolov8-pose-m",
}


def __getattr__(name: str):
    """평가기별 선택 의존성이 서로를 막지 않도록 필요한 클래스만 import합니다."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_LAZY_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value

def create_evaluator(model_spec: Model_Spec, **kwargs) -> Evaluator:
    """
    Factory Method for Evaluator

    Model_Spec의 Task 종류를 분석하여 해당 Task를 평가할 수 있는
    적절한 구체 평가기(Concrete Evaluator) 객체를 생성합니다.

    Args:
        model_spec (Model_Spec): 평가할 모델의 코어 스펙 규격서
        **kwargs: top_k 등 평가기에 전달될 추가 옵션 인자

    Returns:
        Evaluator: 추상 베이스 클래스를 상속받은 구체 평가기(Metric Calculator)
    """
    task = model_spec.task

    if task == Task.IMAGE_CLASSIFICATION:
        # 단일 책임 원칙: 이미지 분류 테스크는 ImageClassificationEvaluator가 전담
        # 추후 MobileNet 특화 로직이 별도로 필요하면 model_spec.name 등을 통해 분기 가능
        from .image_classification_evaluator import ImageClassificationEvaluator
        return ImageClassificationEvaluator(**kwargs)

    elif task == Task.NLP_GENERATION:
        from .llama_evaluator import LlamaEvaluator
        return LlamaEvaluator(**kwargs)

    elif task == Task.OBJECT_DETECTION:
        from .object_detection_evaluator import ObjectDetectionEvaluator
        return ObjectDetectionEvaluator(**kwargs)

    elif task in (Task.INSTANCE_SEGMENTATION, Task.POSE_ESTIMATION):
        backend = str(kwargs.get("backend", "")).lower()
        if backend == "deepx" or (
            model_spec.name in _LEGACY_LATENCY_VISION_MODELS
            and not kwargs.get("annotation_file")
        ):
            kwargs.pop("backend", None)
            from .latency_evaluator import LatencyOnlyEvaluator
            return LatencyOnlyEvaluator(task_name=task.name, **kwargs)
        kwargs.pop("backend", None)
        if task == Task.INSTANCE_SEGMENTATION:
            from .instance_segmentation_evaluator import (
                InstanceSegmentationEvaluator,
            )
            return InstanceSegmentationEvaluator(**kwargs)
        from .pose_estimation_evaluator import PoseEstimationEvaluator
        return PoseEstimationEvaluator(**kwargs)

    elif task == Task.NLP_CLASSIFICATION:
        from .bert_classification_evaluator import BertClassificationEvaluator
        return BertClassificationEvaluator(**kwargs)

    elif task == Task.QUESTION_ANSWERING:
        from .bert_qa_evaluator import BertQAEvaluator
        return BertQAEvaluator(**kwargs)

    elif task == Task.TIME_SERIES_FORECASTING:
        from .time_series_forecasting_evaluator import TimeSeriesForecastingEvaluator
        return TimeSeriesForecastingEvaluator(**kwargs)

    else:
        raise ValueError(f"현재 '{task.name}' Task를 지원하는 Evaluator가 구현되어 있지 않습니다.")

__all__ = [
    "Evaluator",
    "ImageClassificationEvaluator",
    "LlamaEvaluator",
    "ObjectDetectionEvaluator",
    "BertClassificationEvaluator",
    "BertQAEvaluator",
    "TimeSeriesForecastingEvaluator",
    "LatencyOnlyEvaluator",
    "InstanceSegmentationEvaluator",
    "PoseEstimationEvaluator",
    "create_evaluator"
]
