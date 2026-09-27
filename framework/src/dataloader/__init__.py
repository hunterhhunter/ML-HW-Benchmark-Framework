"""
DataLoader Package Initialization & Factory

이 모듈은 벤치마크 프레임워크의 다른 컴포넌트(Runtime, Evaluators 등)에게
DataLoader 클래스들에 대한 손쉬운 접근(단일 진입점 API)을 제공합니다.
"""

from importlib import import_module

from core.model_spec import Model_Spec, Task
from .base import DataLoader


_LAZY_EXPORTS = {
    "ImageClassificationLoader": ".image_classification_loader",
    "HailoImageClassificationLoader": ".hailo_image_classification_loader",
    "HailoPoseEstimationLoader": ".hailo_pose_estimation_loader",
    "MobilintImageClassificationLoader": ".mobilint_image_classification_loader",
    "MobilintObjectDetectionLoader": ".mobilint_object_detection_loader",
    "ObjectDetectionLoader": ".object_detection_loader",
    "CocoInstanceSegmentationLoader": ".coco_instance_segmentation_loader",
    "CocoPoseLoader": ".coco_pose_loader",
    "LlamaLoader": ".llama_loader",
    "BertClassificationLoader": ".bert_classification_loader",
    "BertQALoader": ".bert_qa_loader",
    "ETTmLoader": ".ettm_loader",
    "TTMR2ETTh1Loader": ".ttm_r2_etth1_loader",
    "DeepXDataLoader": ".deepx_loader",
    "DeepXObjectDetectionLoader": ".deepx_vision_loader",
    "DeepXInstanceSegmentationLoader": ".deepx_vision_loader",
    "DeepXPoseEstimationLoader": ".deepx_vision_loader",
    "PreprocessStrategy": ".preprocess_strategies",
    "MLPerfResNet50Preprocess": ".preprocess_strategies",
    "MLPerfResNet50RawPreprocess": ".preprocess_strategies",
    "DirectResizePreprocess": ".preprocess_strategies",
    "SQuADPreprocessStrategy": ".preprocess_strategies",
    "TimeSeriesPreprocessStrategy": ".preprocess_strategies",
}


def _load_export(name: str):
    if name in globals():
        return globals()[name]
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __getattr__(name: str):
    """Load only the task-specific loader and its optional dependencies."""
    return _load_export(name)

def create_dataloader(model_spec: Model_Spec, **kwargs) -> DataLoader:
    """
    Factory Method for DataLoader
    
    Model_Spec의 Task 종류(IMAGE_CLASSIFICATION 등)를 분석하여
    해당 Task에 알맞은 구체 로더(Concrete Loader) 객체를 초기화하여 반환합니다.
    
    Args:
        model_spec (Model_Spec): 로드할 모델의 코어 스펙 규격서
        **kwargs: dataset_path, image_dir, mean, std 등 구체 로더에 필요한 추가 인자
        
    Returns:
        DataLoader: 추상 베이스 클래스를 상속받은 구체 로더 인스턴스
        
    Raises:
        ValueError: 모델의 Task에 알맞은 로더가 구현되어 있지 않을 경우 발생
    """
    task = model_spec.task
    backend = str(kwargs.get("backend", "")).lower()

    if model_spec.name == "ttm-r2":
        return _load_export("TTMR2ETTh1Loader")(model_spec, **kwargs)

    if backend == "deepx":
        return _load_export("DeepXDataLoader")(model_spec, **kwargs)
    if backend in ("hailort", "hailo", "hailo8"):
        if task == Task.IMAGE_CLASSIFICATION:
            return _load_export("HailoImageClassificationLoader")(
                model_spec, **kwargs
            )
        if task == Task.POSE_ESTIMATION:
            return _load_export("HailoPoseEstimationLoader")(
                model_spec, **kwargs
            )
    if backend == "mobilint":
        if task is Task.IMAGE_CLASSIFICATION:
            return _load_export("MobilintImageClassificationLoader")(
                model_spec, **kwargs
            )
        if task is Task.OBJECT_DETECTION:
            return _load_export("MobilintObjectDetectionLoader")(
                model_spec, **kwargs
            )
        if task in {
            Task.SEMANTIC_SEGMENTATION,
            Task.INSTANCE_SEGMENTATION,
            Task.POSE_ESTIMATION,
        }:
            raise ValueError(f"Mobilint vision task {task.name} is not supported.")
    
    if task == Task.IMAGE_CLASSIFICATION:
        return _load_export("ImageClassificationLoader")(model_spec, **kwargs)
    elif task == Task.OBJECT_DETECTION:
        return _load_export("ObjectDetectionLoader")(model_spec, **kwargs)
    elif task == Task.INSTANCE_SEGMENTATION:
        return _load_export("CocoInstanceSegmentationLoader")(
            model_spec, **kwargs
        )
    elif task == Task.POSE_ESTIMATION:
        return _load_export("CocoPoseLoader")(model_spec, **kwargs)
    elif task == Task.NLP_GENERATION:
        return _load_export("LlamaLoader")(model_spec, **kwargs)
    elif task == Task.NLP_CLASSIFICATION:
        return _load_export("BertClassificationLoader")(model_spec, **kwargs)
    elif task == Task.QUESTION_ANSWERING:
        return _load_export("BertQALoader")(model_spec, **kwargs)
    elif task == Task.TIME_SERIES_FORECASTING:
        return _load_export("ETTmLoader")(model_spec, **kwargs)
    else:
        raise ValueError(f"현재 '{task.name}' Task를 지원하는 DataLoader가 구현되어 있지 않습니다.")

__all__ = [
    "DataLoader",
    "ImageClassificationLoader",
    "HailoImageClassificationLoader",
    "HailoPoseEstimationLoader",
    "MobilintImageClassificationLoader",
    "MobilintObjectDetectionLoader",
    "ObjectDetectionLoader",
    "CocoInstanceSegmentationLoader",
    "CocoPoseLoader",
    "LlamaLoader",
    "BertClassificationLoader",
    "BertQALoader",
    "ETTmLoader",
    "TTMR2ETTh1Loader",
    "DeepXDataLoader",
    "DeepXObjectDetectionLoader",
    "DeepXInstanceSegmentationLoader",
    "DeepXPoseEstimationLoader",
    "create_dataloader",
    "PreprocessStrategy",
    "MLPerfResNet50Preprocess",
    "MLPerfResNet50RawPreprocess",
    "DirectResizePreprocess",
    "SQuADPreprocessStrategy",
    "TimeSeriesPreprocessStrategy",
]
