"""
Preprocessor Package — 모델별 전처리기 모음

각 모델 타입에 특화된 전처리기 클래스를 제공합니다.
모든 전처리기는 BasePreprocessor를 상속합니다.

핵심 원칙:
  - numpy 캐시가 존재하면 전처리 없이 로드합니다.
  - 캐시가 없으면 전처리를 실행하고 numpy 파일로 저장합니다.

전처리기 유형:
  샘플 단위 (.npy / .npz 캐싱):
    - ImagePreprocessor             : 이미지 분류
    - ObjectDetectionPreprocessor   : 객체 탐지 (YOLO)
    - LlamaPreprocessor             : LLaMA SQuAD QA 토큰화
    - ETTmPreprocessor              : 시계열 RevIN 정규화

  데이터셋 단위 (전체 데이터셋 .npy 저장):
    - BertClassificationPreprocessor : BERT SST-2 등 텍스트 분류
    - BertQAPreprocessor             : BERT SQuAD QA
"""

from importlib import import_module

from .base import BasePreprocessor


_LAZY_EXPORTS = {
    "PreprocessStrategy": ".strategies",
    "DirectResizePreprocess": ".strategies",
    "MLPerfResNet50Preprocess": ".strategies",
    "MLPerfResNet50RawPreprocess": ".strategies",
    "SQuADPreprocessStrategy": ".strategies",
    "TimeSeriesPreprocessStrategy": ".strategies",
    "ImagePreprocessor": ".image_preprocessor",
    "ObjectDetectionPreprocessor": ".object_detection_preprocessor",
    "MobilintResNetCenterCropPreprocess": ".mobilint_vision",
    "MobilintYoloV5Preprocessor": ".mobilint_vision",
    "YoloVisionPreprocessor": ".yolo_vision_preprocessor",
    "LlamaPreprocessor": ".llama_preprocessor",
    "ETTmPreprocessor": ".ettm_preprocessor",
    "BertClassificationPreprocessor": ".bert_classification_preprocessor",
    "BertQAPreprocessor": ".bert_qa_preprocessor",
}


def __getattr__(name: str):
    """Import optional preprocessing stacks only when their task needs them."""
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

__all__ = [
    "BasePreprocessor",
    "PreprocessStrategy",
    "DirectResizePreprocess",
    "MLPerfResNet50Preprocess",
    "MLPerfResNet50RawPreprocess",
    "SQuADPreprocessStrategy",
    "TimeSeriesPreprocessStrategy",
    "ImagePreprocessor",
    "ObjectDetectionPreprocessor",
    "MobilintResNetCenterCropPreprocess",
    "MobilintYoloV5Preprocessor",
    "YoloVisionPreprocessor",
    "LlamaPreprocessor",
    "ETTmPreprocessor",
    "BertClassificationPreprocessor",
    "BertQAPreprocessor",
]
