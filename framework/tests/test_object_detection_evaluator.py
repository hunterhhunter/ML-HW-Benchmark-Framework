import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from evaluators.object_detection_evaluator import ObjectDetectionEvaluator


def test_ground_truth_uses_letterbox_preprocess_context():
    evaluator = ObjectDetectionEvaluator(image_size=640)
    labels = [
        {
            "label": np.array([[0, 0.5, 0.5, 0.25, 0.5]], dtype=np.float32),
            "preprocess_context": {
                "original_width": 320,
                "original_height": 240,
                "scale": 2.0,
                "pad_x": 0,
                "pad_y": 80,
            },
        }
    ]

    gts = evaluator._process_ground_truths(labels, img_idx_offset=0)

    assert gts == [[0, 0.0, 1.0, 240.0, 200.0, 400.0, 440.0]]


def test_ground_truth_without_context_keeps_legacy_square_scaling():
    evaluator = ObjectDetectionEvaluator(image_size=640)
    labels = [np.array([[0, 0.5, 0.5, 0.25, 0.5]], dtype=np.float32)]

    gts = evaluator._process_ground_truths(labels, img_idx_offset=0)

    assert gts == [[0, 0.0, 1.0, 240.0, 160.0, 400.0, 480.0]]
