import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from core.model_spec import Model_Spec, Task
from decoders import create_decoder
from decoders.object_detection import (
    DETECTIONS_KEY,
    HailoYoloNMSDecoder,
    RawYoloDetectionDecoder,
)


def _detection_spec():
    return Model_Spec(
        name="yolov5m",
        task=Task.OBJECT_DETECTION,
        input_shapes={"images": (1, 3, 640, 640)},
        input_dtype={"images": "float32"},
        output_shapes={"output": (1, 25200, 85)},
        model_paths={"onnx": "unused.onnx"},
    )


def test_raw_yolo_decoder_returns_canonical_detections():
    preds = np.zeros((1, 2, 85), dtype=np.float32)
    preds[0, 0, :5] = [320.0, 320.0, 100.0, 120.0, 0.9]
    preds[0, 0, 5 + 7] = 0.8

    decoded = RawYoloDetectionDecoder(conf_threshold=0.25).decode({"output": preds})

    detections = decoded[DETECTIONS_KEY]
    assert detections.shape == (1, 7)
    assert detections[0, 0] == 0
    assert detections[0, 1] == 7
    np.testing.assert_allclose(detections[0, 2], 0.72, rtol=1e-6)
    np.testing.assert_allclose(detections[0, 3:], [270.0, 260.0, 370.0, 380.0])


def test_hailo_nms_decoder_decodes_class_major_tensor():
    nms = np.zeros((80, 5, 80), dtype=np.float32)
    nms[3, :, 0] = [0.1, 0.2, 0.4, 0.6, 0.9]

    decoded = HailoYoloNMSDecoder(conf_threshold=0.25, image_size=640).decode(
        {"yolov5_nms_postprocess": nms}
    )

    detections = decoded[DETECTIONS_KEY]
    assert detections.shape == (1, 7)
    assert detections[0, 0] == 0
    assert detections[0, 1] == 3
    np.testing.assert_allclose(detections[0, 2], 0.9, rtol=1e-6)
    np.testing.assert_allclose(detections[0, 3:], [64.0, 128.0, 256.0, 384.0])


def test_hailo_nms_decoder_accepts_batched_last_axis_layout():
    nms = np.zeros((1, 80, 80, 5), dtype=np.float32)
    nms[0, 4, 0, :] = [0.1, 0.2, 0.4, 0.6, 0.95]

    decoded = HailoYoloNMSDecoder(conf_threshold=0.25, image_size=(640, 640)).decode(
        {"yolov5_nms_postprocess": nms}
    )

    detections = decoded[DETECTIONS_KEY]
    assert detections.shape == (1, 7)
    assert detections[0, 1] == 4
    np.testing.assert_allclose(detections[0, 3:], [64.0, 128.0, 256.0, 384.0])


def test_create_decoder_selects_hailo_nms_decoder_for_hailort_object_detection():
    decoder = create_decoder(_detection_spec(), backend="hailort")

    assert isinstance(decoder, HailoYoloNMSDecoder)
