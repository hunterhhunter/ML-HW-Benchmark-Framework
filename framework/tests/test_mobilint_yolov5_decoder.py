import os
import sys
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from core.model_spec import Model_Spec, Task
from decoders import create_decoder
from decoders.mobilint_yolov5 import MobilintYoloV5HeadDecoder
from decoders.object_detection import DETECTIONS_KEY, nms_pure_numpy


_PROFILE_PATH = Path(__file__).parents[1] / "src" / "dataloader" / "mobilint_vision_profiles.py"
_PROFILE_SPEC = spec_from_file_location("_test_mobilint_vision_profiles", _PROFILE_PATH)
assert _PROFILE_SPEC is not None and _PROFILE_SPEC.loader is not None
_PROFILE_MODULE = module_from_spec(_PROFILE_SPEC)
sys.modules[_PROFILE_SPEC.name] = _PROFILE_MODULE
_PROFILE_SPEC.loader.exec_module(_PROFILE_MODULE)
MOBILINT_RESNET50_IMAGENET1K_V2 = _PROFILE_MODULE.MOBILINT_RESNET50_IMAGENET1K_V2
MOBILINT_YOLOV5M_DEFAULT = _PROFILE_MODULE.MOBILINT_YOLOV5M_DEFAULT


FEATURES = 85
CHANNELS = 3 * FEATURES


def _logit(probability):
    return np.log(probability / (1.0 - probability))


def _raw_heads(fill=-100.0, *, batch=1, unbatched_strides=()):
    heads = {}
    for name, stride in (("untrusted_large", 32), ("untrusted_small", 8), ("middle", 16)):
        spatial = 640 // stride
        shape = (batch, spatial, spatial, CHANNELS)
        head = np.full(shape, fill, dtype=np.float32)
        if stride in unbatched_strides:
            if batch != 1:
                raise ValueError("Only batch-one fixtures can be unbatched.")
            head = head[0]
        heads[name] = head
    return heads


def _set_anchor(head, *, y, x, anchor, xy=(0.0, 0.0), wh=(0.0, 0.0), obj=0.9, classes):
    values = head[0] if head.ndim == 4 else head
    offset = anchor * FEATURES
    values[y, x, offset : offset + 5] = [*xy, *wh, _logit(obj)]
    for class_id, probability in classes.items():
        values[y, x, offset + 5 + class_id] = _logit(probability)


def _decoder(**kwargs):
    return MobilintYoloV5HeadDecoder(MOBILINT_YOLOV5M_DEFAULT, **kwargs)


def _detection_spec():
    return Model_Spec(
        name="yolov5m",
        task=Task.OBJECT_DETECTION,
        input_shapes={"images": (1, 640, 640, 3)},
        input_dtype={"images": "uint8"},
        output_shapes={"head": (1, 80, 80, CHANNELS)},
        model_paths={"mxq": "yolov5m.mxq"},
    )


def test_raw_heads_match_by_spatial_shape_and_normalize_mixed_batch_axes():
    heads = _raw_heads(fill=0.0, unbatched_strides=(8, 32))

    decoded, raw_objectness = _decoder()._decode_heads(heads)

    assert decoded.shape == (1, 25_200, FEATURES)
    assert raw_objectness.shape == (1, 25_200)
    np.testing.assert_allclose(decoded[0, 0, :4], [4.0, 4.0, 10.0, 13.0])


def test_raw_head_decode_matches_yolov5_anchor_grid_formula():
    heads = _raw_heads(fill=0.0)
    stride8 = heads["untrusted_small"]
    y, x, anchor = 3, 7, 0
    raw_xy = np.array([0.25, -0.75], dtype=np.float32)
    raw_wh = np.array([0.4, -0.2], dtype=np.float32)
    raw_obj = 0.8
    raw_class = -0.3
    offset = anchor * FEATURES
    stride8[0, y, x, offset : offset + 5] = [*raw_xy, *raw_wh, raw_obj]
    stride8[0, y, x, offset + 5 + 4] = raw_class

    decoded, raw_objectness = _decoder()._decode_heads(heads)

    index = (y * 80 + x) * 3 + anchor
    row = decoded[0, index]
    sigmoid = lambda value: 1.0 / (1.0 + np.exp(-value))
    expected_xy = (sigmoid(raw_xy) * 2.0 - 0.5 + np.array([x, y])) * 8.0
    expected_wh = (sigmoid(raw_wh) * 2.0) ** 2 * np.array([10.0, 13.0])
    expected_score = sigmoid(raw_obj) * sigmoid(raw_class)
    np.testing.assert_allclose(row[:2], expected_xy, rtol=1e-6)
    np.testing.assert_allclose(row[2:4], expected_wh, rtol=1e-6)
    np.testing.assert_allclose(row[4] * row[5 + 4], expected_score, rtol=1e-6)
    np.testing.assert_allclose(raw_objectness[0, index], raw_obj)


def test_combined_confidence_rejects_high_objectness_low_class_score():
    heads = _raw_heads()
    _set_anchor(
        heads["untrusted_small"],
        y=0,
        x=0,
        anchor=0,
        obj=0.9,
        classes={3: 0.2},
    )

    detections = _decoder(conf_threshold=0.25).decode(heads)[DETECTIONS_KEY]

    assert detections.shape == (0, 7)


def test_one_anchor_emits_multiple_class_candidates():
    heads = _raw_heads()
    _set_anchor(
        heads["untrusted_small"],
        y=4,
        x=5,
        anchor=0,
        obj=0.9,
        classes={2: 0.8, 7: 0.7},
    )

    detections = _decoder(conf_threshold=0.25).decode(heads)[DETECTIONS_KEY]

    assert detections.shape == (2, 7)
    assert set(detections[:, 1].astype(int)) == {2, 7}
    np.testing.assert_allclose(detections[1, 3:], detections[0, 3:])


def test_class_aware_nms_keeps_identical_boxes_from_different_classes():
    heads = _raw_heads()
    stride8 = heads["untrusted_small"]
    _set_anchor(stride8, y=2, x=2, anchor=0, obj=0.9, classes={4: 0.9})
    anchor1_wh = np.log(
        (0.5 * np.sqrt(np.array([10.0, 13.0]) / np.array([16.0, 30.0])))
        / (1.0 - 0.5 * np.sqrt(np.array([10.0, 13.0]) / np.array([16.0, 30.0])))
    )
    _set_anchor(
        stride8,
        y=2,
        x=2,
        anchor=1,
        wh=anchor1_wh,
        obj=0.8,
        classes={9: 0.9},
    )

    detections = _decoder(conf_threshold=0.25).decode(heads)[DETECTIONS_KEY]

    assert detections.shape == (2, 7)
    assert set(detections[:, 1].astype(int)) == {4, 9}
    np.testing.assert_allclose(detections[1, 3:], detections[0, 3:], rtol=1e-5)


def test_class_aware_nms_suppresses_identical_boxes_from_same_class():
    heads = _raw_heads()
    stride8 = heads["untrusted_small"]
    _set_anchor(stride8, y=2, x=2, anchor=0, obj=0.9, classes={4: 0.9})
    target_over_anchor = np.array([10.0, 13.0]) / np.array([16.0, 30.0])
    sigmoid_wh = 0.5 * np.sqrt(target_over_anchor)
    anchor1_wh = np.log(sigmoid_wh / (1.0 - sigmoid_wh))
    _set_anchor(
        stride8,
        y=2,
        x=2,
        anchor=1,
        wh=anchor1_wh,
        obj=0.8,
        classes={4: 0.9},
    )

    detections = _decoder(conf_threshold=0.25).decode(heads)[DETECTIONS_KEY]

    assert detections.shape == (1, 7)
    assert detections[0, 1] == 4


def test_max_det_truncates_score_sorted_detections():
    heads = _raw_heads()
    stride8 = heads["untrusted_small"]
    _set_anchor(stride8, y=1, x=1, anchor=0, obj=0.9, classes={1: 0.9})
    _set_anchor(stride8, y=10, x=10, anchor=0, obj=0.9, classes={2: 0.8})
    _set_anchor(stride8, y=20, x=20, anchor=0, obj=0.9, classes={3: 0.7})

    detections = _decoder(conf_threshold=0.25, max_det=2).decode(heads)[DETECTIONS_KEY]

    assert detections.shape == (2, 7)
    assert detections[:, 1].astype(int).tolist() == [1, 2]


def test_public_numpy_nms_suppresses_overlapping_lower_score_box():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    assert nms_pure_numpy(boxes, scores, 0.5) == [0, 2]


def test_raw_head_decoder_rejects_wrong_head_count():
    heads = _raw_heads()
    heads.pop("middle")

    with pytest.raises(ValueError, match="exactly 3"):
        _decoder().decode(heads)


def test_raw_head_decoder_rejects_duplicate_spatial_size():
    heads = _raw_heads()
    heads["middle"] = np.zeros((1, 80, 80, CHANNELS), dtype=np.float32)

    with pytest.raises(ValueError, match="duplicate spatial"):
        _decoder().decode(heads)


def test_raw_head_decoder_rejects_wrong_channel_count():
    heads = _raw_heads()
    heads["middle"] = np.zeros((1, 40, 40, CHANNELS - 1), dtype=np.float32)

    with pytest.raises(ValueError, match="channel"):
        _decoder().decode(heads)


def test_raw_head_decoder_rejects_nchw_layout():
    heads = _raw_heads()
    heads["untrusted_small"] = np.zeros((1, CHANNELS, 80, 80), dtype=np.float32)

    with pytest.raises(ValueError, match="NHWC"):
        _decoder().decode(heads)


def test_raw_head_decoder_rejects_batch_mismatch():
    heads = _raw_heads()
    heads["middle"] = np.zeros((2, 40, 40, CHANNELS), dtype=np.float32)

    with pytest.raises(ValueError, match="batch"):
        _decoder().decode(heads)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"conf_threshold": 0.0}, "conf_threshold"),
        ({"conf_threshold": 1.0}, "conf_threshold"),
        ({"max_nms": 0}, "max_nms"),
        ({"max_det": 0}, "max_det"),
        ({"max_class_offset": 0}, "max_class_offset"),
    ],
)
def test_raw_head_decoder_rejects_invalid_limits(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _decoder(**kwargs)


def test_create_decoder_selects_mobilint_raw_head_decoder_with_profile_defaults():
    decoder = create_decoder(
        _detection_spec(),
        backend="mobilint",
        mobilint_vision_profile=MOBILINT_YOLOV5M_DEFAULT,
    )

    assert isinstance(decoder, MobilintYoloV5HeadDecoder)
    assert decoder.conf_threshold == 0.001
    assert decoder.iou_threshold == 0.65
    assert decoder.max_det == 300
    assert decoder.max_nms == 30_000
    assert decoder.max_class_offset == 7_680


def test_create_decoder_runtime_options_override_mobilint_profile_defaults():
    decoder = create_decoder(
        _detection_spec(),
        backend="mobilint",
        mobilint_vision_profile=MOBILINT_YOLOV5M_DEFAULT,
        runtime_options={
            "conf_threshold": 0.2,
            "iou_threshold": 0.4,
            "max_nms": 123,
            "max_det": 7,
            "max_class_offset": 4_096,
        },
    )

    assert isinstance(decoder, MobilintYoloV5HeadDecoder)
    assert decoder.conf_threshold == 0.2
    assert decoder.iou_threshold == 0.4
    assert decoder.max_nms == 123
    assert decoder.max_det == 7
    assert decoder.max_class_offset == 4_096


def test_create_decoder_explicit_options_override_mobilint_profile_defaults():
    decoder = create_decoder(
        _detection_spec(),
        backend="mobilint",
        mobilint_vision_profile=MOBILINT_YOLOV5M_DEFAULT,
        runtime_options={
            "conf_threshold": 0.1,
            "iou_threshold": 0.3,
            "max_nms": 11,
            "max_det": 3,
            "max_class_offset": 900,
        },
        conf_threshold=0.2,
        iou_threshold=0.4,
        max_det=4,
        max_nms=12,
        max_class_offset=1_000,
    )

    assert isinstance(decoder, MobilintYoloV5HeadDecoder)
    assert decoder.conf_threshold == 0.2
    assert decoder.iou_threshold == 0.4
    assert decoder.max_det == 4
    assert decoder.max_nms == 12
    assert decoder.max_class_offset == 1_000


def test_create_decoder_requires_mobilint_vision_profile():
    with pytest.raises(ValueError, match="mobilint_vision_profile"):
        create_decoder(_detection_spec(), backend="mobilint")


def test_create_decoder_rejects_mobilint_profile_without_raw_head_recipe():
    invalid_profile = replace(
        MOBILINT_RESNET50_IMAGENET1K_V2,
        task=Task.OBJECT_DETECTION,
    )

    with pytest.raises(ValueError, match="YoloV5RawHeadRecipe"):
        create_decoder(
            _detection_spec(),
            backend="mobilint",
            mobilint_vision_profile=invalid_profile,
        )
