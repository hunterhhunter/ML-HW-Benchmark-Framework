import numpy as np
import pytest

from decoders.instance_segmentation import YoloV8SegmentationDecoder


def _seg_outputs():
    prediction = np.zeros((1, 116, 2), dtype=np.float32)
    prediction[0, :4, 0] = [4, 4, 4, 4]
    prediction[0, 4, 0] = 0.9
    prediction[0, 84, 0] = 10.0
    prediction[0, :4, 1] = [4.2, 4.2, 4, 4]
    prediction[0, 4, 1] = 0.8
    prediction[0, 84, 1] = 10.0
    prototype = np.zeros((1, 32, 2, 2), dtype=np.float32)
    prototype[0, 0] = 1.0
    return {"proto": prototype, "prediction": prediction}


def test_decoder_applies_nms_and_builds_aligned_binary_masks():
    result = YoloV8SegmentationDecoder(
        conf_threshold=0.25,
        iou_threshold=0.5,
        max_detections=10,
    ).decode(_seg_outputs())

    assert result["detections"].shape == (1, 7)
    assert result["detections"][0, :3].tolist() == pytest.approx(
        [0, 0, 0.9]
    )
    assert result["masks"].shape == (1, 8, 8)
    assert result["masks"].dtype == np.uint8
    assert result["masks"][0, 2:6, 2:6].all()
    assert not result["masks"][0, :2].any()
    assert not result["masks"][0, 6:].any()


def test_decoder_keeps_overlapping_instances_of_different_classes_aligned():
    outputs = _seg_outputs()
    outputs["prediction"][0, 4, 1] = 0.0
    outputs["prediction"][0, 5, 1] = 0.8
    outputs["prediction"][0, 84, 1] = -10.0

    result = YoloV8SegmentationDecoder(iou_threshold=0.5).decode(outputs)

    assert result["detections"][:, 1].tolist() == [0.0, 1.0]
    assert result["masks"].shape == (2, 8, 8)
    assert result["masks"][0].any()
    assert not result["masks"][1].any()


def test_decoder_returns_typed_empty_arrays():
    outputs = _seg_outputs()
    outputs["prediction"][:, 4:84, :] = 0.0

    result = YoloV8SegmentationDecoder().decode(outputs)

    assert result["detections"].shape == (0, 7)
    assert result["detections"].dtype == np.float32
    assert result["masks"].shape == (0, 8, 8)
    assert result["masks"].dtype == np.uint8


def test_decoder_rejects_prediction_prototype_batch_mismatch():
    outputs = _seg_outputs()
    outputs["proto"] = np.repeat(outputs["proto"], 2, axis=0)

    with pytest.raises(ValueError, match="batch mismatch"):
        YoloV8SegmentationDecoder().decode(outputs)


def test_decoder_rejects_ambiguous_and_non_finite_outputs():
    outputs = _seg_outputs()
    outputs["another_prediction"] = outputs["prediction"].copy()
    with pytest.raises(ValueError, match="ambiguous segmentation prediction"):
        YoloV8SegmentationDecoder().decode(outputs)

    outputs = _seg_outputs()
    outputs["proto"][0, 0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        YoloV8SegmentationDecoder().decode(outputs)


def test_decoder_rejects_wrong_output_shapes():
    with pytest.raises(ValueError, match="segmentation prediction"):
        YoloV8SegmentationDecoder().decode(
            {
                "prediction": np.zeros((1, 115, 2), dtype=np.float32),
                "proto": np.zeros((1, 32, 2, 2), dtype=np.float32),
            }
        )
